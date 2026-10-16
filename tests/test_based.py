"""
Tests for BASED linear attention (Taylor feature map, chunked causal linear
attention) and sliding-window attention, each checked against a direct
quadratic reference.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn.functional as F

from guac.core.custom_architectures import (
    BasedMixer, MixedLayerConfig, MixedTransformerLayer, causal_attention, sliding_window_attention,
)
from guac.core.linear_attention import causal_linear_attention, taylor_feature_map


def quadratic_linear_attention(q, k, v):
    """y_i = sum_{j<=i} (q_i . k_j) v_j / sum_{j<=i} q_i . k_j, written out directly."""
    scores = (q @ k.transpose(-1, -2)).tril()
    return (scores @ v) / scores.sum(-1, keepdim=True)


def test_taylor_feature_map_is_second_order_exp_expansion():
    torch.manual_seed(0)
    q, k = torch.randn(5, 16, dtype=torch.float64), torch.randn(5, 16, dtype=torch.float64)
    s = (q * k).sum(-1) / 16 ** 0.5
    torch.testing.assert_close((taylor_feature_map(q) * taylor_feature_map(k)).sum(-1), 1 + s + s ** 2 / 2)
    assert taylor_feature_map(q).shape[-1] == 1 + 16 + 16 ** 2


@pytest.mark.parametrize("length, chunk_size", [(1, 4), (7, 4), (16, 4), (16, 16), (33, 8), (20, 64)])
def test_chunked_linear_attention_matches_quadratic_form(length, chunk_size):
    torch.manual_seed(0)
    q = taylor_feature_map(torch.randn(2, 3, length, 4, dtype=torch.float64)).requires_grad_()
    k = taylor_feature_map(torch.randn(2, 3, length, 4, dtype=torch.float64)).requires_grad_()
    v = torch.randn(2, 3, length, 5, dtype=torch.float64, requires_grad=True)

    expected = quadratic_linear_attention(q, k, v)
    actual, _ = causal_linear_attention(q, k, v, chunk_size, eps=0.0)
    torch.testing.assert_close(actual, expected)

    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e)


def test_linear_attention_continues_from_a_carried_state():
    """Splitting a sequence and passing the state across gives the same outputs, as in generation."""
    torch.manual_seed(0)
    q, k = (taylor_feature_map(torch.randn(1, 2, 13, 4, dtype=torch.float64)) for _ in range(2))
    v = torch.randn(1, 2, 13, 3, dtype=torch.float64)

    whole, whole_state = causal_linear_attention(q, k, v, chunk_size=4)
    first, state = causal_linear_attention(q[:, :, :6], k[:, :, :6], v[:, :, :6], chunk_size=4)
    second, final_state = causal_linear_attention(q[:, :, 6:], k[:, :, 6:], v[:, :, 6:], chunk_size=4,
                                                  initial_state=state)

    torch.testing.assert_close(torch.cat([first, second], dim=2), whole)
    for a, e in zip(final_state, whole_state):
        torch.testing.assert_close(a, e)


def test_linear_attention_with_no_keys_outputs_zero():
    q = taylor_feature_map(torch.randn(1, 1, 4, 4))
    y, _ = causal_linear_attention(q, torch.zeros_like(q), torch.randn(1, 1, 4, 3), chunk_size=2)
    assert torch.equal(y, torch.zeros_like(y))


def based_config(**overrides):
    kwargs = dict(vocab_size=50, hidden_size=32, num_attention_heads=4, num_kv_heads=2, num_hidden_layers=1,
                  layer_types=["based"], intermediate_size=64, based_feature_dim=8, based_chunk_size=4)
    kwargs.update(overrides)
    return MixedLayerConfig(**kwargs)


def test_based_mixer_matches_quadratic_reference():
    """The whole mixer, grouped-query heads included, against the formula written out directly."""
    torch.manual_seed(0)
    mixer = BasedMixer(based_config()).double()
    x = torch.randn(2, 11, 32, dtype=torch.float64)

    def heads(t, n):
        return t.view(2, 11, n, -1).transpose(1, 2)

    q = taylor_feature_map(heads(mixer.q_proj(x), 4))
    k = taylor_feature_map(heads(mixer.k_proj(x), 2)).repeat_interleave(2, dim=1)
    v = heads(mixer.v_proj(x), 2).repeat_interleave(2, dim=1)
    expected = mixer.o_proj(quadratic_linear_attention(q, k, v).transpose(1, 2).reshape(2, 11, 32))

    torch.testing.assert_close(mixer(x), expected)


def test_based_attention_is_not_biased_toward_recent_tokens():
    """
    A query that matches one early key should read that key's value. The old
    softmax over prefix sums of features always favored the latest position.
    """
    q = torch.zeros(1, 1, 8, 4, dtype=torch.float64)
    k = torch.zeros_like(q)
    q[0, 0, -1, 0] = k[0, 0, 1, 0] = 6.0  # the last query matches key 1 strongly, other keys weakly
    v = torch.eye(8, dtype=torch.float64)[None, None]

    y, _ = causal_linear_attention(taylor_feature_map(q), taylor_feature_map(k), v, chunk_size=4)
    assert y[0, 0, -1].argmax() == 1


def test_based_layer_ignores_left_padding():
    torch.manual_seed(0)
    layer = MixedTransformerLayer(based_config(), 0).eval()
    real = torch.randn(1, 7, 32)
    padded = torch.cat([torch.randn(1, 3, 32), real], dim=1)
    mask = torch.tensor([[0] * 3 + [1] * 7])

    with torch.no_grad():
        torch.testing.assert_close(layer(padded, mask)[:, 3:], layer(real))


def banded_reference(q, k, v, window, attention_mask=None):
    length = q.size(2)
    positions = torch.arange(length)
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0) & (distance < window)
    if attention_mask is not None:
        allowed = allowed & attention_mask[:, None, None, :].bool()
    allowed = allowed | torch.eye(length, dtype=torch.bool)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)


@pytest.mark.parametrize("length, window", [(5, 2), (16, 4), (17, 4), (9, 8), (3, 1)])
def test_sliding_window_attention_matches_banded_mask(length, window):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, length, 4, dtype=torch.float64) for _ in range(3))
    mask = torch.ones(2, length, dtype=torch.long)
    mask[0, :2] = 0      # left padding
    mask[1, -2:] = 0     # right padding

    for attention_mask in (None, mask):
        torch.testing.assert_close(sliding_window_attention(q, k, v, window, attention_mask),
                                   banded_reference(q, k, v, window, attention_mask))


def test_window_at_least_the_sequence_is_full_attention():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 6, 4) for _ in range(3))
    torch.testing.assert_close(causal_attention(q, k, v, window=6), causal_attention(q, k, v))


def test_sliding_window_layer_sees_exactly_its_window():
    torch.manual_seed(0)
    window, t = 4, 10
    layer = MixedTransformerLayer(based_config(layer_types=["sliding_window_attention"],
                                               sliding_window_size=window), 0).eval()
    x = torch.randn(1, 16, 32)

    def output_change_at_t(perturbed_position):
        x_perturbed = x.clone()
        x_perturbed[0, perturbed_position] += 5.0
        with torch.no_grad():
            return (layer(x_perturbed)[0, t] - layer(x)[0, t]).abs().max().item()

    assert output_change_at_t(t - window + 1) > 1e-4  # oldest position inside the window
    assert output_change_at_t(t - window) == 0.0       # just outside it
