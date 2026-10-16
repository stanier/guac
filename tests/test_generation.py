"""
Tests for cached generation: decoding token by token with MixedLayerCache
gives the same logits as running the whole sequence, for every layer type,
and generate() with the cache matches generate() without it.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch

from guac.core.custom_architectures import SUPPORTED_LAYER_TYPES, MixedLayerConfig, MixedLayerForCausalLM

WINDOW = 4


def make_model(layer_types, seed=0, **overrides):
    torch.manual_seed(seed)
    kwargs = dict(
        vocab_size=40, hidden_size=32, num_attention_heads=4, num_kv_heads=2,
        num_hidden_layers=len(layer_types), layer_types=layer_types, intermediate_size=64,
        num_experts=3, num_experts_per_token=2, moh_top_k=2, moe_noise_std=0.0, moh_noise_std=0.0,
        # Small windows and chunks, so the test sequences cross their boundaries
        sliding_window_size=WINDOW, based_chunk_size=4, mamba_chunk_size=4, based_feature_dim=4,
        mamba_d_state=8, mamba_headdim=16, conv_kernel_size=3, pad_token_id=0,
    )
    kwargs.update(overrides)
    return MixedLayerForCausalLM(MixedLayerConfig(**kwargs)).eval()


def decode_incrementally(model, input_ids, steps, attention_mask=None):
    """Logits for every position, computed by a prefill followed by calls of the given lengths."""
    cache, logits, start = None, [], 0
    for length in steps:
        end = start + length
        mask = None if attention_mask is None else attention_mask[:, :end]
        out = model(input_ids=input_ids[:, start:end], attention_mask=mask, past_key_values=cache, use_cache=True)
        cache, start = out.past_key_values, end
        logits.append(out.logits)
    assert start == input_ids.size(1)
    return torch.cat(logits, dim=1), cache


@pytest.mark.parametrize("layer_type", SUPPORTED_LAYER_TYPES)
def test_token_by_token_decoding_matches_full_forward(layer_type):
    model = make_model([layer_type, layer_type])
    input_ids = torch.randint(1, 40, (2, 13))
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits
        actual, _ = decode_incrementally(model, input_ids, [5] + [1] * 8)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_multi_token_continuation_matches_full_forward():
    model = make_model(list(SUPPORTED_LAYER_TYPES))
    input_ids = torch.randint(1, 40, (2, 15))
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits
        actual, _ = decode_incrementally(model, input_ids, [3, 6, 1, 5])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_left_padded_decoding_matches_unpadded():
    """Batched prompts are left-padded for generation; the padding must not change any real token's logits."""
    model = make_model(list(SUPPORTED_LAYER_TYPES))
    real = torch.randint(1, 40, (1, 9))
    padded = torch.cat([torch.zeros(1, 3, dtype=torch.long), real], dim=1)
    mask = torch.tensor([[0] * 3 + [1] * 9])
    with torch.no_grad():
        expected = model(input_ids=real, use_cache=False).logits
        actual, _ = decode_incrementally(model, padded, [6] + [1] * 6, attention_mask=mask)
    torch.testing.assert_close(actual[:, 3:], expected, atol=1e-5, rtol=1e-4)


def test_generate_with_cache_matches_generate_without():
    model = make_model(list(SUPPORTED_LAYER_TYPES))
    prompts = torch.randint(1, 40, (2, 6))
    kwargs = dict(max_new_tokens=12, do_sample=False, pad_token_id=0)
    with torch.no_grad():
        cached = model.generate(prompts, use_cache=True, **kwargs)
        uncached = model.generate(prompts, use_cache=False, **kwargs)
    assert torch.equal(cached, uncached)


def test_batched_left_padded_generate_matches_one_at_a_time():
    model = make_model(list(SUPPORTED_LAYER_TYPES))
    short, long = torch.randint(1, 40, (1, 4)), torch.randint(1, 40, (1, 7))
    batch = torch.cat([torch.cat([torch.zeros(1, 3, dtype=torch.long), short], dim=1), long])
    mask = torch.tensor([[0] * 3 + [1] * 4, [1] * 7])
    kwargs = dict(max_new_tokens=8, do_sample=False, pad_token_id=0)
    with torch.no_grad():
        batched = model.generate(batch, attention_mask=mask, **kwargs)
        alone = [model.generate(prompt, **kwargs) for prompt in (short, long)]
    # A sequence that stops at EOS first is padded to the batch's length
    for row, single in ((batched[0, 3:], alone[0][0]), (batched[1], alone[1][0])):
        assert torch.equal(row[:len(single)], single)
        assert (row[len(single):] == 0).all()


def test_beam_search_with_cache_matches_without():
    model = make_model(["full_attention", "mamba2", "based", "conv"])
    prompt = torch.randint(1, 40, (1, 5))
    kwargs = dict(max_new_tokens=6, num_beams=3, do_sample=False, pad_token_id=0)
    with torch.no_grad():
        assert torch.equal(model.generate(prompt, use_cache=True, **kwargs),
                           model.generate(prompt, use_cache=False, **kwargs))


def test_cache_size_is_fixed_except_for_full_attention():
    model = make_model(["sliding_window_attention", "mamba2", "based", "conv", "full_attention"])
    input_ids = torch.randint(1, 40, (1, 20))
    with torch.no_grad():
        _, cache = decode_incrementally(model, input_ids, [4] + [1] * 16)
    window, mamba, based, conv, attention = cache.layers
    assert window["key"].size(2) == WINDOW - 1
    assert mamba["ssm_state"].shape == (1, 4, 16, 8)       # heads, head_dim, d_state
    assert based["kv_state"].shape == (1, 4, 1 + 4 + 16, 8)  # heads, Taylor features, head_dim
    assert conv["conv_state"].size(-1) == 2                 # kernel_size - 1
    assert attention["key"].size(2) == 20
    assert cache.seen_tokens == 20


def test_no_cache_in_training_or_with_labels():
    model = make_model(["full_attention"])
    input_ids = torch.randint(1, 40, (1, 6))
    assert model(input_ids=input_ids, labels=input_ids).past_key_values is None
    model.train()
    assert model(input_ids=input_ids, use_cache=True).past_key_values is None
