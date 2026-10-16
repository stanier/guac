"""
Tests for Mixture-of-Head attention: each selected head keeps its own subspace
of the output projection, weighted by its gate.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch

from guac.core.custom_architectures import GroupedQueryAttention, MixedLayerConfig, MixtureOfHeadAttention

HEADS = 4


def make_pair(top_k, router_bias):
    """A MoH layer whose router always outputs router_bias, and a GQA layer with the same weights."""
    torch.manual_seed(0)
    config = MixedLayerConfig(hidden_size=32, num_attention_heads=HEADS, num_kv_heads=2, num_hidden_layers=1,
                              layer_types=["moh_attention"], intermediate_size=64, moh_top_k=top_k)
    moh = MixtureOfHeadAttention(config).eval()
    router_out = moh.head_router[-1]
    with torch.no_grad():
        router_out.weight.zero_()
        router_out.bias.copy_(torch.tensor(router_bias))

    gqa = GroupedQueryAttention(config).eval()
    gqa.load_state_dict({k: v for k, v in moh.state_dict().items() if not k.startswith("head_router")})
    return moh, gqa


def test_all_heads_with_equal_gates_is_multi_head_attention():
    """With every head selected at gate 1/H, MoH is ordinary attention with all H heads, scaled by 1/H."""
    moh, gqa = make_pair(top_k=HEADS, router_bias=[0.0] * HEADS)
    x = torch.randn(2, 9, 32)
    with torch.no_grad():
        torch.testing.assert_close(moh(x), gqa(x) / HEADS)


def test_unselected_heads_contribute_nothing():
    """Routing every token to head 2 alone equals attention with the other heads' output columns removed."""
    moh, gqa = make_pair(top_k=1, router_bias=[0.0, 0.0, 10.0, 0.0])
    head_dim = 32 // HEADS
    with torch.no_grad():
        keep = torch.zeros(32)
        keep[2 * head_dim:3 * head_dim] = 1
        gqa.o_proj.weight.mul_(keep)

    x = torch.randn(2, 9, 32)
    with torch.no_grad():
        torch.testing.assert_close(moh(x), gqa(x))


def test_output_has_more_than_one_heads_capacity():
    """The old layer summed heads into one head_dim vector before o_proj, so its input had rank head_dim."""
    torch.manual_seed(0)
    config = MixedLayerConfig(hidden_size=32, num_attention_heads=HEADS, num_kv_heads=2, num_hidden_layers=1,
                              layer_types=["moh_attention"], intermediate_size=64, moh_top_k=2)
    moh = MixtureOfHeadAttention(config).eval()
    captured = {}
    moh.o_proj.register_forward_hook(lambda module, inputs, output: captured.setdefault("input", inputs[0]))
    with torch.no_grad():
        moh(torch.randn(1, 64, 32))
    assert torch.linalg.matrix_rank(captured["input"][0]) > 32 // HEADS


def test_router_and_heads_receive_gradients():
    torch.manual_seed(0)
    config = MixedLayerConfig(hidden_size=32, num_attention_heads=HEADS, num_kv_heads=2, num_hidden_layers=1,
                              layer_types=["moh_attention"], intermediate_size=64, moh_top_k=2)
    moh = MixtureOfHeadAttention(config).train()
    moh(torch.randn(2, 9, 32)).pow(2).sum().backward()
    for name, param in moh.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, name
