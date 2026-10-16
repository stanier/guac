#!/usr/bin/env python3
"""
Causality test for every layer type supported by MixedTransformerLayer.

Perturbs the input at a single position and checks that outputs at all
earlier positions are unchanged. Any difference means the layer leaks
future tokens, which breaks next-token-prediction training.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from scripts.training.core.custom_architectures import MixedLayerConfig, MixedTransformerLayer

LAYER_TYPES = [
    "full_attention",
    "conv",
    "mlp_only",
    "mamba2",
    "based",
    "flash_attention",
    "moh_attention",
    "shared_expert_moe",
]

SEQ_LEN = 32
PERTURB_POS = 15
TOLERANCE = 1e-5


def check_layer_causality(layer_type):
    """Return the max abs difference at positions < PERTURB_POS after perturbing PERTURB_POS."""
    torch.manual_seed(0)
    config = MixedLayerConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=4,
        num_kv_heads=2,
        num_hidden_layers=1,
        layer_types=[layer_type],
        intermediate_size=128,
    )
    layer = MixedTransformerLayer(config, layer_type).eval()

    x = torch.randn(2, SEQ_LEN, config.hidden_size)
    x_perturbed = x.clone()
    x_perturbed[:, PERTURB_POS] += 10.0 * torch.randn(2, config.hidden_size)

    with torch.no_grad():
        out = layer(x)
        out_perturbed = layer(x_perturbed)

    past_diff = (out[:, :PERTURB_POS] - out_perturbed[:, :PERTURB_POS]).abs().max().item()
    # Sanity check: the perturbed position itself should change, otherwise the test proves nothing
    self_diff = (out[:, PERTURB_POS] - out_perturbed[:, PERTURB_POS]).abs().max().item()
    return past_diff, self_diff


def test_causality():
    """Assert every layer type is causal."""
    print("Testing causality of MixedTransformerLayer types")
    print("=" * 50)

    failures = []
    for layer_type in LAYER_TYPES:
        past_diff, self_diff = check_layer_causality(layer_type)
        ok = past_diff <= TOLERANCE and self_diff > TOLERANCE
        status = "✅" if ok else "❌"
        print(f"  {status} {layer_type:<20} past diff: {past_diff:.2e}  self diff: {self_diff:.2e}")
        if not ok:
            failures.append(layer_type)

    assert not failures, f"Non-causal layer types: {failures}"
    print("\n✅ All layer types are causal!")
    return True


if __name__ == "__main__":
    try:
        test_causality()
    except AssertionError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
