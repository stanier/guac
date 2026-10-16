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

import pytest
import torch
from scripts.training.core.custom_architectures import SUPPORTED_LAYER_TYPES, MixedLayerConfig, MixedTransformerLayer

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


@pytest.mark.parametrize("layer_type", SUPPORTED_LAYER_TYPES)
def test_causality(layer_type):
    """Assert the layer type is causal, and that the perturbation is visible at its own position."""
    past_diff, self_diff = check_layer_causality(layer_type)
    assert past_diff <= TOLERANCE, f"{layer_type} leaks future tokens (past diff {past_diff:.2e})"
    # Otherwise the test proves nothing
    assert self_diff > TOLERANCE, f"{layer_type} ignores its input (self diff {self_diff:.2e})"
