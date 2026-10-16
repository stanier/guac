"""
Correctness tests for MixedLayerForCausalLM and its layer types.

Covers weight tying across save/load, attention-mask handling, and config
validation. Causality is covered separately in test_causality.py.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch

from scripts.training.core.custom_architectures import (
    SUPPORTED_LAYER_TYPES, MixedLayerConfig, MixedLayerForCausalLM, MixedTransformerLayer,
)

# Layer types that use softmax attention over a key padding mask
SOFTMAX_ATTENTION_TYPES = ["full_attention", "flash_attention", "moh_attention"]


def make_config(layer_types, **overrides):
    kwargs = dict(
        vocab_size=50, hidden_size=32, num_attention_heads=4, num_kv_heads=2,
        num_hidden_layers=len(layer_types), layer_types=layer_types, intermediate_size=64,
        max_position_embeddings=64, num_experts=2, num_experts_per_token=1,
    )
    kwargs.update(overrides)
    return MixedLayerConfig(**kwargs)


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_save_load_round_trip_preserves_logits(tmp_path, tie_word_embeddings):
    torch.manual_seed(0)
    model = MixedLayerForCausalLM(
        make_config(list(SUPPORTED_LAYER_TYPES), tie_word_embeddings=tie_word_embeddings)
    ).eval()
    input_ids = torch.randint(0, 50, (2, 12))

    model.save_pretrained(tmp_path)
    reloaded = MixedLayerForCausalLM.from_pretrained(tmp_path).eval()

    with torch.no_grad():
        torch.testing.assert_close(reloaded(input_ids=input_ids).logits, model(input_ids=input_ids).logits)
    shares_storage = reloaded.lm_head.weight.data_ptr() == reloaded.token_embeddings.weight.data_ptr()
    assert shares_storage == tie_word_embeddings


def test_tied_model_shares_embeddings_from_construction():
    model = MixedLayerForCausalLM(make_config(["conv"], tie_word_embeddings=True))
    assert model.lm_head.weight is model.token_embeddings.weight


@pytest.mark.parametrize("layer_type", SUPPORTED_LAYER_TYPES)
def test_all_ones_mask_matches_no_mask(layer_type):
    torch.manual_seed(0)
    layer = MixedTransformerLayer(make_config([layer_type]), layer_type).eval()
    hidden_states = torch.randn(2, 10, 32)

    with torch.no_grad():
        unmasked = layer(hidden_states)
        masked = layer(hidden_states, torch.ones(2, 10, dtype=torch.long))

    assert not torch.isnan(masked).any()
    torch.testing.assert_close(masked, unmasked)


@pytest.mark.parametrize("layer_type", SUPPORTED_LAYER_TYPES)
def test_right_padding_leaves_real_tokens_unchanged(layer_type):
    """Padding appended after a sequence must not change outputs at its real positions."""
    torch.manual_seed(0)
    layer = MixedTransformerLayer(make_config([layer_type]), layer_type).eval()
    real = torch.randn(1, 7, 32)
    padded = torch.cat([real, torch.randn(1, 3, 32)], dim=1)
    mask = torch.tensor([[1] * 7 + [0] * 3])

    with torch.no_grad():
        expected = layer(real)
        actual = layer(padded, mask)

    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual[:, :7], expected)


@pytest.mark.parametrize("layer_type", SOFTMAX_ATTENTION_TYPES)
def test_left_padding_is_masked_without_nan(layer_type):
    """
    With left padding, the first positions have no real token to attend to.
    Those rows must not produce NaN, and real tokens must ignore the padding.
    """
    torch.manual_seed(0)
    layer = MixedTransformerLayer(make_config([layer_type]), layer_type).eval()
    real = torch.randn(1, 7, 32)
    padded = torch.cat([torch.randn(1, 3, 32), real], dim=1)
    mask = torch.tensor([[0] * 3 + [1] * 7])

    with torch.no_grad():
        expected = layer(real)
        actual = layer(padded, mask)

    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual[:, 3:], expected)


def test_model_loss_with_all_ones_mask_matches_no_mask():
    torch.manual_seed(0)
    model = MixedLayerForCausalLM(make_config(list(SUPPORTED_LAYER_TYPES))).eval()
    input_ids = torch.randint(0, 50, (2, 12))

    with torch.no_grad():
        unmasked = model(input_ids=input_ids, labels=input_ids).loss
        masked = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids).loss

    assert torch.isfinite(masked)
    torch.testing.assert_close(masked, unmasked)


def test_unsupported_layer_type_is_rejected_by_config():
    with pytest.raises(ValueError, match="Unsupported layer types"):
        make_config(["attention"])


def test_layer_types_length_must_match_num_hidden_layers():
    with pytest.raises(ValueError, match="must match num_hidden_layers"):
        make_config(["conv", "conv"], num_hidden_layers=3)


def test_based_layer_builds_without_num_kv_heads():
    config = make_config(["based"], num_kv_heads=None)
    assert config.based_num_key_value_heads == config.num_attention_heads
    MixedTransformerLayer(config, "based")
