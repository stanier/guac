"""
Tests for the shared block structure: every layer type is
x + mixer(norm(x)) followed by x + ffn(norm(x)), with only the mixer varying,
plus the per-layer FFN options and the parameter-matching helper.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch

from scripts.training.core.custom_architectures import (
    MIXERLESS_LAYER_TYPES, SUPPORTED_LAYER_TYPES, MixedLayerConfig, MixedLayerForCausalLM,
    MixedTransformerLayer, block_flops_per_token, block_parameter_counts, flop_matched_ffn_hidden_sizes,
    mixer_flops_per_token, model_flops_per_token, param_matched_ffn_hidden_sizes,
)
from scripts.training.core.pipeline_utils import load_model_config
from scripts.training.pipeline.pretrain import build_model_config

HIDDEN = 32
MIXER_TYPES = [t for t in SUPPORTED_LAYER_TYPES if t not in MIXERLESS_LAYER_TYPES]


def make_config(layer_types, **overrides):
    kwargs = dict(
        vocab_size=50, hidden_size=HIDDEN, num_attention_heads=4, num_kv_heads=2,
        num_hidden_layers=len(layer_types), layer_types=layer_types, intermediate_size=96,
        max_position_embeddings=64, num_experts=2, num_experts_per_token=1,
    )
    kwargs.update(overrides)
    return MixedLayerConfig(**kwargs)


@pytest.mark.parametrize("layer_type", SUPPORTED_LAYER_TYPES)
def test_every_layer_type_has_the_same_block_shape(layer_type):
    layer = MixedTransformerLayer(make_config([layer_type]), 0)

    submodules = {name.split(".")[0] for name, _ in layer.named_parameters()}
    has_mixer = layer_type not in MIXERLESS_LAYER_TYPES
    assert submodules == ({"mixer_norm", "mixer"} if has_mixer else set()) | {"ffn_norm", "ffn"}


@pytest.mark.parametrize("layer_type", SUPPORTED_LAYER_TYPES)
def test_block_is_pre_norm_mixer_then_pre_norm_ffn(layer_type):
    torch.manual_seed(0)
    layer = MixedTransformerLayer(make_config([layer_type]), 0).eval()
    x = torch.randn(2, 9, HIDDEN)
    mask = torch.tensor([[1] * 9, [1] * 6 + [0] * 3])

    with torch.no_grad():
        expected = x
        if layer.mixer is not None:
            expected = expected + layer.mixer(layer.mixer_norm(expected), mask)
        expected = expected + layer.ffn(layer.ffn_norm(expected))
        actual = layer(x, mask)

    torch.testing.assert_close(actual, expected)


def test_full_and_flash_attention_are_the_same_block():
    x = torch.randn(2, 9, HIDDEN)
    outputs = []
    for layer_type in ("full_attention", "flash_attention"):
        torch.manual_seed(0)
        layer = MixedTransformerLayer(make_config([layer_type]), 0).eval()
        with torch.no_grad():
            outputs.append(layer(x))
    torch.testing.assert_close(outputs[0], outputs[1])


def test_ffn_type_none_drops_the_ffn():
    """As in the original Mamba-2 blocks, which have no FFN."""
    torch.manual_seed(0)
    layer = MixedTransformerLayer(make_config(["mamba2"], ffn_types=["none"]), 0).eval()
    assert layer.ffn is None and layer.ffn_norm is None

    x = torch.randn(2, 9, HIDDEN)
    with torch.no_grad():
        torch.testing.assert_close(layer(x), x + layer.mixer(layer.mixer_norm(x)))


@pytest.mark.parametrize("layer_type", MIXER_TYPES)
def test_moe_ffn_can_follow_any_mixer(layer_type):
    layer = MixedTransformerLayer(make_config([layer_type], ffn_types=["moe"]), 0)
    assert hasattr(layer.ffn, "router")
    layer(torch.randn(1, 5, HIDDEN)).sum().backward()
    assert layer.ffn.router.weight.grad is not None


def test_ffn_hidden_sizes_set_swiglu_width_per_layer():
    model = MixedLayerForCausalLM(make_config(["conv", "conv"], ffn_hidden_sizes=[16, 40]))
    assert [layer.ffn.up_proj.out_features for layer in model.layers] == [16, 40]


def test_default_swiglu_width_matches_a_two_matrix_ffn():
    config = make_config(["mlp_only"])
    assert config.ffn_hidden_sizes == [64]  # 2/3 of intermediate_size 96, so 3 * 64 == 2 * 96 rows


@pytest.mark.parametrize("overrides, message", [
    (dict(layer_types=["mlp_only"], ffn_types=["none"]), "neither a mixer nor an FFN"),
    (dict(layer_types=["shared_expert_moe"], ffn_types=["none"]), "neither a mixer nor an FFN"),
    (dict(layer_types=["conv"], ffn_types=["gelu"]), "Unsupported FFN types"),
    (dict(layer_types=["conv"], ffn_types=["swiglu", "none"]), "ffn_types length"),
    (dict(layer_types=["conv"], ffn_hidden_sizes=[8, 8]), "ffn_hidden_sizes length"),
    (dict(layer_types=["conv"], ffn_hidden_sizes=[0]), "must all be positive"),
])
def test_config_rejects_invalid_ffn_settings(overrides, message):
    layer_types = overrides.pop("layer_types")
    with pytest.raises(ValueError, match=message):
        make_config(layer_types, **overrides)


def test_ffn_settings_survive_save_and_load(tmp_path):
    torch.manual_seed(0)
    config = make_config(["mamba2", "full_attention", "mlp_only"],
                         ffn_types=["none", "moe", "swiglu"], ffn_hidden_sizes=[8, 8, 24])
    model = MixedLayerForCausalLM(config).eval()
    input_ids = torch.randint(0, 50, (2, 10))

    model.save_pretrained(tmp_path)
    reloaded = MixedLayerForCausalLM.from_pretrained(tmp_path).eval()

    assert reloaded.config.ffn_types == config.ffn_types
    assert reloaded.config.ffn_hidden_sizes == config.ffn_hidden_sizes
    with torch.no_grad():
        torch.testing.assert_close(reloaded(input_ids=input_ids).logits, model(input_ids=input_ids).logits)


@pytest.mark.parametrize("reference", ["full_attention", "conv", "mlp_only"])
def test_param_matching_equalizes_block_sizes(reference):
    config = make_config(list(SUPPORTED_LAYER_TYPES), intermediate_size=300)
    target = block_parameter_counts(make_config([reference], intermediate_size=300))[0]

    config.ffn_hidden_sizes = param_matched_ffn_hidden_sizes(config, reference)
    counts = block_parameter_counts(config)

    for layer_type, ffn_type, count in zip(config.layer_types, config.ffn_types, counts):
        if ffn_type == "swiglu":
            assert abs(count - target) <= 3 * HIDDEN / 2, f"{layer_type}: {count} vs {target}"
    # The reference type itself keeps the default width
    assert config.ffn_hidden_sizes[config.layer_types.index(reference)] == 200


def test_param_matching_leaves_moe_and_ffn_free_layers_alone():
    config = make_config(["conv", "mamba2", "shared_expert_moe"], ffn_types=["swiglu", "none", "moe"],
                         ffn_hidden_sizes=[7, 11, 13], intermediate_size=300)
    assert param_matched_ffn_hidden_sizes(config)[1:] == [11, 13]


def test_param_matching_rejects_a_mixer_larger_than_the_target():
    """With a tiny FFN, a Mamba-2 mixer alone outweighs a whole conv block."""
    config = make_config(["conv", "mamba2"], intermediate_size=3)
    with pytest.raises(ValueError, match="no FFN width can match"):
        param_matched_ffn_hidden_sizes(config, "conv")


def test_preset_can_be_param_matched():
    config = load_model_config("guac-43m")
    model_config = build_model_config({**config, "param_match": "conv"}, vocab_size=100)

    counts = block_parameter_counts(model_config)
    assert max(counts) - min(counts) <= 3 * model_config.hidden_size
    assert len(set(model_config.ffn_hidden_sizes)) == 2  # conv and mamba2 layers get different widths


def test_preset_rejects_param_match_with_explicit_widths():
    config = load_model_config("guac-43m")
    with pytest.raises(ValueError, match="at most one of param_match, flop_match and ffn_hidden_sizes"):
        build_model_config({**config, "param_match": True, "ffn_hidden_sizes": [8] * 4}, vocab_size=100)


def test_attention_flops_match_the_hand_count():
    """Per token: q and o projections 2*d*d each, k and v 2*d*kv_dim each, then QK^T and AV 2*L*d each."""
    config = make_config(["full_attention"])
    d, kv_dim, length = HIDDEN, 2 * (HIDDEN // 4), 16
    expected = 2 * (2 * d * d) + 2 * (2 * d * kv_dim) + 2 * (2 * length * d)
    assert mixer_flops_per_token(config, "full_attention", length) == expected


def test_flops_per_token_grow_with_length_only_for_full_attention_and_mamba2_chunks():
    config = make_config(["full_attention", "sliding_window_attention", "conv", "based", "mamba2"],
                         sliding_window_size=8, mamba_chunk_size=8, based_chunk_size=8)
    short, long = (block_flops_per_token(config, n) for n in (64, 128))
    attention, window, conv, based, mamba = (b / a for a, b in zip(short, long))
    assert attention > 1.3                                    # quadratic in length
    assert window == conv == based == 1.0                     # linear in length
    # The inter-chunk pass of the SSD scan is quadratic in the number of chunks (a known issue)
    assert 1.0 < mamba < attention


@pytest.mark.parametrize("reference", ["full_attention", "conv"])
def test_flop_matching_equalizes_block_flops(reference):
    config = make_config(list(SUPPORTED_LAYER_TYPES), intermediate_size=300, mamba_d_state=16, mamba_headdim=16,
                         mamba_chunk_size=8, based_chunk_size=8, based_feature_dim=4)
    config.ffn_hidden_sizes = flop_matched_ffn_hidden_sizes(config, reference, seq_len=32)
    flops = block_flops_per_token(config, 32)
    target = flops[config.layer_types.index(reference)]
    for layer_type, ffn_type, count in zip(config.layer_types, config.ffn_types, flops):
        if ffn_type == "swiglu":
            assert abs(count - target) <= 3 * HIDDEN, f"{layer_type}: {count} vs {target}"


def test_model_flops_add_the_output_projection():
    config = make_config(["conv", "mamba2"])
    assert model_flops_per_token(config, 16) == sum(block_flops_per_token(config, 16)) + 2 * HIDDEN * 50


def test_moe_flops_count_only_the_experts_a_token_uses():
    config = make_config(["shared_expert_moe"], num_experts=8, num_experts_per_token=2, num_shared_experts=1)
    expert = 6 * HIDDEN * 64  # SwiGLU of width 2/3 * 96
    assert block_flops_per_token(config, 16) == [2 * HIDDEN * 8 + 3 * expert]


def test_preset_can_be_flop_matched():
    config = load_model_config("guac-43m")
    model_config = build_model_config({**config, "flop_match": "conv", "max_sequence_length": 128}, vocab_size=100)
    flops = block_flops_per_token(model_config, 128)
    assert max(flops) - min(flops) <= 6 * model_config.hidden_size
