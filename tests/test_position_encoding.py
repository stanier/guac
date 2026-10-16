"""
Tests for position encoding: rotary embeddings in the attention mixers, the
optional absolute position table, and preset names that match model sizes.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch
import yaml

from guac.core.custom_architectures import (
    ROTARY_LAYER_TYPES, MixedLayerConfig, MixedLayerForCausalLM, MixedTransformerLayer, RotaryEmbedding,
)
from guac.core.pipeline_utils import load_model_config
from guac.pipeline.pretrain import build_model_config

project_root = Path(__file__).parent.parent


def make_config(layer_types, **overrides):
    kwargs = dict(vocab_size=50, hidden_size=32, num_attention_heads=4, num_kv_heads=2,
                  num_hidden_layers=len(layer_types), layer_types=layer_types, intermediate_size=64,
                  max_position_embeddings=64, sliding_window_size=64)
    kwargs.update(overrides)
    return MixedLayerConfig(**kwargs)


def test_rotary_scores_depend_only_on_relative_position():
    torch.manual_seed(0)
    rotary = RotaryEmbedding(8)
    q, k = torch.randn(1, 1, 6, 8), torch.randn(1, 1, 6, 8)

    q0, k0 = rotary(q, k)
    q1, k1 = rotary(q, k, offset=37)
    torch.testing.assert_close(q0 @ k0.transpose(-1, -2), q1 @ k1.transpose(-1, -2), atol=1e-4, rtol=1e-4)
    assert not torch.allclose(q0, q1)  # the vectors themselves do move


def test_rotary_preserves_norms():
    rotary = RotaryEmbedding(8)
    q = torch.randn(2, 3, 5, 8)
    rotated, _ = rotary(q, q)
    torch.testing.assert_close(rotated.norm(dim=-1), q.norm(dim=-1))


@pytest.mark.parametrize("layer_type", ROTARY_LAYER_TYPES)
def test_attention_sees_token_order_only_with_rope(layer_type):
    """
    Without position information, causal attention at the last position is a
    weighted average over a set, so swapping two earlier tokens can't change it.
    With RoPE it does.
    """
    x = torch.randn(1, 8, 32)
    swapped = x.clone()
    swapped[0, [1, 4]] = x[0, [4, 1]]

    changes = {}
    for use_rope in (False, True):
        torch.manual_seed(0)
        layer = MixedTransformerLayer(make_config([layer_type], use_rope=use_rope, moh_top_k=4), 0).eval()
        with torch.no_grad():
            changes[use_rope] = (layer(x)[0, -1] - layer(swapped)[0, -1]).abs().max().item()

    assert changes[False] < 1e-5
    assert changes[True] > 1e-3


def test_rope_adds_no_parameters():
    with_rope = MixedLayerForCausalLM(make_config(["full_attention"], use_rope=True))
    without = MixedLayerForCausalLM(make_config(["full_attention"], use_rope=False))
    assert with_rope.state_dict().keys() == without.state_dict().keys()


def test_absolute_position_table_is_off_by_default():
    model = MixedLayerForCausalLM(make_config(["conv"]))
    assert model.position_embeddings is None

    with_table = MixedLayerForCausalLM(make_config(["conv"], use_absolute_position_embeddings=True))
    assert with_table.num_parameters() - model.num_parameters() == 64 * 32


def test_absolute_position_table_round_trips(tmp_path):
    torch.manual_seed(0)
    model = MixedLayerForCausalLM(make_config(["full_attention"], use_absolute_position_embeddings=True)).eval()
    input_ids = torch.randint(0, 50, (2, 10))
    model.save_pretrained(tmp_path)
    reloaded = MixedLayerForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(input_ids=input_ids).logits, model(input_ids=input_ids).logits)


def test_odd_head_size_with_rope_is_rejected():
    with pytest.raises(ValueError, match="even head size"):
        make_config(["full_attention"], hidden_size=36, num_attention_heads=4, num_kv_heads=4)
    make_config(["conv"], hidden_size=36, num_attention_heads=4, num_kv_heads=4)  # no attention, no RoPE


def test_presets_pass_position_settings_through():
    config = {**load_model_config('guac-43m'), 'use_rope': False, 'rope_theta': 500000,
              'use_absolute_position_embeddings': True}
    model_config = build_model_config(config, vocab_size=100)
    assert model_config.use_rope is False
    assert model_config.rope_theta == 500000
    assert model_config.use_absolute_position_embeddings is True


PRESETS = yaml.safe_load((project_root / "model_configs.yaml").read_text())
ALIASES = PRESETS['preset_aliases']


@pytest.mark.parametrize("old_name, new_name", ALIASES.items())
def test_former_preset_names_still_load(old_name, new_name):
    assert load_model_config(old_name) == load_model_config(new_name)


@pytest.mark.parametrize("preset", [name for name in PRESETS['model_training_configs']
                                    if name in ALIASES.values()])
def test_renamed_presets_match_their_names(preset):
    """The renamed presets carry their size in the name; check it stays within 10%."""
    config = load_model_config(preset)
    with torch.device("meta"):
        params = MixedLayerForCausalLM(build_model_config(config, config['vocab_size'])).num_parameters()
    size = preset.split('-')[1]
    named = float(size[:-1]) * (1e9 if size.endswith('b') else 1e6)
    assert abs(params - named) / named < 0.1, f"{preset} has {params:,} parameters"


def test_absolute_positions_ignore_left_padding():
    """Positions count real tokens, so batched generation with left padding sees the same positions."""
    torch.manual_seed(0)
    model = MixedLayerForCausalLM(make_config(["full_attention", "conv"], use_absolute_position_embeddings=True)).eval()
    real = torch.randint(1, 50, (1, 6))
    padded = torch.cat([torch.zeros(1, 4, dtype=torch.long), real], dim=1)
    mask = torch.tensor([[0] * 4 + [1] * 6])
    with torch.no_grad():
        expected = model(input_ids=real).logits
        torch.testing.assert_close(model(input_ids=padded, attention_mask=mask).logits[:, 4:], expected,
                                   atol=1e-5, rtol=1e-4)
        batched = model.generate(padded, attention_mask=mask, max_new_tokens=5, do_sample=False,
                                 pad_token_id=0, eos_token_id=None)
        alone = model.generate(real, max_new_tokens=5, do_sample=False, pad_token_id=0, eos_token_id=None)
    assert torch.equal(batched[0, 4:], alone[0])
