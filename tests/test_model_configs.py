"""
Tests that every preset in model_configs.yaml is valid and that pretraining
uses the config's training settings as given.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pytest
import yaml

from scripts.training.core.pipeline_utils import load_model_config
from scripts.training.pipeline.pretrain import build_model_config, create_training_arguments

PRESETS = list(yaml.safe_load((project_root / "model_configs.yaml").read_text())['model_training_configs'])


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_builds_valid_model_config(preset):
    """Builds the config only (not the model), so even the 48b preset is cheap to check."""
    config = load_model_config(preset)
    model_config = build_model_config(config, vocab_size=config['vocab_size'])

    assert len(model_config.layer_types) == config['num_hidden_layers']
    assert model_config.hidden_size % model_config.num_attention_heads == 0
    assert model_config.num_attention_heads % model_config.num_kv_heads == 0


def test_build_model_config_passes_moe_settings_through():
    config = load_model_config('guac-3b')
    model_config = build_model_config(config, vocab_size=100)

    assert model_config.num_experts == config['num_experts']
    assert model_config.num_shared_experts == config['num_shared_experts']
    assert model_config.tie_word_embeddings is True


def test_training_arguments_use_config_values_as_given(tmp_path):
    config = {'pdbs_train': 16, 'pdbs_eval': 24, 'grad_accum_steps': 1, 'num_train_epochs': 3, 'fp16': False}
    args = create_training_arguments(config, str(tmp_path))

    assert args.per_device_train_batch_size == 16
    assert args.per_device_eval_batch_size == 24
    assert args.gradient_accumulation_steps == 1
    assert args.max_steps == -1
    assert args.num_train_epochs == 3


def test_training_arguments_respect_explicit_max_steps(tmp_path):
    args = create_training_arguments({'fp16': False}, str(tmp_path), max_steps=50)
    assert args.max_steps == 50


def test_build_model_config_passes_layer_type_settings_through():
    config = {**load_model_config('guac-43m'), 'mamba_d_state': 32, 'based_feature_dim': 8,
              'based_chunk_size': 16, 'conv_kernel_size': 5, 'sliding_window_size': 128}
    model_config = build_model_config(config, vocab_size=100)

    assert model_config.mamba_d_state == 32
    assert model_config.based_feature_dim == 8
    assert model_config.based_chunk_size == 16
    assert model_config.conv_kernel_size == 5
    assert model_config.sliding_window_size == 128
