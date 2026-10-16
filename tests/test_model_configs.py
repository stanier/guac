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

from guac.core.pipeline_utils import load_model_config
from guac.pipeline.pretrain import build_model_config, create_training_arguments

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


def test_preset_numbers_in_scientific_notation_are_numbers():
    """PyYAML reads 8e-4 as a string; presets must still give floats."""
    config = load_model_config('guac-3b')
    assert isinstance(config['learning_rate'], float)


def test_training_arguments_share_stage_conventions(tmp_path):
    from guac.core.pipeline_utils import make_training_arguments
    import torch

    args = make_training_arguments(str(tmp_path), fp16=True, report_to='none', per_device_train_batch_size=2)
    assert args.fp16 == torch.cuda.is_available()  # Trainer refuses fp16 without CUDA
    assert args.report_to == []
    assert args.dataloader_drop_last is False      # keeps evaluation sets smaller than a batch
    assert make_training_arguments(str(tmp_path), save_total_limit=1).save_total_limit == 1


def test_midtraining_text_tasks_load_and_tokenize_datasets_from_disk(tmp_path):
    from datasets import Dataset
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from guac.pipeline.midtrain import prepare_task_dataset

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    Dataset.from_dict({'text': ["a b a", "", "b b"]}).save_to_disk(str(tmp_path / "texts"))

    tokenized = prepare_task_dataset('custom', tokenizer, [str(tmp_path / "texts")])
    assert tokenized['input_ids'] == [[1, 2, 1], [2, 2]]
    with pytest.raises(ValueError, match="needs --datasets"):
        prepare_task_dataset('custom', tokenizer)
