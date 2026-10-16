#!/usr/bin/env python3
"""
Shared utilities for Guac training pipeline stages.

This module contains common functions used across pretrain, midtrain, and posttrain stages
to reduce code duplication and maintain consistency.
"""

import json
import math
import os
import subprocess
import sys
import time
import yaml
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path
from transformers import Trainer, TrainerCallback, TrainingArguments, EarlyStoppingCallback

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Project setup
project_root = Path(__file__).parent.parent.parent.parent


def as_number(value: Any) -> Any:
    """
    value, with a string that spells a number converted to it. PyYAML follows
    YAML 1.1, which reads 3e-3 (no decimal point) as a string.
    """
    if isinstance(value, str):
        for convert in (int, float):
            try:
                return convert(value)
            except ValueError:
                pass
    return value


def numbers_in(value: Any) -> Any:
    """value with every numeric string inside it (in nested dicts and lists too) converted by as_number."""
    if isinstance(value, dict):
        return {key: numbers_in(item) for key, item in value.items()}
    if isinstance(value, list):
        return [numbers_in(item) for item in value]
    return as_number(value)


def load_model_config(config_name: str) -> Dict[str, Any]:
    """Load a model_configs.yaml preset, merged over the defaults. Former preset names still resolve."""
    config_path = project_root / "model_configs.yaml"

    with open(config_path, 'r') as f:
        configs = numbers_in(yaml.safe_load(f))

    aliases = configs.get('preset_aliases') or {}
    if config_name in aliases:
        print(f"Note: preset '{config_name}' is now called '{aliases[config_name]}' (renamed to match its size)")
        config_name = aliases[config_name]

    if config_name not in configs['model_training_configs']:
        raise ValueError(f"Model config '{config_name}' not found. Available: {list(configs['model_training_configs'].keys())}")

    # Merge defaults with specific config
    config = configs['defaults'].copy()
    config.update(configs['model_training_configs'][config_name])

    return config


def load_base_model(model_path: str) -> tuple:
    """
    Load pre-trained or mid-trained base model and tokenizer.

    Args:
        model_path: Path to model directory

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If model loading fails, or if the checkpoint's weights don't
            match the current architecture code (from_pretrained alone would
            randomly initialize the missing weights and carry on)
    """
    print(f"Loading base model from {model_path}")

    try:
        from transformers import AutoTokenizer
        from .custom_architectures import MixedLayerForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model, loading_info = MixedLayerForCausalLM.from_pretrained(model_path, output_loading_info=True)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")

    mismatched = {kind: loading_info[kind] for kind in ('missing_keys', 'unexpected_keys', 'mismatched_keys')
                  if loading_info.get(kind)}
    if mismatched:
        details = "\n".join(f"  {kind}: {', '.join(map(str, keys[:5]))}{' ...' if len(keys) > 5 else ''}"
                            for kind, keys in mismatched.items())
        raise ValueError(f"Checkpoint at {model_path} doesn't match the current model code, so some weights "
                         f"would be randomly initialized. It was probably saved by an older version of a layer "
                         f"type; retrain it.\n{details}")

    print(f"Loaded model with {model.num_parameters():,} parameters")
    return model, tokenizer


def create_standard_data_collator(training_args: TrainingArguments, tokenizer: Any) -> Any:
    """
    Create standard data collator for language modeling.

    Args:
        training_args: Training arguments (used to check fp16 setting)
        tokenizer: Tokenizer (required by recent transformers versions)

    Returns:
        DataCollatorForLanguageModeling instance
    """
    from transformers import DataCollatorForLanguageModeling

    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )


def get_output_directory(
    training_type: str,
    model_name: str,
    output_dir: Optional[str] = None,
    **kwargs
) -> str:
    """
    Get output directory with path manager fallback.

    Args:
        training_type: Type of training ('pretraining', 'midtraining', 'posttraining')
        model_name: Name of model or base model
        output_dir: Optional explicit output directory
        **kwargs: Additional parameters (e.g., task, mode)

    Returns:
        Output directory path
    """
    if output_dir:
        return output_dir

    try:
        from .path_manager import get_model_output_path
        return str(get_model_output_path(training_type, model_name, **kwargs))
    except ImportError:
        suffix = ""
        if 'task' in kwargs:
            suffix = f"_{kwargs['task']}"
        elif 'mode' in kwargs:
            suffix = f"_{kwargs['mode']}"

        default_dir = f"./models/{training_type.replace('training', 'train')}{suffix}_{model_name}"
        os.makedirs(default_dir, exist_ok=True)
        return default_dir


def run_training_loop(trainer: Any, resume_checkpoint: Optional[str] = None) -> Any:
    """
    Run training, resuming from resume_checkpoint if given. Returns trainer.train()'s TrainOutput.
    """
    return trainer.train(resume_from_checkpoint=resume_checkpoint)


def apply_debug_mode(dataset: Any, max_samples: int = 1000) -> Any:
    """
    Apply debug mode by reducing dataset size.

    Args:
        dataset: Dataset to reduce
        max_samples: Maximum number of samples to keep

    Returns:
        Reduced dataset
    """
    print(f"Debug mode: using smaller dataset ({max_samples} samples)")
    return dataset.select(range(min(max_samples, len(dataset))))


def save_trained_model(
    trainer: Any,
    tokenizer: Any,
    output_dir: str,
    save_config: bool = True
) -> None:
    """
    Save trained model, tokenizer, and optionally config.

    Args:
        trainer: HuggingFace Trainer instance
        tokenizer: Tokenizer to save
        output_dir: Directory to save to
        save_config: Whether to save model config
    """
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    if save_config and hasattr(trainer.model, 'config'):
        trainer.model.config.save_pretrained(output_dir)

# ---------------------------------------------------------------------------
# Run records: metrics, perplexity and the settings needed to reproduce a run
# ---------------------------------------------------------------------------


class GuacTrainer(Trainer):
    """Trainer that adds perplexity (exp of the loss) next to every logged loss."""

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        for key in ('loss', 'eval_loss'):
            if key in logs:
                # Capped so a diverged run logs a large number rather than overflowing
                logs[key.replace('loss', 'perplexity')] = math.exp(min(logs[key], 50.0))
        super().log(logs, *args, **kwargs)


class MetricsLoggerCallback(TrainerCallback):
    """
    Appends training logs and evaluation results to a JSONL file, one object per line.

    Evaluation results are written from on_evaluate rather than on_log, so they
    include metrics that callbacks listed before this one add during evaluation
    (such as midtraining's held-out accuracy).
    """

    def __init__(self, path: str):
        self.path = path

    def write(self, state, values):
        if not values or not state.is_world_process_zero:
            return
        entry = {'step': state.global_step, 'epoch': state.epoch, 'time': time.time(), **values}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, 'a') as f:
            f.write(json.dumps(entry, default=float) + '\n')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and not any(key.startswith('eval_') for key in logs):
            self.write(state, logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.write(state, metrics)


def git_state() -> Dict[str, Any]:
    """The checked-out commit and whether the working tree has uncommitted changes."""
    def git(*args):
        result = subprocess.run(['git', *args], cwd=project_root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    status = git('status', '--porcelain', '--untracked-files=no')
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(status) if status is not None else None}


def write_run_info(output_dir: str, stage: str, config: Dict[str, Any], cli_args: Any = None,
                   model: Any = None, seed: Optional[int] = None, **extra: Any) -> str:
    """
    Write run_info.json: everything needed to reproduce a run. That is the
    resolved config (preset merged with overrides), command-line arguments,
    seed, git commit, library versions, model size and any extra fields.
    """
    import platform
    import transformers
    import datasets

    info = {
        'stage': stage,
        'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'command': ' '.join(sys.argv),
        'cli_args': vars(cli_args) if cli_args is not None and hasattr(cli_args, '__dict__') else cli_args,
        'config': config,
        'seed': seed,
        'git': git_state(),
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'transformers': transformers.__version__, 'datasets': datasets.__version__},
        **extra,
    }
    if model is not None:
        info['parameters'] = model.num_parameters()
        info['model_config'] = model.config.to_dict()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'run_info.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    return path
