#!/usr/bin/env python3
"""
Shared utilities for Guac training pipeline stages.

This module contains common functions used across pretrain, midtrain, and posttrain stages
to reduce code duplication and maintain consistency.
"""

import os
import yaml
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path
from transformers import TrainingArguments, EarlyStoppingCallback

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Project setup
project_root = Path(__file__).parent.parent.parent.parent


def load_model_config(config_name: str) -> Dict[str, Any]:
    """Load model configuration from YAML."""
    config_path = project_root / "model_configs.yaml"

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    if config_name not in configs['model_training_configs']:
        raise ValueError(f"Model config '{config_name}' not found. Available: {list(configs['model_training_configs'].keys())}")

    # Merge defaults with specific config
    config = configs['defaults'].copy()
    config.update(configs['model_training_configs'][config_name])

    return config


def setup_gpu_optimizations(model: Any) -> Any:
    """Apply GPU optimizations to model."""
    # Clear GPU cache before model creation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Move to GPU with memory optimization
    if torch.cuda.is_available():
        model = model.cuda()
        # Enable gradient checkpointing if model supports it
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

    return model


def create_base_training_arguments(
    output_dir: str,
    config: Dict[str, Any],
    stage: str = "pretrain",
    enable_evaluation: bool = False
) -> TrainingArguments:
    """Create base training arguments, using the config's batch settings as given."""

    train_batch_size = config.get('pdbs_train', 16)
    eval_batch_size = config.get('pdbs_eval', 16)
    grad_accum = config.get('grad_accum_steps', 1)

    # Adjust settings based on stage
    if stage == "pretrain":
        # Pre-training: config's max_steps if set, else epochs; no evaluation
        max_steps = config.get('max_steps', -1)
        num_epochs = config.get('num_train_epochs', 1)
        learning_rate = float(config.get('learning_rate', 8e-4))
        eval_strategy = "no"
        save_steps = config.get('step_val', 100) * 10

    elif stage == "midtrain":
        # Mid-training: Conservative specialization
        max_steps = -1  # Use epochs
        num_epochs = 3
        learning_rate = 1e-5
        eval_strategy = "steps" if enable_evaluation else "no"
        save_steps = 500

    elif stage == "posttrain":
        # Post-training: Instruction tuning
        max_steps = -1  # Use epochs
        num_epochs = 2
        learning_rate = 5e-6
        eval_strategy = "steps" if enable_evaluation else "no"
        save_steps = 200

    else:
        raise ValueError(f"Unknown stage: {stage}")

    training_args = TrainingArguments(
        output_dir=output_dir,

        # Training schedule
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,

        # Optimization
        learning_rate=learning_rate,
        weight_decay=float(config.get('weight_decay', 0.01)),
        warmup_ratio=float(config.get('warmup_ratio', 0.05)),
        lr_scheduler_type=config.get('lr_scheduler_type', 'constant_with_warmup'),

        # Logging and saving
        logging_steps=config.get('step_val', 100),
        save_steps=save_steps,
        save_total_limit=3,

        # Performance optimizations
        fp16=config.get('fp16', True),
        gradient_checkpointing=True,  # Memory savings
        dataloader_drop_last=True,
        dataloader_num_workers=2,  # Reduced for memory
        dataloader_pin_memory=False,  # Disabled for memory

        # Evaluation
        eval_strategy=eval_strategy,
        eval_steps=save_steps // 2 if enable_evaluation else None,
        load_best_model_at_end=enable_evaluation,
        metric_for_best_model="eval_loss" if enable_evaluation else None,
        greater_is_better=False,

        # Other
        report_to=[],
        seed=42,
    )

    return training_args


def create_standard_callbacks(
    tokenizer: Any,
    config: Dict[str, Any],
    enable_profiling: bool = False,
    enable_early_stopping: bool = False
) -> List[Any]:
    """Create standard callbacks for training."""
    try:
        from .callbacks import CustomProgressCallback, ProfilingCallback
    except ImportError:
        from scripts.training.core.callbacks import CustomProgressCallback, ProfilingCallback

    callbacks = []

    # Progress callback
    callbacks.append(CustomProgressCallback(tokenizer, config.get('step_val', 100)))

    # Profiling callbacks if enabled
    if enable_profiling:
        callbacks.append(ProfilingCallback(
            log_every_n_steps=config.get('profiling_steps', 50),
            detailed_profiling=True
        ))

    # Early stopping (only if evaluation is enabled)
    if enable_early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        ))

    return callbacks


def save_model_and_tokenizer(trainer: Any, tokenizer: Any, output_dir: str) -> None:
    """Save final model and tokenizer with error handling."""
    try:
        # Save tokenizer and config first
        tokenizer.save_pretrained(output_dir)
        if hasattr(trainer.model, 'config'):
            trainer.model.config.save_pretrained(output_dir)

        # Save final model
        trainer.save_model()
        print(f"✅ Model and tokenizer saved to {output_dir}")

    except Exception as e:
        print(f"⚠️  Error saving model: {e}")
        # Try to save just the model state dict as fallback
        try:
            torch.save(trainer.model.state_dict(), os.path.join(output_dir, "model_state.pt"))
            print(f"💾 Saved model state dict as fallback")
        except Exception as e2:
            print(f"❌ Failed to save model: {e2}")


def print_training_info(
    stage: str,
    model_config: str,
    model: Any,
    dataset_info: str,
    train_samples: int,
    output_dir: str
) -> None:
    """Print standardized training information."""
    stage_emoji = {
        "pretrain": "🚀",
        "midtrain": "🎯",
        "posttrain": "🤖"
    }

    emoji = stage_emoji.get(stage, "🔧")
    stage_name = stage.replace("train", "-training").title()

    print(f"{emoji} Starting {stage_name}...")
    print(f"📊 Model: {model_config} ({model.num_parameters():,} parameters)")
    print(f"📚 Dataset: {dataset_info} ({train_samples:,} training samples)")
    print(f"💾 Output: {output_dir}")


def filter_short_sequences(dataset: Any, min_length: int = 10) -> Any:
    """Filter out sequences that are too short."""
    if hasattr(dataset, 'filter'):
        return dataset.filter(
            lambda x: len(x.get('input_ids', [])) > min_length,
            desc="Filtering short sequences"
        )
    return dataset


def create_cache_dir_name(
    prefix: str,
    tokenizer: Any = None,
    vocab_size: Optional[int] = None,
    max_length: Optional[int] = None,
    stage: str = "pretraining",
    **kwargs
) -> str:
    """Create consistent cache directory names using path manager."""
    try:
        from .path_manager import get_tokenized_dataset_path, create_cache_key

        # Build cache key components
        cache_kwargs = {}
        if tokenizer and hasattr(tokenizer, '__len__'):
            cache_kwargs['vocab'] = len(tokenizer)
        elif vocab_size:
            cache_kwargs['vocab'] = vocab_size

        if max_length:
            cache_kwargs['len'] = max_length

        cache_kwargs.update(kwargs)

        # Create cache key and get managed path
        cache_key = create_cache_key(prefix, **cache_kwargs)
        managed_path = get_tokenized_dataset_path(stage, cache_key)
        return str(managed_path)
    except ImportError:
        # Fallback to old method
        parts = [prefix]

        if tokenizer and hasattr(tokenizer, '__len__'):
            parts.append(f"vocab{len(tokenizer)}")
        elif vocab_size:
            parts.append(f"vocab{vocab_size}")

        if max_length:
            parts.append(f"len{max_length}")

        for key, value in kwargs.items():
            if value is not None:
                parts.append(f"{key}{value}")

        return f"./tokenized_cache_{'_'.join(parts)}"


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


def run_training_loop(trainer: Any, resume_checkpoint: Optional[str] = None) -> None:
    """
    Run training loop with optional checkpoint resumption.

    Args:
        trainer: HuggingFace Trainer instance
        resume_checkpoint: Optional path to checkpoint to resume from
    """
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()


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