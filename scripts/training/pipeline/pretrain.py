#!/usr/bin/env python3
"""
Guac Pre-training Script

Handles pre-training on large-scale language data.

Usage:
    python pretrain.py --model-config guac-3b --datasets wikitext-103
    python pretrain.py --model-config guac-910m --datasets wikitext-103,openwebtext
    python pretrain.py --model-config guac-430m --resume-from-checkpoint /path/to/checkpoint
"""

import os, sys, json, yaml, argparse, torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, EarlyStoppingCallback,
    TrainerCallback, set_seed
)
from datasets import load_dataset, load_from_disk, concatenate_datasets, IterableDataset

try:
    from scripts.training.core.callbacks import CustomProgressCallback, ProfilingCallback, LayerTimingCallback
    from scripts.training.core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM, param_matched_ffn_hidden_sizes, flop_matched_ffn_hidden_sizes
    from scripts.training.core.pipeline_utils import load_model_config, get_output_directory, run_training_loop, save_trained_model, GuacTrainer, MetricsLoggerCallback, write_run_info
    from scripts.training.core.dataset_utils import load_pretraining_splits, tokenize_dataset_with_cache, get_pretraining_dataset_settings, PackedSequenceCollator
    from scripts.training.core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size, content_hash, tokenizer_fingerprint
except ImportError:
    try:
        from ..core.callbacks import CustomProgressCallback, ProfilingCallback, LayerTimingCallback
        from ..core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM, param_matched_ffn_hidden_sizes, flop_matched_ffn_hidden_sizes
        from ..core.pipeline_utils import load_model_config, get_output_directory, run_training_loop, save_trained_model, GuacTrainer, MetricsLoggerCallback, write_run_info
        from ..core.dataset_utils import load_pretraining_splits, tokenize_dataset_with_cache, get_pretraining_dataset_settings, PackedSequenceCollator
        from ..core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size, content_hash, tokenizer_fingerprint
    except ImportError:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        from core.callbacks import CustomProgressCallback, ProfilingCallback, LayerTimingCallback
        from core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM, param_matched_ffn_hidden_sizes, flop_matched_ffn_hidden_sizes
        from core.pipeline_utils import load_model_config, get_output_directory, run_training_loop, save_trained_model, GuacTrainer, MetricsLoggerCallback, write_run_info
        from core.dataset_utils import load_pretraining_splits, tokenize_dataset_with_cache, get_pretraining_dataset_settings, PackedSequenceCollator
        from core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size, content_hash, tokenizer_fingerprint


def load_tokenizer(config: Dict[str, Any]) -> Any:
    """Load pre-trained tokenizer for pre-training."""
    default_tokenizer = 'scratch/tokenizers/sentencepiece/shared_60k'
    try:
        from scripts.training.core.path_manager import get_tokenizer_output_path
        default_tokenizer = str(get_tokenizer_output_path("sentencepiece", "shared_60k"))
    except ImportError:
        pass

    tokenizer_path = config.get('tokenizer_model_path', default_tokenizer)

    # Take the containing directory if a .model file path was given
    if tokenizer_path.endswith('.model'):
        tokenizer_path = os.path.dirname(tokenizer_path)

    # The path manager creates tokenizer directories up front, so check for a
    # saved tokenizer rather than for the directory
    if not os.path.isfile(os.path.join(tokenizer_path, 'tokenizer_config.json')):
        vocab_size = config.get('vocab_size', 60000)
        raise ValueError(
            f"Tokenizer not found at {tokenizer_path}\n"
            f"Please train a tokenizer first using:\n"
            f"  python scripts/training/pipeline/train_tokenizer.py --vocab-size {vocab_size}\n"
            f"Or use the Makefile:\n"
            f"  make train-tokenizer VOCAB_SIZE={vocab_size}"
        )

    print(f"Loading tokenizer from {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Loaded tokenizer with vocab size: {len(tokenizer):,}")
        return tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer from {tokenizer_path}: {e}")


def print_parameter_breakdown(model: Any, config: Dict[str, Any]) -> None:
    """Print detailed parameter breakdown by component and layer type."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter Breakdown:")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*60}")

    component_params = {}

    if hasattr(model, 'token_embeddings'):
        embedding_params = sum(p.numel() for p in model.token_embeddings.parameters())
        if hasattr(model, 'position_embeddings'):
            embedding_params += sum(p.numel() for p in model.position_embeddings.parameters())
        component_params['Embeddings'] = embedding_params

    if hasattr(model, 'lm_head'):
        component_params['LM Head'] = sum(p.numel() for p in model.lm_head.parameters())

    if hasattr(model, 'final_layer_norm'):
        component_params['Final Layer Norm'] = sum(p.numel() for p in model.final_layer_norm.parameters())

    layer_type_params = {}
    layers_total_params = 0

    if hasattr(model, 'layers'):
        layer_types = None
        if hasattr(model, 'config') and hasattr(model.config, 'layer_types'):
            layer_types = model.config.layer_types
        elif isinstance(config, dict) and 'layer_types' in config:
            layer_types = config['layer_types']
        elif hasattr(config, 'layer_types'):
            layer_types = config.layer_types

        for i, layer in enumerate(model.layers):
            if layer_types and i < len(layer_types):
                layer_type = layer_types[i]
            else:
                layer_class = layer.__class__.__name__.lower()
                if 'attention' in layer_class or 'moh' in layer_class:
                    layer_type = 'attention'
                elif 'conv' in layer_class:
                    layer_type = 'conv'
                elif 'mamba' in layer_class:
                    layer_type = 'mamba'
                elif 'based' in layer_class:
                    layer_type = 'based'
                elif 'moe' in layer_class:
                    layer_type = 'moe'
                else:
                    layer_type = f'layer_{i}'

            layer_params = sum(p.numel() for p in layer.parameters())
            layers_total_params += layer_params

            if layer_type not in layer_type_params:
                layer_type_params[layer_type] = {'count': 0, 'params': 0}
            layer_type_params[layer_type]['count'] += 1
            layer_type_params[layer_type]['params'] += layer_params

        component_params['All Layers'] = layers_total_params

    print(f"\nComponent Breakdown:")
    for component, params in component_params.items():
        percentage = (params / total_params) * 100
        print(f"  {component:<20}: {params:>12,} ({percentage:5.1f}%)")

    if layer_type_params:
        print(f"\nLayer Type Breakdown:")
        for layer_type, info in sorted(layer_type_params.items()):
            count = info['count']
            params = info['params']
            avg_params = params // count if count > 0 else 0
            percentage = (params / total_params) * 100
            print(f"  {layer_type:<15} (x{count:2d}): {params:>12,} ({percentage:5.1f}%) - avg {avg_params:,}/layer")

    accounted_params = sum(component_params.values())
    remaining = total_params - accounted_params
    if remaining > 0:
        percentage = (remaining / total_params) * 100
        print(f"  {'Other':<20}: {remaining:>12,} ({percentage:5.1f}%)")

    print(f"{'='*60}\n")


class MemoryUsageCallback(TrainerCallback):
    """Callback to print memory usage breakdown after first training step."""

    def __init__(self, config=None, tokenized_dataset=None):
        self.step_completed = False
        self.config = config
        self.tokenized_dataset = tokenized_dataset

    def on_step_end(self, args, state, control, **kwargs):
        if not self.step_completed and state.global_step == 1:
            self.step_completed = True
            model = kwargs.get('model')
            if model is not None:
                self._print_memory_breakdown(model, args)

    def _print_memory_breakdown(self, model, args):
        if not torch.cuda.is_available():
            return

        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
        except RuntimeError:
            return

        print(f"\nMemory Usage Breakdown (after first training step):")
        print(f"{'='*70}")
        print(f"Current GPU memory: {initial_memory / 1024**3:.2f} GB")
        print(f"Peak GPU memory: {max_memory / 1024**3:.2f} GB")
        print(f"{'='*70}")

        component_memory = {}

        try:
            if hasattr(model, 'token_embeddings'):
                component_memory['Token Embeddings'] = self._measure_component_memory(model.token_embeddings)
            if hasattr(model, 'position_embeddings'):
                component_memory['Position Embeddings'] = self._measure_component_memory(model.position_embeddings)
            if hasattr(model, 'lm_head'):
                component_memory['LM Head'] = self._measure_component_memory(model.lm_head)
            if hasattr(model, 'final_layer_norm'):
                component_memory['Final Layer Norm'] = self._measure_component_memory(model.final_layer_norm)

            if hasattr(model, 'layers') and len(model.layers) > 0:
                layer_memories = []
                for i, layer in enumerate(model.layers[:min(3, len(model.layers))]):
                    layer_mem = self._measure_component_memory(layer)
                    layer_memories.append(layer_mem)
                    if i < 3:
                        component_memory[f'Layer {i}'] = layer_mem

                if layer_memories:
                    avg_layer_mem = sum(layer_memories) / len(layer_memories)
                    component_memory['All Layers (est)'] = avg_layer_mem * len(model.layers)

            print(f"\nComponent Memory Usage:")
            total_component_mem = 0
            for component, mem_bytes in component_memory.items():
                mem_gb = mem_bytes / 1024**3
                total_component_mem += mem_bytes
                if mem_gb >= 0.1:
                    print(f"  {component:<25}: {mem_gb:6.2f} GB")
                else:
                    print(f"  {component:<25}: {mem_bytes / 1024**2:6.1f} MB")

            print(f"\nMemory Summary:")
            print(f"  Measured components: {total_component_mem / 1024**3:.2f} GB")
            print(f"  Current allocation:  {initial_memory / 1024**3:.2f} GB")
            print(f"  Unaccounted:         {max(0, initial_memory - total_component_mem) / 1024**3:.2f} GB")

        except Exception as e:
            print(f"Error measuring memory: {e}")

        dataset_memory = self._print_dataset_memory(args)
        self._print_memory_summary(initial_memory, max_memory, total_component_mem, dataset_memory, args)
        print(f"{'='*70}\n")

    def _print_dataset_memory(self, args):
        try:
            print(f"\nDataset Memory Usage:")
            batch_size = getattr(args, 'per_device_train_batch_size', 8)
            grad_accum = getattr(args, 'gradient_accumulation_steps', 4)
            num_devices = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
            effective_batch = batch_size * grad_accum * num_devices

            max_length = 512
            if self.config:
                max_length = self.config.get('max_sequence_length', 512)
            elif hasattr(args, 'max_length'):
                max_length = args.max_length

            print(f"  Batch size per device:  {batch_size}")
            print(f"  Gradient accumulation:  {grad_accum}")
            print(f"  Effective batch size:   {effective_batch}")
            print(f"  Sequence length:        {max_length}")

            bytes_per_token = 4
            tokens_per_batch = effective_batch * max_length
            batch_memory = tokens_per_batch * bytes_per_token * 3

            workers = getattr(args, 'dataloader_num_workers', 2)
            if workers > 0:
                dataloader_memory = batch_memory * workers * 2
            else:
                dataloader_memory = batch_memory

            total_data_memory = dataloader_memory + batch_memory

            if self.tokenized_dataset:
                dataset_size = len(self.tokenized_dataset)
                print(f"  Total samples:          {dataset_size:,}")
                print(f"  Batches per epoch:      {dataset_size // effective_batch:,}")

            if total_data_memory > 1024**3:
                print(f"  Total data memory:      {total_data_memory / 1024**3:.2f} GB")
            else:
                print(f"  Total data memory:      {total_data_memory / 1024**2:.1f} MB")

            return {
                'batch_memory': batch_memory,
                'dataloader_memory': dataloader_memory,
                'total_data_memory': total_data_memory,
                'effective_batch_size': effective_batch,
                'sequence_length': max_length
            }

        except Exception:
            return {
                'batch_memory': 0, 'dataloader_memory': 0,
                'total_data_memory': 0, 'effective_batch_size': 0,
                'sequence_length': 0
            }

    def _print_memory_summary(self, current_gpu, peak_gpu, model_memory, dataset_info, args):
        try:
            print(f"\nComprehensive Memory Summary:")
            print(f"{'-'*50}")

            current_gb = current_gpu / 1024**3
            peak_gb = peak_gpu / 1024**3
            model_gb = model_memory / 1024**3
            dataset_gb = dataset_info.get('total_data_memory', 0) / 1024**3

            total_est = model_gb * 4 + dataset_gb

            if total_est > current_gb * 1.2:
                scale = (current_gb * 0.8) / total_est
                model_gb *= scale
                optimizer_gb = model_gb * 1.5
                gradient_gb = model_gb * 0.8
            else:
                optimizer_gb = model_gb * 1.5
                gradient_gb = model_gb * 0.8

            overhead_gb = min(model_gb * 0.3, current_gb * 0.15)
            accounted = model_gb + optimizer_gb + gradient_gb + dataset_gb + overhead_gb
            framework_gb = max(0, current_gb - accounted)

            print(f"Memory Breakdown:")
            print(f"  Model Parameters:    {model_gb:6.2f} GB ({model_gb/current_gb*100:5.1f}%)")
            print(f"  Optimizer States:    {optimizer_gb:6.2f} GB ({optimizer_gb/current_gb*100:5.1f}%)")
            print(f"  Gradients:           {gradient_gb:6.2f} GB ({gradient_gb/current_gb*100:5.1f}%)")
            print(f"  Dataset/Batches:     {dataset_gb:6.2f} GB ({dataset_gb/current_gb*100:5.1f}%)")
            print(f"  Training Overhead:   {overhead_gb:6.2f} GB ({overhead_gb/current_gb*100:5.1f}%)")
            print(f"  Framework/Other:     {framework_gb:6.2f} GB ({framework_gb/current_gb*100:5.1f}%)")
            print(f"  {'-'*30}")
            print(f"  Total GPU Memory:    {current_gb:6.2f} GB")
            print(f"  Peak GPU Memory:     {peak_gb:6.2f} GB")

        except Exception:
            pass

    def _measure_component_memory(self, component):
        try:
            param_count = sum(p.numel() for p in component.parameters())
            return param_count * 12
        except Exception:
            return 0


def pretraining_cache_key(dataset_names: List[str], tokenizer: Any, max_length: int,
                          debug_samples: Optional[int] = None, split: str = 'train', pack: bool = True,
                          validation_docs: int = 1000, seed: int = 0) -> str:
    """
    Cache key for one split of a tokenized pre-training dataset.

    The readable part names the datasets (by basename, for on-disk paths),
    vocab size, sequence length and split. The hash covers everything else that
    changes the tokenized content: the tokenizer itself, each dataset's full
    name and site_config.yml settings, packing, the validation hold-out (which
    changes the training split too) and the debug subset, so a --debug run
    never fills the cache used by full runs.
    """
    names = '_'.join(sorted(Path(name).name for name in dataset_names))
    readable = f"{names}_vocab{get_vocab_size(tokenizer)}_len{max_length}_{split}"
    if debug_samples is not None:
        readable += f"_debug{debug_samples}"
    settings = {
        'tokenizer': tokenizer_fingerprint(tokenizer),
        'datasets': get_pretraining_dataset_settings(dataset_names),
        'max_length': max_length,
        'debug_samples': debug_samples,
        'pack': pack,
        'validation_docs': validation_docs,
        'seed': seed,
    }
    return f"{readable}_{content_hash(settings)}"


def prepare_pretraining_data(dataset_names: List[str], tokenizer: Any, max_length: int, pack: bool = True,
                             validation_docs: int = 1000, debug: bool = False,
                             force_retokenize: bool = False) -> Tuple[Any, Optional[Any]]:
    """
    Load, split and tokenize the pre-training datasets as (train, validation).

    validation is None when validation_docs is 0. With debug, training uses
    its first 1000 documents and validation its first 200. Each split is cached
    under pretraining_cache_key, which tokenize_datasets.py shares.
    """
    train, validation = load_pretraining_splits(dataset_names, validation_docs)

    debug_samples = None
    if debug:
        print("Debug mode: using smaller dataset")
        if hasattr(train, 'select'):
            debug_samples = min(1000, len(train))
            train = train.select(range(debug_samples))
        if validation is not None:
            validation = validation.select(range(min(200, len(validation))))

    key_args = dict(debug_samples=debug_samples, pack=pack, validation_docs=validation_docs)
    splits = {'train': train, 'validation': validation}
    tokenized = {}
    for split, dataset in splits.items():
        if dataset is None:
            tokenized[split] = None
            continue
        cache_key = pretraining_cache_key(dataset_names, tokenizer, max_length, split=split, **key_args)
        tokenized[split] = tokenize_dataset_with_cache(dataset, tokenizer, get_cache_directory("pretraining", cache_key),
                                                       max_length, force_retokenize, pack=pack)
    return tokenized['train'], tokenized['validation']


# Preset keys passed straight through to MixedLayerConfig (per-layer-type and position-encoding settings)
MODEL_SETTING_PREFIXES = (
    'num_experts', 'num_shared_experts', 'moe_', 'moh_', 'mamba_', 'based_', 'conv_', 'sliding_window_',
    'use_rope', 'rope_', 'use_absolute_position_embeddings',
)


def build_model_config(config: Dict[str, Any], vocab_size: int) -> MixedLayerConfig:
    """
    Build the MixedLayerConfig for a model_configs.yaml entry (validates layer types).

    With `param_match: <layer type>` (or `true` for full_attention), every SwiGLU
    layer's FFN width is set so its block has the parameters of a block of that type.
    `flop_match` does the same for forward FLOPs per token at max_sequence_length.
    """
    model_config = MixedLayerConfig(
        vocab_size=vocab_size,
        hidden_size=config['hidden_size'],
        num_hidden_layers=config['num_hidden_layers'],
        num_attention_heads=config['num_attention_heads'],
        num_kv_heads=config.get('num_kv_heads', config['num_attention_heads']),
        intermediate_size=config.get('intermediate_size', config['hidden_size'] * 4),
        max_position_embeddings=config.get('max_position_embeddings', 4096),
        layer_types=config.get('layer_types'),  # None -> all full_attention
        use_rms_norm=config.get('use_rms_norm', True),
        layer_norm_eps=config.get('layer_norm_eps', 1e-6),
        tie_word_embeddings=config.get('tie_word_embeddings', True),
        ffn_types=config.get('ffn_types'),
        ffn_hidden_sizes=config.get('ffn_hidden_sizes'),
        **{k: v for k, v in config.items() if k.startswith(MODEL_SETTING_PREFIXES)}
    )
    param_match, flop_match = config.get('param_match'), config.get('flop_match')
    if sum(bool(x) for x in (param_match, flop_match, config.get('ffn_hidden_sizes') is not None)) > 1:
        raise ValueError("Set at most one of param_match, flop_match and ffn_hidden_sizes")
    if param_match:
        reference = 'full_attention' if param_match is True else param_match
        model_config.ffn_hidden_sizes = param_matched_ffn_hidden_sizes(model_config, reference)
    if flop_match:
        reference = 'full_attention' if flop_match is True else flop_match
        model_config.ffn_hidden_sizes = flop_matched_ffn_hidden_sizes(
            model_config, reference, config.get('max_sequence_length', 1024))
    return model_config


def create_model(config: Dict[str, Any], tokenizer: Any, print_params: bool = False) -> Any:
    """Create model from configuration."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = MixedLayerForCausalLM(build_model_config(config, len(tokenizer)))
    print(f"Created model with {model.num_parameters():,} parameters")

    if print_params:
        print_parameter_breakdown(model, config)

    if torch.cuda.is_available():
        model = model.cuda()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

    return model


def create_training_arguments(config: Dict[str, Any], output_dir: str, max_steps: int = -1,
                              evaluate: bool = False) -> TrainingArguments:
    """
    Create training arguments from config, using its values as given.

    max_steps <= 0 trains for config['num_train_epochs'] instead. With evaluate,
    the validation set is scored every config['eval_steps'] steps.
    """
    report_to = config.get('report_to', 'none')
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get('num_train_epochs', 1),
        max_steps=max_steps,
        per_device_train_batch_size=config.get('pdbs_train', 16),
        per_device_eval_batch_size=config.get('pdbs_eval', 16),
        gradient_accumulation_steps=config.get('grad_accum_steps', 1),
        learning_rate=float(config.get('learning_rate', 8e-4)),
        weight_decay=float(config.get('weight_decay', 0.01)),
        warmup_ratio=float(config.get('warmup_ratio', 0.05)),
        lr_scheduler_type=config.get('lr_scheduler_type', 'constant_with_warmup'),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        logging_steps=config.get('step_val', 100),
        save_steps=config.get('save_steps', 1000),
        eval_steps=config.get('eval_steps', 500),
        save_total_limit=3,
        fp16=config.get('fp16', True),
        gradient_checkpointing=True,
        # Dropping the last partial batch would also drop a validation set smaller than one batch
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        eval_strategy="steps" if evaluate else "no",
        report_to=[] if report_to in (None, 'none') else report_to,
        seed=config.get('seed', 42),
    )


def print_training_settings(training_args: TrainingArguments, max_length: int, train_dataset: Any) -> None:
    """Print the settings training will actually use."""
    per_device = training_args.per_device_train_batch_size
    grad_accum = training_args.gradient_accumulation_steps
    effective_batch = per_device * grad_accum * training_args.world_size
    if training_args.max_steps > 0:
        duration = f"{training_args.max_steps:,} steps"
    else:
        duration = f"{training_args.num_train_epochs:g} epochs"
        if not isinstance(train_dataset, IterableDataset):
            steps_per_epoch = len(train_dataset) // (per_device * training_args.world_size) // grad_accum
            duration += f" (~{steps_per_epoch:,} steps each)"

    print("Training settings:")
    print(f"  Sequence length:      {max_length}")
    print(f"  Batch per device:     {per_device}")
    print(f"  Grad accumulation:    {grad_accum}")
    print(f"  Devices:              {training_args.world_size}")
    print(f"  Effective batch:      {effective_batch} sequences (up to {effective_batch * max_length:,} tokens/step)")
    print(f"  Learning rate:        {training_args.learning_rate:g} ({training_args.lr_scheduler_type.value}, "
          f"warmup ratio {training_args.warmup_ratio:g})")
    print(f"  Duration:             {duration}")


def create_callbacks(config: Dict[str, Any], tokenizer: Any, enable_profiling: bool = False,
                     enable_memory_profiling: bool = False, tokenized_dataset: Any = None) -> List[Any]:
    """Create training callbacks."""
    callbacks = [CustomProgressCallback(tokenizer, config.get('sample_every', 200))]

    if enable_memory_profiling:
        callbacks.append(MemoryUsageCallback(config=config, tokenized_dataset=tokenized_dataset))

    if enable_profiling:
        callbacks.append(ProfilingCallback(
            log_every_n_steps=config.get('profiling_steps', 50),
            detailed_profiling=True
        ))

    return callbacks


def main():
    parser = argparse.ArgumentParser(description="Guac Pre-training Script")

    parser.add_argument('--model-config', type=str, required=True,
                       help='Model configuration name from model_configs.yaml')
    parser.add_argument('--datasets', type=str, default='wikitext-103',
                       help='Comma-separated list of datasets')
    parser.add_argument('--force-retokenize', action='store_true',
                       help='Force re-tokenize dataset even if cached')
    parser.add_argument('--vocab-size', type=int,
                       help='Override vocab size from config')
    parser.add_argument('--output-dir', type=str,
                       help='Training output directory')
    parser.add_argument('--resume-from-checkpoint', type=str,
                       help='Resume training from checkpoint')
    parser.add_argument('--max-length', type=int, default=None,
                       help='Maximum sequence length')
    parser.add_argument('--enable-profiling', action='store_true',
                       help='Enable detailed profiling callbacks')
    parser.add_argument('--print-mem', action='store_true',
                       help='Print detailed memory usage after first step')
    parser.add_argument('--print-params', action='store_true',
                       help='Print detailed parameter breakdown')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with smaller dataset')
    parser.add_argument('--max-steps', type=int, default=None,
                       help="Optimizer steps to train for (default: the config's max_steps if set, "
                            "else num_train_epochs; 20 with --debug)")
    parser.add_argument('--batch-size', type=int, default=None,
                       help="Per-device batch size (default: the config's pdbs_train)")
    parser.add_argument('--grad-accum', type=int, default=None,
                       help="Gradient accumulation steps (default: the config's grad_accum_steps)")
    parser.add_argument('--validation-docs', type=int, default=None,
                       help="Documents held out per dataset for validation, where a dataset has no "
                            "validation split (default: the config's validation_docs; 0 disables)")
    parser.add_argument('--pack', action=argparse.BooleanOptionalAction, default=None,
                       help="Pack documents into full-length blocks instead of truncating each one "
                            "(default: the config's pack_sequences)")
    parser.add_argument('--eval-steps', type=int, default=None,
                       help="Evaluate on the validation set every N steps (default: the config's eval_steps)")
    parser.add_argument('--save-steps', type=int, default=None,
                       help="Checkpoint every N steps (default: the config's save_steps)")
    parser.add_argument('--seed', type=int, default=None,
                       help="Seed for model initialization and data order (default: the config's seed)")
    parser.add_argument('--report-to', type=str, default=None,
                       help="Also log to this tracker, e.g. wandb (default: the config's report_to; "
                            "metrics.jsonl in the output directory is always written)")

    args = parser.parse_args()

    print(f"Loading model config: {args.model_config}")
    config = load_model_config(args.model_config)

    overrides = {'vocab_size': args.vocab_size, 'pdbs_train': args.batch_size, 'grad_accum_steps': args.grad_accum,
                 'validation_docs': args.validation_docs, 'pack_sequences': args.pack,
                 'eval_steps': args.eval_steps, 'save_steps': args.save_steps, 'seed': args.seed,
                 'report_to': args.report_to}
    config.update({key: value for key, value in overrides.items() if value is not None})
    seed = config.get('seed', 42)
    validation_docs = config.get('validation_docs', 1000)
    pack = config.get('pack_sequences', True)

    dataset_names = args.datasets.split(',')

    tokenizer = load_tokenizer(config)

    max_length = args.max_length if args.max_length is not None else config.get('max_sequence_length', 1024)

    train_dataset, eval_dataset = prepare_pretraining_data(
        dataset_names, tokenizer, max_length, pack=pack, validation_docs=validation_docs,
        debug=args.debug, force_retokenize=args.force_retokenize,
    )

    # Seed before building the model, so its initialization is reproducible too
    set_seed(seed)
    model = create_model(config, tokenizer, args.print_params)

    output_dir = get_output_directory("pretraining", args.model_config, args.output_dir)

    if args.max_steps is not None:
        max_steps = args.max_steps
    elif args.debug:
        max_steps = 20
    else:
        max_steps = config.get('max_steps', -1)
    if max_steps <= 0 and isinstance(train_dataset, IterableDataset):
        raise ValueError("Streaming datasets have no length, so training can't be measured in epochs. "
                         "Pass --max-steps (MAX_STEPS= with make) or set max_steps in the model config.")
    training_args = create_training_arguments(config, output_dir, max_steps, evaluate=eval_dataset is not None)
    print_training_settings(training_args, max_length, train_dataset)
    print(f"  Packing:              {'on' if pack else 'off'}")
    print(f"  Validation:           {f'{len(eval_dataset):,} sequences, every {training_args.eval_steps} steps' if eval_dataset is not None else 'none'}")
    callbacks = create_callbacks(config, tokenizer, args.enable_profiling, args.print_mem, train_dataset)
    callbacks.append(MetricsLoggerCallback(os.path.join(output_dir, 'metrics.jsonl')))

    if pack:
        data_collator = PackedSequenceCollator()
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if training_args.fp16 else None
        )

    trainer = GuacTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    write_run_info(output_dir, 'pretraining', config, args, model, seed=seed,
                   model_config_name=args.model_config, datasets=dataset_names, max_length=max_length)

    print(f"Starting pre-training...")
    print(f"Model: {args.model_config} ({model.num_parameters():,} parameters)")

    try:
        size_info = f"{len(train_dataset):,} sequences"
    except TypeError:
        size_info = "streaming dataset"

    print(f"Dataset: {args.datasets} ({size_info})")
    print(f"Output: {output_dir}")

    train_output = run_training_loop(trainer, args.resume_from_checkpoint)
    save_trained_model(trainer, tokenizer, output_dir, save_config=True)

    results = {'train_loss': train_output.training_loss, 'steps': trainer.state.global_step,
               'runtime_seconds': train_output.metrics.get('train_runtime')}
    if eval_dataset is not None:
        final_step = trainer.state.global_step
        logged = [entry for entry in trainer.state.log_history if 'eval_loss' in entry and entry['step'] == final_step]
        results.update(logged[-1] if logged else trainer.evaluate())
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    if eval_dataset is not None:
        print(f"Validation loss {results['eval_loss']:.4f} (perplexity {results['eval_perplexity']:.2f})")

    print(f"Pre-training complete! Model saved to {output_dir}")


if __name__ == "__main__": main()
