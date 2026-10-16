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

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, EarlyStoppingCallback,
    TrainerCallback, set_seed
)
from datasets import load_dataset, load_from_disk, concatenate_datasets, IterableDataset

from guac.core.callbacks import CustomProgressCallback, ProfilingCallback, LayerTimingCallback
from guac.core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM, param_matched_ffn_hidden_sizes, flop_matched_ffn_hidden_sizes
from guac.core.pipeline_utils import load_model_config, get_output_directory, run_training_loop, save_trained_model, GuacTrainer, MetricsLoggerCallback, make_training_arguments, write_run_info
from guac.core.dataset_utils import load_pretraining_splits, tokenize_dataset_with_cache, get_pretraining_dataset_settings, PackedSequenceCollator
from guac.core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size, content_hash, tokenizer_fingerprint


def load_tokenizer(config: Dict[str, Any]) -> Any:
    """Load pre-trained tokenizer for pre-training."""
    from guac.core.path_manager import get_tokenizer_output_path
    default_tokenizer = str(get_tokenizer_output_path("sentencepiece", "shared_60k"))

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
            f"  python guac/pipeline/train_tokenizer.py --vocab-size {vocab_size}\n"
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


def print_parameter_breakdown(model: Any) -> None:
    """Print parameter counts by component and by layer type."""
    total = model.num_parameters()
    print(f"\nParameter Breakdown:\n{'=' * 60}\nTotal parameters: {total:,}\n{'=' * 60}")

    def count(module):
        return sum(p.numel() for p in module.parameters()) if module is not None else 0

    tied = model.lm_head.weight is model.token_embeddings.weight
    components = {
        'Token embeddings': count(model.token_embeddings),
        'Position embeddings': count(model.position_embeddings),
        'LM head' + (' (tied to embeddings)' if tied else ''): 0 if tied else count(model.lm_head),
        'Final norm': count(model.final_layer_norm),
        'Layers': sum(count(layer) for layer in model.layers),
    }
    print("\nComponent Breakdown:")
    for name, params in components.items():
        if params or name.startswith('LM head'):
            print(f"  {name:<30}: {params:>14,} ({params / total:6.1%})")

    by_type = {}
    for layer in model.layers:
        key = f"{layer.layer_type} + {layer.ffn_type}"
        layers, params = by_type.get(key, (0, 0))
        by_type[key] = (layers + 1, params + count(layer))
    print("\nLayer Type Breakdown (mixer + FFN):")
    for key, (layers, params) in sorted(by_type.items()):
        print(f"  {key:<30} (x{layers:2d}): {params:>14,} ({params / total:6.1%}) - {params // layers:,}/layer")
    print(f"{'=' * 60}\n")


class MemoryUsageCallback(TrainerCallback):
    """
    After the first optimizer step, print where GPU memory went: parameters
    and optimizer state measured from the tensors themselves, gradients (the
    size of the trainable parameters, freed after each step), and the rest of
    the peak (activations, temporary buffers and allocator overhead).
    """

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if state.global_step != 1 or model is None or not torch.cuda.is_available():
            return

        def gigabytes(tensors):
            return sum(t.numel() * t.element_size() for t in tensors) / 1024**3

        parameters = gigabytes(model.parameters())
        gradients = gigabytes(p for p in model.parameters() if p.requires_grad)
        optimizer_state = gigabytes(value for param_state in (optimizer.state.values() if optimizer else [])
                                    for value in param_state.values() if torch.is_tensor(value))
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nGPU memory after the first step (device {torch.cuda.current_device()}):")
        print(f"  Parameters:        {parameters:7.2f} GB")
        print(f"  Gradients:         {gradients:7.2f} GB (at peak; freed after each step)")
        print(f"  Optimizer state:   {optimizer_state:7.2f} GB")
        print(f"  Activations, etc.: {max(0.0, peak - parameters - gradients - optimizer_state):7.2f} GB")
        print(f"  Peak allocated:    {peak:7.2f} GB  (currently {torch.cuda.memory_allocated() / 1024**3:.2f} GB)")


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
        print_parameter_breakdown(model)

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
    return make_training_arguments(
        output_dir,
        fp16=config.get('fp16', True),
        report_to=config.get('report_to', 'none'),
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
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        eval_strategy="steps" if evaluate else "no",
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
        callbacks.append(MemoryUsageCallback())

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
