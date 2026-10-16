#!/usr/bin/env python3
"""
Guac Mid-training Script - Specialized Task Training

Handles mid-training on specific tasks like arithmetic, code, reasoning, etc.

Usage:
    python midtrain.py --base-model ./models/pretrain_nano --task arithmetic
    python midtrain.py --base-model ./models/pretrain_femto --task code
    python midtrain.py --base-model ./models/pretrain_atto --task custom --datasets /path/to/data

    # Long arithmetic run: fresh data, reversed answers, best checkpoint by held-out accuracy
    python midtrain.py --base-model ./scratch/models/pretraining/guac-13m-arith --task arithmetic \\
        --num-arithmetic-samples 1000000 --epochs 1 --learning-rate 1e-3 --weight-decay 0.1 \\
        --warmup-ratio 0.02 --reverse-answers --eval-steps 2000 --early-stopping-patience 0
"""

import os, sys, yaml, argparse, torch, random
from typing import List, Dict, Any, Optional
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    AutoTokenizer, AutoModel, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, EarlyStoppingCallback, TrainerCallback
)
from datasets import load_dataset, load_from_disk, concatenate_datasets

from guac.core.callbacks import CustomProgressCallback, ProfilingCallback, LayerTimingCallback
from guac.core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM
from guac.core.arithmetic import (
    CATEGORIES, SPLIT_VERSION, generate_problems, format_answer, save_format, evaluate_accuracy
)
from guac.core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size, tokenizer_cache_tag
from guac.core.pipeline_utils import (
    load_base_model, create_standard_data_collator, get_output_directory,
    run_training_loop, apply_debug_mode, save_trained_model,
    GuacTrainer, MetricsLoggerCallback, make_training_arguments, write_run_info
)


def create_arithmetic_dataset(tokenizer: Any, num_samples: int = 50000, force_retokenize: bool = False,
                              reverse_answers: bool = False, answer_only_loss: bool = True,
                              held_out: bool = False, seed: int = 42) -> Any:
    """
    Create an arithmetic dataset with caching support.

    Problems come from the train pool (or the held-out pool when held_out=True;
    see core/arithmetic.py), split evenly across categories. With
    answer_only_loss, prompt tokens get label -100 so loss covers only the
    answer and EOS - the operands are random and cannot be predicted.
    """
    from datasets import Dataset

    cache_key = (f"arithmetic_split{SPLIT_VERSION}_{tokenizer_cache_tag(tokenizer)}_samples{num_samples}"
                 f"{'_heldout' if held_out else '_train'}"
                 f"{'_rev' if reverse_answers else ''}{'_ansloss' if answer_only_loss else ''}_seed{seed}")
    cache_dir = get_cache_directory("midtraining", cache_key)

    cached = load_cached_dataset(cache_dir, force_retokenize)
    if cached is not None:
        return cached

    rng = random.Random(seed)
    per_category = num_samples // len(CATEGORIES)
    prompts, texts = [], []
    print(f"Generating {per_category * len(CATEGORIES):,} {'held-out' if held_out else 'training'} "
          f"arithmetic problems{' (reversed answers)' if reverse_answers else ''}...")
    for category in CATEGORIES:
        for _, prompt, result in generate_problems(category, per_category, rng, held_out=held_out):
            prompts.append(prompt)
            texts.append(prompt + format_answer(result, reverse_answers))
    order = list(range(len(texts)))
    rng.shuffle(order)
    prompts, texts = [prompts[i] for i in order], [texts[i] for i in order]

    tokenized = tokenizer(texts, truncation=True, padding=False, max_length=512, return_tensors=None)
    dataset_dict = {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'text': texts,
    }
    if answer_only_loss:
        # Assumes the prompt tokenizes as a prefix of the full text (exact for
        # character-level tokenizers, close for subword ones)
        prompt_lengths = [len(ids) for ids in tokenizer(prompts, add_special_tokens=False)['input_ids']]
        dataset_dict['labels'] = [
            [-100] * n + ids[n:] for ids, n in zip(tokenized['input_ids'], prompt_lengths)
        ]
    dataset = Dataset.from_dict(dataset_dict)

    save_dataset_to_cache(dataset, cache_dir)
    return dataset


class ArithmeticAccuracyCallback(TrainerCallback):
    """
    Measure exact-match accuracy on held-out problems at each evaluation.

    Adds eval_heldout_accuracy (and per-category values) to the eval metrics,
    so it can drive metric_for_best_model and early stopping. Must be placed
    before EarlyStoppingCallback in the callback list.
    """

    def __init__(self, tokenizer, num_per_category: int, reverse_answers: bool):
        self.tokenizer = tokenizer
        self.num_per_category = num_per_category
        self.reverse_answers = reverse_answers

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        result = evaluate_accuracy(model, self.tokenizer, self.num_per_category,
                                   reverse=self.reverse_answers)
        entry = {'eval_heldout_accuracy': result['overall']}
        for category, accuracy in result['per_category'].items():
            entry[f"eval_heldout_accuracy_{category.replace('/', '_')}"] = accuracy
        if metrics is not None:
            metrics.update(entry)
        state.log_history.append({**entry, 'step': state.global_step})
        per_cat = '  '.join(f"{c} {a:.1%}" for c, a in result['per_category'].items())
        print(f"\n[step {state.global_step}] held-out accuracy {result['overall']:.1%}  ({per_cat})")


# Default data for the text tasks: Hugging Face (name, config, split) and how a record becomes training text
TEXT_TASKS = {
    'code': (('codeparrot/codeparrot-clean-valid', None, 'train'), lambda record: record['content']),
    'reasoning': (('openai/gsm8k', 'main', 'train'),
                  lambda record: f"Question: {record['question']}\nAnswer: {record['answer']}"),
}


def load_text_dataset(name: str) -> Any:
    """A dataset saved on disk, or a Hugging Face dataset's train split."""
    if os.path.exists(name):
        dataset = load_from_disk(name)
        return dataset['train'] if hasattr(dataset, 'keys') else dataset
    return load_dataset(name, split='train')


def prepare_task_dataset(task: str, tokenizer: Any, custom_datasets: Optional[List[str]] = None,
                         max_length: int = 512) -> Any:
    """
    Tokenized training text for the code, reasoning and custom tasks.

    code and reasoning load their default dataset (TEXT_TASKS) unless
    --datasets names others; custom needs --datasets. Datasets given by name
    or path must have a 'text' (or, for code, 'content') column.
    """
    if task == 'custom' and not custom_datasets:
        raise ValueError("--task custom needs --datasets")
    if custom_datasets:
        datasets = [load_text_dataset(name) for name in custom_datasets]
        to_text = lambda record: record.get('text', record.get('content'))
    else:
        (name, config_name, split), to_text = TEXT_TASKS[task]
        datasets = [load_dataset(name, config_name, split=split)]
    combined = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    def tokenize(examples):
        records = [dict(zip(examples, values)) for values in zip(*examples.values())]
        texts = [text for text in map(to_text, records) if text and text.strip()]
        return tokenizer(texts, truncation=True, padding=False, max_length=max_length)

    return combined.map(tokenize, batched=True, remove_columns=combined.column_names, desc="Tokenizing")


def create_midtraining_arguments(output_dir: str, learning_rate: float = 1e-5, num_train_epochs: float = 3,
                                 max_steps: int = -1, batch_size: int = 8, grad_accum: int = 4,
                                 weight_decay: float = 0.001, warmup_ratio: float = 0.1,
                                 eval_steps: int = 250, save_steps: Optional[int] = None,
                                 best_metric: str = "eval_loss", seed: int = 42,
                                 report_to: str = "none") -> TrainingArguments:
    """Create training arguments optimized for mid-training."""
    save_steps = save_steps or eval_steps * 2
    if save_steps % eval_steps:
        raise ValueError(f"--save-steps ({save_steps}) must be a multiple of --eval-steps ({eval_steps})")

    return make_training_arguments(
        output_dir,
        report_to=report_to,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type='cosine',
        logging_steps=50,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=2,
        dataloader_num_workers=2,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=best_metric != "eval_loss",
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(description="Guac Mid-training Script")

    parser.add_argument('--base-model', type=str, required=True,
                       help='Path to pre-trained base model')
    parser.add_argument('--task', type=str, required=True,
                       choices=['arithmetic', 'code', 'reasoning', 'custom'],
                       help='Specialization task')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='Datasets (paths or Hugging Face names) with a text column: required for '
                            '--task custom; replaces the default data for code and reasoning')
    parser.add_argument('--num-arithmetic-samples', type=int, default=50000,
                       help='Number of arithmetic samples to generate')
    parser.add_argument('--output-dir', type=str,
                       help='Training output directory')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                       help='Learning rate for mid-training')
    parser.add_argument('--resume-from-checkpoint', type=str,
                       help='Resume from checkpoint')
    parser.add_argument('--enable-profiling', action='store_true',
                       help='Enable profiling callbacks')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with smaller datasets')
    parser.add_argument('--force-retokenize', action='store_true',
                       help='Force re-tokenize dataset even if cached')

    # Training schedule
    parser.add_argument('--epochs', type=float, default=3,
                       help='Number of training epochs (ignored if --max-steps is set)')
    parser.add_argument('--max-steps', type=int, default=-1,
                       help='Total optimizer steps (overrides --epochs)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Per-device batch size')
    parser.add_argument('--grad-accum', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                       help='AdamW weight decay')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                       help='Fraction of steps used for LR warmup')
    parser.add_argument('--eval-steps', type=int, default=250,
                       help='Evaluate every N steps')
    parser.add_argument('--save-steps', type=int,
                       help='Checkpoint every N steps (default: 2x --eval-steps; must be a multiple of it)')
    parser.add_argument('--early-stopping-patience', type=int, default=2,
                       help='Evaluations without improvement before stopping (0 disables)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed for data order and dropout')
    parser.add_argument('--report-to', type=str, default='none',
                       help='Also log to this tracker, e.g. wandb (metrics.jsonl is always written)')

    # Arithmetic task options
    parser.add_argument('--reverse-answers', action='store_true',
                       help='Arithmetic: write answers least-significant digit first')
    parser.add_argument('--answer-only-loss', action=argparse.BooleanOptionalAction, default=True,
                       help='Arithmetic: compute loss on answer tokens only (default: on)')
    parser.add_argument('--num-eval-samples', type=int, default=2000,
                       help='Arithmetic: held-out problems for eval loss')
    parser.add_argument('--accuracy-eval-per-category', type=int, default=50,
                       help='Arithmetic: held-out problems per category for accuracy at each eval (0 disables)')

    args = parser.parse_args()


    model, tokenizer = load_base_model(args.base_model)

    print(f"Preparing dataset for task: {args.task}")
    is_arithmetic = args.task == 'arithmetic'
    if is_arithmetic:
        # Train and eval come from disjoint problem pools (see core/arithmetic.py)
        arithmetic_kwargs = dict(reverse_answers=args.reverse_answers, answer_only_loss=args.answer_only_loss,
                                 force_retokenize=args.force_retokenize)
        train_dataset = create_arithmetic_dataset(tokenizer, args.num_arithmetic_samples, **arithmetic_kwargs)
        eval_dataset = create_arithmetic_dataset(tokenizer, args.num_eval_samples, held_out=True, seed=43,
                                                 **arithmetic_kwargs)
    else:
        dataset = prepare_task_dataset(args.task, tokenizer, args.datasets)
        split_dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset, eval_dataset = split_dataset['train'], split_dataset['test']

    if args.debug:
        train_dataset = apply_debug_mode(train_dataset, 1000)
        eval_dataset = apply_debug_mode(eval_dataset, 200)

    base_model_name = Path(args.base_model).name
    output_dir = get_output_directory("midtraining", base_model_name, args.output_dir, task=args.task)

    track_accuracy = is_arithmetic and args.accuracy_eval_per_category > 0
    training_args = create_midtraining_arguments(
        output_dir, args.learning_rate, num_train_epochs=args.epochs, max_steps=args.max_steps,
        batch_size=args.batch_size, grad_accum=args.grad_accum, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio, eval_steps=args.eval_steps, save_steps=args.save_steps,
        best_metric="eval_heldout_accuracy" if track_accuracy else "eval_loss",
        seed=args.seed, report_to=args.report_to,
    )

    callbacks = [CustomProgressCallback(tokenizer, sample_every=200)]
    if track_accuracy:
        # Before EarlyStoppingCallback, which reads the metric this adds
        callbacks.append(ArithmeticAccuracyCallback(tokenizer, args.accuracy_eval_per_category,
                                                    args.reverse_answers))
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience,
                                               early_stopping_threshold=0.01))

    if args.enable_profiling:
        callbacks.append(ProfilingCallback(log_every_n_steps=25))
    callbacks.append(MetricsLoggerCallback(os.path.join(output_dir, 'metrics.jsonl')))

    if 'labels' in train_dataset.column_names:
        # Pads labels with -100 so masked prompt tokens stay masked
        data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100,
                                               pad_to_multiple_of=8 if training_args.fp16 else None)
    else:
        data_collator = create_standard_data_collator(training_args, tokenizer)

    trainer = GuacTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    write_run_info(output_dir, 'midtraining', vars(args), model=model, seed=args.seed, base_model=args.base_model)

    print(f"Starting {args.task} specialization...")
    print(f"Base model: {args.base_model}")
    print(f"Training samples: {len(train_dataset):,} | Eval samples: {len(eval_dataset):,}")
    print(f"Output: {output_dir}")

    run_training_loop(trainer, args.resume_from_checkpoint)
    save_trained_model(trainer, tokenizer, output_dir, save_config=False)
    if is_arithmetic:
        save_format(output_dir, args.reverse_answers)

    print(f"Mid-training complete! Specialized model saved to {output_dir}")


if __name__ == "__main__":
    main()
