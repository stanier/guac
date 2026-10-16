#!/usr/bin/env python3
"""
Guac Post-training Script - Instruction Tuning & Supervised Fine-Tuning

Fine-tunes a pre- or mid-trained model on instruction or conversation data.
Loss covers only the response tokens (assistant turns): prompts are masked.

Datasets (--datasets, any mix):
    alpaca, dolly         instruction / input / output records (Hugging Face)
    ultrachat             multi-turn conversations ("messages"; Hugging Face)
    path.json, .jsonl,    either format from disk: records with instruction
    or a saved dataset    (+ optional input/context) and output/response, or
                          with messages = [{role, content}, ...]
    synthetic             a small generated arithmetic set, for offline smoke
                          tests only

Usage:
    python posttrain.py --base-model ./scratch/models/midtraining/guac-3b_arithmetic --mode instruction
    python posttrain.py --base-model ./scratch/models/pretraining/guac-3b --mode sft --datasets ultrachat
    python posttrain.py --base-model ./model --mode instruction --datasets /path/to/instructions.json
"""

import os, sys, argparse, json, random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
from transformers import TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset

from guac.core.arithmetic import CATEGORIES, generate_problems
from guac.core.callbacks import CustomProgressCallback, ProfilingCallback
from guac.core.pipeline_utils import (
    load_base_model, get_output_directory, run_training_loop, apply_debug_mode, save_trained_model,
    GuacTrainer, MetricsLoggerCallback, make_training_arguments, write_run_info,
)

INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
INSTRUCTION_WITH_INPUT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
ROLE_PREFIXES = {'system': "System: ", 'user': "User: ", 'assistant': "Assistant: "}

# Default datasets when --datasets isn't given
DEFAULT_DATASETS = {'instruction': ['alpaca'], 'sft': ['ultrachat']}


def synthetic_instruction_records(num_samples: int = 2000, seed: int = 0) -> List[Dict[str, str]]:
    """Arithmetic instructions from the training pool (see core/arithmetic.py). Smoke tests only."""
    rng = random.Random(seed)
    records = []
    for category in CATEGORIES:
        for expression, _, result in generate_problems(category, num_samples // len(CATEGORIES), rng, held_out=False):
            records.append({'instruction': f"What is {expression}?", 'output': f"{expression} = {result}"})
    rng.shuffle(records)
    return records


def load_records(name: str) -> Dataset:
    """Load one dataset by name or path, unmodified."""
    if name == 'alpaca':
        return load_dataset("tatsu-lab/alpaca", split='train')
    if name == 'dolly':
        return load_dataset("databricks/databricks-dolly-15k", split='train')
    if name == 'ultrachat':
        return load_dataset("HuggingFaceH4/ultrachat_200k", split='train_sft')
    if name == 'synthetic':
        print("⚠️  Using the synthetic arithmetic instruction set: for smoke tests, not real post-training")
        return Dataset.from_list(synthetic_instruction_records())
    if name.endswith('.jsonl') and os.path.isfile(name):
        with open(name) as f:
            return Dataset.from_list([json.loads(line) for line in f if line.strip()])
    if name.endswith('.json') and os.path.isfile(name):
        with open(name) as f:
            return Dataset.from_list(json.load(f))
    if os.path.isdir(name):
        dataset = load_from_disk(name)
        return dataset['train'] if hasattr(dataset, 'keys') else dataset
    raise ValueError(f"Unknown post-training dataset {name!r}: use alpaca, dolly, ultrachat, synthetic, "
                     f"a .json/.jsonl file or a saved dataset directory")


def to_segments(record: Dict[str, Any]) -> List[Tuple[str, bool]]:
    """
    A record as (text, trained) segments: prompts and user turns are not
    trained on, responses and assistant turns are.

    Accepts instruction records (instruction, optional input or context, and
    output or response; Alpaca and Dolly) and conversations (messages).
    """
    if record.get('messages'):
        segments = []
        for message in record['messages']:
            role = message['role']
            if role not in ROLE_PREFIXES:
                raise ValueError(f"Unknown message role {role!r}")
            if role == 'assistant':
                segments += [(ROLE_PREFIXES[role], False), (message['content'] + "\n", True)]
            else:
                segments.append((ROLE_PREFIXES[role] + message['content'] + "\n", False))
        return segments

    instruction = record.get('instruction')
    # Not get('output', default): mixed datasets have every column, with None where a record lacks one
    response = record.get('output')
    if response is None:
        response = record.get('response')
    if instruction is None or response is None:
        raise ValueError(f"Record has neither messages nor instruction plus output/response: {sorted(record)}")
    context = record.get('input') or record.get('context')
    template = INSTRUCTION_WITH_INPUT_TEMPLATE if context else INSTRUCTION_TEMPLATE
    return [(template.format(instruction=instruction, input=context), False), (response, True)]


def tokenize_segments(segments: List[Tuple[str, bool]], tokenizer: Any, max_length: int) -> Dict[str, List[int]]:
    """
    Token ids of the segments joined, followed by EOS, truncated to
    max_length. Labels copy the trained segments (and EOS) and are -100 elsewhere.
    """
    input_ids, labels = [], []
    for text, trained in segments:
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        input_ids += ids
        labels += ids if trained else [-100] * len(ids)
    if tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    return {'input_ids': input_ids[:max_length], 'attention_mask': [1] * min(len(input_ids), max_length),
            'labels': labels[:max_length]}


def prepare_posttraining_dataset(dataset_names: List[str], tokenizer: Any, max_length: int = 512,
                                 train_on_prompts: bool = False) -> Dataset:
    """Load, format and tokenize the datasets, dropping examples with no response left after truncation."""
    datasets = [load_records(name) for name in dataset_names]
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    def tokenize(record):
        segments = to_segments(record)
        if train_on_prompts:
            segments = [(text, True) for text, _ in segments]
        return tokenize_segments(segments, tokenizer, max_length)

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names, desc="Tokenizing")
    return tokenized.filter(lambda example: any(label != -100 for label in example['labels']),
                            desc="Dropping examples with no response tokens")


def create_posttraining_arguments(output_dir: str, mode: str, overrides: Dict[str, Any]) -> TrainingArguments:
    """Training arguments for a mode, with command-line overrides applied (None leaves a default)."""
    defaults = {
        'instruction': dict(num_train_epochs=2, per_device_train_batch_size=4, gradient_accumulation_steps=8,
                            learning_rate=5e-6, weight_decay=0.0001, warmup_ratio=0.05, logging_steps=25,
                            save_steps=200, eval_steps=100),
        'sft': dict(num_train_epochs=3, per_device_train_batch_size=2, gradient_accumulation_steps=16,
                    learning_rate=1e-5, weight_decay=0.001, warmup_ratio=0.1, logging_steps=50,
                    save_steps=500, eval_steps=250),
    }[mode]
    defaults.update({key: value for key, value in overrides.items() if value is not None})
    return make_training_arguments(
        output_dir,
        per_device_eval_batch_size=defaults['per_device_train_batch_size'],
        lr_scheduler_type='cosine',
        save_total_limit=2,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        **defaults,
    )


def main():
    parser = argparse.ArgumentParser(description="Guac Post-training Script")

    parser.add_argument('--base-model', type=str, required=True,
                        help='Path to base model (pre-trained or mid-trained)')
    parser.add_argument('--mode', type=str, required=True, choices=['instruction', 'sft'],
                        help='Post-training mode: sets default datasets and hyperparameters')
    parser.add_argument('--datasets', type=str, nargs='+',
                        help='Datasets (default: alpaca for instruction, ultrachat for sft; see the module docstring)')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--train-on-prompts', action='store_true',
                        help='Also compute loss on prompts and user turns (default: responses only)')
    parser.add_argument('--output-dir', type=str, help='Training output directory')
    parser.add_argument('--resume-from-checkpoint', type=str, help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller datasets')
    parser.add_argument('--enable-profiling', action='store_true', help='Enable profiling callbacks')
    parser.add_argument('--force-retokenize', action='store_true',
                        help='Accepted for pipeline compatibility; post-training data is not cached')
    parser.add_argument('--epochs', type=float, help='Training epochs (default depends on --mode)')
    parser.add_argument('--max-steps', type=int, help='Optimizer steps (overrides --epochs)')
    parser.add_argument('--batch-size', type=int, help='Per-device batch size')
    parser.add_argument('--grad-accum', type=int, help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, help='Peak learning rate')
    parser.add_argument('--eval-steps', type=int, help='Evaluate every N steps')
    parser.add_argument('--save-steps', type=int, help='Checkpoint every N steps (a multiple of --eval-steps)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the eval split, data order and dropout')
    parser.add_argument('--report-to', type=str, default='none',
                        help='Also log to this tracker, e.g. wandb (metrics.jsonl is always written)')

    args = parser.parse_args()

    model, tokenizer = load_base_model(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_names = args.datasets or DEFAULT_DATASETS[args.mode]
    print(f"Preparing {args.mode} data from {', '.join(dataset_names)}...")
    dataset = prepare_posttraining_dataset(dataset_names, tokenizer, args.max_length, args.train_on_prompts)
    if args.debug:
        dataset = apply_debug_mode(dataset, 500)
    split = dataset.train_test_split(test_size=min(0.1, 2000 / len(dataset)), seed=args.seed)

    base_model_name = Path(args.base_model).name
    output_dir = get_output_directory("posttraining", base_model_name, args.output_dir, mode=args.mode)

    training_args = create_posttraining_arguments(output_dir, args.mode, {
        'num_train_epochs': args.epochs, 'max_steps': args.max_steps,
        'per_device_train_batch_size': args.batch_size, 'gradient_accumulation_steps': args.grad_accum,
        'learning_rate': args.learning_rate, 'eval_steps': args.eval_steps, 'save_steps': args.save_steps,
        'seed': args.seed, 'report_to': args.report_to,
    })

    callbacks = [
        CustomProgressCallback(tokenizer, sample_every=200),
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001),
        MetricsLoggerCallback(os.path.join(output_dir, 'metrics.jsonl')),
    ]
    if args.enable_profiling:
        callbacks.append(ProfilingCallback(log_every_n_steps=25))

    # Pads labels with -100 so masked prompt tokens stay masked
    data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100,
                                           pad_to_multiple_of=8 if training_args.fp16 else None)

    trainer = GuacTrainer(
        model=model,
        args=training_args,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        data_collator=data_collator,
        callbacks=callbacks,
    )

    write_run_info(output_dir, 'posttraining', vars(args), model=model, seed=args.seed,
                   base_model=args.base_model, datasets=dataset_names)
    print(f"Starting {args.mode} post-training...")
    print(f"Base model: {args.base_model}")
    print(f"Training examples: {len(split['train']):,} | Eval examples: {len(split['test']):,}")
    print(f"Output: {output_dir}")

    run_training_loop(trainer, args.resume_from_checkpoint)
    save_trained_model(trainer, tokenizer, output_dir, save_config=False)

    print(f"Post-training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
