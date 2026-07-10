#!/usr/bin/env python3
"""
GUAC Mid-training Script - Specialized Task Training

Handles mid-training on specific tasks like arithmetic, code, reasoning, etc.

Usage:
    python midtrain.py --base-model ./models/pretrain_nano --task arithmetic
    python midtrain.py --base-model ./models/pretrain_femto --task code
    python midtrain.py --base-model ./models/pretrain_atto --task custom --datasets /path/to/data
"""

import os, sys, yaml, argparse, torch, random
from typing import List, Dict, Any, Optional
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    AutoTokenizer, AutoModel, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from datasets import load_dataset, load_from_disk, concatenate_datasets

try:
    from scripts.training.core.callbacks import CustomProgressCallback, ProfilingCallback, LayerTimingCallback
    from scripts.training.core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM
    from scripts.training.core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size
    from scripts.training.core.pipeline_utils import (
        load_base_model, create_standard_data_collator, get_output_directory,
        run_training_loop, apply_debug_mode, save_trained_model
    )
except ImportError:
    try:
        from ..core.callbacks import CustomProgressCallback, ProfilingCallback, LayerTimingCallback
        from ..core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM
        from ..core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size
        from ..core.pipeline_utils import (
            load_base_model, create_standard_data_collator, get_output_directory,
            run_training_loop, apply_debug_mode, save_trained_model
        )
    except ImportError:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        from core.callbacks import CustomProgressCallback, ProfilingCallback, LayerTimingCallback
        from core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM
        from core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size
        from core.pipeline_utils import (
            load_base_model, create_standard_data_collator, get_output_directory,
            run_training_loop, apply_debug_mode, save_trained_model
        )


def create_arithmetic_dataset(tokenizer: Any, num_samples: int = 50000, force_retokenize: bool = False) -> Any:
    """Create arithmetic dataset for mid-training with caching support."""
    from datasets import Dataset

    vocab_size = get_vocab_size(tokenizer)
    cache_key = f"arithmetic_vocab{vocab_size}_samples{num_samples}"
    cache_dir = get_cache_directory("midtraining", cache_key)

    cached = load_cached_dataset(cache_dir, force_retokenize)
    if cached is not None:
        return cached

    def generate_arithmetic_problems():
        problems = []

        for _ in range(num_samples // 4):
            a, b = random.randint(1, 1000), random.randint(1, 1000)
            op = random.choice(['+', '-'])
            result = a + b if op == '+' else a - b
            problems.append(f"{a} {op} {b} = {result}")

        for _ in range(num_samples // 4):
            a, b = random.randint(1, 100), random.randint(1, 100)
            result = a * b
            problems.append(f"{a} * {b} = {result}")

        for _ in range(num_samples // 4):
            b = random.randint(1, 50)
            result = random.randint(1, 100)
            a = b * result
            problems.append(f"{a} / {b} = {result}")

        for _ in range(num_samples // 4):
            a, b, c = random.randint(1, 50), random.randint(1, 50), random.randint(1, 50)
            result = (a + b) * c
            problems.append(f"({a} + {b}) * {c} = {result}")

        return problems

    print(f"Generating {num_samples} arithmetic problems...")
    problems = generate_arithmetic_problems()

    tokenized = tokenizer(
        problems,
        truncation=True,
        padding=False,
        max_length=512,
        return_tensors=None
    )

    dataset_dict = {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'text': problems
    }
    dataset = Dataset.from_dict(dataset_dict)

    save_dataset_to_cache(dataset, cache_dir)
    return dataset


def create_code_dataset(tokenizer: Any) -> Any:
    """Create code dataset for mid-training."""
    print("Loading code datasets...")

    try:
        datasets = []

        try:
            python_data = load_dataset("codeparrot/github-code", languages=["Python"])
            datasets.append(python_data['train'].select(range(min(10000, len(python_data['train'])))))
        except:
            print("Could not load Python code dataset")

        if not datasets:
            code_examples = [
                "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "class Calculator:\n    def add(self, a, b):\n        return a + b\n    def multiply(self, a, b):\n        return a * b",
                "import math\n\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            return False\n    return True"
            ]

            expanded_examples = []
            for _ in range(1000):
                expanded_examples.append(random.choice(code_examples))

            tokenized = tokenizer(
                expanded_examples,
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors=None
            )

            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'text': expanded_examples
            }

        combined = concatenate_datasets(datasets)

        def tokenize_code(examples):
            texts = [example for example in examples['code'] if example and len(example.strip()) > 50]
            if not texts:
                return {'input_ids': [], 'attention_mask': []}

            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors=None
            )
            return tokenized

        return combined.map(tokenize_code, batched=True, remove_columns=combined.column_names)

    except Exception as e:
        print(f"Error loading code dataset: {e}")
        return None


def create_reasoning_dataset(tokenizer: Any, force_retokenize: bool = False) -> Any:
    """Create reasoning dataset for mid-training with caching support."""
    from datasets import Dataset

    vocab_size = get_vocab_size(tokenizer)
    cache_key = f"reasoning_vocab{vocab_size}_samples5000"
    cache_dir = get_cache_directory("midtraining", cache_key)

    cached = load_cached_dataset(cache_dir, force_retokenize)
    if cached is not None:
        return cached

    print("Creating reasoning dataset...")

    reasoning_examples = [
        "If all cats are animals and Fluffy is a cat, then Fluffy is an animal.",
        "The number of apples in the basket decreased by 3. If there were originally 10 apples, how many are left? Answer: 7",
        "If it takes 2 hours to walk 6 miles, how long does it take to walk 1 mile? Answer: 20 minutes",
        "A train leaves at 2 PM and arrives at 5 PM. How long was the journey? Answer: 3 hours",
        "If A > B and B > C, then A > C. This is an example of transitivity.",
    ]

    expanded = []
    for _ in range(5000):
        expanded.append(random.choice(reasoning_examples))

    tokenized = tokenizer(
        expanded,
        truncation=True,
        padding=False,
        max_length=512,
        return_tensors=None
    )

    dataset_dict = {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'text': expanded
    }
    dataset = Dataset.from_dict(dataset_dict)

    save_dataset_to_cache(dataset, cache_dir)
    return dataset


def prepare_task_dataset(task: str, tokenizer: Any, custom_datasets: Optional[List[str]] = None,
                        num_arithmetic_samples: int = 50000, force_retokenize: bool = False) -> Any:
    """Prepare dataset based on task type."""
    if task == 'arithmetic':
        return create_arithmetic_dataset(tokenizer, num_arithmetic_samples, force_retokenize)

    elif task == 'code':
        return create_code_dataset(tokenizer)

    elif task == 'reasoning':
        return create_reasoning_dataset(tokenizer, force_retokenize)

    elif task == 'custom' and custom_datasets:
        datasets = []
        for dataset_path in custom_datasets:
            try:
                if os.path.exists(dataset_path):
                    dataset = load_from_disk(dataset_path)
                    datasets.append(dataset)
                else:
                    dataset = load_dataset(dataset_path)
                    datasets.append(dataset['train'])
            except Exception as e:
                print(f"Could not load dataset {dataset_path}: {e}")

        if not datasets:
            raise ValueError("No valid custom datasets loaded!")

        combined = concatenate_datasets(datasets)

        def tokenize_function(examples):
            texts = [text for text in examples['text'] if text and text.strip()]
            if not texts:
                return {'input_ids': [], 'attention_mask': []}

            return tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors=None
            )

        return combined.map(tokenize_function, batched=True, remove_columns=combined.column_names)

    else:
        raise ValueError(f"Unknown task: {task}. Available: arithmetic, code, reasoning, custom")


def create_midtraining_arguments(output_dir: str, learning_rate: float = 1e-5) -> TrainingArguments:
    """Create training arguments optimized for mid-training."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=50,
        save_steps=500,
        eval_steps=250,
        save_total_limit=2,
        fp16=True,
        dataloader_drop_last=True,
        dataloader_num_workers=2,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=42,
    )


def main():
    parser = argparse.ArgumentParser(description="GUAC Mid-training Script")

    parser.add_argument('--base-model', type=str, required=True,
                       help='Path to pre-trained base model')
    parser.add_argument('--task', type=str, required=True,
                       choices=['arithmetic', 'code', 'reasoning', 'custom'],
                       help='Specialization task')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='Custom dataset paths (required for --task custom)')
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

    args = parser.parse_args()

    if args.task == 'custom' and not args.datasets:
        raise ValueError("--datasets required when --task is 'custom'")

    model, tokenizer = load_base_model(args.base_model)

    print(f"Preparing dataset for task: {args.task}")
    if args.task == 'arithmetic':
        dataset = create_arithmetic_dataset(tokenizer, args.num_arithmetic_samples, args.force_retokenize)
    else:
        dataset = prepare_task_dataset(args.task, tokenizer, args.datasets,
                                     args.num_arithmetic_samples, args.force_retokenize)

    if args.debug:
        dataset = apply_debug_mode(dataset, 1000)

    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    base_model_name = Path(args.base_model).name
    output_dir = get_output_directory("midtraining", base_model_name, args.output_dir, task=args.task)

    training_args = create_midtraining_arguments(output_dir, args.learning_rate)

    callbacks = [
        CustomProgressCallback(tokenizer, 50),
        EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)
    ]

    if args.enable_profiling:
        callbacks.append(ProfilingCallback(log_every_n_steps=25))

    data_collator = create_standard_data_collator(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print(f"Starting {args.task} specialization...")
    print(f"Base model: {args.base_model}")
    print(f"Training samples: {len(split_dataset['train']):,}")
    print(f"Output: {output_dir}")

    run_training_loop(trainer, args.resume_from_checkpoint)
    save_trained_model(trainer, tokenizer, output_dir, save_config=False)

    print(f"Mid-training complete! Specialized model saved to {output_dir}")


if __name__ == "__main__":
    main()
