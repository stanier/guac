#!/usr/bin/env python3
"""
GUAC Post-training Script - Instruction Tuning & Alignment

Handles post-training phases like instruction tuning and supervised fine-tuning.

Usage:
    python posttrain.py --base-model ./models/midtrain_arithmetic --mode instruction
    python posttrain.py --base-model ./models/pretrain_nano --mode sft --dataset alpaca
    python posttrain.py --base-model ./models/midtrain_code --mode instruction --datasets /path/to/instructions.json
"""

import os, sys, yaml, argparse, torch, json, random
from typing import List, Dict, Any, Optional
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from datasets import load_dataset, load_from_disk, Dataset

try:
    from scripts.training.core.callbacks import CustomProgressCallback, ProfilingCallback
    from scripts.training.core.custom_architectures import MixedLayerForCausalLM
    from scripts.training.core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size
    from scripts.training.core.pipeline_utils import (
        load_base_model, create_standard_data_collator, get_output_directory,
        run_training_loop, apply_debug_mode, save_trained_model
    )
except ImportError:
    try:
        from ..core.callbacks import CustomProgressCallback, ProfilingCallback
        from ..core.custom_architectures import MixedLayerForCausalLM
        from ..core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size
        from ..core.pipeline_utils import (
            load_base_model, create_standard_data_collator, get_output_directory,
            run_training_loop, apply_debug_mode, save_trained_model
        )
    except ImportError:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        from core.callbacks import CustomProgressCallback, ProfilingCallback
        from core.custom_architectures import MixedLayerForCausalLM
        from core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size
        from core.pipeline_utils import (
            load_base_model, create_standard_data_collator, get_output_directory,
            run_training_loop, apply_debug_mode, save_trained_model
        )

try:
    from scripts.training.core.prompt_templates import apply_default_template
except ImportError:
    def apply_default_template(text):
        return text


def create_instruction_dataset(tokenizer: Any, num_samples: int = 10000, force_retokenize: bool = False) -> Dataset:
    """Create synthetic instruction dataset with caching support."""
    vocab_size = get_vocab_size(tokenizer)
    cache_key = f"instruction_vocab{vocab_size}_samples{num_samples}"
    cache_dir = get_cache_directory("posttraining", cache_key)

    cached = load_cached_dataset(cache_dir, force_retokenize)
    if cached is not None:
        return cached

    print(f"Generating {num_samples} instruction examples...")

    instructions = []

    math_templates = [
        ("What is {a} + {b}?", "The answer is {result}."),
        ("Calculate {a} * {b}", "{a} * {b} = {result}"),
        ("Solve: {a} - {b}", "{a} - {b} equals {result}"),
        ("What's {a} divided by {b}?", "{a} ÷ {b} = {result}"),
    ]

    for _ in range(num_samples // 4):
        template = random.choice(math_templates)
        a, b = random.randint(1, 100), random.randint(1, 100)

        if "divided by" in template[0]:
            result = random.randint(1, 20)
            a = b * result
        elif "*" in template[0]:
            result = a * b
        elif "+" in template[0]:
            result = a + b
        elif "-" in template[0]:
            result = a - b

        instruction = template[0].format(a=a, b=b)
        response = template[1].format(a=a, b=b, result=result)
        instructions.append({"instruction": instruction, "response": response})

    general_examples = [
        {
            "instruction": "Explain what artificial intelligence is.",
            "response": "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving."
        },
        {
            "instruction": "Write a Python function to check if a number is prime.",
            "response": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
        },
        {
            "instruction": "What are the benefits of exercise?",
            "response": "Exercise has numerous benefits including improved cardiovascular health, stronger muscles and bones, better mental health, increased energy levels, and reduced risk of chronic diseases."
        }
    ]

    for _ in range(num_samples // 4):
        instructions.append(random.choice(general_examples))

    qa_examples = [
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "instruction": "How do you make a sandwich?",
            "response": "To make a sandwich: 1) Take two slices of bread, 2) Add your desired fillings like meat, cheese, vegetables, 3) Add condiments if desired, 4) Place the second slice on top."
        },
        {
            "instruction": "What is photosynthesis?",
            "response": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. It's essential for life on Earth as it provides food for plants and oxygen for other organisms."
        }
    ]

    for _ in range(num_samples // 4):
        instructions.append(random.choice(qa_examples))

    creative_examples = [
        {
            "instruction": "Write a short poem about nature.",
            "response": "Trees sway gently in the breeze,\nLeaves dancing with such ease.\nBirds sing songs both sweet and clear,\nNature's music fills the air."
        },
        {
            "instruction": "Create a story about a robot learning to cook.",
            "response": "CHEF-3000 stared at the kitchen with confusion. Its sensors detected various ingredients, but cooking was not in its programming. Through trial and error, burning several meals, the robot slowly learned. By week's end, CHEF-3000 was creating delicious dishes, having discovered that cooking was both science and art."
        }
    ]

    for _ in range(num_samples // 4):
        instructions.append(random.choice(creative_examples))

    dataset = Dataset.from_list(instructions)
    save_dataset_to_cache(dataset, cache_dir)
    return dataset


def load_instruction_datasets(dataset_names: List[str]) -> Dataset:
    """Load instruction datasets."""
    datasets = []

    for dataset_name in dataset_names:
        print(f"Loading instruction dataset: {dataset_name}")

        try:
            if dataset_name == 'alpaca':
                dataset = load_dataset("tatsu-lab/alpaca")
                datasets.append(dataset['train'])
            elif dataset_name == 'dolly':
                dataset = load_dataset("databricks/databricks-dolly-15k")
                datasets.append(dataset['train'])
            elif os.path.exists(dataset_name):
                if dataset_name.endswith('.json'):
                    with open(dataset_name) as f:
                        data = json.load(f)
                    datasets.append(Dataset.from_list(data))
                else:
                    datasets.append(load_from_disk(dataset_name))
        except Exception as e:
            print(f"Could not load dataset {dataset_name}: {e}")

    if not datasets:
        print("No external datasets loaded, using synthetic instruction data")
        return create_instruction_dataset(None, 5000)

    if len(datasets) == 1:
        return datasets[0]
    else:
        from datasets import concatenate_datasets
        return concatenate_datasets(datasets)


def format_instruction_data(examples: Dict[str, List], tokenizer: Any, max_length: int = 512) -> Dict[str, List]:
    """Format instruction data with proper templates."""
    formatted_texts = []

    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        response = examples['response'][i]
        formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{tokenizer.eos_token}"
        formatted_texts.append(formatted_text)

    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None
    )

    return tokenized


def create_conversation_dataset(tokenizer: Any, force_retokenize: bool = False) -> Dataset:
    """Create conversation dataset for SFT with caching support."""
    vocab_size = get_vocab_size(tokenizer)
    cache_key = f"conversation_vocab{vocab_size}_samples1000"
    cache_dir = get_cache_directory("posttraining", cache_key)

    cached = load_cached_dataset(cache_dir, force_retokenize)
    if cached is not None:
        return cached

    print("Creating conversation dataset...")

    conversations = [
        {
            "messages": [
                {"role": "user", "content": "Hello! How are you today?"},
                {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. I'm here and ready to help with any questions or tasks you might have. How can I assist you today?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Can you help me understand machine learning?"},
                {"role": "assistant", "content": "Absolutely! Machine learning is a subset of AI where computers learn patterns from data without being explicitly programmed for each task. There are three main types: supervised learning (learning from labeled examples), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error). Would you like me to explain any of these in more detail?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "I don't have access to real-time weather data, so I can't tell you the current weather conditions. To get accurate weather information, I'd recommend checking a weather app, website like weather.com, or asking a voice assistant that has internet access. Is there anything else I can help you with?"}
            ]
        }
    ]

    expanded = []
    for _ in range(1000):
        expanded.append(random.choice(conversations))

    dataset = Dataset.from_list(expanded)
    save_dataset_to_cache(dataset, cache_dir)
    return dataset


def format_conversation_data(examples: Dict[str, List], tokenizer: Any, max_length: int = 512) -> Dict[str, List]:
    """Format conversation data for training."""
    formatted_texts = []

    for messages in examples['messages']:
        formatted_parts = []
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'user':
                formatted_parts.append(f"User: {content}")
            else:
                formatted_parts.append(f"Assistant: {content}")

        formatted_text = "\n".join(formatted_parts) + tokenizer.eos_token
        formatted_texts.append(formatted_text)

    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None
    )

    return tokenized


def prepare_posttraining_dataset(mode: str, tokenizer: Any, dataset_names: Optional[List[str]] = None,
                               force_retokenize: bool = False) -> Dataset:
    """Prepare dataset based on post-training mode."""
    if mode == 'instruction':
        if dataset_names:
            dataset = load_instruction_datasets(dataset_names)
        else:
            dataset = create_instruction_dataset(tokenizer, force_retokenize=force_retokenize)

        formatted_dataset = dataset.map(
            lambda examples: format_instruction_data(examples, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        return formatted_dataset

    elif mode == 'sft':
        if dataset_names:
            return load_instruction_datasets(dataset_names)
        else:
            dataset = create_conversation_dataset(tokenizer, force_retokenize)
            formatted_dataset = dataset.map(
                lambda examples: format_conversation_data(examples, tokenizer),
                batched=True,
                remove_columns=dataset.column_names
            )
            return formatted_dataset

    else:
        raise ValueError(f"Unknown mode: {mode}. Available: instruction, sft")


def create_posttraining_arguments(output_dir: str, mode: str) -> TrainingArguments:
    """Create training arguments for post-training."""
    if mode == 'instruction':
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            weight_decay=0.0001,
            warmup_ratio=0.05,
            lr_scheduler_type='cosine',
            logging_steps=25,
            save_steps=200,
            eval_steps=100,
            save_total_limit=2,
            fp16=True,
            dataloader_drop_last=True,
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to=[],
            seed=42,
        )
    else:
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=1e-5,
            weight_decay=0.001,
            warmup_ratio=0.1,
            lr_scheduler_type='cosine',
            logging_steps=50,
            save_steps=500,
            eval_steps=250,
            save_total_limit=2,
            fp16=True,
            dataloader_drop_last=True,
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to=[],
            seed=42,
        )


def main():
    parser = argparse.ArgumentParser(description="GUAC Post-training Script")

    parser.add_argument('--base-model', type=str, required=True,
                       help='Path to base model (pre-trained or mid-trained)')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['instruction', 'sft'],
                       help='Post-training mode')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='Instruction/conversation datasets')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--output-dir', type=str,
                       help='Training output directory')
    parser.add_argument('--resume-from-checkpoint', type=str,
                       help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with smaller datasets')
    parser.add_argument('--enable-profiling', action='store_true',
                       help='Enable profiling callbacks')
    parser.add_argument('--force-retokenize', action='store_true',
                       help='Force re-tokenize dataset even if cached')

    args = parser.parse_args()

    model, tokenizer = load_base_model(args.base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Preparing dataset for {args.mode} mode...")
    dataset = prepare_posttraining_dataset(args.mode, tokenizer, args.datasets, args.force_retokenize)

    if args.debug:
        dataset = apply_debug_mode(dataset, 500)

    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    base_model_name = Path(args.base_model).name
    output_dir = get_output_directory("posttraining", base_model_name, args.output_dir, mode=args.mode)

    training_args = create_posttraining_arguments(output_dir, args.mode)

    callbacks = [
        CustomProgressCallback(tokenizer, 25),
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)
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

    print(f"Starting {args.mode} post-training...")
    print(f"Base model: {args.base_model}")
    print(f"Training samples: {len(split_dataset['train']):,}")
    print(f"Output: {output_dir}")

    run_training_loop(trainer, args.resume_from_checkpoint)
    save_trained_model(trainer, tokenizer, output_dir, save_config=False)

    print(f"Post-training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
