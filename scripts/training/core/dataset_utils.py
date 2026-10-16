#!/usr/bin/env python3
"""
Shared dataset utilities for Guac training pipeline.

This module contains common dataset loading, tokenization, and caching functions
used across all training stages.
"""

import os
from typing import Any, List, Dict, Optional, Tuple
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset, DatasetDict, IterableDataset

try:
    from .cache_utils import is_saved_dataset
except ImportError:
    from scripts.training.core.cache_utils import is_saved_dataset


def _get_pretraining_dataset_configs() -> Dict[str, Any]:
    """Look up HF dataset definitions (fineweb, fineweb-edu, etc.) from site_config.yml."""
    try:
        from .path_manager import get_dataset_configs
    except ImportError:
        from scripts.training.core.path_manager import get_dataset_configs
    return get_dataset_configs().get('pretraining_datasets', {})


def _load_huggingface_pretraining_dataset(dataset_config: Dict[str, Any], validation_docs: int,
                                          seed: int) -> Tuple[Any, Optional[Dataset]]:
    """
    Load one HF dataset configured in site_config.yml as (train, validation).

    The validation documents come off the front of the dataset (a seeded
    random sample for map-style datasets, the first documents when
    streaming), and max_articles then limits the training documents.
    """
    config_name = dataset_config.get('config_name')
    streaming = dataset_config.get('streaming', False)
    max_articles = dataset_config.get('max_articles')
    text_field = dataset_config.get('text_field', 'text')

    print(f"  Loading {dataset_config['dataset_name']}" +
          (f"/{config_name}" if config_name else "") +
          f" (streaming: {streaming}, max_articles: {max_articles})")

    dataset = load_dataset(dataset_config['dataset_name'], config_name, streaming=streaming, split='train')

    if text_field != 'text':
        dataset = dataset.rename_column(text_field, 'text')

    train, validation = split_validation(dataset, validation_docs, seed)

    if max_articles:
        if streaming:
            train = train.take(max_articles)
        elif len(train) > max_articles:
            train = train.select(range(max_articles))

    return train, validation


def is_streaming(dataset: Any) -> bool:
    return isinstance(dataset, IterableDataset)


def split_validation(dataset: Any, validation_docs: int, seed: int = 0) -> Tuple[Any, Optional[Dataset]]:
    """
    Hold out validation_docs documents as (train, validation); validation is None for 0.

    Map-style datasets get a seeded random split. Streaming datasets have no
    random access, so their first validation_docs documents are read into
    memory as the validation set and skipped for training.
    """
    if not validation_docs:
        return dataset, None
    if is_streaming(dataset):
        validation = Dataset.from_list(list(dataset.take(validation_docs)))
        return dataset.skip(validation_docs), validation
    if validation_docs >= len(dataset):
        raise ValueError(f"Can't hold out {validation_docs} validation documents from a dataset of {len(dataset)}")
    split = dataset.train_test_split(test_size=validation_docs, seed=seed)
    return split['train'], split['test']


def load_pretraining_splits(dataset_names: List[str], validation_docs: int = 1000,
                            seed: int = 0) -> Tuple[Any, Optional[Dataset]]:
    """
    Load pre-training datasets as (train, validation).

    Datasets with an official validation split (wikitext-103, or an on-disk
    DatasetDict with a 'validation' split) use it; the others hold out
    validation_docs documents each (see split_validation). Pass
    validation_docs=0 for no validation set (validation is then None).
    """
    hf_dataset_configs = _get_pretraining_dataset_configs()
    trains, validations = [], []

    for dataset_name in dataset_names:
        print(f"📚 Loading dataset: {dataset_name}")

        if dataset_name in hf_dataset_configs:
            train, validation = _load_huggingface_pretraining_dataset(
                hf_dataset_configs[dataset_name], validation_docs, seed)

        elif dataset_name == 'wikitext-103':
            dataset = load_dataset('wikitext', 'wikitext-103-v1')
            train, validation = dataset['train'], dataset['validation'] if validation_docs else None

        elif dataset_name == 'openwebtext':
            train, validation = split_validation(load_dataset('openwebtext')['train'], validation_docs, seed)

        elif os.path.exists(dataset_name):
            dataset = load_from_disk(dataset_name)
            if isinstance(dataset, DatasetDict):
                if 'validation' in dataset and validation_docs:
                    train, validation = dataset['train'], dataset['validation']
                else:
                    train, validation = split_validation(dataset['train'], validation_docs, seed)
            else:
                train, validation = split_validation(dataset, validation_docs, seed)

        else:
            raise ValueError(f"Unknown dataset {dataset_name!r}: not defined in site_config.yml, "
                             f"not a built-in name (wikitext-103, openwebtext), and not a path on disk")

        trains.append(train)
        if validation is not None:
            validations.append(validation)

    train = trains[0] if len(trains) == 1 else concatenate_datasets(trains)
    validation = None
    if validations:
        validation = validations[0] if len(validations) == 1 else concatenate_datasets(validations)
    return train, validation


def load_pretraining_datasets(dataset_names: List[str]) -> Any:
    """Load the training split of each pre-training dataset, concatenated (e.g. for tokenizer training)."""
    return load_pretraining_splits(dataset_names, validation_docs=0)[0]


def get_pretraining_dataset_settings(dataset_names: List[str]) -> Dict[str, Any]:
    """
    Settings that determine each dataset's content, for use in cache keys.

    Datasets defined in site_config.yml map to their full entry (HF name,
    config, max_articles, text field, ...); built-in and on-disk datasets map
    to their name.
    """
    hf_dataset_configs = _get_pretraining_dataset_configs()
    return {name: hf_dataset_configs.get(name, name) for name in sorted(dataset_names)}


def pack_documents(documents: List[List[int]], block_size: int, eos_token_id: Optional[int]) -> List[List[int]]:
    """
    Concatenate token lists (each ending in eos_token_id, appended if missing)
    and cut the stream into blocks of block_size tokens. The tokens after the
    last full block are dropped. Documents run across block boundaries, so no
    text is lost to truncation.
    """
    stream = []
    for ids in documents:
        stream.extend(ids)
        if eos_token_id is not None and (not ids or ids[-1] != eos_token_id):
            stream.append(eos_token_id)
    usable = len(stream) - len(stream) % block_size
    return [stream[i:i + block_size] for i in range(0, usable, block_size)]


class PackedSequenceCollator:
    """
    Batches packed blocks (all the same length) for causal LM training: labels
    are the input ids, with nothing masked. DataCollatorForLanguageModeling
    would mask every pad_token_id, which drops EOS separators from the loss
    for tokenizers whose pad token is EOS.
    """

    def __call__(self, examples):
        import torch
        input_ids = torch.tensor([example['input_ids'] for example in examples], dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids), 'labels': input_ids.clone()}


def tokenize_dataset_with_cache(
    dataset: Any,
    tokenizer: Any,
    cache_dir: str,
    max_length: int = 1024,
    force_retokenize: bool = False,
    pack: bool = False,
) -> Any:
    """
    Tokenize a text dataset, reusing the cached result when there is one.

    With pack, documents are concatenated with EOS between them and cut into
    blocks of exactly max_length tokens (pack_documents). Without it, each
    document is truncated to max_length and those of 10 tokens or fewer are
    dropped. Streaming datasets are processed lazily and not cached.
    """
    if is_saved_dataset(cache_dir) and not force_retokenize:
        print(f"📦 Loading cached tokenized dataset from {cache_dir}")
        cached_dataset = load_from_disk(cache_dir)
        print(f"✅ Loaded {len(cached_dataset):,} cached samples")
        return cached_dataset

    def tokenize_function(examples):
        texts = [text for text in examples['text'] if text and text.strip()]
        if pack:
            ids = tokenizer(texts, truncation=False, padding=False, return_attention_mask=False)['input_ids'] if texts else []
            return {'input_ids': pack_documents(ids, max_length, tokenizer.eos_token_id)}
        if not texts:
            return {'input_ids': [], 'attention_mask': []}
        return tokenizer(texts, truncation=True, padding=False, max_length=max_length, return_tensors=None)

    print(f"🔤 Tokenizing dataset{' (packing into ' + str(max_length) + '-token blocks)' if pack else ''}...")
    progress = {} if is_streaming(dataset) else {'desc': "Tokenizing"}
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, **progress)
    if not pack:
        progress = {} if is_streaming(dataset) else {'desc': "Filtering short sequences"}
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 10, **progress)

    if is_streaming(dataset):
        print("🔄 Streaming dataset - skipping cache save")
    else:
        print(f"💾 Saving tokenized dataset to cache: {cache_dir}")
        tokenized_dataset.save_to_disk(cache_dir)
        print(f"✅ Cached {len(tokenized_dataset):,} samples")

    return tokenized_dataset


def load_instruction_datasets(dataset_names: List[str]) -> Dataset:
    """Load instruction datasets for post-training."""
    datasets = []

    for dataset_name in dataset_names:
        print(f"📚 Loading instruction dataset: {dataset_name}")

        try:
            if dataset_name == 'alpaca':
                dataset = load_dataset("tatsu-lab/alpaca")
                datasets.append(dataset['train'])

            elif dataset_name == 'dolly':
                dataset = load_dataset("databricks/databricks-dolly-15k")
                datasets.append(dataset['train'])

            elif os.path.exists(dataset_name):
                # Load from file
                if dataset_name.endswith('.json'):
                    import json
                    with open(dataset_name) as f:
                        data = json.load(f)
                    datasets.append(Dataset.from_list(data))
                else:
                    datasets.append(load_from_disk(dataset_name))

        except Exception as e:
            print(f"⚠️  Could not load dataset {dataset_name}: {e}")

    if not datasets:
        print("📝 No external datasets loaded, returning None")
        return None

    # Combine datasets
    if len(datasets) == 1:
        return datasets[0]
    else:
        return concatenate_datasets(datasets)


def format_instruction_data(examples: Dict[str, List], tokenizer: Any, max_length: int = 512) -> Dict[str, List]:
    """Format instruction data with proper templates."""
    formatted_texts = []

    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        response = examples['response'][i]

        # Use a simple instruction template
        formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{tokenizer.eos_token}"
        formatted_texts.append(formatted_text)

    # Tokenize
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None
    )

    return tokenized


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

    # Tokenize
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None
    )

    return tokenized


def create_synthetic_instruction_dataset(tokenizer: Any, num_samples: int = 10000, force_retokenize: bool = False) -> Dataset:
    """Create synthetic instruction dataset with caching support."""
    from .pipeline_utils import create_cache_dir_name

    # Create cache directory
    cache_dir = create_cache_dir_name("instruction", tokenizer, num_samples=num_samples)

    # Check if cached version exists
    if os.path.exists(cache_dir) and not force_retokenize:
        print(f"📦 Loading cached instruction dataset from {cache_dir}")
        try:
            cached_dataset = load_from_disk(cache_dir)
            print(f"✅ Loaded {len(cached_dataset):,} cached instruction samples")
            return cached_dataset
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}, regenerating...")

    print(f"📝 Generating {num_samples} instruction examples...")

    import random
    instructions = []

    # Math instructions
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
            # Ensure clean division
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

    # General instructions
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

    # Expand general examples
    for _ in range(num_samples // 4):
        instructions.append(random.choice(general_examples))

    # Q&A examples
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

    # Creative tasks
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

    # Create dataset
    dataset = Dataset.from_list(instructions)

    # Save to cache
    try:
        print(f"💾 Saving instruction dataset to cache: {cache_dir}")
        dataset.save_to_disk(cache_dir)
        print(f"✅ Cached {len(dataset):,} instruction samples")
    except Exception as e:
        print(f"⚠️  Failed to save cache: {e}")

    return dataset


def create_synthetic_conversation_dataset(tokenizer: Any, force_retokenize: bool = False) -> Dataset:
    """Create conversation dataset for SFT with caching support."""
    from .pipeline_utils import create_cache_dir_name
    import random

    # Create cache directory
    cache_dir = create_cache_dir_name("conversation", tokenizer, samples=1000)

    # Check if cached version exists
    if os.path.exists(cache_dir) and not force_retokenize:
        print(f"📦 Loading cached conversation dataset from {cache_dir}")
        try:
            cached_dataset = load_from_disk(cache_dir)
            print(f"✅ Loaded {len(cached_dataset):,} cached conversation samples")
            return cached_dataset
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}, regenerating...")

    print("💬 Creating conversation dataset...")

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

    # Expand conversations
    expanded = []
    for _ in range(1000):
        expanded.append(random.choice(conversations))

    # Create dataset
    dataset = Dataset.from_list(expanded)

    # Save to cache
    try:
        print(f"💾 Saving conversation dataset to cache: {cache_dir}")
        dataset.save_to_disk(cache_dir)
        print(f"✅ Cached {len(dataset):,} conversation samples")
    except Exception as e:
        print(f"⚠️  Failed to save cache: {e}")

    return dataset