#!/usr/bin/env python3
"""
Guac Dataset Tokenization Script

Pre-tokenize and cache datasets for training.

Usage:
    python tokenize_datasets.py --tokenizer scratch/tokenizers/sentencepiece/shared_60k --datasets wikitext-103
    python tokenize_datasets.py --tokenizer ./my_tokenizer --datasets fineweb,wikitext-103 --max-length 2048
    python tokenize_datasets.py --tokenizer ./my_tokenizer --datasets fineweb --stage pretraining
"""

import os, sys, argparse
from typing import List, Optional
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, concatenate_datasets

try:
    from scripts.training.core.dataset_utils import load_pretraining_datasets
    from scripts.training.core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size
except ImportError:
    try:
        from ..core.dataset_utils import load_pretraining_datasets
        from ..core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size
    except ImportError:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        from core.dataset_utils import load_pretraining_datasets
        from core.cache_utils import get_cache_directory, load_cached_dataset, save_dataset_to_cache, get_vocab_size


def tokenize_dataset(dataset: Any, tokenizer: Any, max_length: int = 1024,
                     dataset_names: List[str] = None, cache_type: str = "pretraining",
                     force_retokenize: bool = False) -> Any:
    """Tokenize dataset for language modeling with caching support."""
    dataset_key = "_".join(sorted(dataset_names)) if dataset_names else "unknown"
    vocab_size = get_vocab_size(tokenizer)
    cache_key = f"{dataset_key}_vocab{vocab_size}_len{max_length}"
    cache_dir = get_cache_directory(cache_type, cache_key)

    cached = load_cached_dataset(cache_dir, force_retokenize)
    if cached is not None:
        return cached

    def tokenize_function(examples):
        texts = [text for text in examples['text'] if text and text.strip()]
        if not texts:
            return {'input_ids': [], 'attention_mask': []}

        return tokenizer(texts, truncation=True, padding=False,
                        max_length=max_length, return_tensors=None)

    print("Tokenizing dataset...")
    is_streaming = hasattr(dataset, 'iter') and not hasattr(dataset, 'select')

    if is_streaming:
        print("Processing streaming dataset...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                       remove_columns=dataset.column_names)
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 10)

        # For streaming datasets, we need to materialize a portion for caching
        print("Materializing streaming dataset for caching...")
        max_samples = 100000  # Limit for streaming datasets
        materialized = []
        count = 0
        for item in tokenized_dataset:
            materialized.append(item)
            count += 1
            if count >= max_samples:
                break
            if count % 10000 == 0:
                print(f"  Processed {count:,} samples...")

        from datasets import Dataset
        tokenized_dataset = Dataset.from_list(materialized)
        print(f"Materialized {len(tokenized_dataset):,} samples")
    else:
        tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                       remove_columns=dataset.column_names, desc="Tokenizing")
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 10,
                                                     desc="Filtering short sequences")

    save_dataset_to_cache(tokenized_dataset, cache_dir)
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Guac Dataset Tokenization Script")

    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to trained tokenizer directory')
    parser.add_argument('--datasets', type=str, required=True,
                       help='Comma-separated list of datasets to tokenize')
    parser.add_argument('--max-length', type=int, default=1024,
                       help='Maximum sequence length for tokenization')
    parser.add_argument('--stage', type=str, default='pretraining',
                       choices=['pretraining', 'midtraining', 'posttraining'],
                       help='Training stage for cache organization')
    parser.add_argument('--force-retokenize', action='store_true',
                       help='Force re-tokenization even if cached dataset exists')
    parser.add_argument('--max-samples', type=int,
                       help='Limit dataset to this many samples (for testing)')

    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"Loaded tokenizer with vocab size: {len(tokenizer):,}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Make sure to train a tokenizer first:")
        print("  python scripts/training/pipeline/train_tokenizer.py")
        return

    dataset_names = args.datasets.split(',')
    print(f"\nLoading datasets: {', '.join(dataset_names)}")

    try:
        dataset = load_pretraining_datasets(dataset_names)

        # Get dataset size info
        try:
            dataset_size = len(dataset)
            print(f"Loaded dataset with {dataset_size:,} samples")
        except TypeError:
            print("Loaded streaming dataset")
            dataset_size = None

        # Limit samples if requested
        if args.max_samples and dataset_size:
            if args.max_samples < dataset_size:
                print(f"Limiting to {args.max_samples:,} samples for testing")
                dataset = dataset.select(range(args.max_samples))
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    print(f"\nTokenizing with max length: {args.max_length}")
    print(f"Cache type: {args.stage}")

    tokenized_dataset = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        dataset_names=dataset_names,
        cache_type=args.stage,
        force_retokenize=args.force_retokenize
    )

    try:
        final_size = len(tokenized_dataset)
        print(f"\nTokenization complete!")
        print(f"Tokenized samples: {final_size:,}")
        print(f"Average tokens per sample: {sum(len(x['input_ids']) for x in tokenized_dataset.select(range(min(1000, final_size)))) / min(1000, final_size):.1f}")
    except:
        print("\nTokenization complete!")

    dataset_key = "_".join(sorted(dataset_names))
    vocab_size = get_vocab_size(tokenizer)
    cache_key = f"{dataset_key}_vocab{vocab_size}_len{args.max_length}"
    cache_dir = get_cache_directory(args.stage, cache_key)
    print(f"Cache location: {cache_dir}")


if __name__ == "__main__":
    main()
