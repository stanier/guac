#!/usr/bin/env python3
"""
Cache utilities for GUAC training pipeline.

Provides unified caching functionality for tokenized datasets across all training stages.
"""

import os
from typing import Any, Optional
from datasets import load_from_disk


def get_cache_directory(cache_type: str, cache_key: str) -> str:
    """
    Get cache directory path with fallback handling.

    Args:
        cache_type: Type of cache (e.g., 'pretraining', 'midtraining', 'posttraining')
        cache_key: Unique cache key identifier

    Returns:
        Path to cache directory
    """
    try:
        from .path_manager import get_tokenized_dataset_path
        return str(get_tokenized_dataset_path(cache_type, cache_key))
    except ImportError:
        return f"./tokenized_cache_{cache_key}"


def load_cached_dataset(cache_dir: str, force_retokenize: bool = False) -> Optional[Any]:
    """
    Load cached dataset if it exists and force_retokenize is False.

    Args:
        cache_dir: Directory containing cached dataset
        force_retokenize: If True, skip loading cache

    Returns:
        Loaded dataset or None if cache doesn't exist or failed to load
    """
    if not os.path.exists(cache_dir) or force_retokenize:
        return None

    print(f"Loading cached dataset from {cache_dir}")
    try:
        cached_dataset = load_from_disk(cache_dir)
        print(f"Loaded {len(cached_dataset):,} cached samples")
        return cached_dataset
    except Exception as e:
        print(f"Failed to load cache: {e}, regenerating")
        return None


def save_dataset_to_cache(dataset: Any, cache_dir: str) -> None:
    """
    Save dataset to cache directory.

    Args:
        dataset: Dataset to cache
        cache_dir: Directory to save cache to
    """
    try:
        print(f"Saving to cache: {cache_dir}")
        dataset.save_to_disk(cache_dir)
        print(f"Cached {len(dataset):,} samples")
    except Exception as e:
        print(f"Failed to save cache: {e}")


def get_vocab_size(tokenizer: Any) -> Any:
    """
    Get tokenizer vocabulary size with fallback.

    Args:
        tokenizer: Tokenizer instance

    Returns:
        Vocabulary size (int) or 'unknown' if unavailable
    """
    return len(tokenizer) if hasattr(tokenizer, '__len__') else 'unknown'
