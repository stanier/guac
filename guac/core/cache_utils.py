#!/usr/bin/env python3
"""
Cache utilities for Guac training pipeline.

Provides unified caching functionality for tokenized datasets across all training stages.
"""

import hashlib
import json
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
    from .path_manager import get_tokenized_dataset_path
    return str(get_tokenized_dataset_path(cache_type, cache_key))


def is_saved_dataset(path: str) -> bool:
    """
    Check whether a directory holds a dataset written by save_to_disk.

    Cache directories are created empty before anything is saved into them,
    so the directory existing is not enough.
    """
    return any(os.path.isfile(os.path.join(path, name)) for name in ('state.json', 'dataset_dict.json'))


def load_cached_dataset(cache_dir: str, force_retokenize: bool = False) -> Optional[Any]:
    """
    Load cached dataset if it exists and force_retokenize is False.

    Args:
        cache_dir: Directory containing cached dataset
        force_retokenize: If True, skip loading cache

    Returns:
        Loaded dataset or None if cache doesn't exist or failed to load
    """
    if not is_saved_dataset(cache_dir) or force_retokenize:
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


def content_hash(value: Any, length: int = 10) -> str:
    """
    Short, stable hash of a string or JSON-serializable value, for cache keys.

    Args:
        value: String, or value to serialize as JSON with sorted keys
        length: Number of hex characters to keep

    Returns:
        Hex digest prefix
    """
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:length]


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """
    Hash identifying a tokenizer's behavior, so caches built with one tokenizer
    are never reused with another that happens to have the same vocab size.

    Fast tokenizers are hashed from their full serialized definition (vocab,
    normalizer, pre-tokenizer, post-processor); others from vocab and special tokens.

    Args:
        tokenizer: Tokenizer instance

    Returns:
        Hex digest prefix
    """
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is not None:
        definition = json.loads(backend.to_str())
        # The serialization also records the truncation and padding of the last call, which
        # aren't part of the tokenizer and would make the key depend on what ran before
        definition.pop('truncation', None)
        definition.pop('padding', None)
        description = json.dumps(definition, sort_keys=True)
    else:
        description = json.dumps(sorted(tokenizer.get_vocab().items()))
    return content_hash(description + json.dumps(tokenizer.special_tokens_map, sort_keys=True, default=str))


def tokenizer_cache_tag(tokenizer: Any) -> str:
    """
    Readable cache-key fragment naming a tokenizer, e.g. 'vocab100_tok3fa2b1c9'.

    Args:
        tokenizer: Tokenizer instance

    Returns:
        Vocab size plus a short tokenizer fingerprint
    """
    return f"vocab{get_vocab_size(tokenizer)}_tok{tokenizer_fingerprint(tokenizer)[:8]}"
