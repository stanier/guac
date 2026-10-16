#!/usr/bin/env python3
"""
Shared dataset utilities for Guac training pipeline.

This module contains common dataset loading, tokenization, and caching functions
used across all training stages.
"""

import os
from typing import Any, List, Dict, Optional, Tuple
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset, DatasetDict, IterableDataset

from .cache_utils import is_saved_dataset


def _get_pretraining_dataset_configs() -> Dict[str, Any]:
    """Look up HF dataset definitions (fineweb, fineweb-edu, etc.) from site_config.yml."""
    from .path_manager import get_dataset_configs
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
