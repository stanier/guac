"""
Tests that tokenized-dataset cache keys change whenever the cached content would.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from guac.core import dataset_utils
from guac.core.cache_utils import tokenizer_cache_tag
from guac.pipeline.pretrain import pretraining_cache_key


def make_tokenizer(words):
    vocab = {"[UNK]": 0, **{word: i + 1 for i, word in enumerate(words)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


@pytest.fixture
def site_datasets(monkeypatch):
    """Stand-in for the pretraining_datasets section of site_config.yml."""
    configs = {'fineweb': {'dataset_name': 'HuggingFaceFW/fineweb', 'streaming': True, 'max_articles': 1000}}
    monkeypatch.setattr(dataset_utils, '_get_pretraining_dataset_configs', lambda: configs)
    return configs


def test_key_is_stable(site_datasets):
    tokenizer = make_tokenizer(["a", "b", "c"])
    assert pretraining_cache_key(['fineweb'], tokenizer, 256) == pretraining_cache_key(['fineweb'], tokenizer, 256)


def test_debug_subset_gets_its_own_key(site_datasets):
    tokenizer = make_tokenizer(["a", "b", "c"])
    full = pretraining_cache_key(['wikitext-103'], tokenizer, 256)
    debug = pretraining_cache_key(['wikitext-103'], tokenizer, 256, debug_samples=1000)
    assert full != debug
    assert "_debug1000_" in debug


def test_tokenizers_with_same_vocab_size_get_different_keys(site_datasets):
    first, second = make_tokenizer(["a", "b", "c"]), make_tokenizer(["x", "y", "z"])
    assert len(first) == len(second)
    assert pretraining_cache_key(['wikitext-103'], first, 256) != pretraining_cache_key(['wikitext-103'], second, 256)
    assert tokenizer_cache_tag(first) != tokenizer_cache_tag(second)


def test_site_config_dataset_settings_change_key(site_datasets):
    tokenizer = make_tokenizer(["a", "b", "c"])
    before = pretraining_cache_key(['fineweb'], tokenizer, 256)
    site_datasets['fineweb']['max_articles'] = 2000
    assert pretraining_cache_key(['fineweb'], tokenizer, 256) != before


def test_sequence_length_changes_key(site_datasets):
    tokenizer = make_tokenizer(["a", "b", "c"])
    assert pretraining_cache_key(['fineweb'], tokenizer, 256) != pretraining_cache_key(['fineweb'], tokenizer, 512)


@pytest.mark.parametrize("change", [dict(split='validation'), dict(pack=False), dict(validation_docs=500)])
def test_split_packing_and_holdout_get_their_own_keys(site_datasets, change):
    """The hold-out size changes the training split too, so it is part of the key."""
    tokenizer = make_tokenizer(["a", "b", "c"])
    assert pretraining_cache_key(['fineweb'], tokenizer, 256, **change) != pretraining_cache_key(['fineweb'], tokenizer, 256)


def test_tokenizer_fingerprint_ignores_call_time_truncation(site_datasets):
    """Tokenizing with truncation records it in the backend; the cache key must not change with it."""
    tokenizer = make_tokenizer(["a", "b", "c"])
    before = pretraining_cache_key(['fineweb'], tokenizer, 256)
    tokenizer(["a b c"], truncation=True, max_length=2)
    assert pretraining_cache_key(['fineweb'], tokenizer, 256) == before
