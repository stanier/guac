"""
Tests for pre-training data and run records: validation hold-outs, sequence
packing, the packed-batch collator, and the metrics and run-info files.
"""

import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch
from datasets import Dataset, DatasetDict, IterableDataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from guac.core import dataset_utils
from guac.core.dataset_utils import (
    PackedSequenceCollator, load_pretraining_splits, pack_documents, split_validation, tokenize_dataset_with_cache,
)
from guac.core.pipeline_utils import GuacTrainer, MetricsLoggerCallback, write_run_info


def make_tokenizer():
    words = [f"w{i}" for i in range(20)]
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, **{word: i + 3 for i, word in enumerate(words)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")


def test_pack_documents_joins_with_eos_and_cuts_full_blocks():
    blocks = pack_documents([[5, 6, 7], [8, 9], [10, 2]], block_size=4, eos_token_id=2)
    # stream: 5 6 7 2 8 9 2 10 2 -> two full blocks, the trailing token is dropped
    assert blocks == [[5, 6, 7, 2], [8, 9, 2, 10]]


def test_packing_keeps_every_document_token():
    tokenizer = make_tokenizer()
    texts = [" ".join(f"w{(i * 7 + j) % 20}" for j in range(n)) for i, n in enumerate([3, 25, 1, 9, 14])]

    stream = [token for text in texts for token in tokenizer(text)['input_ids'] + [2]]
    blocks = pack_documents([tokenizer(text)['input_ids'] for text in texts], 8, tokenizer.eos_token_id)
    assert all(len(block) == 8 for block in blocks)
    assert sum(blocks, []) == stream[:len(blocks) * 8]
    assert len(stream) - len(blocks) * 8 < 8


def test_tokenize_with_packing_writes_full_blocks(tmp_path):
    tokenizer = make_tokenizer()
    texts = [" ".join(f"w{j % 20}" for j in range(n)) for n in [3, 30, 2, 12, 40]]
    packed = tokenize_dataset_with_cache(Dataset.from_dict({'text': texts}), tokenizer, str(tmp_path / "cache"),
                                         max_length=16, pack=True)
    assert set(len(ids) for ids in packed['input_ids']) == {16}
    assert len(packed) == (sum(len(t.split()) + 1 for t in texts)) // 16
    # Reloads from the cache
    again = tokenize_dataset_with_cache(Dataset.from_dict({'text': ["w1"]}), tokenizer, str(tmp_path / "cache"),
                                        max_length=16, pack=True)
    assert again['input_ids'] == packed['input_ids']


def test_unpacked_tokenization_truncates_and_drops_short_documents(tmp_path):
    tokenizer = make_tokenizer()
    texts = [" ".join(["w1"] * n) for n in [5, 30]]
    tokenized = tokenize_dataset_with_cache(Dataset.from_dict({'text': texts}), tokenizer, str(tmp_path / "c"),
                                            max_length=16, pack=False)
    assert [len(ids) for ids in tokenized['input_ids']] == [16]


def test_packed_collator_keeps_eos_in_the_loss_even_when_it_is_the_pad_token():
    batch = PackedSequenceCollator()([{'input_ids': [5, 0, 6]}, {'input_ids': [0, 7, 8]}])
    assert torch.equal(batch['labels'], batch['input_ids'])
    assert batch['attention_mask'].all()


def test_split_validation_is_seeded_and_disjoint():
    dataset = Dataset.from_dict({'text': [f"doc {i}" for i in range(50)]})
    train, validation = split_validation(dataset, 10, seed=3)
    again_train, again_validation = split_validation(dataset, 10, seed=3)
    assert len(validation) == 10 and len(train) == 40
    assert validation['text'] == again_validation['text']
    assert not set(train['text']) & set(validation['text'])


def test_split_validation_of_a_stream_takes_the_first_documents():
    stream = IterableDataset.from_generator(lambda: ({'text': f"doc {i}"} for i in range(20)))
    train, validation = split_validation(stream, 5)
    assert isinstance(validation, Dataset)
    assert validation['text'] == [f"doc {i}" for i in range(5)]
    assert [row['text'] for row in train] == [f"doc {i}" for i in range(5, 20)]


def test_no_validation_when_disabled():
    dataset = Dataset.from_dict({'text': ["a", "b"]})
    train, validation = split_validation(dataset, 0)
    assert validation is None and len(train) == 2


def test_on_disk_datasets_use_their_own_validation_split(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_utils, '_get_pretraining_dataset_configs', lambda: {})
    plain = tmp_path / "plain"
    Dataset.from_dict({'text': [f"p{i}" for i in range(30)]}).save_to_disk(str(plain))
    with_split = tmp_path / "with_split"
    DatasetDict({'train': Dataset.from_dict({'text': ["t1", "t2", "t3"]}),
                 'validation': Dataset.from_dict({'text': ["v1"]})}).save_to_disk(str(with_split))

    train, validation = load_pretraining_splits([str(plain), str(with_split)], validation_docs=4)
    assert len(train) == 26 + 3
    assert len(validation) == 4 + 1 and "v1" in validation['text']


def test_unknown_dataset_is_an_error(monkeypatch):
    monkeypatch.setattr(dataset_utils, '_get_pretraining_dataset_configs', lambda: {})
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_pretraining_splits(["no-such-dataset"])


def test_trainer_logs_perplexity_and_metrics_file(tmp_path):
    from transformers import TrainingArguments
    from guac.core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM

    torch.manual_seed(0)
    model = MixedLayerForCausalLM(MixedLayerConfig(vocab_size=23, hidden_size=32, num_attention_heads=4,
                                                   num_hidden_layers=1, layer_types=["conv"], intermediate_size=64))
    data = Dataset.from_dict({'input_ids': [[(i + j) % 23 for j in range(8)] for i in range(16)]})
    args = TrainingArguments(output_dir=str(tmp_path), max_steps=4, per_device_train_batch_size=4,
                             per_device_eval_batch_size=4, logging_steps=2, eval_strategy="steps", eval_steps=2,
                             save_strategy="no", report_to=[], use_cpu=True)
    metrics_path = tmp_path / "metrics.jsonl"
    trainer = GuacTrainer(model=model, args=args, train_dataset=data, eval_dataset=data,
                          data_collator=PackedSequenceCollator(), callbacks=[MetricsLoggerCallback(str(metrics_path))])
    trainer.train()

    entries = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    evals = [e for e in entries if 'eval_loss' in e]
    trains = [e for e in entries if 'loss' in e]
    assert [e['step'] for e in evals] == [2, 4]
    assert [e['step'] for e in trains] == [2, 4]
    for entry in evals:
        assert entry['eval_perplexity'] == pytest.approx(math.exp(entry['eval_loss']))
    for entry in trains:
        assert entry['perplexity'] == pytest.approx(math.exp(entry['loss']))


def test_run_info_records_what_reproduces_a_run(tmp_path):
    from guac.core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM
    model = MixedLayerForCausalLM(MixedLayerConfig(vocab_size=23, hidden_size=32, num_attention_heads=4,
                                                   num_hidden_layers=1, layer_types=["conv"], intermediate_size=64))
    path = write_run_info(str(tmp_path), 'pretraining', {'hidden_size': 32, 'learning_rate': 1e-3}, model=model,
                          seed=7, datasets=['x'])
    info = json.loads(Path(path).read_text())
    assert info['seed'] == 7
    assert info['config']['learning_rate'] == 1e-3
    assert info['parameters'] == model.num_parameters()
    assert info['model_config']['layer_types'] == ["conv"]
    assert len(info['git']['commit']) == 40
    assert info['datasets'] == ['x']


def test_packing_a_stream_with_unknown_columns(tmp_path):
    """Streams built from generators or CSV report no column names; packing must still drop them."""
    tokenizer = make_tokenizer()
    stream = IterableDataset.from_generator(lambda: ({'text': "w1 w2 w3 w4 w5", 'id': i} for i in range(20)))
    assert stream.column_names is None
    packed = list(tokenize_dataset_with_cache(stream, tokenizer, str(tmp_path / "c"), max_length=8, pack=True))
    assert packed and all(set(row) == {'input_ids'} and len(row['input_ids']) == 8 for row in packed)


def test_debug_runs_never_fill_the_full_runs_cache(tmp_path, monkeypatch):
    """A --debug run on a stream used to cache its 200-document validation set under the full run's key."""
    from guac.pipeline import pretrain

    def splits(names, validation_docs, seed=0):
        stream = IterableDataset.from_generator(lambda: ({'text': f"w{i % 20} w1 w2 w3"} for i in range(3000)))
        return stream, Dataset.from_dict({'text': [" ".join(["w2"] * 12) for _ in range(500)]})

    monkeypatch.setattr(pretrain, 'load_pretraining_splits', splits)
    monkeypatch.setattr(pretrain, 'get_cache_directory', lambda kind, key: str(tmp_path / key))
    tokenizer = make_tokenizer()
    _, debug_validation = pretrain.prepare_pretraining_data(['s'], tokenizer, 16, pack=False, validation_docs=500, debug=True)
    _, full_validation = pretrain.prepare_pretraining_data(['s'], tokenizer, 16, pack=False, validation_docs=500)
    assert len(debug_validation) == 200 and len(full_validation) == 500
