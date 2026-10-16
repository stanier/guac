"""
Tests for post-training data: Alpaca, Dolly and conversation records all
format, and loss covers only responses and assistant turns.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from scripts.training.pipeline.posttrain import (
    load_records, prepare_posttraining_dataset, synthetic_instruction_records, to_segments, tokenize_segments,
)


def make_tokenizer():
    """Character-level, so every character is one token and lengths are easy to check."""
    chars = sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 #:\n?=+-*/().,'"))
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, **{c: i + 3 for i, c in enumerate(chars)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")


def trained_text(tokenizer, example):
    ids = [t for t, label in zip(example['input_ids'], example['labels']) if label != -100]
    return "".join(tokenizer.convert_ids_to_tokens(ids))


def test_alpaca_records_use_output_and_input():
    """Alpaca's response field is 'output'; formatting it used to raise KeyError."""
    segments = to_segments({'instruction': "Add them", 'input': "2 and 3", 'output': "5"})
    assert segments[0] == ("### Instruction:\nAdd them\n\n### Input:\n2 and 3\n\n### Response:\n", False)
    assert segments[1] == ("5", True)


def test_dolly_records_use_context_and_response():
    segments = to_segments({'instruction': "Summarize", 'context': "Long text", 'response': "Short"})
    assert "### Input:\nLong text" in segments[0][0] and segments[1] == ("Short", True)


def test_records_without_an_input_skip_that_section():
    prompt, _ = to_segments({'instruction': "Say hi", 'input': "", 'output': "hi"})[0]
    assert "### Input" not in prompt


def test_conversations_train_only_on_assistant_turns():
    segments = to_segments({'messages': [
        {'role': 'user', 'content': "Hi"}, {'role': 'assistant', 'content': "Hello"},
        {'role': 'user', 'content': "Bye"}, {'role': 'assistant', 'content': "Goodbye"},
    ]})
    assert [text for text, trained in segments if trained] == ["Hello\n", "Goodbye\n"]
    assert "".join(text for text, _ in segments) == "User: Hi\nAssistant: Hello\nUser: Bye\nAssistant: Goodbye\n"


def test_unrecognized_records_are_an_error():
    with pytest.raises(ValueError, match="neither messages nor instruction"):
        to_segments({'question': "?", 'answer': "!"})


def test_prompt_tokens_are_masked_and_eos_is_trained():
    tokenizer = make_tokenizer()
    example = tokenize_segments([("Q: 2+2=", False), ("4", True)], tokenizer, max_length=64)
    assert example['labels'][:7] == [-100] * 7
    assert trained_text(tokenizer, example) == "4[EOS]"
    assert example['input_ids'][-1] == tokenizer.eos_token_id


def test_truncation_keeps_inputs_and_labels_aligned():
    tokenizer = make_tokenizer()
    example = tokenize_segments([("abcdef", False), ("ghij", True)], tokenizer, max_length=8)
    assert len(example['input_ids']) == len(example['labels']) == len(example['attention_mask']) == 8
    assert trained_text(tokenizer, example) == "gh"


def test_sft_datasets_from_disk_are_tokenized(tmp_path):
    """--mode sft --datasets used to return the raw records, untokenized."""
    path = tmp_path / "chats.jsonl"
    rows = [{'messages': [{'role': 'user', 'content': "hi"}, {'role': 'assistant', 'content': "hello"}]}] * 3
    path.write_text("\n".join(json.dumps(row) for row in rows))
    dataset = prepare_posttraining_dataset([str(path)], make_tokenizer(), max_length=64)
    assert set(dataset.column_names) == {'input_ids', 'attention_mask', 'labels'} and len(dataset) == 3


def test_examples_truncated_before_their_response_are_dropped(tmp_path):
    path = tmp_path / "long.json"
    path.write_text(json.dumps([{'instruction': "x" * 100, 'output': "y"}, {'instruction': "short", 'output': "ok"}]))
    dataset = prepare_posttraining_dataset([str(path)], make_tokenizer(), max_length=60)
    assert len(dataset) == 1


def test_train_on_prompts_unmasks_everything(tmp_path):
    path = tmp_path / "one.json"
    path.write_text(json.dumps([{'instruction': "hi", 'output': "ok"}]))
    dataset = prepare_posttraining_dataset([str(path)], make_tokenizer(), max_length=64, train_on_prompts=True)
    assert -100 not in dataset[0]['labels']


def test_synthetic_set_comes_from_the_arithmetic_training_pool():
    records = synthetic_instruction_records(40)
    assert len(records) == 40 and all(r['output'].startswith(r['instruction'][8:-1]) for r in records)


def test_unknown_dataset_names_are_an_error():
    with pytest.raises(ValueError, match="Unknown post-training dataset"):
        load_records("not-a-dataset")
