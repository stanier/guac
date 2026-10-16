"""
Tests for the synthetic diagnostic tasks: each generator's answers follow its
rule, accuracy is computed on the answer positions only, and the training
loop runs end to end.
"""

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch

from guac.core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM
from guac.core.diagnostics import (
    IGNORE, TASKS, answer_accuracy, make_batch, run_diagnostic, run_language_modeling,
)
from guac.pipeline.run_diagnostics import diagnostic_model_config, parse_assignments


def answers(batch):
    """(positions, values) of the labeled answers in example 0."""
    labels = batch['labels'][0]
    positions = (labels != IGNORE).nonzero().squeeze(-1)
    return positions, labels[positions]


def test_mqar_answers_are_the_values_stored_under_the_queried_keys():
    batch = make_batch('mqar', 20, seed=0, seq_len=64, num_kv_pairs=6, vocab_size=64)
    for ids, labels in zip(batch['input_ids'], batch['labels']):
        pairs = dict(zip(ids[0:12:2].tolist(), ids[1:12:2].tolist()))
        assert len(pairs) == 6  # distinct keys
        positions = (labels != IGNORE).nonzero().squeeze(-1)
        assert len(positions) == 6
        for position in positions:
            assert labels[position] == pairs[ids[position - 1].item()]


def test_mqar_never_shows_an_answer_in_the_input():
    """Fed-back answers would let later queries be answered by elimination instead of recall."""
    batch = make_batch('mqar', 20, seed=0, seq_len=64, num_kv_pairs=6, vocab_size=64)
    answered = batch['labels'] != IGNORE
    assert (batch['input_ids'][answered] == 1).all()  # the blank token
    query_region = batch['input_ids'][:, 12:24]
    assert not torch.isin(query_region, batch['input_ids'][:, 1:12:2]).any()


def test_selective_copy_answers_are_the_data_tokens_in_order():
    batch = make_batch('selective_copy', 10, seed=0, seq_len=40, num_tokens=5)
    for ids, labels in zip(batch['input_ids'], batch['labels']):
        context = ids[:34]
        data = context[context > 2]
        assert ids[34] == 2  # separator
        assert torch.equal(labels[35:], data)
        assert (labels[:35] == IGNORE).all()


def test_induction_answer_follows_the_earlier_trigger():
    batch = make_batch('induction', 20, seed=0, seq_len=32, vocab_size=30)
    for ids, labels in zip(batch['input_ids'], batch['labels']):
        triggers = (ids == 1).nonzero().squeeze(-1)
        assert len(triggers) == 2 and triggers[1] == 30
        assert labels[-1] == ids[triggers[0] + 1]
        assert (labels[:-1] == IGNORE).all()


def test_passkey_answer_repeats_the_hidden_digits():
    batch = make_batch('passkey', 20, seed=0, seq_len=50, vocab_size=40, passkey_len=4)
    for ids, labels in zip(batch['input_ids'], batch['labels']):
        start = (ids == 1).nonzero().item()
        assert torch.equal(labels[-4:], ids[start + 1:start + 5])
        assert ids[-5] == 2 and ((labels[-4:] >= 3) & (labels[-4:] < 13)).all()


@pytest.mark.parametrize("task", TASKS)
def test_batches_are_reproducible_from_the_seed(task):
    first, second, other = (make_batch(task, 4, seed=s) for s in (5, 5, 6))
    assert torch.equal(first['input_ids'], second['input_ids'])
    assert not torch.equal(first['input_ids'], other['input_ids'])


def test_accuracy_counts_only_labeled_positions():
    labels = torch.tensor([[IGNORE, IGNORE, 3, 4], [IGNORE, IGNORE, 5, 6]])
    logits = torch.zeros(2, 4, 8)
    # Position t predicts labels[t + 1]: row 0 gets both right, row 1 only the first
    logits[0, 1, 3] = logits[0, 2, 4] = 1
    logits[1, 1, 5] = logits[1, 2, 0] = 1
    accuracy = answer_accuracy(logits, labels)
    assert accuracy == {'token_accuracy': 0.75, 'example_accuracy': 0.5}


def test_run_diagnostic_trains_and_reports():
    torch.manual_seed(0)
    config = MixedLayerConfig(vocab_size=64, hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                              layer_types=["conv", "full_attention"], intermediate_size=64)
    model = MixedLayerForCausalLM(config)
    results = run_diagnostic(model, 'induction', dict(seq_len=16, vocab_size=64), steps=6, batch_size=8,
                             warmup_steps=2, eval_examples=16, eval_every=3, device='cpu', log=lambda _: None)
    assert results['steps'] == 6 and len(results['history']) == 1
    assert set(results['eval']) == {'loss', 'token_accuracy', 'example_accuracy'}
    assert results['parameters'] == model.num_parameters()
    assert results['best_example_accuracy'] >= results['eval']['example_accuracy']


def test_language_modeling_loop_reports_validation_perplexity():
    torch.manual_seed(0)
    config = MixedLayerConfig(vocab_size=20, hidden_size=32, num_attention_heads=4, num_hidden_layers=1,
                              layer_types=["conv"], intermediate_size=64)
    model = MixedLayerForCausalLM(config)
    blocks = torch.arange(20).repeat(12, 2)[:, :16]  # a periodic sequence the model can learn
    results = run_language_modeling(model, blocks[:10], blocks[10:], steps=40, batch_size=4, warmup_steps=5,
                                    learning_rate=3e-3, device='cpu', log=lambda _: None)
    assert results['eval']['perplexity'] == pytest.approx(math.exp(results['eval']['loss']))
    assert results['eval']['perplexity'] < 20  # better than uniform over the vocabulary


def test_diagnostic_model_config_layers_settings_over_a_preset():
    config = diagnostic_model_config(['conv', 'based'], 'guac-43m', hidden_size=64, num_heads=2,
                                     param_match='conv', settings=parse_assignments(['based_feature_dim=8']))
    assert config['hidden_size'] == 64 and 'intermediate_size' not in config
    assert config['num_hidden_layers'] == 2 and config['num_kv_heads'] == 2
    assert config['param_match'] == 'conv' and config['based_feature_dim'] == 8


def test_parse_assignments_reads_yaml_values():
    assert parse_assignments(['a=3', 'b=[conv, based]', 'c=true', 'd=5e5']) == {'a': 3, 'b': ['conv', 'based'],
                                                                             'c': True, 'd': 500000.0}
    with pytest.raises(ValueError):
        parse_assignments(['no_equals'])
