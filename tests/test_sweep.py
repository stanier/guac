"""
Tests for the sweep harness: specs are validated, budgets reach the model
config, results land in one table, and an interrupted sweep resumes.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import yaml

from guac.pipeline import sweep
from guac.core.custom_architectures import block_parameter_counts
from guac.pipeline.pretrain import build_model_config

SPEC = {
    'name': 'tiny',
    'base': {'hidden_size': 32, 'num_attention_heads': 4, 'intermediate_size': 96},
    'budget': {'param_match': 'full_attention'},
    'architectures': {'attention': ['full_attention', 'full_attention'], 'conv': ['conv', 'conv']},
    'tasks': [{'task': 'induction', 'args': {'seq_len': 16, 'vocab_size': 16}, 'steps': 3, 'eval_examples': 8}],
    'seeds': [0, 1],
    'train': {'batch_size': 4, 'warmup_steps': 1, 'eval_every': 0},
}


def write_spec(tmp_path, spec=SPEC):
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec))
    return str(path)


def run_sweep(monkeypatch, spec_path, output_dir, *extra):
    monkeypatch.setattr(sys, 'argv', ['sweep.py', spec_path, '--output-dir', str(output_dir), *extra])
    sweep.main()


def test_sweep_writes_one_row_per_run_and_a_table(tmp_path, monkeypatch):
    run_sweep(monkeypatch, write_spec(tmp_path), tmp_path / "out")

    rows = [json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    assert sorted((r['architecture'], r['seed']) for r in rows) == [('attention', 0), ('attention', 1),
                                                                     ('conv', 0), ('conv', 1)]
    table = (tmp_path / "out" / "results.md").read_text()
    assert "| attention | full_attention, full_attention |" in table and "±" in table
    assert (tmp_path / "out" / "results.csv").exists()
    assert (tmp_path / "out" / "run_info.json").exists()


def test_sweep_resumes_without_rerunning_finished_runs(tmp_path, monkeypatch):
    spec_path = write_spec(tmp_path)
    run_sweep(monkeypatch, spec_path, tmp_path / "out")
    before = (tmp_path / "out" / "results.jsonl").read_text()

    def fail(*args, **kwargs):
        raise AssertionError("finished runs must not rerun")

    monkeypatch.setattr(sweep, 'run_one', fail)
    run_sweep(monkeypatch, spec_path, tmp_path / "out")
    assert (tmp_path / "out" / "results.jsonl").read_text() == before


def test_param_budget_reaches_every_architecture():
    counts = [block_parameter_counts(build_model_config(sweep.architecture_settings(SPEC, name), 16))
              for name in SPEC['architectures']]
    flat = [c for arch in counts for c in arch]
    assert max(flat) - min(flat) <= 3 * 32


def test_architecture_entries_can_carry_their_own_settings():
    spec = {**SPEC, 'architectures': {'m': {'layer_types': ['mamba2'], 'mamba_d_state': 8}}}
    settings = sweep.architecture_settings(spec, 'm')
    assert settings['mamba_d_state'] == 8 and settings['num_hidden_layers'] == 1


@pytest.mark.parametrize("bad, message", [
    ({'budget': {'param_match': 'conv', 'flop_match': 'conv'}}, "param_match or flop_match"),
    ({'tasks': [{'task': 'sorting'}]}, "Unknown sweep task"),
])
def test_invalid_specs_are_rejected(tmp_path, bad, message):
    with pytest.raises(ValueError, match=message):
        sweep.load_spec(write_spec(tmp_path, {**SPEC, **bad}))


def test_example_spec_is_valid():
    spec = sweep.load_spec(str(Path(__file__).parent.parent / "sweeps" / "recall_small.yaml"))
    for name in spec['architectures']:
        build_model_config(sweep.architecture_settings(spec, name), 128)


def test_scientific_notation_in_specs_is_read_as_numbers(tmp_path):
    """PyYAML reads 3e-3 as a string; the sweep must still get a float."""
    path = tmp_path / "spec.yaml"
    path.write_text("name: s\narchitectures: {a: [conv]}\ntasks: [{task: induction, learning_rate: 3e-3}]\n"
                    "train: {weight_decay: 1e-2, batch_size: 8}\n")
    spec = sweep.load_spec(str(path))
    assert spec['tasks'][0]['learning_rate'] == 0.003
    assert spec['train'] == {'weight_decay': 0.01, 'batch_size': 8}
