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


@pytest.mark.parametrize("spec_file", sorted(p.name for p in (Path(__file__).parent.parent / "sweeps").glob("*.yaml")))
def test_example_specs_are_valid(spec_file):
    spec = sweep.load_spec(str(Path(__file__).parent.parent / "sweeps" / spec_file))
    for task in spec['tasks']:
        if task['task'] != 'lm':
            sweep.TASKS[task['task']](2, **(task.get('args') or {}))
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


def test_shards_split_the_runs_and_share_one_table(tmp_path, monkeypatch):
    spec_path = write_spec(tmp_path)
    run_sweep(monkeypatch, spec_path, tmp_path / "out", '--shard', '0/3')
    run_sweep(monkeypatch, spec_path, tmp_path / "out", '--shard', '2/3')
    shard_rows = {name: [json.loads(line) for line in (tmp_path / "out" / name).read_text().splitlines()]
                  for name in ('results.shard0-of-3.jsonl', 'results.shard2-of-3.jsonl')}
    # Runs in order (attention, 0), (attention, 1), (conv, 0), (conv, 1): shard 0 gets the 1st and 4th
    assert [(r['architecture'], r['seed']) for r in shard_rows['results.shard0-of-3.jsonl']] == \
        [('attention', 0), ('conv', 1)]
    assert [(r['architecture'], r['seed']) for r in shard_rows['results.shard2-of-3.jsonl']] == [('conv', 0)]
    table = (tmp_path / "out" / "results.md").read_text()
    assert "| attention |" in table and "| conv |" in table

    run_sweep(monkeypatch, spec_path, tmp_path / "out", '--shard', '1/3')
    (tmp_path / "out" / "results.md").unlink()
    run_sweep(monkeypatch, spec_path, tmp_path / "out", '--summarize')
    rows = [line for line in (tmp_path / "out" / "results.md").read_text().splitlines() if line.startswith("| ")]
    assert len(rows) == 3 and all("±" in row for row in rows[1:])


def test_force_with_shard_is_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_sweep(monkeypatch, write_spec(tmp_path), tmp_path / "out", '--shard', '0/2', '--force')


def test_budget_override_replaces_the_spec_budget():
    class Args:
        param_match, flop_match = None, 'conv'
    spec = sweep.apply_budget_override(SPEC, Args)
    assert spec['budget'] == {'flop_match': 'conv', 'seq_len': 16} and spec['name'] == 'tiny-flop'
    assert sweep.architecture_settings(spec, 'attention')['flop_match'] == 'conv'
