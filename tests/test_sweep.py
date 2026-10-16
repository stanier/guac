"""
Tests for the sweep harness: specs are validated, budgets reach the model
config, results land in one table, an interrupted sweep resumes, and rows
whose settings no longer match the spec are rerun rather than reused.
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


def read_rows(output_dir):
    return [json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines()]


def count_runs(monkeypatch):
    """Patch run_one to count the runs it is asked for, still running them."""
    calls = []
    real_run_one = sweep.run_one

    def counting(spec, arch_name, task, seed, lm_data):
        calls.append((arch_name, sweep.task_name(task), seed))
        return real_run_one(spec, arch_name, task, seed, lm_data)

    monkeypatch.setattr(sweep, 'run_one', counting)
    return calls


def test_editing_the_spec_reruns_what_changed_and_ignores_stale_rows(tmp_path, monkeypatch, capsys):
    """Rows were keyed by (architecture, task name, seed), so a rerun after changing steps reused stale results."""
    spec_path = write_spec(tmp_path)
    run_sweep(monkeypatch, spec_path, tmp_path / "out")
    edited = {**SPEC, 'tasks': [{**SPEC['tasks'][0], 'steps': 4}]}
    write_spec(tmp_path, edited)
    calls = count_runs(monkeypatch)
    capsys.readouterr()
    run_sweep(monkeypatch, spec_path, tmp_path / "out")
    assert len(calls) == 4
    assert "ignoring 4 of 4 results rows" in capsys.readouterr().out

    rows = read_rows(tmp_path / "out")
    assert [r['steps'] for r in rows] == [3] * 4 + [4] * 4
    current, stale = sweep.current_rows(edited, rows)
    assert stale == 4 and sorted(r['steps'] for r in current) == [4] * 4

    # The table holds the new runs only, and --summarize reports the stale ones
    run_sweep(monkeypatch, spec_path, tmp_path / "out", '--summarize')
    assert "ignoring 4 of 8 results rows" in capsys.readouterr().out
    csv_rows = (tmp_path / "out" / "results.csv").read_text().splitlines()
    new_mean = sum(r['example_accuracy'] for r in current if r['architecture'] == 'attention') / 2
    assert float(csv_rows[1].split(',')[-1]) == pytest.approx(new_mean)


def test_adding_a_seed_runs_only_the_new_runs(tmp_path, monkeypatch):
    spec_path = write_spec(tmp_path)
    run_sweep(monkeypatch, spec_path, tmp_path / "out")
    write_spec(tmp_path, {**SPEC, 'seeds': [0, 1, 2]})
    calls = count_runs(monkeypatch)
    run_sweep(monkeypatch, spec_path, tmp_path / "out")
    task = sweep.task_name(SPEC['tasks'][0])
    assert calls == [('attention', task, 2), ('conv', task, 2)]


def test_rows_without_a_config_hash_are_stale(tmp_path, monkeypatch, capsys):
    """Rows from before config hashes can't show which settings they ran with, so they are rerun."""
    spec_path = write_spec(tmp_path)
    run_sweep(monkeypatch, spec_path, tmp_path / "out")
    rows = read_rows(tmp_path / "out")
    assert all(len(r['config_hash']) == 16 for r in rows)
    (tmp_path / "out" / "results.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in r.items() if k != 'config_hash'}) + "\n" for r in rows))
    calls = count_runs(monkeypatch)
    capsys.readouterr()
    run_sweep(monkeypatch, spec_path, tmp_path / "out")
    assert len(calls) == 4 and "ignoring 4 of 4 results rows" in capsys.readouterr().out


def test_config_hash_covers_model_training_task_and_seed():
    task = SPEC['tasks'][0]
    reference = sweep.run_config_hash(SPEC, 'attention', task, 0)
    assert sweep.run_config_hash(SPEC, 'attention', dict(task), 0) == reference
    changed = [
        sweep.run_config_hash({**SPEC, 'base': {**SPEC['base'], 'intermediate_size': 128}}, 'attention', task, 0),
        sweep.run_config_hash({**SPEC, 'budget': {}}, 'attention', task, 0),
        sweep.run_config_hash({**SPEC, 'train': {**SPEC['train'], 'learning_rate': 3e-3}}, 'attention', task, 0),
        sweep.run_config_hash(SPEC, 'attention', {**task, 'steps': 4}, 0),
        sweep.run_config_hash(SPEC, 'attention', {**task, 'eval_examples': 16}, 0),
        sweep.run_config_hash(SPEC, 'attention', {**task, 'args': {**task['args'], 'seq_len': 32}}, 0),
        sweep.run_config_hash(SPEC, 'conv', task, 0),
        sweep.run_config_hash(SPEC, 'attention', task, 1),
    ]
    assert len({reference, *changed}) == len(changed) + 1


def test_lm_tasks_are_named_by_their_seq_len():
    assert sweep.task_name({'task': 'lm', 'seq_len': 128}) == 'lm seq_len=128'
    assert sweep.task_name({'task': 'lm'}) == 'lm seq_len=256'
    assert sweep.task_name({'task': 'lm', 'seq_len': 128, 'name': 'ppl'}) == 'ppl'


@pytest.mark.parametrize("tasks", [
    [{'task': 'lm', 'seq_len': 128, 'datasets': ['a']}, {'task': 'lm', 'seq_len': 128, 'datasets': ['b']}],
    [{'task': 'induction', 'steps': 10}, {'task': 'induction', 'steps': 20}],
    [{'task': 'induction', 'name': 'x'}, {'task': 'mqar', 'name': 'x'}],
])
def test_tasks_sharing_a_name_are_rejected(tmp_path, tasks):
    """Two lm tasks at seq_len 128 and 512 were both 'lm', and the second trained on the first's data."""
    with pytest.raises(ValueError, match="distinct 'name'"):
        sweep.load_spec(write_spec(tmp_path, {**SPEC, 'tasks': tasks}))
    named = [{**task, 'name': f"task {i}"} for i, task in enumerate(tasks)]
    sweep.load_spec(write_spec(tmp_path, {**SPEC, 'tasks': named}))


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
