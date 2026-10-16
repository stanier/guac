#!/usr/bin/env python3
"""
Sweep a grid of layer-type patterns over tasks at a matched parameter or FLOP
budget, and collect one results table.

A sweep spec is a YAML file (see sweeps/README.md and the examples there):

    name: recall-small
    base:                       # model settings shared by every architecture
      hidden_size: 64
      num_attention_heads: 4
      intermediate_size: 256
    budget:                     # optional: size every block like a reference block
      param_match: full_attention     # or flop_match: full_attention (plus seq_len)
    architectures:
      attention: [full_attention, full_attention]
      based_hybrid: [based, sliding_window_attention]
      mamba2: {layer_types: [mamba2, mamba2], mamba_d_state: 16}
    tasks:
      - task: mqar
        args: {num_kv_pairs: 8, seq_len: 64, vocab_size: 128}
        steps: 2000
      - task: lm                 # language modeling on a dataset, reports validation perplexity
        tokenizer: scratch/tokenizers/character/ascii
        datasets: [scratch/smoke/tiny_text]
        seq_len: 128
        steps: 500
    seeds: [0, 1]
    train: {batch_size: 64, learning_rate: 1e-3}

Usage:
    python guac/pipeline/sweep.py sweeps/recall_small.yaml
    python guac/pipeline/sweep.py sweeps/recall_small.yaml --dry-run    # sizes only, no training

Results go to scratch/sweeps/<name>/ (or --output-dir): results.jsonl gets one
line per finished run, so an interrupted sweep resumes where it stopped (pass
--force to rerun everything), and results.md / results.csv hold the table.

To spread a sweep over several jobs (jobscripts/sweep.sbatch), give each one
--shard I/N: it runs every Nth run starting at I and writes its own
results.shardI-of-N.jsonl; every shard's table includes all the shards'
finished runs, and --summarize rebuilds the table from them. --param-match or
--flop-match replaces the spec's budget, so one spec covers both.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import yaml

from guac.core.custom_architectures import MixedLayerForCausalLM, model_flops_per_token
from guac.core.diagnostics import DEFAULT_VOCAB, TASKS, run_diagnostic, run_language_modeling
from guac.core.pipeline_utils import numbers_in, write_run_info
from guac.pipeline.pretrain import build_model_config

TRAIN_KEYS = ('batch_size', 'learning_rate', 'weight_decay', 'warmup_steps', 'eval_every', 'bf16')


def load_spec(path: str) -> Dict[str, Any]:
    with open(path) as f:
        spec = numbers_in(yaml.safe_load(f))
    for key in ('name', 'architectures', 'tasks'):
        if key not in spec:
            raise ValueError(f"Sweep spec {path} needs a '{key}' entry")
    budget = spec.get('budget') or {}
    unknown = set(budget) - {'param_match', 'flop_match', 'seq_len'}
    if unknown or ('param_match' in budget and 'flop_match' in budget):
        raise ValueError(f"budget takes param_match or flop_match (plus seq_len), got {sorted(budget)}")
    for task in spec['tasks']:
        if task.get('task') not in (*TASKS, 'lm'):
            raise ValueError(f"Unknown sweep task {task.get('task')!r}; choose from {[*TASKS, 'lm']}")
    return spec


def task_name(task: Dict[str, Any]) -> str:
    """A task's column name: its 'name', or the task plus its arguments."""
    if task.get('name'):
        return task['name']
    args = task.get('args') or {}
    return task['task'] + (''.join(f" {k}={v}" for k, v in sorted(args.items())) if args else '')


def task_seq_len(task: Dict[str, Any]) -> int:
    if task['task'] == 'lm':
        return task.get('seq_len', 256)
    signature_default = {'passkey': 256}.get(task['task'], 128)
    return (task.get('args') or {}).get('seq_len', signature_default)


def architecture_settings(spec: Dict[str, Any], name: str) -> Dict[str, Any]:
    """The model_configs.yaml-style settings for one architecture: base, budget, then its own settings."""
    entry = spec['architectures'][name]
    entry = {'layer_types': entry} if isinstance(entry, list) else dict(entry)
    config = {**(spec.get('base') or {}), **entry}
    config.setdefault('hidden_size', 128)
    config.setdefault('num_attention_heads', 4)
    config.setdefault('num_kv_heads', config['num_attention_heads'])
    config['num_hidden_layers'] = len(config['layer_types'])
    budget = spec.get('budget') or {}
    if 'param_match' in budget:
        config['param_match'] = budget['param_match']
    if 'flop_match' in budget:
        config['flop_match'] = budget['flop_match']
        config['max_sequence_length'] = budget.get('seq_len', 1024)
    return config


def load_lm_data(task: Dict[str, Any]) -> Tuple[Any, torch.Tensor, torch.Tensor]:
    """(tokenizer, train blocks, validation blocks) for an 'lm' task, packed to its seq_len."""
    from transformers import AutoTokenizer
    from guac.pipeline.pretrain import prepare_pretraining_data

    tokenizer = AutoTokenizer.from_pretrained(task['tokenizer'])
    train, validation = prepare_pretraining_data(task['datasets'], tokenizer, task_seq_len(task), pack=True,
                                                 validation_docs=task.get('validation_docs', 200))
    if validation is None or not hasattr(train, '__len__'):
        raise ValueError("lm sweep tasks need a map-style dataset and validation_docs > 0")
    return tokenizer, torch.tensor(train['input_ids']), torch.tensor(validation['input_ids'])


def run_one(spec, arch_name: str, task: Dict[str, Any], seed: int, lm_data) -> Dict[str, Any]:
    settings = architecture_settings(spec, arch_name)
    seq_len = task_seq_len(task)
    if task['task'] == 'lm':
        tokenizer, train_blocks, validation_blocks = lm_data
        vocab_size = len(tokenizer)
    else:
        task_args = dict(task.get('args') or {})
        vocab_size = task_args.setdefault('vocab_size', DEFAULT_VOCAB[task['task']])

    torch.manual_seed(seed)
    model_config = build_model_config(settings, vocab_size)
    model = MixedLayerForCausalLM(model_config)
    train = {**(spec.get('train') or {}), **{k: task[k] for k in TRAIN_KEYS if k in task}}
    steps = task.get('steps', spec.get('steps', 1000))
    print(f"\n=== {arch_name} | {task_name(task)} | seed {seed} | {model.num_parameters():,} parameters ===")

    if task['task'] == 'lm':
        result = run_language_modeling(model, train_blocks, validation_blocks, steps=steps, seed=seed, **train)
        metric = {'perplexity': result['eval']['perplexity'], 'loss': result['eval']['loss']}
    else:
        result = run_diagnostic(model, task['task'], task_args, steps=steps, seed=seed,
                                eval_examples=task.get('eval_examples', 1000), **train)
        metric = {'example_accuracy': result['eval']['example_accuracy'],
                  'token_accuracy': result['eval']['token_accuracy'],
                  'best_example_accuracy': result['best_example_accuracy']}

    return {
        'architecture': arch_name, 'layer_types': model_config.layer_types, 'task': task_name(task), 'seed': seed,
        'parameters': model.num_parameters(),
        'non_embedding_parameters': model.num_parameters() - model_config.vocab_size * model_config.hidden_size,
        'flops_per_token': model_flops_per_token(model_config, seq_len), 'seq_len': seq_len,
        'ffn_hidden_sizes': model_config.ffn_hidden_sizes, **metric,
        'steps': steps, 'seconds': result['seconds'], 'train_loss': result['train_loss'],
    }


def summarize(spec, rows: List[Dict[str, Any]], output_dir: str) -> str:
    """Write results.md and results.csv: one row per architecture, one column per task (mean over seeds)."""
    tasks = [task_name(t) for t in spec['tasks']]
    lines = [f"# Sweep: {spec['name']}", ""]
    budget = spec.get('budget') or {}
    if budget:
        lines += [f"Budget: {', '.join(f'{k} {v}' for k, v in budget.items())}. "
                  "Accuracy is held-out example accuracy (every answer token right); lm is validation perplexity. "
                  "Mean ± standard deviation over seeds.", ""]
    header = ["Architecture", "Layer types", "Params (non-emb.)", "MFLOP/token"] + tasks
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    csv_rows = []
    for arch in spec['architectures']:
        arch_rows = [r for r in rows if r['architecture'] == arch]
        if not arch_rows:
            continue
        first = arch_rows[0]
        cells = [arch, ", ".join(first['layer_types']), f"{first['non_embedding_parameters']:,}",
                 f"{first['flops_per_token'] / 1e6:.2f}"]
        csv_row = {'architecture': arch, 'layer_types': " ".join(first['layer_types']),
                   'non_embedding_parameters': first['non_embedding_parameters'],
                   'flops_per_token': first['flops_per_token']}
        for task in tasks:
            values = [r.get('perplexity', r.get('example_accuracy')) for r in arch_rows if r['task'] == task]
            if not values:
                cells.append("")
                continue
            is_lm = 'perplexity' in next(r for r in arch_rows if r['task'] == task)
            mean = statistics.mean(values)
            spread = f" ± {statistics.stdev(values):.{2 if is_lm else 1}{'f' if is_lm else '%'}}" if len(values) > 1 else ""
            cells.append(f"{mean:.2f}{spread}" if is_lm else f"{mean:.1%}{spread}")
            csv_row[task] = mean
        lines.append("| " + " | ".join(cells) + " |")
        csv_rows.append(csv_row)

    table = "\n".join(lines) + "\n"
    # Written to a temporary file and renamed, as shards running at once rewrite these
    temporary = os.path.join(output_dir, f'.results.md.{os.getpid()}')
    with open(temporary, 'w') as f:
        f.write(table)
    os.replace(temporary, os.path.join(output_dir, 'results.md'))
    if csv_rows:
        fields = list(dict.fromkeys(key for row in csv_rows for key in row))
        temporary = os.path.join(output_dir, f'.results.csv.{os.getpid()}')
        with open(temporary, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        os.replace(temporary, os.path.join(output_dir, 'results.csv'))
    return table


def read_results(output_dir: str) -> List[Dict[str, Any]]:
    """Every finished run, from results.jsonl and any shards' files."""
    rows = []
    for path in sorted(glob.glob(os.path.join(output_dir, 'results*.jsonl'))):
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def parse_shard(text: str) -> Tuple[int, int]:
    try:
        index, count = (int(part) for part in text.split('/'))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--shard takes I/N, such as 0/4; got {text!r}")
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError(f"--shard {text}: need 0 <= I < N")
    return index, count


def apply_budget_override(spec: Dict[str, Any], args) -> Dict[str, Any]:
    """Replace the spec's budget with --param-match/--flop-match, and name the sweep after it."""
    if not (args.param_match or args.flop_match):
        return spec
    budget = {'param_match': args.param_match} if args.param_match else {'flop_match': args.flop_match}
    if args.flop_match:
        budget['seq_len'] = (spec.get('budget') or {}).get('seq_len') or task_seq_len(spec['tasks'][0])
    suffix = 'param' if args.param_match else 'flop'
    return {**spec, 'budget': budget, 'name': f"{spec['name']}-{suffix}"}


def dry_run(spec) -> None:
    """Print each architecture's size and cost without training."""
    seq_len = (spec.get('budget') or {}).get('seq_len') or task_seq_len(spec['tasks'][0])
    print(f"{'Architecture':<20} {'Params':>12} {'MFLOP/token':>12}  FFN widths   (vocab 256, seq_len {seq_len})")
    for name in spec['architectures']:
        config = build_model_config(architecture_settings(spec, name), 256)
        with torch.device("meta"):
            params = MixedLayerForCausalLM(config).num_parameters()
        flops = model_flops_per_token(config, seq_len)
        print(f"{name:<20} {params:>12,} {flops / 1e6:>12.2f}  {config.ffn_hidden_sizes}")


def main():
    parser = argparse.ArgumentParser(description="Run a layer-type sweep and tabulate the results")
    parser.add_argument('spec', help='Sweep spec YAML')
    parser.add_argument('--output-dir', help='Default: scratch/sweeps/<name>')
    parser.add_argument('--dry-run', action='store_true', help='Print sizes and FLOPs only')
    parser.add_argument('--force', action='store_true', help='Rerun runs already in the results files')
    parser.add_argument('--shard', type=parse_shard, metavar='I/N',
                        help='Run only every Nth run, starting at I (0-based), for parallel jobs')
    parser.add_argument('--summarize', action='store_true', help='Rebuild results.md/csv from finished runs only')
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--param-match', metavar='LAYER', help="Replace the spec's budget with param_match LAYER")
    budget.add_argument('--flop-match', metavar='LAYER', help="Replace the spec's budget with flop_match LAYER")
    args = parser.parse_args()
    if args.force and args.shard:
        parser.error("--force with --shard would let one shard delete the others' results; "
                     "remove the output directory instead")

    spec = apply_budget_override(load_spec(args.spec), args)
    if args.dry_run:
        dry_run(spec)
        return

    output_dir = args.output_dir or os.path.join(project_root, 'scratch', 'sweeps', spec['name'])
    os.makedirs(output_dir, exist_ok=True)
    if args.summarize:
        print(summarize(spec, read_results(output_dir), output_dir))
        return
    if args.force:
        for path in glob.glob(os.path.join(output_dir, 'results*.jsonl')):
            os.remove(path)
    index, count = args.shard or (0, 1)
    results_path = os.path.join(output_dir, f'results.shard{index}-of-{count}.jsonl' if args.shard
                                else 'results.jsonl')
    finished = {(r['architecture'], r['task'], r['seed']) for r in read_results(output_dir)}
    write_run_info(output_dir, 'sweep', spec, args, seed=None, spec_path=os.path.abspath(args.spec))

    runs = [(task, arch_name, seed) for task in spec['tasks'] for arch_name in spec['architectures']
            for seed in spec.get('seeds', [0])][index::count]
    lm_data = {}
    for task, arch_name, seed in runs:
        if (arch_name, task_name(task), seed) in finished:
            continue
        if task['task'] == 'lm' and task_name(task) not in lm_data:
            # Shards share the tokenization cache; the first to arrive fills it while the others wait
            from filelock import FileLock
            with FileLock(os.path.join(output_dir, '.lm-data.lock')):
                lm_data = {task_name(task): load_lm_data(task)}
        row = run_one(spec, arch_name, task, seed, lm_data.get(task_name(task)))
        with open(results_path, 'a') as f:
            f.write(json.dumps(row) + '\n')
        summarize(spec, read_results(output_dir), output_dir)

    print("\n" + summarize(spec, read_results(output_dir), output_dir))
    print(f"Results in {output_dir}")


if __name__ == "__main__":
    main()
