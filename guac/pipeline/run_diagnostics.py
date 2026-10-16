#!/usr/bin/env python3
"""
Train a small model from scratch on a synthetic diagnostic task and report
held-out accuracy (see guac/core/diagnostics.py for the tasks).

Usage:
    # Associative recall with a BASED hybrid
    python guac/pipeline/run_diagnostics.py --task mqar \\
        --layer-types conv,based,sliding_window_attention,based --hidden-size 128 --steps 3000 \\
        --task-arg num_kv_pairs=16 --task-arg seq_len=128

    # Architecture from a preset, at its own width
    python guac/pipeline/run_diagnostics.py --task induction --model-config guac-43m

    # Any model setting
    python guac/pipeline/run_diagnostics.py --task passkey --layer-types mamba2,mamba2 \\
        --set mamba_d_state=32 --param-match full_attention --output scratch/diagnostics/passkey.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import yaml

from guac.core.custom_architectures import MixedLayerForCausalLM
from guac.core.diagnostics import DEFAULT_VOCAB, TASKS, run_diagnostic
from guac.core.pipeline_utils import as_number, load_model_config, write_run_info
from guac.pipeline.pretrain import build_model_config


def parse_assignments(items: Optional[List[str]]) -> Dict[str, Any]:
    """['a=1', 'b=[conv, conv]', 'c=3e-3'] -> {'a': 1, 'b': ['conv', 'conv'], 'c': 0.003} (values parsed as YAML)."""
    result = {}
    for item in items or []:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f"Expected key=value, got {item!r}")
        result[key.strip()] = as_number(yaml.safe_load(value))
    return result


def diagnostic_model_config(layer_types: Optional[List[str]] = None, model_config: Optional[str] = None,
                            hidden_size: Optional[int] = None, num_heads: Optional[int] = None,
                            param_match: Any = None, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Model settings for a diagnostic run, as a model_configs.yaml-style dict:
    a preset's architecture keys (if given), then explicit sizes, layer types
    and other settings on top. Without a preset, defaults to hidden size 128,
    4 heads and an FFN of 4x the hidden size.
    """
    config = dict(load_model_config(model_config)) if model_config else {}
    config.setdefault('hidden_size', 128)
    config.setdefault('num_attention_heads', 4)
    if hidden_size:
        config['hidden_size'] = hidden_size
        config.pop('intermediate_size', None)
    if num_heads:
        config['num_attention_heads'] = num_heads
        config['num_kv_heads'] = num_heads
    if layer_types:
        config['layer_types'] = list(layer_types)
    if 'layer_types' not in config:
        raise ValueError("Give --layer-types or a --model-config with layer_types")
    config['num_hidden_layers'] = len(config['layer_types'])
    config.setdefault('num_kv_heads', config['num_attention_heads'])
    if param_match:
        config['param_match'] = param_match
    config.update(settings or {})
    return config


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a model on a synthetic diagnostic task")
    parser.add_argument('--task', required=True, choices=list(TASKS))
    parser.add_argument('--layer-types', type=lambda s: s.split(','), help='Comma-separated layer types')
    parser.add_argument('--model-config', help='Take the architecture from this model_configs.yaml preset')
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--heads', type=int, help='Attention heads (and KV heads)')
    parser.add_argument('--param-match', help="Size each block's FFN to match this layer type's block")
    parser.add_argument('--set', action='append', metavar='KEY=VALUE',
                        help='Model setting, e.g. --set mamba_d_state=32 (repeatable)')
    parser.add_argument('--task-arg', action='append', metavar='KEY=VALUE',
                        help='Task setting, e.g. --task-arg num_kv_pairs=16 --task-arg seq_len=256 (repeatable)')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--eval-examples', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', help='Write the results (plus run info) as JSON here')
    args = parser.parse_args()

    task_kwargs = parse_assignments(args.task_arg)
    task_kwargs.setdefault('vocab_size', DEFAULT_VOCAB[args.task])
    config = diagnostic_model_config(args.layer_types, args.model_config, args.hidden_size, args.heads,
                                     args.param_match, parse_assignments(args.set))

    torch.manual_seed(args.seed)
    model = MixedLayerForCausalLM(build_model_config(config, task_kwargs['vocab_size']))
    print(f"{args.task}: {', '.join(config['layer_types'])} (hidden {config['hidden_size']}, "
          f"{model.num_parameters():,} parameters), task {task_kwargs}")

    results = run_diagnostic(model, args.task, task_kwargs, steps=args.steps, batch_size=args.batch_size,
                             learning_rate=args.learning_rate, eval_examples=args.eval_examples,
                             eval_every=args.eval_every, seed=args.seed)
    final = results['eval']
    print(f"{args.task}: example accuracy {final['example_accuracy']:.1%}, token accuracy "
          f"{final['token_accuracy']:.1%}, answer loss {final['loss']:.4f} ({results['seconds']:.0f}s)")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({**results, 'model_settings': config}, f, indent=2)
        write_run_info(os.path.dirname(os.path.abspath(args.output)), 'diagnostics', config, args, model,
                       seed=args.seed, task=args.task, task_kwargs=task_kwargs)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
