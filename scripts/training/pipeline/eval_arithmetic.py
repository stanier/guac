#!/usr/bin/env python3
"""
Guac Arithmetic Evaluation - exact-match accuracy on held-out arithmetic problems

Draws problems only from the held-out pool that mid-training never trains on
(see core/arithmetic.py), prompts the model with "<expr> = ", decodes greedily
and compares the first word of the completion to the true answer. The answer
format (normal or reversed digits) is read from the model directory's
arithmetic_format.json, written by midtrain.py.

Usage:
    python eval_arithmetic.py --model scratch/models/midtraining/guac-13m-arith_arithmetic
    python eval_arithmetic.py --model <dir> --num-per-category 200 --show 10
"""

import os, sys, argparse, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import torch

from scripts.training.core.arithmetic import CATEGORIES, SPLIT_VERSION, evaluate_accuracy, load_format
from scripts.training.core.pipeline_utils import load_base_model


def main():
    parser = argparse.ArgumentParser(description="Guac Arithmetic Evaluation")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model directory (with tokenizer)')
    parser.add_argument('--num-per-category', type=int, default=50,
                       help='Problems per category')
    parser.add_argument('--categories', type=str, nargs='+', default=CATEGORIES,
                       choices=CATEGORIES, help='Categories to evaluate')
    parser.add_argument('--reverse-answers', action=argparse.BooleanOptionalAction, default=None,
                       help='Override the answer format saved with the model')
    parser.add_argument('--max-new-tokens', type=int, default=8,
                       help='Generation budget per answer')
    parser.add_argument('--show', type=int, default=4,
                       help='Example completions to print per category')
    parser.add_argument('--seed', type=int, default=123,
                       help='Problem-sampling seed')
    args = parser.parse_args()

    saved_format = load_format(args.model)
    reverse = args.reverse_answers
    if reverse is None:
        reverse = saved_format['reverse_answers']
    if saved_format.get('split_version', 1) != SPLIT_VERSION and 'holdout_fraction' in saved_format:
        print(f"⚠️  This model was trained with train/held-out split version {saved_format.get('split_version', 1)} "
              f"(current: {SPLIT_VERSION}). Some held-out problems' commuted twins (b + a for a + b) were in its "
              f"training data, so accuracy may be inflated.")

    model, tokenizer = load_base_model(args.model)
    if torch.cuda.is_available():
        model = model.cuda()

    print(f"Answer format: {'reversed digits' if reverse else 'normal'} | held-out problems only")
    start = time.time()
    result = evaluate_accuracy(model, tokenizer, args.num_per_category, reverse=reverse, seed=args.seed,
                               categories=args.categories, num_examples=args.show,
                               max_new_tokens=args.max_new_tokens)

    n = args.num_per_category
    print(f"\n{'category':10s} {'correct':>12s}")
    for category in args.categories:
        accuracy = result['per_category'][category]
        print(f"{category:10s} {round(accuracy * n):>5d}/{n:<5d} {accuracy:6.1%}")
        for line in result['examples'][category]:
            print(f"    {line}")
    total = n * len(args.categories)
    print(f"{'overall':10s} {round(result['overall'] * total):>5d}/{total:<5d} {result['overall']:6.1%}"
          f"   ({time.time() - start:.1f}s)")


if __name__ == '__main__':
    main()
