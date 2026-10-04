#!/usr/bin/env python3
"""
Guac Arithmetic Evaluation - exact-match accuracy on fresh arithmetic problems

Generates problems with the same distributions as the mid-training arithmetic
task (see create_arithmetic_dataset in midtrain.py), prompts the model with
"<expr> = ", decodes greedily and compares the first token-free word of the
completion to the true answer.

Usage:
    python eval_arithmetic.py --model scratch/models/midtraining/guac-106m-arith_arithmetic
    python eval_arithmetic.py --model <dir> --num-per-category 200 --show 10
"""

import os, sys, argparse, random, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import torch

from scripts.training.core.pipeline_utils import load_base_model


def make_problem(category: str, rng: random.Random) -> tuple:
    """Return (prompt, answer) for one problem; mirrors midtrain.py's generator."""
    if category == 'add/sub':
        a, b = rng.randint(1, 1000), rng.randint(1, 1000)
        op = rng.choice(['+', '-'])
        return f"{a} {op} {b} = ", str(a + b if op == '+' else a - b)
    if category == 'mul':
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        return f"{a} * {b} = ", str(a * b)
    if category == 'div':
        b, result = rng.randint(1, 50), rng.randint(1, 100)
        return f"{b * result} / {b} = ", str(result)
    if category == 'paren':
        a, b, c = rng.randint(1, 50), rng.randint(1, 50), rng.randint(1, 50)
        return f"({a} + {b}) * {c} = ", str((a + b) * c)
    raise ValueError(f"Unknown category: {category}")


CATEGORIES = ['add/sub', 'mul', 'div', 'paren']


@torch.no_grad()
def complete(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False,
                       return_token_type_ids=False)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    words = text.strip().split()
    return words[0] if words else ''


def main():
    parser = argparse.ArgumentParser(description="Guac Arithmetic Evaluation")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model directory (with tokenizer)')
    parser.add_argument('--num-per-category', type=int, default=50,
                       help='Problems per category')
    parser.add_argument('--categories', type=str, nargs='+', default=CATEGORIES,
                       choices=CATEGORIES, help='Categories to evaluate')
    parser.add_argument('--max-new-tokens', type=int, default=8,
                       help='Generation budget per answer')
    parser.add_argument('--show', type=int, default=4,
                       help='Example completions to print per category')
    parser.add_argument('--seed', type=int, default=123,
                       help='Problem-generation seed (differs from training seed 42)')
    args = parser.parse_args()

    model, tokenizer = load_base_model(args.model)
    model.eval()
    rng = random.Random(args.seed)

    start = time.time()
    total_correct = total = 0
    print(f"\n{'category':10s} {'correct':>12s}")
    for category in args.categories:
        correct, examples = 0, []
        for _ in range(args.num_per_category):
            prompt, answer = make_problem(category, rng)
            prediction = complete(model, tokenizer, prompt, args.max_new_tokens)
            correct += prediction == answer
            if len(examples) < args.show:
                mark = '✓' if prediction == answer else '✗'
                examples.append(f"    {mark} {prompt}{prediction or '<empty>'}   (true {answer})")
        total_correct += correct
        total += args.num_per_category
        print(f"{category:10s} {correct:>5d}/{args.num_per_category:<5d} {correct / args.num_per_category:6.1%}")
        for line in examples:
            print(line)

    print(f"{'overall':10s} {total_correct:>5d}/{total:<5d} {total_correct / total:6.1%}"
          f"   ({time.time() - start:.1f}s)")


if __name__ == '__main__':
    main()
