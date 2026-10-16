"""
Arithmetic task definitions shared by mid-training and evaluation.

Problems are split into train and held-out pools by a hash of the expression,
so evaluation never measures recall of problems seen during training.
Commuted twins (a + b and b + a) hash the same, so one can't be trained on
while the other is evaluated. This
matters because some categories have small problem spaces (div has 5,000
distinct problems, mul 10,000) that a 50k-sample training set mostly covers.

Answers can be written with their digits reversed (sign kept in front), e.g.
"54 + 275 = 923". Emitting the least-significant digit first lets the model
propagate carries in generation order, which helps small models a lot.
"""

import hashlib
import json
import os
import random
import re
from typing import Iterator, Tuple

CATEGORIES = ['add/sub', 'mul', 'div', 'paren']

# Fraction of each category's problem space reserved for evaluation
HOLDOUT_FRACTION = 0.1

FORMAT_FILENAME = 'arithmetic_format.json'

# Version of the train/held-out assignment, recorded with trained models. Version 1
# hashed expressions as written, so b + a could be trained on while a + b was evaluated
SPLIT_VERSION = 2


def sample_problem(category: str, rng: random.Random) -> Tuple[str, int]:
    """Return (expression, result) for one random problem of the given category."""
    if category == 'add/sub':
        a, b = rng.randint(1, 1000), rng.randint(1, 1000)
        op = rng.choice(['+', '-'])
        return f"{a} {op} {b}", a + b if op == '+' else a - b
    if category == 'mul':
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        return f"{a} * {b}", a * b
    if category == 'div':
        b, result = rng.randint(1, 50), rng.randint(1, 100)
        return f"{b * result} / {b}", result
    if category == 'paren':
        a, b, c = rng.randint(1, 50), rng.randint(1, 50), rng.randint(1, 50)
        return f"({a} + {b}) * {c}", (a + b) * c
    raise ValueError(f"Unknown category: {category}. Available: {CATEGORIES}")


def canonical_expression(expression: str) -> str:
    """The expression with commutative operands sorted: '7 + 3' -> '3 + 7', '(9 + 2) * 4' -> '(2 + 9) * 4'."""
    match = re.fullmatch(r"(\d+) ([+*]) (\d+)", expression)
    if match:
        a, op, b = match.groups()
        a, b = sorted((int(a), int(b)))
        return f"{a} {op} {b}"
    match = re.fullmatch(r"\((\d+) \+ (\d+)\) \* (\d+)", expression)
    if match:
        a, b, c = map(int, match.groups())
        a, b = sorted((a, b))
        return f"({a} + {b}) * {c}"
    return expression


def is_held_out(expression: str) -> bool:
    """Deterministically assign an expression (and its commuted twins) to the held-out (eval) pool."""
    digest = hashlib.md5(canonical_expression(expression).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64 < HOLDOUT_FRACTION


def format_answer(result: int, reverse: bool = False) -> str:
    """Render a result, optionally with digits reversed and the sign kept in front."""
    sign, digits = ('-', str(-result)) if result < 0 else ('', str(result))
    return sign + (digits[::-1] if reverse else digits)


def unreverse_answer(text: str) -> str:
    """Undo format_answer(..., reverse=True) on a string, keeping any leading zeros' meaning."""
    sign, digits = ('-', text[1:]) if text.startswith('-') else ('', text)
    return sign + digits[::-1]


def format_prompt(expression: str) -> str:
    return f"{expression} = "


def generate_problems(category: str, n: int, rng: random.Random,
                      held_out: bool) -> Iterator[Tuple[str, str, int]]:
    """Yield n (expression, prompt, result) tuples from the train or held-out pool."""
    produced = 0
    while produced < n:
        expression, result = sample_problem(category, rng)
        if is_held_out(expression) == held_out:
            produced += 1
            yield expression, format_prompt(expression), result


def save_format(output_dir: str, reverse: bool) -> None:
    """Record the answer format next to a trained model so evaluation can match it."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, FORMAT_FILENAME), 'w') as f:
        json.dump({'reverse_answers': reverse, 'holdout_fraction': HOLDOUT_FRACTION,
                   'split_version': SPLIT_VERSION}, f, indent=2)


def load_format(model_dir: str) -> dict:
    """Load the answer format saved by save_format, or the default format."""
    path = os.path.join(model_dir, FORMAT_FILENAME)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'reverse_answers': False}


def predict_answer(model, tokenizer, prompt: str, max_new_tokens: int = 8) -> str:
    """Greedy-decode a completion for prompt and return its first word."""
    import torch

    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False,
                       return_token_type_ids=False).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    words = text.strip().split()
    return words[0] if words else ''


def evaluate_accuracy(model, tokenizer, num_per_category: int, reverse: bool = False,
                      seed: int = 123, categories=CATEGORIES, num_examples: int = 0,
                      max_new_tokens: int = 8) -> dict:
    """
    Exact-match accuracy on held-out problems.

    Returns {'overall': float, 'per_category': {cat: float}, 'examples': {cat: [str]}}.
    Examples are shown with answers un-reversed so they read naturally.
    """
    was_training = model.training
    model.eval()
    rng = random.Random(seed)
    per_category, examples = {}, {}
    total_correct = 0
    for category in categories:
        correct, shown = 0, []
        for expression, prompt, result in generate_problems(category, num_per_category, rng, held_out=True):
            target = format_answer(result, reverse)
            prediction = predict_answer(model, tokenizer, prompt, max_new_tokens)
            correct += prediction == target
            if len(shown) < num_examples:
                readable = unreverse_answer(prediction) if reverse else prediction
                mark = '✓' if prediction == target else '✗'
                shown.append(f"{mark} {prompt}{readable or '<empty>'}   (true {result})")
        per_category[category] = correct / num_per_category
        examples[category] = shown
        total_correct += correct
    if was_training:
        model.train()
    return {
        'overall': total_correct / (num_per_category * len(categories)),
        'per_category': per_category,
        'examples': examples,
    }
