"""
Tests for the arithmetic train/held-out split: commuted twins share a pool,
so held-out accuracy never measures recall of a problem's mirror image.
"""

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest

from guac.core.arithmetic import (
    CATEGORIES, HOLDOUT_FRACTION, canonical_expression, generate_problems, is_held_out,
)


@pytest.mark.parametrize("expression, canonical", [
    ("7 + 3", "3 + 7"), ("3 + 7", "3 + 7"), ("12 * 5", "5 * 12"),
    ("(9 + 2) * 4", "(2 + 9) * 4"), ("7 - 3", "7 - 3"), ("12 / 4", "12 / 4"),
])
def test_canonical_form_sorts_only_commutative_operands(expression, canonical):
    assert canonical_expression(expression) == canonical


def test_commuted_twins_land_in_the_same_pool():
    for a in range(1, 60):
        for b in range(1, 60):
            assert is_held_out(f"{a} + {b}") == is_held_out(f"{b} + {a}")
            assert is_held_out(f"{a} * {b}") == is_held_out(f"{b} * {a}")
            assert is_held_out(f"({a} + {b}) * 3") == is_held_out(f"({b} + {a}) * 3")


def test_held_out_fraction_is_about_ten_percent():
    held_out = sum(is_held_out(f"{a} * {b}") for a in range(1, 101) for b in range(a, 101))
    assert abs(held_out / 5050 - HOLDOUT_FRACTION) < 0.02


@pytest.mark.parametrize("category", CATEGORIES)
def test_training_and_held_out_problems_never_share_a_canonical_form(category):
    train = {canonical_expression(e) for e, _, _ in generate_problems(category, 2000, random.Random(0), held_out=False)}
    held_out = {canonical_expression(e) for e, _, _ in generate_problems(category, 300, random.Random(1), held_out=True)}
    assert not train & held_out
