#!/usr/bin/env python3
"""
Smoke tests for the hierarchical tokenizer pipeline.

These tests exercise the pure-Python logic in VocabSpec, VocabPipeline, and
deduplicate_partitions without training real SentencePiece models (which would
require actual dataset I/O).  extract_tokens_from_model is integration-only
and not covered here.

Usage:
    python tests/test_vocab_pipeline.py
    python tests/run_tests.py --test vocab_pipeline
"""

import io
import sys
import tempfile
import textwrap
from contextlib import redirect_stdout
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.training.tokenizer.vocab_spec import VocabSpec, PartitionSpec
from scripts.training.tokenizer.pipeline import VocabPipeline
from scripts.training.tokenizer.utils import deduplicate_partitions


# ---------------------------------------------------------------------------
# VocabSpec.from_yaml
# ---------------------------------------------------------------------------

def test_from_yaml_minimal():
    """from_yaml loads a minimal valid spec (no partitions) without error."""
    yaml_text = textwrap.dedent("""\
        total_size: 8000
        core_size: 4000
        output_dir: /tmp/test_vocab_minimal
        final_training:
          data:
            - wikitext-103
    """)
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
        f.write(yaml_text)
        path = f.name

    spec = VocabSpec.from_yaml(path)
    assert spec.total_size == 8000
    assert spec.core_size == 4000
    assert spec.output_dir == '/tmp/test_vocab_minimal'
    assert spec.final_data == ['wikitext-103']
    assert spec.partitions == []
    assert spec.dedup_strategy == 'coverage'
    assert spec.dedup_min_coverage == 1
    print("  ✓ from_yaml minimal spec")


def test_from_yaml_with_partitions():
    """from_yaml populates PartitionSpec objects and deduplication config."""
    yaml_text = textwrap.dedent("""\
        total_size: 16000
        core_size: 8000
        output_dir: /tmp/test_vocab_parts
        final_training:
          data: [wikitext-103]
        partitions:
          - name: code
            trainer_data: [github-code]
            budget: 2000
          - name: math
            trainer_data: [arxiv-math]
            budget: 1500
        deduplication:
          strategy: first-wins
          min_coverage: 2
    """)
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
        f.write(yaml_text)
        path = f.name

    spec = VocabSpec.from_yaml(path)
    assert len(spec.partitions) == 2
    assert spec.partitions[0].name == 'code'
    assert spec.partitions[0].budget == 2000
    assert spec.partitions[1].name == 'math'
    assert spec.partitions[1].trainer_data == ['arxiv-math']
    assert spec.dedup_strategy == 'first-wins'
    assert spec.dedup_min_coverage == 2
    print("  ✓ from_yaml with partitions and deduplication config")


def test_from_yaml_with_vocab_spec_wrapper():
    """from_yaml handles optional top-level 'vocab_spec' wrapper key."""
    yaml_text = textwrap.dedent("""\
        vocab_spec:
          total_size: 4000
          core_size: 2000
          output_dir: /tmp/test_vocab_wrapped
          final_training:
            data: [wikitext-103]
    """)
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
        f.write(yaml_text)
        path = f.name

    spec = VocabSpec.from_yaml(path)
    assert spec.total_size == 4000
    assert spec.core_size == 2000
    assert spec.output_dir == '/tmp/test_vocab_wrapped'
    print("  ✓ from_yaml with vocab_spec wrapper key")


# ---------------------------------------------------------------------------
# VocabSpec.validate
# ---------------------------------------------------------------------------

def test_validate_passes_for_valid_spec():
    """validate() does not raise for a well-formed spec."""
    spec = VocabSpec(
        total_size=8000,
        core_size=4000,
        output_dir='/tmp/ok',
        final_data=['wikitext-103'],
    )
    spec.validate(anchor_count=10)   # must not raise
    print("  ✓ validate passes for valid spec")


def test_validate_raises_when_core_too_small_for_anchors():
    """validate() raises ValueError when core_size < 4 special tokens + anchors + 1."""
    spec = VocabSpec(
        total_size=8000,
        core_size=10,   # 4 special + 20 anchors + 1 = 25 minimum; 10 < 25
        output_dir='/tmp/bad',
        final_data=['wikitext-103'],
    )
    try:
        spec.validate(anchor_count=20)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'core_size' in str(e)
    print("  ✓ validate raises when core_size is too small for anchor count")


def test_validate_raises_when_total_not_larger_than_core():
    """validate() raises ValueError when total_size <= core_size."""
    spec = VocabSpec(
        total_size=4000,
        core_size=4000,
        output_dir='/tmp/bad2',
        final_data=['wikitext-103'],
    )
    try:
        spec.validate(anchor_count=0)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'total_size' in str(e)
    print("  ✓ validate raises when total_size <= core_size")


# ---------------------------------------------------------------------------
# deduplicate_partitions
# ---------------------------------------------------------------------------

def test_dedup_first_wins():
    """first-wins: each token is kept only in the first partition that listed it."""
    partitions = {
        'a': ['tok1', 'tok2', 'tok3'],
        'b': ['tok2', 'tok4', 'tok5'],
        'c': ['tok3', 'tok5', 'tok6'],
    }
    result = deduplicate_partitions(partitions, strategy='first-wins')
    assert result['a'] == ['tok1', 'tok2', 'tok3']
    assert result['b'] == ['tok4', 'tok5']    # tok2 already in 'a'
    assert result['c'] == ['tok6']            # tok3, tok5 already seen
    print("  ✓ deduplicate first-wins")


def test_dedup_coverage_min1():
    """coverage with min_coverage=1: every token qualifies; first-seen wins."""
    partitions = {
        'a': ['tok1', 'tok2'],
        'b': ['tok2', 'tok3'],
    }
    result = deduplicate_partitions(partitions, strategy='coverage', min_coverage=1)
    # tok2 must appear in exactly one partition
    tok2_count = result['a'].count('tok2') + result['b'].count('tok2')
    assert tok2_count == 1
    assert 'tok1' in result['a']
    assert 'tok3' in result['b']
    print("  ✓ deduplicate coverage min_coverage=1")


def test_dedup_coverage_drops_singleton_tokens():
    """coverage with min_coverage=2: tokens in only one partition are dropped."""
    partitions = {
        'a': ['shared', 'only_a'],
        'b': ['shared', 'only_b'],
    }
    result = deduplicate_partitions(partitions, strategy='coverage', min_coverage=2)
    all_tokens = result['a'] + result['b']
    assert 'shared' in all_tokens
    assert 'only_a' not in all_tokens
    assert 'only_b' not in all_tokens
    print("  ✓ deduplicate coverage drops tokens below min_coverage")


def test_dedup_invalid_strategy_raises():
    """An unknown dedup strategy name raises ValueError."""
    try:
        deduplicate_partitions({'a': ['tok1']}, strategy='unknown')
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'unknown' in str(e)
    print("  ✓ deduplicate raises on unknown strategy")


# ---------------------------------------------------------------------------
# VocabPipeline.status
# ---------------------------------------------------------------------------

def test_pipeline_status_fresh_spec():
    """status() runs without error on an unbuilt spec and reports pending stages."""
    spec = VocabSpec(
        total_size=8000,
        core_size=4000,
        output_dir='/tmp/fresh_pipeline_never_built',
        final_data=['wikitext-103'],
        partitions=[
            PartitionSpec(name='code', trainer_data=['github-code'], budget=1000),
        ],
    )
    pipeline = VocabPipeline(spec)

    buf = io.StringIO()
    with redirect_stdout(buf):
        pipeline.status()   # must not raise

    output = buf.getvalue()
    # Status should mention pending or the '○' symbol for at least one stage
    assert 'pending' in output or '\u25cb' in output
    print("  ✓ VocabPipeline.status() runs on unbuilt spec")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS = [
    test_from_yaml_minimal,
    test_from_yaml_with_partitions,
    test_from_yaml_with_vocab_spec_wrapper,
    test_validate_passes_for_valid_spec,
    test_validate_raises_when_core_too_small_for_anchors,
    test_validate_raises_when_total_not_larger_than_core,
    test_dedup_first_wins,
    test_dedup_coverage_min1,
    test_dedup_coverage_drops_singleton_tokens,
    test_dedup_invalid_strategy_raises,
    test_pipeline_status_fresh_spec,
]


def main() -> bool:
    print(f"\n{'='*60}")
    print("Tokenizer Pipeline Tests")
    print(f"{'='*60}\n")

    passed = failed = 0
    for test in _TESTS:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
