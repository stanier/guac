#!/usr/bin/env python3
"""
The three training stages of the hierarchical tokenizer pipeline.

    Stage 1 (train_core):       Generalist core tokenizer seeded with anchor tokens.
    Stage 2 (train_partition):  Domain tokenizer seeded with core vocab; returns new tokens only.
    Stage 3 (train_final):      Full-size tokenizer seeded with the complete combined vocab.
"""

import os
from pathlib import Path
from typing import List, Set

from .vocab_spec import VocabSpec, PartitionSpec
from .utils import (
    extract_tokens_from_model,
    get_new_tokens,
    prepare_training_text,
    train_spm,
)


def _load_datasets(dataset_names: List[str]):
    """load_pretraining_datasets, imported when needed (it pulls in the datasets library)."""
    from ..core.dataset_utils import load_pretraining_datasets
    return load_pretraining_datasets(dataset_names)


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def train_core(
    spec: VocabSpec,
    anchor_tokens: List[str],
    output_dir: Path,
    force: bool = False,
) -> List[str]:
    """
    Stage 1: Train the generalist core tokenizer.

    Uses the full final_data corpus so the core covers the widest possible
    distribution before the domain partitions narrow in on their specialties.

    Args:
        spec:          The VocabSpec driving this pipeline.
        anchor_tokens: Predefined tokens forced into the core vocab.
        output_dir:    Root output directory.
        force:         Re-train even if the model already exists.

    Returns:
        Full token list from the trained core model (anchors + learned tokens).
    """
    model_path = spec.core_model_path

    if model_path.exists() and not force:
        print(f"  Core tokenizer exists at {model_path}, skipping.")
        return extract_tokens_from_model(str(model_path))

    print(f"  Training core tokenizer (target size={spec.core_size:,})")
    if not spec.final_data:
        raise ValueError("VocabSpec.final_data must list at least one dataset for Stage 1 training.")

    datasets = _load_datasets(spec.final_data)
    text_file = str(output_dir / 'core' / 'training_data.txt')
    prepare_training_text(datasets, text_file, max_texts=100_000)

    model_prefix = str(output_dir / 'core' / 'tokenizer')
    train_spm(text_file, model_prefix, vocab_size=spec.core_size, user_defined_symbols=anchor_tokens,
              byte_fallback=spec.byte_fallback)

    os.remove(text_file)
    tokens = extract_tokens_from_model(str(model_path))
    print(f"  Core vocab: {len(tokens):,} tokens ({len(tokens) - len(anchor_tokens):,} learned, {len(anchor_tokens):,} anchors)")
    return tokens


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def train_partition(
    partition: PartitionSpec,
    core_tokens: List[str],
    output_dir: Path,
    force: bool = False,
    byte_fallback: bool = True,
) -> List[str]:
    """
    Stage 2: Train a single domain partition tokenizer.

    The core vocab is passed as user_defined_symbols so it is preserved
    verbatim. SentencePiece then fills the remaining budget with tokens
    specific to the domain corpus — these are the tokens this partition
    contributes to the final vocab.

    Args:
        partition:    The PartitionSpec for this domain.
        core_tokens:  Token list from Stage 1 (anchors + generalist learned tokens).
        output_dir:   Root output directory.
        force:        Re-train even if the model already exists.

    Returns:
        New tokens learned by this partition (i.e., tokens not in core_tokens).
    """
    model_path = output_dir / 'partitions' / partition.name / 'tokenizer.model'
    core_set: Set[str] = set(core_tokens)

    if model_path.exists() and not force:
        print(f"  Partition '{partition.name}' exists, skipping.")
        return get_new_tokens(str(model_path), core_set)

    print(f"  Training partition '{partition.name}' (budget={partition.budget:,})")
    datasets = _load_datasets(partition.trainer_data)

    text_file = str(output_dir / 'partitions' / partition.name / 'training_data.txt')
    prepare_training_text(datasets, text_file, max_texts=50_000)

    # vocab_size = 4 special tokens + len(core) user-defined + partition budget
    vocab_size = 4 + len(core_tokens) + partition.budget
    model_prefix = str(output_dir / 'partitions' / partition.name / 'tokenizer')
    train_spm(text_file, model_prefix, vocab_size=vocab_size, user_defined_symbols=core_tokens,
              byte_fallback=byte_fallback)

    os.remove(text_file)
    new_tokens = get_new_tokens(str(model_path), core_set)
    print(f"  '{partition.name}': {len(new_tokens):,} new tokens learned")
    return new_tokens


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

def train_final(
    spec: VocabSpec,
    combined_tokens: List[str],
    output_dir: Path,
    force: bool = False,
) -> str:
    """
    Stage 3: Train the full-size final tokenizer.

    All combined tokens (core + deduplicated partition tokens) are passed as
    user_defined_symbols. They receive the lowest IDs in the final model,
    making them stable if you later want to extend the vocab further.
    SentencePiece learns the remaining (total_size - 4 - len(combined)) slots
    from the full corpus.

    Args:
        spec:             The VocabSpec driving this pipeline.
        combined_tokens:  Ordered, deduplicated token list from Stages 1 + 2.
        output_dir:       Root output directory.
        force:            Re-train even if the model already exists.

    Returns:
        Path to the trained .model file.
    """
    model_path = spec.final_model_path

    if model_path.exists() and not force:
        print(f"  Final tokenizer exists at {model_path}, skipping.")
        return str(model_path)

    new_slot_count = spec.total_size - 4 - len(combined_tokens)
    if new_slot_count <= 0:
        raise ValueError(
            f"combined_tokens ({len(combined_tokens):,}) fills the entire vocab budget. "
            f"Increase total_size beyond {len(combined_tokens) + 4:,} to leave room for "
            f"the final training step."
        )

    print(f"  Training final tokenizer (total_size={spec.total_size:,})")
    print(f"  Seeded with {len(combined_tokens):,} tokens; learning {new_slot_count:,} more from data")

    if not spec.final_data:
        raise ValueError("VocabSpec.final_data must list at least one dataset for Stage 3 training.")

    datasets = _load_datasets(spec.final_data)
    text_file = str(output_dir / 'final' / 'training_data.txt')
    prepare_training_text(datasets, text_file, max_texts=500_000)

    model_prefix = str(output_dir / 'final' / 'tokenizer')
    train_spm(text_file, model_prefix, vocab_size=spec.total_size, user_defined_symbols=combined_tokens,
              byte_fallback=spec.byte_fallback)

    os.remove(text_file)
    print(f"  Final tokenizer saved to {model_path}")
    return str(model_path)
