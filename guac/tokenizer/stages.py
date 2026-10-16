#!/usr/bin/env python3
"""
The three training stages of the hierarchical tokenizer pipeline.

    Stage 1 (train_core):       Generalist core tokenizer seeded with the prompt template
                                tokens and anchor tokens.
    Stage 2 (train_partition):  Domain tokenizer; returns the pieces it adds to the core.
    Stage 3 (train_final):      The core model extended with the partitions' pieces, then
                                with more pieces from the full corpus up to total_size.

Stages 2 and 3 can't seed SentencePiece with the core vocab (see utils.extend_model),
so they train their own models and append those models' pieces to the core model.
Every core piece keeps its id in the final tokenizer.
"""

import os
from pathlib import Path
from typing import List, Set

from .vocab_spec import VocabSpec, PartitionSpec
from .utils import (
    extend_model,
    extract_tokens_from_model,
    load_model_proto,
    prepare_training_text,
    ranked_pieces,
    select_reachable,
    train_spm,
)


def prompt_special_tokens() -> List[str]:
    """The prompt template's special tokens, which every stage-1 core reserves (lowest ids)."""
    from ..core.prompt_templates import create_prompt_tokenizer
    return create_prompt_tokenizer().get_all_special_tokens()


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
    train_spm(text_file, model_prefix, vocab_size=spec.core_size,
              user_defined_symbols=list(dict.fromkeys(prompt_special_tokens() + anchor_tokens)),
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

    The partition's tokenizer is trained from scratch on its corpus (seeding it with
    the core vocab would leave it nothing to merge). Its learned pieces that the core
    lacks, in merge order and only those BPE can build from the core plus earlier
    picks, are this partition's contribution, up to its budget.

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
        return select_reachable(ranked_pieces(str(model_path)), core_set, partition.budget)

    print(f"  Training partition '{partition.name}' (budget={partition.budget:,})")
    datasets = _load_datasets(partition.trainer_data)

    text_file = str(output_dir / 'partitions' / partition.name / 'training_data.txt')
    prepare_training_text(datasets, text_file, max_texts=50_000)

    # Big enough to learn the core's merges plus `budget` more; a small corpus may stop sooner
    vocab_size = len(core_tokens) + partition.budget + (256 if byte_fallback else 0) + 4
    model_prefix = str(output_dir / 'partitions' / partition.name / 'tokenizer')
    train_spm(text_file, model_prefix, vocab_size=vocab_size, user_defined_symbols=[],
              byte_fallback=byte_fallback, hard_vocab_limit=False)

    os.remove(text_file)
    new_tokens = select_reachable(ranked_pieces(str(model_path)), core_set, partition.budget)
    print(f"  '{partition.name}': {len(new_tokens):,} new tokens learned")
    return new_tokens


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

def train_final(
    spec: VocabSpec,
    partition_tokens: List[str],
    output_dir: Path,
    force: bool = False,
) -> str:
    """
    Stage 3: Build the full-size final tokenizer.

    The core model, extended first with the (deduplicated) partition tokens, then with
    pieces from a tokenizer trained on the full corpus until the vocab reaches
    total_size. Core pieces keep their ids, and partition tokens come right after, so
    a later build that extends this one keeps all of them.

    Args:
        spec:              The VocabSpec driving this pipeline.
        partition_tokens:  Ordered, deduplicated tokens from Stage 2.
        output_dir:        Root output directory.
        force:             Re-train even if the model already exists.

    Returns:
        Path to the final .model file.
    """
    model_path = spec.final_model_path

    if model_path.exists() and not force:
        print(f"  Final tokenizer exists at {model_path}, skipping.")
        return str(model_path)

    core_model = str(spec.core_model_path)
    vocab = set(_pieces(core_model))
    core_size = len(vocab)
    partition_pieces = select_reachable(partition_tokens, vocab, spec.total_size - core_size)
    vocab |= set(partition_pieces)

    new_slot_count = spec.total_size - core_size - len(partition_pieces)
    if new_slot_count <= 0:
        raise ValueError(
            f"The core ({core_size:,}) and partition tokens ({len(partition_pieces):,}) fill the entire "
            f"vocab budget. Increase total_size beyond {core_size + len(partition_pieces):,} to leave room "
            f"for the final training step."
        )

    print(f"  Building final tokenizer (total_size={spec.total_size:,})")
    print(f"  Core {core_size:,} + {len(partition_pieces):,} partition tokens; learning {new_slot_count:,} more from data")

    if not spec.final_data:
        raise ValueError("VocabSpec.final_data must list at least one dataset for Stage 3 training.")

    datasets = _load_datasets(spec.final_data)
    final_dir = output_dir / 'final'
    text_file = str(final_dir / 'training_data.txt')
    prepare_training_text(datasets, text_file, max_texts=500_000)

    # A full-size tokenizer of its own: its first merges repeat the core's, the rest are new
    full_prefix = str(final_dir / 'full_corpus')
    train_spm(text_file, full_prefix, vocab_size=spec.total_size, user_defined_symbols=[],
              byte_fallback=spec.byte_fallback, hard_vocab_limit=False)
    os.remove(text_file)
    corpus_pieces = select_reachable(ranked_pieces(f"{full_prefix}.model"), vocab, new_slot_count)

    extend_model(core_model, partition_pieces + corpus_pieces, str(model_path))
    total = len(_pieces(str(model_path)))
    if total < spec.total_size:
        print(f"  Note: the corpus ran out of merges at {total:,} tokens (total_size {spec.total_size:,})")
    print(f"  Final tokenizer saved to {model_path}")
    return str(model_path)


def _pieces(model_path: str) -> List[str]:
    return [p.piece for p in load_model_proto(model_path).pieces]
