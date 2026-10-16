#!/usr/bin/env python3
"""
VocabSpec — declarative configuration for the hierarchical tokenizer pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class PartitionSpec:
    """Configuration for a single domain partition (Stage 2 trainer)."""
    name: str
    trainer_data: List[str]   # dataset names passed to load_pretraining_datasets
    budget: int               # number of new tokens this partition may contribute


@dataclass
class VocabSpec:
    """
    Full specification for a hierarchical tokenizer build.

    The pipeline has three stages:
        Stage 1 (core):       Train a generalist tokenizer seeded with anchor tokens.
        Stage 2 (partitions): Train N domain tokenizers seeded with the full core vocab,
                              extract only the new domain-specific tokens each learns.
        Stage 3 (final):      Combine core + all partition tokens, train the full-size
                              tokenizer with the combined set as user_defined_symbols.

    Token ID stability:
        SentencePiece places user_defined_symbols at the beginning of the ID space
        (after the 4 reserved special tokens). Passing core + partition tokens as
        user_defined_symbols in Stage 3 means those tokens get the lowest IDs in the
        final model — they stay stable if you later extend the vocab.
    """

    total_size: int               # target vocab size for the final tokenizer
    core_size: int                # vocab size for the Stage 1 generalist core
    output_dir: str               # root directory for all trained artifacts
    final_data: List[str]         # datasets used for Stage 1 and Stage 3 training
    anchors_path: Optional[str] = None       # path to predefined anchor tokens file
    partitions: List[PartitionSpec] = field(default_factory=list)
    dedup_strategy: str = 'coverage'         # 'coverage' or 'first-wins'
    dedup_min_coverage: int = 1              # for coverage strategy: min partitions a token must appear in
    byte_fallback: bool = True               # add 256 byte-fallback tokens to every stage

    # -------------------------------------------------------------------------
    # Derived paths
    # -------------------------------------------------------------------------

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def core_model_path(self) -> Path:
        return self.output_path / 'core' / 'tokenizer.model'

    @property
    def final_model_path(self) -> Path:
        return self.output_path / 'final' / 'tokenizer.model'

    @property
    def final_hf_dir(self) -> Path:
        """Directory where the HuggingFace-compatible tokenizer is saved."""
        return self.output_path / 'final' / 'hf'

    def partition_model_path(self, name: str) -> Path:
        return self.output_path / 'partitions' / name / 'tokenizer.model'

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self, anchor_count: int = 0) -> None:
        """Raise ValueError for obviously invalid configurations."""
        special_tokens = 4  # pad, bos, eos, unk
        min_core = special_tokens + anchor_count + 1
        if self.core_size < min_core:
            raise ValueError(
                f"core_size ({self.core_size}) must be > {min_core} "
                f"(4 special tokens + {anchor_count} anchors + at least 1 learned token)"
            )

        if self.total_size <= self.core_size:
            raise ValueError(
                f"total_size ({self.total_size}) must be larger than core_size ({self.core_size}) "
                f"to leave room for the final training step to learn new tokens"
            )

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> 'VocabSpec':
        """Load a VocabSpec from a YAML file."""
        with open(path, encoding='utf-8') as f:
            raw = yaml.safe_load(f)

        # Support optional top-level 'vocab_spec' wrapper key
        spec = raw.get('vocab_spec', raw)

        partitions = [
            PartitionSpec(
                name=p['name'],
                trainer_data=p['trainer_data'],
                budget=int(p['budget']),
            )
            for p in spec.get('partitions', [])
        ]

        dedup = spec.get('deduplication', {})
        final_training = spec.get('final_training', {})

        return cls(
            total_size=int(spec['total_size']),
            core_size=int(spec['core_size']),
            output_dir=spec['output_dir'],
            final_data=final_training.get('data', []),
            anchors_path=spec.get('anchors', {}).get('path'),
            partitions=partitions,
            dedup_strategy=dedup.get('strategy', 'coverage'),
            dedup_min_coverage=int(dedup.get('min_coverage', 1)),
            byte_fallback=bool(spec.get('byte_fallback', True)),
        )
