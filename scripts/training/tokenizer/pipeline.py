#!/usr/bin/env python3
"""
VocabPipeline — orchestrates the three-stage hierarchical tokenizer build.

Usage (Python):
    from scripts.training.tokenizer import VocabPipeline, VocabSpec

    spec = VocabSpec.from_yaml("vocab_spec.yaml")
    pipeline = VocabPipeline(spec)
    pipeline.run()

Usage (CLI):
    python -m scripts.training.tokenizer.pipeline --spec vocab_spec.yaml
    python -m scripts.training.tokenizer.pipeline --spec vocab_spec.yaml --force
    python -m scripts.training.tokenizer.pipeline --spec vocab_spec.yaml --status
"""

import argparse
from pathlib import Path
from typing import List

from .vocab_spec import VocabSpec
from .utils import load_anchor_tokens, deduplicate_partitions, wrap_as_hf_tokenizer
from .stages import train_core, train_partition, train_final


class VocabPipeline:
    """
    Hierarchical tokenizer training pipeline.

    Stages
    ------
    1. Core       Train a generalist tokenizer seeded with predefined anchor tokens.
    2. Partitions Train N domain tokenizers seeded with the full core vocab; extract
                  only the new tokens each domain trainer learns.
    3. Final      Combine core + deduplicated partition tokens, train the full-size
                  tokenizer with this combined set as user_defined_symbols, then wrap
                  and save as a HuggingFace-compatible tokenizer.

    Token ID stability
    ------------------
    Because user_defined_symbols are assigned IDs 4..N (after the four special tokens)
    in SentencePiece, passing the same combined token list in future extended builds
    preserves those IDs. New tokens from extended partitions simply append to the end.
    """

    def __init__(self, spec: VocabSpec):
        self.spec = spec
        self.output_dir = Path(spec.output_dir)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def run(self, force: bool = False) -> str:
        """
        Execute the full pipeline.

        Args:
            force: Re-run all stages even if their outputs already exist.

        Returns:
            Path to the HuggingFace tokenizer directory (loadable with
            AutoTokenizer.from_pretrained).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load anchor tokens
        anchor_tokens: List[str] = []
        if self.spec.anchors_path:
            anchor_tokens = load_anchor_tokens(self.spec.anchors_path)
            print(f"Loaded {len(anchor_tokens):,} anchor tokens from {self.spec.anchors_path}")

        self.spec.validate(anchor_count=len(anchor_tokens))

        # ── Stage 1 ──────────────────────────────────────────────────────────
        print("\n── Stage 1: Core ────────────────────────────────────────────────")
        core_tokens = train_core(self.spec, anchor_tokens, self.output_dir, force=force)
        print(f"Core vocab complete: {len(core_tokens):,} tokens")

        # ── Stage 2 ──────────────────────────────────────────────────────────
        if self.spec.partitions:
            print("\n── Stage 2: Partitions ──────────────────────────────────────────")
            raw_partition_tokens = {}
            for partition in self.spec.partitions:
                new_tokens = train_partition(partition, core_tokens, self.output_dir, force=force,
                                             byte_fallback=self.spec.byte_fallback)
                raw_partition_tokens[partition.name] = new_tokens

            deduped = deduplicate_partitions(
                raw_partition_tokens,
                strategy=self.spec.dedup_strategy,
                min_coverage=self.spec.dedup_min_coverage,
            )

            total_raw = sum(len(t) for t in raw_partition_tokens.values())
            all_partition_tokens = [t for tokens in deduped.values() for t in tokens]
            print(f"\nDeduplication ({self.spec.dedup_strategy}, min_coverage={self.spec.dedup_min_coverage}):")
            print(f"  Before: {total_raw:,} tokens across {len(deduped)} partitions")
            print(f"  After:  {len(all_partition_tokens):,} tokens")
            for name, tokens in deduped.items():
                print(f"    {name}: {len(tokens):,}")
        else:
            all_partition_tokens = []
            print("\nNo partitions configured, skipping Stage 2.")

        # Combine, preserving order and deduplicating any residual overlap
        combined = list(dict.fromkeys(core_tokens + all_partition_tokens))
        print(f"\nCombined seed vocab: {len(combined):,} tokens")

        # ── Stage 3 ──────────────────────────────────────────────────────────
        print("\n── Stage 3: Final ───────────────────────────────────────────────")
        final_model_path = train_final(self.spec, combined, self.output_dir, force=force)

        # Wrap as HuggingFace tokenizer
        hf_dir = str(self.spec.final_hf_dir)
        print(f"  Wrapping as HuggingFace tokenizer → {hf_dir}")
        core_token_map = self._get_core_tokens()
        wrap_as_hf_tokenizer(final_model_path, core_token_map, hf_dir)

        print(f"\n✅ Pipeline complete.")
        print(f"   HuggingFace tokenizer: {hf_dir}")
        print(f"   Raw SentencePiece model: {final_model_path}")
        print(f"   Load with: AutoTokenizer.from_pretrained('{hf_dir}')")
        return hf_dir

    def status(self) -> None:
        """Print the current build status of each pipeline stage."""
        spec = self.spec
        print(f"VocabPipeline status  [{spec.output_dir}]")
        print(f"  Stage 1 — core  (size={spec.core_size:,}):   "
              f"{'✓ built' if spec.core_model_path.exists() else '○ pending'}")
        for p in spec.partitions:
            path = spec.partition_model_path(p.name)
            print(f"  Stage 2 — '{p.name}'  (budget={p.budget:,}):  "
                  f"{'✓ built' if path.exists() else '○ pending'}")
        print(f"  Stage 3 — final (size={spec.total_size:,}): "
              f"{'✓ built' if spec.final_model_path.exists() else '○ pending'}")
        print(f"  HuggingFace wrap:                     "
              f"{'✓ built' if spec.final_hf_dir.exists() else '○ pending'}")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_core_tokens(self) -> dict:
        """Resolve core special tokens from prompt_templates or use defaults."""
        try:
            try:
                from scripts.training.core.prompt_templates import create_prompt_tokenizer
            except ImportError:
                from ..core.prompt_templates import create_prompt_tokenizer
            pt = create_prompt_tokenizer()
            return pt.get_core_tokens()
        except Exception:
            return {
                'pad_token': '<pad>',
                'unk_token': '<unk>',
                'bos_token': '<s>',
                'eos_token': '</s>',
            }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical tokenizer pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--spec', required=True, help='Path to vocab_spec.yaml')
    parser.add_argument('--force', action='store_true', help='Re-run all stages even if outputs exist')
    parser.add_argument('--status', action='store_true', help='Print build status and exit')
    args = parser.parse_args()

    spec = VocabSpec.from_yaml(args.spec)
    pipeline = VocabPipeline(spec)

    if args.status:
        pipeline.status()
    else:
        pipeline.run(force=args.force)


if __name__ == '__main__':
    main()
