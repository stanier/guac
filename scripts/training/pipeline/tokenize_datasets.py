#!/usr/bin/env python3
"""
Guac Dataset Tokenization Script

Pre-tokenize pre-training datasets into the cache that pretrain.py reads, so a
training run can start without tokenizing first. It splits, packs and caches
exactly as pretrain.py does (prepare_pretraining_data), so pass the same
settings the training run will use.

Usage:
    python tokenize_datasets.py --tokenizer scratch/tokenizers/sentencepiece/shared_60k --datasets wikitext-103
    python tokenize_datasets.py --tokenizer ./my_tokenizer --datasets fineweb,wikitext-103 --max-length 2048
"""

import os, sys, argparse
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from transformers import AutoTokenizer

from scripts.training.pipeline.pretrain import prepare_pretraining_data


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize pre-training datasets into pretrain.py's cache")

    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path to trained tokenizer directory (the preset\'s tokenizer_model_path)')
    parser.add_argument('--datasets', type=str, required=True,
                        help='Comma-separated list of datasets to tokenize')
    parser.add_argument('--max-length', type=int, default=1024,
                        help="Sequence length (the preset's max_sequence_length)")
    parser.add_argument('--pack', action=argparse.BooleanOptionalAction, default=True,
                        help='Pack documents into full-length blocks (as pretrain.py does by default)')
    parser.add_argument('--validation-docs', type=int, default=1000,
                        help='Documents held out per dataset for validation (as pretrain.py)')
    parser.add_argument('--debug', action='store_true',
                        help="Tokenize the debug subset pretrain.py --debug uses")
    parser.add_argument('--force-retokenize', action='store_true',
                        help='Force re-tokenization even if cached dataset exists')

    args = parser.parse_args()

    tokenizer_path = args.tokenizer[:-len('/tokenizer.model')] if args.tokenizer.endswith('tokenizer.model') else args.tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Loaded tokenizer with vocab size: {len(tokenizer):,}")

    train, validation = prepare_pretraining_data(
        args.datasets.split(','), tokenizer, args.max_length, pack=args.pack,
        validation_docs=args.validation_docs, debug=args.debug, force_retokenize=args.force_retokenize,
    )

    for name, split in (('train', train), ('validation', validation)):
        if split is None:
            continue
        try:
            print(f"{name}: {len(split):,} sequences")
        except TypeError:
            print(f"{name}: streaming (not cached; tokenized during training)")


if __name__ == "__main__":
    main()
