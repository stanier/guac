#!/usr/bin/env python3
"""
Guac Dataset Tokenization Script

Pre-tokenize pre-training datasets into the cache that pretrain.py reads, so a
training run can start without tokenizing first. It splits, packs and caches
exactly as pretrain.py does (prepare_pretraining_data), so pass the same
settings the training run will use.

With --model-config, the tokenizer, sequence length, packing and validation
hold-out default to that preset's, which is what pretrain.py --model-config uses.

Usage:
    python tokenize_datasets.py --model-config guac-3b --datasets wikitext-103
    python tokenize_datasets.py --tokenizer ./my_tokenizer --datasets fineweb,wikitext-103 --max-length 2048
"""

import os, sys, argparse
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


from guac.core.pipeline_utils import load_model_config, load_tokenizer, resolve_scratch_path
from guac.pipeline.pretrain import load_tokenizer as load_preset_tokenizer, prepare_pretraining_data


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize pre-training datasets into pretrain.py's cache")

    parser.add_argument('--model-config', type=str, default=None,
                        help="Preset whose tokenizer and data settings to use, as pretrain.py --model-config")
    parser.add_argument('--tokenizer', type=str, default=None,
                        help="Tokenizer directory (default: the preset's tokenizer_model_path)")
    parser.add_argument('--datasets', type=str, required=True,
                        help='Comma-separated list of datasets to tokenize')
    parser.add_argument('--max-length', type=int, default=None,
                        help="Sequence length (default: the preset's max_sequence_length, else 1024)")
    parser.add_argument('--pack', action=argparse.BooleanOptionalAction, default=None,
                        help="Pack documents into full-length blocks (default: the preset's pack_sequences, else on)")
    parser.add_argument('--validation-docs', type=int, default=None,
                        help="Documents held out per dataset for validation (default: the preset's, else 1000)")
    parser.add_argument('--debug', action='store_true',
                        help="Tokenize the debug subset pretrain.py --debug uses")
    parser.add_argument('--force-retokenize', action='store_true',
                        help='Force re-tokenization even if cached dataset exists')

    args = parser.parse_args()

    if args.model_config is None and args.tokenizer is None:
        parser.error("pass --model-config, --tokenizer, or both")
    config = load_model_config(args.model_config) if args.model_config else {}
    if args.tokenizer:
        tokenizer_path = resolve_scratch_path(args.tokenizer)
        if tokenizer_path.endswith('.model'):
            tokenizer_path = os.path.dirname(tokenizer_path)
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"Loaded tokenizer with vocab size: {len(tokenizer):,}")
    else:
        tokenizer = load_preset_tokenizer(config)

    max_length = args.max_length or config.get('max_sequence_length', 1024)
    pack = args.pack if args.pack is not None else config.get('pack_sequences', True)
    validation_docs = args.validation_docs if args.validation_docs is not None else config.get('validation_docs', 1000)
    print(f"Max length {max_length}, packing {'on' if pack else 'off'}, {validation_docs} validation documents")

    train, validation = prepare_pretraining_data(
        args.datasets.split(','), tokenizer, max_length, pack=pack,
        validation_docs=validation_docs, debug=args.debug, force_retokenize=args.force_retokenize,
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
