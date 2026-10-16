#!/usr/bin/env python3
"""
Rewrite a SentencePiece tokenizer directory as a correct HuggingFace fast tokenizer.

Tokenizers built before the fix were saved as T5Tokenizer, which AutoTokenizer
loads as T5TokenizerFast: newlines and unseen characters become UNK, runs of
spaces collapse, and "</s>" is appended instead of the EOS token. This rebuilds
the directory's tokenizer files from its SentencePiece model. Token ids don't
change, so models trained with the slow tokenizer stay compatible; text tokenized
through the broken fast one (pre-tokenized caches, trained models) does not
match the fixed tokenizer and is worth redoing.

Usage:
    python -m guac.tokenizer.convert scratch/tokenizers/sentencepiece/shared_60k
    python -m guac.tokenizer.convert scratch/models/pretraining/guac-3b   # a model directory
"""

import argparse
from pathlib import Path

from guac.core.prompt_templates import create_prompt_tokenizer
from guac.tokenizer.utils import wrap_as_hf_tokenizer


def convert(directory: str) -> None:
    directory = Path(directory)
    model = next((directory / name for name in ('tokenizer.model', 'spiece.model')
                  if (directory / name).exists()), None)
    if model is None:
        raise FileNotFoundError(f"No tokenizer.model or spiece.model in {directory}")
    tokenizer = wrap_as_hf_tokenizer(str(model), create_prompt_tokenizer().get_core_tokens(), str(directory))
    print(f"Rewrote {directory} as a fast tokenizer ({len(tokenizer):,} tokens, eos id {tokenizer.eos_token_id})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directories', nargs='+')
    for directory in parser.parse_args().directories:
        convert(directory)


if __name__ == '__main__':
    main()
