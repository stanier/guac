#!/usr/bin/env python3
"""
Build a character-level HuggingFace tokenizer (printable ASCII + newline).

Useful for small arithmetic / smoke-test runs where a learned subword vocab
is overkill or lacks the needed symbols.

Usage:
    python -m guac.tokenizer.build_char_tokenizer
    python -m guac.tokenizer.build_char_tokenizer --output scratch/tokenizers/character/ascii
"""

import argparse
import string

from tokenizers import Tokenizer, Regex, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from guac.core.prompt_templates import create_prompt_tokenizer


def build_char_tokenizer(output_dir: str) -> PreTrainedTokenizerFast:
    core = create_prompt_tokenizer().get_core_tokens()
    specials = [core['pad_token'], core['unk_token'], core['bos_token'], core['eos_token']]
    chars = sorted(set(string.printable) - set('\t\r\x0b\x0c'))

    vocab = {tok: i for i, tok in enumerate(specials + chars)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token=core['unk_token']))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tok.decoder = decoders.Fuse()
    # Append EOS so the model learns where an answer ends
    tok.post_processor = processors.TemplateProcessing(
        single=["$A", core['eos_token']],  # list form: core tokens contain spaces
        special_tokens=[(core['eos_token'], vocab[core['eos_token']])],
    )

    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, **core)
    hf_tok.save_pretrained(output_dir)
    return hf_tok


def main():
    parser = argparse.ArgumentParser(description="Build a character-level tokenizer")
    parser.add_argument('--output', default='scratch/tokenizers/character/ascii')
    args = parser.parse_args()

    tok = build_char_tokenizer(args.output)
    sample = "(12 + 7) * 3 = 57"
    print(f"Saved {len(tok)}-token character tokenizer to {args.output}")
    print(f"  {sample!r} -> {tok.convert_ids_to_tokens(tok(sample)['input_ids'])}")


if __name__ == '__main__':
    main()
