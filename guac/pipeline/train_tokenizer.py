#!/usr/bin/env python3
"""
Guac Tokenizer Training Script

Train a SentencePiece or BPE tokenizer on specified datasets.

Usage:
    python train_tokenizer.py --vocab-size 60000 --datasets wikitext-103
    python train_tokenizer.py --vocab-size 32000 --datasets fineweb,wikitext-103 --output tokenizers/custom
"""

import os, sys, yaml, argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from datasets import load_dataset, load_from_disk, concatenate_datasets

from guac.core.dataset_utils import load_pretraining_datasets


def get_special_tokens() -> tuple[Dict[str, str], List[str]]:
    """The core special tokens (pad, unk, bos, eos) and every special token, from prompt_templates."""
    from guac.core.prompt_templates import create_prompt_tokenizer
    prompt_tokenizer = create_prompt_tokenizer()
    return prompt_tokenizer.get_core_tokens(), prompt_tokenizer.get_all_special_tokens()


def load_datasets_for_tokenizer(dataset_names: List[str]) -> Any:
    """Load the training split of each dataset for tokenizer training."""
    return load_pretraining_datasets(dataset_names)


def get_output_path(output_dir: Optional[str], vocab_size: int, tokenizer_type: str) -> str:
    """Get output path for tokenizer with fallback."""
    if output_dir:
        return output_dir

    from guac.core.path_manager import get_tokenizer_output_path
    return str(get_tokenizer_output_path(tokenizer_type, f"shared_{vocab_size//1000}k"))


def create_training_text_file(datasets: Any, all_special_tokens: List[str],
                              max_samples: int = 50000) -> str:
    """Create temporary text file for tokenizer training."""
    temp_file = "tokenizer_training_data.txt"

    with open(temp_file, 'w', encoding='utf-8') as f:
        text_count = 0

        for item in datasets:
            if text_count >= max_samples:
                break
            text = (item.get('text') or '').strip()
            if text:
                f.write(text + '\n')
                text_count += 1

        if text_count < 100:
            raise ValueError(f"Only {text_count} non-empty texts loaded; a tokenizer needs real training text")

        for token in all_special_tokens[:10]:
            f.write(f"Example with {token} in text.\n")
            text_count += 1

    print(f"Created training file with {text_count:,} samples")
    return temp_file


def train_sentencepiece_tokenizer(training_file: str, output_path: str,
                                  vocab_size: int, all_special_tokens: List[str],
                                  core_tokens: Dict[str, str]) -> Any:
    """Train a SentencePiece BPE tokenizer and save it as a HuggingFace fast tokenizer."""
    from guac.tokenizer.utils import train_spm, wrap_as_hf_tokenizer

    print(f"Training SentencePiece tokenizer (vocab_size={vocab_size})")
    model_path = train_spm(training_file, os.path.join(output_path, "tokenizer"), vocab_size,
                           user_defined_symbols=all_special_tokens, character_coverage=0.9995)
    tokenizer = wrap_as_hf_tokenizer(model_path, core_tokens, output_path)
    print(f"SentencePiece tokenizer saved to {output_path}")
    return tokenizer


def train_bpe_tokenizer(training_file: str, output_path: str, vocab_size: int,
                        all_special_tokens: List[str], core_tokens: Dict[str, str]) -> Any:
    """Train a byte-level BPE tokenizer (every string encodes, nothing becomes UNK)."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    print(f"Training byte-level BPE tokenizer (vocab_size={vocab_size})")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # The core tokens first, so they get ids 0-3 (pad, unk, bos, eos)
    core = [core_tokens[k] for k in ('pad_token', 'unk_token', 'bos_token', 'eos_token')]
    special = [AddedToken(t, normalized=False, special=True) for t in core]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train([training_file], trainer)
    tokenizer.add_tokens([AddedToken(t, normalized=False) for t in all_special_tokens if t not in core])
    eos = core_tokens['eos_token']
    tokenizer.post_processor = processors.TemplateProcessing(
        single=["$A", eos], pair=["$A", eos, "$B:1", eos], special_tokens=[(eos, tokenizer.token_to_id(eos))])

    os.makedirs(output_path, exist_ok=True)
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **core_tokens)
    hf_tokenizer.save_pretrained(output_path)
    print(f"BPE tokenizer saved to {output_path}")
    return hf_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Guac Tokenizer Training Script")

    parser.add_argument('--vocab-size', type=int, default=60000,
                       help='Vocabulary size for tokenizer')
    parser.add_argument('--datasets', type=str, default='wikitext-103',
                       help='Comma-separated list of datasets for training')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for trained tokenizer')
    parser.add_argument('--tokenizer-type', type=str, default='sentencepiece',
                       choices=['sentencepiece', 'bpe'],
                       help='Type of tokenizer to train')
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='Maximum number of text samples to use')
    parser.add_argument('--force-bpe', action='store_true',
                       help='Same as --tokenizer-type bpe')

    args = parser.parse_args()
    if args.force_bpe:
        args.tokenizer_type = 'bpe'

    core_tokens, all_special_tokens = get_special_tokens()

    output_path = get_output_path(args.output_dir, args.vocab_size, args.tokenizer_type)

    if os.path.exists(os.path.join(output_path, "tokenizer_config.json")):
        print(f"Tokenizer already exists at {output_path}")
        print("Use --output-dir to specify a different location or delete the existing tokenizer")
        return

    dataset_names = args.datasets.split(',')
    print(f"Loading datasets: {', '.join(dataset_names)}")
    datasets = load_datasets_for_tokenizer(dataset_names)

    training_file = create_training_text_file(datasets, all_special_tokens, args.max_samples)

    try:
        if args.tokenizer_type == 'bpe':
            tokenizer = train_bpe_tokenizer(training_file, output_path, args.vocab_size,
                                            all_special_tokens, core_tokens)
        else:
            tokenizer = train_sentencepiece_tokenizer(
                training_file, output_path, args.vocab_size,
                all_special_tokens, core_tokens
            )

        print(f"\nTokenizer training complete!")
        print(f"Vocabulary size: {len(tokenizer):,}")
        print(f"Saved to: {output_path}")

    finally:
        if os.path.exists(training_file):
            os.remove(training_file)
            print(f"Cleaned up temporary file: {training_file}")


if __name__ == "__main__":
    main()
