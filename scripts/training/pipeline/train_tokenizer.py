#!/usr/bin/env python3
"""
GUAC Tokenizer Training Script

Train a SentencePiece or BPE tokenizer on specified datasets.

Usage:
    python train_tokenizer.py --vocab-size 60000 --datasets wikitext-103
    python train_tokenizer.py --vocab-size 32000 --datasets fineweb,wikitext-103 --output tokenizers/custom
"""

import os, sys, yaml, argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, concatenate_datasets

try:
    from scripts.training.core.dataset_utils import load_pretraining_datasets
except ImportError:
    try:
        from ..core.dataset_utils import load_pretraining_datasets
    except ImportError:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        from core.dataset_utils import load_pretraining_datasets


def get_special_tokens() -> tuple[Dict[str, str], List[str]]:
    """Get special tokens for tokenizer."""
    try:
        from scripts.training.core.prompt_templates import create_prompt_tokenizer
        prompt_tokenizer = create_prompt_tokenizer()
        core_tokens = prompt_tokenizer.get_core_tokens()
        all_special_tokens = prompt_tokenizer.get_all_special_tokens()
    except ImportError:
        core_tokens = {
            'pad_token': '<pad>', 'unk_token': '<unk>',
            'bos_token': '<s>', 'eos_token': '</s>'
        }
        all_special_tokens = list(core_tokens.values())

    return core_tokens, all_special_tokens


def load_datasets_for_tokenizer(dataset_names: List[str]) -> Any:
    """Load datasets for tokenizer training."""
    try:
        return load_pretraining_datasets(dataset_names)
    except:
        print("Failed to load datasets, using fallback")
        return None


def get_output_path(output_dir: Optional[str], vocab_size: int, tokenizer_type: str) -> str:
    """Get output path for tokenizer with fallback."""
    if output_dir:
        return output_dir

    try:
        from scripts.training.core.path_manager import get_tokenizer_output_path
        path = get_tokenizer_output_path(tokenizer_type, f"shared_{vocab_size//1000}k")
        return str(path)
    except ImportError:
        return f"scratch/tokenizers/{tokenizer_type}/shared_{vocab_size//1000}k"


def create_training_text_file(datasets: Any, all_special_tokens: List[str],
                              max_samples: int = 50000) -> str:
    """Create temporary text file for tokenizer training."""
    temp_file = "tokenizer_training_data.txt"

    with open(temp_file, 'w', encoding='utf-8') as f:
        text_count = 0

        if datasets is not None:
            if hasattr(datasets, '__iter__'):
                for item in datasets:
                    if text_count >= max_samples:
                        break
                    text = item.get('text', '').strip()
                    if text:
                        f.write(text + '\n')
                        text_count += 1
            elif hasattr(datasets, 'select'):
                subset = datasets.select(range(min(max_samples, len(datasets))))
                for item in subset:
                    text = item.get('text', '').strip()
                    if text:
                        f.write(text + '\n')
                        text_count += 1
            else:
                for item in datasets:
                    if text_count >= max_samples:
                        break
                    text = str(item).strip() if item else ""
                    if text:
                        f.write(text + '\n')
                        text_count += 1

        if text_count < 100:
            print("Warning: Very few samples loaded, using fallback data")
            basic_texts = [
                "Hello world", "This is a test", "Machine learning is fun",
                "The quick brown fox jumps over the lazy dog",
                "Python is a programming language", "Deep learning with neural networks",
                "Natural language processing", "Transformers are powerful models"
            ] * 100

            for text in basic_texts:
                f.write(text + '\n')
                text_count += 1

        for token in all_special_tokens[:10]:
            f.write(f"Example with {token} in text.\n")
            text_count += 1

    print(f"Created training file with {text_count:,} samples")
    return temp_file


def train_sentencepiece_tokenizer(training_file: str, output_path: str,
                                  vocab_size: int, all_special_tokens: List[str],
                                  core_tokens: Dict[str, str]) -> Optional[Any]:
    """Train SentencePiece tokenizer."""
    try:
        import sentencepiece as spm
        from transformers import T5Tokenizer

        print(f"Training SentencePiece tokenizer (vocab_size={vocab_size})")

        model_prefix = os.path.join(output_path, "tokenizer")
        os.makedirs(output_path, exist_ok=True)

        spm.SentencePieceTrainer.train(
            input=training_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=0.9995,
            split_digits=True,
            byte_fallback=True,
            normalization_rule_name='nfkc',
            add_dummy_prefix=True,
            remove_extra_whitespaces=False,
            max_sentence_length=32768,
            user_defined_symbols=all_special_tokens,
            minloglevel=1,
            pad_id=0, bos_id=1, eos_id=2, unk_id=3
        )

        tokenizer = T5Tokenizer(vocab_file=f"{model_prefix}.model", extra_ids=0)
        tokenizer.pad_token = core_tokens['pad_token']
        tokenizer.bos_token = core_tokens['bos_token']
        tokenizer.eos_token = core_tokens['eos_token']
        tokenizer.unk_token = core_tokens['unk_token']

        tokenizer.save_pretrained(output_path)
        print(f"SentencePiece tokenizer saved to {output_path}")
        return tokenizer

    except Exception as e:
        print(f"SentencePiece training failed: {e}")
        return None


def train_bpe_tokenizer(training_file: str, output_path: str,
                       vocab_size: int, core_tokens: Dict[str, str]) -> Any:
    """Train BPE tokenizer as fallback."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    print(f"Training BPE tokenizer (vocab_size={vocab_size})")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    special_tokens_list = list(core_tokens.values())
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens_list)

    with open(training_file, 'r', encoding='utf-8') as f:
        training_data = [line.strip() for line in f if line.strip()]

    tokenizer.train_from_iterator(training_data, trainer)

    os.makedirs(output_path, exist_ok=True)
    tokenizer.save(os.path.join(output_path, "tokenizer.json"))

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    hf_tokenizer.pad_token = core_tokens['pad_token']
    hf_tokenizer.bos_token = core_tokens['bos_token']
    hf_tokenizer.eos_token = core_tokens['eos_token']
    hf_tokenizer.unk_token = core_tokens['unk_token']

    hf_tokenizer.save_pretrained(output_path)
    print(f"BPE tokenizer saved to {output_path}")
    return hf_tokenizer


def main():
    parser = argparse.ArgumentParser(description="GUAC Tokenizer Training Script")

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
                       help='Force use of BPE tokenizer instead of SentencePiece')

    args = parser.parse_args()

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
        if args.force_bpe:
            tokenizer = train_bpe_tokenizer(training_file, output_path, args.vocab_size, core_tokens)
        else:
            tokenizer = train_sentencepiece_tokenizer(
                training_file, output_path, args.vocab_size,
                all_special_tokens, core_tokens
            )

            if tokenizer is None:
                print("Falling back to BPE tokenizer")
                tokenizer = train_bpe_tokenizer(training_file, output_path, args.vocab_size, core_tokens)

        print(f"\nTokenizer training complete!")
        print(f"Vocabulary size: {len(tokenizer):,}")
        print(f"Saved to: {output_path}")

    finally:
        if os.path.exists(training_file):
            os.remove(training_file)
            print(f"Cleaned up temporary file: {training_file}")


if __name__ == "__main__":
    main()
