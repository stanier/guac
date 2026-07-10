#!/usr/bin/env python3

"""
Tokenizer Training Script with SentencePiece Integration

This script trains tokenizers from datasets and saves them to configurable locations.
It now supports SentencePiece tokenization with special token integration from prompt_templates,
while maintaining backward compatibility with character-level tokenizers.

Features:
- SentencePiece BPE tokenization (default)
- Automatic integration of prompt template special tokens
- Arithmetic data generation for mathematical content
- Character-level tokenization (legacy support)
- Site configuration integration

Usage:
    python train_tokenizer.py [--datasets DATASET1,DATASET2] [--output-path PATH] [--tokenizer-type TYPE] [--vocab-size SIZE]

Examples:
    # Train SentencePiece tokenizer with default config (includes arithmetic data)
    python train_tokenizer.py

    # Train on specific datasets with custom vocab size
    python train_tokenizer.py --datasets wikitext-103 --vocab-size 16000

    # Train SentencePiece without arithmetic data
    python train_tokenizer.py --tokenizer-type sentencepiece --no-include-arithmetic

    # Train character-level tokenizer (legacy)
    python train_tokenizer.py --tokenizer-type character --vocab-size 64000

    # Train with custom SentencePiece parameters
    python train_tokenizer.py --character-coverage 0.999 --max-sentence-length 2048
"""

import os
import sys
import yaml
import argparse
from typing import Dict, Any, List, Optional
from collections import Counter

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import sentencepiece as spm
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, trainers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from datasets import load_from_disk, load_dataset

try:
    from .prompt_templates import create_prompt_tokenizer
    from scripts.training.legacy.sentencepiece_tokenizer import GUACSentencePieceTokenizer
    from scripts.training.legacy.train_sentencepiece import SentencePieceTrainer
except ImportError:
    from scripts.training.core.prompt_templates import create_prompt_tokenizer
    from scripts.training.legacy.sentencepiece_tokenizer import GUACSentencePieceTokenizer
    from scripts.training.legacy.train_sentencepiece import SentencePieceTrainer


def load_site_config():
    """Load site-specific configurations from YAML config file."""
    config_path = os.path.join(project_root, "site_config.yml")
    example_path = os.path.join(project_root, "site_config.example.yml")

    if not os.path.exists(config_path):
        if os.path.exists(example_path):
            print(f"Warning: {config_path} not found. Please copy {example_path} to {config_path} and customize it.")
            print("Using example config for now...")
            config_path = example_path
        else:
            raise FileNotFoundError(f"Neither {config_path} nor {example_path} found. Please create site configuration.")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Expand path variables in the config
    def expand_vars(text, var_dict):
        """Expand ${var} references in text using var_dict."""
        if not isinstance(text, str) or "${" not in text:
            return text

        result = text
        for var_name, var_value in var_dict.items():
            if isinstance(var_value, str):
                result = result.replace(f"${{{var_name}}}", var_value)
        return result

    # Build a flat dictionary of all variables for expansion
    all_vars = {}

    # First, collect base directory values
    dirs = config.get('directories', {})
    if 'scratch_dir' in dirs:
        all_vars['scratch_dir'] = dirs['scratch_dir']

    # Then expand base_dir references within each section
    for section_name, section_data in dirs.items():
        if isinstance(section_data, dict):
            if 'base_dir' in section_data:
                base_dir = expand_vars(section_data['base_dir'], all_vars)
                all_vars['base_dir'] = base_dir
                section_data['base_dir'] = base_dir

            # Expand all other paths in this section
            for key, value in section_data.items():
                if key != 'base_dir' and isinstance(value, str):
                    section_data[key] = expand_vars(value, all_vars)

    return config


def create_text_vocab_from_datasets(datasets: Dict[str, Any], vocab_size: int = 64000, min_frequency: int = 2) -> Dict[str, int]:
    """Creates a vocabulary from multiple datasets without special tokens."""
    # Collect all text from all datasets
    all_text = []
    for dataset in datasets.values():
        # Handle both DatasetDict and dict of IterableDataset
        if hasattr(dataset, 'keys'):
            for split in dataset.values():
                if hasattr(split, '__iter__') and hasattr(split, 'map'):
                    # This is a regular Dataset or IterableDataset
                    if hasattr(split, '__len__'):
                        # Regular Dataset
                        all_text.extend(split["text"])
                    else:
                        # IterableDataset - we need to iterate and collect
                        for item in split:
                            all_text.append(item["text"])
                            if len(all_text) >= 50000:  # Limit for memory and speed
                                break
                else:
                    # This might be a list or other iterable
                    all_text.extend(split)
        else:
            # Single dataset, not a dict
            all_text.extend(dataset["text"])

    print(f"Collecting vocabulary from {len(all_text)} total texts across {len(datasets)} datasets")

    # Count character frequency
    char_counts = Counter()
    for text in all_text:
        char_counts.update(text)

    # Get most common characters
    common_chars = [char for char, count in char_counts.most_common() if count >= min_frequency]

    # Limit vocabulary size
    if len(common_chars) > vocab_size:
        common_chars = common_chars[:vocab_size]

    return {char: idx for idx, char in enumerate(common_chars)}


def prepare_training_data_from_datasets(datasets: Dict[str, Any], output_path: str = "training_data.txt") -> str:
    """Prepare training data from multiple datasets for SentencePiece training."""
    import tempfile

    # Create temporary file for training data
    temp_dir = os.path.dirname(output_path)
    os.makedirs(temp_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as outf:
        total_lines = 0
        for dataset_name, dataset in datasets.items():
            print(f"Processing dataset: {dataset_name}")

            # Handle different dataset formats
            if hasattr(dataset, 'keys'):
                for split_name, split in dataset.items():
                    if hasattr(split, '__iter__') and hasattr(split, 'map'):
                        # This is a regular Dataset or IterableDataset
                        if hasattr(split, '__len__'):
                            # Regular Dataset
                            for text in split["text"]:
                                if text.strip():  # Skip empty lines
                                    outf.write(text.strip() + '\n')
                                    total_lines += 1
                        else:
                            # IterableDataset - iterate and collect
                            for item in split:
                                text = item.get("text", "")
                                if text.strip():
                                    outf.write(text.strip() + '\n')
                                    total_lines += 1
                                if total_lines >= 100000:  # Limit for memory
                                    break
                    else:
                        # This might be a list or other iterable
                        for text in split:
                            if text.strip():
                                outf.write(text.strip() + '\n')
                                total_lines += 1
            else:
                # Single dataset
                for text in dataset["text"]:
                    if text.strip():
                        outf.write(text.strip() + '\n')
                        total_lines += 1

    print(f"Prepared {total_lines} lines of training data in {output_path}")
    return output_path


def train_tokenizer_from_datasets(datasets: Dict[str, Any], tokenizer_type: str = "sentencepiece", vocab_size: int = 32000, **kwargs) -> GUACSentencePieceTokenizer:
    """Creates a tokenizer based on multiple datasets with prompt tokens added globally."""

    # Get prompt tokenizer for special tokens integration
    prompt_tokenizer = create_prompt_tokenizer()
    special_tokens = prompt_tokenizer.get_all_special_tokens()
    core_tokens = prompt_tokenizer.get_core_tokens()

    print(f"Integrating {len(special_tokens)} special prompt tokens with SentencePiece")

    if tokenizer_type == "sentencepiece" or tokenizer_type == "bpe":
        # Use SentencePiece for modern tokenization
        try:
            from .path_manager import get_path_manager
            temp_dir = kwargs.get('temp_dir', str(get_path_manager().get_temp_dir('tokenizer_training')))
        except ImportError:
            temp_dir = kwargs.get('temp_dir', './temp_tokenizer')
        os.makedirs(temp_dir, exist_ok=True)

        # Prepare training data
        training_data_path = os.path.join(temp_dir, "training_data.txt")
        prepare_training_data_from_datasets(datasets, training_data_path)

        # Create SentencePiece trainer with integrated special tokens
        sp_trainer = SentencePieceTrainer(output_dir=temp_dir)

        # Combine prompt special tokens with arithmetic tokens
        combined_special_tokens = list(special_tokens)

        # Update the trainer's arithmetic tokens to include prompt tokens
        original_arithmetic_tokens = sp_trainer.arithmetic_tokens.copy()
        sp_trainer.arithmetic_tokens = combined_special_tokens + original_arithmetic_tokens

        print(f"Training SentencePiece with {len(sp_trainer.arithmetic_tokens)} total special tokens")
        print(f"Special token breakdown:")
        print(f"  - Prompt tokens: {len(special_tokens)}")
        print(f"  - Arithmetic tokens: {len(original_arithmetic_tokens)}")

        # Train the SentencePiece model
        model_path = sp_trainer.train_tokenizer(
            input_file=training_data_path,
            model_prefix="guac_integrated",
            vocab_size=vocab_size,
            model_type='bpe' if tokenizer_type == "bpe" else 'bpe',
            character_coverage=kwargs.get('character_coverage', 0.9995),
            max_sentence_length=kwargs.get('max_sentence_length', 4096)
        )

        # Create GUAC tokenizer with core tokens properly set
        tokenizer = GUACSentencePieceTokenizer(
            vocab_file=model_path,
            pad_token=core_tokens['pad_token'],
            unk_token=core_tokens['unk_token'],
            bos_token=core_tokens['bos_token'],
            eos_token=core_tokens['eos_token'],
            mask_token=core_tokens.get('mask_token', '<mask>'),
            sep_token=core_tokens.get('sep_token', '<sep>'),
            cls_token=core_tokens.get('cls_token', '<cls>')
        )

        print(f"Successfully created SentencePiece tokenizer with vocab size: {tokenizer.vocab_size}")
        return tokenizer

    elif tokenizer_type == "character":
        # Fallback to original character-level implementation
        base_vocab = create_text_vocab_from_datasets(datasets, vocab_size, kwargs.get('min_frequency', 2))

        print(f"Adding {len(special_tokens)} special prompt tokens to vocabulary")

        # Reserve space for special tokens by reducing base vocab size if needed
        max_base_vocab_size = vocab_size - len(special_tokens)
        if len(base_vocab) > max_base_vocab_size:
            # Keep only the most frequent tokens
            base_vocab = {char: idx for idx, char in enumerate(list(base_vocab.keys())[:max_base_vocab_size])}

        # Create final vocabulary with special tokens first
        final_vocab = {}

        # Add special tokens first
        for idx, token in enumerate(special_tokens):
            final_vocab[token] = idx

        # Add base vocabulary tokens after special tokens
        for char, _ in base_vocab.items():
            final_vocab[char] = len(final_vocab)

        tokenizer = Tokenizer(WordLevel(vocab=final_vocab, unk_token=core_tokens['unk_token']))
        tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")
        tokenizer.normalizer = None

        # Add special tokens to the tokenizer
        tokenizer.add_special_tokens(list(core_tokens.values()))

        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token=core_tokens['bos_token'],
            eos_token=core_tokens['eos_token'],
            unk_token=core_tokens['unk_token'],
            pad_token=core_tokens['pad_token']
        )
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}. Supported: 'sentencepiece', 'bpe', 'character'")


def load_huggingface_dataset(dataset_config: Dict[str, Any], dataset_name: str) -> Any:
    """Load a HuggingFace dataset based on configuration."""
    config_name = dataset_config.get('config_name')
    streaming = dataset_config.get('streaming', False)
    max_articles = dataset_config.get('max_articles')

    print(f"Loading {dataset_name}: {dataset_config['dataset_name']}" +
          (f"/{config_name}" if config_name else "") +
          f" (streaming: {streaming}, max_articles: {max_articles})")

    # Use path manager for HuggingFace cache directory
    cache_dir = None
    try:
        from .path_manager import get_huggingface_cache_dir
        cache_dir = str(get_huggingface_cache_dir())
        print(f"  Using cache directory: {cache_dir}")
    except ImportError:
        pass

    # Load the dataset
    if config_name:
        dataset = load_dataset(dataset_config['dataset_name'], config_name, streaming=streaming, cache_dir=cache_dir)
    else:
        dataset = load_dataset(dataset_config['dataset_name'], streaming=streaming, cache_dir=cache_dir)

    # Limit articles if specified
    if max_articles and not streaming:
        # For non-streaming datasets, we can slice directly
        for split_name in dataset.keys():
            if len(dataset[split_name]) > max_articles:
                dataset[split_name] = dataset[split_name].select(range(max_articles))
                print(f"Limited {split_name} split to {max_articles} articles")

    elif max_articles and streaming:
        # For streaming datasets, we need to take only the first N examples
        from datasets import DatasetDict
        limited_dataset = {}
        for split_name in dataset.keys():
            limited_dataset[split_name] = dataset[split_name].take(max_articles)
            print(f"Will stream first {max_articles} articles from {split_name} split")
        dataset = DatasetDict(limited_dataset)

    return dataset


def load_datasets(site_config: Dict[str, Any], dataset_names: Optional[List[str]] = None, use_defaults: bool = True) -> Dict[str, Any]:
    """Load datasets specified in site configuration."""
    datasets = {}
    dataset_configs = site_config.get('dataset_configs', {})

    # For tokenizer training, we use pretraining datasets (raw text)
    hf_datasets = dataset_configs.get('pretraining_datasets', {})
    # Fallback to old structure for compatibility
    if not hf_datasets:
        hf_datasets = dataset_configs.get('huggingface_datasets', {})

    dirs = site_config['directories']

    # Determine which datasets to load
    if dataset_names is None and use_defaults:
        # Use default tokenizer datasets from config
        default_datasets = dataset_configs.get('default_tokenizer_datasets', ['wikitext-103'])
        # Filter out None values and empty lists
        if default_datasets:
            dataset_names = [name for name in default_datasets if name]

        if not dataset_names:
            # Fallback to available HuggingFace datasets if no defaults specified
            available_hf = list(hf_datasets.keys())
            if available_hf:
                dataset_names = available_hf[:2]  # Take first 2 datasets
                print(f"No default tokenizer datasets specified, using first available: {dataset_names}")
            else:
                dataset_names = ['wikitext-103']  # Ultimate fallback
                print(f"No datasets configured, using fallback: {dataset_names}")
        else:
            print(f"Using default tokenizer datasets: {dataset_names}")

    if dataset_names is None:
        # Fallback to available local datasets from config
        available_datasets = [key for key in dirs.get('datasets', {}).keys()
                            if key not in ['base_dir']]
        dataset_names = available_datasets

    # Load individual datasets
    for name in dataset_names:
        # First try to load from HuggingFace dataset configs
        if name in hf_datasets:
            try:
                datasets[name] = load_huggingface_dataset(hf_datasets[name], name)
                continue
            except Exception as e:
                print(f"Failed to load HuggingFace dataset {name}: {e}")

        # Fallback to local dataset paths
        dataset_path = dirs['datasets'].get(name)
        if dataset_path and os.path.exists(dataset_path):
            datasets[name] = load_from_disk(dataset_path)
            print(f"Loaded local dataset: {name} from {dataset_path}")
        else:
            print(f"Warning: Dataset {name} not found in HuggingFace configs or local path {dataset_path}")

    if not datasets:
        raise ValueError("No datasets could be loaded. Check your site configuration.")

    return datasets


def get_tokenizer_output_path(site_config: Dict[str, Any], custom_path: Optional[str] = None, tokenizer_type: str = "sentencepiece") -> str:
    """Determine the output path for the trained tokenizer using path manager."""
    if custom_path:
        return custom_path

    # Use the new path manager for consistent path handling
    try:
        from .path_manager import get_tokenizer_output_path as get_managed_path
        return str(get_managed_path(tokenizer_type))
    except ImportError:
        # Fallback to old method if path manager not available
        dirs = site_config['directories']
        tokenizer_dirs = dirs.get('tokenizers', {})

        # Try different fallback paths
        for key in ['char_arithmetic', 'base_dir']:
            if key in tokenizer_dirs:
                return tokenizer_dirs[key]

        # Ultimate fallback
        return os.path.join(dirs.get('scratch_dir', './scratch'), 'tokenizer')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train tokenizers from datasets using site configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ''
    )

    parser.add_argument(
        '--datasets',
        type=str,
        help='Comma-separated list of dataset names to use (default: from site config)'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        help='Custom output path for the trained tokenizer (overrides site config)'
    )

    parser.add_argument(
        '--tokenizer-type',
        type=str,
        default='sentencepiece',
        choices=['sentencepiece', 'bpe', 'character'],
        help='Type of tokenizer to train (default: sentencepiece)'
    )

    parser.add_argument(
        '--vocab-size',
        type=int,
        default=60000,
        help='Vocabulary size for the tokenizer (default: 60000)'
    )

    parser.add_argument(
        '--min-frequency',
        type=int,
        default=2,
        help='Minimum frequency for tokens to be included in vocabulary (default: 2)'
    )

    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets from site config and exit'
    )

    parser.add_argument(
        '--list-available',
        action='store_true',
        help='List all available datasets from site config and exit'
    )

    parser.add_argument(
        '--no-defaults',
        action='store_true',
        help='Disable default dataset usage from site config'
    )

    # SentencePiece specific arguments
    parser.add_argument(
        '--character-coverage',
        type=float,
        default=0.9995,
        help='Character coverage for SentencePiece training (default: 0.9995)'
    )

    parser.add_argument(
        '--max-sentence-length',
        type=int,
        default=4096,
        help='Maximum sentence length for SentencePiece training (default: 4096)'
    )

    parser.add_argument(
        '--temp-dir',
        type=str,
        default='./temp_tokenizer',
        help='Temporary directory for training files (default: ./temp_tokenizer)'
    )

    parser.add_argument(
        '--include-arithmetic',
        action='store_true',
        default=True,
        help='Include arithmetic training data generation (default: True)'
    )

    parser.add_argument(
        '--no-include-arithmetic',
        dest='include_arithmetic',
        action='store_false',
        help='Disable arithmetic training data generation'
    )

    parser.add_argument(
        '--arithmetic-samples',
        type=int,
        default=10000,
        help='Number of arithmetic samples to generate if --include-arithmetic (default: 10000)'
    )

    return parser.parse_args()


def main():
    """Main function for tokenizer training."""
    args = parse_args()

    # Load site configuration
    try:
        site_config = load_site_config()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Handle dataset listing
    if args.list_datasets:
        dirs = site_config['directories']
        available_datasets = [key for key in dirs.get('datasets', {}).keys()
                            if key not in ['base_dir']]
        print("Available local datasets from site configuration:")
        for dataset_name in available_datasets:
            dataset_path = dirs['datasets'][dataset_name]
            exists = "✓" if os.path.exists(dataset_path) else "✗"
            print(f"  {exists} {dataset_name}: {dataset_path}")
        return

    # Handle available datasets listing
    if args.list_available:
        dataset_configs = site_config.get('dataset_configs', {})

        print("=== PRETRAINING DATASETS (Raw Text) ===")
        pretraining_datasets = dataset_configs.get('pretraining_datasets', {})
        for name, config in pretraining_datasets.items():
            streaming_info = "(streaming)" if config.get('streaming') else "(download)"
            max_articles = config.get('max_articles', 'unlimited')
            print(f"  • {name}: {config['dataset_name']}")
            print(f"    Config: {config.get('config_name', 'default')}")
            print(f"    Mode: {streaming_info}, Max articles: {max_articles}")

        print("\n=== INSTRUCTION DATASETS (Chat/Templates) ===")
        instruction_datasets = dataset_configs.get('instruction_datasets', {})
        for name, config in instruction_datasets.items():
            streaming_info = "(streaming)" if config.get('streaming') else "(download)"
            max_articles = config.get('max_articles', 'unlimited')
            print(f"  • {name}: {config['dataset_name']}")
            print(f"    Config: {config.get('config_name', 'default')}")
            print(f"    Mode: {streaming_info}, Max articles: {max_articles}")

        print(f"\nDefault tokenizer datasets: {dataset_configs.get('default_tokenizer_datasets', [])}")
        print(f"Default pretrain datasets: {dataset_configs.get('default_pretrain_datasets', [])}")
        print(f"Default finetune datasets: {dataset_configs.get('default_finetune_datasets', [])}")
        return

    # Determine which datasets to load
    dataset_names = None
    if args.datasets:
        dataset_names = [name.strip() for name in args.datasets.split(',')]

    # Load datasets
    print("Loading datasets...")
    try:
        use_defaults = not args.no_defaults
        datasets = load_datasets(site_config, dataset_names=dataset_names, use_defaults=use_defaults)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Add arithmetic data if SentencePiece and requested
    if (args.tokenizer_type in ['sentencepiece', 'bpe']) and args.include_arithmetic:
        print(f"\nGenerating {args.arithmetic_samples} arithmetic training samples...")

        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)

        # Create temporary SentencePiece trainer just for arithmetic data generation
        sp_trainer = SentencePieceTrainer(output_dir=temp_dir)
        arithmetic_file = sp_trainer.create_arithmetic_training_data(
            output_file="arithmetic_training.txt",
            num_samples=args.arithmetic_samples
        )

        # Add arithmetic data to datasets for training
        print(f"Generated arithmetic training data: {arithmetic_file}")
        with open(arithmetic_file, 'r', encoding='utf-8') as f:
            arithmetic_texts = [line.strip() for line in f if line.strip()]

        # Create a simple dataset-like structure for arithmetic data
        datasets['arithmetic_generated'] = {'text': arithmetic_texts}
        print(f"Added {len(arithmetic_texts)} arithmetic examples to training data")

    # Train tokenizer
    print(f"\nTraining {args.tokenizer_type} tokenizer with vocab size {args.vocab_size}...")
    tokenizer = train_tokenizer_from_datasets(
        datasets,
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        temp_dir=args.temp_dir,
        character_coverage=args.character_coverage,
        max_sentence_length=args.max_sentence_length
    )

    # Determine output path
    output_path = get_tokenizer_output_path(site_config, args.output_path, args.tokenizer_type)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save tokenizer
    if isinstance(tokenizer, GUACSentencePieceTokenizer):
        tokenizer.save_pretrained(output_path)
        print(f"\nSentencePiece tokenizer saved to: {output_path}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")

        # Also save the raw SentencePiece model for direct loading
        sp_model_path = os.path.join(output_path, "tokenizer.model")
        if os.path.exists(sp_model_path):
            print(f"SentencePiece model file: {sp_model_path}")

        print(f"\nSpecial tokens integrated:")
        print(f"  - Pad token: {tokenizer.pad_token}")
        print(f"  - Unknown token: {tokenizer.unk_token}")
        print(f"  - BOS token: {tokenizer.bos_token}")
        print(f"  - EOS token: {tokenizer.eos_token}")

        # Show arithmetic token integration
        arithmetic_ids = tokenizer.get_arithmetic_token_ids()
        print(f"  - Arithmetic tokens: {len(arithmetic_ids)} tokens integrated")
    else:
        tokenizer.save_pretrained(output_path)
        print(f"\nCharacter-level tokenizer saved to: {output_path}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Print information for integration with other scripts
    print(f"\nTo use this tokenizer in training scripts:")
    print(f"  Update model_configs.yaml:")
    print(f"    tokenizer_type: \"{args.tokenizer_type}\"")
    print(f"    tokenizer_model_path: \"{output_path}\"")
    print(f"    vocab_size: {tokenizer.vocab_size}")
    print(f"\nOr set the tokenizer path in site_config.yml:")
    print(f"  directories:")
    print(f"    tokenizers:")
    print(f"      custom_trained: \"{output_path}\"")


if __name__ == "__main__":
    main()