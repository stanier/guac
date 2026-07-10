#!/usr/bin/env python3
"""
SentencePiece tokenizer training script for Guac architecture.

This script trains a SentencePiece tokenizer optimized for the Guac mixed layer
architecture, with support for arithmetic operations and mathematical content.
"""

import os
import argparse
import sentencepiece as spm
from pathlib import Path
from typing import List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentencePieceTrainer:
    """
    Trains SentencePiece tokenizers optimized for Guac arithmetic models.
    """

    def __init__(self, output_dir: Optional[str] = None):
        # Use path manager for default output directory
        if output_dir is None:
            try:
                # Add project root to path for imports
                import sys
                project_root = Path(__file__).parent.parent.parent.parent
                sys.path.append(str(project_root))

                from scripts.training.core.path_manager import get_tokenizer_output_path
                output_dir = str(get_tokenizer_output_path("sentencepiece", "default"))
            except ImportError:
                # Fallback to old behavior if path manager not available
                output_dir = "tokenizers"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Special tokens for arithmetic and mathematical operations
        # Note: Standard control tokens (pad, unk, bos, eos) are handled separately
        self.arithmetic_tokens = [
            '<mask>', '<sep>', '<cls>',       # Additional utility tokens
            # Arithmetic-specific tokens
            '<num>', '<eq>', '<add>', '<sub>', '<mul>', '<div>',
            '<pow>', '<sqrt>', '<log>', '<sin>', '<cos>', '<tan>',
            '<sum>', '<prod>', '<avg>', '<max>', '<min>',
            # Parentheses and brackets
            '<lparen>', '<rparen>', '<lbracket>', '<rbracket>',
            # Mathematical constants
            '<pi>', '<e>', '<inf>', '<nan>',
            # Comparison operators
            '<lt>', '<gt>', '<le>', '<ge>', '<ne>',
        ]

    def prepare_training_data(
        self,
        input_files: List[str],
        output_file: str = "training_data.txt",
        max_lines: Optional[int] = None
    ) -> str:
        """
        Prepare and combine training data from multiple sources.

        Args:
            input_files: List of input file paths
            output_file: Output file for combined data
            max_lines: Maximum number of lines to process

        Returns:
            Path to the prepared training data file
        """
        output_path = self.output_dir / output_file

        logger.info(f"Preparing training data from {len(input_files)} files...")

        lines_written = 0
        with open(output_path, 'w', encoding='utf-8') as outf:
            for input_file in input_files:
                if not os.path.exists(input_file):
                    logger.warning(f"Input file not found: {input_file}")
                    continue

                logger.info(f"Processing: {input_file}")
                with open(input_file, 'r', encoding='utf-8') as inf:
                    for line in inf:
                        if max_lines and lines_written >= max_lines:
                            break

                        line = line.strip()
                        if line:  # Skip empty lines
                            outf.write(line + '\n')
                            lines_written += 1

                if max_lines and lines_written >= max_lines:
                    break

        logger.info(f"Prepared {lines_written} lines of training data")
        return str(output_path)

    def train_tokenizer(
        self,
        input_file: str,
        model_prefix: str,
        vocab_size: int = 32000,
        model_type: str = 'bpe',
        character_coverage: float = 0.9995,
        max_sentence_length: int = 4096,
        split_by_whitespace: bool = True,
        byte_fallback: bool = True,
        normalization_rule_name: str = 'nmt_nfkc_cf',
        add_dummy_prefix: bool = True,
        remove_extra_whitespaces: bool = True,
        hard_vocab_limit: bool = False,
        use_all_vocab: bool = False,
        **kwargs
    ) -> str:
        """
        Train a SentencePiece tokenizer.

        Args:
            input_file: Input training data file
            model_prefix: Prefix for output model files
            vocab_size: Target vocabulary size
            model_type: Model type ('bpe', 'unigram', 'char', 'word')
            character_coverage: Character coverage for vocabulary
            max_sentence_length: Maximum sentence length
            split_by_whitespace: Whether to split by whitespace
            byte_fallback: Enable byte fallback for unknown characters
            normalization_rule_name: Text normalization rule
            add_dummy_prefix: Add dummy prefix for subword regularization
            remove_extra_whitespaces: Remove extra whitespaces
            hard_vocab_limit: Enforce hard vocabulary size limit
            use_all_vocab: Use all vocabulary slots
            **kwargs: Additional SentencePiece arguments

        Returns:
            Path to the trained model file
        """

        model_path = self.output_dir / f"{model_prefix}.model"
        vocab_path = self.output_dir / f"{model_prefix}.vocab"

        # Prepare user-defined symbols (special tokens)
        user_defined_symbols = ','.join(self.arithmetic_tokens)

        # SentencePiece training arguments
        train_args = {
            'input': input_file,
            'model_prefix': str(self.output_dir / model_prefix),
            'vocab_size': vocab_size,
            'model_type': model_type,
            'character_coverage': character_coverage,
            'max_sentence_length': max_sentence_length,
            'split_by_whitespace': split_by_whitespace,
            'byte_fallback': byte_fallback,
            'normalization_rule_name': normalization_rule_name,
            'add_dummy_prefix': add_dummy_prefix,
            'remove_extra_whitespaces': remove_extra_whitespaces,
            'hard_vocab_limit': hard_vocab_limit,
            'use_all_vocab': use_all_vocab,
            'user_defined_symbols': user_defined_symbols,
            'pad_id': 0,
            'unk_id': 1,
            'bos_id': 2,
            'eos_id': 3,
            'pad_piece': '<pad>',
            'unk_piece': '<unk>',
            'bos_piece': '<s>',
            'eos_piece': '</s>',
        }

        # Add any additional arguments
        train_args.update(kwargs)

        logger.info(f"Training SentencePiece tokenizer with {vocab_size} vocabulary size...")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Character coverage: {character_coverage}")

        # Train the tokenizer
        spm.SentencePieceTrainer.train(**train_args)

        logger.info(f"Tokenizer saved to: {model_path}")
        logger.info(f"Vocabulary saved to: {vocab_path}")

        # Load and test the trained tokenizer
        self.test_tokenizer(str(model_path))

        return str(model_path)

    def test_tokenizer(self, model_path: str):
        """Test the trained tokenizer with sample inputs."""

        logger.info("Testing trained tokenizer...")

        sp = spm.SentencePieceProcessor()
        sp.load(model_path)

        # Test samples including arithmetic expressions
        test_samples = [
            "Hello world! This is a test sentence.",
            "2 + 3 = 5",
            "Calculate the square root of 144.",
            "The result of 15 * 7 is 105.",
            "Find the derivative of x^2 + 3x - 5.",
            "π ≈ 3.14159265359",
            "Solve for x: 2x + 5 = 13",
            "The sum of 1 through 10 is 55.",
        ]

        logger.info("Sample tokenizations:")
        for sample in test_samples:
            tokens = sp.encode_as_pieces(sample)
            ids = sp.encode_as_ids(sample)
            decoded = sp.decode_pieces(tokens)

            logger.info(f"Input: {sample}")
            logger.info(f"Tokens: {tokens}")
            logger.info(f"IDs: {ids}")
            logger.info(f"Decoded: {decoded}")
            logger.info("-" * 50)

        logger.info(f"Vocabulary size: {sp.get_piece_size()}")
        logger.info(f"Special token IDs:")
        for token in self.arithmetic_tokens[:10]:  # Show first 10 special tokens
            if sp.piece_to_id(token) != sp.unk_id():
                logger.info(f"  {token}: {sp.piece_to_id(token)}")

    def create_arithmetic_training_data(self, output_file: str = "arithmetic_data.txt", num_samples: int = 100000):
        """
        Generate synthetic arithmetic training data.

        Args:
            output_file: Output file path
            num_samples: Number of arithmetic samples to generate
        """
        import random
        import math

        output_path = self.output_dir / output_file

        logger.info(f"Generating {num_samples} arithmetic training samples...")

        operations = ['+', '-', '*', '/', '**', '%']
        functions = ['sqrt', 'log', 'sin', 'cos', 'tan', 'abs']

        with open(output_path, 'w', encoding='utf-8') as f:
            for _ in range(num_samples):
                sample_type = random.choice(['basic_arithmetic', 'functions', 'equations', 'word_problems'])

                if sample_type == 'basic_arithmetic':
                    a = random.randint(1, 1000)
                    b = random.randint(1, 1000)
                    op = random.choice(operations[:4])  # +, -, *, /

                    if op == '+':
                        result = a + b
                    elif op == '-':
                        result = a - b
                    elif op == '*':
                        result = a * b
                    elif op == '/':
                        result = round(a / b, 2) if b != 0 else 'undefined'

                    f.write(f"{a} {op} {b} = {result}\n")

                elif sample_type == 'functions':
                    x = random.uniform(0.1, 10)
                    func = random.choice(functions)

                    if func == 'sqrt':
                        result = round(math.sqrt(x), 4)
                        f.write(f"sqrt({x:.2f}) = {result}\n")
                    elif func == 'log':
                        result = round(math.log(x), 4)
                        f.write(f"log({x:.2f}) = {result}\n")
                    elif func == 'sin':
                        result = round(math.sin(x), 4)
                        f.write(f"sin({x:.2f}) = {result}\n")
                    elif func == 'cos':
                        result = round(math.cos(x), 4)
                        f.write(f"cos({x:.2f}) = {result}\n")
                    elif func == 'tan':
                        result = round(math.tan(x), 4)
                        f.write(f"tan({x:.2f}) = {result}\n")
                    elif func == 'abs':
                        x = random.uniform(-10, 10)
                        result = abs(x)
                        f.write(f"abs({x:.2f}) = {result:.2f}\n")

                elif sample_type == 'equations':
                    a = random.randint(1, 20)
                    b = random.randint(1, 50)
                    c = random.randint(1, 100)
                    x = (c - b) / a if a != 0 else 0
                    f.write(f"Solve for x: {a}x + {b} = {c}. Answer: x = {x:.2f}\n")

                elif sample_type == 'word_problems':
                    problems = [
                        f"If you have {random.randint(10, 50)} apples and give away {random.randint(5, 20)}, how many are left?",
                        f"A rectangle has width {random.randint(5, 15)} and height {random.randint(8, 20)}. What is its area?",
                        f"Calculate the sum of numbers from 1 to {random.randint(10, 25)}.",
                        f"What is {random.randint(2, 9)} percent of {random.randint(100, 500)}?",
                    ]
                    f.write(random.choice(problems) + "\n")

        logger.info(f"Generated arithmetic training data: {output_path}")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer for Guac")
    parser.add_argument('--input_files', nargs='+', help='Input training data files')
    parser.add_argument('--output_dir', default=None, help='Output directory (default: uses path manager)')
    parser.add_argument('--model_prefix', default='guac_sp', help='Model prefix')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--model_type', default='bpe', choices=['bpe', 'unigram', 'char', 'word'], help='Model type')
    parser.add_argument('--character_coverage', type=float, default=0.9995, help='Character coverage')
    parser.add_argument('--max_lines', type=int, help='Maximum lines to process')
    parser.add_argument('--generate_arithmetic', action='store_true', help='Generate synthetic arithmetic data')
    parser.add_argument('--arithmetic_samples', type=int, default=100000, help='Number of arithmetic samples')

    args = parser.parse_args()

    trainer = SentencePieceTrainer(args.output_dir)

    input_files = args.input_files or []

    # Generate arithmetic data if requested
    if args.generate_arithmetic:
        arithmetic_file = trainer.create_arithmetic_training_data(
            num_samples=args.arithmetic_samples
        )
        input_files.append(arithmetic_file)

    if not input_files:
        logger.error("No input files provided. Use --input_files or --generate_arithmetic")
        return

    # Prepare training data
    training_data = trainer.prepare_training_data(
        input_files=input_files,
        max_lines=args.max_lines
    )

    # Train tokenizer
    model_path = trainer.train_tokenizer(
        input_file=training_data,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage
    )

    logger.info(f"Training complete! Model saved to: {model_path}")


if __name__ == "__main__":
    main()