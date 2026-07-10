#!/usr/bin/env python3
"""
SentencePiece tokenizer wrapper for Guac mixed layer architecture.

This module provides a unified interface for SentencePiece tokenization
that integrates seamlessly with the Guac training pipeline and supports
arithmetic and mathematical content tokenization.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple
import sentencepiece as spm
from transformers import PreTrainedTokenizer
import torch

logger = logging.getLogger(__name__)


class GUACSentencePieceTokenizer(PreTrainedTokenizer):
    """
    Guac-specific SentencePiece tokenizer that extends HuggingFace's PreTrainedTokenizer.

    This tokenizer is optimized for arithmetic and mathematical content while
    maintaining compatibility with the transformers library ecosystem.
    """

    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        do_lower_case: bool = False,
        remove_space: bool = True,
        keep_accents: bool = True,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        mask_token: str = "<mask>",
        sep_token: str = "<sep>",
        cls_token: str = "<cls>",
        **kwargs
    ):
        """
        Initialize the Guac SentencePiece tokenizer.

        Args:
            vocab_file: Path to the SentencePiece model file
            do_lower_case: Whether to lowercase the input
            remove_space: Whether to remove spaces
            keep_accents: Whether to keep accents
            sp_model_kwargs: Additional SentencePiece model arguments
            pad_token: Padding token
            unk_token: Unknown token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            mask_token: Mask token for MLM
            sep_token: Separator token
            cls_token: Classification token
            **kwargs: Additional tokenizer arguments
        """

        self.vocab_file = vocab_file
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.sp_model_kwargs = sp_model_kwargs or {}

        # Initialize SentencePiece processor
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(vocab_file)

        # Special tokens for arithmetic operations
        self.arithmetic_tokens = {
            '<num>': 'number placeholder',
            '<eq>': 'equals',
            '<add>': 'addition',
            '<sub>': 'subtraction',
            '<mul>': 'multiplication',
            '<div>': 'division',
            '<pow>': 'power',
            '<sqrt>': 'square root',
            '<log>': 'logarithm',
            '<sin>': 'sine',
            '<cos>': 'cosine',
            '<tan>': 'tangent',
            '<sum>': 'summation',
            '<prod>': 'product',
            '<avg>': 'average',
            '<max>': 'maximum',
            '<min>': 'minimum',
            '<lparen>': 'left parenthesis',
            '<rparen>': 'right parenthesis',
            '<lbracket>': 'left bracket',
            '<rbracket>': 'right bracket',
            '<pi>': 'pi constant',
            '<e>': 'euler constant',
            '<inf>': 'infinity',
            '<nan>': 'not a number',
            '<lt>': 'less than',
            '<gt>': 'greater than',
            '<le>': 'less than or equal',
            '<ge>': 'greater than or equal',
            '<ne>': 'not equal',
        }

        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            sep_token=sep_token,
            cls_token=cls_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary."""
        vocab = {}
        for i in range(self.vocab_size):
            piece = self.sp_model.id_to_piece(i)
            vocab[piece] = i
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string using SentencePiece.

        Args:
            text: Input text to tokenize

        Returns:
            List of token strings
        """
        if self.do_lower_case:
            text = text.lower()

        # Preprocess arithmetic expressions
        text = self._preprocess_arithmetic(text)

        # Tokenize using SentencePiece
        pieces = self.sp_model.encode_as_pieces(text)

        return pieces

    def _preprocess_arithmetic(self, text: str) -> str:
        """
        Preprocess text to better handle arithmetic expressions.

        Args:
            text: Input text

        Returns:
            Preprocessed text with arithmetic tokens
        """
        import re

        # Replace common mathematical symbols with special tokens
        replacements = {
            r'\+': ' <add> ',
            r'-': ' <sub> ',
            r'\*': ' <mul> ',
            r'/': ' <div> ',
            r'\*\*': ' <pow> ',
            r'=': ' <eq> ',
            r'\(': ' <lparen> ',
            r'\)': ' <rparen> ',
            r'\[': ' <lbracket> ',
            r'\]': ' <rbracket> ',
            r'<': ' <lt> ',
            r'>': ' <gt> ',
            r'<=': ' <le> ',
            r'>=': ' <ge> ',
            r'!=': ' <ne> ',
            r'π': ' <pi> ',
            r'∞': ' <inf> ',
        }

        # Apply replacements
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        # Handle numbers with special token
        text = re.sub(r'\b\d+(?:\.\d+)?\b', r'<num>', text)

        # Handle mathematical functions
        math_functions = {
            r'\bsqrt\b': '<sqrt>',
            r'\blog\b': '<log>',
            r'\bsin\b': '<sin>',
            r'\bcos\b': '<cos>',
            r'\btan\b': '<tan>',
            r'\bsum\b': '<sum>',
            r'\bmax\b': '<max>',
            r'\bmin\b': '<min>',
        }

        for pattern, replacement in math_functions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token string to its corresponding ID."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its corresponding token string."""
        return self.sp_model.id_to_piece(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens back to a string.

        Args:
            tokens: List of token strings

        Returns:
            Decoded string
        """
        # Handle special arithmetic tokens in post-processing
        decoded = self.sp_model.decode_pieces(tokens)
        decoded = self._postprocess_arithmetic(decoded)
        return decoded

    def _postprocess_arithmetic(self, text: str) -> str:
        """
        Post-process decoded text to restore arithmetic expressions.

        Args:
            text: Decoded text

        Returns:
            Post-processed text with restored arithmetic symbols
        """
        # Reverse the preprocessing replacements
        replacements = {
            '<add>': '+',
            '<sub>': '-',
            '<mul>': '*',
            '<div>': '/',
            '<pow>': '**',
            '<eq>': '=',
            '<lparen>': '(',
            '<rparen>': ')',
            '<lbracket>': '[',
            '<rbracket>': ']',
            '<lt>': '<',
            '<gt>': '>',
            '<le>': '<=',
            '<ge>': '>=',
            '<ne>': '!=',
            '<pi>': 'π',
            '<inf>': '∞',
            '<sqrt>': 'sqrt',
            '<log>': 'log',
            '<sin>': 'sin',
            '<cos>': 'cos',
            '<tan>': 'tan',
            '<sum>': 'sum',
            '<max>': 'max',
            '<min>': 'min',
        }

        for token, symbol in replacements.items():
            text = text.replace(token, symbol)

        # Clean up extra spaces around operators
        import re
        text = re.sub(r'\s*([+\-*/=<>()[\]])\s*', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs by adding special tokens.

        Args:
            token_ids_0: First sequence token IDs
            token_ids_1: Optional second sequence token IDs

        Returns:
            List of token IDs with special tokens added
        """
        bos_token_id = [self.bos_token_id] if self.bos_token_id is not None else []
        eos_token_id = [self.eos_token_id] if self.eos_token_id is not None else []
        sep_token_id = [self.sep_token_id] if self.sep_token_id is not None else []

        if token_ids_1 is None:
            return bos_token_id + token_ids_0 + eos_token_id
        else:
            return bos_token_id + token_ids_0 + sep_token_id + token_ids_1 + eos_token_id

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Get special tokens mask.

        Args:
            token_ids_0: First sequence token IDs
            token_ids_1: Optional second sequence token IDs
            already_has_special_tokens: Whether tokens already have special tokens

        Returns:
            List indicating special token positions (1 for special, 0 for regular)
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.bos_token_id is not None else []
        eos_token_id = [1] if self.eos_token_id is not None else []
        sep_token_id = [1] if self.sep_token_id is not None else []

        if token_ids_1 is None:
            return bos_token_id + [0] * len(token_ids_0) + eos_token_id
        else:
            return bos_token_id + [0] * len(token_ids_0) + sep_token_id + [0] * len(token_ids_1) + eos_token_id

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer vocabulary.

        Args:
            save_directory: Directory to save vocabulary
            filename_prefix: Optional filename prefix

        Returns:
            Tuple of saved file paths
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            import shutil
            shutil.copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def encode_arithmetic_expression(self, expression: str) -> List[int]:
        """
        Encode an arithmetic expression with special handling.

        Args:
            expression: Arithmetic expression string

        Returns:
            List of token IDs
        """
        # Preprocess the expression
        processed = self._preprocess_arithmetic(expression)

        # Tokenize and convert to IDs
        tokens = self._tokenize(processed)
        token_ids = self.convert_tokens_to_ids(tokens)

        return token_ids

    def decode_arithmetic_expression(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to an arithmetic expression.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded arithmetic expression
        """
        # Convert IDs to tokens
        tokens = self.convert_ids_to_tokens(token_ids)

        # Decode tokens to string
        decoded = self.convert_tokens_to_string(tokens)

        return decoded

    def get_arithmetic_token_ids(self) -> Dict[str, int]:
        """
        Get the token IDs for arithmetic special tokens.

        Returns:
            Dictionary mapping arithmetic tokens to their IDs
        """
        arithmetic_ids = {}
        for token in self.arithmetic_tokens:
            token_id = self._convert_token_to_id(token)
            if token_id != self.unk_token_id:  # Only include tokens that exist in vocabulary
                arithmetic_ids[token] = token_id

        return arithmetic_ids

    def create_attention_mask_for_arithmetic(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask with special handling for arithmetic tokens.

        Args:
            input_ids: Input token IDs tensor

        Returns:
            Attention mask tensor
        """
        # Standard attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token_id).long()

        # Could add special logic here for arithmetic tokens if needed
        # For example, giving higher attention weights to operators

        return attention_mask


def load_guac_tokenizer(model_path: str, **kwargs) -> GUACSentencePieceTokenizer:
    """
    Load a Guac SentencePiece tokenizer.

    Args:
        model_path: Path to the SentencePiece model file
        **kwargs: Additional tokenizer arguments

    Returns:
        Loaded Guac tokenizer
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

    tokenizer = GUACSentencePieceTokenizer(vocab_file=model_path, **kwargs)

    logger.info(f"Loaded Guac tokenizer from {model_path}")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    logger.info(f"Special tokens: {list(tokenizer.special_tokens_map.keys())}")

    return tokenizer


