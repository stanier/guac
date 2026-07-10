#!/usr/bin/env python3
"""
Test script for SentencePiece integration with Guac architecture.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.training.train_sentencepiece import SentencePieceTrainer


def test_sentencepiece_training():
    """Test SentencePiece tokenizer training."""

    print("Testing SentencePiece Tokenizer Training")
    print("=" * 50)

    # Create trainer
    # Use path manager for test tokenizer directory
    try:
        from scripts.training.core.path_manager import get_path_manager
        pm = get_path_manager()
        test_tokenizer_dir = str(pm.get_temp_dir("test_tokenizers"))
    except ImportError:
        test_tokenizer_dir = "test_tokenizers"

    trainer = SentencePieceTrainer(output_dir=test_tokenizer_dir)

    # Generate arithmetic training data
    print("Generating arithmetic training data...")
    arithmetic_file = trainer.create_arithmetic_training_data(
        output_file="test_arithmetic.txt",
        num_samples=1000
    )

    # Train a small tokenizer for testing
    print("Training SentencePiece tokenizer...")
    model_path = trainer.train_tokenizer(
        input_file=arithmetic_file,
        model_prefix="test_guac_sp",
        vocab_size=1000,  # Small vocab for testing
        model_type='bpe',
        character_coverage=0.9995
    )

    print(f"✅ Tokenizer training completed!")
    print(f"Model saved to: {model_path}")

    return model_path


def test_tokenizer_wrapper():
    """Test the tokenizer wrapper functionality."""

    print("\nTesting Tokenizer Wrapper")
    print("=" * 50)

    try:
        from scripts.training.sentencepiece_tokenizer import load_guac_tokenizer

        # Use the trained tokenizer from previous test
        model_path = "test_tokenizers/test_guac_sp.model"

        if not os.path.exists(model_path):
            print("Training tokenizer first...")
            model_path = test_sentencepiece_training()

        print(f"Loading tokenizer from: {model_path}")
        tokenizer = load_guac_tokenizer(model_path)

        # Test arithmetic expressions
        test_expressions = [
            "2 + 3 = 5",
            "Calculate 15 * 7",
            "What is sqrt(144)?",
            "Solve for x: 2x + 5 = 13",
            "The sum of 1 through 10 is 55",
        ]

        print("\nTesting arithmetic expression tokenization:")
        for expr in test_expressions:
            tokens = tokenizer.encode_arithmetic_expression(expr)
            decoded = tokenizer.decode_arithmetic_expression(tokens)

            print(f"Original: {expr}")
            print(f"Tokens:   {tokens}")
            print(f"Decoded:  {decoded}")
            print("-" * 30)

        # Test special token IDs
        arithmetic_ids = tokenizer.get_arithmetic_token_ids()
        print(f"Arithmetic token IDs found: {len(arithmetic_ids)}")
        for token, token_id in list(arithmetic_ids.items())[:5]:
            print(f"  {token}: {token_id}")

        print("✅ Tokenizer wrapper test completed!")

    except Exception as e:
        print(f"❌ Tokenizer wrapper test failed: {e}")
        return False

    return True


def test_model_config_integration():
    """Test integration with model configurations."""

    print("\nTesting Model Config Integration")
    print("=" * 50)

    try:
        import yaml

        # Load model configs
        with open("model_configs.yaml", 'r') as f:
            configs = yaml.safe_load(f)

        # Check if SentencePiece settings are present
        defaults = configs.get('defaults', {})
        tokenizer_type = defaults.get('tokenizer_type')
        tokenizer_path = defaults.get('tokenizer_model_path')
        vocab_size = defaults.get('vocab_size')

        print(f"Tokenizer type: {tokenizer_type}")
        print(f"Tokenizer path: {tokenizer_path}")
        print(f"Vocab size: {vocab_size}")

        # Check model-specific configs
        model_configs = configs.get('model_training_configs', {})
        for model_name, config in model_configs.items():
            if 'tokenizer_model_path' in config or 'vocab_size' in config:
                print(f"Model '{model_name}' has SentencePiece config:")
                if 'vocab_size' in config:
                    print(f"  Vocab size: {config['vocab_size']}")
                if 'tokenizer_model_path' in config:
                    print(f"  Tokenizer path: {config['tokenizer_model_path']}")

        print("✅ Model config integration check completed!")

    except Exception as e:
        print(f"❌ Model config integration test failed: {e}")
        return False

    return True


def main():
    """Run all tests."""

    print("Guac SentencePiece Integration Tests")
    print("=" * 60)

    # Test 1: Train tokenizer
    try:
        test_sentencepiece_training()
    except Exception as e:
        print(f"❌ Tokenizer training test failed: {e}")

    # Test 2: Test wrapper
    try:
        test_tokenizer_wrapper()
    except Exception as e:
        print(f"❌ Tokenizer wrapper test failed: {e}")

    # Test 3: Test config integration
    try:
        test_model_config_integration()
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("\nNext steps:")
    print("1. Run: python scripts/training/train_sentencepiece.py --generate_arithmetic")
    print("2. Run: python scripts/training/train_with_sentencepiece.py --model guac-3b")
    print("3. Monitor training with --use_wandb flag")


if __name__ == "__main__":
    main()