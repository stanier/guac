#!/usr/bin/env python3
"""
Example training script for GUAC mixed layer architecture with SentencePiece tokenization.

This script demonstrates how to integrate SentencePiece tokenization with the GUAC
mixed layer models for arithmetic and mathematical content training.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import Dataset, load_dataset
import wandb

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.training.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM
from scripts.training.sentencepiece_tokenizer import GUACSentencePieceTokenizer, load_guac_tokenizer
from scripts.training.train_sentencepiece import SentencePieceTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GUACSentencePieceTrainer:
    """
    Trainer class for GUAC models with SentencePiece tokenization.
    """

    def __init__(self, config_path: str, model_name: str):
        """
        Initialize the trainer.

        Args:
            config_path: Path to model configuration YAML file
            model_name: Name of the model configuration to use
        """
        self.config_path = config_path
        self.model_name = model_name
        self.config = self._load_config()
        self.tokenizer = None
        self.model = None

    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            configs = yaml.safe_load(f)

        # Get default config
        defaults = configs.get('defaults', {})

        # Get model-specific config
        model_configs = configs.get('model_training_configs', {})
        if self.model_name not in model_configs:
            raise ValueError(f"Model '{self.model_name}' not found in config file")

        model_config = model_configs[self.model_name]

        # Merge defaults with model-specific config
        config = {**defaults, **model_config}

        logger.info(f"Loaded configuration for model: {self.model_name}")
        logger.info(f"Architecture: {config.get('architecture', 'basic_transformer')}")
        logger.info(f"Hidden size: {config.get('hidden_size', 512)}")
        logger.info(f"Layers: {config.get('num_hidden_layers', 6)}")

        return config

    def setup_tokenizer(self, force_retrain: bool = False) -> GUACSentencePieceTokenizer:
        """
        Setup or train the SentencePiece tokenizer.

        Args:
            force_retrain: Whether to force retraining the tokenizer

        Returns:
            Configured GUAC SentencePiece tokenizer
        """
        tokenizer_path = self.config.get('tokenizer_model_path', 'tokenizers/guac_sp.model')
        vocab_size = self.config.get('vocab_size', 32000)

        # Check if tokenizer exists and force_retrain is False
        if os.path.exists(tokenizer_path) and not force_retrain:
            logger.info(f"Loading existing tokenizer from: {tokenizer_path}")
            self.tokenizer = load_guac_tokenizer(tokenizer_path)
            return self.tokenizer

        # Train new tokenizer
        logger.info(f"Training new SentencePiece tokenizer with vocab size: {vocab_size}")

        # Use path manager for tokenizer output directory
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            sys.path.append(str(project_root))
            from scripts.training.core.path_manager import get_tokenizer_output_path
            tokenizer_dir = str(get_tokenizer_output_path("sentencepiece", "default"))
        except ImportError:
            tokenizer_dir = "tokenizers"

        trainer = SentencePieceTrainer(output_dir=tokenizer_dir)

        # Generate arithmetic training data
        arithmetic_file = trainer.create_arithmetic_training_data(
            output_file="arithmetic_training.txt",
            num_samples=50000
        )

        # You can add additional training files here
        input_files = [arithmetic_file]

        # Prepare training data
        training_data = trainer.prepare_training_data(input_files)

        # Train tokenizer
        model_prefix = Path(tokenizer_path).stem
        model_path = trainer.train_tokenizer(
            input_file=training_data,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=0.9995
        )

        # Load the trained tokenizer
        self.tokenizer = load_guac_tokenizer(model_path)
        logger.info(f"Trained and loaded new tokenizer: {model_path}")

        return self.tokenizer

    def setup_model(self) -> MixedLayerForCausalLM:
        """
        Setup the GUAC mixed layer model.

        Returns:
            Configured model
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be setup before model")

        # Create model configuration
        model_config = MixedLayerConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=self.config.get('hidden_size', 512),
            num_hidden_layers=self.config.get('num_hidden_layers', 6),
            num_attention_heads=self.config.get('num_attention_heads', 8),
            num_kv_heads=self.config.get('num_kv_heads', None),
            intermediate_size=self.config.get('hidden_size', 512) * self.config.get('intermediate_layers_fac', 4),
            max_position_embeddings=self.config.get('max_position_embeddings', 4096),
            layer_types=self.config.get('layer_types', None),
            # MoH parameters
            moh_top_k=self.config.get('moh_top_k', None),
            moh_temperature=self.config.get('moh_temperature', 1.0),
            moh_noise_std=self.config.get('moh_noise_std', 0.1),
            # Other parameters
            hidden_dropout_prob=self.config.get('hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=self.config.get('attention_probs_dropout_prob', 0.1),
        )

        # Create model
        self.model = MixedLayerForCausalLM(model_config)

        logger.info(f"Created model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logger.info(f"Layer types: {model_config.layer_types}")

        return self.model

    def prepare_dataset(self, dataset_name: str = "wikitext", subset: str = "wikitext-2-raw-v1", split: str = "train") -> Dataset:
        """
        Prepare training dataset with SentencePiece tokenization.

        Args:
            dataset_name: Name of the dataset to load
            subset: Dataset subset
            split: Dataset split (train/validation/test)

        Returns:
            Tokenized dataset
        """
        logger.info(f"Loading dataset: {dataset_name}/{subset} ({split})")

        # Load dataset
        dataset = load_dataset(dataset_name, subset, split=split)

        # Add arithmetic examples to the dataset
        arithmetic_examples = self._generate_arithmetic_examples(num_examples=10000)

        # Combine with existing dataset
        combined_texts = list(dataset['text']) + arithmetic_examples
        combined_dataset = Dataset.from_dict({'text': combined_texts})

        # Tokenize dataset
        def tokenize_function(examples):
            # Tokenize texts
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.config.get('max_position_embeddings', 1024),
                return_special_tokens_mask=False,
            )

            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].copy()

            return tokenized

        logger.info("Tokenizing dataset...")
        tokenized_dataset = combined_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=combined_dataset.column_names,
            desc="Tokenizing"
        )

        logger.info(f"Dataset prepared: {len(tokenized_dataset)} examples")
        return tokenized_dataset

    def _generate_arithmetic_examples(self, num_examples: int = 10000) -> List[str]:
        """Generate arithmetic training examples."""
        import random

        examples = []
        operations = ['+', '-', '*', '/']

        for _ in range(num_examples):
            a = random.randint(1, 1000)
            b = random.randint(1, 100)
            op = random.choice(operations)

            if op == '+':
                result = a + b
                examples.append(f"Calculate {a} + {b} = {result}")
            elif op == '-':
                result = a - b
                examples.append(f"What is {a} - {b}? The answer is {result}")
            elif op == '*':
                result = a * b
                examples.append(f"Multiply {a} by {b}: {result}")
            elif op == '/':
                if b != 0:
                    result = round(a / b, 2)
                    examples.append(f"Divide {a} by {b} to get {result}")

        return examples

    def train(
        self,
        output_dir: str = "outputs",
        num_train_epochs: int = None,
        per_device_train_batch_size: int = None,
        gradient_accumulation_steps: int = None,
        learning_rate: float = None,
        warmup_ratio: float = None,
        weight_decay: float = None,
        logging_steps: int = None,
        save_steps: int = None,
        eval_steps: int = None,
        use_wandb: bool = False,
        **kwargs
    ):
        """
        Train the model.

        Args:
            output_dir: Output directory for model checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio
            weight_decay: Weight decay
            logging_steps: Logging frequency
            save_steps: Save frequency
            eval_steps: Evaluation frequency
            use_wandb: Whether to use Weights & Biases logging
            **kwargs: Additional training arguments
        """
        # Use config values as defaults
        num_train_epochs = num_train_epochs or self.config.get('num_train_epochs', 3)
        per_device_train_batch_size = per_device_train_batch_size or self.config.get('pdbs_train', 8)
        gradient_accumulation_steps = gradient_accumulation_steps or self.config.get('grad_accum_steps', 1)
        learning_rate = learning_rate or self.config.get('learning_rate', 5e-5)
        warmup_ratio = warmup_ratio or self.config.get('warmup_ratio', 0.1)
        weight_decay = weight_decay or self.config.get('weight_decay', 0.01)
        logging_steps = logging_steps or self.config.get('step_val', 100)
        save_steps = save_steps or self.config.get('step_val', 100)
        eval_steps = eval_steps or self.config.get('step_val', 100)

        # Prepare datasets
        train_dataset = self.prepare_dataset(split="train")
        eval_dataset = self.prepare_dataset(split="validation")

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if use_wandb else None,
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=True,
            remove_unused_columns=False,
            **kwargs
        )

        # Initialize Weights & Biases if requested
        if use_wandb:
            wandb.init(
                project="guac-sentencepiece",
                name=f"{self.model_name}-{self.config.get('architecture', 'mixed_layer')}",
                config=self.config
            )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logger.info("Starting training...")
        trainer.train()

        # Save final model and tokenizer
        final_output_dir = os.path.join(output_dir, "final")
        trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)

        logger.info(f"Training completed! Model saved to: {final_output_dir}")

        # Test the trained model
        self._test_model()

    def _test_model(self):
        """Test the trained model with arithmetic examples."""
        if self.model is None or self.tokenizer is None:
            logger.warning("Model or tokenizer not available for testing")
            return

        logger.info("Testing trained model...")

        test_examples = [
            "Calculate 15 + 27",
            "What is 144 - 89?",
            "Multiply 12 by 8",
            "Find the square root of 64",
        ]

        self.model.eval()
        with torch.no_grad():
            for example in test_examples:
                # Tokenize input
                inputs = self.tokenizer(example, return_tensors="pt")

                # Generate
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode output
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                logger.info(f"Input: {example}")
                logger.info(f"Output: {generated_text}")
                logger.info("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Train GUAC model with SentencePiece tokenization")
    parser.add_argument('--config', default='model_configs.yaml', help='Path to model configuration file')
    parser.add_argument('--model', default='nano', help='Model configuration to use')
    parser.add_argument('--output_dir', default='outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--retrain_tokenizer', action='store_true', help='Force retrain tokenizer')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')

    args = parser.parse_args()

    # Initialize trainer
    trainer = GUACSentencePieceTrainer(args.config, args.model)

    # Setup tokenizer and model
    trainer.setup_tokenizer(force_retrain=args.retrain_tokenizer)
    trainer.setup_model()

    # Train
    train_kwargs = {}
    if args.epochs:
        train_kwargs['num_train_epochs'] = args.epochs
    if args.batch_size:
        train_kwargs['per_device_train_batch_size'] = args.batch_size
    if args.learning_rate:
        train_kwargs['learning_rate'] = args.learning_rate

    trainer.train(
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        **train_kwargs
    )


if __name__ == "__main__":
    main()