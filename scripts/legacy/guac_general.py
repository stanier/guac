#!/usr/bin/env python3

"""
General-purpose text model training with automatic tokenizer management.

This script automatically checks for existing tokenizers and trains new ones using
the integrated train_tokenizer.py if needed. Supports SentencePiece, BPE, and
character-level tokenizers with special token integration.

Usage:
    # Basic training with automatic SentencePiece tokenizer
    python guac_general.py --model-config guac-3b

    # Train with specific datasets and custom vocab size
    python guac_general.py --datasets wikitext-103,openwebtext --vocab-size 16000

    # Force retrain tokenizer even if one exists
    python guac_general.py --force-retrain-tokenizer --tokenizer-type sentencepiece

    # Use existing tokenizer from site config
    python guac_general.py --load-tokenizer custom_trained

    # Use explicit tokenizer path
    python guac_general.py --tokenizer-path /path/to/tokenizer
"""

import os, sys, yaml, re, torch, argparse
from typing import List, Tuple, Optional, Dict, Any
from torch import nn
from tqdm.auto import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from transformers import (AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling,
                          PreTrainedTokenizerFast, PrinterCallback, EarlyStoppingCallback)
from tokenizers import Tokenizer, trainers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from datasets import load_from_disk, concatenate_datasets, DatasetDict, load_dataset
try:
    from ..callbacks import CustomProgressCallback, ProfilingCallback, LayerTimingCallback
    from ..prompt_templates import create_prompt_tokenizer, apply_default_template, apply_template_file
    from ..train_tokenizer import main as train_tokenizer_main, load_site_config as train_tokenizer_load_site_config
except ImportError:
    from scripts.training.callbacks import CustomProgressCallback, ProfilingCallback, LayerTimingCallback
    from scripts.training.prompt_templates import create_prompt_tokenizer, apply_default_template, apply_template_file
    from scripts.training.train_tokenizer import main as train_tokenizer_main, load_site_config as train_tokenizer_load_site_config

def create_text_vocab_from_dataset(dataset, vocab_size=1000, min_frequency=2):
    """Creates a vocabulary from the dataset text without special tokens."""
    from collections import Counter

    # Collect all text from the dataset
    all_text = []
    for split in dataset.values():
        all_text.extend(split["text"])

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

def create_text_vocab_from_datasets(datasets, vocab_size=1000, min_frequency=2):
    """Creates a vocabulary from multiple datasets without special tokens."""
    from collections import Counter

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
                            if len(all_text) >= 10000:  # Limit for memory
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

def train_tokenizer_from_datasets(datasets, tokenizer_type="character", vocab_size=1000, **kwargs):
    """Creates a tokenizer based on multiple datasets with prompt tokens added globally."""
    # Create vocabulary from all datasets without special tokens
    base_vocab = create_text_vocab_from_datasets(datasets, vocab_size, kwargs.get('min_frequency', 2))

    # Get prompt tokenizer for special tokens
    prompt_tokenizer = create_prompt_tokenizer()
    special_tokens = prompt_tokenizer.get_all_special_tokens()

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

    # Get core tokens from prompt system
    core_tokens = prompt_tokenizer.get_core_tokens()

    if tokenizer_type == "character":
        tokenizer = Tokenizer(WordLevel(vocab=final_vocab, unk_token=core_tokens['unk_token']))
        tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")
        tokenizer.normalizer = None
    elif tokenizer_type == "bpe":
        # Use BPE tokenizer for more general text
        from tokenizers import ByteLevelBPETokenizer
        tokenizer = Tokenizer(WordLevel(vocab=final_vocab, unk_token=core_tokens['unk_token']))
        tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")
        tokenizer.normalizer = None
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    # Add special tokens to the tokenizer
    tokenizer.add_special_tokens(list(core_tokens.values()))

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=core_tokens['bos_token'],
        eos_token=core_tokens['eos_token'],
        unk_token=core_tokens['unk_token'],
        pad_token=core_tokens['pad_token']
    )

def train_tokenizer_from_dataset(dataset, tokenizer_type="character", vocab_size=1000, **kwargs):
    """Creates a tokenizer based on the dataset content with prompt tokens added globally."""
    # Create vocabulary from dataset without special tokens
    base_vocab = create_text_vocab_from_dataset(dataset, vocab_size, kwargs.get('min_frequency', 2))

    # Get prompt tokenizer for special tokens
    prompt_tokenizer = create_prompt_tokenizer()
    special_tokens = prompt_tokenizer.get_all_special_tokens()

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

    # Get core tokens from prompt system
    core_tokens = prompt_tokenizer.get_core_tokens()

    if tokenizer_type == "character":
        tokenizer = Tokenizer(WordLevel(vocab=final_vocab, unk_token=core_tokens['unk_token']))
        tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")
        tokenizer.normalizer = None
    elif tokenizer_type == "bpe":
        # Use BPE tokenizer for more general text
        from tokenizers import ByteLevelBPETokenizer
        # For BPE, we'd need to write text to temporary files, but for simplicity
        # we'll fall back to character-level for now
        tokenizer = Tokenizer(WordLevel(vocab=final_vocab, unk_token=core_tokens['unk_token']))
        tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")
        tokenizer.normalizer = None
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    # Add special tokens to the tokenizer
    tokenizer.add_special_tokens(list(core_tokens.values()))

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=core_tokens['bos_token'],
        eos_token=core_tokens['eos_token'],
        unk_token=core_tokens['unk_token'],
        pad_token=core_tokens['pad_token']
    )


def apply_prompt_template_to_dataset(dataset, template_config=None, template_file="default.jinja"):
    """Apply prompt templates to dataset texts."""
    prompt_tokenizer = create_prompt_tokenizer()

    # Default template configuration
    default_template_config = {
        "system_guidance": "You are a helpful AI assistant trained on text data.",
        "organization_guidance": "Follow standard guidelines for text processing.",
        "user_guidance": "Process the input text appropriately.",
        "task": "Process the following text sequence.",
        "sequence.label": "text_sequence",
        "sequence.modality_hint": "text"
    }

    if template_config:
        default_template_config.update(template_config)

    def apply_template(examples):
        templated_texts = []
        for text in examples["text"]:
            # Create context for this specific text
            context = default_template_config.copy()
            context["input_text"] = text

            # Use the specified template file
            try:
                templated_text = apply_template_file(template_file, context)
            except Exception as e:
                print(f"Warning: Failed to apply template file '{template_file}': {e}")
                print("Falling back to default template")
                templated_text = apply_default_template(context)

            templated_texts.append(templated_text)

        return {"text": templated_texts}

    # IterableDatasets don't support the desc parameter
    if hasattr(dataset, '__len__'):
        # Regular Dataset
        return dataset.map(apply_template, batched=True, desc=f"Applying prompt templates ({template_file})")
    else:
        # IterableDataset
        return dataset.map(apply_template, batched=True)

def tokenize_dataset(tokenizer, dataset, apply_templates=True, template_config=None, template_file="default.jinja"):
    """Tokenize dataset with optional prompt template application."""

    # Handle both DatasetDict and dict of IterableDatasets
    if isinstance(dataset, dict) and not hasattr(dataset, 'map'):
        # This is a dict of IterableDatasets, convert each split
        from datasets import DatasetDict
        tokenized_splits = {}

        for split_name, split_dataset in dataset.items():
            # Apply templates if requested
            if apply_templates:
                split_dataset = apply_prompt_template_to_dataset(split_dataset, template_config, template_file)

            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    padding='max_length',
                    max_length=256,  # Increased for template content
                    truncation=True,
                    return_tensors=None,
                )

            # IterableDatasets don't support desc or remove_columns parameters
            tokenized_split = split_dataset.map(
                tokenize_function,
                batched=True
            )
            tokenized_splits[split_name] = tokenized_split

        # For streaming datasets, we can't easily create a DatasetDict
        # Return the dict directly and handle it downstream
        return tokenized_splits
    else:
        # Regular DatasetDict processing
        # Apply templates if requested
        if apply_templates:
            dataset = apply_prompt_template_to_dataset(dataset, template_config, template_file)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding='max_length',
                max_length=256,  # Increased for template content
                truncation=True,
                return_tensors=None,
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[],
            desc="Tokenizing dataset"
        )
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        return tokenized_dataset

def init_model(tokenizer, model_kwargs, architecture_name=None):
    """Initialize model with specified architecture from config."""
    # Load architecture config
    arch_config = load_architecture_config()

    if architecture_name is None: architecture_name = select_runtime_architecture(arch_config)

    config_class, model_class = get_architecture_classes(arch_config, architecture_name)

    arch_defaults = arch_config['architecture_defaults'].get(architecture_name, {})

    runtime_overrides = arch_config['runtime_config'].get('override_params', {})
    if runtime_overrides == 'pass': runtime_overrides = {}

    config_params = {
        'vocab_size': tokenizer.vocab_size,
        **arch_defaults,
        **runtime_overrides,
        **model_kwargs,
    }

    print(f"Initializing {architecture_name} model with config: {config_class.__name__}")
    config = config_class(**config_params)
    model = model_class(config)

    return model

def train_model(model, tokenizer, training_dataset, validation_dataset, training_kwargs, log_steps):
    # If no validation dataset, disable evaluation and adjust scheduler
    if validation_dataset is None:
        eval_strategy = "no"
        metric_for_best_model = None
        print("No validation dataset available - disabling evaluation")

        # Override scheduler that requires evaluation
        if training_kwargs.get("lr_scheduler_type") == "reduce_lr_on_plateau":
            training_kwargs = training_kwargs.copy()
            training_kwargs["lr_scheduler_type"] = "cosine"
            print("Changed scheduler from reduce_lr_on_plateau to cosine (no eval dataset)")
    else:
        eval_strategy = "steps"
        metric_for_best_model = 'eval_input_ids_loss'

    # Handle streaming datasets that don't have __len__
    training_kwargs = training_kwargs.copy()
    if not hasattr(training_dataset, '__len__'):
        # For streaming datasets, set max_steps if not already set
        if 'max_steps' not in training_kwargs:
            # Calculate reasonable max_steps based on epochs and estimated dataset size
            num_epochs = training_kwargs.get('num_train_epochs', 1)
            batch_size = training_kwargs.get('per_device_train_batch_size', 8)
            grad_accum = training_kwargs.get('gradient_accumulation_steps', 1)

            # Estimate steps: assume ~100K articles, ~1000 tokens each, 256 max_length
            estimated_samples = 100000
            steps_per_epoch = estimated_samples // (batch_size * grad_accum)
            max_steps = int(steps_per_epoch * num_epochs)

            training_kwargs['max_steps'] = max_steps
            # Remove num_train_epochs when using max_steps
            training_kwargs.pop('num_train_epochs', None)
            print(f"Set max_steps to {max_steps} for streaming dataset")

    training_args = TrainingArguments(
        report_to="none",
        eval_strategy=eval_strategy,
        metric_for_best_model=metric_for_best_model,
        disable_tqdm=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",          # Better for older GPUs like 1080 Ti
        #optim="adamw_torch_fused",   # Requires newer CUDA
        #optim="schedule_free_radamw",
        torch_compile=False,         # May not work well on 1080 Ti
        fp16=True,                   # Essential for 1080 Ti efficiency
        **training_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
        callbacks=_create_callbacks(tokenizer, log_steps, model, args)
    )

    trainer.remove_callback(PrinterCallback)
    trainer.train()
    return trainer


def _create_callbacks(tokenizer, log_steps, model, args):
    """Create training callbacks based on arguments."""
    callbacks = [CustomProgressCallback(tokenizer, log_steps=log_steps)]

    # Add profiling callbacks if requested
    if args.enable_profiling:
        profiling_callback = ProfilingCallback(
            log_every_n_steps=args.profiling_steps,
            detailed_profiling=True
        )
        callbacks.append(profiling_callback)
        print("🔍 Enabled detailed profiling (timing, memory, performance)")

    if args.enable_layer_profiling:
        layer_callback = LayerTimingCallback(
            model=model,
            log_every_n_steps=args.profiling_steps * 2  # Less frequent for layer timing
        )
        callbacks.append(layer_callback)
        print("🧬 Enabled layer-by-layer timing analysis")

    return callbacks

def load_model_configs():
    """Load model configurations from the YAML config file."""
    config_path = os.path.join(project_root, "model_configs.yaml")
    with open(config_path, 'r') as f: config = yaml.safe_load(f)

    def convert_numeric_values(obj):
        if isinstance(obj, dict): return {k: convert_numeric_values(v) for k, v in obj.items()}
        elif isinstance(obj, str) and ('e-' in obj.lower() or 'e+' in obj.lower()):
            try: return float(obj)
            except ValueError: return obj
        else: return obj

    return convert_numeric_values(config)

def load_architecture_config():
    """Load architecture configurations from the YAML config file."""
    config_path = os.path.join(project_root, "architecture_configs.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    def convert_numeric_values(obj):
        if isinstance(obj, dict):
            return {k: convert_numeric_values(v) for k, v in obj.items()}
        elif isinstance(obj, str) and ('e-' in obj.lower() or 'e+' in obj.lower()):
            try:
                return float(obj)
            except ValueError:
                return obj
        else:
            return obj

    return convert_numeric_values(config)

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
        if not isinstance(text, str) or "${" not in text: return text

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


def check_and_train_tokenizer(site_config, dataset_names=None, tokenizer_type="sentencepiece", vocab_size=32000):
    """
    Check if a tokenizer exists in the configured location. If not, train one using train_tokenizer.py.

    Args:
        site_config: Site configuration dictionary
        dataset_names: List of dataset names to use for training (None = use defaults)
        tokenizer_type: Type of tokenizer to train ('sentencepiece', 'bpe', 'character')
        vocab_size: Vocabulary size for the tokenizer

    Returns:
        Path to the tokenizer (either existing or newly trained)
    """
    dirs = site_config['directories']
    tokenizer_dirs = dirs.get('tokenizers', {})

    # Determine the expected tokenizer path
    expected_tokenizer_path = None

    # Check various possible tokenizer locations in order of preference
    tokenizer_keys = ['general', 'custom_trained', 'char_arithmetic', 'base_dir']
    for key in tokenizer_keys:
        if key in tokenizer_dirs and key != 'base_dir':
            expected_tokenizer_path = tokenizer_dirs[key]
            break

    # Fallback if no specific path found
    if not expected_tokenizer_path:
        expected_tokenizer_path = tokenizer_dirs.get('base_dir', './tokenizer')

    # Check if tokenizer files exist
    tokenizer_exists = False
    required_files = ['tokenizer.json', 'tokenizer_config.json']  # Standard HuggingFace tokenizer files
    sentencepiece_files = ['tokenizer.model', 'tokenizer_config.json']  # SentencePiece files

    if os.path.exists(expected_tokenizer_path):
        # Check for HuggingFace tokenizer files
        hf_files_exist = all(os.path.exists(os.path.join(expected_tokenizer_path, f)) for f in required_files)
        # Check for SentencePiece files
        sp_files_exist = all(os.path.exists(os.path.join(expected_tokenizer_path, f)) for f in sentencepiece_files)

        tokenizer_exists = hf_files_exist or sp_files_exist

        if tokenizer_exists:
            print(f"✓ Found existing tokenizer at: {expected_tokenizer_path}")
            if sp_files_exist:
                print("  Type: SentencePiece tokenizer detected")
            elif hf_files_exist:
                print("  Type: HuggingFace tokenizer detected")
            return expected_tokenizer_path

    # If tokenizer doesn't exist, train a new one
    print(f"✗ No tokenizer found at: {expected_tokenizer_path}")
    print(f"Training new {tokenizer_type} tokenizer with vocab size {vocab_size}...")

    # Prepare arguments for train_tokenizer.py
    import subprocess

    # Create the command to run train_tokenizer.py
    cmd = [
        sys.executable,
        os.path.join(project_root, 'scripts', 'training', 'train_tokenizer.py'),
        '--tokenizer-type', tokenizer_type,
        '--vocab-size', str(vocab_size),
        '--output-path', expected_tokenizer_path
    ]

    # Add dataset arguments if specified
    if dataset_names:
        cmd.extend(['--datasets', ','.join(dataset_names)])

    # Add arithmetic data for SentencePiece by default
    if tokenizer_type in ['sentencepiece', 'bpe']:
        cmd.append('--include-arithmetic')
        cmd.extend(['--arithmetic-samples', '10000'])

    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=project_root)
        print("Tokenizer training completed successfully!")
        print("Training output:")
        print(result.stdout)

        if result.stderr:
            print("Training warnings/errors:")
            print(result.stderr)

        return expected_tokenizer_path

    except subprocess.CalledProcessError as e:
        print(f"Error training tokenizer: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command errors: {e.stderr}")
        raise RuntimeError(f"Failed to train tokenizer: {e}")
    except Exception as e:
        print(f"Unexpected error while training tokenizer: {e}")
        raise

def get_architecture_classes(arch_config, architecture_name):
    """Dynamically import and return the config and model classes for the specified architecture."""
    arch_info = arch_config['architectures'][architecture_name]

    if arch_info['import_path'] == 'transformers':
        from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
            Lfm2MoeConfig, Lfm2MoeForCausalLM, Lfm2Config, Lfm2ForCausalLM,
            GraniteConfig, GraniteForCausalLM, GraniteMoeConfig, GraniteMoeForCausalLM,
            GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM,
            GraniteMoeSharedConfig, GraniteMoeSharedForCausalLM)

        class_map = {
            'LlamaConfig': LlamaConfig, 'LlamaForCausalLM': LlamaForCausalLM,
            'Qwen2Config': Qwen2Config, 'Qwen2ForCausalLM': Qwen2ForCausalLM,
            'Lfm2MoeConfig': Lfm2MoeConfig, 'Lfm2MoeForCausalLM': Lfm2MoeForCausalLM,
            'Lfm2Config': Lfm2Config, 'Lfm2ForCausalLM': Lfm2ForCausalLM,
            'GraniteConfig': GraniteConfig, 'GraniteForCausalLM': GraniteForCausalLM,
            'GraniteMoeConfig': GraniteMoeConfig, 'GraniteMoeForCausalLM': GraniteMoeForCausalLM,
            'GraniteMoeHybridConfig': GraniteMoeHybridConfig, 'GraniteMoeHybridForCausalLM': GraniteMoeHybridForCausalLM,
            'GraniteMoeSharedConfig': GraniteMoeSharedConfig, 'GraniteMoeSharedForCausalLM': GraniteMoeSharedForCausalLM,
        }

        return class_map[arch_info['config_class']], class_map[arch_info['model_class']]

    elif arch_info['import_path'].startswith('scripts.training.'):
        # Custom architecture import
        import importlib

        try:
            # Dynamic import of custom architecture module
            module = importlib.import_module(arch_info['import_path'])

            # Get the config and model classes from the module
            config_class = getattr(module, arch_info['config_class'])
            model_class = getattr(module, arch_info['model_class'])

            return config_class, model_class

        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import custom architecture '{architecture_name}' from '{arch_info['import_path']}': {e}")

    else:
        raise ValueError(f"Unsupported import path: {arch_info['import_path']}")

def select_runtime_architecture(arch_config, override_arch=None):
    """Select the architecture to use at runtime."""
    # Priority: override_arch > environment variable > config file > default
    selected = override_arch
    if selected is None: selected = os.environ.get('GUAC_ARCHITECTURE')
    if selected is None: selected = arch_config['runtime_config']['selected_architecture']
    if selected is None: selected = arch_config['default_architecture']
    if selected not in arch_config['architectures']:
        available = list(arch_config['architectures'].keys())
        raise ValueError(f"Unknown architecture '{selected}'. Available: {available}")

    print(f"Selected architecture: {selected}")
    return selected

def do_train(model, tokenizer, mc, train, val, step_val, output_dir):
    trainer = train_model(
        model = model,
        tokenizer = tokenizer,
        training_dataset = train,
        validation_dataset = val,
        log_steps = step_val,
        training_kwargs = {
            "output_dir": output_dir,
            "eval_steps": step_val,
            "logging_steps": step_val,
            "save_steps": 5*step_val,
            "learning_rate": mc['learning_rate'],
            "lr_scheduler_type": "reduce_lr_on_plateau",
            "warmup_ratio": mc["warmup_ratio"],
            "weight_decay": mc["weight_decay"],
            "per_device_train_batch_size": mc["pdbs_train"],
            "per_device_eval_batch_size": mc["pdbs_eval"],
            "gradient_accumulation_steps": mc["grad_accum_steps"],
            "num_train_epochs": mc["num_train_epochs"],
        },
    )

    trainer.save_model(f"{output_dir}_complete")

def load_huggingface_dataset(dataset_config, dataset_name):
    """Load a HuggingFace dataset based on configuration."""
    config_name = dataset_config.get('config_name')
    streaming = dataset_config.get('streaming', False)
    max_articles = dataset_config.get('max_articles')

    print(f"Loading {dataset_name}: {dataset_config['dataset_name']}" +
          (f"/{config_name}" if config_name else "") +
          f" (streaming: {streaming}, max_articles: {max_articles})")

    # Load the dataset
    if config_name:
        dataset = load_dataset(dataset_config['dataset_name'], config_name, streaming=streaming)
    else:
        dataset = load_dataset(dataset_config['dataset_name'], streaming=streaming)

    # Limit articles if specified
    if max_articles and not streaming:
        # For non-streaming datasets, we can slice directly
        for split_name in dataset.keys():
            if len(dataset[split_name]) > max_articles:
                dataset[split_name] = dataset[split_name].select(range(max_articles))
                print(f"Limited {split_name} split to {max_articles} articles")

    elif max_articles and streaming:
        # For streaming datasets, we need to take only the first N examples
        # Keep as raw dict since IterableDataset can't be used in DatasetDict
        limited_dataset = {}
        for split_name in dataset.keys():
            limited_dataset[split_name] = dataset[split_name].take(max_articles)
            print(f"Will stream first {max_articles} articles from {split_name} split")
        dataset = limited_dataset

    return dataset


def load_and_prepare_datasets(site_config, dataset_names=None, combine_datasets=True, dataset_type="pretraining"):
    """Load and optionally combine multiple datasets."""
    datasets = {}
    dataset_configs = site_config.get('dataset_configs', {})

    # Choose the right dataset section based on type
    if dataset_type == "pretraining":
        hf_datasets = dataset_configs.get('pretraining_datasets', {})
        default_key = 'default_pretrain_datasets'
    elif dataset_type == "instruction":
        hf_datasets = dataset_configs.get('instruction_datasets', {})
        default_key = 'default_finetune_datasets'
    else:
        # Fallback to old structure for compatibility
        hf_datasets = dataset_configs.get('huggingface_datasets', {})
        default_key = 'default_pretrain_datasets'

    dirs = site_config['directories']

    if dataset_names is None:
        # Use default datasets from config based on type
        default_datasets = dataset_configs.get(default_key, ['wikitext-103'])
        # Filter out None values and empty lists
        if default_datasets:
            dataset_names = [name for name in default_datasets if name]

        if not dataset_names:
            # Fallback to available HuggingFace datasets if no defaults specified
            available_hf = list(hf_datasets.keys())
            if available_hf:
                dataset_names = available_hf[:2]  # Take first 2 datasets
                print(f"No default {dataset_type} datasets specified, using first available: {dataset_names}")
            else:
                dataset_names = ['wikitext-103']  # Ultimate fallback
                print(f"No datasets configured, using fallback: {dataset_names}")
        else:
            print(f"Using default {dataset_type} datasets: {dataset_names}")

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

    if combine_datasets and len(datasets) > 1:
        # Check if we have streaming datasets (can't use concatenate_datasets)
        has_streaming = any(
            isinstance(ds, dict) and any(hasattr(split, 'take') for split in ds.values())
            for ds in datasets.values()
        )

        if has_streaming:
            # For streaming datasets, return as-is without combining
            print("Streaming datasets detected - skipping combination")
            return list(datasets.values())[0], datasets
        else:
            # Combine regular datasets
            all_train = [ds['train'] for ds in datasets.values() if 'train' in ds]
            all_val = [ds['validation'] for ds in datasets.values() if 'validation' in ds]

            combined_dataset = DatasetDict({
                'train': concatenate_datasets(all_train) if all_train else None,
                'validation': concatenate_datasets(all_val) if all_val else None
            })
            return combined_dataset, datasets
    else:
        # Return the first dataset and all individual datasets
        return list(datasets.values())[0], datasets

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='General-purpose text model training')
    parser.add_argument('--datasets', type=str, help='Comma-separated list of dataset names to use')
    parser.add_argument('--model-config', type=str, default='iota', help='Model configuration to use')
    parser.add_argument('--tokenizer-type', type=str, default='sentencepiece', choices=['sentencepiece', 'bpe', 'character'],
                       help='Type of tokenizer to use/train (default: sentencepiece)')
    parser.add_argument('--vocab-size', type=int, default=32000, help='Vocabulary size for tokenizer (default: 32000)')
    parser.add_argument('--architecture', type=str, help='Model architecture to use (overrides config)')
    parser.add_argument('--combine-datasets', action='store_true', default=True,
                       help='Combine multiple datasets for training')
    parser.add_argument('--use-templates', action='store_true', default=False,
                       help='Apply prompt templates to input data (default: False for pretraining)')
    parser.add_argument('--no-templates', action='store_true',
                       help='Disable prompt template application')
    parser.add_argument('--template-task', type=str,
                       help='Override default task description in templates')
    parser.add_argument('--template-file', type=str, default='default.jinja',
                       help='Jinja template file to use (default: default.jinja)')
    parser.add_argument('--list-templates', action='store_true',
                       help='List available template files and exit')
    parser.add_argument('--load-tokenizer', type=str,
                       help='Load pre-trained tokenizer from path (relative to site config)')
    parser.add_argument('--tokenizer-path', type=str,
                       help='Explicit path to pre-trained tokenizer directory')
    parser.add_argument('--force-retrain-tokenizer', action='store_true',
                       help='Force re-training of tokenizer even if one exists')
    parser.add_argument('--force-retokenize', action='store_true',
                       help='Force re-tokenization even if cached datasets exist')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable dataset caching (don\'t save or load tokenized datasets)')

    # Profiling arguments
    parser.add_argument('--enable-profiling', action='store_true',
                       help='Enable detailed training profiling (timing, memory, etc.)')
    parser.add_argument('--enable-layer-profiling', action='store_true',
                       help='Enable layer-by-layer timing analysis (more overhead)')
    parser.add_argument('--profiling-steps', type=int, default=10,
                       help='Log profiling stats every N steps (default: 10)')

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Handle template listing
    if args.list_templates:
        tokenizer = create_prompt_tokenizer()
        templates = tokenizer.list_available_templates()
        print("Available template files:")
        for template in templates:
            print(f"  {template}")
        if not templates:
            print("  No template files found in templates directory")
            print(f"  Template directory: {tokenizer.template_dir}")
        exit(0)

    # Load site-specific configuration
    site_config = load_site_config()

    # Override tokenizer config with command line args
    if 'tokenizer' not in site_config:
        site_config['tokenizer'] = {}
    site_config['tokenizer']['type'] = args.tokenizer_type
    site_config['tokenizer']['vocab_size'] = args.vocab_size

    # Parse dataset names for tokenizer training
    dataset_names = None
    if args.datasets:
        dataset_names = [name.strip() for name in args.datasets.split(',')]

    # Load datasets first (needed for both tokenizer training and model training)
    main_dataset, individual_datasets = load_and_prepare_datasets(
        site_config, dataset_names=dataset_names, combine_datasets=args.combine_datasets, dataset_type="pretraining")

    # Load or train tokenizer
    if args.tokenizer_path:
        # Load from explicit path
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        print(f"Loaded tokenizer from: {args.tokenizer_path}")
    elif args.load_tokenizer:
        # Load from site config path
        dirs = site_config['directories']
        tokenizer_path = dirs['tokenizers'].get(args.load_tokenizer)
        if not tokenizer_path:
            available_keys = [k for k in dirs['tokenizers'].keys() if k != 'base_dir']
            raise ValueError(f"Tokenizer key '{args.load_tokenizer}' not found in site config. Available: {available_keys}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Loaded tokenizer from site config path '{args.load_tokenizer}': {tokenizer_path}")
    else:
        # Check for existing tokenizer or train a new one using train_tokenizer.py
        if args.force_retrain_tokenizer:
            print("Force retraining tokenizer (--force-retrain-tokenizer specified)")
            # Remove existing tokenizer directory to force retrain
            dirs = site_config['directories']
            tokenizer_dirs = dirs.get('tokenizers', {})
            for key in ['general', 'custom_trained', 'char_arithmetic']:
                if key in tokenizer_dirs:
                    potential_path = tokenizer_dirs[key]
                    if os.path.exists(potential_path):
                        import shutil
                        print(f"Removing existing tokenizer at: {potential_path}")
                        shutil.rmtree(potential_path)
                        break

        # Use the new check_and_train_tokenizer function
        tokenizer_path = check_and_train_tokenizer(
            site_config=site_config,
            dataset_names=dataset_names,
            tokenizer_type=args.tokenizer_type,
            vocab_size=args.vocab_size
        )

        # Load the tokenizer (either existing or newly trained)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Using tokenizer from: {tokenizer_path}")

    # Determine template settings
    use_templates = args.use_templates and not args.no_templates
    template_config = None
    if use_templates and args.template_task:
        template_config = {"task": args.template_task}

    print(f"Template application: {'enabled' if use_templates else 'disabled'}")

    # Get directories for caching
    dirs = site_config['directories']

    # Tokenize datasets (with caching)
    tokenized_datasets = {}
    for name, dataset in individual_datasets.items():
        # Check if tokenized dataset already exists on disk
        save_path = dirs['tokenized_datasets'].get(name, f"./tokenized_{name}")

        # Create a unique cache key based on tokenizer vocab size, template settings, and dataset name
        template_suffix = f"_templates_{args.template_file.replace('.jinja', '')}" if use_templates else "_no_templates"
        vocab_suffix = f"_vocab_{tokenizer.vocab_size}"
        cache_path = f"{save_path}{vocab_suffix}{template_suffix}"

        # Try to load existing tokenized dataset (unless forced to retokenize or caching disabled)
        if not args.force_retokenize and not args.no_cache and os.path.exists(cache_path):
            try:
                print(f"Loading cached tokenized dataset: {name} from {cache_path}")
                tokenized_dataset = load_from_disk(cache_path)
                print(f"Successfully loaded cached dataset {name}")
                tokenized_datasets[name] = tokenized_dataset
                continue
            except Exception as e:
                print(f"Failed to load cached dataset {name}: {e}")
                print(f"Will re-tokenize dataset {name}")

        # Tokenize dataset if not cached or loading failed
        print(f"Tokenizing dataset: {name}")
        tokenized_dataset = tokenize_dataset(tokenizer, dataset, apply_templates=use_templates, template_config=template_config, template_file=args.template_file)

        # Save non-streaming datasets to disk (unless caching disabled)
        if not args.no_cache and hasattr(tokenized_dataset, 'save_to_disk'):
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                tokenized_dataset.save_to_disk(cache_path)
                print(f"Tokenized dataset {name} saved to: {cache_path}")
            except Exception as e:
                print(f"Failed to save tokenized dataset {name}: {e}")
        else:
            if args.no_cache:
                print(f"Tokenized dataset {name} (caching disabled)")
            else:
                print(f"Tokenized streaming dataset {name} (not saved to disk)")

        tokenized_datasets[name] = tokenized_dataset

    # Use the first tokenized dataset for training (or combined if available)
    if 'train' in main_dataset and 'validation' in main_dataset:
        # Try to load cached combined training dataset
        combined_cache_path = f"{dirs['tokenized_datasets'].get('combined', './tokenized_combined')}{vocab_suffix}{template_suffix}"

        if not args.force_retokenize and not args.no_cache and os.path.exists(combined_cache_path):
            try:
                print(f"Loading cached combined training dataset from {combined_cache_path}")
                training_dataset = load_from_disk(combined_cache_path)
                print("Successfully loaded cached combined training dataset")
            except Exception as e:
                print(f"Failed to load cached combined dataset: {e}")
                print("Will re-tokenize combined dataset")
                training_dataset = tokenize_dataset(tokenizer, main_dataset, apply_templates=use_templates, template_config=template_config, template_file=args.template_file)

                # Save combined training dataset
                if not args.no_cache and hasattr(training_dataset, 'save_to_disk'):
                    try:
                        os.makedirs(os.path.dirname(combined_cache_path), exist_ok=True)
                        training_dataset.save_to_disk(combined_cache_path)
                        print(f"Combined training dataset saved to: {combined_cache_path}")
                    except Exception as e:
                        print(f"Failed to save combined training dataset: {e}")
        else:
            print("Tokenizing combined training dataset")
            training_dataset = tokenize_dataset(tokenizer, main_dataset, apply_templates=use_templates, template_config=template_config, template_file=args.template_file)

            # Save combined training dataset
            if not args.no_cache and hasattr(training_dataset, 'save_to_disk'):
                try:
                    os.makedirs(os.path.dirname(combined_cache_path), exist_ok=True)
                    training_dataset.save_to_disk(combined_cache_path)
                    print(f"Combined training dataset saved to: {combined_cache_path}")
                except Exception as e:
                    print(f"Failed to save combined training dataset: {e}")
    else:
        training_dataset = list(tokenized_datasets.values())[0]

    # Load model configurations from YAML file
    config = load_model_configs()
    model_training_configs = config['model_training_configs']
    model_training_configs['defaults'] = config['defaults']
    common_training_kwargs = config['common_training_kwargs']

    # Use specified model configuration
    target = args.model_config
    targets = [target]

    for target in targets:
        print(f"> TRAINING: {target}")

        if target not in model_training_configs:
            available_configs = list(model_training_configs.keys())
            print(f"Error: Model config '{target}' not found. Available: {available_configs}")
            continue

        selected_config = model_training_configs[target]
        d = common_training_kwargs | model_training_configs["defaults"]
        mc = d | selected_config

        model = init_model(tokenizer, {
            "hidden_size": mc['hidden_size'], "intermediate_size": mc['hidden_size'] * mc['intermediate_layers_fac'],
            "num_hidden_layers": mc['num_hidden_layers'], "num_attention_heads": mc['num_attention_heads'],
            "max_position_embeddings": mc['max_position_embeddings'], "num_key_value_heads": mc['num_kv_heads'],
            "torch_dtype": torch.float16, "num_experts": mc['num_experts'],
            "num_experts_per_tok": mc['num_experts_per_token'], "layer_types": mc['layer_types']
        }, architecture_name=args.architecture)

        print(f"Model has {model.num_parameters()} parameters")

        if model.config.vocab_size != tokenizer.vocab_size:
            print(f"MODEL ERROR: Mismatch between model vocab_size ({model.config.vocab_size}) and tokenizer vocab_size ({tokenizer.vocab_size})")

        step_val = mc['step_val']

        # Prepare training data
        if hasattr(training_dataset, 'keys') and 'train' in training_dataset:
            train_data = training_dataset["train"]
            val_data = training_dataset["validation"][:100] if "validation" in training_dataset else None
        else:
            # Fall back to first available tokenized dataset
            first_dataset = list(tokenized_datasets.values())[0]
            if isinstance(first_dataset, dict):
                # Streaming dataset dict
                train_data = first_dataset.get("train")
                val_data = first_dataset.get("validation")
                # For streaming datasets, we can't slice easily, so take first 100 via take()
                if val_data and hasattr(val_data, 'take'):
                    val_data = val_data.take(100)
            else:
                # Regular DatasetDict
                train_data = first_dataset["train"]
                val_data = first_dataset["validation"][:100] if "validation" in first_dataset else None

        # Determine output directory
        output_dir = dirs['checkpoints'].get('pretrain_general', dirs['checkpoints'].get('pretrain_simple', './checkpoints'))

        do_train(model, tokenizer, mc, train_data, val_data, step_val, output_dir)
        print("\n")

