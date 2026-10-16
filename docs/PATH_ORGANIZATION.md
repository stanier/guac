# Guac Path Organization

This document describes the new centralized path management system for Guac training, which ensures all outputs are organized under the configured `SCRATCH_DIR`.

## Overview

All training outputs (datasets, tokenizers, models, caches) are now centrally managed through the `PathManager` class and organized in a consistent hierarchy under the `SCRATCH_DIR` specified in `site_config.yml`.

## Configuration

Paths are configured in `site_config.yml` under the `directories` section:

```yaml
directories:
  scratch_dir: "./scratch"  # Main directory for all outputs

  datasets:
    base_dir: "${scratch_dir}/datasets"
    huggingface_cache: "${base_dir}/huggingface_cache"
    generated: "${base_dir}/generated"

  tokenized_datasets:
    base_dir: "${scratch_dir}/tokenized_datasets"
    cache_dir: "${base_dir}/cache"
    pretraining: "${base_dir}/pretraining"
    midtraining: "${base_dir}/midtraining"
    posttraining: "${base_dir}/posttraining"

  tokenizers:
    base_dir: "${scratch_dir}/tokenizers"
    sentencepiece: "${base_dir}/sentencepiece"
    character: "${base_dir}/character"
    custom: "${base_dir}/custom"

  models:
    base_dir: "${scratch_dir}/models"
    pretraining: "${base_dir}/pretraining"
    midtraining: "${base_dir}/midtraining"
    posttraining: "${base_dir}/posttraining"
    checkpoints: "${base_dir}/checkpoints"
    final: "${base_dir}/final"

  temp:
    base_dir: "${scratch_dir}/temp"
    tokenizer_training: "${base_dir}/tokenizer_training"
    data_processing: "${base_dir}/data_processing"
```

## Directory Structure

With the default configuration, the complete directory structure looks like:

```
./scratch/
├── datasets/
│   ├── huggingface_cache/          # HuggingFace dataset downloads
│   └── generated/                  # Custom generated datasets
├── tokenized_datasets/
│   ├── cache/                      # General tokenized dataset cache
│   ├── pretraining/                # Pre-training tokenized datasets
│   ├── midtraining/                # Mid-training tokenized datasets
│   └── posttraining/               # Post-training tokenized datasets
├── tokenizers/
│   ├── sentencepiece/              # SentencePiece tokenizers
│   ├── character/                  # Character-level tokenizers
│   └── custom/                     # Custom tokenizers
├── models/
│   ├── pretraining/                # Pre-trained models
│   ├── midtraining/                # Mid-trained models
│   ├── posttraining/               # Post-trained models
│   ├── checkpoints/                # Training checkpoints
│   └── final/                      # Final production models
└── temp/
    ├── tokenizer_training/         # Temporary files for tokenizer training
    └── data_processing/            # Temporary files for data processing
```

## Usage

### Path Manager API

The `PathManager` class provides programmatic access to all paths:

```python
from guac.core.path_manager import get_path_manager

pm = get_path_manager()

# Get base directories
scratch_dir = pm.get_scratch_dir()
dataset_dir = pm.get_dataset_base_dir()

# Get specific paths
tokenizer_path = pm.get_tokenizer_output_path("sentencepiece", "my_tokenizer")
model_path = pm.get_model_output_path("pretraining", "guac-3b")
cache_path = pm.get_tokenized_dataset_path("pretraining", "wikitext_vocab32k")
```

### Convenience Functions

For common operations, convenience functions are available:

```python
from guac.core.path_manager import (
    get_scratch_dir,
    get_dataset_path,
    get_tokenizer_output_path,
    get_model_output_path,
    get_huggingface_cache_dir
)

# Use directly
model_dir = get_model_output_path("pretraining", "guac-3b")
hf_cache = get_huggingface_cache_dir()
```

## Who Uses It

- `guac/core/cache_utils.py` (`get_cache_directory`): tokenized-dataset caches
- `guac/core/pipeline_utils.py` (`get_output_directory`): default output directories of the three stages
- `guac/pipeline/pretrain.py` and `train_tokenizer.py`: the default tokenizer location
- `guac/core/dataset_utils.py`: the `dataset_configs` section (Hugging Face datasets by name)

## Migration from Old Paths

The path manager includes backward compatibility for legacy paths. Existing paths in the configuration are still accessible through `get_legacy_path()` method:

```python
pm = get_path_manager()
old_tokenizer_path = pm.get_legacy_path("tokenizers", "char_arithmetic")
```

However, it's recommended to migrate to the new organized structure.

## Benefits

1. **Centralized Configuration**: All paths configured in one place
2. **Consistent Organization**: Logical hierarchy for all outputs
3. **SCRATCH_DIR Integration**: Everything under configurable scratch directory
4. **Variable Expansion**: Support for `${var}` references in paths
5. **No Empty Directories**: getters only compute paths; a directory appears when something is saved there, so its existence means it has content (`get_temp_dir` and `ensure_dir_exists` do create directories)
6. **Backward Compatibility**: Legacy paths still accessible

## Cleanup

The Makefile has been updated to clean the new directory structure:

```bash
make clean-cache    # Removes tokenized dataset caches
make clean-models   # Removes trained models
make clean          # Removes all outputs
```

## Testing

Test the path manager implementation:

```bash
python -m pytest tests/test_path_manager.py
```

This will validate that all paths are correctly organized under `SCRATCH_DIR`.