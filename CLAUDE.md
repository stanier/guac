# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

GUAC is a research framework for training and experimenting with **heterogeneous transformer architectures** — models where different layers use entirely different computational mechanisms (MLP-only, Conv1D, Mamba-2, BASED, Flash Attention, standard attention) stacked together in a single network. The research goal is understanding how mixed layer types interact and what inductive biases they contribute.

## Commands

**Setup:**
```bash
make install
```

**Three-stage training pipeline (primary workflow):**
```bash
make pipeline                         # Full pre→mid→post training
make pipeline CONFIG=nano             # With specific model size
make pipeline TASK=arithmetic         # With task specialization
make pipeline DEBUG=true              # Smaller datasets for debugging
make pipeline FORCE_RETOKENIZE=true   # Force re-tokenization
```

**Individual stages:**
```bash
make pretrain
make midtrain
make posttrain
```

**Tests:**
```bash
make test
python tests/run_tests.py
python tests/run_tests.py --test path_manager   # Single test module
```

**Utilities:**
```bash
make status      # Pipeline status
make cache       # Show cached dataset info
make configs     # Show available model configs
make clean       # Clean all outputs
```

## Architecture

### Mixed Layer Composition

The core novelty is `MixedLayerConfig` in `scripts/training/core/custom_architectures.py`, which allows heterogeneous layer stacks. Supported layer types:

- **MLP-only**: Point-wise feed-forward, O(d²), no sequence mixing
- **Conv1D**: Local pattern recognition with configurable kernel size
- **Mamba-2**: Selective state space — efficient long-range dependencies
- **BASED**: Linear attention via feature maps, O(L·F)
- **Flash Attention**: Tiling-based full attention, memory-efficient
- **Standard Attention**: Multi-head with grouped query support

### Three-Stage Training Pipeline

| Stage | Script | Purpose | Default Datasets |
|-------|--------|---------|-----------------|
| Pre-training | `scripts/training/pipeline/pretrain.py` | Foundation LM | FineWeb, FineWeb-EDU, WikiText-103 |
| Mid-training | `scripts/training/pipeline/midtrain.py` | Task specialization | Task-specific (arithmetic, code, reasoning) |
| Post-training | `scripts/training/pipeline/posttrain.py` | Instruction tuning | OpenOrca, Alpaca, UltraChat |

### Key Source Files

- `scripts/training/core/custom_architectures.py` — All layer type implementations
- `scripts/training/core/matryoshka_layers.py` — Layer type definitions
- `scripts/training/core/callbacks.py` — Training monitoring, profiling, live inference
- `scripts/training/core/dataset_utils.py` — Dataset loading, tokenization, caching
- `scripts/training/core/path_manager.py` — Centralized path resolution from `site_config.yml`
- `scripts/training/core/prompt_templates.py` — Jinja2-based prompt system with custom tokens

### Configuration

**`model_configs.yaml`** — Model size presets. Sizes range from `yocto` (512 hidden, 4 layers) through `nano` (384 hidden, 18 layers, development default) to `chungus` (8192 hidden, 80 layers). Each preset includes hyperparameters (lr, batch size, warmup ratio).

**`architecture_configs.yaml`** — Architecture definitions (`mixed_layer`, `basic_transformer`, `llama`, `qwen2`, `granite`). Controls layer type progressions, normalization, initialization ranges.

**`site_config.yml`** — Environment-specific paths and dataset configurations (git-ignored; copy from `site_config.example.yml`). Controls `scratch_dir` layout, HuggingFace dataset streaming settings, and which datasets are used per stage.

All output paths (tokenized datasets, checkpoints, models) resolve through `PathManager` from `site_config.yml`, not hardcoded. The `scratch/` directory holds all generated artifacts.

### Dataset Management

`datasets_download.csv` is a manifest for `hf_partial_download.py`. Supports partial downloads by size (`100MB`, `50K`, `10%`).

```bash
make download-datasets
python hf_partial_download.py --csv datasets_download.csv
```

## Testing

Tests live in `tests/`. They cover path manager, architecture config loading, site config parsing, tokenizer, and mixture-of-heads layers. No external services required for unit tests.
