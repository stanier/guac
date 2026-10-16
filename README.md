# Guac

Guac is an experimental research codebase for training and experimenting with heterogeneous transformer architectures, featuring mixed layer types and a custom prompt formatting system designed for complex sequence-to-sequence tasks. It's active research code, not a polished library — expect rough edges.

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch (CUDA-capable GPU recommended)
- HuggingFace `transformers`

### Installation
```bash
git clone github.com/stanier/guac
cd guac
make install
# (copies site_config.example.yml -> site_config.yml if it doesn't exist yet)
```

Then edit `site_config.yml` with your paths and dataset preferences — this file is git-ignored so each environment can have its own local configuration.

### Basic Training
```bash
# Recommended: three-stage pipeline (pre-training -> mid-training -> post-training)
make pipeline                 # Full pipeline, default guac-3b config
make pipeline CONFIG=guac-590m     # With a specific model size
make pipeline TASK=arithmetic # With a mid-training task specialization
make pipeline DEBUG=true      # Smaller datasets for a fast debug run

# Individual stages
make pretrain
make midtrain
make posttrain

# Quick smoke test: cap pre-training at a handful of steps.
# Training length comes from the config (max_steps if set, else num_train_epochs);
# DEBUG=true also shrinks the dataset and caps training at 20 steps; MAX_STEPS overrides both.
make pretrain CONFIG=guac-106m DEBUG=true MAX_STEPS=5

# Batch size, grad accumulation and sequence length come from the config as given
# (pretrain.py prints the values it uses); override them per run with PRETRAIN_ARGS
make pretrain PRETRAIN_ARGS="--batch-size 8 --grad-accum 4 --max-length 512"

# Legacy shortcuts (still wired to the pipeline scripts)
make debug   # guac-3b config, debug mode
make stable  # guac-590m config
```

## Architecture Overview

Guac introduces **Mixed Layer Architectures** - models that use different computational patterns within the same network. For example, the smaller presets in `model_configs.yaml` repeat a `conv, conv, mamba2, conv` block, while larger presets add mixture-of-experts and mixture-of-heads layers on top.

### Supported Layer Types
Defined in `scripts/training/core/custom_architectures.py`:
- **`full_attention`**: Standard multi-head attention
- **`flash_attention`**: Memory-efficient, tiling-based attention
- **`conv`**: 1D convolution for local pattern recognition
- **`mamba2`**: Selective state space modeling
- **`based`**: BASED linear attention (2nd-order Taylor feature map), fixed-size state
- **`sliding_window_attention`**: Softmax attention over the last `sliding_window_size` positions
- **`mlp_only`**: Feed-forward only (no sequence mixing)
- **`moh_attention`**: Mixture-of-Heads attention
- **`shared_expert_moe`**: Mixture-of-Experts with shared experts

## Key Features

### 1. **Mixed Layer Architectures**
- Combine different computational patterns in a single model via `MixedLayerConfig`
- Enables studying different inductive biases within one model

### 2. **Custom Prompt System**
- Jinja2-based templating (`scripts/training/core/prompt_templates.py`)
- Custom tokens for multi-sequence boundaries and inference control

### 3. **Flexible Configuration**
- Scalable model size progression: `guac-106m` (512 hidden, 4 layers) up through `guac-48b` (8192 hidden, 64 layers) — see the full table below
- Environment-agnostic `site_config.yml`, resolved via `PathManager`

### 4. **Advanced Training Features**
- Gradient checkpointing for memory efficiency
- Dataset caching for faster iteration
- Live inference during training
- Streaming dataset support

## Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)**: Detailed architecture explanations
- **[Training Guide](docs/TRAINING.md)**: Training workflows and best practices
- **[API Reference](docs/API.md)**: Code API documentation
- **[Contributing](docs/CONTRIBUTING.md)**: How to contribute to Guac

## Research Questions

Guac is built to explore:

1. **Heterogeneous Architectures**: How do different layer types interact?
2. **Computational Complexity**: Optimal ordering of computation patterns
3. **Sequence-to-Sequence Learning**: Multi-modal and multi-sequence tasks
4. **Efficient Attention**: Comparing attention mechanisms within one model
5. **Mathematical Reasoning**: Arithmetic and symbolic computation tasks

## Model Configurations

Guac provides pre-configured model sizes in `model_configs.yaml` (all use the `mixed_layer` architecture unless noted):

| Config        | Hidden Size | Layers | Attention Heads  |
|---------------|-------------|--------|------------------|
| `guac-106m`   | 512         | 4      | 4                |
| `guac-280m`   | 1024        | 8      | 8                |
| `guac-590m`   | 1536        | 12     | 12               |
| `guac-1.1b`   | 2048        | 16     | 16               |
| `guac-1.9b`   | 2560        | 20     | 20               |
| `guac-3b`     | 3072        | 24     | 24               |
| `guac-4.5b`   | 3584        | 28     | 28               |
| `guac-6.5b`   | 4096        | 32     | 32               |
| `guac-9b`     | 4608        | 36     | 36               |
| `guac-12b`    | 5120        | 40     | 40               |
| `guac-16b`    | 5632        | 44     | 44               |
| `guac-21b`    | 6144        | 48     | 48               |
| `guac-26b`    | 6656        | 52     | 52               |
| `guac-32b`    | 7168        | 56     | 56               |
| `guac-39b`    | 7680        | 60     | 60               |
| `guac-48b`    | 8192        | 64     | 64               |

There's also a `guac-106m-1080ti` variant of `guac-106m` tuned for smaller GPUs. `guac-3b` is the default development config referenced throughout `CLAUDE.md` and the `Makefile`.

## Workflows

### Training a Mixed Architecture Model
```bash
# Full pipeline with cached/tokenized datasets (default)
make pipeline CONFIG=guac-3b

# Force re-tokenization
make pipeline FORCE_RETOKENIZE=true

# Different mid-training task specialization
make pipeline TASK=code
make pipeline TASK=reasoning

# Debug mode with smaller datasets
make pipeline DEBUG=true

# Cap pre-training at a fixed step count (useful for quick smoke tests)
make pretrain MAX_STEPS=50
```

Pre-training datasets (`fineweb`, `fineweb-edu`, `finewiki`, plus `wikitext-103`/`openwebtext`) are resolved via the `pretraining_datasets` section of `site_config.yml`, so `DATASETS=` accepts any key defined there.

### Tokenizer & Dataset Preparation
```bash
make train-tokenizer VOCAB_SIZE=60000 DATASETS=fineweb
make tokenize-datasets TOKENIZER_PATH=scratch/tokenizers/sentencepiece/shared_60k
```

### Utilities
```bash
make status   # Pipeline status and available models
make configs  # Show available model configs
make cache    # Show cached dataset info
make clean    # Clean all outputs
```

## Attributions

Guac's mixed-layer architectures build on the following work:

- Dao, T. & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2). arXiv:2405.21060.
- Arora, S., Eyuboglu, S., Timalsina, A., Johnson, I., Poli, M., Zou, J., Rudra, A., & Ré, C. (2024). *Simple linear attention language models balance the recall-throughput tradeoff* (BASED). arXiv:2402.18668.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. arXiv:2205.14135.
- Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing* (HuggingFace `transformers`). arXiv:1910.03771.
