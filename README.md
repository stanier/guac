# GUAC: General Understanding and Computational Architecture

GUAC is a research framework for training and experimenting with heterogeneous transformer architectures, featuring mixed layer types and novel prompt formatting designed for complex sequence-to-sequence tasks.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch (CUDA-capable GPU recommended)
- HuggingFace `transformers`

### Installation
```bash
git clone <repository>
cd guac
make install
# (copies site_config.example.yml -> site_config.yml if it doesn't exist yet)
```

> Note: `make install` runs `pip install -r requirements.txt`, but this repo does not currently ship a `requirements.txt`. Create one (or install `torch`/`transformers`/`pyyaml`/`sentencepiece`/`jinja2` manually) before running `make install`.

Then edit `site_config.yml` with your paths and dataset preferences — this file is git-ignored so each environment can have its own local configuration.

### Basic Training
```bash
# Recommended: three-stage pipeline (pre-training -> mid-training -> post-training)
make pipeline                 # Full pipeline, default nano config
make pipeline CONFIG=atto     # With a specific model size
make pipeline TASK=arithmetic # With a mid-training task specialization
make pipeline DEBUG=true      # Smaller datasets for a fast debug run

# Individual stages
make pretrain
make midtrain
make posttrain

# Legacy shortcuts (still wired to the pipeline scripts)
make debug   # nano config, debug mode
make stable  # atto config
```

## 🏗️ Architecture Overview

GUAC introduces **Mixed Layer Architectures** - models that use different computational patterns within the same network. For example, the smaller presets in `model_configs.yaml` repeat a `conv, conv, mamba2, conv` block, while larger presets add mixture-of-experts and mixture-of-heads layers on top.

### Supported Layer Types
Defined in `scripts/training/core/custom_architectures.py`:
- **`full_attention`**: Standard multi-head attention
- **`flash_attention`**: Memory-efficient, tiling-based attention
- **`conv`**: 1D convolution for local pattern recognition
- **`mamba2`**: Selective state space modeling
- **`based`**: Linear attention via feature maps
- **`mlp_only`**: Feed-forward only (no sequence mixing)
- **`moh_attention`**: Mixture-of-Heads attention
- **`shared_expert_moe`**: Mixture-of-Experts with shared experts

## 📁 Project Structure

```
guac/
├── README.md                     # This file
├── CLAUDE.md                     # Guidance for Claude Code
├── Makefile                      # Pipeline commands
├── model_configs.yaml            # Model size presets (yocto -> zetta)
├── architecture_configs.yaml     # Architecture definitions (mixed_layer, llama, qwen2, granite, ...)
├── site_config.example.yml       # Template for local site_config.yml (git-ignored)
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md
│   ├── TRAINING.md
│   ├── CONTRIBUTING.md
│   ├── API.md
│   ├── PATH_ORGANIZATION.md
│   └── DATASET_DOWNLOAD.md
├── jobscripts/                   # Slurm job scripts (current + legacy)
├── scripts/
│   ├── training/
│   │   ├── pipeline/            # pretrain.py, midtrain.py, posttrain.py, tokenize_datasets.py
│   │   ├── tokenizer/           # Tokenizer training pipeline
│   │   ├── core/                # custom_architectures.py, callbacks.py, dataset_utils.py,
│   │   │                        # path_manager.py, prompt_templates.py, matryoshka_layers.py
│   │   ├── finetune/
│   │   └── slurm/
│   ├── datasets/                # Dataset generation (gen_arithm.py, ingest_dialogue*.py)
│   ├── legacy/                  # Superseded scripts, kept for reference
│   └── contrib/
├── templates/                    # Jinja2 prompt templates
├── tests/                        # Unit tests (see `make test`)
└── scratch/                      # Generated artifacts: checkpoints, tokenizers, tokenized datasets (git-ignored)
```

## 🎯 Key Features

### 1. **Mixed Layer Architectures**
- Combine different computational patterns in a single model via `MixedLayerConfig`
- Enables studying different inductive biases within one model

### 2. **Novel Prompt System**
- Jinja2-based templating (`scripts/training/core/prompt_templates.py`)
- Custom tokens for multi-sequence boundaries and inference control

### 3. **Flexible Configuration**
- Scalable model size progression: `yocto` (512 hidden, 4 layers) up through `zetta` (8192 hidden, 64 layers) — see the full table below
- Environment-agnostic `site_config.yml`, resolved via `PathManager`

### 4. **Advanced Training Features**
- Gradient checkpointing for memory efficiency
- Dataset caching for faster iteration
- Live inference during training
- Streaming dataset support

## 📖 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)**: Detailed architecture explanations
- **[Training Guide](docs/TRAINING.md)**: Training workflows and best practices
- **[API Reference](docs/API.md)**: Code API documentation
- **[Contributing](docs/CONTRIBUTING.md)**: How to contribute to GUAC

## 🧪 Research Goals

GUAC enables research into:

1. **Heterogeneous Architectures**: How do different layer types interact?
2. **Computational Complexity**: Optimal ordering of computation patterns
3. **Sequence-to-Sequence Learning**: Multi-modal and multi-sequence tasks
4. **Efficient Attention**: Comparing attention mechanisms within one model
5. **Mathematical Reasoning**: Arithmetic and symbolic computation tasks

## 📊 Model Configurations

GUAC provides pre-configured model sizes in `model_configs.yaml` (all use the `mixed_layer` architecture unless noted):

| Config    | Hidden Size | Layers | Attention Heads |
|-----------|-------------|--------|------------------|
| `yocto`   | 512         | 4      | 4                |
| `zepto`   | 1024        | 8      | 8                |
| `atto`    | 1536        | 12     | 12               |
| `femto`   | 2048        | 16     | 16               |
| `pico`    | 2560        | 20     | 20               |
| `nano`    | 3072        | 24     | 24               |
| `micro`   | 3584        | 28     | 28               |
| `milli`   | 4096        | 32     | 32               |
| `base`    | 4608        | 36     | 36               |
| `kilo`    | 5120        | 40     | 40               |
| `mega`    | 5632        | 44     | 44               |
| `giga`    | 6144        | 48     | 48               |
| `tera`    | 6656        | 52     | 52               |
| `peta`    | 7168        | 56     | 56               |
| `exa`     | 7680        | 60     | 60               |
| `zetta`   | 8192        | 64     | 64               |

There's also a `yocto_1080ti` variant of `yocto` tuned for smaller GPUs. `nano` is the default development config referenced throughout `CLAUDE.md` and the `Makefile`.

## 🔄 Workflows

### Training a Mixed Architecture Model
```bash
# Full pipeline with cached/tokenized datasets (default)
make pipeline CONFIG=nano

# Force re-tokenization
make pipeline FORCE_RETOKENIZE=true

# Different mid-training task specialization
make pipeline TASK=code
make pipeline TASK=reasoning

# Debug mode with smaller datasets
make pipeline DEBUG=true
```

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

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on:
- Code style and standards
- Adding new layer types
- Submitting experiments
- Documentation improvements

## 📄 License

MIT License — Copyright (c) 2026 Keyton Stanier. See [LICENSE](LICENSE) for the full text.

## 🙏 Acknowledgments

GUAC builds upon work from:
- **Mamba-2**: Selective state space models
- **BASED**: Linear attention via feature maps
- **Flash Attention**: Memory-efficient attention
- **Transformers**: HuggingFace ecosystem
