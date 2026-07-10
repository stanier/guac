# GUAC Training Directory Structure

This directory contains all training-related scripts organized into logical subdirectories.

## Directory Structure

```
scripts/training/
├── core/                    # Core modules (shared across all training stages)
│   ├── callbacks.py         # Training callbacks (progress, profiling, etc.)
│   ├── custom_architectures.py  # GUAC model architectures (MixedLayer, MoE, etc.)
│   ├── prompt_templates.py  # Prompt template utilities
│   └── train_tokenizer.py   # Tokenizer training utilities
├── pipeline/                # Three-stage training pipeline (main scripts)
│   ├── pretrain.py          # Stage 1: Foundation language modeling
│   ├── midtrain.py          # Stage 2: Task specialization
│   ├── posttrain.py         # Stage 3: Instruction tuning/alignment
│   └── README.md            # Pipeline documentation
├── legacy/                  # Legacy/experimental scripts
│   ├── guac_arithmetic.py   # Old arithmetic training
│   ├── guac_general.py      # Old general training
│   ├── profile_training.py  # Training profiling tools
│   ├── sentencepiece_tokenizer.py  # Legacy tokenizer
│   ├── train_sentencepiece.py      # Legacy SP trainer
│   └── train_with_sentencepiece.py # Legacy training script
└── finetune/               # Specialized fine-tuning scripts
    └── train_guac0.py      # GUAC v0 fine-tuning
```

## Core Modules

### `core/custom_architectures.py`
- **MixedLayerForCausalLM**: Main GUAC model class
- **MixedLayerConfig**: Configuration for mixed architectures
- Supports: Mamba2, BASED, Conv1D, Attention, MoE, MoH
- Modern features: RMSNorm, Flash Attention 2, Gradient Checkpointing

### `core/callbacks.py`
- **CustomProgressCallback**: Enhanced progress tracking
- **ProfilingCallback**: Memory and performance profiling
- **LayerTimingCallback**: Per-layer timing analysis

### `core/train_tokenizer.py`
- SentencePiece tokenizer training
- Automatic special token integration
- Character-level fallback support

## Pipeline Scripts

### Three-Stage Training Pipeline

1. **`pipeline/pretrain.py`** - Foundation Language Modeling
   - Large-scale language data (WikiText-103, OpenWebText)
   - Basic language understanding and generation
   - No evaluation (for speed)

2. **`pipeline/midtrain.py`** - Task Specialization
   - Arithmetic, code, reasoning tasks
   - Domain-specific knowledge acquisition
   - Specialized datasets

3. **`pipeline/posttrain.py`** - Instruction Tuning & Alignment
   - Instruction following
   - Conversational capabilities
   - Safety and helpfulness alignment

## Usage

### Quick Start
```bash
# Complete pipeline
make pipeline

# Individual stages
make pretrain CONFIG=nano
make midtrain TASK=arithmetic
make posttrain MODE=instruction

# Debug mode
make pipeline-debug
```

### Configuration
- Model configs: `model_configs.yaml`
- Architecture configs: `architecture_configs.yaml`
- Site config: `site_config.yml`

### Features
- **Dataset caching**: Automatic tokenized dataset caching
- **Memory optimization**: 1080 Ti compatible settings
- **Modern architecture**: Mixed layers, MoE, Flash Attention
- **Flexible configuration**: YAML-based model configs

## Import Structure

All scripts use robust import fallbacks:
```python
try:
    from scripts.training.core.custom_architectures import MixedLayerForCausalLM
except ImportError:
    from ..core.custom_architectures import MixedLayerForCausalLM
except ImportError:
    from core.custom_architectures import MixedLayerForCausalLM
```

This ensures scripts work when run directly or as modules.