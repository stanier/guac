# Guac Training Pipeline

A comprehensive three-stage training pipeline for Guac language models.

## Training Stages

### 1. **Pre-training** (`guac_pretrain.py`)
Foundation training on large-scale language data.

```bash
# Basic pre-training
python guac_pretrain.py --model-config guac-3b --dataset wikitext-103

# Multi-dataset pre-training
python guac_pretrain.py --model-config guac-1.1b --datasets wikitext-103,openwebtext

# Continue from checkpoint
python guac_pretrain.py --model-config guac-590m --resume-from-checkpoint /path/to/checkpoint
```

**Purpose**: Teach general language understanding, grammar, world knowledge
**Data**: WikiText-103, OpenWebText, Common Crawl, etc.
**Output**: Base language model with general capabilities

### 2. **Mid-training** (`guac_midtrain.py`)
Task specialization on domain-specific data.

```bash
# Arithmetic specialization
python guac_midtrain.py --base-model ./models/pretrain_nano --task arithmetic

# Code specialization
python guac_midtrain.py --base-model ./models/pretrain_femto --task code

# Reasoning specialization
python guac_midtrain.py --base-model ./models/pretrain_atto --task reasoning

# Custom domain
python guac_midtrain.py --base-model ./models/pretrain_pico --task custom --datasets /path/to/domain/data
```

**Purpose**: Specialize model for specific domains or tasks
**Data**: Math problems, code repositories, reasoning datasets, domain-specific text
**Output**: Specialized model with enhanced domain performance

### 3. **Post-training** (`guac_posttrain.py`)
Instruction tuning and alignment.

```bash
# Instruction tuning
python guac_posttrain.py --base-model ./models/midtrain_arithmetic --mode instruction

# Supervised fine-tuning on conversations
python guac_posttrain.py --base-model ./models/pretrain_nano --mode sft --datasets alpaca

# Custom instructions
python guac_posttrain.py --base-model ./models/midtrain_code --mode instruction --datasets /path/to/instructions.json
```

**Purpose**: Make model helpful, harmless, honest through instruction following
**Data**: Instruction datasets (Alpaca, Dolly), conversation logs, human feedback
**Output**: Aligned model ready for deployment

## Model Configurations

16 model sizes from edge to datacenter:

| Config | Parameters | Use Case |
|--------|------------|----------|
| `guac-3b` | 1M | IoT devices, edge computing |
| `guac-1.9b` | 3M | Mobile phones, embedded |
| `guac-1.1b` | 14M | Laptops, local inference |
| `guac-590m` | 35M | Desktop applications |
| `guac-280m` | 75M | Single GPU training |
| `guac-106m` | 120M | Professional workstations |
| `guac-4.5b` | 250M | Multi-GPU setups |
| `guac-6.5b` | 500M | Small clusters |
| `centi` | 1.5B | GPT-2 scale |
| `deci` | 3.5B | GPT-3 scale |
| `guac-9b` | 7B | LLaMA-7B scale |
| `guac-12b` | 15B | GPT-3.5 scale |
| `guac-16b` | 35B | GPT-4 scale |
| `guac-21b` | 80B | GPT-4 Turbo scale |
| `guac-26b` | 250B | Future models |
| `chungus` | 1T+ | Theoretical AGI scale |

## Features

- **Modern Architecture**: RMSNorm, Flash Attention 2, Mixture-of-Experts, Mixture-of-Heads
- **Mixed Layers**: Mamba2, Convolution, BASED, Attention layers in same model
- **Flexible Training**: Supports all scales from 1M to 1T+ parameters
- **Automatic Tokenization**: SentencePiece with domain-specific vocabularies
- **Comprehensive Callbacks**: Progress tracking, profiling, layer timing analysis

## Quick Start

```bash
# 1. Pre-train a small model
python guac_pretrain.py --model-config guac-3b --debug

# 2. Specialize for arithmetic
python guac_midtrain.py --base-model ./models/pretrain_nano --task arithmetic --debug

# 3. Add instruction following
python guac_posttrain.py --base-model ./models/midtrain_arithmetic_pretrain_nano --mode instruction --debug
```

## Advanced Usage

### Custom Training Pipeline
```bash
# Full pipeline with profiling
python guac_pretrain.py --model-config guac-1.1b --datasets wikitext-103 --enable-profiling
python guac_midtrain.py --base-model ./models/pretrain_femto --task code --enable-profiling
python guac_posttrain.py --base-model ./models/midtrain_code_pretrain_femto --mode sft --datasets alpaca --enable-profiling
```

### Multi-Stage Training
```bash
# Pre-train -> Math specialization -> Instruction tuning
python guac_pretrain.py --model-config guac-590m --datasets wikitext-103,openwebtext
python guac_midtrain.py --base-model ./models/pretrain_atto --task arithmetic --num-arithmetic-samples 100000
python guac_posttrain.py --base-model ./models/midtrain_arithmetic_pretrain_atto --mode instruction --datasets alpaca,dolly
```

## Configuration

Edit `model_configs.yaml` to:
- Add new model sizes
- Modify layer architectures
- Adjust training hyperparameters
- Enable/disable features (Flash Attention, RMSNorm, MoE)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Flash Attention 2 (optional): `pip install flash-attn --no-build-isolation`
- Datasets library
- SentencePiece

## Performance Tips

- **Memory**: Use gradient checkpointing for large models
- **Speed**: Enable Flash Attention 2, use FP16/BF16
- **Quality**: Use progressive training (small→large context), curriculum learning
- **Debugging**: Start with `--debug` flag to test on small data subsets