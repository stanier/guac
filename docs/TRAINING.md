# GUAC Training Guide

This guide covers training workflows, best practices, and troubleshooting for GUAC models.

## 🚀 Quick Start Training

### Basic Commands

```bash
# Train with default mixed architecture
python scripts/training/pretrain/guac_general.py

# Train with specific model size
python scripts/training/pretrain/guac_general.py --model-config mixed

# Train arithmetic models
python scripts/training/pretrain/guac_arithmetic.py --config nano
```

### Command Line Options

```bash
# Dataset control
--datasets wikitext-103,fineweb-edu   # Specify datasets
--force-retokenize                    # Ignore cached tokenized datasets
--no-cache                           # Disable dataset caching entirely

# Model configuration
--model-config mixed                  # Use specific model config
--architecture mixed_layer            # Override architecture
--vocab-size 2000                    # Set tokenizer vocabulary size

# Template control
--use-templates                       # Enable prompt templates
--no-templates                       # Disable templates (default for pretraining)
--template-file custom.jinja          # Use custom template
--template-task "Custom task desc"    # Override task description

# Tokenizer options
--tokenizer-type character            # Character vs BPE tokenization
--load-tokenizer char_arithmetic      # Load pre-trained tokenizer
--tokenizer-path /path/to/tokenizer   # Explicit tokenizer path
```

## 📊 Model Configurations

### Pre-defined Sizes

| Config      | Use Case              | Hidden Size | Layers | Parameters | Memory   |
|-------------|-----------------------|-------------|--------|------------|----------|
| `iota0`     | Debugging             | 256         | 1      | ~65K       | <1GB     |
| `iota`      | Small experiments     | 256         | 16     | ~2M        | ~2GB     |
| `nano`      | Development           | 384         | 18     | ~8M        | ~4GB     |
| `micro`     | Small models          | 768         | 22     | ~25M       | ~8GB     |
| `milli`     | Medium experiments    | 896         | 21     | ~40M       | ~12GB    |
| `mixed`     | Mixed architecture    | 512         | 8      | ~48M       | ~6GB     |
| `centi`     | Production small      | 1536        | 28     | ~180M      | ~16GB    |
| `base`      | Production medium     | 2560        | 36     | ~1.3B      | ~24GB    |
| `chungus`   | Large scale           | 8192        | 80     | ~65B       | >80GB    |

### Custom Configuration

Add custom configs to `model_configs.yaml`:

```yaml
model_training_configs:
  my_custom:
    step_val: 100
    hidden_size: 640
    num_hidden_layers: 12
    num_attention_heads: 10
    num_kv_heads: 5
    learning_rate: 1e-4
    warmup_ratio: 0.1
    weight_decay: 0.01
    pdbs_train: 16          # Per-device batch size
    pdbs_eval: 16
    grad_accum_steps: 4
    num_train_epochs: 3
```

## 🗂️ Dataset Management

### Supported Datasets

**HuggingFace Datasets** (configured in `site_config.yml`):
- `fineweb-edu`: Educational web content (high-quality subset)
- `wikitext-103`: Wikipedia articles (clean, structured text)
- `openwebtext`: Web crawl data (diverse internet content)
- Custom datasets via configuration

**Fine-tuning Datasets**:
- `Alpaca`: 52k instruction-following examples (SFT)
- `OpenOrca`: 2.91M instruction examples with system prompts
- `Capybara`: 16k conversational examples

**Local Datasets**:
- Arithmetic: Generated via `gen_arithm.py`
- Custom: Place in configured dataset directories

### Dataset Caching

GUAC automatically caches tokenized datasets:

```bash
# Cache locations (configurable in site_config.yml)
./tokenized_wikitext-103_vocab_1000_no_templates/
./tokenized_fineweb-edu_vocab_1000_templates_default/
./tokenized_combined_vocab_1000_no_templates/
```

**Cache Management**:
```bash
# Force re-tokenization
python scripts/training/pretrain/guac_general.py --force-retokenize

# Disable caching
python scripts/training/pretrain/guac_general.py --no-cache

# Clean cache manually
rm -rf ./tokenized_*
```

### Adding Custom Datasets

1. **Configure in site_config.yml**:
```yaml
dataset_configs:
  pretraining_datasets:
    my_dataset:
      dataset_name: "username/dataset-name"
      config_name: "default"
      streaming: false
      max_articles: 10000
```

2. **Use in training**:
```bash
python scripts/training/pretrain/guac_general.py --datasets my_dataset
```

## 🔧 Architecture Configuration

### Mixed Layer Setup

Configure layer types in `architecture_configs.yaml`:

```yaml
architecture_defaults:
  mixed_layer:
    layer_types: [
      "mlp_only",        # Layer 1: Simple
      "conv",            # Layer 2: Local
      "conv",            # Layer 3: More local
      "mamba2",          # Layer 4: State space
      "based",           # Layer 5: Efficient attention
      "mamba2",          # Layer 6: More state space
      "flash_attention", # Layer 7: Full attention
      "mlp_only"         # Layer 8: Final processing
    ]
```

### Layer Type Parameters

```yaml
# Mamba-2 configuration
mamba_d_state: 8           # State space dimension (higher = more memory)
mamba_d_conv: 3            # Convolution kernel size
mamba_expand: 1.5          # Expansion factor

# BASED configuration
based_feature_dim: 8       # Feature map size (trade-off: accuracy vs speed)
based_num_heads: 8         # Number of attention heads
based_num_key_value_heads: 4 # KV heads (≤ num_heads)

# Convolutional configuration
conv_kernel_size: 3        # 1D convolution kernel
conv_groups: 1             # Depthwise convolution groups

# Flash Attention configuration
flash_attn_causal: true    # Enable causal masking
flash_attn_softmax_scale: null # Auto-scale (1/√d)
```

## 🎯 Training Best Practices

### Memory Optimization

1. **Gradient Checkpointing**: Enabled by default in training scripts
2. **Mixed Precision**: Use FP16 training
3. **Batch Size Tuning**: Reduce batch size for longer sequences
4. **Layer Ordering**: Memory-intensive layers later in stack

### Numerical Stability

**Monitor for Issues**:
- Loss = 0.0 (model collapse)
- grad_norm = NaN (exploding gradients)
- Loss spikes (instability)

**Solutions**:
```bash
# Reduce learning rate
--learning-rate 1e-5

# Increase gradient clipping
# (configured in training script: max_grad_norm: 1.0)

# Use more stable layer arrangements
layer_types: ["mlp_only", "conv", "full_attention", "mlp_only"]
```

### Hyperparameter Guidelines

**Learning Rate**:
- Small models (< 10M): 1e-3 to 1e-4
- Medium models (10M-100M): 1e-4 to 5e-5
- Large models (> 100M): 1e-4 to 1e-5

**Batch Size**:
- Start with largest that fits in memory
- Use gradient accumulation for effective larger batches
- Mixed architectures may need smaller batches

**Warmup**:
- 5-10% of total steps for small models
- 1-5% for larger models
- Mixed architectures benefit from longer warmup

## 📈 Monitoring and Evaluation

### Training Metrics

**Key Metrics to Watch**:
- `loss`: Should decrease steadily
- `grad_norm`: Should be finite and stable
- `learning_rate`: Should follow schedule
- Memory usage: Monitor GPU memory

**Live Inference**: Training scripts show periodic inference examples during training.

### Checkpointing

**Automatic Checkpointing**:
- Every `save_steps` (default: 5 × log_steps)
- Best model based on evaluation loss
- Final model at training completion

**Manual Checkpointing**:
```python
# In training script
trainer.save_model("/path/to/checkpoint")
```

### Evaluation

**During Training**:
- Validation loss (if validation set available)
- Live inference examples
- Gradient norms and learning rate

**Post-Training**:
- Perplexity on held-out data
- Task-specific evaluations
- Generation quality assessment

## 🛠️ Troubleshooting

### Common Issues

**1. Loss = 0.0, grad_norm = NaN**
```bash
# Symptoms: Model produces zeros, gradients explode
# Causes: Numerical instability in custom layers

# Solutions:
# - Reduce learning rate
# - Check Mamba-2 parameters (d_state, expand factor)
# - Use simpler layer arrangement initially
# - Enable gradient clipping
```

**2. Out of Memory (OOM)**
```bash
# Solutions:
# - Reduce batch size: pdbs_train and pdbs_eval
# - Increase gradient accumulation: grad_accum_steps
# - Enable gradient checkpointing (default)
# - Use fewer attention layers
# - Reduce sequence length
```

**3. Very Slow Training**
```bash
# Solutions:
# - Check if Flash Attention is installed
# - Reduce sequence length
# - Use fewer complex layers (Mamba-2, BASED)
# - Enable dataset caching (default)
```

**4. Poor Convergence**
```bash
# Solutions:
# - Increase learning rate gradually
# - Extend warmup period
# - Check data quality and preprocessing
# - Try simpler architecture first
```

### Debug Mode

For debugging, use minimal configurations:

```bash
# Tiny model for debugging
python scripts/training/pretrain/guac_general.py --model-config iota0

# Simple architecture
# Temporarily modify architecture_configs.yaml:
layer_types: ["mlp_only", "full_attention", "mlp_only"]
```

### Performance Profiling

```python
# Add to training script for profiling
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler')
) as prof:
    # Training step
    trainer.train()
```

## 🔄 Advanced Workflows

### Multi-GPU Training

```bash
# Data parallel (multiple GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 scripts/training/pretrain/guac_general.py

# Model parallel (for very large models)
# Configure in training script: device_map="auto"
```

### Resuming Training

```bash
# Training automatically resumes from last checkpoint in output_dir
python scripts/training/pretrain/guac_general.py --model-config mixed
```

### Fine-tuning

```bash
# Load pre-trained model for fine-tuning
python scripts/training/finetune/train_guac0.py \
  --model-path /path/to/checkpoint \
  --task-specific-config
```

### Experiment Management

**Organized Experiments**:
```bash
# Use descriptive output directories
export EXPERIMENT="mixed_arch_v1_$(date +%Y%m%d)"
python scripts/training/pretrain/guac_general.py \
  --model-config mixed \
  --datasets wikitext-103 \
  --output-dir ./experiments/$EXPERIMENT
```

**Version Control**:
- Commit code changes before experiments
- Tag successful experiment checkpoints
- Document hyperparameter changes

This guide should get you started with GUAC training. For specific issues or advanced use cases, refer to the API documentation or open an issue on the repository.