# Guac API Reference

This document provides detailed API reference for Guac's core components, training scripts, and configuration systems.

## 🏗️ Core Architecture Classes

### MixedLayerConfig

Configuration class for mixed layer architectures.

```python
class MixedLayerConfig(PretrainedConfig):
    """Configuration for MixedLayerForCausalLM models."""

    model_type = "mixed_layer"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        layer_types: List[str] = None,
        ffn_types: Optional[List[str]] = None,
        ffn_hidden_sizes: Optional[List[int]] = None,
        # Layer-specific parameters
        mamba_d_state: int = 128,
        mamba_d_conv: int = 4,
        mamba_expand: float = 2.0,
        based_feature_dim: int = 16,
        based_num_heads: Optional[int] = None,            # num_attention_heads
        based_num_key_value_heads: Optional[int] = None,  # num_kv_heads
        based_chunk_size: int = 64,
        sliding_window_size: int = 64,
        # Position encoding
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        use_absolute_position_embeddings: bool = False,
        max_position_embeddings: int = 4096,   # rows of the absolute table, if used
        conv_kernel_size: int = 3,
        conv_groups: int = 1,
        # Standard parameters
        layer_norm_epsilon: float = 1e-6,
        **kwargs
    ):
```

**Parameters**:
- `layer_types` (List[str]): Mixer for each layer. Options: `"full_attention"`, `"flash_attention"` (same mixer), `"sliding_window_attention"`, `"conv"`, `"mamba2"`, `"based"`, `"moh_attention"`, `"mlp_only"`, `"shared_expert_moe"`
- `ffn_types` (List[str]): FFN for each layer: `"swiglu"`, `"moe"` or `"none"`. Defaults to `"moe"` for `shared_expert_moe` and `"swiglu"` elsewhere
- `ffn_hidden_sizes` (List[int]): SwiGLU width for each layer. Defaults to 2/3 of `intermediate_size`
- `mamba_d_state` (int): State space dimension for Mamba-2 layers
- `mamba_scan_impl` (str): `"torch"` (default, any device) or `"mamba_ssm"` (mamba_ssm's fused kernel for CUDA inputs)
- `mamba_expand` (float): Expansion factor for Mamba-2 internal dimension
- `based_feature_dim` (int): Query/key size per head in BASED layers, before the Taylor feature map expands it to 1 + f + f²
- `sliding_window_size` (int): Positions each token sees in `sliding_window_attention` layers, itself included
- `use_rope` (bool): Rotary position embeddings in the softmax-attention mixers (`ROTARY_LAYER_TYPES`). Needs an even head size
- `rope_theta` (float): RoPE base frequency
- `use_absolute_position_embeddings` (bool): Add a learned table of `max_position_embeddings` positions to the token embeddings (off by default)
- `conv_kernel_size` (int): Kernel size for convolutional layers

### MixedLayerForCausalLM

Main model class implementing mixed layer architecture.

```python
class MixedLayerForCausalLM(PreTrainedModel, GenerationMixin):
    """Mixed layer transformer for causal language modeling."""

    config_class = MixedLayerConfig

    def __init__(self, config: MixedLayerConfig):
        """Initialize mixed layer model."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithCrossAttentions:
        """Forward pass through mixed layer model."""

    def generate(self, **kwargs):
        """Generate text using the model."""

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
```

**Methods**:
- `forward()`: Standard transformer forward pass with mixed layers
- `generate()`: Text generation with configurable parameters
- `gradient_checkpointing_enable/disable()`: Memory optimization controls

## 🧱 Layer Types

### MixedTransformerLayer

The block shared by every layer type. Only the mixer depends on the layer type.

```python
class MixedTransformerLayer(nn.Module):
    def __init__(self, config: MixedLayerConfig, layer_idx: int):
        """Builds layer `layer_idx`: config.layer_types[layer_idx] picks the mixer,
        config.ffn_types[layer_idx] and config.ffn_hidden_sizes[layer_idx] the FFN."""

    def forward(self, hidden_states, attention_mask=None, cache=None):
        # h = x + mixer(mixer_norm(x))   (no mixer for mlp_only, shared_expert_moe)
        # y = h + ffn(ffn_norm(h))       (no FFN when the FFN type is "none")
```

### Mixers

Each takes `(hidden_states, attention_mask=None, cache=None)` and returns a tensor
of the same shape, `[batch, seq_len, hidden_size]`. They are registered in `MIXERS`.
`attention_mask` is `[batch, total_len]` (it includes cached tokens during
generation); padding never reaches real tokens, left padding included. `cache` is
the layer's `LayerCache` during generation, else `None`.

### Generation

`model.generate(...)` keeps per-layer state in a `MixedLayerCache` (returned as
`past_key_values`) and feeds one new token per step. Attention layers cache keys
and values (sliding-window layers only the last `sliding_window_size - 1`); conv,
Mamba-2 and BASED layers keep fixed-size recurrent states. Batched prompts should
be left-padded with an `attention_mask`. The cache is on by default
(`config.use_cache`) and is never built in training or when `labels` are given.
`tests/test_generation.py` checks that token-by-token decoding matches a full
forward pass for every layer type.

| Layer type | Mixer | Config parameters |
|------------|-------|-------------------|
| `full_attention`, `flash_attention` | `GroupedQueryAttention` (`F.scaled_dot_product_attention`) | `num_attention_heads`, `num_kv_heads` |
| `conv` | `ConvMixer` (causal Conv1D) | `conv_kernel_size`, `conv_groups` |
| `mamba2` | `ConfigMamba2Mixer` (`Mamba2Mixer` from `mamba2.py`) | `mamba_d_state`, `mamba_d_conv`, `mamba_expand`, `mamba_headdim`, `mamba_ngroups`, `mamba_chunk_size` |
| `sliding_window_attention` | `SlidingWindowAttention` (`GroupedQueryAttention` over a local window) | `num_attention_heads`, `num_kv_heads`, `sliding_window_size` |
| `based` | `BasedMixer` (Taylor-feature linear attention, `linear_attention.py`) | `based_feature_dim`, `based_num_heads`, `based_num_key_value_heads`, `based_chunk_size` |
| `moh_attention` | `MixtureOfHeadAttention` | `moh_top_k`, `moh_temperature`, `moh_noise_std` |
| `mlp_only`, `shared_expert_moe` | none | |

### FFNs

- `SwiGLUFeedForward(hidden_size, intermediate_size)`: the default FFN.
- `SharedExpertMoE(config)`: shared plus routed SwiGLU experts (`num_experts`,
  `num_shared_experts`, `moe_top_k`, ...). Used where `ffn_types` says `moe`.
  `forward(hidden_states, attention_mask=None)` sets `aux_loss`, the Switch
  load-balancing loss (`moe_load_balancing_weight * N * sum_i f_i * P_i`).
  `MixedLayerForCausalLM` returns the sum over layers as `outputs.aux_loss`
  and adds it to `outputs.loss` in training mode only.

### Parameter helpers

```python
param_matched_ffn_hidden_sizes(config, reference_layer_type="full_attention") -> List[int]
    """SwiGLU widths that give every SwiGLU block the parameter count of a
    reference block; pass the result as ffn_hidden_sizes."""

block_parameter_counts(config) -> List[int]
    """Parameter count of each block, built on the meta device."""
```

## 🎯 Training Scripts

### guac_general.py

Main pretraining script for general datasets.

```python
def main():
    """Main training function with argument parsing."""

def setup_model_and_tokenizer(args):
    """
    Initialize model and tokenizer.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Model and tokenizer
    """

def setup_datasets(args, tokenizer):
    """
    Setup and tokenize datasets with caching.

    Args:
        args: Command line arguments
        tokenizer: Tokenizer for text processing

    Returns:
        Dataset: Concatenated training dataset
    """

def train_model(model, tokenizer, train_dataset, args):
    """
    Execute model training.

    Args:
        model: Model to train
        tokenizer: Text tokenizer
        train_dataset: Training data
        args: Training configuration
    """
```

**Command Line Options**:
```bash
python scripts/training/pretrain/guac_general.py [OPTIONS]

# Model configuration
--model-config TEXT          # Model size (guac-3b, guac-4.5b, mixed, etc.)
--architecture TEXT          # Architecture type (mixed_layer, etc.)
--vocab-size INTEGER         # Tokenizer vocabulary size

# Dataset options
--datasets TEXT              # Comma-separated dataset names
--force-retokenize          # Ignore cached tokenized datasets
--no-cache                  # Disable dataset caching

# Template options
--use-templates             # Enable prompt templates
--no-templates              # Disable templates (default)
--template-file TEXT        # Custom template file
--template-task TEXT        # Override task description

# Tokenizer options
--tokenizer-type TEXT       # character or bpe
--load-tokenizer TEXT       # Load pre-trained tokenizer
--tokenizer-path TEXT       # Explicit tokenizer path
```

### guac_arithmetic.py

Specialized training for arithmetic tasks.

```python
def main():
    """Main arithmetic training function."""

def generate_arithmetic_data(num_samples: int, complexity: str) -> List[str]:
    """
    Generate arithmetic problem datasets.

    Args:
        num_samples: Number of samples to generate
        complexity: Problem complexity (simple, medium, hard)

    Returns:
        List of arithmetic problems and solutions
    """
```

## 🔧 Configuration System

### Model Configurations

Model sizes are defined in `model_configs.yaml`:

```yaml
model_training_configs:
  config_name:
    # Model architecture
    hidden_size: 768                    # Model dimension
    num_hidden_layers: 12              # Number of layers
    num_attention_heads: 12            # Attention heads
    num_kv_heads: 4                    # Key-value heads (GQA)

    # Training parameters
    learning_rate: 1e-4                # Base learning rate
    warmup_ratio: 0.1                  # Warmup fraction
    weight_decay: 0.01                 # L2 regularization
    pdbs_train: 8                      # Per-device batch size
    pdbs_eval: 8                       # Evaluation batch size
    grad_accum_steps: 2                # Gradient accumulation
    num_train_epochs: 3                # Training epochs

    # Evaluation
    step_val: 100                      # Evaluation frequency

    # Optional layer types
    layer_types: ["mlp_only", "conv", "mamba2", "flash_attention"]
```

### Architecture Configurations

Architecture definitions in `architecture_configs.yaml`:

```yaml
# Default architecture to use
default_architecture: "mixed_layer"

# Available architectures
architectures:
  mixed_layer:
    config_class: "MixedLayerConfig"
    model_class: "MixedLayerForCausalLM"
    import_path: "scripts.training.custom_architectures"

# Architecture-specific defaults
architecture_defaults:
  mixed_layer:
    layer_types: [
      "mlp_only",
      "conv",
      "conv",
      "mamba2",
      "based",
      "mamba2",
      "flash_attention",
      "mlp_only"
    ]

    # Layer-specific parameters
    mamba_d_state: 8
    mamba_d_conv: 3
    mamba_expand: 1.5
    based_feature_dim: 8
    based_num_heads: 8
    based_num_key_value_heads: 4
    conv_kernel_size: 3
    conv_groups: 1
```

### Site Configuration

Environment settings in `site_config.yml`:

```yaml
# Dataset directories
dataset_configs:
  local_dataset_dir: "/path/to/datasets"
  cache_dir: "/path/to/cache"

  pretraining_datasets:
    wikitext-103:
      dataset_name: "wikitext"
      config_name: "wikitext-103-raw-v1"
      streaming: false
      max_articles: 50000

    fineweb-edu:
      dataset_name: "HuggingFaceFW/fineweb-edu"
      config_name: "sample-10BT"
      streaming: true
      max_articles: 100000

# Training defaults
training_defaults:
  output_dir: "./output"
  logging_dir: "./logs"
  save_strategy: "steps"
  save_steps: 500
  evaluation_strategy: "steps"
  eval_steps: 100
```

## 🎨 Prompt Template System

### Template Structure

Templates use Jinja2 syntax with special tokens:

```jinja2
{# Basic template structure #}
{{ FLAG_TOKEN }}TASK:{{ SERIAL_TOKEN }}{{ task_description }}{{ END_TASK_TOKEN }}
{{ FLAG_TOKEN }}INPUT:{{ SERIAL_TOKEN }}{{ input_text }}{{ END_INPUT_TOKEN }}
{{ FLAG_TOKEN }}OUTPUT:{{ SERIAL_TOKEN }}{{ output_text }}{{ END_OUTPUT_TOKEN }}
```

### Template Variables

Available in all templates:

- `FLAG_TOKEN`: Boundary marker for sections
- `SERIAL_TOKEN`: Sequence separator
- `END_TASK_TOKEN`, `END_INPUT_TOKEN`, `END_OUTPUT_TOKEN`: Section terminators
- `task_description`: Task description text
- `input_text`: Input content
- `output_text`: Expected output

### Template Registration

```python
# In prompt_templates.py
TASK_TEMPLATES = {
    "default": "templates/default.jinja",
    "arithmetic": "templates/arithmetic.jinja",
    "qa": "templates/question_answer.jinja",
    # Add custom templates here
}

def apply_template(
    text: str,
    template_name: str = "default",
    task_description: str = "General text completion",
    **kwargs
) -> str:
    """Apply template to input text."""
```

## 🧪 Utility Functions

### Dataset Processing

```python
def load_and_tokenize_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    use_templates: bool = False,
    template_file: str = "default.jinja"
) -> Dataset:
    """Load and tokenize dataset with optional templating."""

def create_cache_key(
    dataset_name: str,
    vocab_size: int,
    use_templates: bool,
    template_file: str
) -> str:
    """Create unique cache key for tokenized dataset."""
```

### Model Utilities

```python
def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""

def estimate_memory_usage(
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int
) -> float:
    """Estimate GPU memory usage in GB."""

def check_numerical_stability(
    model: nn.Module,
    input_ids: torch.Tensor
) -> Dict[str, bool]:
    """Check for numerical issues in model outputs."""
```

## 📊 Performance Monitoring

### Training Callbacks

```python
class StabilityCallback(TrainerCallback):
    """Monitor training stability and numerical issues."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Check for loss=0.0, NaN gradients, etc."""

class InferenceCallback(TrainerCallback):
    """Periodic inference during training."""

    def on_evaluate(self, args, state, control, **kwargs):
        """Run inference examples and log outputs."""
```

### Metrics

Standard metrics logged during training:

- `loss`: Training loss
- `eval_loss`: Validation loss
- `grad_norm`: Gradient norm
- `learning_rate`: Current learning rate
- `step`: Training step
- `epoch`: Training epoch

## 🔍 Debugging Tools

### Model Inspection

```python
def analyze_layer_outputs(
    model: MixedLayerForCausalLM,
    input_ids: torch.Tensor
) -> Dict[int, torch.Tensor]:
    """Extract intermediate layer outputs for analysis."""

def visualize_attention_patterns(
    model: MixedLayerForCausalLM,
    input_ids: torch.Tensor,
    layer_idx: int
) -> np.ndarray:
    """Extract attention patterns from attention layers."""
```

### Error Diagnostics

```python
def diagnose_training_issues(
    trainer: Trainer,
    model: nn.Module
) -> Dict[str, Any]:
    """
    Diagnose common training problems.

    Returns:
        Dictionary with diagnostic information
    """
```

This API reference covers the core components of Guac. For implementation details, see the source code and architecture documentation.