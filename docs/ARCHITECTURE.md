# Guac Architecture Guide

This document provides detailed information about Guac's mixed layer architectures, layer types, and design principles.

## 🏗️ Mixed Layer Architecture Concept

### Core Innovation

Guac introduces **Mixed Layer Architectures** - transformer models where different layers use entirely different computational patterns. This allows studying how various inductive biases interact within a single model.

```
Traditional Transformer:    Mixed Layer Architecture:
┌─────────────────────┐    ┌─────────────────────┐
│   Attention + FFN   │    │      MLP-only       │ ← Simple
├─────────────────────┤    ├─────────────────────┤
│   Attention + FFN   │    │   Convolutional     │ ← Local patterns
├─────────────────────┤    ├─────────────────────┤
│   Attention + FFN   │    │   Convolutional     │ ← More local
├─────────────────────┤    ├─────────────────────┤
│   Attention + FFN   │    │      Mamba-2        │ ← State space
├─────────────────────┤    ├─────────────────────┤
│   Attention + FFN   │    │       BASED         │ ← Efficient attn
├─────────────────────┤    ├─────────────────────┤
│   Attention + FFN   │    │      Mamba-2        │ ← More state
├─────────────────────┤    ├─────────────────────┤
│   Attention + FFN   │    │  Flash Attention    │ ← Full attention
└─────────────────────┘    ├─────────────────────┤
                           │      MLP-only       │ ← Final processing
                           └─────────────────────┘
```

### Design Principles

1. **Computational Complexity Progression**: Layers progress from simple to complex computations
2. **Complementary Inductive Biases**: Each layer type captures different aspects of the data
3. **Gradient Flow**: Careful ordering to maintain stable training dynamics
4. **Memory Efficiency**: Strategic placement of memory-intensive layers

## 🧱 Layer Type Specifications

### 1. MLP-Only Layers

**Purpose**: Pure feed-forward computation without attention mechanisms.

**Implementation**:
```python
class MLPOnlyLayer:
    - LayerNorm
    - Linear(hidden_size → intermediate_size)
    - GELU activation
    - Linear(intermediate_size → hidden_size)
    - Dropout + residual connection
```

**Characteristics**:
- **Complexity**: O(d²) per token
- **Memory**: Minimal
- **Use Cases**: Simple transformations, warm-up processing, final output layers

### 2. Convolutional Layers

**Purpose**: Local pattern recognition and sequence modeling via 1D convolution.

**Implementation**:
```python
class ConvolutionalLayer:
    - Input LayerNorm
    - Conv1D(kernel_size=3, groups=conv_groups)
    - Feed-forward network
    - Output LayerNorm
    - Residual connections
```

**Characteristics**:
- **Complexity**: O(d² + d·k·L) where k=kernel_size, L=sequence_length
- **Memory**: O(d·k)
- **Use Cases**: Local feature extraction, translation-like tasks, image-to-text

### 3. Mamba-2 Layers

**Purpose**: Selective state space modeling for efficient long-range dependencies.

**Implementation** (`scripts/training/core/mamba2.py`, following Dao & Gu 2024):
```python
class Mamba2Layer:                 # pre-norm residual block, no separate FFN
    norm -> Mamba2Mixer -> + residual

class Mamba2Mixer:
    - in_proj -> gate z, input x, B, C (shared by heads within a group), dt per head
    - Depthwise causal conv1d over x, B, C, then SiLU
    - Scalar decay per head: A = -exp(A_log), discretized as exp(A * dt),
      dt = softplus(dt + dt_bias)
    - Chunked SSD scan + D skip connection
    - Gated RMSNorm: norm(y * silu(z)), then out_proj
```

**The scan**: the chunked structured state space duality (SSD) algorithm in plain
PyTorch. Inside each chunk it is a causal, attention-like product; between chunks
it is a linear recurrence over chunk states. It computes exactly the sequential
recurrence `h_t = exp(A·dt_t)·h_{t-1} + dt_t·x_t⊗B_t`, `y_t = h_t·C_t + D·x_t`
(tested against it), runs on any device, and keeps the scan in float32 under
mixed precision. Parameter names follow the reference implementations
(mamba_ssm, transformers' `Mamba2Mixer`), and outputs are tested against
transformers' version.

**Characteristics**:
- **Complexity**: O(L·Q·(N+P)) per head for the scan, with chunk length Q, state size N, head size P
- **Memory**: O(N·P) state per head at inference; linear in L during training
- **Use Cases**: Long sequences, time series, language modeling

### 4. BASED Layers

**Purpose**: Bidirectional Attention with Sparse Efficient Decoding using feature maps.

**Implementation**:
```python
class BasedLayer:
    - Query/Key/Value projections
    - Feature map transformations
    - Cumulative sum operations
    - Efficient attention via feature maps
    - Feed-forward network
```

**Key Innovation**:
- **Feature Maps**: ReLU(Linear(head_dim → feature_dim))
- **Efficient Computation**: O(L·F) instead of O(L²) where F=feature_dim
- **Causality**: Maintained through cumulative operations

**Characteristics**:
- **Complexity**: O(d² + h·f·L) where h=heads, f=feature_dim
- **Memory**: O(h·f·L)
- **Use Cases**: Medium-range dependencies, attention approximation

### 5. Flash Attention Layers

**Purpose**: Memory-efficient full attention computation.

**Implementation**:
```python
class FlashAttentionLayer:
    - Standard Q/K/V projections
    - torch F.scaled_dot_product_attention: FlashAttention kernels on
      supported GPUs/dtypes, math kernel elsewhere (CPU included)
    - Grouped query attention support
```

**Optimizations**:
- **Tiling**: Process attention in blocks
- **Recomputation**: Trade computation for memory
- **Online Softmax**: Streaming softmax computation

**Characteristics**:
- **Complexity**: O(d² + h·L²) - same as standard attention
- **Memory**: O(h·L) instead of O(h·L²)
- **Use Cases**: Full attention when memory permits, long sequences

### 6. Standard Attention Layers

**Purpose**: Traditional multi-head attention with grouped query attention.

**Implementation**:
```python
class GroupedQueryAttention:
    - Q projection: (d → num_heads × head_dim)
    - K/V projections: (d → num_kv_heads × head_dim)
    - Attention computation
    - Output projection
```

**Features**:
- **Grouped Query**: Fewer KV heads than Q heads for efficiency
- **Causal Masking**: Lower triangular attention
- **Stable Computation**: Proper scaling and normalization

## 🔧 Configuration System

### Architecture Selection

Architectures are defined in `architecture_configs.yaml`:

```yaml
architectures:
  mixed_layer:
    config_class: "MixedLayerConfig"
    model_class: "MixedLayerForCausalLM"
    import_path: "scripts.training.custom_architectures"

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
```

### Layer Type Configuration

Each layer type has specific parameters:

```yaml
# Mamba-2 parameters
mamba_d_state: 128        # State size N per head
mamba_d_conv: 4           # Causal convolution kernel size
mamba_expand: 2           # Inner size = expand * hidden_size
mamba_headdim: 64         # Head size P; inner size must be divisible by it
mamba_ngroups: 1          # B/C groups shared across heads (heads must divide evenly)
mamba_chunk_size: 64      # Scan chunk length: speed/memory only, results are identical

# BASED parameters
based_feature_dim: 8      # Feature map dimension
based_num_heads: 8        # Number of attention heads
based_num_key_value_heads: 4  # Number of KV heads

# Convolutional parameters
conv_kernel_size: 3       # 1D convolution kernel
conv_groups: 1            # Depthwise grouping
```

## 📊 Performance Characteristics

### Computational Complexity Comparison

| Layer Type        | Per-Token Compute | Memory Usage    | Sequence Scaling |
|-------------------|-------------------|-----------------|------------------|
| MLP-only          | O(d²)            | O(d)           | O(1)            |
| Convolutional     | O(d² + d·k)      | O(d·k)         | O(1)            |
| Mamba-2           | O(d² + d·s)      | O(d·s)         | O(1)            |
| BASED             | O(d² + h·f·L)    | O(h·f·L)       | O(L)            |
| Flash Attention   | O(d² + h·L²)     | O(h·L)         | O(L²)           |
| Standard Attention| O(d² + h·L²)     | O(h·L²)        | O(L²)           |

### Memory Efficiency Tips

1. **Layer Ordering**: Place memory-intensive layers (attention) later in the stack
2. **Gradient Checkpointing**: Enable for memory-constrained training
3. **Batch Size**: Reduce batch size for longer sequences
4. **Mixed Precision**: Use FP16 training when possible

## 🔄 Training Dynamics

### Gradient Flow

The mixed architecture is designed to maintain stable gradient flow:

1. **Early Layers**: Simple operations provide stable gradients
2. **Middle Layers**: Complex operations benefit from stable inputs
3. **Late Layers**: High-capacity layers can fit complex patterns
4. **Residual Connections**: All layers maintain skip connections

### Numerical Stability

- **Mamba-2**: the scan's exponentials and cumulative sums run in float32 even under
  mixed precision; decays are built from masked cumulative sums rather than
  differences of large running totals.
- **Attention**: softmax attention runs through `F.scaled_dot_product_attention`.
  Positions whose keys are all padding attend to themselves, so masked rows can't
  produce NaN.
- **No silent recovery**: NaNs are not reset to zero; a diverging run should fail
  visibly. The tests check causality, padding handling and save/load for every layer type.

## 🎯 Design Guidelines

### Adding New Layer Types

To add a new layer type:

1. **Implement the layer class** in `custom_architectures.py`
2. **Add to MixedTransformerLayer** switch statement
3. **Update configuration** with new parameters
4. **Add stability checks** for numerical issues
5. **Document** complexity and use cases

### Best Practices

1. **Start Simple**: Begin with well-understood layer types
2. **Monitor Stability**: Watch for NaN gradients or loss=0
3. **Profile Memory**: Check memory usage with new combinations
4. **Validate Outputs**: Ensure layer outputs have expected statistics
5. **Test Incrementally**: Add one layer type at a time

## 🎯 Design Philosophy

### Prompt System Innovation

Guac's prompt system intentionally deviates from convention in several key areas:

1. **Multi-Sequence Handling**: Uses explicit boundary tokens (FLAG, SERIAL) instead of turn-based markers to support MultiSeq-to-MultiSeq tasks where linear ordering doesn't imply temporal sequence.

2. **Inference Control**: Employs inference trigger and cessation tokens rather than turn boundaries, preventing models from conflating turn information with timestep information.

3. **Optional Guidance**: All guidance and task definitions are optional by design to reduce complacency and improve adherence to instructions.

4. **Content Boundaries**: Sequence content lacks explicit internal boundaries as content should be implicitly expected within the outer sequence containers.

## 🔬 Research Applications

### Architectural Studies

- **Layer Importance**: Ablation studies removing specific layer types
- **Ordering Effects**: Different arrangements of computational patterns
- **Scaling Laws**: How mixed architectures scale with size
- **Transfer Learning**: How mixed models transfer across tasks

### Comparison Studies

- **vs Homogeneous**: Mixed vs. all-attention or all-convolution
- **vs MoE**: Mixed layers vs. mixture of experts
- **vs Hybrid**: Mixed vs. traditional hybrid architectures
- **Efficiency Analysis**: FLOPs, memory, and quality trade-offs

This architecture enables unprecedented flexibility in studying computational patterns within transformer models while maintaining practical training efficiency.