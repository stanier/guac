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

1. **Only the mixer varies**: every layer type sits in the same pre-norm block with
   the same FFN, so differences between them come from how they mix positions
2. **Comparable budgets**: FFN widths can be set so every block has the same
   parameters or FLOPs (`param_match`, `flop_match`)
3. **Faithful mechanisms**: Mamba-2, BASED and MoH follow their papers, and each
   is tested against a direct reference implementation
4. **Measured, not assumed**: synthetic diagnostics and sweeps report what each
   layer type contributes

## 🧱 Layer Type Specifications

### Block Structure (shared by every layer type)

Every layer is the same pre-norm residual block (`MixedTransformerLayer`), and
only its **mixer** (the part that moves information between positions) depends on
the layer type:

```python
h = x + mixer(norm(x))     # skipped for mlp_only and shared_expert_moe, which have no mixer
y = h + ffn(norm(h))       # skipped when the layer's FFN type is "none"
```

This follows hybrid models such as Jamba and Granite, and means a comparison
between layer types isn't confounded by norm placement or a missing FFN.

**Generation** keeps each mixer's state between steps (`MixedLayerCache`), so
each new token costs one step per layer rather than a pass over the whole
sequence: keys and values for attention (only the window for sliding-window
attention), and fixed-size states for conv (last k-1 inputs), Mamba-2 (conv and
SSM state) and BASED (the running `sum phi(k) v^T`).
Mixers are looked up in the `MIXERS` registry in `custom_architectures.py`.

**Packed documents** can be kept apart with `reset_at_document_boundaries: true`
(or `pretrain.py --reset-at-document-boundaries`). Each document ends at
`eos_token_id`; without a generation cache, attention then masks keys from
other documents, conv windows treat earlier documents' inputs as zeros, Mamba-2
and BASED restart their state, and absolute positions restart. A document
packed among others gets exactly the outputs (and gradients) it would get
alone; `tests/test_document_boundaries.py` checks this for every layer type.
It costs 10–40% per training step on CPU, and attention loses PyTorch's fastest
kernels to the explicit mask, so it is off by default (GPT-style packing).
Recurrent layers are where it matters most: without it, Mamba-2 and BASED carry
one document's state into the next, where attention can learn to ignore what
came before. Generation ignores it. The model's pad, BOS and EOS ids come
from the tokenizer.

**The FFN is chosen per layer** with `ffn_types` (one entry per layer):

| `ffn_types` entry | FFN |
|-------------------|-----|
| `swiglu` (default) | SwiGLU of width `ffn_hidden_sizes[i]`; by default 2/3 of `intermediate_size`, which gives it the parameters of a two-matrix FFN of width `intermediate_size` |
| `moe` (default for `shared_expert_moe`) | `SharedExpertMoE`: shared plus routed SwiGLU experts |
| `none` | No FFN, e.g. for original-style Mamba-2 blocks |

**Matching parameters across layer types.** Mixers differ in size (at the preset
settings, a Mamba-2 mixer has about twice the parameters of a conv mixer), so a raw
swap changes model size too. Set `param_match: <layer type>` (or `true` for
`full_attention`) in a `model_configs.yaml` preset, and every SwiGLU layer's FFN
width is set so its block has the parameters of a block of that type with the
default width. In code, use `param_matched_ffn_hidden_sizes(config, reference)`;
`block_parameter_counts(config)` reports each block's size without allocating it.

### Position Encoding

The softmax-attention mixers (`full_attention`, `flash_attention`,
`sliding_window_attention`, `moh_attention`) apply rotary position embeddings
(RoPE) to queries and keys, so their scores depend only on relative position.
Set `use_rope: false` for position-free (NoPE) attention, and `rope_theta`
(default 10000) for the base frequency. RoPE has no parameters.

Conv, Mamba-2 and BASED mixers see token order through their causal structure,
so they need no position encoding. A learned absolute position table
(`use_absolute_position_embeddings: true`, `max_position_embeddings` rows) is
available but off by default. Until October 2026 it was always on, with
65,536 rows: 33.5M parameters in the smallest preset that its short sequences
never used.

### 1. MLP-Only Layers

**Purpose**: Pure feed-forward computation without attention mechanisms.

**Implementation**:
```python
mlp_only: no mixer; the block is just  x + ffn(norm(x))
```

**Characteristics**:
- **Complexity**: O(d²) per token
- **Memory**: Minimal
- **Use Cases**: Simple transformations, warm-up processing, final output layers

### 2. Convolutional Layers

**Purpose**: Local pattern recognition and sequence modeling via 1D convolution.

**Implementation**:
```python
class ConvMixer:
    - Causal Conv1D(kernel_size=conv_kernel_size, groups=conv_groups)
```

**Characteristics**:
- **Complexity**: O(d² + d·k·L) where k=kernel_size, L=sequence_length
- **Memory**: O(d·k)
- **Use Cases**: Local feature extraction, translation-like tasks, image-to-text

### 3. Mamba-2 Layers

**Purpose**: Selective state space modeling for efficient long-range dependencies.

**Implementation** (`guac/core/mamba2.py`, following Dao & Gu 2024):
```python
class Mamba2Mixer:                 # its block has a SwiGLU FFN unless ffn_types says "none"
    - in_proj -> gate z, input x, B, C (shared by heads within a group), dt per head
    - Depthwise causal conv1d over x, B, C, then SiLU
    - Scalar decay per head: A = -exp(A_log), discretized as exp(A * dt),
      dt = softplus(dt + dt_bias)
    - Chunked SSD scan + D skip connection
    - Gated RMSNorm: norm(y * silu(z)), then out_proj
```

**The scan**: the chunked structured state space duality (SSD) algorithm in plain
PyTorch. Inside each chunk it is a causal, attention-like product; between chunks
it is a linear recurrence over chunk states, itself scanned in chunks once there
are more than `2 * mamba_chunk_size` of them, so cost stays linear in the
sequence length at any length. It computes exactly the sequential
recurrence `h_t = exp(A·dt_t)·h_{t-1} + dt_t·x_t⊗B_t`, `y_t = h_t·C_t + D·x_t`
(tested against it), runs on any device, and keeps the scan in float32 under
mixed precision. Parameter names follow the reference implementations
(mamba_ssm, transformers' `Mamba2Mixer`), and outputs are tested against
transformers' version.

**GPU kernel (optional)**: with `mamba_scan_impl: mamba_ssm`, CUDA inputs go
through `mamba_ssm`'s fused `mamba_chunk_scan_combined` instead of the PyTorch
scan (CPU inputs still use the PyTorch scan, so the same checkpoint runs
anywhere). It takes the same parameters, so checkpoints are interchangeable.
`tests/test_mamba2.py` compares the two, but that test only runs with a GPU
and `mamba_ssm` installed, so it hasn't been run yet. The default stays `torch`
until it has.

**Characteristics**:
- **Complexity**: O(L·Q·(N+P) + L·N·P) per head for the scan, with chunk length Q, state size N, head size P
- **Memory**: O(N·P) state per head at inference; linear in L during training
- **Use Cases**: Long sequences, time series, language modeling

### 4. BASED Layers

**Purpose**: Linear attention that keeps a fixed-size state, from BASED (Arora et al. 2024).

**Implementation** (`BasedMixer`; the scan is in `guac/core/linear_attention.py`):
```python
class BasedMixer:
    - q_proj, k_proj: hidden -> based_feature_dim per head (small, 16 by default)
    - v_proj: hidden -> head_dim per KV head (grouped-query heads supported)
    - 2nd-order Taylor feature map: phi(x) = [1, x / d^(1/4), vec(x x^T) / sqrt(2d)],
      so phi(q).phi(k) = 1 + s + s^2/2 with s = q.k / sqrt(d)  (approximates exp(s))
    - Normalized causal linear attention:
      y_i = sum_{j<=i} phi(q_i).phi(k_j) v_j / sum_{j<=i} phi(q_i).phi(k_j)
    - Output projection
```

**The scan** is chunked: quadratic inside each chunk of `based_chunk_size` tokens,
plus a running state `sum_j phi(k_j) v_j^T` between chunks. It is tested against
the quadratic form written out directly, values and gradients, and runs in float32
under mixed precision. Padded keys are zeroed (the feature map has a constant
term, so they would otherwise count), so left and right padding are both handled.

The BASED paper pairs these layers with small sliding-window attention layers
(and short convolutions) for exact local recall. Build that as a `layer_types`
pattern, for example `[conv, based, sliding_window_attention, ...]`.

**Characteristics**:
- **Complexity**: O(L·F·P) per head, with F = 1 + f + f² expanded features (273 for f = 16) and head size P
- **Memory**: O(F·P) state per head at inference
- **Use Cases**: Associative recall at linear cost; pair with sliding-window attention

### 4b. Sliding-Window Attention Layers

**Purpose**: Exact softmax attention over a short local window.

`sliding_window_attention` is `GroupedQueryAttention` in which each position sees
only the last `sliding_window_size` positions (itself included; 64 by default, as
in BASED). The sequence is split into window-sized blocks that each attend to
themselves and the previous block, so cost grows as L·w rather than L².

### 5. Flash Attention Layers

**Purpose**: Memory-efficient full attention computation.

**Implementation**:
```python
flash_attention uses the same mixer as full_attention (GroupedQueryAttention):
    - torch F.scaled_dot_product_attention: FlashAttention kernels on
      supported GPUs/dtypes, math kernel elsewhere (CPU included)
```

The two names are kept so existing configs keep working; they build identical blocks.

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

### 7. Mixture-of-Head Attention

**Purpose**: Attention heads as experts (MoH, Jin et al. 2024): each token uses its top-k heads.

`moh_attention` computes every head, then a per-token router (a small MLP on the
token's own hidden state, so routing stays causal) picks `moh_top_k` heads and
gives them softmax gates. The output is `o_proj(concat_h(g_h * head_h))`: each
selected head keeps its own subspace of the output projection, and unselected
heads contribute nothing. With all heads selected at equal gates it reduces to
multi-head attention scaled by 1/H (tested).

### 8. Shared-Expert MoE (an FFN option)

`SharedExpertMoE` replaces the SwiGLU FFN where `ffn_types` says `moe` (the
default for `shared_expert_moe` layers, and allowed after any mixer). Shared
experts always run; each token also goes to its top `moe_top_k` of `num_experts`
routed experts, weighted by renormalized router probabilities. Dispatch sorts
assignments by expert so each expert runs once per batch.

**Load balancing**: each MoE layer computes the Switch Transformer loss
`moe_load_balancing_weight * N * sum_i f_i * P_i` (f_i: fraction of assignments
to expert i, P_i: its mean router probability; padding excluded). The model sums
it over layers, returns it as `aux_loss`, and adds it to `loss` in training mode
only, so evaluation loss stays the language-modeling loss.

## 🔧 Configuration System

### Layer Stacks

A model's layers are set per preset in `model_configs.yaml` (built by
`build_model_config` in `guac/pipeline/pretrain.py`), or directly with
`MixedLayerConfig`:

```yaml
layer_types: [conv, conv, mamba2, sliding_window_attention]   # one mixer per layer
ffn_types: [swiglu, swiglu, none, moe]                        # optional, per layer
ffn_hidden_sizes: [1365, 1365, 1365, 1365]                    # optional SwiGLU widths
param_match: full_attention    # optional: size each FFN to a full_attention block's parameters
# flop_match: full_attention   #   ...or to its FLOPs per token at max_sequence_length
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
mamba_scan_impl: torch    # or mamba_ssm: its fused Triton kernel for CUDA inputs (needs the package)
reset_at_document_boundaries: false   # true: packed documents don't see each other (see above)

# BASED parameters
based_feature_dim: 16     # Query/key size per head before the Taylor feature map
based_num_heads: 8        # Number of heads (default num_attention_heads)
based_num_key_value_heads: 4  # Number of KV heads (default num_kv_heads)
based_chunk_size: 64      # Scan chunk length: speed/memory only, results are identical

# Sliding-window attention
sliding_window_size: 64   # Positions each token sees, itself included

# Convolutional parameters
conv_kernel_size: 3       # 1D convolution kernel
conv_groups: 1            # Depthwise grouping

# Mixture-of-Head attention
moh_top_k: 4              # Heads each token uses (default num_attention_heads // 2)

# MoE FFN (ffn_types: moe, the default for shared_expert_moe layers)
num_experts: 8            # Routed experts
num_shared_experts: 1     # Always-active experts
moe_top_k: 2              # Routed experts per token
moe_load_balancing_weight: 0.01

# Position encoding
use_rope: true            # RoPE in the softmax-attention mixers
rope_theta: 10000.0
use_absolute_position_embeddings: false
```

## 📊 Performance Characteristics

### Computational Complexity Comparison

| Layer Type        | Per-Token Compute | Memory Usage    | Sequence Scaling |
|-------------------|-------------------|-----------------|------------------|
| MLP-only          | O(d²)            | O(d)           | O(1)            |
| Convolutional     | O(d² + d·k)      | O(d·k)         | O(1)            |
| Mamba-2           | O(d² + d·s)      | O(d·s)         | O(1)            |
| BASED             | O(d² + h·F·P)    | O(h·F·P) state | O(1)            |
| Sliding window    | O(d² + h·w)      | O(h·w) KV      | O(1)            |
| Flash Attention   | O(d² + h·L²)     | O(h·L)         | O(L²)           |
| Standard Attention| O(d² + h·L²)     | O(h·L²)        | O(L²)           |

### Memory Efficiency Tips

1. **Layer Ordering**: Place memory-intensive layers (attention) later in the stack
2. **Gradient Checkpointing**: Enable for memory-constrained training
3. **Batch Size**: Reduce batch size for longer sequences
4. **Mixed Precision**: Use FP16 training when possible

## 🔄 Training Dynamics

### Gradient Flow

Every layer is a pre-norm residual block, so each layer's output is added to
an identity path and gradients reach every layer regardless of its mixer.
Which orderings of layer types train best is an open question the sweep
harness is for (`sweeps/README.md`).

### Numerical Stability

- **Mamba-2**: the scan's exponentials and cumulative sums run in float32 even under
  mixed precision; decays are built from masked cumulative sums rather than
  differences of large running totals.
- **BASED**: the Taylor feature map and the linear-attention sums run in float32;
  the normalizer is at least 1/2 per key, so it never divides by zero.
- **Attention**: softmax attention runs through `F.scaled_dot_product_attention`.
  Positions whose keys are all padding attend to themselves, so masked rows can't
  produce NaN.
- **No silent recovery**: NaNs are not reset to zero; a diverging run should fail
  visibly. The tests check causality, padding handling and save/load for every layer type.

## 🎯 Design Guidelines

### Adding New Layer Types

To add a new layer type:

1. **Implement the mixer** in `custom_architectures.py`: an `nn.Module` built
   from the config whose `forward(hidden_states, attention_mask=None, cache=None)`
   returns a tensor of the same shape. No norm, residual or FFN; the block adds
   those. Padding must not reach real tokens, and with a `cache` (generation)
   the mixer stores the state it needs to continue (see `docs/CONTRIBUTING.md`).
2. **Register it** in `MIXERS` and `SUPPORTED_LAYER_TYPES`
3. **Update configuration** with new parameters
4. **Test it**: the causality, padding, block-structure and generation tests run
   over every registered type automatically
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