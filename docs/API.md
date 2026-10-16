# Guac API Reference

The main classes and functions, by module. Docstrings in the source have the
details; `docs/ARCHITECTURE.md` explains the model and `docs/TRAINING.md` the
workflows.

## 🏗️ Model (`guac.core.custom_architectures`)

### MixedLayerConfig

```python
MixedLayerConfig(
    vocab_size=32000, hidden_size=4096, num_hidden_layers=32,
    num_attention_heads=32, num_kv_heads=None,          # None -> num_attention_heads
    intermediate_size=11008,                            # SwiGLU width defaults to 2/3 of this
    layer_types=None,                                   # None -> all full_attention
    ffn_types=None, ffn_hidden_sizes=None,              # per layer
    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    layer_norm_eps=1e-6, use_rms_norm=True, initializer_range=0.02,
    use_cache=True,                                     # generation cache (never in training)
    # Position encoding
    use_rope=True, rope_theta=10000.0,
    use_absolute_position_embeddings=False, max_position_embeddings=4096,
    # Conv
    conv_kernel_size=3, conv_groups=1,
    # Mamba-2
    mamba_d_state=128, mamba_d_conv=4, mamba_expand=2, mamba_headdim=64,
    mamba_ngroups=1, mamba_chunk_size=64, mamba_scan_impl="torch",
    # BASED and sliding-window attention
    based_feature_dim=16, based_num_heads=None, based_num_key_value_heads=None,
    based_chunk_size=64, sliding_window_size=64,
    # MoH and MoE (keyword arguments)
    moh_top_k=num_attention_heads // 2, moh_temperature=1.0, moh_noise_std=0.1,
    num_experts=1, num_experts_per_token=1, num_shared_experts=0, moe_top_k=num_experts_per_token,
    moe_temperature=1.0, moe_noise_std=0.1, moe_load_balancing_weight=0.01,
    tie_word_embeddings=True,                           # (PretrainedConfig)
)
```

- `layer_types`: one mixer per layer: `full_attention`, `flash_attention` (the
  same mixer), `sliding_window_attention`, `moh_attention`, `conv`, `mamba2`,
  `based`, `mlp_only`, `shared_expert_moe` (`SUPPORTED_LAYER_TYPES`)
- `ffn_types`: `swiglu`, `moe` or `none` per layer (default `moe` for
  `shared_expert_moe`, else `swiglu`); `ffn_hidden_sizes`: SwiGLU width per layer
- `use_rope`: RoPE in the softmax-attention mixers (`ROTARY_LAYER_TYPES`); needs an even head size
- `mamba_scan_impl`: `"torch"` (any device) or `"mamba_ssm"` (its fused kernel for CUDA inputs)
- The config validates layer types, lengths, head divisibility and FFN choices when built.

### MixedLayerForCausalLM

```python
model = MixedLayerForCausalLM(config)
out = model(input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=None)
# MoeCausalLMOutputWithPast: loss, aux_loss (MoE load balancing; added to loss in training only),
# logits, past_key_values (a MixedLayerCache when caching)
model.generate(input_ids, attention_mask=..., max_new_tokens=..., ...)  # uses the cache
```

Weights are tied (`lm_head` shares the token embeddings) when
`tie_word_embeddings` is set, from construction and after `from_pretrained`.

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

flop_matched_ffn_hidden_sizes(config, reference_layer_type="full_attention", seq_len=1024) -> List[int]
    """SwiGLU widths that give every SwiGLU block the forward FLOPs per token of a reference block."""

mixer_flops_per_token(config, layer_type, seq_len) -> float
block_flops_per_token(config, seq_len) -> List[float]
model_flops_per_token(config, seq_len) -> float
    """Forward FLOPs per token (matmuls, convolutions, attention), counted with
    torch.utils.flop_counter on the meta device."""
```

## 🎯 Training (`guac.core.pipeline_utils`, `guac.pipeline.pretrain`)

```python
load_model_config(name) -> dict
    """A model_configs.yaml preset merged over defaults (former names resolve via preset_aliases;
    numbers in scientific notation are read as numbers)."""

build_model_config(preset: dict, vocab_size) -> MixedLayerConfig       # guac.pipeline.pretrain
    """Applies param_match / flop_match and passes layer-type settings through."""

prepare_pretraining_data(dataset_names, tokenizer, max_length, pack=True, validation_docs=1000,
                         debug=False, force_retokenize=False) -> (train, validation)
create_training_arguments(preset, output_dir, max_steps=-1, evaluate=False) -> TrainingArguments

load_base_model(path) -> (model, tokenizer)
    """Raises if the checkpoint's weights don't match the current layer code."""

make_training_arguments(output_dir, fp16=True, report_to="none", **kwargs) -> TrainingArguments
    """Shared conventions: fp16 only on CUDA, no trackers for 'none', last partial batch kept."""

GuacTrainer(Trainer)                 # logs perplexity next to every loss
MetricsLoggerCallback(path)          # appends training and evaluation entries to a JSONL file
write_run_info(output_dir, stage, config, cli_args=None, model=None, seed=None, **extra) -> path
```

## 🗂️ Data (`guac.core.dataset_utils`, `guac.core.cache_utils`)

```python
load_pretraining_splits(dataset_names, validation_docs=1000, seed=0) -> (train, validation)
split_validation(dataset, validation_docs, seed=0) -> (train, validation)
pack_documents(token_lists, block_size, eos_token_id) -> List[List[int]]
tokenize_dataset_with_cache(dataset, tokenizer, cache_dir, max_length, force_retokenize=False, pack=False)
PackedSequenceCollator()             # batches packed blocks; labels = input_ids, nothing masked
tokenizer_fingerprint(tokenizer), content_hash(value)    # cache-key ingredients
```

## 🧪 Diagnostics (`guac.core.diagnostics`) and arithmetic (`guac.core.arithmetic`)

```python
make_batch(task, num_examples, seed, **task_kwargs) -> {'input_ids', 'labels'}
    # task in TASKS: mqar, selective_copy, induction, passkey
run_diagnostic(model, task, task_kwargs, steps=2000, batch_size=64, ...) -> results
run_language_modeling(model, train_blocks, validation_blocks, steps=1000, ...) -> results
answer_accuracy(logits, labels) -> {'token_accuracy', 'example_accuracy'}

generate_problems(category, n, rng, held_out) -> iterator of (expression, prompt, result)
is_held_out(expression)              # commuted twins share a pool (SPLIT_VERSION 2)
evaluate_accuracy(model, tokenizer, num_per_category, reverse=False) -> results
```

## 🔧 Configuration Files

| File | Contents |
|------|----------|
| `model_configs.yaml` | `defaults`, `model_training_configs` (presets), `preset_aliases` (former names) |
| `site_config.yml` | `directories` (scratch layout, read by `PathManager`) and `dataset_configs` (Hugging Face datasets by name). Copy `site_config.example.yml` |
| `sweeps/*.yaml` | Sweep specs (`sweeps/README.md`) |
| `datasets_download.csv` | Manifest for `make download-datasets` |

## 🎨 Prompt Templates (`guac.core.prompt_templates`)

```python
create_prompt_tokenizer() -> PromptTokenizer
PromptTokenizer.get_core_tokens()        # pad, unk, bos, eos
PromptTokenizer.get_all_special_tokens() # flag and serial boundary tokens, e.g. "<& THINK &>", "<& /INFERENCE &>"
apply_default_template(context) -> str   # renders templates/default.jinja
apply_template_file(filename, context) -> str
```

Templates are Jinja2 files in `templates/`, using explicit boundary tokens
(FLAG and SERIAL) rather than turn markers; see `docs/ARCHITECTURE.md`.
