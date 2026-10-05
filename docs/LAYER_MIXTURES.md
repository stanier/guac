# Layer Mixtures: Recommendations

Notes on choosing `layer_types` for `mixed_layer` models, based on a review of
the current presets and the layer implementations in
`scripts/training/core/custom_architectures.py`.

## What the current presets do

Every preset in `model_configs.yaml` repeats the same 4-layer motif
`[conv, conv, mamba2, conv]`: 75% conv, 25% Mamba-2, **0% attention**.

Problems with that mixture:

1. **No exact-recall layer.** Conv layers see only `kernel_size` tokens, and a
   Mamba-2 layer compresses the whole prefix into a fixed-size state
   (`d_state=16`). Neither can do exact lookup or copying over long contexts
   (associative recall, copying, retrieving a fact from earlier in the
   prompt). Zoology/BASED, Jamba, Samba, Nemotron-H and LFM2 all found that a
   few global softmax-attention layers close most of that gap.
2. **The motif doesn't change with depth.** A 48B model gets the same 3:1
   ratio as a 106M model, with the same local-only receptive field per conv
   layer.
3. **The MoE settings have no effect.** `num_experts`, `moe_*` and `moh_*` only
   matter if `shared_expert_moe` / `moh_attention` appear in `layer_types`.
   None of the presets list them, so every expert setting from 3b up is
   ignored. Some values are also inconsistent: for example, `guac-48b` asks
   for 256 shared experts, 128 routed experts and top-64.

## Bugs fixed alongside this doc

These had to be fixed before mixture comparisons would mean anything:

| Issue | Effect | Fix |
|---|---|---|
| `ConvolutionalLayer` used symmetric padding (`k//2`) | **Future-token leak.** With k=3, position *t* saw *t+1*. Stacked over 18 conv layers in guac-3b, that reaches 18 tokens ahead. Training loss looked great, but generation would be poor. | Causal left-padding (`k-1, 0`) |
| `FlashAttentionLayer` non-flash path used `(1-mask) * -inf` | `0 * -inf = NaN` whenever an `attention_mask` was passed (always the case at eval, since flash is training-only) | `masked_fill` with `finfo.min` |
| `guac-48b` listed 60 layer types for 64 layers | `ValueError` at model construction | Added the missing motif |
| `pretrain.create_model` dropped `mamba_*`, `conv_*`, `based_*`, `flash_attn_*`, `num_shared_experts` | These settings in `model_configs.yaml` were silently ignored | Passed through |
| `bf16` not forwarded to `TrainingArguments` | Could only train in fp16 | `bf16: true` is now honored |

Verified with a causality probe: change token 10, then check that logits at
positions 0–9 stay the same, for every layer type.

## Layer types: what to use

| Layer type | Use in mixtures? | Notes |
|---|---|---|
| `conv` | Yes | Short-range token mixing plus a GELU FFN, so it doubles as the channel-mixing ("MLP") layer. Now causal. |
| `mamba2` | Yes, but slow | No FFN of its own; pair it with conv/MLP layers. The selective scan is a Python loop over `seq_len`, so it is by far the slowest layer to train. `C` is tied to `B`. |
| `flash_attention` | **Yes: the recommended attention layer** | Pre-norm, GQA, has an FFN. Uses flash-attn when installed, otherwise a correct masked fallback. |
| `full_attention` | Avoid in mixtures | **Post-norm** (norm after residual), while every other layer type is pre-norm. Mixing the two in one stack hurts stability at depth. |
| `mlp_only` | Occasionally | Pure FFN. Useful after `mamba2` layers if you want more channel mixing. |
| `based` | **Avoid** | Not real BASED: it computes an L×L softmax over `q · cumsum(k)`. That is quadratic and doesn't approximate attention. |
| `moh_attention` | **Avoid** | Head routing uses `hidden_states.mean(dim=1)` over the *whole* sequence, which **leaks future tokens** (verified by the probe). It also sums the selected heads into a single `head_dim` and tiles it to `hidden_size`. Needs per-token routing before it can be used causally. |
| `shared_expert_moe` | Not yet | No sequence mixing. Post-norm output. Per-token Python loop for dispatch. Fix it before using it to scale. |

## Recommended mixtures

Two families, both added to `model_configs.yaml` at the same widths as the
existing presets:

### `*-hybrid`: Jamba / Nemotron-H style (SSM-heavy)

An 8-layer block, repeated:

```
conv, mamba2, conv, mamba2, flash_attention, mamba2, conv, mamba2
```

- 12.5% attention, 50% Mamba-2, 37.5% conv (Jamba uses 1:7 attention:Mamba;
  Nemotron-H uses about 8% attention).
- Attention is **never the first layer**. Early layers build local features
  that attention can then use. Attention sits in the middle of each block, so
  it is spread evenly through the depth.
- Conv layers sit next to Mamba layers and supply the FFN capacity that
  `mamba2` lacks.
- Presets: `guac-106m-hybrid` (scaled down to `conv, mamba2, flash_attention, conv`),
  `guac-1.1b-hybrid`, `guac-3b-hybrid`.

### `*-lfm2`: LFM2 style (conv + attention, fully parallel)

```
conv, conv, flash_attention, conv, conv, flash_attention, ...
(attention gets denser toward the top)
```

- About 37.5–40% attention, the rest short causal convs, following LFM2's
  10 conv + 6 GQA layout.
- No sequential scan, so it is **the fastest mixture to train with the current
  code**. Use it as the default until `mamba2` gets a parallel or chunked scan
  (or `mamba_ssm` kernels).
- Presets: `guac-1.1b-lfm2`, `guac-3b-lfm2`.

### Scaling rule of thumb

Keep the attention *fraction* roughly fixed as depth grows, about 1 in 8 for
SSM-heavy stacks and about 1 in 3 for conv-heavy stacks. Repeat the block
rather than appending more conv/SSM layers. Always end with a layer that has
an FFN (conv, attention or MLP), not a bare `mamba2`.

## Config recommendations

Applied in the new presets:

| Setting | Current | Recommended | Why |
|---|---|---|---|
| `max_position_embeddings` | 65536 (default) | = `max_sequence_length` | The learned position table is mostly never trained. That's 34M of guac-106m's 106M parameters and 201M params in guac-3b. |
| `warmup_ratio` | 0.1 → 0.5 as size grows | 0.02–0.05 | `pretrain.py` pins `max_steps=1000`, so a 0.5 ratio spends half the run warming up. |
| `fp16` | true | `bf16: true` | The Mamba recurrence (`exp(A·Δ)`) and the clamp/NaN-reset code are symptoms of fp16 range problems. |
| `weight_decay` | 0.01–0.05 at most sizes | 0.1 | Standard for LM pretraining at every size listed. |

Not yet applied:

- **Learning rate at large sizes** decays too fast: 2e-5 at 32B and 1e-5 at
  48B. Published runs at those sizes use about 1–1.5e-4. With
  `constant_with_warmup`, consider `cosine` or WSD (warmup-stable-decay) so
  there is a decay phase.
- **MoE (once `shared_expert_moe` is fixed):** 1–2 shared experts, 32–128
  routed fine-grained experts, top-k 4–8. Use MoE in place of the FFN in
  every other block (Jamba style), not as a standalone layer.
- **Position encoding:** `use_rope` is read into the config but never applied.
  Hybrids with causal conv/SSM layers already encode position, so it's worth
  ablating NoPE (no position table at all) against the learned table.
- **Batch size typos:** `guac-280m` has `pdbs_train: 4096` while its
  neighbours use 1–32. `pretrain.py` clamps to 8, which hides this.

## Suggested ablation

At `guac-106m` scale (cheap) and then `guac-1.1b`, with matched tokens, compare:

1. `guac-*` (current motif, now causal). Expect a much worse loss than before
   the conv fix; the old numbers were inflated by the leak.
2. `guac-*-lfm2`
3. `guac-*-hybrid`

Evaluate perplexity **and** a recall probe (multi-query associative recall or
needle-in-a-haystack at 2× the training length). Recall is where the
attention-free motif should fall short.
