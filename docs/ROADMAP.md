# Guac Roadmap

The working plan for Guac, kept current as items land. It comes from a full
codebase review in October 2026. Each finding below was reproduced with a
probe script or on real checkpoints, not just read off the code.

**Status (2026-10-16):** Phase 0, the Mamba-2 rewrite and the unified block
structure (1.1) are done. Next are the BASED rewrite (1.2), the MoH and MoE
fixes (1.3) and position encoding (1.4), which are independent of each other.

## Where Things Stand

| Branch | Commits on top of `main` | Purpose |
|--------|--------------------------|---------|
| `phase-0-correctness` | `3219380`, `761c30f` | Pre-review pending work, then the Phase 0 fixes |
| `phase-1-mamba2` | the above + `446bd60` + `8fdeb01` (this doc) | Mamba-2 rewrite |
| `claude/vibrant-darwin-vs3o5o` | the above + `4fd35cb` + this update | Unified block structure; **continue from here** |

None of these branches is merged into `main`. Each Phase 1 item should land as its own
commit (or stacked branch) with tests.

## Setting Up a Fresh Clone

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is enough for tests
pip install -e ".[test]"
python -m pytest                                                     # 127 tests, ~6 s, no network
```

- `site_config.yml` is git-ignored. Without it, the path manager falls back to
  `site_config.example.yml` with a warning, which is fine for tests and smoke runs.
- `scratch/` (tokenizers, datasets, checkpoints) is git-ignored. A fresh clone has none of it.
- The character tokenizer used by `guac-106m-arith` builds offline:
  `python -m scripts.training.tokenizer.build_char_tokenizer`.
  Every other preset needs the 60k SentencePiece tokenizer (`make train-tokenizer`),
  which downloads data.

### End-to-end smoke test (CPU, offline, about 1 minute)

Unit tests don't exercise `Trainer`, the collators or the stage scripts. After
changing model or pipeline code, also run a real pretrain → midtrain:

```bash
python -m scripts.training.tokenizer.build_char_tokenizer
python - <<'EOF'
import random
from datasets import Dataset
rng = random.Random(0)
words = "the model learns which layer types mix information across tokens".split()
texts = [" ".join(rng.choice(words) for _ in range(rng.randint(8, 120))) for _ in range(400)]
Dataset.from_dict({"text": texts}).save_to_disk("scratch/smoke/tiny_text")
EOF
python scripts/training/pipeline/pretrain.py --model-config guac-13m-arith \
    --datasets scratch/smoke/tiny_text --output-dir scratch/smoke/pre --max-steps 12 --batch-size 4 --grad-accum 2 \
    --eval-steps 6 --validation-docs 40
python scripts/training/pipeline/midtrain.py --base-model scratch/smoke/pre --task arithmetic \
    --output-dir scratch/smoke/mid --num-arithmetic-samples 400 --num-eval-samples 40 \
    --accuracy-eval-per-category 2 --max-steps 4 --eval-steps 2 --batch-size 8 --grad-accum 1
```

Loss should fall steadily (about 4.7 → 0.4 over the 12 pretraining steps), and
midtraining should load the checkpoint and finish. The presets only use `conv`,
`mamba2` and (in one preset) `flash_attention`. To test other layer types under `Trainer`, build a model
config dict with the layer types you want (plus `ffn_types` and `param_match` if
relevant) and call `build_model_config` and `create_training_arguments` from
`pretrain.py` directly.

## Done

### Phase 0: correctness (`761c30f`)

- **Weight tying:** models tie `lm_head` to the embeddings at construction via
  `post_init()`, controlled by `tie_word_embeddings` (new preset key, default true).
  Previously they trained untied, and `from_pretrained` then replaced the trained
  head with the embedding matrix.
- **Attention:** every softmax attention layer goes through one
  `causal_attention` helper on `F.scaled_dot_product_attention`. This fixed NaN
  loss whenever a mask was passed (CPU, and every eval pass) and removed the
  `flash-attn` dependency.
- **Config is authoritative:** removed the hidden batch cap (8), the grad-accum
  floor (4), the sequence-length shrinking heuristic and the fixed 1000 steps.
  `pretrain.py` prints the settings it uses and takes `--batch-size`/`--grad-accum`
  (`PRETRAIN_ARGS=` in make).
- **Cache keys:** these now cover the tokenizer itself, the site-config dataset
  settings and the debug subset. A `--debug` run used to fill the cache that full
  runs then read.
- Fixed `guac-48b` (60 layer types for 64 layers) and BASED crashing without
  `num_kv_heads`. Layer types are validated when the config is built, and
  `num_shared_experts` now reaches the model.
- **Tests:** moved to pytest with CPU CI (`.github/workflows/tests.yml`; not yet
  observed running on GitHub). Added round-trip, mask, preset and cache-key tests,
  and deleted three tests of legacy modules.

### Phase 1: Mamba-2 (`446bd60`)

- `scripts/training/core/mamba2.py`: Mamba-2 per Dao & Gu (2024), with a chunked
  SSD scan in plain PyTorch. It is exact against the sequential recurrence (values
  and gradients) and matches transformers' `Mamba2Mixer` given the same weights.
  It is about 10× faster than the old layer at 512 tokens, with linear memory.
- `load_base_model` raises when checkpoint weights don't match the current layer
  code, instead of loading a partly random model.

### Phase 1.1: unified block structure (`4fd35cb`)

- Every layer is one pre-norm block, `x + mixer(norm(x))` then `x + ffn(norm(x))`
  (`MixedTransformerLayer(config, layer_idx)`); only the mixer depends on the
  layer type, looked up in the `MIXERS` registry. `full_attention` used to
  normalize after the residual add; `mamba2`, `mlp_only` and `shared_expert_moe`
  had no mixer/FFN pairing.
- `full_attention` and `flash_attention` are the same mixer and build identical
  blocks; both names stay so configs keep working.
- The FFN is chosen per layer with `ffn_types`: `swiglu` (default), `moe`
  (`SharedExpertMoE`, now an FFN option after any mixer; the default for
  `shared_expert_moe`) or `none` (for example original-style Mamba-2 blocks).
  `ffn_hidden_sizes` sets the SwiGLU width per layer; the default is 2/3 of
  `intermediate_size`, which has the parameters of the old two-matrix GELU FFN.
- Parameter matching: `param_matched_ffn_hidden_sizes(config, reference)` sizes
  each SwiGLU layer's FFN so its block has the parameters of a reference block,
  and presets can set `param_match: <layer type>` (or `true` for
  `full_attention`). `block_parameter_counts(config)` reports block sizes on the
  meta device.
- Tests: `tests/test_block_structure.py`.

## Behavior Changes to Keep in Mind

- **Tied weights shrink parameter counts.** For example, guac-106m goes from
  106.3M to 75.6M. Preset names haven't been updated.
- **Preset batch sizes now take effect, and some will run out of memory:**
  `guac-280m` (`pdbs_train: 4096`), `guac-1.9b` (512), `guac-3b` (256 × 12
  accumulation; the make default `CONFIG`). Fixing the presets is the user's call;
  `PRETRAIN_ARGS` overrides them per run.
- **Training length comes from the config:** `max_steps` if set, else
  `num_train_epochs`. Streaming datasets (the default `fineweb`) need an explicit
  `MAX_STEPS` or `max_steps`.
- **Old checkpoints don't load.** The Mamba-2 rewrite and then the block
  restructuring (1.1) renamed weights, so every checkpoint from before `4fd35cb`
  fails to load by design. Retrain.
- **Mamba-2 blocks now have an FFN** (1.1), so presets grew: guac-106m from
  74.7M to 76.7M parameters, guac-3b from 2.60B to 3.05B. Conv blocks are about
  the same size as before. Set `ffn_types` to `none` for the `mamba2` layers to
  get the old sizes back.
- **All FFNs are SwiGLU** (1.1), replacing the GELU FFNs of the conv, attention,
  BASED and MoH layers, at about the same parameter count.
- `mamba_d_state` defaults to 128 (Mamba-2 paper), up from 16.

## Decisions Made

- **Block structure (2026-10-16):** every block is `x + mixer(norm(x))` then
  `x + ffn(norm(x))`, so only the mixer varies (as in the Jamba and Granite
  hybrids), with a per-layer option to drop the FFN. This was the review's
  recommendation; the user had no preference, so it went ahead as 1.1.

## Next: Rest of Phase 1 (Mechanism Fidelity)

Line numbers refer to `scripts/training/core/custom_architectures.py` at `4fd35cb`.

### 1.2 BASED rewrite (`BasedMixer`, line 609; `_based_attention`, line 670)

- **Currently:** an O(L²) softmax over `φ(q_i)·cumsum(φ(k))_j`. Because the
  features are ReLU (non-negative), scores never decrease with j, so attention is
  biased toward the most recent token. It is not linear attention.
- **Target** (Arora et al. 2024): a 2nd-order Taylor feature map, normalized
  causal linear attention (a prefix sum of φ(k)⊗v), and small sliding-window
  softmax attention.
- **Test:** the chunked or prefix-sum form matches the explicit quadratic
  linear-attention form.

### 1.3 MoH and MoE

- **MoH** (lines ~784–801): head outputs are summed into one head-sized vector,
  then tiled H times and scaled by H, so the layer has a single head's capacity.
  Fix: `o_proj(concat_h(g_h · head_h))`, with gates scattered per token (routing is
  already per token).
- **MoE** (`SharedExpertMoE`, line 839):
  - The load-balancing loss (line 912) is the constant `w·N` with zero gradient,
    and is never added to the model loss. Implement the Switch-style
    `N·Σ f_i·P_i` and surface it through the model output.
  - The dispatch has a per-token Python loop (line 960), about 15× slower than
    attention. Vectorize it with sort/`index_add_`.
- **Tests:** vectorized MoE equals the loop version; the aux loss has a nonzero
  gradient to the router.

### 1.4 Position encoding

- Learned absolute position embeddings (line 1103) default to 65,536 positions
  (`model_configs.yaml:3`): 33.5M parameters in guac-106m and 201M in guac-3b.
- `use_rope` (line 456) is read but nothing uses it.
- Add RoPE to the attention mixers, and drop or shrink the absolute embeddings so
  preset names reflect real sizes.

### 1.5 Optional: Mamba-2 GPU kernel

Wire `mamba_ssm`'s `mamba_chunk_scan_combined` in as a scan backend when it's
installed and the tensors are on CUDA. Parameter names already match.
Verification needs an Ampere+ GPU: add a test, skipped without CUDA, comparing
it to `ssd_chunked`.

## Phase 2: Research Infrastructure

This is what makes the layer comparisons meaningful.

1. **Validation and metrics.** Pretraining has no held-out set:
   `dataset_utils.py:68` merges wikitext's validation split into train, and
   `pretrain.py:460-461` sets `eval_strategy="no"` and `report_to=[]`. Add
   validation perplexity and metric logging (wandb or JSONL), and save the
   resolved config, git SHA and seed for every run.
2. **Sequence packing.** Documents are truncated at `max_length` and the rest is
   discarded (`dataset_utils.py:133`). Concatenate and chunk instead.
3. **Synthetic diagnostics.** Cheap tasks that isolate what each layer type
   contributes: associative recall (MQAR), selective copying, induction heads,
   passkey retrieval, plus the existing arithmetic.
4. **Sweep harness.** A grid over `layer_types` at a fixed parameter or FLOP
   budget, producing one results table.
5. **Faster generation.** KV-cache / recurrent-state generation.
   `prepare_inputs_for_generation` currently re-runs the full sequence per token
   and drops `attention_mask`.

## Phase 3: Cleanup

- **Dead code:**
  - `matryoshka_layers.py` (never imported; also has no causal mask).
  - `scripts/legacy/` (~3.4k lines; history keeps it).
  - `architecture_configs.yaml` (read only by legacy scripts).
  - Duplicate synthetic-data helpers in `dataset_utils.py` and `posttrain.py`.
- **Consolidate `TrainingArguments`:** each stage script builds its own;
  `pipeline_utils.create_base_training_arguments` was meant to be shared but is
  unused. (Carried over from the former `REMAINING_IMPROVEMENTS.md`.)
- **Silent exception handlers:** audit the remaining `except Exception: print`
  sites in the pipeline scripts and let unexpected errors propagate. (Carried over.)
- **Packaging:**
  - The installed package is literally named `scripts`; rename it `guac`.
  - Drop the triple `try/except ImportError` import fallbacks in 12 files.
  - `trl`, `peft` and `wandb` are only used by legacy code.
  - Set `requires-python` to 3.9+.
- **Broken references:**
  - `core/train_tokenizer.py:57` imports `scripts.training.legacy.*`, which
    doesn't exist; legacy code is in `scripts/legacy/`.
  - The Makefile calls `hf_partial_download.py` at the repo root (lines 469 and
    522); it's in `scripts/`.
  - `datasets_download.csv` isn't shipped.
- Make targets `progressive`, `benchmark` and `expert` launch 9B-scale runs; trim them.

## Known Issues Not Yet Fixed

| Where | Issue |
|-------|-------|
| `posttrain.py:198` | Alpaca's field is `output`, not `response`: KeyError in instruction mode |
| `posttrain.py:301` | `--mode sft --datasets ...` returns an untokenized dataset |
| `posttrain.py:332,355` | `fp16=True` hard-coded; no prompt masking; the synthetic fallback is 8 canned responses |
| `callbacks.py:53` | LR printed with `.2f`, so it always shows `0.00` |
| `callbacks.py:63` | 64-token cache-less generation every logging step (every step when `step_val: 1`) |
| `arithmetic.py:46` | Held-out split hashes `a + b` and `b + a` separately, so commuted twins can leak |
| `path_manager.py` | Creates directories eagerly. Existence checks must look for saved files (already done for caches and tokenizers) |
| transformers 4.46.3 `Mamba2Mixer` | Its CPU path is slightly wrong in the final partial chunk; `tests/test_mamba2.py` works around it |

## Working Conventions

- **Reproduce before claiming.** Check suspected bugs with a probe script or a
  real checkpoint.
- **Every fix gets the test that would have caught it.** Prefer behavioral
  tests: equivalence to a simple reference implementation, perturbation tests,
  round trips.
- **Run the end-to-end smoke test** after model or pipeline changes; unit tests
  don't cover `Trainer`.
- **Keep the docs honest.** Update `docs/ARCHITECTURE.md`, `docs/TRAINING.md` and
  `CLAUDE.md` when behavior changes, and update this file as items land.
