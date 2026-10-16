# Guac Roadmap

The working plan for Guac, kept current as items land. It comes from a full
codebase review in October 2026. Each finding below was reproduced with a
probe script or on real checkpoints, not just read off the code.

**Status (2026-10-16):** every phase of the review plan has landed: Phase 0
(correctness), Phase 1 (mechanism fidelity), Phase 2 (research
infrastructure) and Phase 3 (cleanup), plus the known issues it listed, and
the open items that could be done without a GPU or network access (Mamba-2's
linear inter-chunk scan, per-document resets for packed sequences, GPU-scale
sweep specs), and the fixes from a second audit of everything the first review
didn't reach (tokenizers, the stage plumbing, the Makefile and jobscripts,
routing, initialization). What is left is in [Open Items](#open-items):
running things on a cluster, and decisions for the user.

## Where Things Stand

| Branch | Commits on top of `main` | Purpose |
|--------|--------------------------|---------|
| `phase-0-correctness` | `3219380`, `761c30f` | Pre-review pending work, then the Phase 0 fixes |
| `phase-1-mamba2` | the above + `446bd60` + `8fdeb01` | Mamba-2 rewrite and the first version of this doc |
| `claude/vibrant-darwin-vs3o5o` | the above + everything below | Phases 1.1 to 3 and the follow-ups; **continue from here** |

None of these branches is merged into `main`. Each item landed as its own commit with tests.

## Setting Up a Fresh Clone

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is enough for tests
pip install -e ".[test]"
python -m pytest                                                     # ~390 tests, ~15 s, no network
```

- `site_config.yml` is git-ignored. Without it, the path manager falls back to
  `site_config.example.yml` with a warning, which is fine for tests and smoke runs.
- `scratch/` (tokenizers, datasets, checkpoints) is git-ignored. A fresh clone has none of it.
- The character tokenizer used by `guac-13m-arith` builds offline:
  `python -m guac.tokenizer.build_char_tokenizer`.
  Every other preset needs the 60k SentencePiece tokenizer (`make train-tokenizer`),
  which downloads data.
- When something else is using the CPU, `OMP_NUM_THREADS=1 python -m pytest` keeps
  the suite from fighting it for cores (tests use tiny models).

### End-to-end smoke test (CPU, offline, about 1 minute)

Unit tests don't exercise `Trainer`, the collators or the stage scripts. After
changing model or pipeline code, also run a real pretrain → midtrain:

```bash
python -m guac.tokenizer.build_char_tokenizer
python - <<'EOF'
import random
from datasets import Dataset
rng = random.Random(0)
words = "the model learns which layer types mix information across tokens".split()
texts = [" ".join(rng.choice(words) for _ in range(rng.randint(8, 120))) for _ in range(400)]
Dataset.from_dict({"text": texts}).save_to_disk("scratch/smoke/tiny_text")
EOF
python guac/pipeline/pretrain.py --model-config guac-13m-arith \
    --datasets scratch/smoke/tiny_text --output-dir scratch/smoke/pre --max-steps 12 --batch-size 4 --grad-accum 2 \
    --eval-steps 6 --validation-docs 40
python guac/pipeline/midtrain.py --base-model scratch/smoke/pre --task arithmetic \
    --output-dir scratch/smoke/mid --num-arithmetic-samples 400 --num-eval-samples 40 \
    --accuracy-eval-per-category 2 --max-steps 4 --eval-steps 2 --batch-size 8 --grad-accum 1
# Optional: post-training on the offline synthetic set
python guac/pipeline/posttrain.py --base-model scratch/smoke/mid --mode instruction --datasets synthetic \
    --output-dir scratch/smoke/post --max-steps 4 --eval-steps 2 --save-steps 4 --batch-size 4 --grad-accum 1
```

Pre-training loss should fall steadily (about 4.7 → 0.4 over the 12 steps) and
end with validation loss 0.39 (perplexity 1.48); midtraining should load the
checkpoint and finish. Each output directory gets `metrics.jsonl` and
`run_info.json`. To check a layer type under training without `Trainer`,
`make diagnostics DIAG_TASK=induction LAYERS=<type>,<type>` takes about a
minute on CPU (2-layer attention reaches 100% by step 500).

## Done

### Phase 0: correctness (`761c30f`)

Weight tying at construction; one `causal_attention` helper on SDPA (fixed NaN
loss with masks); the config is authoritative (no hidden batch caps or step
counts); cache keys cover the tokenizer, dataset settings and debug subset;
`guac-48b` and BASED-without-`num_kv_heads` fixed; layer types validated;
pytest with CPU CI.

### Phase 1: mechanism fidelity

- **Mamba-2 (`446bd60`):** Dao & Gu (2024) with a chunked SSD scan in plain
  PyTorch, exact against the sequential recurrence and against transformers'
  `Mamba2Mixer`. `load_base_model` refuses checkpoints that don't match the code.
- **1.1 Unified block (`4fd35cb`):** every layer is `x + mixer(norm(x))` then
  `x + ffn(norm(x))`; per-layer `ffn_types` (SwiGLU, MoE, none) and widths;
  `param_match`.
- **1.2 BASED (`a09a680`):** Taylor feature map (φ(q)·φ(k) = 1 + s + s²/2),
  normalized causal linear attention computed in chunks with a carried state
  (`linear_attention.py`), padded keys zeroed. The old layer was a softmax over
  prefix sums of ReLU features, biased toward the latest token. New
  `sliding_window_attention` layer type, computed in window-sized blocks.
- **1.3 MoH and MoE (`5acf60c`):** MoH concatenates gated heads before `o_proj`
  (it had summed them into one head's width). MoE has the Switch load-balancing
  loss (it was a constant with no gradient), added to the training loss only and
  returned as `aux_loss`, and vectorized dispatch (sort by expert, `index_add_`).
- **1.4 Position encoding (`acab222`):** RoPE in all softmax-attention mixers;
  the 65,536-row absolute table is opt-in (`use_absolute_position_embeddings`).
  Presets below `guac-3b` were renamed to their real sizes, old names aliased.
- **1.5 Mamba-2 kernel (`99eb536`):** `mamba_scan_impl: mamba_ssm` routes CUDA
  inputs through `mamba_chunk_scan_combined`; opt-in until the GPU test has run.

### Phase 2: research infrastructure

- **2.1 Validation and metrics, 2.2 packing (`050e492`):** pre-training holds out
  validation data (wikitext's own split, or `validation_docs` per dataset) and
  reports loss and perplexity every `eval_steps` and at the end
  (`results.json`). Documents are packed into full blocks with EOS between
  them (`pack_sequences`, default on). Every stage writes `metrics.jsonl` and
  `run_info.json` (resolved config, command, seed, git commit, versions);
  `report_to` passes through (e.g. wandb). The seed is set before the model is
  built. `tokenize_datasets.py` fills the cache pretraining actually reads.
- **2.3 Diagnostics, 2.4 sweeps (`798b1fb`):** MQAR, selective copying,
  induction and passkey tasks (`guac/core/diagnostics.py`,
  `guac/pipeline/run_diagnostics.py`); FLOP counting on the meta device and
  `flop_match`; `guac/pipeline/sweep.py` runs architectures × tasks × seeds at
  a matched budget into one table, resuming interrupted sweeps. First results
  are below.
- **2.5 Cached generation (`d0ce20c`):** `MixedLayerCache` keeps KV for attention
  (only the window for sliding-window layers) and recurrent states for conv,
  Mamba-2 and BASED. Token-by-token decoding matches a full forward pass for
  every type; 128 tokens went from 3.3 s to 0.5 s on CPU. Conv and Mamba-2 now
  zero padding, so left-padded batches work.

### Phase 3: cleanup, and the former known issues

- **Post-training (`090227d`):** Alpaca's `output` field, untokenized
  `--mode sft --datasets`, hard-coded fp16, no prompt masking and the
  8-canned-response fallback are fixed; defaults are alpaca and ultrachat.
- **Callbacks (`090227d`):** LR shown as `0.00`, a KeyError without `grad_norm`,
  and 64-token sampling every logging step are fixed (`sample_every`).
- **Arithmetic split (`090227d`):** commuted twins (`a + b`, `b + a`) now share a
  pool; 90% of held-out a+b problems had their twin in training. Split version 2
  is in cache keys, and `eval_arithmetic.py` warns on older models.
- **Dead code (`ca69b93`):** `scripts/legacy/`, `architecture_configs.yaml`,
  `matryoshka_layers.py`, the unimportable `core/train_tokenizer.py`,
  `BasicTransformer*`, duplicate dataset helpers, unused pipeline helpers.
- **TrainingArguments and errors (`c203e56`):** one `make_training_arguments`
  for the shared conventions; silent fallbacks to stand-in data in tokenizer
  training and mid-training now raise; the memory report measures instead of
  guessing.
- **Make targets (`f4d8e4e`):** download targets call `scripts/hf_partial_download.py`,
  `datasets_download.csv` ships, the 1-9B `progressive`/`benchmark`/`expert` targets are gone.
- **Packaging (`dfa8bec`):** the package is `guac` (`guac/core`, `guac/pipeline`,
  `guac/tokenizer`); no more `try/except ImportError` chains; trl, peft and
  wandb dropped from the dependencies (wandb, mamba-ssm and the data-script
  libraries are extras); Python 3.9+.
- **Paths (`3d07ef2`):** path getters no longer create directories.
- **Docs (`1e8d5da`):** TRAINING, API, CONTRIBUTING and the package READMEs
  rewritten from the code; `site_config.example.yml` defines the default `fineweb`.
- **SLURM (`8d63ece`):** `jobscripts/full_pipeline.sbatch` had a syntax error,
  wrong flags and a commented-out `main`; it now runs the whole pipeline.
- **Review fixes (`7027ee8`):** a streaming `--debug` run polluted the full run's
  validation cache; alpaca + dolly couldn't be mixed; packing failed on streams
  with unknown columns; absolute positions shifted under left padding; the
  tokenizer fingerprint depended on the last call's truncation settings.

### Follow-ups from the open items

- **SLURM guide (`8271aa3`):** an example exported without `ALL`, the size
  ordering was wrong, and it pointed at a `configs/models/` that doesn't exist.
- **GPU-scale sweeps (`e9ae932`):** `sweeps/recall_scaled.yaml` (4 layers,
  hidden 256, 9 architectures, MQAR with 16/32/64 pairs, induction, selective
  copying, passkey, 3 seeds: 162 runs) and `sweeps/lm_scaled.yaml` (WikiText-103
  perplexity). `jobscripts/sweep.sbatch` runs a spec as a job array
  (`sweep.py --shard I/N`, one shared table, `--summarize`); `--param-match` /
  `--flop-match` override a spec's budget; `bf16: true` trains under bfloat16
  autocast.
- **Linear Mamba-2 scan (`bf3bdff`):** the recurrence over chunk states recurses
  into `ssd_chunked` past `2 * chunk_size` chunks, so cost is linear at any
  length (1.53 MFLOP/token at 64k tokens for a 256-wide mixer, was 3.49).
- **Document boundaries (`ec57c32`):** `reset_at_document_boundaries` (or
  `pretrain.py --reset-at-document-boundaries`) splits packed rows at EOS; every
  mixer then keeps documents apart, exactly (tested per layer type, values and
  gradients). Off by default. Models now take pad/BOS/EOS ids from the
  tokenizer (configs all said EOS 2; the character tokenizer's is 3).

### Second audit (October 2026)

Three parallel audits of the code the review hadn't covered; every finding was
reproduced on CPU before it was fixed, and each fix has its test.

- **Tokenizers (`fb0fe51`):** the 60k SentencePiece tokenizer was saved as
  `T5Tokenizer`, which `AutoTokenizer` loads as a `T5TokenizerFast` that turns
  every newline and unseen character into `<unk>`, collapses runs of spaces and
  appends `</s>` (id 2) instead of the EOS token, so packed documents ended
  `</s> <EOS>` and mid-training taught the wrong end token. It is now saved as a
  fast tokenizer built from the model, which encodes exactly as SentencePiece
  does, with the same ids (`python -m guac.tokenizer.convert DIR` fixes an old
  one; the stages refuse unconverted ones). Special-token ids no longer depend
  on `PYTHONHASHSEED`. The vocab pipeline's stages 2 and 3 couldn't learn
  anything (the core vocab passed back as `user_defined_symbols` leaves BPE
  nothing to merge); they now append pieces to the core model. `--tokenizer-type
  bpe` was ignored and the BPE path dropped characters; it is byte-level BPE.
  The character tokenizer turned `"\n\n"` into `<unk>`.
- **Stages (`4796a09`, `20b59a5`):** `--resume-from-checkpoint` crashed in
  every stage (torch >= 2.6 vs transformers 4.46's RNG-state load).
  Checkpoints lacked the tokenizer, so a stage couldn't start from
  `checkpoint-N`. Mid-training early-stopped at 0% held-out accuracy while eval
  loss fell, and both mid- and post-training could only pick the best model at
  every other evaluation. posttrain crashed on `--eval-steps` alone. Packing
  dropped a partial block per 1000-document map batch. `.jsonl` downloads
  weren't loadable and streaming plus on-disk datasets couldn't be mixed.
  `pretrain.py --vocab-size` did nothing (now `--tokenizer`), and `--max-length`
  missed `flop_match`. `metrics.jsonl` repeated steps after a resume.
- **Makefile and jobscripts (`61ba460`):** `full_pipeline.sbatch` trained and
  pre-tokenized with a tokenizer pretrain never loaded (`VOCAB_SIZE`, or
  guac-13m-arith), and hard-coded `scratch/` paths; `make tokenize-datasets`
  wrote a cache `make pretrain` never read (length 1024 vs the preset's).
  `make cache`, `make configs` and the site-config download targets were broken.
- **Model and diagnostics (`0344c2a`..`d8ab676`, `b28bf09`):** with top-1 routing
  the renormalized gate is always 1, so MoE routers learned only from the
  balancing loss and mixture-of-heads routers (top-1 by default for 2-3 heads)
  not at all. Mamba-2's depthwise conv started 10-37x too small (Kaiming fan-out
  ignores groups). Weight decay hit norm gains and Mamba-2's `A_log`, `D` and
  `dt_bias`, in the diagnostics and in the stages' Trainer. Seed 0's training
  batch at step 10,000 was the diagnostics' held-out set. A plain eval-mode
  forward ignored `reset_at_document_boundaries`. Passing a cache to
  `generate()` crashed. Sweeps keyed results by task name, which ignored lm
  `seq_len` and `steps`, so two lm tasks shared data and table columns, and
  edited specs reused stale rows; rows now carry a hash of their settings.

## First Sweep Results

`sweeps/recall_small.yaml` (2 layers, hidden 64, every block parameter-matched
to a `full_attention` block, one seed, CPU). Held-out example accuracy:

| Architecture | Layer types | MQAR (8 pairs, 64 tokens) | Induction (64 tokens) |
|--------------|-------------|---------------------------|------------------------|
| attention | full_attention ×2 | 99.5% | 100% |
| mamba2_attention | mamba2, full_attention | 99.8% | 100% |
| conv_based | conv, based | 99.9% | 100% |
| sliding_window | sliding_window_attention ×2 (window 16) | 63.1% | 4.0% |
| based | based ×2 | 0.0% | 66.3% |
| mamba2 | mamba2 ×2 | 0.4% (61% of answer tokens) | 3.8% |
| conv | conv ×2 | 0.0% | 3.7% (chance) |

What the table shows, at this small budget:
- Attention solves both tasks, and one attention layer is enough to rescue a Mamba-2 model.
- The sliding window recalls only the pairs within reach of its 16 positions.
- BASED needs something to bind each key to the token after it. Alone it
  fails MQAR, since it has no position information. With one convolution
  first, as in the paper, it matches attention.
- Mamba-2 at this size and training length learns part of MQAR and none of
  induction. That reflects the setting (state size 16, lr 1e-3, 1,500 steps), not
  the mixer: with `mamba_d_state: 64`, lr 3e-3 and 3,000 steps (not parameter-matched)
  the same 2-layer model reaches 100% on induction by step 2,000.
- These are one seed of 1,500 to 3,000 CPU steps; read them as a check that the
  harness discriminates, not as findings.
- **After the second audit**, rerunning the same spec for three rows (one seed)
  gives mamba2 **96.0%** on MQAR (was 0.4%) and 22.3% on induction (was 3.8%);
  based 0.0% / 69.6% and conv 0.0% / 2.8% barely move. Mamba-2's conv init and
  weight decay changed (and the diagnostic data, through the new seeds), so the
  table above understates Mamba-2; the other rows predate the fixes too, and
  the whole table is worth rerunning.

## Behavior Changes to Keep in Mind

- **Old checkpoints don't load.** Mamba-2 (446bd60), the block restructuring
  (4fd35cb), BASED (a09a680) and the absolute-position change (acab222) all
  change weights; `load_base_model` refuses mismatched checkpoints. Retrain.
- **Presets are smaller and renamed.** Tied embeddings and dropping the
  position table took guac-106m from 106M to 43M parameters (now `guac-43m`);
  `guac-3b` is 2.85B. Old names still load through `preset_aliases`.
- **Mamba-2 blocks have an FFN** by default; `ffn_types: none` restores the old shape.
- **All FFNs are SwiGLU**, at about the old GELU FFNs' parameter count.
- **Attention layers use RoPE**, and nothing adds absolute positions by default.
- **Pre-training packs documents and evaluates** by default (`pack_sequences`,
  `validation_docs`, `eval_steps: 500`, `save_steps: 1000`); caches built before
  are not reused, since the keys changed.
- **MoE training loss includes the load-balancing term**; eval loss doesn't.
- **Training settings come from the config**, and some presets' batch sizes
  (`guac-160m` 4096, `guac-1.7b` 512, `guac-3b` 256 × 12) won't fit one GPU.
  Fixing them is the user's call; `PRETRAIN_ARGS` overrides per run.
- **Streaming datasets need `max_steps`** (`MAX_STEPS=`).
- **Mid- and post-training use real datasets by default** (codeparrot-clean-valid,
  GSM8K, alpaca, ultrachat), so they need network access; `--datasets synthetic`
  is the offline post-training smoke set.
- **Paths:** scripts moved from `scripts/training/` to `guac/`
  (`python guac/pipeline/pretrain.py`); imports are `from guac.core...`.
- **SentencePiece tokenizers from before the second audit are refused**
  until converted (`python -m guac.tokenizer.convert DIR`, ids unchanged).
  Caches and models built through the broken loading are worth redoing; the
  cache keys change with the tokenizer, so caches redo themselves. Special-token
  ids of newly trained tokenizers differ from older ones (fixed order now).
- **Training changed for new runs:** top-1 MoE and mixture-of-heads gates are
  the router probability (below 1); Mamba-2's conv and grouped convs initialize
  differently; norm gains, biases and `A_log`/`D`/`dt_bias` aren't decayed;
  diagnostics draw different data. The First Sweep Results won't reproduce
  exactly, and existing sweep directories rerun once (old rows have no hash).
- **Mid- and post-training checkpoint at every evaluation** by default, and
  mid-training's early stopping waits while either eval loss or held-out
  accuracy improves.
- **`pretrain.py --vocab-size` is gone**; pass `--tokenizer DIR`. The sbatch
  `VOCAB_SIZE` / `TOKENIZER_TYPE` now select (and train) the tokenizer every
  stage uses; unset, it is the preset's.
- **With `reset_at_document_boundaries`, `model(x)` returns no cache** unless
  `use_cache=True` is passed.

## Decisions Made

- **Block structure (1.1):** every block is `x + mixer(norm(x))` then
  `x + ffn(norm(x))`, so only the mixer varies (as in Jamba and Granite), with
  a per-layer option to drop the FFN. The review's recommendation; the user
  had no preference.
- **BASED's sliding-window attention is its own layer type**, not part of the
  BASED mixer. The paper builds BASED models as stacks of linear-attention,
  sliding-window and convolution layers, and separate types let sweeps measure
  each part. Note that BASED layers have no position information of their own
  (as in the paper), so a stack needs a convolution or attention layer to bind
  keys to the tokens that follow them.
- **Absolute position embeddings are off by default**: attention gets RoPE,
  and conv, Mamba-2 and BASED see order through their causal structure.
- **Renamed presets keep their old names as aliases** rather than breaking
  existing commands. Only presets more than 10% from their names were renamed.
- **The mamba_ssm kernel is opt-in** until its comparison test has run on a GPU.
- **MQAR shows answers only as labels** (as Zoology does). Feeding answers back
  let every mixer reach 36% by elimination.

## Open Items

| Item | Notes |
|------|-------|
| Run the GPU tests | `test_kernel_scan_matches_torch_scan` (needs CUDA + `mamba_ssm`); then consider defaulting `mamba_scan_impl` to the kernel on CUDA |
| Network-dependent defaults are unrun here | codeparrot-clean-valid (`content`), GSM8K (`question`/`answer`), alpaca, ultrachat (`messages`) and the `fineweb` site-config entries were written from their dataset cards; the offline paths are tested |
| Run the scaled sweeps | `sbatch jobscripts/sweep.sbatch` (parameter-matched), again with `BUDGET="--flop-match full_attention"`, and with `SPEC=sweeps/lm_scaled.yaml`. Untested on a GPU: the specs' settings (steps, learning rate, batch size) are first guesses, and one learning rate for every architecture is a confound (see the spec's comment). Also worth a run: `lm_scaled.yaml` with `reset_at_document_boundaries: true` in `base` |
| Document resets on GPU | `reset_at_document_boundaries` is tested on CPU only; the `mamba_ssm` kernel path passes `seq_idx` untested. With it on, attention uses an explicit mask instead of the causal FlashAttention path; a varlen kernel would avoid that |
| Preset batch sizes | See Behavior Changes; the user's call |
| MoE dispatch syncs | `bincount(...).tolist()` is one host sync per MoE layer per step; fine at research scale |
| No fused kernels for BASED and sliding-window attention | Plain PyTorch; BASED's step is ~1.5× attention's at small sizes on CPU |
| transformers 4.46.3 `Mamba2Mixer` | Its CPU path is slightly wrong in the final partial chunk; `tests/test_mamba2.py` works around it |
| CI | `.github/workflows/tests.yml` has been pushed on several branches, but GitHub lists no workflows and no runs for the repository (checked 2026-10-16), so Actions is probably disabled in the repository settings |
| `docs/modules.txt` | Cluster module versions (Transformers 4.39/4.44) predate the pinned 4.46.3 |
| Convert the cluster's tokenizers | Any SentencePiece tokenizer (and model directory) made before the second audit: `python -m guac.tokenizer.convert DIR`, then re-tokenize and retrain what used it |
| `huggingface_cache` in site_config | Never read (HF caches stay in `~/.cache/huggingface`); wire it to `HF_HOME` or drop the key — the user's call, since it moves caches on the cluster |
| `Auto*` registration | The model isn't registered with `AutoConfig`/`AutoModelForCausalLM`, so tools that load through `Auto*` can't open checkpoints; load with `MixedLayerForCausalLM` |
| Passed cache with beam search | `generate(past_key_values=..., num_beams>1)` is untested |

## Working Conventions

- **Reproduce before claiming.** Check suspected bugs with a probe script or a
  real checkpoint.
- **Every fix gets the test that would have caught it.** Prefer behavioral
  tests: equivalence to a simple reference implementation, perturbation tests,
  round trips.
- **Check a diagnostic is learnable before comparing on it**: attention should
  solve it at your settings.
- **Run the end-to-end smoke test** after model or pipeline changes; unit tests
  don't cover `Trainer`.
- **Keep the docs honest.** Update `docs/ARCHITECTURE.md`, `docs/TRAINING.md` and
  `CLAUDE.md` when behavior changes, and update this file as items land.
