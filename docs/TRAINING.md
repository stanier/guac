# Guac Training Guide

How to train Guac models: the three-stage pipeline, presets, data, and the
experiment tools. `guac/pipeline/README.md` lists every script's options.

## 🚀 Quick Start

```bash
make install                              # pip install -e ".[test]"
make train-tokenizer                      # 60k SentencePiece tokenizer (downloads data)
make pipeline CONFIG=guac-43m MAX_STEPS=2000   # pre -> mid -> post training
make help                                 # every target and variable
```

Without network access or a trained tokenizer, the offline smoke test in
`docs/ROADMAP.md` runs the character-level `guac-13m-arith` preset end to end
on CPU in about a minute.

### The three stages

| Stage | Command | Data | Output (`scratch/models/...`) |
|-------|---------|------|-------------------------------|
| Pre-training | `make pretrain` / `guac/pipeline/pretrain.py` | `DATASETS` (default `fineweb`; see below) | `pretraining/<config>` |
| Mid-training | `make midtrain TASK=arithmetic` / `midtrain.py` | arithmetic (generated), code (codeparrot-clean-valid), reasoning (GSM8K), or `--datasets` | `midtraining/<config>_<task>` |
| Post-training | `make posttrain MODE=instruction` / `posttrain.py` | alpaca (instruction) or ultrachat (sft), or `--datasets`; loss on responses only | `posttraining/<config>_<task>_<mode>` |

Useful make variables: `CONFIG`, `DATASETS`, `MAX_STEPS`, `TASK`, `MODE`,
`DEBUG=true` (small data subsets), `FORCE_RETOKENIZE=true`, and
`PRETRAIN_ARGS="--batch-size 8 --grad-accum 4"` for any other pretrain flag.

## 📊 Presets (`model_configs.yaml`)

A preset merges `defaults` with its own entry, and is named after its size
(`README.md` lists all of them with parameter counts). Former names such as
`guac-106m` still work through `preset_aliases`. The training settings are
used as given:

```yaml
model_training_configs:
  my-model:
    hidden_size: 512
    num_hidden_layers: 4
    num_attention_heads: 4
    num_kv_heads: 1
    layer_types: [conv, conv, mamba2, sliding_window_attention]
    # Optional: per-layer FFN types/widths, or a budget that sizes them
    # ffn_types: [swiglu, swiglu, none, moe]
    # param_match: full_attention        # or flop_match: full_attention
    learning_rate: 1e-3
    warmup_ratio: 0.1
    weight_decay: 0.1
    pdbs_train: 32           # per-device batch size
    pdbs_eval: 32
    grad_accum_steps: 1
    max_sequence_length: 512
    max_steps: 20000         # or num_train_epochs (map-style datasets only)
```

Command-line flags override presets for one run (`--batch-size`,
`--grad-accum`, `--max-steps`, `--seed`, ...). Every layer-type setting below
(`mamba_*`, `based_*`, `conv_*`, `sliding_window_*`, `moe_*`, `moh_*`,
position encoding) can be set in a preset.

### Layer type settings

```yaml
# Mamba-2 configuration
mamba_d_state: 128         # State size per head (higher = more memory)
mamba_d_conv: 4            # Causal convolution kernel size
mamba_expand: 2            # Inner size = expand * hidden_size
mamba_headdim: 64          # Head size; inner size must be divisible by it
mamba_ngroups: 1           # B/C groups shared across heads
mamba_chunk_size: 64       # Scan chunk length (speed/memory only)
mamba_scan_impl: torch     # or mamba_ssm (fused kernel on CUDA; pip install mamba-ssm)

# BASED configuration
based_feature_dim: 16      # Query/key size per head before the Taylor feature map
based_num_heads: 8         # Number of heads (default num_attention_heads)
based_num_key_value_heads: 4 # KV heads (≤ num_heads; default num_kv_heads)
based_chunk_size: 64       # Scan chunk length (speed/memory only)

# Sliding-window attention
sliding_window_size: 64    # Positions each token sees, itself included

# Position encoding
use_rope: true             # Rotary embeddings in the softmax-attention mixers
rope_theta: 10000.0
use_absolute_position_embeddings: false  # Learned table of max_position_embeddings rows

# Convolutional configuration
conv_kernel_size: 3        # 1D convolution kernel
conv_groups: 1             # Depthwise convolution groups

# Output head
tie_word_embeddings: true  # Share the token embedding matrix with lm_head
```

All softmax-attention layers (`full_attention`, `flash_attention`, `sliding_window_attention`, `moh_attention`) are always causal and run through PyTorch's `F.scaled_dot_product_attention`, which uses FlashAttention kernels on supported GPUs and dtypes. The separate `flash-attn` package is not used.

## 🗂️ Data

### Pre-training datasets

`--datasets` (make `DATASETS=`) takes a comma-separated list of:

- names defined under `dataset_configs.pretraining_datasets` in
  `site_config.yml` (copy `site_config.example.yml`, which defines `fineweb` and
  `fineweb-edu`). Each sets the Hugging Face id, config, `streaming`,
  `max_articles` and `text_field`. Streaming datasets have no length, so they
  need `--max-steps`.
- the built-in `wikitext-103` and `openwebtext`
- a path to a dataset saved with `save_to_disk` (a `DatasetDict` with a
  `validation` split uses it)

An unknown name is an error. To download part of a dataset ahead of time,
see `docs/DATASET_DOWNLOAD.md` (`make download-datasets` reads
`datasets_download.csv`).

## 📈 Data, Evaluation and Run Records

### Pre-training data

- **Validation**: every pre-training run holds out a validation set and reports
  its loss and perplexity every `eval_steps` (default 500) and at the end.
  wikitext-103 (and on-disk `DatasetDict`s with a `validation` split) use their
  own validation split. Other datasets hold out `validation_docs` documents each
  (default 1000; a seeded random sample, or the first documents of a streaming
  dataset). Set `validation_docs: 0` (or `--validation-docs 0`) to skip evaluation.
- **Packing** (`pack_sequences`, default on; `--no-pack` to turn off): documents
  are joined with EOS between them and cut into blocks of exactly
  `max_sequence_length` tokens, so only the final partial block is thrown
  away and no batch holds padding. Documents run across block boundaries, and
  by default layers aren't reset at EOS (as in GPT-style pre-training);
  `--reset-at-document-boundaries` (or `reset_at_document_boundaries: true` in
  the config) keeps each document's context to itself, for attention and
  recurrent layers alike (see `docs/ARCHITECTURE.md`). Unpacked, each
  document is truncated to `max_sequence_length` and the rest discarded.
- Tokenized splits are cached under keys that cover the tokenizer, dataset
  settings, sequence length, packing and hold-out size.
  `make tokenize-datasets CONFIG=<preset>` fills the same cache ahead of time
  (pass `make pretrain` the same `TOKENIZER_PATH` / `MAX_LENGTH`, if any).
- Datasets are site-config names, built-in names (`wikitext-103`,
  `openwebtext`), datasets saved with `save_to_disk`, or data files
  (`.jsonl`, `.json`, `.parquet`, `.csv`, `.txt`, or a directory of them, such
  as `hf_partial_download.py` writes). Streaming and on-disk datasets mix.

### What every run writes

| File | Contents |
|------|----------|
| `metrics.jsonl` | One JSON object per log entry: training loss, perplexity, learning rate and gradient norm every `step_val` steps; validation loss and perplexity (and midtraining's held-out accuracy) at each evaluation |
| `run_info.json` | Everything needed to reproduce the run: resolved config (preset plus overrides), command line, seed, git commit and whether the tree was dirty, library versions, parameter count, model config |
| `results.json` | Pre-training only: final training loss, steps, runtime, and final validation loss and perplexity |

Set `report_to: wandb` (or `--report-to wandb`) to also send metrics to
Weights & Biases. `seed` (default 42; `--seed`) seeds model initialization as
well as data order. Checkpoints are written every `save_steps` (default 1000)
and include the tokenizer, so a later stage can start from one. Mid- and
post-training checkpoint at every evaluation by default, since the best model
(`load_best_model_at_end`) is chosen among checkpoints. Mid-training stops
early after `--early-stopping-patience` evaluations (default 2) in which neither
eval loss nor held-out accuracy improved.

## 🧪 Synthetic Diagnostics and Sweeps

`guac/core/diagnostics.py` has four synthetic tasks, each
isolating one ability. They generate token ids directly, with no tokenizer,
and put labels only on the answer:

| Task | Tests | Format |
|------|-------|--------|
| `mqar` | Associative recall (Zoology) | key-value pairs, then queried keys; predict each key's value |
| `selective_copy` | Content-dependent memory (Mamba) | data tokens scattered among blanks; repeat them in order after a separator |
| `induction` | Copying a matched context | `TRIGGER X` once early on; predict `X` after the final `TRIGGER` |
| `passkey` | Long-range retrieval | digits hidden at a random depth in filler; repeat them at the end |

Arithmetic, through `midtrain.py` and `make eval-arithmetic`, completes the set.

```bash
# One model, one task: trains on fresh batches, reports held-out example and token accuracy
python guac/pipeline/run_diagnostics.py --task mqar --layer-types conv,based,sliding_window_attention,based \
    --hidden-size 128 --steps 3000 --task-arg num_kv_pairs=16 --task-arg seq_len=128 --param-match full_attention
make diagnostics DIAG_TASK=induction LAYERS=mamba2,mamba2

# A grid of architectures x tasks at a matched parameter or FLOP budget, one results table
python guac/pipeline/sweep.py sweeps/recall_small.yaml
```

See `sweeps/README.md` for the sweep spec. `param_match` and `flop_match`
(also usable in presets) size each block's FFN so every block has the
parameters, or the forward FLOPs per token at `max_sequence_length`, of a
reference block.

## 🛠️ Troubleshooting

**Loss becomes NaN or diverges.** Nothing in the model masks NaNs, so a
diverging run fails visibly. Lower `learning_rate`, lengthen warmup, keep
`max_grad_norm: 1.0`, and check whether one layer type is responsible by
training a small model of only that type (`make diagnostics`). Mamba-2 and
BASED compute their scans in float32 under mixed precision.

**Out of memory.** Reduce `pdbs_train` (or `--batch-size`) and raise
`grad_accum_steps` to keep the effective batch, or shorten
`max_sequence_length`. Pre-training uses gradient checkpointing by default.
`--print-mem` reports parameter, gradient, optimizer-state and activation
memory after the first step. Several large presets' batch sizes don't fit
on one GPU as written.

**Slow training.** Softmax attention uses FlashAttention kernels through
`F.scaled_dot_product_attention` on supported GPUs. For long Mamba-2
sequences, try `mamba_scan_impl: mamba_ssm` (CUDA, `pip install -e ".[kernels]"`)
or a larger `mamba_chunk_size`. `--enable-profiling` adds timing callbacks.

**Poor results.** Evaluate on held-out data rather than training loss:
pre-training reports validation perplexity, mid-training held-out arithmetic
accuracy (`make eval-arithmetic`), and the diagnostics isolate single abilities.

## 🔄 Other Workflows

**Multiple GPUs.** The stage scripts use the Hugging Face Trainer, so
`torchrun --nproc_per_node=4 guac/pipeline/pretrain.py --model-config ...`
runs data-parallel training. `docs/SLURM.md` covers cluster jobs.

**Resuming.** Pass `--resume-from-checkpoint scratch/models/pretraining/<config>/checkpoint-N`
(any stage).

**Tokenizers trained before October 2026.** SentencePiece tokenizers used to be
saved as `T5Tokenizer`, which `AutoTokenizer` loads as a `T5TokenizerFast` that
turns newlines and unseen characters into `<unk>` and ends text with `</s>`
instead of the EOS token. The stages refuse them;
`python -m guac.tokenizer.convert <dir>` rewrites one in place (ids are
unchanged). Caches and models made with the broken one are worth redoing.

**Reproducing a run.** `run_info.json` in the output directory has the
resolved config, command line, seed and git commit.
