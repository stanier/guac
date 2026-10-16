# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Guac is a research framework for training and experimenting with **heterogeneous transformer architectures** — models where different layers use entirely different computational mechanisms (attention, sliding-window attention, Conv1D, Mamba-2, BASED, mixture-of-heads, MLP-only) stacked together in a single network. The research goal is understanding how mixed layer types interact and what inductive biases they contribute.

**Current work plan:** `docs/ROADMAP.md` has the status, decisions made, the next steps (with file and line pointers), known unfixed issues, and an offline end-to-end smoke test. Read it before starting work, and update it as items land.

## Commands

**Setup:**
```bash
make install
```

**Three-stage training pipeline (primary workflow):**
```bash
make pipeline                         # Full pre→mid→post training
make pipeline CONFIG=guac-3b             # With specific model size
make pipeline TASK=arithmetic         # With task specialization
make pipeline DEBUG=true              # Smaller datasets for debugging
make pipeline FORCE_RETOKENIZE=true   # Force re-tokenization
```

**Individual stages:**
```bash
make pretrain
make midtrain
make posttrain
```

**Tests:**
```bash
make test                                      # python -m pytest
python -m pytest tests/test_path_manager.py    # Single test module
```

**Comparing layer types:**
```bash
make diagnostics DIAG_TASK=mqar LAYERS=conv,based   # one model on a synthetic task (guac/pipeline/run_diagnostics.py)
make sweep SPEC=sweeps/recall_small.yaml            # architectures x tasks at a matched budget (sweeps/README.md)
```

**Utilities:**
```bash
make status      # Pipeline status
make cache       # Show cached dataset info
make configs     # Show available model configs
make clean       # Clean all outputs
```

## Architecture

### Mixed Layer Composition

The core novelty is `MixedLayerConfig` in `guac/core/custom_architectures.py`, which allows heterogeneous layer stacks. Every layer is the same pre-norm block, `x + mixer(norm(x))` then `x + ffn(norm(x))`; only the mixer depends on the layer type (the `MIXERS` registry), and the FFN is chosen per layer (`ffn_types`: SwiGLU, MoE or none). `param_match`/`flop_match` size FFNs so blocks match a reference block's parameters or FLOPs. Mixers:

- **Attention** (`full_attention`, `flash_attention`): grouped-query softmax attention with RoPE, via PyTorch SDPA
- **Sliding-window attention**: the same over the last `sliding_window_size` positions, as paired with BASED
- **Mixture-of-Head attention**: per-token top-k heads with gates
- **Conv1D**: causal convolution with configurable kernel size
- **Mamba-2**: selective state space, chunked SSD scan (`mamba2.py`; optional `mamba_ssm` kernel)
- **BASED**: linear attention with a Taylor feature map and a fixed-size state (`linear_attention.py`)
- **MLP-only / shared_expert_moe**: no mixer, only the FFN (SwiGLU or MoE)

Generation caches each mixer's state (KV for attention, recurrent states otherwise).

### Three-Stage Training Pipeline

| Stage | Script | Purpose | Default Datasets |
|-------|--------|---------|-----------------|
| Pre-training | `guac/pipeline/pretrain.py` | Foundation LM, validation perplexity | `fineweb` (site config), WikiText-103, saved datasets |
| Mid-training | `guac/pipeline/midtrain.py` | Task specialization | Arithmetic (generated, held-out split), code (codeparrot-clean-valid), reasoning (GSM8K) |
| Post-training | `guac/pipeline/posttrain.py` | Instruction tuning, loss on responses only | Alpaca (instruction), UltraChat (sft) |

Every run writes `metrics.jsonl` and `run_info.json` (resolved config, seed, git commit) to its output directory.

### Key Source Files

- `guac/core/custom_architectures.py` — The config, mixers, shared block, model, generation cache, parameter/FLOP matching; Mamba-2's scan is in `guac/core/mamba2.py` and BASED's in `guac/core/linear_attention.py`
- `guac/core/diagnostics.py` — Synthetic tasks (MQAR, selective copying, induction, passkey) and the small training loop used by `run_diagnostics.py` and `sweep.py`
- `guac/core/arithmetic.py` — Arithmetic problems and the train/held-out split
- `guac/core/pipeline_utils.py` — Preset loading, model loading, shared TrainingArguments, metrics and run records
- `guac/core/callbacks.py` — Progress bar with live samples, profiling
- `guac/core/dataset_utils.py` — Pre-training data: validation splits, packing, tokenization cache
- `guac/core/path_manager.py` — Centralized path resolution from `site_config.yml`
- `guac/core/prompt_templates.py` — Jinja2-based prompt system with custom tokens

### Configuration

**`model_configs.yaml`** — Model size presets, named after their approximate parameter count. Sizes range from `guac-43m` (512 hidden, 4 layers) through `guac-3b` (3072 hidden, 24 layers, development default) up to `guac-48b` (8192 hidden, 64 layers). Each preset includes hyperparameters (lr, batch size, warmup ratio).

**`site_config.yml`** — Environment-specific paths and dataset configurations (git-ignored; copy from `site_config.example.yml`). Controls `scratch_dir` layout, HuggingFace dataset streaming settings, and which datasets are used per stage.

All output paths (tokenized datasets, checkpoints, models) resolve through `PathManager` from `site_config.yml`, not hardcoded. The `scratch/` directory holds all generated artifacts.

### Dataset Management

`datasets_download.csv` is a manifest that `make download-datasets` feeds line by line to `scripts/hf_partial_download.py`. Supports partial downloads by size (`100MB`, `50K`, `10%`).

```bash
make download-datasets                                          # every line of datasets_download.csv
python scripts/hf_partial_download.py HuggingFaceFW/fineweb-edu --size 100MB --config sample-10BT
```

## Testing

Tests live in `tests/` and run with pytest on CPU, with no GPU, network or trained artifacts needed; CI runs them on every push. They cover save/load round trips, mask handling, causality and cached generation for every layer type, the Mamba-2, BASED, sliding-window, mixture-of-heads and MoE layers against reference implementations, parameter/FLOP matching, preset validity, pre-training data (splits, packing, cache keys), the diagnostics and sweep harness, post-training data formatting, the arithmetic split, the path manager and the tokenizer vocab pipeline. See `tests/README.md`.
