# Guac Tests

Tests run with [pytest](https://docs.pytest.org/) on CPU, need no GPU, network access or trained artifacts, and run in CI on every push (`.github/workflows/tests.yml`).

## Running Tests

```bash
pip install -e ".[test]"                       # once (make install does this)
make test                                      # or: python -m pytest
python -m pytest tests/test_architecture.py    # one module
python -m pytest -k causality                  # tests matching a name
```

## What's Covered

| File | Covers |
|------|--------|
| `test_architecture.py` | Save/load round trip (tied and untied embeddings), `load_base_model` rejecting checkpoints that don't match the current layer code, attention-mask handling for every layer type (all-ones, right padding, left padding), config validation |
| `test_generation.py` | Cached generation: token-by-token and multi-token decoding match a full forward pass for every layer type, left-padded batches match single prompts, `generate()` gives the same tokens with and without the cache (greedy and beam search) |
| `test_position_encoding.py` | RoPE (relative, norm-preserving, gives attention token order), the optional absolute table, former preset names, renamed presets within 10% of their names |
| `test_mamba2.py` | Chunked SSD scan against the sequential recurrence (values and gradients, with document resets, and with more chunks than the chunk size, where the inter-chunk scan recurses), chunk-size invariance, agreement with transformers' reference `Mamba2Mixer`, initialization ranges |
| `test_causality.py` | Every layer type is causal: perturbing position t never changes outputs before t |
| `test_model_configs.py` | Every preset in `model_configs.yaml` builds a valid model config; settings pass through to the model; scientific notation reads as numbers; pretraining uses the config's batch/step settings as given; shared TrainingArguments conventions; midtraining text tasks |
| `test_cache_keys.py` | Tokenized-dataset cache keys change with the tokenizer, dataset settings, sequence length, split, packing, hold-out size and debug subset |
| `test_pretraining_data.py` | Validation hold-outs (seeded, streaming, on-disk splits), packing keeps every token, the packed collator keeps EOS in the loss, perplexity in logs and `metrics.jsonl`, `run_info.json` |
| `test_block_structure.py` | Every layer type is the same pre-norm mixer + FFN block; per-layer FFN types and widths; parameter and FLOP matching; FLOP counts against a hand count |
| `test_based.py` | BASED: Taylor feature map, chunked linear attention against the quadratic form (values, gradients, carried state, document resets); sliding-window attention against a banded mask |
| `test_document_boundaries.py` | `reset_at_document_boundaries`: every layer type gives packed documents exactly their outputs alone, no gradient crosses a document, batched rows with padding, generation ignores it, the tokenizer's EOS reaches the config |
| `test_moh.py` | Mixture-of-Head attention: with all heads at equal gates it is multi-head attention, unselected heads contribute nothing |
| `test_moe.py` | MoE: vectorized dispatch against a per-token loop, the Switch load-balancing loss (formula, router gradient, padding), added to the training loss only, under gradient checkpointing too |
| `test_diagnostics.py` | Each synthetic task's answers follow its rule (MQAR never shows an answer in the input), accuracy counts only labeled positions, the training loops run |
| `test_sweep.py` | Sweep specs (the examples included) are validated, budgets reach every architecture, results land in one table, an interrupted sweep resumes, shards split the runs and share one table, budget overrides |
| `test_posttrain.py` | Alpaca, Dolly and conversation records format; loss covers only responses and assistant turns; truncation |
| `test_arithmetic.py` | The train/held-out split keeps commuted twins (`a + b`, `b + a`) in the same pool |
| `test_path_manager.py` | All generated paths resolve under `SCRATCH_DIR`; getters don't create directories |
| `test_vocab_pipeline.py` | Tokenizer vocab spec loading, validation and partition deduplication |

## Configuration

Tests use `site_config.yml`, falling back to `site_config.example.yml` (with a warning) when it doesn't exist, as in CI. `test_path_manager.py` creates temp directories under the configured scratch directory; everything else writes to pytest's `tmp_path`. One test (`test_kernel_scan_matches_torch_scan`) needs a CUDA GPU and `mamba_ssm`, and is skipped otherwise.

## Adding Tests

Name files `test_<component>.py` and test functions `test_<behavior>`, and use plain `assert`. A test that signals failure by returning `False` passes under pytest. Prefer `@pytest.mark.parametrize` over loops, so a failure names the case that broke, and keep models tiny (hidden size 32-64) so the suite stays fast.

Start each file with the project-root path setup the existing tests use:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
```
