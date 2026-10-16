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
| `test_mamba2.py` | Chunked SSD scan against the sequential recurrence (values and gradients), chunk-size invariance, agreement with transformers' reference `Mamba2Mixer`, initialization ranges |
| `test_causality.py` | Every layer type is causal: perturbing position t never changes outputs before t |
| `test_model_configs.py` | Every preset in `model_configs.yaml` builds a valid model config; pretraining uses the config's batch/step settings as given |
| `test_cache_keys.py` | Tokenized-dataset cache keys change with the tokenizer, dataset settings, sequence length and debug subset |
| `test_block_structure.py` | Every layer type is the same pre-norm mixer + FFN block; per-layer FFN types and widths; parameter matching |
| `test_based.py` | BASED: Taylor feature map, chunked linear attention against the quadratic form (values, gradients, carried state); sliding-window attention against a banded mask |
| `test_moh.py` | Mixture-of-Head attention: with all heads at equal gates it is multi-head attention, unselected heads contribute nothing |
| `test_moe.py` | MoE: vectorized dispatch against a per-token loop, the Switch load-balancing loss (formula, router gradient, padding), added to the training loss only, under gradient checkpointing too |
| `test_path_manager.py` | All generated paths resolve under `SCRATCH_DIR` |
| `test_vocab_pipeline.py` | Tokenizer vocab spec loading, validation and partition deduplication |

## Configuration

Tests use `site_config.yml`, falling back to `site_config.example.yml` (with a warning) when it doesn't exist, as in CI. `test_path_manager.py` creates empty directories under the configured scratch directory.

## Adding Tests

Name files `test_<component>.py` and test functions `test_<behavior>`, and use plain `assert`. A test that signals failure by returning `False` passes under pytest. Prefer `@pytest.mark.parametrize` over loops, so a failure names the case that broke, and keep models tiny (hidden size 32-64) so the suite stays fast.

Start each file with the project-root path setup the existing tests use:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
```
