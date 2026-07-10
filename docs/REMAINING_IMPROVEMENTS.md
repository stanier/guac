# Remaining Improvements

Items identified during the October 2026 refactor session that would close the gap from A- to A.

## 1. Tokenizer pipeline tests

No tests exist for `scripts/training/tokenizer/`. At minimum, smoke tests covering:

- `VocabSpec.from_yaml()` loads and validates correctly
- `VocabSpec.validate()` raises on invalid budget configurations
- `deduplicate_partitions()` with both `first-wins` and `coverage` strategies
- `extract_tokens_from_model()` returns non-empty list from a trained model
- `VocabPipeline.status()` runs without error on a fresh (unbuilt) spec

The existing test runner is at `tests/run_tests.py`. Add `tests/test_vocab_pipeline.py` following the pattern of `tests/test_architecture_config.py`.

## 2. Unify stage-specific TrainingArguments constructors

Each pipeline script owns its own `TrainingArguments` factory:

- `pretrain.py` → `create_training_arguments(config, output_dir)`
- `midtrain.py` → `create_midtraining_arguments(output_dir, learning_rate)`
- `posttrain.py` → `create_posttraining_arguments(output_dir, mode)`

`pipeline_utils.py` already has `create_base_training_arguments(output_dir, config, stage)` which handles all three stages via a `stage` switch. The per-script constructors are redundant and have drifted from the shared one (different batch sizes, schedulers, eval strategies). Consolidate: remove the per-script constructors and route each script through `create_base_training_arguments` with its appropriate `stage` argument.

## 3. Audit remaining silent exception handlers

The bulk of silent swallows were fixed this session. A few `except Exception as e: print(...)` patterns in older code paths were not reviewed:

- `pipeline_utils.py` — `save_model_and_tokenizer` was fixed, but scan the rest of the file
- `pretrain.py` — several `load_site_config` and `check_for_downloaded_dataset` exception handlers print and return empty rather than propagating
- `midtrain.py` / `posttrain.py` — dataset loading fallbacks

The rule: catch specific exceptions at system boundaries (user input, external I/O), let unexpected exceptions propagate so they surface as tracebacks rather than silent empty returns.
