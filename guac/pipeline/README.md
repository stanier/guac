# Pipeline scripts

Each script is a command-line entry point; `make` targets wrap the common
cases (`make help`). Every training run writes `metrics.jsonl` and
`run_info.json` (resolved config, command, seed, git commit, versions) to its
output directory. See `docs/TRAINING.md` for details.

## Three-stage training

```bash
# 1. Pre-training: a preset from model_configs.yaml on one or more datasets
python guac/pipeline/pretrain.py --model-config guac-43m --datasets wikitext-103 --max-steps 2000

# 2. Mid-training: specialize a pre-trained model
python guac/pipeline/midtrain.py --base-model scratch/models/pretraining/guac-43m --task arithmetic
python guac/pipeline/midtrain.py --base-model ... --task code        # codeparrot-clean-valid, or --datasets
python guac/pipeline/midtrain.py --base-model ... --task reasoning   # GSM8K, or --datasets
python guac/pipeline/midtrain.py --base-model ... --task custom --datasets /path/to/saved_dataset

# 3. Post-training: instruction tuning (alpaca by default) or conversations (ultrachat by default)
python guac/pipeline/posttrain.py --base-model scratch/models/midtraining/... --mode instruction
python guac/pipeline/posttrain.py --base-model ... --mode sft --datasets chats.jsonl
```

| Script | Key options |
|--------|-------------|
| `pretrain.py` | `--model-config`, `--datasets`, `--max-steps`, `--batch-size`, `--grad-accum`, `--max-length`, `--validation-docs` (0 disables evaluation), `--pack/--no-pack`, `--eval-steps`, `--save-steps`, `--seed`, `--report-to`, `--debug`, `--print-params`, `--print-mem` |
| `midtrain.py` | `--base-model`, `--task {arithmetic,code,reasoning,custom}`, `--datasets`, schedule flags (`--epochs`, `--max-steps`, `--learning-rate`, ...), arithmetic: `--reverse-answers`, `--answer-only-loss`, `--accuracy-eval-per-category` |
| `posttrain.py` | `--base-model`, `--mode {instruction,sft}`, `--datasets` (alpaca, dolly, ultrachat, `.json`/`.jsonl`, a saved dataset, or `synthetic` for smoke tests), `--train-on-prompts`, schedule flags |

Pre-training holds out a validation set and reports perplexity; preset keys
(`pack_sequences`, `validation_docs`, `eval_steps`, `save_steps`, `seed`,
`report_to`, `sample_every`) set the defaults the flags override.

## Evaluation and experiments

```bash
python guac/pipeline/eval_arithmetic.py --model scratch/models/midtraining/...   # exact-match accuracy
python guac/pipeline/run_diagnostics.py --task mqar --layer-types conv,based,sliding_window_attention,based
python guac/pipeline/sweep.py sweeps/recall_small.yaml                         # see sweeps/README.md
```

## Tokenizers and data

```bash
python guac/pipeline/train_tokenizer.py --vocab-size 60000 --datasets wikitext-103
python -m guac.tokenizer.build_char_tokenizer                 # offline, for guac-13m-arith
python guac/pipeline/tokenize_datasets.py --tokenizer scratch/tokenizers/sentencepiece/shared_60k \
    --datasets wikitext-103 --max-length 1024                 # fills pretrain.py's cache ahead of time
```
