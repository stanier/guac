# The `guac` package

```
guac/
├── core/                       # Models and shared training code
│   ├── custom_architectures.py # MixedLayerConfig, the mixers, the shared block, MixedLayerForCausalLM,
│   │                           #   cached generation, parameter/FLOP matching
│   ├── mamba2.py               # Mamba-2 mixer and chunked SSD scan (optional mamba_ssm kernel)
│   ├── linear_attention.py     # BASED's Taylor feature map and chunked causal linear attention
│   ├── diagnostics.py          # Synthetic tasks (MQAR, selective copying, induction, passkey) and a small training loop
│   ├── arithmetic.py           # Arithmetic problems, the train/held-out split, accuracy evaluation
│   ├── dataset_utils.py        # Pre-training data: validation splits, packing, tokenization cache
│   ├── cache_utils.py          # Cache keys and cache directories
│   ├── pipeline_utils.py       # Presets, model loading, TrainingArguments, metrics and run records
│   ├── callbacks.py            # Progress bar with live samples, profiling, layer timing
│   ├── path_manager.py         # Output paths from site_config.yml
│   └── prompt_templates.py     # Special tokens and Jinja2 prompt templates
├── pipeline/                   # Command-line entry points (see pipeline/README.md)
│   ├── pretrain.py, midtrain.py, posttrain.py
│   ├── eval_arithmetic.py
│   ├── run_diagnostics.py, sweep.py
│   └── train_tokenizer.py, tokenize_datasets.py
└── tokenizer/                  # Hierarchical tokenizer vocabulary pipeline and the character tokenizer
```

Imports are plain: `from guac.core.custom_architectures import MixedLayerForCausalLM`.
`make install` (`pip install -e ".[test]"`) installs the package; the scripts
in `pipeline/` also run directly (`python guac/pipeline/pretrain.py ...`),
since they add the repository root to `sys.path`.

The model is described in `docs/ARCHITECTURE.md`, its API in `docs/API.md`,
and training in `docs/TRAINING.md`.
