# Contributing to Guac

Guac is a research framework for heterogeneous transformers: models whose
layers use different mechanisms (attention, convolution, Mamba-2, BASED, ...).
Contributions that make comparisons between layer types more faithful or
easier to measure are especially welcome. `docs/ROADMAP.md` lists planned work.

## 🚀 Getting Started

```bash
git clone <repository> && cd guac
make install            # pip install -e ".[test]" and copies site_config.example.yml
make test               # the CPU test suite (about 10 s; no GPU, network or data)
```

Then run the offline end-to-end smoke test in `docs/ROADMAP.md`, which trains
the character-level `guac-13m-arith` preset through pre- and mid-training on CPU.

### Code structure

```
guac/
├── core/            # models (custom_architectures.py, mamba2.py, linear_attention.py),
│                    # data, diagnostics, training utilities
├── pipeline/        # command-line entry points: pretrain, midtrain, posttrain, sweep, ...
└── tokenizer/       # tokenizer vocabulary pipeline
tests/               # pytest suite (tests/README.md)
sweeps/              # sweep specs (sweeps/README.md)
templates/           # Jinja2 prompt templates
scripts/             # standalone utilities (hf_partial_download.py, data scripts)
model_configs.yaml   # presets
site_config.yml      # paths and datasets for your machine (from site_config.example.yml)
```

`guac/README.md` describes each module.

## 🧱 Adding New Layer Types

### Implementation Steps

1. **Create the mixer**: the part of the block that moves information between
positions. The block (`MixedTransformerLayer`) adds the norms, residuals and FFN,
so the mixer returns only its own output:
```python
class MyMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, cache=None):
        # Must be causal: position t may only read positions <= t.
        # attention_mask (batch, total_len) marks padding, which must not reach
        # real tokens (left padding is used in batched generation).
        # cache is a LayerCache (a dict, plus .offset = tokens seen before this
        # call) during generation and None otherwise. Store whatever state lets
        # the next call continue the sequence, e.g. cache["state"] = ...
        return self.proj(my_sequence_mixing(hidden_states))
```

2. **Register it** in `custom_architectures.py`:
```python
SUPPORTED_LAYER_TYPES = (..., "my_mixer")
MIXERS = {..., "my_mixer": MyMixer}
```

3. **Add config parameters** to `MixedLayerConfig` (and pass them through
`build_model_config` in `pretrain.py` if presets should set them).

4. **Run the tests.** The causality, padding, save/load, block-structure and
generation tests (token-by-token decoding must match a full forward pass)
parametrize over `SUPPORTED_LAYER_TYPES`, so they cover the new type
automatically. Don't paper over numerical problems by zeroing NaNs or clamping
outputs: a diverging run should fail visibly.

### Design Guidelines

1. **Faithfulness**: Follow the paper's formulation, and test the mixer against a
   direct reference (a quadratic form, a sequential recurrence, or a reference
   implementation), as `test_based.py` and `test_mamba2.py` do
2. **Complexity**: Document time and memory in sequence length L and hidden size d
3. **Generation**: Keep a fixed-size state if the mechanism allows it
4. **Numerical Stability**: Run scans, exponentials and long sums in float32 under mixed precision (see `mamba2.py`, `linear_attention.py`). Don't mask NaNs or clamp outputs
5. **Documentation**: Include usage examples and parameter explanations

## 📊 Presets

Add a preset to `model_configs.yaml` under `model_training_configs`; it is
merged over `defaults`. `docs/TRAINING.md` lists the keys. Name it after its
parameter count (`print_parameter_breakdown`, or
`python guac/pipeline/pretrain.py --print-params`), and
`tests/test_model_configs.py` checks that every preset builds.

## 🎨 Prompt Templates

Templates are Jinja2 files in `templates/` rendered by
`guac.core.prompt_templates` (`apply_template_file(filename, context)`). They
mark structure with explicit boundary tokens (FLAG tokens such as
`<& THINK &>`, SERIAL start/end pairs) instead of turn markers; the tokens are
defined in `FLAG_TOKEN_NAMES` and `SERIAL_TOKEN_NAMES` and added to every
tokenizer as special tokens.

## 🧪 Experiments

- **Compare layer types with the diagnostics and sweeps** (`docs/TRAINING.md`,
  `sweeps/README.md`), at a matched parameter or FLOP budget, over several seeds.
- **Check a task is learnable before comparing on it**: attention should reach
  near 100% on recall tasks at your settings. MQAR once leaked its answers into
  the inputs, and every mixer scored 36% by elimination.
- **Every run records itself**: `run_info.json` has the resolved config, seed
  and git commit (with a dirty flag), so commit code before running.

## ✅ Before Submitting

- [ ] `python -m pytest` passes, including new tests for the change
- [ ] The smoke test in `docs/ROADMAP.md` still runs, for model or pipeline changes
- [ ] Behavior changes are documented (`docs/ARCHITECTURE.md`, `docs/TRAINING.md`,
      `CLAUDE.md`) and `docs/ROADMAP.md` is updated
- [ ] Suspected bugs were reproduced, and each fix has the test that would have caught it

Write commit messages that say what changed and why, and keep one logical
change per commit.

## 📚 References

- Mamba-2: Dao & Gu (2024), *Transformers are SSMs: Generalized Models and
  Efficient Algorithms Through Structured State Space Duality*
- BASED: Arora et al. (2024), *Simple linear attention language models balance
  the recall-throughput tradeoff*
- MQAR: Arora et al. (2023), *Zoology: Measuring and Improving Recall in
  Efficient Language Models*
- MoH: Jin et al. (2024), *MoH: Multi-Head Attention as Mixture-of-Head Attention*
- Switch Transformers: Fedus et al. (2021), for the MoE load-balancing loss
- RoPE: Su et al. (2021), *RoFormer*
