# Sweeps

A sweep trains every architecture in a grid on every task, at a matched
parameter or FLOP budget, and writes one results table. It is the way to
answer "what does this layer type contribute?" with numbers.

```bash
python guac/pipeline/sweep.py sweeps/recall_small.yaml --dry-run   # sizes and FLOPs only
python guac/pipeline/sweep.py sweeps/recall_small.yaml             # or: make sweep SPEC=sweeps/recall_small.yaml
```

Output goes to `scratch/sweeps/<name>/` (or `--output-dir`):

| File | Contents |
|------|----------|
| `results.md`, `results.csv` | One row per architecture: layer types, non-embedding parameters, forward MFLOP/token, and per task the held-out example accuracy (or validation perplexity for `lm`), mean ± std over seeds |
| `results.jsonl` | One line per finished run, with every metric. An interrupted sweep skips these runs when restarted; `--force` reruns all |
| `run_info.json` | The spec, command, git commit and versions |

## Spec format

```yaml
name: my-sweep                 # output directory name
base:                          # model_configs.yaml keys shared by every architecture
  hidden_size: 128
  num_attention_heads: 4
  intermediate_size: 512
budget:                        # optional
  param_match: full_attention  # every block gets the parameters of a full_attention block...
  # flop_match: full_attention #   ...or its forward FLOPs per token at seq_len
  # seq_len: 512
architectures:                 # name: layer types, or a dict with layer_types plus any settings
  attention: [full_attention, full_attention, full_attention, full_attention]
  based_hybrid: [conv, based, sliding_window_attention, based]
  small_state_mamba: {layer_types: [mamba2, mamba2, mamba2, mamba2], mamba_d_state: 16}
tasks:
  - task: mqar                 # mqar, selective_copy, induction, passkey (guac/core/diagnostics.py)
    args: {num_kv_pairs: 16, seq_len: 128, vocab_size: 256}
    steps: 3000                # plus optional batch_size, learning_rate, warmup_steps, eval_every, eval_examples
  - task: lm                   # language modeling: validation perplexity on packed text
    tokenizer: scratch/tokenizers/character/ascii
    datasets: [wikitext-103]
    seq_len: 256
    validation_docs: 200
    steps: 2000
    name: wikitext ppl         # optional column name
seeds: [0, 1, 2]
train: {batch_size: 64, learning_rate: 1e-3}   # defaults for every task
```

Budgets size each block's SwiGLU FFN (see `param_matched_ffn_hidden_sizes` and
`flop_matched_ffn_hidden_sizes` in `custom_architectures.py`), so architectures
with the same number of layers end up the same size. FLOPs are forward FLOPs
per token counted by `torch.utils.flop_counter` on the meta device, so they
include full attention's quadratic term at the given sequence length.

## Reading the results

Synthetic tasks isolate one ability each. Expect attention (full or windowed,
within its window) to solve recall and induction fastest, Mamba-2 and BASED to
degrade as MQAR's key-value pairs outgrow their fixed state, and conv layers
alone to fail anything beyond their kernel. Check that each task is learnable
at your settings (attention near 100%) before comparing others on it.

Mind what each mixer can see. BASED layers have no position information (as
in the paper), and causal linear attention without it is order-invariant over
its prefix: a stack of BASED layers, or BASED with a single attention layer,
can't bind a key to the value after it. The BASED paper pairs it with short
convolutions for that, hence `conv_based` in the example. Mamba-2 has its own
short convolution inside the mixer.
