#!/usr/bin/env python3
"""
Synthetic diagnostic tasks that isolate what a layer type contributes.

Each task generates token-id sequences directly (no tokenizer) with labels
only at the answer positions (-100 elsewhere), so loss and accuracy measure
one ability:

- mqar: multi-query associative recall (Arora et al. 2023, "Zoology"). The
  context lists key-value pairs, and the model must recall the value for each
  queried key (answers are labels only, so they can't be deduced by elimination). Tests content-based lookup, where attention excels and
  fixed-state recurrences degrade as pairs outnumber their state.
- selective_copy: data tokens scattered among blank tokens must be repeated
  in order after a separator (Gu & Dao 2023, "Mamba"). Tests content-dependent
  filtering of what to remember.
- induction: a trigger token followed by token X appears once; at the end
  the trigger reappears and the model must predict X (Olsson et al. 2022).
  Tests copying from a matched earlier context.
- passkey: a short digit passkey hidden at a random depth in random filler
  must be repeated at the end (Mohtashami & Jaggi 2023). Tests retrieval
  across long distances.

Arithmetic, the remaining diagnostic, runs through midtrain.py and
eval_arithmetic.py since it uses text.

run_diagnostic trains a model from scratch on fresh batches and reports
accuracy on a fixed held-out set.
"""

import math
import time
from typing import Any, Callable, Dict, List, Optional

import torch

from .custom_architectures import no_weight_decay_parameter_names

PAD = 0
IGNORE = -100


def _rand(generator, low, high, size):
    return torch.randint(low, high, size, generator=generator)


def mqar(num_examples: int, seq_len: int = 128, vocab_size: int = 256, num_kv_pairs: int = 8,
         num_queries: Optional[int] = None, generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    """
    [k1 v1 ... kD vD  q1 _ q2 _ ... qQ _  pad...], where each _ is a blank
    token labeled with the value stored under the query key before it.

    As in Zoology, answers are only labels, never inputs: if they were fed
    back, a model could answer later queries by elimination (the values not
    yet given) rather than by recall. Keys and values come from disjoint
    halves of the vocabulary, each example uses distinct keys, and the
    queries are distinct keys in random order.
    """
    num_queries = num_queries or num_kv_pairs
    blank, first_key = 1, 2
    num_keys = (vocab_size - first_key) // 2
    if num_kv_pairs > num_keys or num_queries > num_kv_pairs:
        raise ValueError(f"mqar needs num_queries <= num_kv_pairs <= {num_keys} for vocab_size {vocab_size}")
    if 2 * (num_kv_pairs + num_queries) > seq_len:
        raise ValueError(f"mqar with {num_kv_pairs} pairs and {num_queries} queries needs seq_len >= "
                         f"{2 * (num_kv_pairs + num_queries)}")

    keys = torch.stack([torch.randperm(num_keys, generator=generator)[:num_kv_pairs]
                        for _ in range(num_examples)]) + first_key
    values = _rand(generator, first_key + num_keys, first_key + 2 * num_keys, (num_examples, num_kv_pairs))
    order = torch.stack([torch.randperm(num_kv_pairs, generator=generator)[:num_queries] for _ in range(num_examples)])
    query_keys, answers = keys.gather(1, order), values.gather(1, order)

    input_ids = torch.full((num_examples, seq_len), PAD, dtype=torch.long)
    labels = torch.full_like(input_ids, IGNORE)
    context = torch.stack([keys, values], dim=2).flatten(1)
    input_ids[:, :context.size(1)] = context
    start = context.size(1)
    input_ids[:, start:start + 2 * num_queries:2] = query_keys
    input_ids[:, start + 1:start + 2 * num_queries:2] = blank
    labels[:, start + 1:start + 2 * num_queries:2] = answers
    return {'input_ids': input_ids, 'labels': labels}


def selective_copy(num_examples: int, seq_len: int = 128, vocab_size: int = 16, num_tokens: int = 8,
                   generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    """
    [context of blanks with num_tokens data tokens at random positions, SEP, d1 ... dN],
    labels on d1 ... dN (the data tokens in order of appearance).
    """
    blank, separator, first_data = 1, 2, 3
    context_len = seq_len - num_tokens - 1
    if vocab_size <= first_data or num_tokens > context_len:
        raise ValueError(f"selective_copy needs vocab_size > {first_data} and seq_len > 2 * num_tokens")

    data = _rand(generator, first_data, vocab_size, (num_examples, num_tokens))
    positions = torch.stack([torch.randperm(context_len, generator=generator)[:num_tokens].sort().values
                             for _ in range(num_examples)])
    input_ids = torch.full((num_examples, seq_len), blank, dtype=torch.long)
    input_ids.scatter_(1, positions, data)
    input_ids[:, context_len] = separator
    input_ids[:, context_len + 1:] = data
    labels = torch.full_like(input_ids, IGNORE)
    labels[:, context_len + 1:] = data
    return {'input_ids': input_ids, 'labels': labels}


def induction(num_examples: int, seq_len: int = 128, vocab_size: int = 64,
              generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    """
    Random tokens with [TRIGGER, X] inserted once at a random position in the
    first half, ending with [TRIGGER, X]; the label is the final X.
    """
    trigger, first_token = 1, 2
    if vocab_size <= first_token + 1 or seq_len < 8:
        raise ValueError("induction needs vocab_size > 3 and seq_len >= 8")
    input_ids = _rand(generator, first_token, vocab_size, (num_examples, seq_len))
    targets = _rand(generator, first_token, vocab_size, (num_examples,))
    positions = _rand(generator, 0, seq_len // 2, (num_examples,))
    rows = torch.arange(num_examples)
    input_ids[rows, positions] = trigger
    input_ids[rows, positions + 1] = targets
    input_ids[:, -2] = trigger
    input_ids[:, -1] = targets
    labels = torch.full_like(input_ids, IGNORE)
    labels[:, -1] = targets
    return {'input_ids': input_ids, 'labels': labels}


def passkey(num_examples: int, seq_len: int = 256, vocab_size: int = 64, passkey_len: int = 5,
            generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    """
    Random filler with [KEY, d1 ... dK] at a random depth, ending with
    [QUERY, d1 ... dK]; labels on the final d1 ... dK. Digits are 10 dedicated tokens.
    """
    key_marker, query_marker, first_digit, first_filler = 1, 2, 3, 13
    block = passkey_len + 1
    if vocab_size <= first_filler or seq_len < 2 * block + 1:
        raise ValueError(f"passkey needs vocab_size > {first_filler} and seq_len > {2 * block}")
    digits = _rand(generator, first_digit, first_digit + 10, (num_examples, passkey_len))
    input_ids = _rand(generator, first_filler, vocab_size, (num_examples, seq_len))
    depths = _rand(generator, 0, seq_len - 2 * block + 1, (num_examples,))
    offsets = torch.arange(block)
    rows = torch.arange(num_examples)[:, None]
    input_ids[rows, depths[:, None] + offsets] = torch.cat([torch.full((num_examples, 1), key_marker), digits], dim=1)
    input_ids[:, -block] = query_marker
    input_ids[:, -passkey_len:] = digits
    labels = torch.full_like(input_ids, IGNORE)
    labels[:, -passkey_len:] = digits
    return {'input_ids': input_ids, 'labels': labels}


TASKS: Dict[str, Callable[..., Dict[str, torch.Tensor]]] = {
    'mqar': mqar,
    'selective_copy': selective_copy,
    'induction': induction,
    'passkey': passkey,
}

# Default vocabulary size per task (the model's vocab_size)
DEFAULT_VOCAB = {'mqar': 256, 'selective_copy': 16, 'induction': 64, 'passkey': 64}


def make_batch(task: str, num_examples: int, seed: int, **task_kwargs) -> Dict[str, torch.Tensor]:
    if task not in TASKS:
        raise ValueError(f"Unknown diagnostic task {task!r}; choose from {list(TASKS)}")
    generator = torch.Generator().manual_seed(seed)
    task_kwargs.setdefault('vocab_size', DEFAULT_VOCAB[task])
    return TASKS[task](num_examples, generator=generator, **task_kwargs)


def answer_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Accuracy of greedy next-token predictions at the labeled positions:
    per answer token, and per example (every answer token right).
    """
    predictions = logits[:, :-1].argmax(-1)
    targets = labels[:, 1:]
    answered = targets != IGNORE
    correct = (predictions == targets) & answered
    return {
        'token_accuracy': (correct.sum() / answered.sum()).item(),
        'example_accuracy': (correct.sum(1) == answered.sum(1)).float().mean().item(),
    }


@torch.no_grad()
def evaluate(model, data: Dict[str, torch.Tensor], batch_size: int = 256) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    totals = {'loss': 0.0, 'token_accuracy': 0.0, 'example_accuracy': 0.0}
    num_examples = data['input_ids'].size(0)
    num_answers = (data['labels'][:, 1:] != IGNORE).sum().item()
    for start in range(0, num_examples, batch_size):
        input_ids = data['input_ids'][start:start + batch_size].to(device)
        labels = data['labels'][start:start + batch_size].to(device)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        batch_answers = (labels[:, 1:] != IGNORE).sum().item()
        accuracy = answer_accuracy(out.logits, labels)
        totals['loss'] += out.loss.item() * batch_answers
        totals['token_accuracy'] += accuracy['token_accuracy'] * batch_answers
        totals['example_accuracy'] += accuracy['example_accuracy'] * input_ids.size(0)
    if was_training:
        model.train()
    return {'loss': totals['loss'] / num_answers, 'token_accuracy': totals['token_accuracy'] / num_answers,
            'example_accuracy': totals['example_accuracy'] / num_examples}


def parameter_groups(model, weight_decay: float) -> List[Dict[str, Any]]:
    """
    AdamW parameter groups: weight_decay for the weights, none for biases, norm
    gains and Mamba-2's A_log, D and dt_bias (see no_weight_decay_parameter_names).
    """
    exempt = no_weight_decay_parameter_names(model)
    decayed, not_decayed = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (not_decayed if name in exempt else decayed).append(param)
    return [{'params': decayed, 'weight_decay': weight_decay}, {'params': not_decayed, 'weight_decay': 0.0}]


def train_and_evaluate(model, next_batch: Callable[[int], Dict[str, torch.Tensor]],
                       evaluate_fn: Callable[[Any], Dict[str, float]], steps: int, learning_rate: float = 1e-3,
                       weight_decay: float = 0.1, warmup_steps: int = 100, eval_every: int = 0,
                       device: Optional[str] = None, bf16: bool = False,
                       log: Callable[[str], None] = print) -> Dict[str, Any]:
    """
    The training loop shared by the diagnostics and sweeps: AdamW (weight
    decay as in parameter_groups), linear warmup then cosine decay, gradient
    clipping at 1.0. next_batch(step)
    returns input_ids and labels; evaluate_fn(model) returns metrics, run
    every eval_every steps (if set) and at the end. bf16 runs forward passes
    under bfloat16 autocast (weights and optimizer state stay float32).

    Returns the final metrics, intermediate evaluations, the smoothed training
    loss and the wall time. Raises FloatingPointError if the loss diverges.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).train()
    optimizer = torch.optim.AdamW(parameter_groups(model, weight_decay), lr=learning_rate, betas=(0.9, 0.98))

    def lr_scale(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    device_type = torch.device(device).type

    def autocast():
        return torch.autocast(device_type, dtype=torch.bfloat16, enabled=bf16)

    history, running_loss = [], None
    started = time.time()
    for step in range(steps):
        batch = next_batch(step)
        with autocast():
            out = model(input_ids=batch['input_ids'].to(device), labels=batch['labels'].to(device))
        loss = out.loss.item()
        if not math.isfinite(loss):
            raise FloatingPointError(f"Loss became {loss} at step {step}")
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss = loss if running_loss is None else 0.98 * running_loss + 0.02 * loss
        if eval_every and (step + 1) % eval_every == 0 and step + 1 < steps:
            with autocast():
                metrics = evaluate_fn(model)
            history.append({'step': step + 1, **metrics})
            log(f"  step {step + 1:6d}  train loss {running_loss:.4f}  " +
                "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

    with autocast():
        final = evaluate_fn(model)
    return {
        'steps': steps,
        'train_loss': running_loss,
        'eval': final,
        'history': history,
        'seconds': time.time() - started,
        'parameters': sum(p.numel() for p in model.parameters()),
    }


def run_diagnostic(model, task: str, task_kwargs: Optional[Dict[str, Any]] = None, steps: int = 2000,
                   batch_size: int = 64, eval_examples: int = 1000, seed: int = 0,
                   **train_kwargs) -> Dict[str, Any]:
    """
    Train model on fresh batches of a task for `steps` steps and evaluate on a
    fixed held-out set drawn with a different seed (see train_and_evaluate for
    the optimizer and the other keyword arguments).

    Returns the final evaluation (loss on answer tokens, token and example
    accuracy), the best example accuracy over all evaluations, the training
    loss and the wall time.
    """
    task_kwargs = dict(task_kwargs or {})
    eval_data = make_batch(task, eval_examples, seed=10_000 + seed, **task_kwargs)
    results = train_and_evaluate(
        model, lambda step: make_batch(task, batch_size, seed=seed * 1_000_003 + step, **task_kwargs),
        lambda m: evaluate(m, eval_data), steps, **train_kwargs,
    )
    accuracies = [entry['example_accuracy'] for entry in results['history']] + [results['eval']['example_accuracy']]
    return {'task': task, 'task_kwargs': task_kwargs, **results, 'best_example_accuracy': max(accuracies)}


@torch.no_grad()
def language_model_loss(model, blocks: torch.Tensor, batch_size: int = 32) -> Dict[str, float]:
    """Mean next-token loss and perplexity over packed token blocks (every block the same length)."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total = 0.0
    for start in range(0, blocks.size(0), batch_size):
        batch = blocks[start:start + batch_size].to(device)
        total += model(input_ids=batch, labels=batch, use_cache=False).loss.item() * batch.size(0)
    if was_training:
        model.train()
    loss = total / blocks.size(0)
    return {'loss': loss, 'perplexity': math.exp(min(loss, 50.0))}


def run_language_modeling(model, train_blocks: torch.Tensor, validation_blocks: torch.Tensor, steps: int = 1000,
                          batch_size: int = 32, seed: int = 0, **train_kwargs) -> Dict[str, Any]:
    """
    Train on random batches of packed token blocks (num_blocks, seq_len) and
    report validation loss and perplexity (see train_and_evaluate).
    """
    generator = torch.Generator().manual_seed(seed)

    def next_batch(step):
        batch = train_blocks[torch.randint(0, train_blocks.size(0), (batch_size,), generator=generator)]
        return {'input_ids': batch, 'labels': batch}

    results = train_and_evaluate(model, next_batch, lambda m: language_model_loss(m, validation_blocks),
                                 steps, **train_kwargs)
    return {'task': 'lm', **results}
