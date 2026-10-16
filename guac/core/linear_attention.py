#!/usr/bin/env python3
"""
Causal linear attention with the 2nd-order Taylor feature map used by BASED.

Follows Arora et al. (2024), "Simple linear attention language models balance
the recall-throughput tradeoff" (arXiv:2402.18668). The feature map matches the
reference implementation's TaylorExp, so feature_map(q) . feature_map(k)
approximates exp(q . k / sqrt(d)).

The attention is plain PyTorch and runs on any device. It works in chunks:
quadratic within each chunk of chunk_size tokens, plus a running state
sum_j phi(k_j) v_j^T carried between chunks, so time and memory are linear in
the sequence length. The chunk size only trades memory against speed.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .mamba2 import scan_chunk_states


def taylor_feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    phi(x) = [1, x / d^(1/4), vec(x x^T) / sqrt(2d)] over the last dimension (size d).

    phi(q) . phi(k) = 1 + s + s^2 / 2 with s = q . k / sqrt(d), the 2nd-order
    Taylor expansion of exp(s). That is at least 1/2 for any q and k, so the
    attention normalizer is always positive. The output has 1 + d + d^2 features.
    """
    d = x.size(-1)
    second_order = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / math.sqrt(2 * d)
    return torch.cat([torch.ones_like(x[..., :1]), x / d ** 0.25, second_order], dim=-1)


def causal_linear_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int = 64,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, eps: float = 1e-12,
    document_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Normalized causal linear attention over feature-mapped queries and keys:

        y_i = sum_{j<=i} (q_i . k_j) v_j / sum_{j<=i} (q_i . k_j)

    Args:
        q, k: (batch, heads, length, features), with q_i . k_j >= 0 (as for taylor_feature_map)
        v: (batch, heads, length, dim)
        chunk_size: Tokens per chunk
        initial_state: Optional (kv, k_sum) of shapes (batch, heads, features, dim) and
            (batch, heads, features) summarizing earlier tokens, for continuing a sequence
        eps: Added to the normalizer. A query with no keys at all (every key masked
            by zeroing it) gets output 0
        document_ids: Optional (batch, length) non-decreasing ids; the sums then run
            only over j in i's document (j <= i with the same id), so packed documents
            don't see each other. initial_state belongs to the first document

    Returns:
        y: (batch, heads, length, dim)
        final_state: (kv, k_sum) after the last token (the last document's alone)
    """
    batch, heads, length, _ = q.shape
    pad = (-length) % chunk_size
    if pad:
        # Zero keys and values after the end add nothing to the sums
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        if document_ids is not None:
            document_ids = torch.cat([document_ids, document_ids[:, -1:].expand(-1, pad)], dim=1)
    num_chunks = q.size(2) // chunk_size

    # (batch, heads, chunks, chunk_len, dim)
    q, k, v = (t.reshape(batch, heads, num_chunks, chunk_size, -1) for t in (q, k, v))

    # 1. Earlier tokens in the same chunk (causal, attention-like)
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).tril()
    if document_ids is not None:
        # (batch, 1, chunks, chunk_len)
        document_ids = document_ids.reshape(batch, 1, num_chunks, chunk_size)
        causal = causal & (document_ids[..., :, None] == document_ids[..., None, :])
    scores = (q @ k.transpose(-1, -2)).masked_fill(~causal, 0)
    numerator = scores @ v
    denominator = scores.sum(-1)

    # 2. Tokens in earlier chunks, through the state entering each chunk
    if document_ids is not None:
        # A chunk passes on only its last document's keys
        chunk_end_ids = document_ids[..., -1]
        in_last_document = (document_ids == chunk_end_ids[..., None])[..., None].to(k.dtype)
        chunk_kv = (k * in_last_document).transpose(-1, -2) @ v
        chunk_k_sum = (k * in_last_document).sum(-2)
    else:
        chunk_kv = k.transpose(-1, -2) @ v   # (batch, heads, chunks, features, dim)
        chunk_k_sum = k.sum(-2)              # (batch, heads, chunks, features)
    if initial_state is None:
        initial_kv = torch.zeros_like(chunk_kv[:, :, 0])
        initial_k_sum = torch.zeros_like(chunk_k_sum[:, :, 0])
    else:
        initial_kv, initial_k_sum = (s.to(chunk_kv.dtype) for s in initial_state)
    kv = torch.cat([initial_kv[:, :, None], chunk_kv], dim=2)
    k_sum = torch.cat([initial_k_sum[:, :, None], chunk_k_sum], dim=2)
    if document_ids is None:
        kv, k_sum = kv.cumsum(dim=2), k_sum.cumsum(dim=2)
    else:
        # Entry z of the running sums (the state after chunk z - 1, z = 0 the
        # initial state) sums only its own document's chunks. Mamba-2's scan over
        # chunk states with zero decay does that by masking (exact, and linear in
        # the number of chunks), where subtracting cumulative sums would leave
        # rounding error from earlier documents
        state_ids = torch.cat([document_ids[..., :1, 0], chunk_end_ids], dim=-1)  # (batch, 1, chunks + 1)
        no_decay = kv.new_zeros(batch, heads, num_chunks + 1)

        def running_sum(t):  # (batch, heads, chunks + 1, features, dim)
            return scan_chunk_states(t.transpose(1, 2), no_decay, chunk_size, state_ids[:, 0]).transpose(1, 2)

        kv, k_sum = running_sum(kv), running_sum(k_sum[..., None])[..., 0]
    carried_numerator = q @ kv[:, :, :-1]
    carried_denominator = (q * k_sum[:, :, :-1, None]).sum(-1)
    if document_ids is not None:
        # Each token uses the entering state only if it is in the state's document
        uses_state = (document_ids == state_ids[..., :-1, None]).to(q.dtype)
        carried_numerator = carried_numerator * uses_state[..., None]
        carried_denominator = carried_denominator * uses_state
    numerator = numerator + carried_numerator
    denominator = denominator + carried_denominator

    # Drop the padding before dividing, since padded rows have a zero normalizer
    numerator = numerator.reshape(batch, heads, num_chunks * chunk_size, -1)[:, :, :length]
    denominator = denominator.reshape(batch, heads, num_chunks * chunk_size)[:, :, :length]
    y = numerator / (denominator[..., None] + eps)
    return y, (kv[:, :, -1], k_sum[:, :, -1])
