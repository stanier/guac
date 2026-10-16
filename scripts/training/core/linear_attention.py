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

    Returns:
        y: (batch, heads, length, dim)
        final_state: (kv, k_sum) after the last token
    """
    batch, heads, length, _ = q.shape
    pad = (-length) % chunk_size
    if pad:
        # Zero keys and values after the end add nothing to the sums
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    num_chunks = q.size(2) // chunk_size

    # (batch, heads, chunks, chunk_len, dim)
    q, k, v = (t.reshape(batch, heads, num_chunks, chunk_size, -1) for t in (q, k, v))

    # 1. Earlier tokens in the same chunk (causal, attention-like)
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).tril()
    scores = (q @ k.transpose(-1, -2)).masked_fill(~causal, 0)
    numerator = scores @ v
    denominator = scores.sum(-1)

    # 2. Tokens in earlier chunks, through the state entering each chunk
    chunk_kv = k.transpose(-1, -2) @ v       # (batch, heads, chunks, features, dim)
    chunk_k_sum = k.sum(-2)                  # (batch, heads, chunks, features)
    if initial_state is None:
        initial_kv = torch.zeros_like(chunk_kv[:, :, 0])
        initial_k_sum = torch.zeros_like(chunk_k_sum[:, :, 0])
    else:
        initial_kv, initial_k_sum = (s.to(chunk_kv.dtype) for s in initial_state)
    kv = torch.cat([initial_kv[:, :, None], chunk_kv], dim=2).cumsum(dim=2)
    k_sum = torch.cat([initial_k_sum[:, :, None], chunk_k_sum], dim=2).cumsum(dim=2)
    numerator = numerator + q @ kv[:, :, :-1]
    denominator = denominator + (q * k_sum[:, :, :-1, None]).sum(-1)

    # Drop the padding before dividing, since padded rows have a zero normalizer
    numerator = numerator.reshape(batch, heads, num_chunks * chunk_size, -1)[:, :, :length]
    denominator = denominator.reshape(batch, heads, num_chunks * chunk_size)[:, :, :length]
    y = numerator / (denominator[..., None] + eps)
    return y, (kv[:, :, -1], k_sum[:, :, -1])
