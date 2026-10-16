#!/usr/bin/env python3
"""
Mamba-2 mixer built on the chunked structured state space duality (SSD) scan.

Follows Dao & Gu (2024), "Transformers are SSMs: Generalized Models and
Efficient Algorithms Through Structured State Space Duality" (arXiv:2405.21060).
Parameter names and the in_proj output layout match the reference
implementations (mamba_ssm's Mamba2 and transformers' Mamba2Mixer).

The scan is plain PyTorch, so it runs on any device: attention-like work
inside chunks of chunk_size tokens, plus a linear recurrence over chunk states
between chunks. It computes exactly the sequential recurrence for any chunk
size; the chunk size only trades memory against speed.

With scan_impl="mamba_ssm", CUDA inputs go through mamba_ssm's fused Triton
kernel (mamba_chunk_scan_combined) instead; CPU inputs still use the PyTorch
scan. The kernel computes the same function (test_mamba2.py compares the two
when a GPU and mamba_ssm are available).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SCAN_IMPLS = ("torch", "mamba_ssm")


def load_mamba_ssm_scan():
    """mamba_ssm's fused chunked scan, or an ImportError explaining how to get it."""
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError as e:
        raise ImportError("mamba_scan_impl='mamba_ssm' needs the mamba_ssm package (pip install mamba-ssm, "
                          "which needs a CUDA toolchain)") from e
    return mamba_chunk_scan_combined


def segment_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Pairwise segment sums over the last dimension.

    out[..., i, j] = x[..., j+1] + ... + x[..., i] for j <= i (0 on the
    diagonal) and -inf for j > i, so exp(out) is a causal decay matrix. Built
    from a masked cumulative sum rather than a difference of cumulative sums,
    which loses precision when the running totals get large.
    """
    length = x.size(-1)
    # x[..., i, j] = x[..., i]
    x = x[..., None].expand(*x.shape, length)
    strictly_lower = torch.tril(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=-1)
    sums = torch.cumsum(x.masked_fill(~strictly_lower, 0), dim=-2)
    lower = torch.tril(torch.ones(length, length, dtype=torch.bool, device=x.device))
    return sums.masked_fill(~lower, float('-inf'))


def ssd_chunked(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                chunk_size: int) -> torch.Tensor:
    """
    Chunked SSD scan. For each head, with h_0 = 0, computes

        h_t = exp(a_t) * h_{t-1} + outer(x_t, b_t)    (h_t is head_dim x state_size)
        y_t = h_t @ c_t

    Args:
        x: (batch, length, heads, head_dim) inputs, already scaled by dt
        a: (batch, length, heads) log decays, A * dt (<= 0)
        b: (batch, length, heads, state_size) input projections
        c: (batch, length, heads, state_size) output projections
        chunk_size: Tokens per chunk

    Returns:
        y: (batch, length, heads, head_dim)
    """
    batch, length, heads, head_dim = x.shape
    pad = (-length) % chunk_size
    if pad:
        # Zero inputs with zero log-decay after the end leave earlier outputs unchanged
        x, b, c = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (x, b, c))
        a = F.pad(a, (0, 0, 0, pad))
    num_chunks = x.size(1) // chunk_size

    # (batch, chunks, chunk_len, heads, dim)
    x, b, c = (t.reshape(batch, num_chunks, chunk_size, heads, -1) for t in (x, b, c))
    # (batch, heads, chunks, chunk_len)
    a = a.reshape(batch, num_chunks, chunk_size, heads).permute(0, 3, 1, 2)
    a_cumsum = torch.cumsum(a, dim=-1)

    # 1. Contributions from earlier tokens in the same chunk (causal, attention-like)
    scores = torch.einsum('bclhn,bcshn->bhcls', c, b) * torch.exp(segment_sum(a))
    y_within = torch.einsum('bhcls,bcshp->bclhp', scores, x)

    # 2. Each chunk's own contribution to the state at its end
    decay_to_end = torch.exp(a_cumsum[..., -1:] - a_cumsum)
    chunk_states = torch.einsum('bclhn,bhcl,bclhp->bchpn', b, decay_to_end, x)

    # 3. State entering each chunk: earlier chunks' states, decayed across the chunks in between
    chunk_states = torch.cat([torch.zeros_like(chunk_states[:, :1]), chunk_states], dim=1)
    chunk_decay = torch.exp(segment_sum(F.pad(a_cumsum[..., -1], (1, 0))))
    entering_states = torch.einsum('bhzc,bchpn->bzhpn', chunk_decay, chunk_states)[:, :-1]

    # 4. Contributions from the state entering the chunk
    y_carried = torch.einsum('bclhn,bchpn,bhcl->bclhp', c, entering_states, torch.exp(a_cumsum))

    y = (y_within + y_carried).reshape(batch, num_chunks * chunk_size, heads, head_dim)
    return y[:, :length]


class GatedRMSNorm(nn.Module):
    """RMSNorm of x * silu(gate), normalized within groups of group_size channels."""

    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_size = group_size
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float() * F.silu(gate.float())
        grouped = x.reshape(*x.shape[:-1], -1, self.group_size)
        grouped = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * grouped.reshape(x.shape).to(input_dtype)


class Mamba2Mixer(nn.Module):
    """
    Mamba-2 sequence mixer.

    in_proj produces a gate z, the SSM input x, the input/output projections
    B and C (shared by the heads within each of ngroups groups), and one dt per
    head. x, B and C pass through a short depthwise causal convolution. Each
    head has a scalar decay A = -exp(A_log), discretized as exp(A * dt) with
    dt = softplus(dt + dt_bias). The scan output plus the D skip connection is
    gated by silu(z), normalized, and projected back to hidden_size.
    """

    def __init__(self, hidden_size: int, d_state: int = 128, d_conv: int = 4, expand: int = 2,
                 headdim: int = 64, ngroups: int = 1, chunk_size: int = 64,
                 dt_min: float = 0.001, dt_max: float = 0.1, dt_init_floor: float = 1e-4,
                 A_init_range: tuple = (1.0, 16.0), norm_eps: float = 1e-5, scan_impl: str = "torch"):
        super().__init__()
        if scan_impl not in SCAN_IMPLS:
            raise ValueError(f"Unknown Mamba-2 scan_impl {scan_impl!r}; choose from {SCAN_IMPLS}")
        self.kernel_scan = load_mamba_ssm_scan() if scan_impl == "mamba_ssm" else None
        self.d_inner = int(expand * hidden_size)
        if self.d_inner % headdim:
            raise ValueError(f"Mamba-2 inner size {self.d_inner} (expand * hidden_size) must be divisible by headdim {headdim}")
        self.nheads = self.d_inner // headdim
        if self.nheads % ngroups:
            raise ValueError(f"Mamba-2 head count {self.nheads} must be divisible by ngroups {ngroups}")

        self.d_state = d_state
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.conv_dim = self.d_inner + 2 * ngroups * d_state

        # Output order: z, x, B, C, dt
        self.in_proj = nn.Linear(hidden_size, self.d_inner + self.conv_dim + self.nheads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=d_conv, groups=self.conv_dim,
                                padding=d_conv - 1, bias=True)

        # dt_bias is the inverse softplus of a dt drawn log-uniformly from [dt_min, dt_max]
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(*A_init_range)))
        self.D = nn.Parameter(torch.ones(self.nheads))

        self.norm = GatedRMSNorm(self.d_inner, group_size=self.d_inner // ngroups, eps=norm_eps)
        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        z, xbc, dt = torch.split(self.in_proj(hidden_states), [self.d_inner, self.conv_dim, self.nheads], dim=-1)

        xbc = F.silu(self.conv1d(xbc.transpose(1, 2))[..., :length].transpose(1, 2))
        x, b, c = torch.split(xbc, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        x = x.reshape(batch, length, self.nheads, self.headdim)
        b = b.reshape(batch, length, self.ngroups, self.d_state)
        c = c.reshape(batch, length, self.ngroups, self.d_state)

        if self.kernel_scan is not None and hidden_states.is_cuda:
            # The fused kernel takes raw dt, grouped B/C and D, and computes the same y as below
            y = self.kernel_scan(x, dt, -torch.exp(self.A_log.float()), b, c, self.chunk_size,
                                 D=self.D, dt_bias=self.dt_bias, dt_softplus=True)
            y = y.reshape(batch, length, self.d_inner)
            return self.out_proj(self.norm(y, z))

        # Head h reads group h // (nheads / ngroups)
        heads_per_group = self.nheads // self.ngroups
        b = b.repeat_interleave(heads_per_group, dim=2)
        c = c.repeat_interleave(heads_per_group, dim=2)

        # The scan's exponentials and cumulative sums need at least float32, even under mixed precision
        scan_dtype = torch.promote_types(x.dtype, torch.float32)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            x, b, c = x.to(scan_dtype), b.to(scan_dtype), c.to(scan_dtype)
            dt = F.softplus(dt.to(scan_dtype) + self.dt_bias.to(scan_dtype))
            A = -torch.exp(self.A_log.to(scan_dtype))
            y = ssd_chunked(x * dt[..., None], A * dt, b, c, self.chunk_size)
            y = y + x * self.D.to(scan_dtype)[:, None]

        y = y.reshape(batch, length, self.d_inner).to(z.dtype)
        return self.out_proj(self.norm(y, z))
