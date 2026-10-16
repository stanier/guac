"""
Tests for the Mamba-2 mixer: the chunked SSD scan against the plain sequential
recurrence, and the full mixer against transformers' reference Mamba2Mixer.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn.functional as F

from scripts.training.core.custom_architectures import MixedLayerConfig
from scripts.training.core.mamba2 import Mamba2Mixer, ssd_chunked


def sequential_scan(x, a, b, c):
    """The recurrence ssd_chunked computes, one token at a time."""
    batch, length, heads, head_dim = x.shape
    state = x.new_zeros(batch, heads, head_dim, b.size(-1))
    outputs = []
    for t in range(length):
        state = torch.exp(a[:, t])[..., None, None] * state + x[:, t, :, :, None] * b[:, t, :, None, :]
        outputs.append(torch.einsum('bhpn,bhn->bhp', state, c[:, t]))
    return torch.stack(outputs, dim=1)


def random_scan_inputs(length, heads=3, head_dim=4, state_size=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)
    x = rand(2, length, heads, head_dim)
    a = -torch.rand(2, length, heads, generator=generator, dtype=torch.float64)  # log decays <= 0
    return x, a, rand(2, length, heads, state_size), rand(2, length, heads, state_size)


@pytest.mark.parametrize("length,chunk_size", [(16, 4), (17, 4), (5, 8), (32, 32), (1, 4)])
def test_chunked_scan_matches_sequential_recurrence(length, chunk_size):
    x, a, b, c = random_scan_inputs(length)
    torch.testing.assert_close(ssd_chunked(x, a, b, c, chunk_size), sequential_scan(x, a, b, c))


def test_chunked_scan_gradients_match_sequential_recurrence():
    inputs = [t.requires_grad_() for t in random_scan_inputs(13)]
    chunked = torch.autograd.grad(ssd_chunked(*inputs, chunk_size=4).square().sum(), inputs)
    sequential = torch.autograd.grad(sequential_scan(*inputs).square().sum(), inputs)
    for got, expected in zip(chunked, sequential):
        torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("ngroups", [1, 2])
def test_mixer_output_does_not_depend_on_chunk_size(ngroups):
    torch.manual_seed(0)
    mixer = Mamba2Mixer(hidden_size=32, d_state=8, headdim=16, ngroups=ngroups).eval()
    hidden_states = torch.randn(2, 37, 32)

    outputs = []
    with torch.no_grad():
        for chunk_size in (4, 16, 64):
            mixer.chunk_size = chunk_size
            outputs.append(mixer(hidden_states))
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0])


def test_mixer_matches_transformers_reference():
    """Same weights through transformers' Mamba2Mixer (CPU path) give the same output."""
    from transformers import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer as ReferenceMixer

    torch.manual_seed(0)
    # Differences in transformers 4.46.3 to work around:
    # - it clamps dt at time_step_min, which the original Mamba-2 does not, so set that to 0
    # - it assigns heads to groups round-robin, so compare with a single group
    # - its final partial chunk is slightly off from the sequential recurrence (up to ~7e-5),
    #   so use a sequence length that fills its chunks exactly
    reference = ReferenceMixer(Mamba2Config(
        hidden_size=32, num_heads=4, head_dim=16, state_size=8, expand=2, n_groups=1,
        conv_kernel=4, chunk_size=8, time_step_min=0.0, use_bias=False, use_conv_bias=True,
        layer_norm_epsilon=1e-5,
    ), layer_idx=0).eval()
    ours = Mamba2Mixer(hidden_size=32, d_state=8, d_conv=4, expand=2, headdim=16, ngroups=1, chunk_size=16).eval()
    ours.load_state_dict(reference.state_dict())  # strict: parameter names and shapes must match

    hidden_states = torch.randn(2, 24, 32)
    with torch.no_grad():
        torch.testing.assert_close(ours(hidden_states), reference.torch_forward(hidden_states), rtol=1e-4, atol=1e-5)


def test_initialization_ranges():
    mixer = Mamba2Mixer(hidden_size=256, headdim=16)
    dt = F.softplus(mixer.dt_bias)
    A = -torch.exp(mixer.A_log)
    assert mixer.nheads == 32
    assert dt.min() >= 1e-4 and dt.max() <= 0.1 + 1e-6
    assert A.min() >= -16 - 1e-4 and A.max() <= -1 + 1e-6


def test_mixer_is_finite_under_bfloat16_autocast():
    mixer = Mamba2Mixer(hidden_size=32, d_state=8, headdim=16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = mixer(torch.randn(2, 20, 32))
    assert torch.isfinite(output.float()).all()


def test_config_rejects_inner_size_not_divisible_by_headdim():
    with pytest.raises(ValueError, match="divisible by mamba_headdim"):
        MixedLayerConfig(hidden_size=48, num_attention_heads=4, num_hidden_layers=1,
                         layer_types=["mamba2"], mamba_headdim=64)


def test_unknown_scan_impl_is_rejected():
    with pytest.raises(ValueError, match="scan_impl"):
        Mamba2Mixer(64, headdim=16, scan_impl="cuda")


def mamba_ssm_available():
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(mamba_ssm_available(), reason="mamba_ssm is installed")
def test_kernel_scan_without_mamba_ssm_explains_how_to_get_it():
    with pytest.raises(ImportError, match="pip install mamba-ssm"):
        Mamba2Mixer(64, headdim=16, scan_impl="mamba_ssm")


@pytest.mark.skipif(not (torch.cuda.is_available() and mamba_ssm_available()),
                    reason="needs a CUDA GPU and the mamba_ssm package")
@pytest.mark.parametrize("ngroups, length", [(1, 256), (2, 100)])
def test_kernel_scan_matches_torch_scan(ngroups, length):
    """mamba_ssm's fused kernel and the PyTorch scan compute the same mixer (values and gradients)."""
    torch.manual_seed(0)
    reference = Mamba2Mixer(256, d_state=64, headdim=32, ngroups=ngroups, chunk_size=64).cuda()
    kernel = Mamba2Mixer(256, d_state=64, headdim=32, ngroups=ngroups, chunk_size=64, scan_impl="mamba_ssm").cuda()
    kernel.load_state_dict(reference.state_dict())

    x = torch.randn(2, length, 256, device="cuda")
    outputs, grads = [], []
    for mixer in (reference, kernel):
        x_in = x.clone().requires_grad_()
        y = mixer(x_in)
        y.pow(2).mean().backward()
        outputs.append(y.detach())
        grads.append((x_in.grad, mixer.A_log.grad, mixer.in_proj.weight.grad))

    torch.testing.assert_close(outputs[1], outputs[0], atol=2e-3, rtol=2e-3)
    for g_kernel, g_reference in zip(grads[1], grads[0]):
        torch.testing.assert_close(g_kernel, g_reference, atol=2e-3, rtol=2e-3)
