"""
Tests for the shared-expert MoE FFN: vectorized dispatch against a per-token
loop, the Switch load-balancing loss, and how the model adds it to the loss.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch

from guac.core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM, SharedExpertMoE


def make_config(**overrides):
    kwargs = dict(vocab_size=50, hidden_size=32, num_attention_heads=4, num_kv_heads=2, num_hidden_layers=2,
                  layer_types=["conv", "shared_expert_moe"], intermediate_size=48, num_experts=4,
                  num_experts_per_token=2, num_shared_experts=1, moe_noise_std=0.0)
    kwargs.update(overrides)
    return MixedLayerConfig(**kwargs)


def loop_dispatch(moe, tokens, top_k_weights, top_k_indices):
    """The reference: every token through each of its experts, one at a time."""
    output = torch.zeros_like(tokens)
    for t in range(tokens.size(0)):
        for weight, expert_idx in zip(top_k_weights[t], top_k_indices[t]):
            output[t] += weight * moe.routing_experts[int(expert_idx)](tokens[t:t + 1])[0]
    return output


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_vectorized_dispatch_matches_per_token_loop(top_k):
    torch.manual_seed(0)
    moe = SharedExpertMoE(make_config(moe_top_k=top_k)).eval()
    tokens = torch.randn(23, 32)
    _, weights, indices = moe.route(tokens)

    with torch.no_grad():
        torch.testing.assert_close(moe.dispatch(tokens, weights, indices),
                                   loop_dispatch(moe, tokens, weights, indices))


def test_expert_with_no_tokens_is_skipped():
    torch.manual_seed(0)
    moe = SharedExpertMoE(make_config(moe_top_k=1)).eval()
    tokens = torch.randn(5, 32)
    indices = torch.zeros(5, 1, dtype=torch.long)  # everything to expert 0
    weights = torch.ones(5, 1)
    with torch.no_grad():
        torch.testing.assert_close(moe.dispatch(tokens, weights, indices), moe.routing_experts[0](tokens))


def test_load_balancing_loss_matches_switch_formula():
    torch.manual_seed(0)
    moe = SharedExpertMoE(make_config()).eval()
    probs, _, indices = moe.route(torch.randn(10, 32))

    fraction = torch.stack([(indices == i).sum() for i in range(4)]).float() / (10 * 2)
    expected = 0.01 * 4 * (fraction * probs.mean(0)).sum()
    torch.testing.assert_close(moe.load_balancing_loss(probs, indices), expected)


def test_uniform_router_gives_the_loss_weight():
    moe = SharedExpertMoE(make_config(moe_load_balancing_weight=0.5)).eval()
    with torch.no_grad():
        moe.router.weight.zero_()
    moe(torch.randn(2, 6, 32))
    torch.testing.assert_close(moe.aux_loss, torch.tensor(0.5))


def test_load_balancing_loss_trains_the_router():
    """The old loss was the constant weight * N, with no gradient."""
    torch.manual_seed(0)
    moe = SharedExpertMoE(make_config()).train()
    moe(torch.randn(2, 6, 32))
    (grad,) = torch.autograd.grad(moe.aux_loss, moe.router.weight)
    assert grad.abs().sum() > 0


def test_load_balancing_loss_ignores_padding():
    torch.manual_seed(0)
    moe = SharedExpertMoE(make_config()).eval()
    real = torch.randn(1, 5, 32)
    moe(real)
    expected = moe.aux_loss

    moe(torch.cat([real, torch.randn(1, 3, 32)], dim=1), torch.tensor([[1] * 5 + [0] * 3]))
    torch.testing.assert_close(moe.aux_loss, expected)


def test_model_adds_aux_loss_in_training_only():
    torch.manual_seed(0)
    model = MixedLayerForCausalLM(make_config())
    input_ids = torch.randint(0, 50, (2, 8))

    model.eval()
    eval_out = model(input_ids=input_ids, labels=input_ids)
    model.train()  # dropout is 0 and router noise is off, so only the aux loss differs
    train_out = model(input_ids=input_ids, labels=input_ids)

    assert eval_out.aux_loss is not None
    torch.testing.assert_close(train_out.loss, eval_out.loss + train_out.aux_loss)


def test_gradient_checkpointing_keeps_aux_loss_and_its_gradient():
    torch.manual_seed(0)
    input_ids = torch.randint(0, 50, (2, 8))
    results = []
    for checkpointing in (False, True):
        torch.manual_seed(0)
        model = MixedLayerForCausalLM(make_config()).train()
        model.gradient_checkpointing = checkpointing
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        router = model.layers[1].ffn.router.weight
        results.append((out.loss.detach(), out.aux_loss.detach(), router.grad.clone()))

    for plain, checkpointed in zip(*results):
        torch.testing.assert_close(checkpointed, plain)
