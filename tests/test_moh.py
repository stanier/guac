#!/usr/bin/env python3
"""
Test script for MoH (Mixture-of-Head) attention implementation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from scripts.training.core.custom_architectures import MixedLayerConfig, MixtureOfHeadAttention

def test_moh_attention():
    """Test the MoH attention implementation."""

    # Create a test configuration
    config = MixedLayerConfig(
        vocab_size=1000,
        hidden_size=384,
        num_attention_heads=8,
        num_kv_heads=2,
        moh_top_k=4,  # Use 50% of heads
        moh_temperature=1.0,
        moh_noise_std=0.1
    )

    # Create MoH attention layer
    moh_layer = MixtureOfHeadAttention(config)

    # Test input
    batch_size = 2
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    print("Testing MoH Attention Implementation")
    print("=" * 50)
    print(f"Config: {config.hidden_size}d, {config.num_attention_heads} heads, {config.num_kv_heads} kv_heads")
    print(f"MoH top_k: {config.moh_top_k} ({config.moh_top_k/config.num_attention_heads:.1%} of heads)")
    print(f"Input shape: {hidden_states.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in moh_layer.parameters())
    attention_params = (
        moh_layer.q_proj.weight.numel() +
        moh_layer.k_proj.weight.numel() +
        moh_layer.v_proj.weight.numel() +
        moh_layer.o_proj.weight.numel()
    )
    router_params = sum(p.numel() for p in moh_layer.head_router.parameters())

    print(f"\nParameter Count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Attention parameters: {attention_params:,}")
    print(f"  Router parameters: {router_params:,}")
    print(f"  Router overhead: {router_params/total_params:.1%}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    moh_layer.eval()  # Test in eval mode first

    with torch.no_grad():
        output = moh_layer(hidden_states)
        print(f"Output shape: {output.shape}")
        print(f"Output mean: {output.mean().item():.4f}")
        print(f"Output std: {output.std().item():.4f}")

    # Test training mode (with noise)
    print(f"\nTesting training mode...")
    moh_layer.train()

    output_train = moh_layer(hidden_states)
    print(f"Training output shape: {output_train.shape}")
    print(f"Training output mean: {output_train.mean().item():.4f}")
    print(f"Training output std: {output_train.std().item():.4f}")

    # Test gradients
    print(f"\nTesting gradients...")
    loss = output_train.sum()
    loss.backward()

    grad_norms = {}
    for name, param in moh_layer.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    print("Gradient norms:")
    for name, norm in grad_norms.items():
        print(f"  {name}: {norm:.6f}")

    print("\n✅ MoH attention test completed successfully!")

def compare_efficiency():
    """Compare parameter efficiency between standard attention and MoH."""

    configs = [
        ("Standard GQA", {"moh_top_k": 8}),  # Use all heads (no MoH)
        ("MoH 75%", {"moh_top_k": 6}),      # Use 75% of heads
        ("MoH 50%", {"moh_top_k": 4}),      # Use 50% of heads
        ("MoH 25%", {"moh_top_k": 2}),      # Use 25% of heads
    ]

    print("\nEfficiency Comparison")
    print("=" * 50)

    base_config = MixedLayerConfig(
        hidden_size=384,
        num_attention_heads=8,
        num_kv_heads=2,
    )

    for name, moh_config in configs:
        config = MixedLayerConfig(
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            num_kv_heads=base_config.num_kv_heads,
            **moh_config
        )

        layer = MixtureOfHeadAttention(config)

        # Calculate effective computation (only for selected heads)
        effective_heads = config.moh_top_k
        total_heads = config.num_attention_heads
        efficiency_ratio = effective_heads / total_heads

        total_params = sum(p.numel() for p in layer.parameters())
        router_params = sum(p.numel() for p in layer.head_router.parameters())

        print(f"{name}:")
        print(f"  Heads used: {effective_heads}/{total_heads} ({efficiency_ratio:.1%})")
        print(f"  Total params: {total_params:,}")
        print(f"  Router overhead: {router_params:,} ({router_params/total_params:.1%})")
        print(f"  Effective computation: {efficiency_ratio:.1%} of full attention")
        print()

if __name__ == "__main__":
    # Run tests
    test_moh_attention()
    compare_efficiency()