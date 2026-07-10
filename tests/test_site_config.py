#!/usr/bin/env python3
"""
Test script for site configuration system.

Usage:
  python test_site_config.py
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from scripts.training.pretrain.guac_arithmetic import load_site_config

def main():
    print("=== Site Configuration Test ===")

    try:
        site_config = load_site_config()
        print("✓ Site configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load site configuration: {e}")
        return

    # Test directory configuration
    print("\n--- Directory Configuration ---")
    dirs = site_config.get('directories', {})

    if dirs:
        print(f"Scratch directory: {dirs.get('scratch_dir', 'NOT SET')}")

        # Test dataset directories
        datasets = dirs.get('datasets', {})
        print(f"Dataset simple: {datasets.get('arithmetic_simple', 'NOT SET')}")
        print(f"Dataset complex: {datasets.get('arithmetic_complex', 'NOT SET')}")

        # Test tokenizer directories
        tokenizers = dirs.get('tokenizers', {})
        print(f"Tokenizer dir: {tokenizers.get('char_arithmetic', 'NOT SET')}")

        # Test checkpoint directories
        checkpoints = dirs.get('checkpoints', {})
        print(f"Checkpoint dir: {checkpoints.get('pretrain_simple', 'NOT SET')}")

        # Test tokenized dataset directories
        tokenized = dirs.get('tokenized_datasets', {})
        print(f"Tokenized simple: {tokenized.get('simple', 'NOT SET')}")
        print(f"Tokenized complex: {tokenized.get('complex', 'NOT SET')}")
    else:
        print("✗ No directory configuration found")

    # Test environment configuration
    print("\n--- Environment Configuration ---")
    env = site_config.get('environment', {})
    if env:
        print(f"Deployment mode: {env.get('deployment_mode', 'NOT SET')}")
        print(f"Log level: {env.get('log_level', 'NOT SET')}")
    else:
        print("✗ No environment configuration found")

    # Test path expansion
    print("\n--- Path Expansion Test ---")
    test_paths = [
        "./scratch/test",
        "${scratch_dir}/test",
        "${base_dir}/models" if 'base_dir' in str(dirs) else "No base_dir found"
    ]

    for path in test_paths:
        print(f"  {path} → (expansion not tested in isolation)")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()