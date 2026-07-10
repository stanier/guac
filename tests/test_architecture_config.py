#!/usr/bin/env python3
"""
Test script demonstrating runtime architecture configuration.

Usage examples:
  python test_architecture_config.py                    # Uses default architecture
  python test_architecture_config.py --arch llama       # Uses Llama architecture
  GUAC_ARCHITECTURE=qwen2 python test_architecture_config.py  # Uses environment variable
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from scripts.training.pretrain.guac_arithmetic import (
    load_architecture_config,
    select_runtime_architecture,
    get_architecture_classes
)

def main():
    parser = argparse.ArgumentParser(description='Test architecture configuration system')
    parser.add_argument('--arch', type=str, help='Architecture to use (overrides config and env var)')
    parser.add_argument('--list', action='store_true', help='List available architectures')
    args = parser.parse_args()

    # Load architecture config
    arch_config = load_architecture_config()

    if args.list:
        print("Available architectures:")
        for arch_name, arch_info in arch_config['architectures'].items():
            print(f"  {arch_name}: {arch_info['description']}")
        return

    # Test different selection methods
    print("=== Architecture Selection Test ===")

    # Method 1: Default selection (from config)
    print("\n1. Default selection:")
    selected = select_runtime_architecture(arch_config)
    config_class, model_class = get_architecture_classes(arch_config, selected)
    print(f"   Selected: {selected}")
    print(f"   Config: {config_class.__name__}")
    print(f"   Model: {model_class.__name__}")

    # Method 2: Command line override
    if args.arch:
        print(f"\n2. Command line override (--arch {args.arch}):")
        try:
            config_class, model_class = get_architecture_classes(arch_config, args.arch)
            print(f"   Selected: {args.arch}")
            print(f"   Config: {config_class.__name__}")
            print(f"   Model: {model_class.__name__}")
        except Exception as e:
            print(f"   Error: {e}")

    # Method 3: Environment variable (if set)
    env_arch = os.environ.get('GUAC_ARCHITECTURE')
    if env_arch:
        print(f"\n3. Environment variable (GUAC_ARCHITECTURE={env_arch}):")
        try:
            config_class, model_class = get_architecture_classes(arch_config, env_arch)
            print(f"   Selected: {env_arch}")
            print(f"   Config: {config_class.__name__}")
            print(f"   Model: {model_class.__name__}")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n=== Architecture Defaults Test ===")
    for arch_name in arch_config['architectures'].keys():
        defaults = arch_config['architecture_defaults'].get(arch_name, {})
        print(f"{arch_name}: {defaults}")

if __name__ == "__main__":
    main()