#!/usr/bin/env python3
"""
Test script for the centralized path manager.

This script validates that all paths are correctly derived from the
SCRATCH_DIR configuration and that the path manager works as expected.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from guac.core.path_manager import get_path_manager


def test_path_manager():
    """Test the path manager functionality."""
    print("🧪 Testing Path Manager Implementation")
    print("=" * 50)

    # Initialize path manager
    pm = get_path_manager()

    # Test scratch directory
    scratch_dir = pm.get_scratch_dir()
    print(f"📁 Scratch Directory: {scratch_dir}")

    # Test dataset paths
    print("\n📚 Dataset Paths:")
    hf_cache = pm.get_huggingface_cache_dir()
    print(f"  HuggingFace Cache: {hf_cache}")

    generated_dir = pm.get_generated_dataset_dir()
    print(f"  Generated Datasets: {generated_dir}")

    custom_dataset = pm.get_dataset_path("custom_dataset")
    print(f"  Custom Dataset: {custom_dataset}")

    # Test tokenized dataset paths
    print("\n💾 Tokenized Dataset Paths:")
    for stage in ['pretraining', 'midtraining', 'posttraining']:
        cache_key = f"test_vocab1000_len512"
        path = pm.get_tokenized_dataset_path(stage, cache_key)
        print(f"  {stage.title()}: {path}")

    # Test tokenizer paths
    print("\n🔤 Tokenizer Paths:")
    for tokenizer_type in ['sentencepiece', 'character', 'custom']:
        path = pm.get_tokenizer_output_path(tokenizer_type, "test_tokenizer")
        print(f"  {tokenizer_type.title()}: {path}")

    # Test model paths
    print("\n🤖 Model Paths:")
    for stage in ['pretraining', 'midtraining', 'posttraining']:
        if stage == 'midtraining':
            path = pm.get_model_output_path(stage, "guac-3b", task="arithmetic")
        elif stage == 'posttraining':
            path = pm.get_model_output_path(stage, "guac-3b", task="arithmetic", mode="instruction")
        else:
            path = pm.get_model_output_path(stage, "guac-3b")
        print(f"  {stage.title()}: {path}")

    # Test checkpoint paths
    print("\n📋 Checkpoint Paths:")
    checkpoint_dir = pm.get_checkpoint_dir("pretraining", "guac-3b")
    print(f"  Checkpoints: {checkpoint_dir}")

    # Test temporary paths
    print("\n🕐 Temporary Paths:")
    temp_tokenizer = pm.get_temp_dir("tokenizer_training")
    print(f"  Tokenizer Training: {temp_tokenizer}")

    temp_data = pm.get_temp_dir("data_processing")
    print(f"  Data Processing: {temp_data}")

    # Test cache key creation
    print("\n🔑 Cache Key Generation:")
    cache_key = pm.create_cache_key("pretrain", vocab=32000, len=512, datasets="wikitext")
    print(f"  Sample Cache Key: {cache_key}")

    # Test that all paths are under scratch_dir
    print("\n✅ Path Validation:")
    all_paths = [
        hf_cache, generated_dir, custom_dataset,
        pm.get_tokenized_dataset_path("pretraining", "test"),
        pm.get_tokenizer_output_path("sentencepiece"),
        pm.get_model_output_path("pretraining", "guac-3b"),
        checkpoint_dir, temp_tokenizer, temp_data
    ]

    all_under_scratch = True
    for path in all_paths:
        try:
            path.relative_to(scratch_dir)
            print(f"  ✓ {path.relative_to(scratch_dir)}")
        except ValueError:
            print(f"  ✗ {path} is NOT under scratch directory!")
            all_under_scratch = False

    if all_under_scratch:
        print("\n🎉 All paths are correctly organized under SCRATCH_DIR!")
    else:
        print("\n❌ Some paths are not under SCRATCH_DIR - configuration issue!")

    assert all_under_scratch, "Some paths are not under SCRATCH_DIR"


if __name__ == "__main__":
    test_path_manager()

def test_getters_do_not_create_directories(tmp_path):
    """Directories appear only when something is written there, so existence means content."""
    import yaml
    from guac.core.path_manager import PathManager

    example = yaml.safe_load((project_root / "site_config.example.yml").read_text())
    example['directories']['scratch_dir'] = str(tmp_path / "scratch")
    config_path = tmp_path / "site_config.yml"
    config_path.write_text(yaml.safe_dump(example))

    pm = PathManager(config_path)
    paths = [pm.get_tokenized_dataset_path("pretraining", "key"), pm.get_tokenizer_output_path("sentencepiece", "t"),
             pm.get_model_output_path("pretraining", "guac-3b"), pm.get_checkpoint_dir("pretraining", "guac-3b"),
             pm.get_dataset_path("d"), pm.get_huggingface_cache_dir()]
    assert all(str(path).startswith(str(tmp_path)) for path in paths)
    assert not (tmp_path / "scratch").exists()
    assert pm.get_temp_dir("work").is_dir()  # temp directories are created on request
