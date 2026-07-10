#!/usr/bin/env python3
"""
Centralized Path Management for GUAC Training System

This module provides a unified interface for managing all filesystem paths
used in the GUAC training pipeline, ensuring everything is organized under
the configured SCRATCH_DIR from site_config.yml.

All output paths (datasets, tokenizers, models, caches) are derived from
the site configuration to maintain consistency and organization.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import warnings


class PathManager:
    """Centralized path manager for all GUAC training outputs."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize path manager with site configuration.

        Args:
            config_path: Optional path to site_config.yml. If None, uses default location.
        """
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.config_path = config_path or (self.project_root / "site_config.yml")
        self._config = None
        self._directories = None

    def _load_config(self) -> Dict[str, Any]:
        """Load and cache site configuration."""
        if self._config is not None:
            return self._config

        # Try to load site_config.yml, fallback to example
        config_file = self.config_path
        if not config_file.exists():
            example_file = self.project_root / "site_config.example.yml"
            if example_file.exists():
                warnings.warn(f"site_config.yml not found, using {example_file}")
                config_file = example_file
            else:
                raise FileNotFoundError(f"Neither {self.config_path} nor {example_file} found")

        with open(config_file, 'r') as f:
            self._config = yaml.safe_load(f)

        return self._config

    def _get_directories(self) -> Dict[str, Any]:
        """Get and expand directory configuration."""
        if self._directories is not None:
            return self._directories

        config = self._load_config()
        dirs = config.get('directories', {})

        # Expand path variables recursively
        self._directories = self._expand_path_variables(dirs)
        return self._directories

    def _expand_path_variables(self, data: Any, vars_dict: Optional[Dict[str, str]] = None) -> Any:
        """Recursively expand ${var} references in configuration."""
        if vars_dict is None:
            vars_dict = {}

        if isinstance(data, dict):
            # First pass: collect base variables
            if 'scratch_dir' in data:
                vars_dict['scratch_dir'] = data['scratch_dir']

            # Process each section
            result = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    # For subsections, collect base_dir first
                    if 'base_dir' in value:
                        expanded_base = self._expand_string(value['base_dir'], vars_dict)
                        value = value.copy()
                        value['base_dir'] = expanded_base
                        vars_dict['base_dir'] = expanded_base

                    result[key] = self._expand_path_variables(value, vars_dict.copy())
                else:
                    result[key] = self._expand_string(value, vars_dict)
            return result
        elif isinstance(data, list):
            return [self._expand_path_variables(item, vars_dict) for item in data]
        else:
            return self._expand_string(data, vars_dict)

    def _expand_string(self, text: Any, vars_dict: Dict[str, str]) -> Any:
        """Expand ${var} references in a string."""
        if not isinstance(text, str) or "${" not in text:
            return text

        result = text
        for var_name, var_value in vars_dict.items():
            if isinstance(var_value, str):
                result = result.replace(f"${{{var_name}}}", var_value)
        return result

    def get_scratch_dir(self) -> Path:
        """Get the main scratch directory."""
        dirs = self._get_directories()
        scratch_dir = dirs.get('scratch_dir', './scratch')
        return Path(scratch_dir).resolve()

    # === Dataset Paths ===

    def get_dataset_base_dir(self) -> Path:
        """Get base directory for all datasets."""
        dirs = self._get_directories()
        path = dirs.get('datasets', {}).get('base_dir', './scratch/datasets')
        return Path(path).resolve()

    def get_huggingface_cache_dir(self) -> Path:
        """Get directory for HuggingFace dataset cache."""
        dirs = self._get_directories()
        path = dirs.get('datasets', {}).get('huggingface_cache', './scratch/datasets/huggingface_cache')
        result = Path(path).resolve()
        result.mkdir(parents=True, exist_ok=True)
        return result

    def get_generated_dataset_dir(self) -> Path:
        """Get directory for generated datasets (arithmetic, etc.)."""
        dirs = self._get_directories()
        path = dirs.get('datasets', {}).get('generated', './scratch/datasets/generated')
        result = Path(path).resolve()
        result.mkdir(parents=True, exist_ok=True)
        return result

    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get path for a specific dataset."""
        dirs = self._get_directories()
        datasets = dirs.get('datasets', {})

        # Check for specific dataset path in config
        if dataset_name in datasets:
            path = datasets[dataset_name]
        else:
            # Default to generated datasets directory
            base_dir = self.get_generated_dataset_dir()
            path = base_dir / dataset_name

        result = Path(path).resolve()
        result.mkdir(parents=True, exist_ok=True)
        return result

    # === Tokenized Dataset Paths ===

    def get_tokenized_dataset_base_dir(self) -> Path:
        """Get base directory for tokenized datasets."""
        dirs = self._get_directories()
        path = dirs.get('tokenized_datasets', {}).get('base_dir', './scratch/tokenized_datasets')
        return Path(path).resolve()

    def get_tokenized_cache_dir(self) -> Path:
        """Get directory for tokenized dataset cache."""
        dirs = self._get_directories()
        path = dirs.get('tokenized_datasets', {}).get('cache_dir', './scratch/tokenized_datasets/cache')
        result = Path(path).resolve()
        result.mkdir(parents=True, exist_ok=True)
        return result

    def get_tokenized_dataset_path(self, stage: str, cache_key: str) -> Path:
        """Get path for tokenized dataset by training stage.

        Args:
            stage: Training stage ('pretraining', 'midtraining', 'posttraining')
            cache_key: Unique identifier for the tokenized dataset

        Returns:
            Path to tokenized dataset directory
        """
        dirs = self._get_directories()
        tokenized_dirs = dirs.get('tokenized_datasets', {})

        if stage in tokenized_dirs:
            base_path = tokenized_dirs[stage]
        else:
            base_path = self.get_tokenized_cache_dir()

        result = Path(base_path) / f"tokenized_cache_{cache_key}"
        result = result.resolve()
        result.mkdir(parents=True, exist_ok=True)
        return result

    # === Tokenizer Paths ===

    def get_tokenizer_base_dir(self) -> Path:
        """Get base directory for trained tokenizers."""
        dirs = self._get_directories()
        path = dirs.get('tokenizers', {}).get('base_dir', './scratch/tokenizers')
        return Path(path).resolve()

    def get_tokenizer_output_path(self, tokenizer_type: str = "sentencepiece", name: Optional[str] = None) -> Path:
        """Get output path for trained tokenizer.

        Args:
            tokenizer_type: Type of tokenizer ('sentencepiece', 'character', 'custom')
            name: Optional custom name for the tokenizer

        Returns:
            Path to tokenizer output directory
        """
        dirs = self._get_directories()
        tokenizer_dirs = dirs.get('tokenizers', {})

        if tokenizer_type in tokenizer_dirs:
            base_path = tokenizer_dirs[tokenizer_type]
        else:
            base_path = self.get_tokenizer_base_dir() / tokenizer_type

        if name:
            result = Path(base_path) / name
        else:
            result = Path(base_path) / "default"

        result = result.resolve()
        result.mkdir(parents=True, exist_ok=True)
        return result

    # === Model Paths ===

    def get_model_base_dir(self) -> Path:
        """Get base directory for trained models."""
        dirs = self._get_directories()
        path = dirs.get('models', {}).get('base_dir', './scratch/models')
        return Path(path).resolve()

    def get_model_output_path(self, stage: str, config_name: str, task: Optional[str] = None, mode: Optional[str] = None) -> Path:
        """Get output path for trained model.

        Args:
            stage: Training stage ('pretraining', 'midtraining', 'posttraining')
            config_name: Model configuration name (nano, pico, etc.)
            task: Optional task name for mid-training
            mode: Optional mode for post-training

        Returns:
            Path to model output directory
        """
        dirs = self._get_directories()
        model_dirs = dirs.get('models', {})

        # Get stage-specific base directory
        if stage in model_dirs:
            stage_base = model_dirs[stage]
        else:
            stage_base = self.get_model_base_dir() / stage

        # Build path components
        path_components = [config_name]
        if task:
            path_components.append(task)
        if mode:
            path_components.append(mode)

        result = Path(stage_base) / "_".join(path_components)
        result = result.resolve()
        result.mkdir(parents=True, exist_ok=True)
        return result

    def get_checkpoint_dir(self, stage: str, config_name: str, task: Optional[str] = None) -> Path:
        """Get directory for model checkpoints during training.

        Args:
            stage: Training stage
            config_name: Model configuration name
            task: Optional task name

        Returns:
            Path to checkpoint directory
        """
        dirs = self._get_directories()
        checkpoint_base = dirs.get('models', {}).get('checkpoints', './scratch/models/checkpoints')

        path_components = [stage, config_name]
        if task:
            path_components.append(task)

        result = Path(checkpoint_base) / "_".join(path_components)
        result = result.resolve()
        result.mkdir(parents=True, exist_ok=True)
        return result

    # === Temporary Paths ===

    def get_temp_dir(self, purpose: str = "general") -> Path:
        """Get temporary directory for processing.

        Args:
            purpose: Purpose of temp directory ('tokenizer_training', 'data_processing', etc.)

        Returns:
            Path to temporary directory
        """
        dirs = self._get_directories()
        temp_dirs = dirs.get('temp', {})

        if purpose in temp_dirs:
            path = temp_dirs[purpose]
        else:
            base_temp = temp_dirs.get('base_dir', './scratch/temp')
            path = Path(base_temp) / purpose

        result = Path(path).resolve()
        result.mkdir(parents=True, exist_ok=True)
        return result

    # === Utility Methods ===

    def create_cache_key(self, prefix: str, **kwargs) -> str:
        """Create a consistent cache key for tokenized datasets.

        Args:
            prefix: Cache key prefix
            **kwargs: Key-value pairs to include in cache key

        Returns:
            Formatted cache key string
        """
        parts = [prefix]
        for key, value in sorted(kwargs.items()):
            if value is not None:
                parts.append(f"{key}{value}")
        return "_".join(parts)

    def ensure_dir_exists(self, path: Union[str, Path]) -> Path:
        """Ensure directory exists and return Path object.

        Args:
            path: Directory path to create

        Returns:
            Path object for the directory
        """
        result = Path(path).resolve()
        result.mkdir(parents=True, exist_ok=True)
        return result

    def get_legacy_path(self, category: str, name: str) -> Optional[Path]:
        """Get legacy path for backward compatibility.

        Args:
            category: Path category ('datasets', 'tokenizers', 'models', etc.)
            name: Specific path name

        Returns:
            Legacy path if found, None otherwise
        """
        dirs = self._get_directories()
        category_dirs = dirs.get(category, {})

        if name in category_dirs:
            return Path(category_dirs[name]).resolve()
        return None


# Global path manager instance
_path_manager = None


def get_path_manager() -> PathManager:
    """Get global path manager instance."""
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager()
    return _path_manager


# Convenience functions for common operations
def get_scratch_dir() -> Path:
    """Get the main scratch directory."""
    return get_path_manager().get_scratch_dir()


def get_dataset_path(dataset_name: str) -> Path:
    """Get path for a dataset."""
    return get_path_manager().get_dataset_path(dataset_name)


def get_tokenized_dataset_path(stage: str, cache_key: str) -> Path:
    """Get path for tokenized dataset."""
    return get_path_manager().get_tokenized_dataset_path(stage, cache_key)


def get_tokenizer_output_path(tokenizer_type: str = "sentencepiece", name: Optional[str] = None) -> Path:
    """Get output path for tokenizer."""
    return get_path_manager().get_tokenizer_output_path(tokenizer_type, name)


def get_model_output_path(stage: str, config_name: str, task: Optional[str] = None, mode: Optional[str] = None) -> Path:
    """Get output path for model."""
    return get_path_manager().get_model_output_path(stage, config_name, task, mode)


def get_huggingface_cache_dir() -> Path:
    """Get HuggingFace dataset cache directory."""
    return get_path_manager().get_huggingface_cache_dir()


def create_cache_key(prefix: str, **kwargs) -> str:
    """Create cache key for tokenized datasets."""
    return get_path_manager().create_cache_key(prefix, **kwargs)