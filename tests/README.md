# GUAC Tests

This directory contains all test scripts for the GUAC training system. Tests validate functionality across different components including path management, configuration systems, model architectures, and training pipelines.

## Test Organization

All test files follow the naming convention `test_*.py` and can be run individually or through the test runner.

### Available Tests

- **`test_path_manager.py`** - Tests the centralized path management system
- **`test_site_config.py`** - Tests site configuration loading and variable expansion
- **`test_sentencepiece.py`** - Tests SentencePiece tokenizer integration
- **`test_moh.py`** - Tests Mixture-of-Head attention implementation
- **`test_architecture_config.py`** - Tests runtime architecture configuration

## Running Tests

### Test Runner (Recommended)

Use the test runner to run all tests or specific tests:

```bash
# Run all tests
python tests/run_tests.py

# Run a specific test
python tests/run_tests.py --test path_manager

# List available tests
python tests/run_tests.py --list
```

### Individual Tests

Run individual test files directly from the project root:

```bash
# Run from project root directory
python tests/test_path_manager.py
python tests/test_site_config.py
python tests/test_sentencepiece.py
python tests/test_moh.py
python tests/test_architecture_config.py
```

## Test Descriptions

### Path Manager Tests (`test_path_manager.py`)

Tests the centralized path management system that ensures all outputs are organized under `SCRATCH_DIR`:

- Validates scratch directory configuration
- Tests dataset path generation (HuggingFace cache, generated datasets)
- Tests tokenized dataset cache paths by training stage
- Tests tokenizer output paths by type
- Tests model output paths by training stage
- Tests checkpoint and temporary directory paths
- Validates all paths are under the configured scratch directory

**Expected Output**: All paths correctly organized under `./scratch/` with proper hierarchy.

### Site Configuration Tests (`test_site_config.py`)

Tests the site configuration system:

- Tests YAML configuration loading
- Tests variable expansion (`${scratch_dir}`, `${base_dir}`)
- Tests fallback to example configuration
- Validates directory structure configuration

**Expected Output**: Configuration loaded successfully with proper variable expansion.

### SentencePiece Tests (`test_sentencepiece.py`)

Tests SentencePiece tokenizer integration:

- Tests tokenizer training with arithmetic data
- Tests special token integration
- Tests model saving and loading
- Validates tokenizer functionality

**Expected Output**: Tokenizer trained successfully with special tokens integrated.

### MoH Attention Tests (`test_moh.py`)

Tests the Mixture-of-Head attention implementation:

- Tests MoH attention layer creation
- Tests forward pass functionality
- Tests tensor shapes and dimensions
- Validates attention mechanism behavior

**Expected Output**: MoH attention layer works correctly with expected tensor shapes.

### Architecture Configuration Tests (`test_architecture_config.py`)

Tests runtime architecture configuration system:

- Tests architecture loading from configuration files
- Tests runtime architecture selection
- Tests environment variable overrides
- Tests command-line architecture selection

**Expected Output**: Architecture configuration system works with different backends.

## Test Environment

### Requirements

Tests require the same dependencies as the main GUAC system:

- PyTorch
- Transformers
- Datasets
- SentencePiece
- PyYAML

### Configuration

Tests use the main `site_config.yml` or fall back to `site_config.example.yml`. When run from the `tests/` directory, they create a local `tests/scratch/` directory for test outputs.

When run from the project root (recommended), they use the main scratch directory configured in `site_config.yml`.

### Cleanup

Test outputs are created in scratch directories and can be cleaned up using:

```bash
# Clean main scratch directory
make clean

# Clean test-specific outputs
rm -rf tests/scratch/
```

## Adding New Tests

To add new tests:

1. Create a new file following the naming convention: `test_<component>.py`
2. Add the standard header with project root path setup:

```python
#!/usr/bin/env python3
"""
Test script for <component>.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Your imports here
```

3. Implement test functions with descriptive names
4. Add a `main()` function that runs the tests and returns exit code
5. Update this README to document the new test

### Test Template

```python
#!/usr/bin/env python3
"""
Test script for new component.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_component_functionality():
    """Test specific component functionality."""
    print("Testing component...")

    # Your test logic here
    assert True, "Test condition"

    print("✅ Component test passed")
    return True

def main():
    """Main test function."""
    print("🧪 Testing New Component")
    print("=" * 50)

    try:
        success = test_component_functionality()
        if success:
            print("🎉 All tests passed!")
            return 0
        else:
            print("❌ Some tests failed!")
            return 1
    except Exception as e:
        print(f"💥 Test crashed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

## Continuous Integration

These tests are designed to be run in CI/CD pipelines. The test runner returns appropriate exit codes:

- `0` - All tests passed
- `1` - One or more tests failed or crashed

## Test Data

Tests that require data files should use minimal datasets or generate synthetic data to keep test execution fast and avoid large file dependencies.

For tests requiring model checkpoints or large datasets, consider:

- Using mock objects for unit tests
- Creating minimal test fixtures
- Downloading data only when needed
- Using environment variables to skip expensive tests in CI