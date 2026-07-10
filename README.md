# GUAC: General Understanding and Computational Architecture

GUAC is a research framework for training and experimenting with heterogeneous transformer architectures, featuring mixed layer types and novel prompt formatting designed for complex sequence-to-sequence tasks.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- transformers 4.40+
- CUDA-capable GPU (recommended)

### Installation
```bash
git clone <repository>
cd guac
pip install -r requirements.txt

# Copy and configure site settings
cp site_config.example.yml site_config.yml
# Edit site_config.yml with your paths and preferences
```

### Basic Training
```bash
# Recommended: Start with stable configuration
make stable

# Or use scripts directly:
# Quick debug with minimal model
python scripts/training/pretrain/guac_general.py --model-config iota0

# Stable training with mixed architecture
python scripts/training/pretrain/guac_general.py --model-config iota

# Train arithmetic models
python scripts/training/pretrain/guac_arithmetic.py --config nano
```

## 🏗️ Architecture Overview

GUAC introduces **Mixed Layer Architectures** - models that use different computational patterns within the same network:

```
Layer 1: MLP-only        (simplest computation)
Layer 2: Convolutional   (local pattern processing)
Layer 3: Convolutional   (more local processing)
Layer 4: Mamba-2         (state space modeling)
Layer 5: BASED           (efficient attention)
Layer 6: Mamba-2         (more state space)
Layer 7: Flash Attention (full attention)
Layer 8: MLP-only        (final processing)
```

### Supported Layer Types
- **Standard Attention**: Multi-head and grouped query attention
- **Flash Attention**: Memory-efficient attention computation
- **Convolutional**: 1D convolution for sequence modeling
- **Mamba-2**: Selective state space models
- **BASED**: Bidirectional attention with feature maps
- **MLP-only**: Feed-forward only layers

## 📁 Project Structure

```
guac/
├── README.md                     # This file
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md          # Architecture details
│   ├── TRAINING.md              # Training guide
│   ├── CONTRIBUTING.md          # Contributor guide
│   └── API.md                   # API reference
├── scripts/
│   ├── training/                # Training scripts
│   │   ├── pretrain/           # Pretraining workflows
│   │   ├── custom_architectures.py # Mixed architecture implementations
│   │   ├── callbacks.py        # Training callbacks
│   │   └── prompt_templates.py # Prompt system
│   └── datasets/               # Dataset generation and processing
├── templates/                   # Jinja2 prompt templates
├── guac/                       # Core library code
├── model_configs.yaml          # Model size configurations
├── architecture_configs.yaml   # Architecture definitions
└── site_config.yml            # Environment configuration
```

## 🎯 Key Features

### 1. **Mixed Layer Architectures**
- Combine different computational patterns in a single model
- Layer types progress from simple to complex computations
- Enables studying different inductive biases within one model

### 2. **Novel Prompt System**
- Custom token design with FLAG and SERIAL tokens
- Multi-sequence boundary handling
- Inference trigger and cessation markers
- Jinja2-based templating system

### 3. **Flexible Configuration**
- Runtime architecture selection
- Scalable model size progression (iota → nano → micro → ... → chungus)
- Environment-agnostic site configuration

### 4. **Advanced Training Features**
- Gradient checkpointing for memory efficiency
- Dataset caching for faster iteration
- Live inference during training
- Streaming dataset support

## 📖 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)**: Detailed architecture explanations
- **[Training Guide](docs/TRAINING.md)**: Training workflows and best practices
- **[API Reference](docs/API.md)**: Code API documentation
- **[Contributing](docs/CONTRIBUTING.md)**: How to contribute to GUAC

## 🧪 Research Goals

GUAC enables research into:

1. **Heterogeneous Architectures**: How do different layer types interact?
2. **Computational Complexity**: Optimal ordering of computation patterns
3. **Sequence-to-Sequence Learning**: Multi-modal and multi-sequence tasks
4. **Efficient Attention**: Comparing attention mechanisms within one model
5. **Mathematical Reasoning**: Arithmetic and symbolic computation tasks

## 📊 Model Configurations

GUAC provides pre-configured model sizes:

| Config    | Hidden Size | Layers | Attention Heads | Parameters |
|-----------|-------------|--------|-----------------|------------|
| `iota`    | 256         | 16     | 4               | ~2M        |
| `nano`    | 384         | 18     | 8               | ~8M        |
| `micro`   | 768         | 22     | 6               | ~25M       |
| `milli`   | 896         | 21     | 16              | ~40M       |
| `mixed`   | 512         | 8      | 8               | ~48M       |
| `base`    | 2560        | 36     | 20              | ~1.3B      |
| `chungus` | 8192        | 80     | 64              | ~65B       |

## 🔄 Workflows

### Training a Mixed Architecture Model
```bash
# Train with cached datasets (default)
python scripts/training/pretrain/guac_general.py --model-config mixed

# Force re-tokenization
python scripts/training/pretrain/guac_general.py --force-retokenize

# Use specific datasets
python scripts/training/pretrain/guac_general.py --datasets wikitext-103,fineweb-edu

# Disable template processing
python scripts/training/pretrain/guac_general.py --no-templates
```

### Generating Arithmetic Datasets
```bash
python scripts/datasets/gen_arithm.py --num-samples 10000 --complexity simple
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on:
- Code style and standards
- Adding new layer types
- Submitting experiments
- Documentation improvements

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

GUAC builds upon work from:
- **Mamba-2**: Selective state space models
- **BASED**: Bidirectional attention mechanisms
- **Flash Attention**: Memory-efficient attention
- **Transformers**: HuggingFace ecosystem

## 📞 Contact

[Add contact information for the research team]