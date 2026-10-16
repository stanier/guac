# Contributing to Guac

Welcome to Guac! This guide will help you contribute effectively to the project.

## 🎯 Project Overview

Guac is a research framework for mixed layer transformer architectures. Our primary goals are:

1. **Novel Architecture Research**: Exploring heterogeneous transformer layers
2. **Computational Efficiency**: Studying different computational patterns within single models
3. **Sequence-to-Sequence Tasks**: Advanced prompt templating for complex tasks
4. **Mathematical Reasoning**: Specialized arithmetic and symbolic computation capabilities

## 🚀 Getting Started

### Development Environment

1. **Clone and Setup**:
```bash
git clone <repository>
cd guac
pip install -r requirements.txt
cp site_config.example.yml site_config.yml
# Configure site_config.yml for your environment
```

2. **Test Installation**:
```bash
# Quick test with minimal model
python scripts/training/pretrain/guac_general.py --model-config iota0
```

### Code Structure

```
guac/
├── scripts/training/           # Training workflows
│   ├── pretrain/              # Pretraining scripts
│   ├── custom_architectures.py # Mixed layer implementations
│   ├── callbacks.py           # Training callbacks
│   └── prompt_templates.py    # Templating system
├── templates/                 # Jinja2 prompt templates
├── docs/                      # Documentation
├── model_configs.yaml         # Model configurations
├── architecture_configs.yaml  # Architecture definitions
└── site_config.yml           # Environment settings
```

## 🧱 Adding New Layer Types

### Implementation Steps

1. **Create the mixer**: the part of the block that moves information between
positions. The block (`MixedTransformerLayer`) adds the norms, residuals and FFN,
so the mixer returns only its own output:
```python
class MyMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        # Must be causal: position t may only read positions <= t
        return self.proj(my_sequence_mixing(hidden_states))
```

2. **Register it** in `custom_architectures.py`:
```python
SUPPORTED_LAYER_TYPES = (..., "my_mixer")
MIXERS = {..., "my_mixer": MyMixer}
```

3. **Add config parameters** to `MixedLayerConfig` (and pass them through
`build_model_config` in `pretrain.py` if presets should set them).

4. **Run the tests.** The causality, padding, save/load and block-structure tests
parametrize over `SUPPORTED_LAYER_TYPES`, so they cover the new type
automatically. Don't paper over numerical problems by zeroing NaNs or clamping
outputs: a diverging run should fail visibly.

### Design Guidelines

1. **Computational Complexity**: Document O(·) complexity for sequence length L and hidden dimension d
2. **Memory Usage**: Specify memory requirements and scaling behavior
3. **Gradient Flow**: Ensure stable gradients through residual connections
4. **Numerical Stability**: Add bounds checking and NaN handling
5. **Documentation**: Include usage examples and parameter explanations

## 📊 Model Configuration

### Adding New Model Sizes

Add to `model_configs.yaml`:

```yaml
model_training_configs:
  my_size:
    step_val: 100              # Evaluation frequency
    hidden_size: 768           # Model dimension
    num_hidden_layers: 12      # Number of layers
    num_attention_heads: 12    # Attention heads
    num_kv_heads: 4           # Key-value heads (GQA)
    learning_rate: 1e-4       # Base learning rate
    warmup_ratio: 0.1         # Warmup fraction
    weight_decay: 0.01        # L2 regularization
    pdbs_train: 8             # Per-device batch size
    pdbs_eval: 8              # Evaluation batch size
    grad_accum_steps: 2       # Gradient accumulation
    num_train_epochs: 3       # Training epochs

    # Layer type progression (optional)
    layer_types: [
      "mlp_only",
      "conv",
      "mamba2",
      "flash_attention"
    ]
```

### Configuration Best Practices

1. **Memory Scaling**: Consider GPU memory constraints for your target size
2. **Layer Progression**: Order from simple to complex computations
3. **Attention Heads**: Ensure `num_kv_heads <= num_attention_heads`
4. **Learning Rate**: Scale appropriately with model size

## 🎨 Prompt Templates

### Creating Custom Templates

1. **Template Structure**:
```jinja2
{# templates/my_task.jinja #}
{{ FLAG_TOKEN }}TASK:{{ SERIAL_TOKEN }}{{ task_description }}{{ END_TASK_TOKEN }}
{{ FLAG_TOKEN }}INPUT:{{ SERIAL_TOKEN }}{{ input_text }}{{ END_INPUT_TOKEN }}
{{ FLAG_TOKEN }}OUTPUT:{{ SERIAL_TOKEN }}{{ output_text }}{{ END_OUTPUT_TOKEN }}
```

2. **Register Template**:
```python
# In prompt_templates.py
TASK_TEMPLATES = {
    "my_task": "templates/my_task.jinja",
    # ... other templates
}
```

3. **Usage**:
```bash
python scripts/training/pretrain/guac_general.py \
  --use-templates \
  --template-file my_task.jinja \
  --template-task "My custom task description"
```

### Template Guidelines

1. **Token Usage**: Always use FLAG and SERIAL tokens for boundaries
2. **Clear Structure**: Separate input/output sections clearly
3. **Flexibility**: Support variable content lengths
4. **Documentation**: Document expected input/output formats

## 🧪 Experiments and Research

### Running Experiments

1. **Systematic Naming**:
```bash
export EXP_NAME="mixed_arch_mamba2_$(date +%Y%m%d)"
python scripts/training/pretrain/guac_general.py \
  --model-config mixed \
  --output-dir ./experiments/$EXP_NAME
```

2. **Configuration Tracking**:
- Commit code changes before experiments
- Document hyperparameter rationale
- Save experiment configurations

3. **Reproducibility**:
```bash
# Set seeds for reproducibility
export PYTHONHASHSEED=42
python scripts/training/pretrain/guac_general.py --seed 42
```

### Evaluation Guidelines

1. **Baseline Comparisons**: Always compare against standard transformer
2. **Ablation Studies**: Test individual layer type contributions
3. **Scaling Analysis**: Test across multiple model sizes
4. **Task-Specific Metrics**: Use appropriate evaluation for your use case

## 🐛 Debugging and Testing

### Common Issues

**1. Numerical Instability**
```bash
# Symptoms: loss=0.0, grad_norm=NaN
# Debug with minimal model
python scripts/training/pretrain/guac_general.py --model-config iota0

# Solutions:
# - Reduce learning rate
# - Check layer implementations for NaN sources
# - Enable gradient clipping (already enabled by default)
```

**2. Memory Issues**
```bash
# Solutions:
# - Reduce batch size: --model-config with lower pdbs_train
# - Enable gradient checkpointing (enabled by default)
# - Use fewer attention layers
```

**3. Layer Compatibility**
```python
# Ensure all layers output same shape
assert output.shape == input.shape, f"Shape mismatch: {output.shape} != {input.shape}"
```

### Testing Checklist

Before submitting changes:

- [ ] Code runs without errors on `iota0` configuration
- [ ] No NaN gradients or infinite losses
- [ ] Memory usage is reasonable
- [ ] Gradient checkpointing works
- [ ] Generation methods function correctly
- [ ] Documentation is updated

## 📋 Submission Guidelines

### Pull Request Process

1. **Branch Naming**:
```bash
git checkout -b feature/layer-type-name
git checkout -b fix/numerical-stability
git checkout -b docs/architecture-update
```

2. **Commit Messages**:
```
Add Mamba-2 layer implementation

- Implement selective state space model
- Add numerical stability measures
- Update architecture configuration
- Add performance documentation
```

3. **PR Description Template**:
```markdown
## Summary
Brief description of changes

## Changes
- [ ] New layer implementation
- [ ] Configuration updates
- [ ] Documentation updates
- [ ] Bug fixes

## Testing
- [ ] Tested on iota0 configuration
- [ ] No numerical stability issues
- [ ] Memory usage reasonable

## Performance
| Layer Type | Complexity | Memory | Notes |
|------------|------------|---------|-------|
| New Layer  | O(...)     | O(...)  | ...   |
```

### Code Style

1. **Python Style**:
- Follow PEP 8
- Use type hints where helpful
- Document complex functions
- Avoid deep nesting

2. **Configuration Style**:
- Use descriptive parameter names
- Add comments for non-obvious settings
- Maintain backwards compatibility

3. **Documentation Style**:
- Use clear, technical language
- Include code examples
- Provide performance characteristics
- Add troubleshooting sections

## 🤝 Community

### Getting Help

1. **Documentation**: Check existing docs first
2. **Issues**: Search existing issues before creating new ones
3. **Discussions**: Use for questions and ideas
4. **Code Review**: Request reviews for significant changes

### Research Collaboration

We encourage:
- **Novel Architecture Ideas**: New layer types and combinations
- **Efficiency Improvements**: Better computational patterns
- **Task Applications**: New use cases for mixed architectures
- **Evaluation Methods**: Better ways to assess mixed models

## 📚 Resources

### Essential Reading

1. **Mamba-2**: Structured State Space Models
2. **BASED**: Bidirectional Attention with Sparse Efficient Decoding
3. **Flash Attention**: Memory-efficient attention computation
4. **Grouped Query Attention**: Efficient attention with fewer KV heads

### Implementation References

- **HuggingFace Transformers**: Model implementation patterns
- **PyTorch**: Tensor operations and autograd
- **Jinja2**: Template syntax and features

This guide should help you contribute effectively to Guac. For specific questions, please open an issue or start a discussion!