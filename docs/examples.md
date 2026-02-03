# Examples

This page provides links to all example scripts included with PyDML.

## Quick Start Examples

### Test Installation

Verify PyDML is installed correctly:

```bash
python examples/test_installation.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/test_installation.py)

### Quick Start

5-minute introduction to PyDML:

```bash
python examples/quick_start.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/quick_start.py)

### Mini Demo

Lightweight demonstration without downloading data:

```bash
python examples/mini_demo.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/mini_demo.py)

## Core Examples

### Complete Demo

Full PyDML workflow with CIFAR-10:

```bash
python examples/complete_demo.py
```

Features:

- Data loading
- Model creation
- DML training
- Evaluation
- Checkpointing

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/complete_demo.py)

### Distillation Demo

Knowledge distillation from teacher to student:

```bash
python examples/distillation_demo.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/distillation_demo.py)

### Co-Distillation Demo

Combining teacher guidance with peer learning:

```bash
python examples/co_distillation_demo.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/co_distillation_demo.py)

### Feature DML Demo

Learning from intermediate representations:

```bash
python examples/feature_dml_demo.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/feature_dml_demo.py)

## Advanced Examples

### Advanced Training

Production-ready training script:

```bash
python examples/advanced_training.py
```

Features:

- LR scheduling with warmup
- Checkpoint management
- Training monitoring
- Overfitting detection
- TensorBoard logging

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/advanced_training.py)

### Advanced Usage

Comprehensive feature demonstration:

```bash
python examples/advanced_usage.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/advanced_usage.py)

### Attention Transfer Demo

Attention transfer between networks:

```bash
python examples/attention_transfer_demo.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/attention_transfer_demo.py)

### Curriculum Learning Demo

Curriculum-based training strategies:

```bash
python examples/curriculum_demo.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/curriculum_demo.py)

### Peer Selection Demo

Dynamic peer selection strategies:

```bash
python examples/peer_selection_demo.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/peer_selection_demo.py)

### Temperature Scaling Demo

Dynamic temperature adjustment:

```bash
python examples/temperature_scaling_demo.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/temperature_scaling_demo.py)

## Validation Examples

### Input Validation Demo

Demonstrates comprehensive input validation:

```bash
python examples/input_validation_demo.py
```

Shows clear error messages and validation features.

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/input_validation_demo.py)

### Validation Demo

Production-safe validation (no assert statements):

```bash
python examples/validation_demo.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/validation_demo.py)

## Benchmarks

### CIFAR-100 Benchmark

Comprehensive CIFAR-100 evaluation:

```bash
python examples/cifar100_benchmark.py
```

Compares:

- Independent training
- DML with 2 networks
- DML with 3+ networks
- Different architectures

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/cifar100_benchmark.py)

## Visualization

### Visualization Demo

Generate analysis visualizations:

```bash
python examples/visualization_demo.py
```

Creates:

- Training curves
- Model comparison plots
- Loss landscapes
- Ensemble diversity
- Confidence distributions
- Peer influence graphs

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/visualization_demo.py)

## Testing Examples

### Test Lightweight

Quick functionality test:

```bash
python examples/test_lightweight.py
```

[View Source](https://github.com/VARUN3WARE/dml-py/blob/main/examples/test_lightweight.py)

## Running Examples

All examples can be run directly:

```bash
cd dml-py
python examples/<example_name>.py
```

Some examples require downloading CIFAR data (automatic on first run).

## Example Template

Create your own training script:

```python
#!/usr/bin/env python
"""
My PyDML Training Script
"""

import torch
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32
from pydml.utils import get_cifar10_loaders

def main():
    # 1. Load data
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=128,
        download=True
    )

    # 2. Create models
    models = [resnet32(num_classes=10) for _ in range(3)]

    # 3. Configure
    config = DMLConfig(
        temperature=3.0,
        supervised_weight=1.0,
        mimicry_weight=1.0
    )

    # 4. Train
    trainer = DMLTrainer(models, config=config, device='cuda')
    history = trainer.fit(train_loader, val_loader, epochs=200)

    # 5. Evaluate
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test Accuracy: {test_metrics['val_acc']:.2f}%")

if __name__ == '__main__':
    main()
```

## Next Steps

- Read the [Quickstart Guide](quickstart.md)
- Explore [Tutorials](tutorials/basic_dml.md) for detailed walkthroughs
- Check [API Reference](api/core.md) for function details
