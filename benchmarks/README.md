# PyDML Benchmarking Infrastructure

This directory contains tools and scripts for reproducible benchmarking of Deep Mutual Learning methods.

## Overview

The benchmarking infrastructure provides:

- **Reproducible Configurations**: JSON-based experiment configs with full parameter tracking
- **Baseline Training**: Clean single-model training for establishing baselines
- **Metrics Logging**: Comprehensive tracking of training metrics with CSV/JSON export
- **Comparison Tools**: Utilities for comparing multiple experiments

## Directory Structure

```
benchmarks/
├── __init__.py              # Package initialization
├── experiment_config.py     # Configuration management
├── baseline_trainer.py      # Single model training
├── metrics_logger.py        # Metrics tracking and logging
├── configs/                 # Experiment configurations
├── results/                 # Training results and metrics
└── scripts/                 # Training scripts
```

## Quick Start

### 1. Basic Baseline Training

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pydml.models.cifar import resnet32
from benchmarks import ExperimentConfig, BaselineTrainer

# Create configuration
config = ExperimentConfig(
    name='baseline_resnet32_cifar10',
    dataset='cifar10',
    model_type='resnet32',
    epochs=200,
    batch_size=128
)

# Setup data
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# Create model
model = resnet32(num_classes=10)

# Train
trainer = BaselineTrainer(model, config, train_loader, val_loader)
results = trainer.train()

print(f"Best validation accuracy: {results['best_val_acc']:.2%}")
```

### 2. Generate Standard Baseline Configs

```python
from benchmarks.experiment_config import create_baseline_configs

# Creates configs for ResNet32, MobileNet, WRN, and DML variants
configs = create_baseline_configs(output_dir='benchmarks/configs')

print(f"Created {len(configs)} baseline configurations")
for config in configs:
    print(f"  - {config.name}")
```

### 3. Compare Experiments

```python
from benchmarks.metrics_logger import compare_experiments

comparison = compare_experiments([
    'baseline_resnet32_cifar10',
    'baseline_mobilenet_cifar10',
    'dml_2resnet_cifar10'
])

print(f"Best: {comparison['best_experiment']} ({comparison['best_accuracy']:.2%})")
print("\nRanking:")
for i, exp in enumerate(comparison['ranking'], 1):
    print(f"  {i}. {exp['name']}: {exp['best_val_acc']:.2%}")
```

## Configuration System

### ExperimentConfig

All experiments are defined using `ExperimentConfig` dataclasses:

```python
config = ExperimentConfig(
    name='my_experiment',           # Unique identifier
    dataset='cifar10',              # Dataset name
    model_type='resnet32',          # Model architecture
    num_models=1,                   # Number of models (1=baseline, >1=DML)
    batch_size=128,
    epochs=200,
    learning_rate=0.1,
    weight_decay=5e-4,
    momentum=0.9,
    seed=42,                        # For reproducibility
    optimizer='sgd',                # 'sgd', 'adam', 'adamw'
    scheduler='multistep',          # 'multistep', 'cosine', 'none'
    scheduler_params={
        'milestones': [60, 120, 160],
        'gamma': 0.2
    },
    notes='Optional description'
)
```

Configurations can be saved and loaded:

```python
# Save
config.save('configs/my_experiment.json')

# Load
loaded_config = ExperimentConfig.load('configs/my_experiment.json')
```

## Metrics Logging

### MetricsLogger

The `MetricsLogger` tracks all training metrics:

```python
logger = MetricsLogger('my_experiment', output_dir='results/')

# During training
for epoch in range(epochs):
    # ... training code ...
    logger.log_epoch(
        epoch=epoch,
        train_loss=train_loss,
        train_acc=train_acc,
        val_loss=val_loss,
        val_acc=val_acc,
        learning_rate=current_lr
    )

# Save results
logger.save()  # Creates JSON and CSV files

# Get summary
summary = logger.get_summary()
print(summary)
```

### Output Files

Each experiment generates two files:

1. **JSON**: `{experiment_name}_metrics.json`
   - Complete metrics with metadata
   - Best epoch and accuracy
   - Timestamps and configuration

2. **CSV**: `{experiment_name}_metrics.csv`
   - Tabular format for easy analysis
   - Compatible with pandas, Excel, etc.

## Baseline Trainer

### BaselineTrainer

Clean interface for single-model training:

```python
trainer = BaselineTrainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader
)

# Run full training
results = trainer.train()

# Save model checkpoint
trainer.save_model('checkpoints/best_model.pth')
```

Features:

- Automatic device management (CPU/GPU)
- Progress bars with tqdm
- Learning rate scheduling
- Periodic metric saving
- Comprehensive logging

## Best Practices

### Reproducibility

1. **Always set seeds** in your config
2. **Save configurations** before training
3. **Use consistent data splits** across experiments
4. **Document environment** (PyTorch version, CUDA version)

### Organization

```
benchmarks/
├── configs/
│   ├── baseline_resnet32_cifar10.json
│   ├── baseline_mobilenet_cifar10.json
│   └── dml_2resnet_cifar10.json
├── results/
│   ├── baseline_resnet32_cifar10_metrics.json
│   ├── baseline_resnet32_cifar10_metrics.csv
│   └── ...
└── scripts/
    ├── run_baselines.py
    └── compare_results.py
```

### Naming Conventions

Use descriptive, consistent names:

- `baseline_{model}_{dataset}` for single models
- `dml_{num}_{model}_{dataset}` for DML experiments
- `kd_{teacher}_{student}_{dataset}` for knowledge distillation

## Example Workflow

```bash
# 1. Generate baseline configs
python -c "from benchmarks.experiment_config import create_baseline_configs; create_baseline_configs()"

# 2. Run experiments
python scripts/run_baselines.py --config configs/baseline_resnet32_cifar10.json

# 3. Compare results
python scripts/compare_results.py --experiments baseline_* dml_*
```

## Integration with PyDML

The benchmarking infrastructure integrates seamlessly with PyDML:

```python
from pydml.trainers import DMLTrainer
from benchmarks import ExperimentConfig, MetricsLogger

# Use same config system for DML
config = ExperimentConfig(
    name='dml_3mixed_cifar10',
    dataset='cifar10',
    model_type='mixed',
    num_models=3
)

# DML trainer can use the same logger
logger = MetricsLogger(config.name)

# Train with DML
dml_trainer = DMLTrainer(models, train_loader, val_loader)
# ... integrate logger ...
```

## Performance Tips

1. **Use multiple workers** for data loading (`num_workers=4`)
2. **Enable CUDA** when available (`device='cuda'`)
3. **Adjust batch size** based on GPU memory
4. **Use mixed precision** training for faster experiments (coming soon)

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Use gradient accumulation
- Try smaller models first

### Slow Training

- Increase `num_workers` for data loading
- Check GPU utilization
- Profile bottlenecks

### Inconsistent Results

- Verify seed is set in config
- Check data shuffling
- Ensure deterministic CUDA operations

## Future Enhancements

- [ ] Multi-GPU support
- [ ] Mixed precision training
- [ ] Automatic hyperparameter tuning
- [ ] Visualization dashboard
- [ ] Statistical significance testing

## Contributing

When adding new benchmarks:

1. Create descriptive configuration files
2. Document expected results
3. Include comparison with baselines
4. Add to comparison scripts

## References

For baseline performance expectations, see:

- CIFAR-10/100 benchmarks: [Link to be added]
- ImageNet results: [Link to be added]
- DML paper comparisons: [Link to be added]
