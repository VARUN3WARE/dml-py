# pytorch-dml

[![PyPI version](https://badge.fury.io/py/pytorch-dml.svg)](https://badge.fury.io/py/pytorch-dml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://varun3ware.github.io/dml-py/)

A collaborative deep learning library implementing Deep Mutual Learning (DML) and related methods in PyTorch. Train multiple neural networks simultaneously so they learn from each other's predictions alongside ground truth labels.

**[Documentation](docs/README.md)** | **[API Reference](docs/api/core.md)** | **[Examples](examples/)** | **[Future Work](FUTURE_WORK.md)**

## Installation

```bash
pip install pytorch-dml
```

Or from source:

```bash
git clone https://github.com/VARUN3WARE/dml-py.git
cd dml-py
pip install -e .
```

Requires Python >= 3.8, PyTorch >= 2.0.0, torchvision >= 0.15.0.

## Quick Start

```python
import torch
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32

models = [resnet32(num_classes=100) for _ in range(2)]

config = DMLConfig(temperature=3.0, supervised_weight=1.0, mimicry_weight=1.0)

optimizers = [
    torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    for m in models
]

trainer = DMLTrainer(models, config=config, device='cuda', optimizers=optimizers)
history = trainer.fit(train_loader, val_loader, epochs=200)

test_metrics = trainer.evaluate(test_loader)
print(f"Test Accuracy: {test_metrics['val_acc']:.2f}%")
```

## Features

| Category        | Details                                                                       |
| --------------- | ----------------------------------------------------------------------------- |
| **Trainers**    | DML, Knowledge Distillation, Co-Distillation, Feature-based DML               |
| **Models**      | ResNet32/110, MobileNetV2, WRN-28-10 (CIFAR variants)                         |
| **Strategies**  | Curriculum learning, Peer selection, Temperature scaling                      |
| **Training**    | Mixed precision (FP16/BF16), Checkpoint management, LR scheduling with warmup |
| **Analysis**    | Overfitting detection, Robustness testing, Loss landscape, Visualization      |
| **Reliability** | CUDA OOM handling, Input validation, Reproducibility via seed management      |

## Benchmark Results (CIFAR-10)

| Method                 | Accuracy | Parameters | Training Time |
| ---------------------- | -------- | ---------- | ------------- |
| ResNet32 (baseline)    | 93.50%   | 467K       | 59 min        |
| MobileNetV2 (baseline) | 92.50%   | 2.2M       | 224 min       |
| WRN-28-10 (baseline)   | 96.05%   | 36.5M      | 1141 min      |
| DML 2x ResNet32        | 93.86%   | 933K       | 84 min        |

DML achieves +0.36% accuracy improvement over a single ResNet32 with modest training overhead. See [benchmarks/](benchmarks/) for full experiment details and reproducible configs.

## Key APIs

**Checkpoint Management:**

```python
from pydml.utils import CheckpointManager, auto_resume

start_epoch = auto_resume(trainer, checkpoint_dir='checkpoints')
trainer.fit(train_loader, val_loader, epochs=200, start_epoch=start_epoch)
```

**LR Scheduling with Warmup:**

```python
from pydml.utils import get_cifar_schedule

schedulers = get_cifar_schedule(optimizers, total_epochs=200, warmup_epochs=5)
trainer = DMLTrainer(models, optimizers=optimizers, schedulers=schedulers, device='cuda')
```

**Training Monitor:**

```python
from pydml import TrainingMonitor

monitor = TrainingMonitor(window_size=5, overfitting_threshold=5.0)
# monitor.update(epoch, train_metrics, val_metrics)
# monitor.is_overfitting(), monitor.should_stop_early(patience=10)
```

See [examples/](examples/) for 17 complete demo scripts covering all features.

## Testing

```bash
pytest tests/ -v
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

Open research directions and contribution ideas are documented in [FUTURE_WORK.md](FUTURE_WORK.md).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Reference

This library implements the method from:

**"Deep Mutual Learning"** - Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu. CVPR 2018. [arXiv:1706.00384](https://arxiv.org/abs/1706.00384)
