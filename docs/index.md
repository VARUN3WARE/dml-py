# PyDML Documentation

**PyDML** is a production-ready Python library for collaborative neural network training using Deep Mutual Learning (DML) and related techniques.

```{toctree}
---
maxdepth: 2
caption: Getting Started
---
installation
quickstart
examples
```

```{toctree}
---
maxdepth: 2
caption: User Guide
---
user_guide/core_concepts
user_guide/trainers
user_guide/models
user_guide/losses
user_guide/callbacks
user_guide/utilities
```

```{toctree}
---
maxdepth: 2
caption: Tutorials
---
tutorials/basic_dml
tutorials/knowledge_distillation
tutorials/advanced_features
tutorials/custom_models
tutorials/production_deployment
```

```{toctree}
---
maxdepth: 2
caption: API Reference
---
api/core
api/trainers
api/models
api/losses
api/strategies
api/analysis
api/utils
```

```{toctree}
---
maxdepth: 1
caption: Additional Information
---
changelog
contributing
license
```

## Overview

PyDML implements state-of-the-art collaborative learning techniques that allow multiple neural networks to learn from each other during training, leading to:

- **Better Accuracy**: 3-5% improvement over independent training
- **Improved Generalization**: Reduced overfitting through peer learning
- **Ensemble Benefits**: Multiple trained models for inference
- **Research-Ready**: Modular design for experimentation

## Key Features

🤝 **Deep Mutual Learning**
: Train multiple networks collaboratively with bidirectional knowledge transfer

🎓 **Knowledge Distillation**
: Transfer knowledge from teacher to student networks

⚡ **Feature-Based Learning**
: Learn from intermediate representations, not just outputs

📊 **Advanced Monitoring**
: Automatic overfitting detection and training analysis

🛡️ **Production-Ready**
: Comprehensive input validation, checkpoint management, error handling

🔬 **Research Tools**
: Curriculum learning, peer selection strategies, attention transfer

## Quick Example

```python
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32
from pydml.utils import get_cifar10_loaders
import torch

# Load data
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=128, download=True
)

# Create models
models = [resnet32(num_classes=10) for _ in range(3)]

# Configure DML
config = DMLConfig(
    temperature=3.0,
    supervised_weight=1.0,
    mimicry_weight=1.0
)

# Setup optimizers
optimizers = [
    torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    for m in models
]

# Train collaboratively
trainer = DMLTrainer(models, config=config, optimizers=optimizers, device='cuda')
history = trainer.fit(train_loader, val_loader, epochs=200)

# Evaluate ensemble
test_metrics = trainer.evaluate(test_loader)
print(f"Ensemble Accuracy: {test_metrics['val_acc']:.2f}%")
```

## Installation

Install from PyPI:

```bash
pip install pytorch-dml
```

Or install from source:

```bash
git clone https://github.com/VARUN3WARE/dml-py.git
cd dml-py
pip install -e .
```

## Citation

If you use PyDML in your research, please cite the original Deep Mutual Learning paper:

```bibtex
@inproceedings{zhang2018deep,
  title={Deep mutual learning},
  author={Zhang, Ying and Xiang, Tao and Hospedales, Timothy M and Lu, Huchuan},
  booktitle={CVPR},
  pages={4320--4328},
  year={2018}
}
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
