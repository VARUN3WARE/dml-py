# Basic Deep Mutual Learning Tutorial

Learn the fundamentals of Deep Mutual Learning with PyDML.

## Overview

This tutorial covers:

- Setting up a basic DML experiment
- Understanding the training process
- Evaluating ensemble performance
- Interpreting results

## Prerequisites

```bash
pip install pytorch-dml
```

## Step 1: Import Dependencies

```python
import torch
import torch.nn as nn
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32
from pydml.utils import get_cifar10_loaders
```

## Step 2: Load Data

```python
# Load CIFAR-10 with 10% validation split
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=128,
    num_workers=4,
    val_split=0.1,
    download=True
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
```

## Step 3: Create Models

```python
# Create 3 ResNet-32 models
num_models = 3
models = [resnet32(num_classes=10) for _ in range(num_models)]

# Verify model architecture
print(f"Number of models: {len(models)}")
print(f"Model parameters: {sum(p.numel() for p in models[0].parameters()):,}")
```

## Step 4: Configure DML

```python
config = DMLConfig(
    temperature=3.0,          # Higher = softer probabilities
    supervised_weight=1.0,    # Weight for CE loss
    mimicry_weight=1.0,       # Weight for peer learning
    peer_selection='all'      # Learn from all peers
)

print(f"Configuration: {config}")
```

## Step 5: Setup Optimizers

```python
# SGD with momentum for each model
optimizers = [
    torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    for model in models
]
```

## Step 6: Create Trainer

```python
trainer = DMLTrainer(
    models,
    config=config,
    optimizers=optimizers,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Trainer created with {len(trainer.models)} models")
print(f"Device: {trainer.device}")
```

## Step 7: Train Models

```python
# Train for 10 epochs (use 200 for full training)
num_epochs = 10

history = trainer.fit(
    train_loader,
    val_loader,
    epochs=num_epochs
)

print(f"Training completed!")
```

## Step 8: Evaluate Performance

```python
# Evaluate on test set
test_metrics = trainer.evaluate(test_loader)

print("\nTest Results:")
print(f"Ensemble Accuracy: {test_metrics['val_acc']:.2f}%")
print(f"Ensemble Loss: {test_metrics['val_loss']:.4f}")

# Individual model performances
if 'individual_accs' in test_metrics:
    for i, acc in enumerate(test_metrics['individual_accs']):
        print(f"Model {i+1} Accuracy: {acc:.2f}%")
```

## Step 9: Save Models

```python
# Save final checkpoint
trainer.save_checkpoint(
    'final_checkpoint.pth',
    epoch=num_epochs,
    metrics=test_metrics
)

print("Checkpoint saved!")
```

## Understanding the Results

### DML vs Independent Training

DML typically provides 3-5% improvement over training models independently:

- **Independent**: Each model ~85% accuracy
- **DML**: Each model ~88% accuracy, ensemble ~90%

### Why DML Works

1. **Regularization**: Peer agreement acts as implicit regularization
2. **Knowledge Transfer**: Models share learned representations
3. **Diversity**: Different initializations lead to complementary errors

## Complete Script

Here's the full tutorial script:

```python
import torch
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32
from pydml.utils import get_cifar10_loaders

def main():
    # Load data
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=128, num_workers=4, val_split=0.1, download=True
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

    # Create trainer
    trainer = DMLTrainer(models, config=config, optimizers=optimizers, device='cuda')

    # Train
    history = trainer.fit(train_loader, val_loader, epochs=200)

    # Evaluate
    test_metrics = trainer.evaluate(test_loader)
    print(f"Ensemble Accuracy: {test_metrics['val_acc']:.2f}%")

    # Save
    trainer.save_checkpoint('final_checkpoint.pth', epoch=200)

if __name__ == '__main__':
    main()
```

## Next Steps

- Try [Knowledge Distillation](knowledge_distillation.md)
- Explore [Advanced Features](advanced_features.md)
- Learn about [Custom Models](custom_models.md)
