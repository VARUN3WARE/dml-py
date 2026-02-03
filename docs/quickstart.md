# Quickstart Guide

This guide will get you started with PyDML in 5 minutes.

## Basic Deep Mutual Learning

Train multiple networks collaboratively:

```python
from pydml import DMLTrainer
from torchvision import models

# Create 3 ResNet18 models
models = [models.resnet18() for _ in range(3)]

# Create trainer
trainer = DMLTrainer(models, device='cuda')

# Train collaboratively
trainer.fit(train_loader, val_loader, epochs=100)

# Evaluate ensemble
metrics = trainer.evaluate(test_loader)
print(f"Ensemble Accuracy: {metrics['val_acc']:.2f}%")
```

## Complete CIFAR-10 Example

A full training pipeline with PyDML's built-in utilities:

```python
import torch
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32
from pydml.utils import get_cifar10_loaders

# 1. Load CIFAR-10 data
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=128,
    num_workers=4,
    val_split=0.1,
    download=True
)

# 2. Create models (3 ResNet-32)
models = [resnet32(num_classes=10) for _ in range(3)]

# 3. Configure DML parameters
config = DMLConfig(
    temperature=3.0,          # Softmax temperature for KD
    supervised_weight=1.0,    # Weight for supervised loss
    mimicry_weight=1.0,       # Weight for peer learning
    peer_selection='all'      # Learn from all peers
)

# 4. Setup optimizers
optimizers = [
    torch.optim.SGD(
        m.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    for m in models
]

# 5. Setup learning rate schedulers (optional)
from pydml.utils import get_cifar_schedule
schedulers = get_cifar_schedule(optimizers, total_epochs=200, warmup_epochs=5)

# 6. Create trainer
trainer = DMLTrainer(
    models,
    config=config,
    optimizers=optimizers,
    schedulers=schedulers,
    device='cuda'
)

# 7. Train with monitoring
from pydml.analysis import TrainingMonitor

monitor = TrainingMonitor(window_size=5, overfitting_threshold=5.0)

for epoch in range(1, 201):
    train_metrics = trainer.train_epoch(train_loader, epoch)
    val_metrics = trainer.evaluate(val_loader)

    # Update monitor
    monitor.update(epoch, train_metrics, val_metrics)

    # Check for overfitting
    if monitor.is_overfitting(strict=True):
        print(monitor.get_overfitting_report())

    # Save checkpoint
    if epoch % 10 == 0:
        trainer.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch=epoch)

# 8. Evaluate final ensemble
test_metrics = trainer.evaluate(test_loader)
print(f"Final Test Accuracy: {test_metrics['val_acc']:.2f}%")

# Get individual model performances
for i, acc in enumerate(test_metrics.get('individual_accs', [])):
    print(f"Model {i+1} Accuracy: {acc:.2f}%")
```

## Knowledge Distillation

Transfer knowledge from a teacher to student:

```python
from pydml import DistillationTrainer, DistillationConfig

# Create teacher and student
teacher = models.resnet50(pretrained=True)
student = models.resnet18()

# Configure distillation
config = DistillationConfig(
    temperature=4.0,
    alpha=0.7  # Balance between hard and soft targets
)

# Train student
trainer = DistillationTrainer(
    student_model=student,
    teacher_model=teacher,
    config=config,
    device='cuda'
)

trainer.fit(train_loader, val_loader, epochs=100)
```

## Feature-Based DML

Learn from intermediate layer representations:

```python
from pydml import FeatureDMLTrainer

# Create models with feature extraction hooks
models = [resnet32(num_classes=10) for _ in range(3)]

# Specify which layers to use for feature matching
feature_layers = ['layer2', 'layer3']

trainer = FeatureDMLTrainer(
    models,
    feature_layers=feature_layers,
    device='cuda'
)

trainer.fit(train_loader, val_loader, epochs=200)
```

## Co-Distillation

Combine teacher guidance with peer learning:

```python
from pydml import CoDistillationTrainer, CoDistillationConfig

# Create teacher and multiple students
teacher = models.resnet50(pretrained=True)
students = [models.resnet18() for _ in range(3)]

config = CoDistillationConfig(
    temperature=3.0,
    teacher_weight=0.5,  # Weight for teacher knowledge
    peer_weight=0.5      # Weight for peer learning
)

trainer = CoDistillationTrainer(
    student_models=students,
    teacher_model=teacher,
    config=config,
    device='cuda'
)

trainer.fit(train_loader, val_loader, epochs=100)
```

## Curriculum Learning

Gradually increase training difficulty:

```python
from pydml import DMLTrainer
from pydml.strategies import CurriculumStrategy

# Create trainer
trainer = DMLTrainer(models, device='cuda')

# Setup curriculum learning
curriculum = CurriculumStrategy(
    strategy='confidence',  # Start with confident samples
    warmup_epochs=10
)

# Apply curriculum to training
curriculum.apply(trainer, train_loader)
trainer.fit(train_loader, val_loader, epochs=200)
```

## Using Callbacks

Monitor and control training with callbacks:

```python
from pydml.core.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

callbacks = [
    # Stop if no improvement for 10 epochs
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.001
    ),

    # Save best model
    ModelCheckpoint(
        filepath='best_model.pth',
        monitor='val_acc',
        mode='max',
        save_best_only=True
    ),

    # Log to TensorBoard
    TensorBoard(log_dir='./logs')
]

trainer = DMLTrainer(models, callbacks=callbacks, device='cuda')
trainer.fit(train_loader, val_loader, epochs=200)
```

## Checkpoint Management

Save and resume training:

```python
from pydml.utils import CheckpointManager, auto_resume

# Automatic resume from latest checkpoint
trainer = DMLTrainer(models, device='cuda')
start_epoch = auto_resume(trainer, checkpoint_dir='checkpoints')

trainer.fit(
    train_loader,
    val_loader,
    epochs=200,
    start_epoch=start_epoch
)

# Manual checkpoint management
manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    max_to_keep=5,
    keep_best=True,
    monitor='val_acc',
    mode='max'
)

for epoch in range(1, 201):
    train_metrics = trainer.train_epoch(train_loader, epoch)
    val_metrics = trainer.evaluate(val_loader)

    # Save checkpoint
    manager.save(trainer, epoch, {**train_metrics, **val_metrics})

# Load best model
best_epoch = manager.load_best(trainer)
print(f"Loaded best model from epoch {best_epoch}")
```

## Robustness Analysis

Evaluate model robustness:

```python
from pydml.analysis import test_robustness

results = test_robustness(
    trainer.models,
    test_loader,
    noise_levels=[0.0, 0.1, 0.2, 0.3],
    attack_types=['gaussian', 'uniform', 'salt_pepper']
)

print(f"Robustness score: {results['robustness_score']:.2f}")
```

## Visualization

Visualize training progress:

```python
from pydml.analysis import plot_training_history, plot_model_comparison

# Plot training curves
plot_training_history(
    history,
    metrics=['loss', 'accuracy'],
    save_path='training_curves.png'
)

# Compare model performances
plot_model_comparison(
    trainer.models,
    test_loader,
    save_path='model_comparison.png'
)
```

## Next Steps

- Explore [Examples](examples.md) for more use cases
- Read the [User Guide](user_guide/core_concepts.md) for in-depth concepts
- Check the [API Reference](api/core.md) for detailed documentation
- See [Tutorials](tutorials/basic_dml.md) for step-by-step guides
