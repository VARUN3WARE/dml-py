# Knowledge Distillation Tutorial

Learn how to use knowledge distillation to transfer knowledge from a teacher to a student network.

## What is Knowledge Distillation?

Knowledge distillation transfers knowledge from a large, complex "teacher" model to a smaller, efficient "student" model.

### Benefits

- **Model Compression**: Deploy smaller models with similar performance
- **Faster Inference**: Student runs faster than teacher
- **Lower Memory**: Smaller models require less memory
- **Transfer Learning**: Leverage pre-trained teachers

## Basic Distillation

### Step 1: Load Teacher Model

```python
from pydml import DistillationTrainer, DistillationConfig
from torchvision import models

# Load pre-trained teacher (ResNet-50)
teacher = models.resnet50(pretrained=True)
teacher.eval()  # Set to evaluation mode
```

### Step 2: Create Student Model

```python
# Create smaller student (ResNet-18)
student = models.resnet18(pretrained=False)

print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")
```

### Step 3: Configure Distillation

```python
config = DistillationConfig(
    temperature=4.0,  # Softening parameter
    alpha=0.7        # Balance hard/soft targets
)
```

The loss function:
$$L = \alpha L_{hard} + (1-\alpha) L_{soft}$$

Where:

- $L_{hard}$ = CE loss with ground truth
- $L_{soft}$ = KL divergence with teacher
- Higher $\alpha$ = more emphasis on ground truth

### Step 4: Train Student

```python
from pydml.utils import get_cifar10_loaders

# Load data
train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=128)

# Create trainer
trainer = DistillationTrainer(
    student_model=student,
    teacher_model=teacher,
    config=config,
    device='cuda'
)

# Train student
history = trainer.fit(train_loader, val_loader, epochs=100)
```

### Step 5: Evaluate

```python
test_metrics = trainer.evaluate(test_loader)

print(f"Student Accuracy: {test_metrics['val_acc']:.2f}%")
print(f"Student vs Teacher: {test_metrics['val_acc'] - teacher_acc:.2f}% gap")
```

## CIFAR-10 Example

Complete example with CIFAR models:

```python
import torch
from pydml import DistillationTrainer, DistillationConfig
from pydml.models.cifar import resnet32, resnet110
from pydml.utils import get_cifar10_loaders

# Load data
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=128, download=True
)

# Teacher: ResNet-110 (larger, pre-trained)
teacher = resnet110(num_classes=10)
teacher.load_state_dict(torch.load('pretrained_resnet110.pth'))
teacher.eval()

# Student: ResNet-32 (smaller)
student = resnet32(num_classes=10)

# Configure
config = DistillationConfig(
    temperature=4.0,
    alpha=0.7
)

# Setup optimizer for student
optimizer = torch.optim.SGD(
    student.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

# Train
trainer = DistillationTrainer(
    student_model=student,
    teacher_model=teacher,
    config=config,
    optimizer=optimizer,
    device='cuda'
)

history = trainer.fit(train_loader, val_loader, epochs=200)

# Evaluate
test_metrics = trainer.evaluate(test_loader)
print(f"Student Accuracy: {test_metrics['val_acc']:.2f}%")
```

## Temperature Analysis

The temperature parameter controls knowledge transfer:

```python
temperatures = [1.0, 2.0, 4.0, 8.0, 16.0]
results = {}

for T in temperatures:
    config = DistillationConfig(temperature=T, alpha=0.7)
    trainer = DistillationTrainer(student, teacher, config=config, device='cuda')

    history = trainer.fit(train_loader, val_loader, epochs=50)
    metrics = trainer.evaluate(test_loader)

    results[T] = metrics['val_acc']
    print(f"T={T}: {metrics['val_acc']:.2f}%")
```

### Temperature Effects

- **T=1**: Hard targets only (no distillation)
- **T=2-4**: Optimal range for most tasks
- **T=8-16**: Very soft targets, may hurt performance

## Alpha Analysis

Balance between hard and soft targets:

```python
alphas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
results = {}

for alpha in alphas:
    config = DistillationConfig(temperature=4.0, alpha=alpha)
    trainer = DistillationTrainer(student, teacher, config=config, device='cuda')

    history = trainer.fit(train_loader, val_loader, epochs=50)
    metrics = trainer.evaluate(test_loader)

    results[alpha] = metrics['val_acc']
    print(f"α={alpha}: {metrics['val_acc']:.2f}%")
```

### Alpha Effects

- **α=0**: Only soft targets (teacher only)
- **α=0.5**: Balanced
- **α=1**: Only hard targets (no teacher)

Optimal: Usually α=0.5-0.7

## Advanced Techniques

### Gradual Unfreezing

Start with frozen teacher, gradually unfreeze:

```python
# Freeze teacher initially
for param in teacher.parameters():
    param.requires_grad = False

# Train student
trainer.fit(train_loader, val_loader, epochs=50)

# Fine-tune teacher
for param in teacher.parameters():
    param.requires_grad = True

optimizer = torch.optim.SGD(
    list(student.parameters()) + list(teacher.parameters()),
    lr=0.01  # Lower LR for fine-tuning
)

trainer.fit(train_loader, val_loader, epochs=50)
```

### Progressive Distillation

Gradually increase teacher influence:

```python
for epoch in range(200):
    # Increase alpha over time
    alpha = min(1.0, 0.5 + (epoch / 200) * 0.5)

    config = DistillationConfig(temperature=4.0, alpha=alpha)
    trainer.config = config

    trainer.train_epoch(train_loader, epoch)
```

## Next Steps

- Try [Co-Distillation](../user_guide/trainers.md) (teacher + peers)
- Explore [Feature-Based Distillation](advanced_features.md)
- Learn about [Custom Teachers](custom_models.md)
