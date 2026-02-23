"""
Advanced Learning Rate Scheduling Demo

This example demonstrates the enhanced LR scheduling capabilities including:
- SchedulerConfig for easy configuration
- Warmup functionality
- Pre-configured training recipes
- Best practices for different scenarios
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pydml.trainers import DMLTrainer
from pydml.utils.lr_scheduling import (
    SchedulerType,
    WarmupConfig,
    SchedulerConfig,
    create_schedulers_from_config,
    get_cifar_schedule,
    get_imagenet_schedule,
    get_fine_tuning_schedule,
)
import matplotlib.pyplot as plt


# Simple model for demonstration
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
    
    def forward(self, x):
        return self.fc(x)


def create_dummy_data(num_samples=500):
    """Create dummy dataset for demonstration."""
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 5, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def visualize_lr_schedule(scheduler, optimizer, num_epochs, title="Learning Rate Schedule"):
    """Visualize learning rate schedule."""
    lrs = []
    for epoch in range(num_epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), lrs, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    return lrs


print("=" * 80)
print("Example 1: Basic SchedulerConfig Usage")
print("=" * 80)
print()

# Create models
models = [SimpleNet() for _ in range(2)]

# Create optimizers
optimizers = [torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9) for m in models]

# Configure scheduler using SchedulerConfig
config = SchedulerConfig(
    scheduler_type=SchedulerType.COSINE,
    base_lr=0.1,
    T_max=100,
    eta_min=0.0
)

# Create schedulers from config
schedulers = create_schedulers_from_config(optimizers, config)

# Create trainer
trainer = DMLTrainer(
    models=models,
    optimizers=optimizers,
    schedulers=schedulers,
    device='cpu'
)

print(f" Created trainer with {len(models)} models")
print(f" Scheduler type: {config.scheduler_type.value}")
print(f" Initial LR: {trainer.get_learning_rates()[0]:.6f}")
print()

# Train for a few epochs
train_loader = create_dummy_data()
for epoch in range(1, 4):
    trainer.train_epoch(train_loader, epoch=epoch)
    current_lr = trainer.get_learning_rates()[0]
    print(f"  Epoch {epoch}: LR = {current_lr:.6f}")

print()


print("=" * 80)
print("Example 2: Warmup + Cosine Annealing")
print("=" * 80)
print()

# Reset models
models = [SimpleNet() for _ in range(2)]
optimizers = [torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9) for m in models]

# Configure with warmup
config = SchedulerConfig(
    scheduler_type=SchedulerType.COSINE,
    base_lr=0.1,
    T_max=50,
    eta_min=1e-6,
    warmup=WarmupConfig(
        warmup_epochs=5,
        warmup_start_lr=1e-6,
        warmup_method='linear'
    )
)

schedulers = create_schedulers_from_config(optimizers, config)
trainer = DMLTrainer(models=models, optimizers=optimizers, schedulers=schedulers, device='cpu')

print("Configuration:")
print(f"  Warmup epochs: {config.warmup.warmup_epochs}")
print(f"  Warmup start LR: {config.warmup.warmup_start_lr:.6f}")
print(f"  Warmup method: {config.warmup.warmup_method}")
print(f"  Main scheduler: {config.scheduler_type.value}")
print()

# Simulate training to show LR progression
print("LR progression:")
for epoch in range(1, 11):
    trainer.train_epoch(train_loader, epoch=epoch)
    current_lr = trainer.get_learning_rates()[0]
    phase = "warmup" if epoch <= 5 else "main schedule"
    print(f"  Epoch {epoch:2d} ({phase:13s}): LR = {current_lr:.6f}")

print()


print("=" * 80)
print("Example 3: CIFAR Training Recipe")
print("=" * 80)
print()

# Reset models
models = [SimpleNet() for _ in range(2)]
optimizers = [torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) for m in models]

# Use pre-configured CIFAR schedule
schedulers = get_cifar_schedule(optimizers, total_epochs=200, warmup_epochs=5)

trainer = DMLTrainer(models=models, optimizers=optimizers, schedulers=schedulers, device='cpu')

print("Using CIFAR training recipe:")
print("  - Optimizer: SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)")
print("  - Scheduler: Cosine annealing with warmup")
print("  - Total epochs: 200")
print("  - Warmup epochs: 5")
print()

# Show LR at key points
epochs_to_check = [1, 5, 10, 50, 100, 150, 200]
print("Learning rate at key epochs:")
for target_epoch in epochs_to_check:
    for epoch in range(1, target_epoch + 1):
        if epoch <= len(lrs_so_far := []):
            continue
        trainer.train_epoch(train_loader, epoch=epoch)
    lr = trainer.get_learning_rates()[0]
    print(f"  Epoch {target_epoch:3d}: LR = {lr:.8f}")

print()


print("=" * 80)
print("Example 4: ImageNet Training Recipe")
print("=" * 80)
print()

# Reset models
models = [SimpleNet() for _ in range(2)]
optimizers = [torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) for m in models]

# Use pre-configured ImageNet schedule
schedulers = get_imagenet_schedule(optimizers, total_epochs=90)

trainer = DMLTrainer(models=models, optimizers=optimizers, schedulers=schedulers, device='cpu')

print("Using ImageNet training recipe:")
print("  - Optimizer: SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)")
print("  - Scheduler: MultiStep with LR drops at ~30, ~60, ~80 epochs")
print("  - Total epochs: 90")
print()

# Simulate to show LR drops
for epoch in range(1, 91):
    prev_lr = trainer.get_learning_rates()[0]
    trainer.train_epoch(train_loader, epoch=epoch)
    new_lr = trainer.get_learning_rates()[0]
    
    # Show when LR drops
    if abs(new_lr - prev_lr) > 1e-6:
        print(f"  Epoch {epoch}: LR dropped from {prev_lr:.6f} to {new_lr:.6f}")

final_lr = trainer.get_learning_rates()[0]
print(f"\n  Final LR at epoch 90: {final_lr:.6f}")
print()


print("=" * 80)
print("Example 5: Fine-tuning Recipe")
print("=" * 80)
print()

# Reset models
models = [SimpleNet() for _ in range(2)]
optimizers = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in models]

# Use pre-configured fine-tuning schedule
schedulers = get_fine_tuning_schedule(optimizers, total_epochs=50)

trainer = DMLTrainer(models=models, optimizers=optimizers, schedulers=schedulers, device='cpu')

print("Using fine-tuning recipe:")
print("  - Optimizer: Adam(lr=1e-4)")
print("  - Scheduler: Cosine annealing with gentle warmup")
print("  - Total epochs: 50")
print("  - Best for: Transfer learning, fine-tuning pretrained models")
print()

# Show LR progression
for epoch in [1, 3, 10, 25, 50]:
    for e in range(1, epoch + 1):
        trainer.train_epoch(train_loader, epoch=e)
    lr = trainer.get_learning_rates()[0]
    print(f"  Epoch {epoch:2d}: LR = {lr:.8f}")

print()


print("=" * 80)
print("Example 6: Polynomial Decay Schedule")
print("=" * 80)
print()

# Reset models
models = [SimpleNet() for _ in range(2)]
optimizers = [torch.optim.SGD(m.parameters(), lr=0.1) for m in models]

# Configure polynomial decay
config = SchedulerConfig(
    scheduler_type=SchedulerType.POLYNOMIAL,
    base_lr=0.1,
    T_max=100,
    power=2.0,
    final_lr=0.001
)

schedulers = create_schedulers_from_config(optimizers, config)
trainer = DMLTrainer(models=models, optimizers=optimizers, schedulers=schedulers, device='cpu')

print("Using polynomial decay schedule:")
print(f"  Power: {config.power}")
print(f"  Initial LR: {config.base_lr}")
print(f"  Final LR: {config.final_lr}")
print()

# Show LR progression
for epoch in [1, 25, 50, 75, 100]:
    for e in range(1, epoch + 1):
        trainer.train_epoch(train_loader, epoch=e)
    lr = trainer.get_learning_rates()[0]
    print(f"  Epoch {epoch:3d}: LR = {lr:.6f}")

print()


print("=" * 80)
print("Example 7: Comparing Different Schedulers")
print("=" * 80)
print()

scheduler_configs = {
    'Step (γ=0.1, step=30)': SchedulerConfig(
        scheduler_type=SchedulerType.STEP,
        base_lr=0.1,
        step_size=30,
        gamma=0.1
    ),
    'Cosine Annealing': SchedulerConfig(
        scheduler_type=SchedulerType.COSINE,
        base_lr=0.1,
        T_max=100,
        eta_min=0.0
    ),
    'Exponential (γ=0.98)': SchedulerConfig(
        scheduler_type=SchedulerType.EXPONENTIAL,
        base_lr=0.1,
        gamma=0.98
    ),
    'MultiStep [30,60,90]': SchedulerConfig(
        scheduler_type=SchedulerType.MULTISTEP,
        base_lr=0.1,
        milestones=[30, 60, 90],
        gamma=0.1
    ),
}

print("Comparing LR schedules at epoch 50:")
print()

for name, config in scheduler_configs.items():
    model = SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = create_schedulers_from_config([optimizer], config)[0]
    
    # Step to epoch 50
    for _ in range(50):
        scheduler.step()
    
    lr_at_50 = optimizer.param_groups[0]['lr']
    print(f"  {name:25s}: LR = {lr_at_50:.6f}")

print()


print("=" * 80)
print("Example 8: Best Practices Summary")
print("=" * 80)
print()

best_practices = """
Best Practices for Learning Rate Scheduling:

1. **CIFAR-10/100 Training**
   - Use: get_cifar_schedule() or cosine annealing
   - LR: 0.1 (SGD with momentum=0.9)
   - Warmup: 5 epochs from 1e-6
   - Total: 200-300 epochs
   
   ```python
   schedulers = get_cifar_schedule(optimizers, total_epochs=200)
   ```

2. **ImageNet Training**
   - Use: get_imagenet_schedule() or multistep
   - LR: 0.1 (SGD with momentum=0.9, weight_decay=1e-4)
   - Drops: At 30%, 67%, 89% of total epochs
   - Warmup: 5 epochs recommended
   
   ```python
   schedulers = get_imagenet_schedule(optimizers, total_epochs=90)
   ```

3. **Fine-tuning Pretrained Models**
   - Use: get_fine_tuning_schedule() or gentle cosine
   - LR: 1e-4 to 1e-5 (Adam/AdamW)
   - Warmup: 3 epochs
   - Total: 20-50 epochs
   
   ```python
   schedulers = get_fine_tuning_schedule(optimizers, total_epochs=50)
   ```

4. **Deep Mutual Learning Specific**
   - Same schedule for all models
   - Warmup helps with collaborative convergence
   - Cosine annealing works well for long training
   
   ```python
   config = SchedulerConfig(
       scheduler_type=SchedulerType.COSINE,
       base_lr=0.1,
       T_max=200,
       warmup=WarmupConfig(warmup_epochs=5)
   )
   schedulers = create_schedulers_from_config(optimizers, config)
   ```

5. **When to Use Different Schedulers**
   - **Cosine Annealing**: Most versatile, smooth decay
   - **MultiStep**: Sharp drops, good for specific milestones
   - **Step**: Simple, predictable decay
   - **Exponential**: Gradual continuous decay
   - **ReduceOnPlateau**: Adaptive, based on validation metrics
   - **Polynomial**: Customizable decay rate

6. **Warmup Guidelines**
   - Always use warmup for large batch sizes (>256)
   - 5-10 epochs for CIFAR, 5-10 for ImageNet
   - Linear warmup is most common
   - Start from 1e-6 or 1% of base LR

7. **Common Pitfalls to Avoid**
   -  Forgetting to call scheduler.step()
   -  Different schedules for different models (in DML)
   -  Too aggressive decay (LR drops too fast)
   -  No warmup with large LR (unstable start)
   -  Wrong scheduler for task (e.g., step LR for fine-tuning)

8. **Monitoring Learning Rate**
   ```python
   # Check current LR
   current_lr = trainer.get_learning_rates()[0]
   print(f"Current LR: {current_lr:.6f}")
   
   # Log LR during training
   for epoch in range(epochs):
       train_metrics = trainer.train_epoch(loader, epoch)
       logger.log({'lr': trainer.get_learning_rates()[0]})
   ```

9. **Hyperparameter Selection**
   - Base LR: 0.1 for SGD, 1e-3 for Adam (standard)
   - For DML: Same as standard training
   - Tune on validation set, not train set
   - Use learning rate finder for initial estimate

10. **Integration with Other Techniques**
    - Works with: AMP, gradient clipping, warmup
    - Checkpoint best model by validation metric
    - Resume training: scheduler state is saved
    
    ```python
    # Saving
    checkpoint = {
        'models': [m.state_dict() for m in models],
        'optimizers': [opt.state_dict() for opt in optimizers],
        'schedulers': [sch.state_dict() for sch in schedulers],
        'epoch': epoch,
    }
    
    # Loading
    for sch, state in zip(schedulers, checkpoint['schedulers']):
        sch.load_state_dict(state)
    ```
"""

print(best_practices)

print()
print("=" * 80)
print("All examples completed!")
print("=" * 80)
print()
print("Key Takeaways:")
print("  1. Use SchedulerConfig for easy configuration")
print("  2. Always include warmup for better convergence")
print("  3. Use pre-configured recipes (get_cifar_schedule, etc.)")
print("  4. Cosine annealing is a safe default choice")
print("  5. Monitor LR during training to verify schedule")
print()
