"""
Learning Rate Schedulers Demo

This example demonstrates how to use various learning rate schedulers
with PyTorch-DML trainers.

Learning rate schedulers adjust the learning rate during training, which can:
- Improve convergence
- Help escape local minima
- Fine-tune models in later epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pydml.trainers import DMLTrainer
from pydml.models.cifar import resnet32, mobilenet_v2
from pydml.utils.schedulers import (
    create_step_schedulers,
    create_cosine_schedulers,
    create_multistep_schedulers,
    create_exponential_schedulers,
    create_cosine_warmrestart_schedulers,
    get_scheduler_info,
)


def demonstrate_schedulers():
    """Demonstrate various learning rate schedulers."""
    print("=" * 80)
    print("LEARNING RATE SCHEDULERS DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create models
    models = [resnet32(num_classes=10), mobilenet_v2(num_classes=10)]
    print(f"Created {len(models)} models: ResNet32, MobileNetV2")
    print()
    
    # 1. StepLR Scheduler
    print("1. StepLR Scheduler")
    print("-" * 80)
    print("Reduces LR by factor of gamma every step_size epochs")
    print()
    
    optimizers = [optim.SGD(m.parameters(), lr=0.1, momentum=0.9) for m in models]
    schedulers = create_step_schedulers(optimizers, step_size=30, gamma=0.1)
    
    trainer = DMLTrainer(
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        device='cpu'
    )
    
    print(f"Initial LR: {trainer.get_learning_rates()[0]:.4f}")
    print(f"Scheduler: StepLR(step_size=30, gamma=0.1)")
    print(f"  - Epoch 1-29: LR = 0.1")
    print(f"  - Epoch 30+: LR = 0.01")
    print(f"  - Epoch 60+: LR = 0.001")
    print()
    
    # 2. CosineAnnealingLR Scheduler
    print("2. CosineAnnealingLR Scheduler")
    print("-" * 80)
    print("Smoothly reduces LR using cosine annealing")
    print()
    
    optimizers = [optim.Adam(m.parameters(), lr=0.001) for m in models]
    schedulers = create_cosine_schedulers(optimizers, T_max=200, eta_min=1e-6)
    
    trainer = DMLTrainer(
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        device='cpu'
    )
    
    print(f"Initial LR: {trainer.get_learning_rates()[0]:.6f}")
    print(f"Scheduler: CosineAnnealingLR(T_max=200, eta_min=1e-6)")
    print(f"  - Smoothly decreases from 0.001 to 0.000001 over 200 epochs")
    print(f"  - Follows cosine curve for smooth transitions")
    
    # Simulate some epochs to show LR changes
    lrs_at_epochs = {}
    for epoch in [0, 50, 100, 150, 199]:
        for _ in range(epoch - len(lrs_at_epochs)):
            for scheduler in schedulers:
                scheduler.step()
        lrs_at_epochs[epoch] = trainer.get_learning_rates()[0]
    
    print(f"\n  Learning rate at different epochs:")
    for epoch, lr in lrs_at_epochs.items():
        print(f"    Epoch {epoch:3d}: LR = {lr:.6f}")
    print()
    
    # 3. MultiStepLR Scheduler
    print("3. MultiStepLR Scheduler")
    print("-" * 80)
    print("Reduces LR at specific milestone epochs")
    print()
    
    models = [resnet32(num_classes=10), mobilenet_v2(num_classes=10)]
    optimizers = [optim.SGD(m.parameters(), lr=0.1, momentum=0.9) for m in models]
    milestones = [60, 120, 160]
    schedulers = create_multistep_schedulers(optimizers, milestones=milestones, gamma=0.2)
    
    trainer = DMLTrainer(
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        device='cpu'
    )
    
    print(f"Initial LR: {trainer.get_learning_rates()[0]:.4f}")
    print(f"Scheduler: MultiStepLR(milestones={milestones}, gamma=0.2)")
    print(f"  - Epoch 1-59: LR = 0.1")
    print(f"  - Epoch 60-119: LR = 0.02")
    print(f"  - Epoch 120-159: LR = 0.004")
    print(f"  - Epoch 160+: LR = 0.0008")
    print()
    
    # 4. ExponentialLR Scheduler
    print("4. ExponentialLR Scheduler")
    print("-" * 80)
    print("Exponentially decays LR every epoch")
    print()
    
    models = [resnet32(num_classes=10), mobilenet_v2(num_classes=10)]
    optimizers = [optim.Adam(m.parameters(), lr=0.01) for m in models]
    schedulers = create_exponential_schedulers(optimizers, gamma=0.95)
    
    trainer = DMLTrainer(
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        device='cpu'
    )
    
    print(f"Initial LR: {trainer.get_learning_rates()[0]:.6f}")
    print(f"Scheduler: ExponentialLR(gamma=0.95)")
    print(f"  - Each epoch: LR = LR * 0.95")
    
    # Show decay
    lrs_decay = [trainer.get_learning_rates()[0]]
    for epoch in range(1, 6):
        for scheduler in schedulers:
            scheduler.step()
        lrs_decay.append(trainer.get_learning_rates()[0])
    
    print(f"\n  Learning rate progression:")
    for epoch, lr in enumerate(lrs_decay):
        print(f"    Epoch {epoch:2d}: LR = {lr:.6f}")
    print()
    
    # 5. CosineAnnealingWarmRestarts
    print("5. CosineAnnealingWarmRestarts Scheduler")
    print("-" * 80)
    print("Cosine annealing with periodic restarts")
    print()
    
    models = [resnet32(num_classes=10), mobilenet_v2(num_classes=10)]
    optimizers = [optim.SGD(m.parameters(), lr=0.1, momentum=0.9) for m in models]
    schedulers = create_cosine_warmrestart_schedulers(optimizers, T_0=10, T_mult=2)
    
    trainer = DMLTrainer(
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        device='cpu'
    )
    
    print(f"Initial LR: {trainer.get_learning_rates()[0]:.4f}")
    print(f"Scheduler: CosineAnnealingWarmRestarts(T_0=10, T_mult=2)")
    print(f"  - Restarts at epochs: 10, 30, 70, ...")
    print(f"  - Each restart period is 2x longer than previous")
    print()
    
    # 6. Scheduler Info Utility
    print("6. Getting Scheduler Information")
    print("-" * 80)
    
    info = get_scheduler_info(schedulers)
    print(f"Number of schedulers: {info['num_schedulers']}")
    print(f"Scheduler types: {info['types']}")
    print(f"Current learning rates: {info['current_lrs']}")
    print()
    
    # 7. Complete Training Example
    print("7. Complete Training Example with Scheduler")
    print("-" * 80)
    
    # Create small dummy dataset
    def create_dummy_data(num_batches=10, batch_size=16):
        return [(torch.randn(batch_size, 3, 32, 32), 
                 torch.randint(0, 10, (batch_size,))) 
                for _ in range(num_batches)]
    
    train_data = create_dummy_data(10)
    val_data = create_dummy_data(3)
    
    # Setup trainer with cosine scheduler
    models = [resnet32(num_classes=10), mobilenet_v2(num_classes=10)]
    optimizers = [optim.SGD(m.parameters(), lr=0.1, momentum=0.9) for m in models]
    schedulers = create_cosine_schedulers(optimizers, T_max=5)
    
    trainer = DMLTrainer(
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        device='cpu'
    )
    
    print("Training for 5 epochs with CosineAnnealingLR...")
    print()
    
    # Track learning rates
    lr_history = []
    
    for epoch in range(1, 6):
        # Get LR before epoch
        current_lr = trainer.get_learning_rates()[0]
        lr_history.append(current_lr)
        
        # Train one epoch
        train_metrics = trainer.train_epoch(train_data, epoch)
        
        # Step schedulers
        for scheduler in trainer.schedulers:
            scheduler.step()
        
        print(f"  Epoch {epoch}: LR={current_lr:.6f}, Loss={train_metrics['train_loss']:.4f}")
    
    print()
    print("Learning Rate Schedule:")
    for epoch, lr in enumerate(lr_history, 1):
        print(f"  Epoch {epoch}: {lr:.6f}")


if __name__ == '__main__':
    demonstrate_schedulers()
