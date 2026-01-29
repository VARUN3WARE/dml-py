"""
Checkpoint and Resume Training Examples

This script demonstrates how to use PyDML's checkpointing system to:
- Save training progress
- Resume from crashes
- Track best models
- Manage checkpoints automatically
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import shutil

from pydml.trainers.dml import DMLTrainer
from pydml.utils.checkpointing import CheckpointManager, auto_resume
from pydml.core.callbacks import ModelCheckpoint
from pydml.models.cifar import resnet32
from pydml.utils.data import get_cifar10_loaders


def example_1_basic_checkpointing():
    """Example 1: Basic save and load checkpoint."""
    print("=" * 70)
    print("Example 1: Basic Checkpointing")
    print("=" * 70)
    
    # Create simple models
    models = [
        nn.Sequential(nn.Linear(10, 5)).cpu()
        for _ in range(2)
    ]
    
    trainer = DMLTrainer(models, device='cpu')
    
    # Create dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)
    
    # Train for a few epochs
    print("\n1. Training for 3 epochs...")
    trainer.fit(loader, epochs=3, verbose=False)
    print(f"   Final epoch: {trainer.current_epoch}")
    
    # Save checkpoint
    checkpoint_path = 'my_checkpoint.pt'
    print(f"\n2. Saving checkpoint to {checkpoint_path}...")
    trainer.save_checkpoint(checkpoint_path)
    print("   ✓ Checkpoint saved")
    
    # Create new trainer and load
    print("\n3. Loading checkpoint into new trainer...")
    new_models = [nn.Sequential(nn.Linear(10, 5)).cpu() for _ in range(2)]
    new_trainer = DMLTrainer(new_models, device='cpu')
    new_trainer.load_checkpoint(checkpoint_path)
    print(f"   ✓ Loaded at epoch: {new_trainer.current_epoch}")
    
    # Cleanup
    os.remove(checkpoint_path)
    print("\n✓ Basic checkpoint save/load complete!")


def example_2_checkpoint_manager():
    """Example 2: Use CheckpointManager for automatic management."""
    print("\n" + "=" * 70)
    print("Example 2: CheckpointManager")
    print("=" * 70)
    
    checkpoint_dir = 'checkpoints_demo'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Create checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_to_keep=3,  # Keep only 3 checkpoints
            keep_best=True,  # Always keep best
            monitor='val_loss',
            mode='min'
        )
        
        # Create models and data
        models = [nn.Sequential(nn.Linear(10, 5)).cpu() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        
        X = torch.randn(200, 10)
        y = torch.randint(0, 5, (200,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32)
        val_loader = DataLoader(dataset, batch_size=32)
        
        print("\nTraining with automatic checkpointing...")
        
        # Train and save checkpoints
        for epoch in range(1, 6):
            trainer.train_epoch(train_loader, epoch=epoch)
            val_metrics = trainer.evaluate(val_loader)
            
            metrics = {
                'val_loss': val_metrics['val_loss'],
                'val_acc': val_metrics['val_acc']
            }
            
            # Save checkpoint
            manager.save(trainer, epoch=epoch, metrics=metrics)
            print(f"  Epoch {epoch}: val_loss={metrics['val_loss']:.4f}, val_acc={metrics['val_acc']:.2f}%")
        
        # Show summary
        print(manager.get_summary())
        
        # List all checkpoints
        checkpoints = manager.list_checkpoints()
        print(f"\nTotal checkpoints saved: {len(checkpoints)}")
        
    finally:
        # Cleanup
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)


def example_3_resume_training():
    """Example 3: Resume training after interruption."""
    print("\n" + "=" * 70)
    print("Example 3: Resume Training")
    print("=" * 70)
    
    checkpoint_dir = 'resume_demo'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Prepare data
        X = torch.randn(200, 10)
        y = torch.randint(0, 5, (200,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32)
        val_loader = DataLoader(dataset, batch_size=32)
        
        # First training session
        print("\n1. First training session (epochs 1-3)...")
        models = [nn.Sequential(nn.Linear(10, 5)).cpu() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        trainer.fit(train_loader, val_loader, epochs=3, verbose=False)
        
        # Save checkpoint
        manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        val_metrics = trainer.evaluate(val_loader)
        manager.save(trainer, epoch=3, metrics={
            'val_loss': val_metrics['val_loss'],
            'val_acc': val_metrics['val_acc']
        })
        print(f"   Trained to epoch {trainer.current_epoch}")
        print(f"   Checkpoint saved")
        
        # Simulate crash and resume
        print("\n2. Simulating crash... Resuming training...")
        new_models = [nn.Sequential(nn.Linear(10, 5)).cpu() for _ in range(2)]
        new_trainer = DMLTrainer(new_models, device='cpu')
        
        # Auto-resume
        start_epoch = auto_resume(new_trainer, checkpoint_dir=checkpoint_dir)
        print(f"   Resumed from epoch {start_epoch-1}, continuing from epoch {start_epoch}")
        
        # Continue training
        print("\n3. Continuing training (epochs 4-6)...")
        new_trainer.fit(train_loader, val_loader, epochs=6, start_epoch=start_epoch-1, verbose=False)
        print(f"   ✓ Training completed to epoch {new_trainer.current_epoch}")
        
    finally:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)


def example_4_best_model_tracking():
    """Example 4: Track and save best model."""
    print("\n" + "=" * 70)
    print("Example 4: Best Model Tracking")
    print("=" * 70)
    
    checkpoint_dir = 'best_model_demo'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            keep_best=True,
            monitor='val_acc',  # Track accuracy
            mode='max'  # Maximize accuracy
        )
        
        models = [nn.Sequential(nn.Linear(10, 5)).cpu() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        
        X = torch.randn(200, 10)
        y = torch.randint(0, 5, (200,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32)
        val_loader = DataLoader(dataset, batch_size=32)
        
        print("\nTraining with best model tracking...")
        
        # Simulate training with varying performance
        fake_accs = [60.0, 65.0, 63.0, 68.0, 67.0, 70.0]
        
        for epoch, acc in enumerate(fake_accs, 1):
            trainer.train_epoch(train_loader, epoch=epoch)
            metrics = {'val_acc': acc, 'val_loss': 2.0 - acc/50}
            manager.save(trainer, epoch=epoch, metrics=metrics)
            print(f"  Epoch {epoch}: val_acc={acc:.2f}%")
        
        print(f"\n✓ Best accuracy: {manager.best_value:.2f}%")
        
        # Load best model
        print("\nLoading best model...")
        new_models = [nn.Sequential(nn.Linear(10, 5)).cpu() for _ in range(2)]
        new_trainer = DMLTrainer(new_models, device='cpu')
        epoch = manager.load_best(new_trainer)
        print(f"✓ Loaded best model from epoch {epoch}")
        
    finally:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)


def example_5_callback_based_checkpointing():
    """Example 5: Use ModelCheckpoint callback."""
    print("\n" + "=" * 70)
    print("Example 5: Callback-Based Checkpointing")
    print("=" * 70)
    
    checkpoint_dir = 'callback_demo'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Create checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_{epoch:04d}_{val_acc:.2f}.pt'),
            monitor='val_acc',
            mode='max',
            save_best_only=True,
            verbose=True
        )
        
        # Create trainer with callback
        models = [nn.Sequential(nn.Linear(10, 5)).cpu() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu', callbacks=[checkpoint_callback])
        
        # Prepare data
        X = torch.randn(200, 10)
        y = torch.randint(0, 5, (200,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32)
        val_loader = DataLoader(dataset, batch_size=32)
        
        print("\nTraining with automatic callbacks...")
        # Callbacks trigger automatically during fit()
        trainer.fit(train_loader, val_loader, epochs=3, verbose=False)
        
        print("\n✓ Checkpoints saved automatically via callback")
        
    finally:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)


def example_6_periodic_checkpointing():
    """Example 6: Periodic checkpoint saving."""
    print("\n" + "=" * 70)
    print("Example 6: Periodic Checkpointing")
    print("=" * 70)
    
    checkpoint_dir = 'periodic_demo'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Save every 2 epochs
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch}.pt'),
            save_best_only=False,
            save_freq=2,  # Save every 2 epochs
            verbose=True
        )
        
        models = [nn.Sequential(nn.Linear(10, 5)).cpu() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu', callbacks=[checkpoint_callback])
        
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32)
        
        print("\nTraining with periodic checkpointing (every 2 epochs)...")
        trainer.fit(train_loader, epochs=5, verbose=False)
        
        # Count saved checkpoints
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint')]
        print(f"\n✓ Saved {len(checkpoints)} checkpoints (epochs 2, 4)")
        
    finally:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)


def example_7_best_practices():
    """Example 7: Best practices for checkpointing."""
    print("\n" + "=" * 70)
    print("Example 7: Checkpointing Best Practices")
    print("=" * 70)
    
    best_practices = """
Best Practices for Checkpoint Management:

1. Use CheckpointManager for Automatic Management:
   ```python
   manager = CheckpointManager(
       checkpoint_dir='checkpoints',
       max_to_keep=5,  # Limit disk usage
       keep_best=True,  # Always preserve best model
       monitor='val_loss',
       mode='min'
   )
   ```

2. Always Save After Each Epoch:
   ```python
   for epoch in range(epochs):
       trainer.train_epoch(train_loader, epoch)
       val_metrics = trainer.evaluate(val_loader)
       manager.save(trainer, epoch, val_metrics)
   ```

3. Use Descriptive Filenames:
   ```python
   # Include epoch and metrics in filename
   filepath = 'checkpoint_epoch{epoch:04d}_acc{val_acc:.2f}.pt'
   ```

4. Resume Training Automatically:
   ```python
   start_epoch = auto_resume(trainer, checkpoint_dir='checkpoints')
   trainer.fit(train_loader, epochs=100, start_epoch=start_epoch)
   ```

5. Save Best Model for Deployment:
   ```python
   # Load best model for final evaluation
   manager.load_best(trainer)
   test_metrics = trainer.evaluate(test_loader)
   ```

6. Limit Checkpoint Storage:
   ```python
   # Keep only recent/best checkpoints to save disk space
   manager = CheckpointManager(max_to_keep=3, keep_best=True)
   ```

7. Use Callbacks for Automation:
   ```python
   callback = ModelCheckpoint(
       filepath='best_model.pt',
       monitor='val_acc',
       save_best_only=True
   )
   trainer = DMLTrainer(models, callbacks=[callback])
   ```

8. Check Checkpoint Status:
   ```python
   # View all checkpoints and their metrics
   print(manager.get_summary())
   checkpoints = manager.list_checkpoints()
   ```

9. Handle Crashes Gracefully:
   ```python
   try:
       start_epoch = auto_resume(trainer, 'checkpoints')
       trainer.fit(loader, epochs=100, start_epoch=start_epoch)
   except KeyboardInterrupt:
       print("Training interrupted, checkpoint saved")
   ```

10. Organize Checkpoints by Experiment:
    ```python
    experiment_name = 'resnet32_dml_cifar10'
    checkpoint_dir = f'checkpoints/{experiment_name}'
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    ```

Common Patterns:

Training with Automatic Checkpointing:
```python
manager = CheckpointManager('checkpoints', max_to_keep=5)

for epoch in range(start_epoch, total_epochs):
    train_metrics = trainer.train_epoch(train_loader, epoch)
    val_metrics = trainer.evaluate(val_loader)
    
    metrics = {**train_metrics, **val_metrics}
    manager.save(trainer, epoch, metrics)
    
    # Early exit on keyboard interrupt
    if keyboard_interrupt:
        break
```

Resume After Crash:
```python
models = [resnet32() for _ in range(2)]
trainer = DMLTrainer(models)

# Automatically resume if checkpoint exists
start_epoch = auto_resume(trainer, 'checkpoints')

# Continue training
trainer.fit(train_loader, val_loader, 
           epochs=200, start_epoch=start_epoch)
```

Deploy Best Model:
```python
# Train with checkpointing
manager = CheckpointManager('checkpoints', monitor='val_acc', mode='max')
# ... training ...

# Load best for deployment
best_epoch = manager.load_best(trainer)
print(f"Loaded best model from epoch {best_epoch}")

# Export for production
torch.save(trainer.models[0].state_dict(), 'production_model.pt')
```
    """
    
    print(best_practices)


if __name__ == "__main__":
    # Run all examples
    example_1_basic_checkpointing()
    example_2_checkpoint_manager()
    example_3_resume_training()
    example_4_best_model_tracking()
    example_5_callback_based_checkpointing()
    example_6_periodic_checkpointing()
    example_7_best_practices()
    
    print("\n" + "=" * 70)
    print("All checkpoint examples completed!")
    print("=" * 70)
