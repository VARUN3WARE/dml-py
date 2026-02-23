"""
Training Monitoring and Overfitting Detection Demo

This example demonstrates how to use TrainingMonitor to:
- Track train vs validation metrics
- Detect overfitting automatically
- Get actionable recommendations
- Make informed decisions about training
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pydml.trainers import DMLTrainer
from pydml.analysis import TrainingMonitor, OverfittingStatus


# Simple model for demonstration
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
    
    def forward(self, x):
        return self.fc(x)


def create_dummy_data(num_samples=500, noise_level=0.0):
    """Create dummy dataset."""
    X = torch.randn(num_samples, 10) + noise_level * torch.randn(num_samples, 10)
    y = torch.randint(0, 5, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


print("=" * 80)
print("Example 1: Basic Training Monitoring")
print("=" * 80)
print()

# Create data
train_loader = create_dummy_data(500)
val_loader = create_dummy_data(200)

# Create models and trainer
models = [SimpleNet() for _ in range(2)]
trainer = DMLTrainer(models=models, device='cpu')

# Create monitor
monitor = TrainingMonitor(
    window_size=5,
    overfitting_threshold=5.0
)

print("Training with monitoring...")
print()

# Train for a few epochs
for epoch in range(1, 11):
    train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
    val_metrics = trainer.evaluate(val_loader)
    
    # Update monitor
    monitor.update(epoch, train_metrics, val_metrics)
    
    # Get current metrics
    current = monitor.get_current_metrics()
    gap = current.generalization_gap
    
    print(f"Epoch {epoch:2d}:")
    print(f"  Train: Loss={current.train_loss:.4f}, Acc={current.train_acc:.2f}%")
    print(f"  Val:   Loss={current.val_loss:.4f}, Acc={current.val_acc:.2f}%")
    print(f"  Gap:   {gap:+.2f}%")
    
    # Check for overfitting
    if monitor.is_overfitting():
        print(f"    Overfitting detected!")
    
    print()

print()
print("=" * 80)
print("Example 2: Detailed Overfitting Report")
print("=" * 80)
print()

# Get comprehensive report
report = monitor.get_overfitting_report()
print(report)
print()


print("=" * 80)
print("Example 3: Training Summary")
print("=" * 80)
print()

summary = monitor.get_summary()
print(summary)
print()


print("=" * 80)
print("Example 4: Simulating Different Training Scenarios")
print("=" * 80)
print()

# Scenario 1: Healthy training (no overfitting)
print("Scenario 1: Healthy Training")
print("-" * 80)

monitor_healthy = TrainingMonitor()
for epoch in range(1, 21):
    train_acc = 50.0 + epoch * 2
    val_acc = 48.0 + epoch * 1.9
    
    monitor_healthy.update(
        epoch,
        {'train_loss': 2.0 - epoch * 0.08, 'train_acc': train_acc},
        {'val_loss': 2.1 - epoch * 0.07, 'val_acc': val_acc}
    )

print(f"Status: {monitor_healthy.get_overfitting_severity().value}")
print(f"Generalization gap: {monitor_healthy.get_generalization_gap():.2f}%")
print(f"Val accuracy trend: {monitor_healthy.get_trend('val_acc')}")
print()

# Scenario 2: Overfitting
print("Scenario 2: Overfitting")
print("-" * 80)

monitor_overfit = TrainingMonitor()
for epoch in range(1, 21):
    # Train acc improves, val acc plateaus then drops
    if epoch <= 10:
        train_acc = 50.0 + epoch * 4
        val_acc = 48.0 + epoch * 3.5
    else:
        train_acc = 90.0 + (epoch - 10) * 1
        val_acc = 83.0 - (epoch - 10) * 0.5
    
    monitor_overfit.update(
        epoch,
        {'train_loss': max(0.1, 2.0 - epoch * 0.12), 'train_acc': train_acc},
        {'val_loss': max(0.3, 2.1 - epoch * 0.05) + (max(0, epoch-10) * 0.03), 'val_acc': val_acc}
    )

print(f"Status: {monitor_overfit.get_overfitting_severity().value}")
print(f"Generalization gap: {monitor_overfit.get_generalization_gap():.2f}%")
print(f"Val accuracy trend: {monitor_overfit.get_trend('val_acc')}")

best_epoch, best_acc = monitor_overfit.get_best_epoch('val_acc')
print(f"Best epoch: {best_epoch} (val_acc={best_acc:.2f}%)")
print()

report = monitor_overfit.get_overfitting_report()
print("Recommendations:")
for i, rec in enumerate(report.recommendations[:3], 1):
    print(f"  {i}. {rec}")
print()

# Scenario 3: Underfitting
print("Scenario 3: Underfitting")
print("-" * 80)

monitor_underfit = TrainingMonitor()
for epoch in range(1, 21):
    # Both train and val acc are low and not improving
    train_acc = 40.0 + epoch * 0.2
    val_acc = 38.0 + epoch * 0.15
    
    monitor_underfit.update(
        epoch,
        {'train_loss': 2.0 - epoch * 0.02, 'train_acc': train_acc},
        {'val_loss': 2.1 - epoch * 0.015, 'val_acc': val_acc}
    )

print(f"Status: {monitor_underfit.get_overfitting_severity().value}")
print(f"Is underfitting: {monitor_underfit.is_underfitting()}")
print(f"Val accuracy trend: {monitor_underfit.get_trend('val_acc')}")
print()

report = monitor_underfit.get_overfitting_report()
print("Recommendations:")
for i, rec in enumerate(report.recommendations[:3], 1):
    print(f"  {i}. {rec}")
print()


print("=" * 80)
print("Example 5: Early Stopping Detection")
print("=" * 80)
print()

monitor_es = TrainingMonitor()

# Simulate training that plateaus
for epoch in range(1, 31):
    if epoch <= 15:
        train_acc = 50.0 + epoch * 3
        val_acc = 48.0 + epoch * 2.8
    else:
        train_acc = 95.0 + (epoch - 15) * 0.2
        val_acc = 90.0 + (epoch - 15) * 0.05  # Plateaus
    
    monitor_es.update(
        epoch,
        {'train_loss': max(0.1, 2.0 - epoch * 0.08), 'train_acc': train_acc},
        {'val_loss': max(0.2, 2.1 - epoch * 0.06) + max(0, epoch-15) * 0.01, 'val_acc': val_acc}
    )
    
    # Check early stopping
    if epoch >= 20:
        should_stop = monitor_es.should_stop_early(patience=5, min_delta=0.1)
        
        if epoch == 20:
            print(f"Epoch {epoch}: Should stop early? {should_stop}")
        
        if should_stop and epoch == 25:
            print(f"Epoch {epoch}: Early stopping triggered!")
            print(f"  Current val_acc: {monitor_es.get_current_metrics().val_acc:.2f}%")
            
            best_epoch, best_acc = monitor_es.get_best_epoch('val_acc')
            print(f"  Best val_acc: {best_acc:.2f}% at epoch {best_epoch}")
            print(f"  Should have stopped {epoch - best_epoch} epochs ago")
            break

print()


print("=" * 80)
print("Example 6: Real-Time Monitoring During Training")
print("=" * 80)
print()

# Create fresh data and trainer
train_loader = create_dummy_data(500)
val_loader = create_dummy_data(200)
models = [SimpleNet() for _ in range(2)]
trainer = DMLTrainer(models=models, device='cpu')

monitor = TrainingMonitor(
    window_size=5,
    overfitting_threshold=5.0
)

print("Training with real-time overfitting checks...")
print()

MAX_EPOCHS = 30
for epoch in range(1, MAX_EPOCHS + 1):
    # Train
    train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
    val_metrics = trainer.evaluate(val_loader)
    
    # Update monitor
    monitor.update(epoch, train_metrics, val_metrics)
    
    # Print metrics
    current = monitor.get_current_metrics()
    print(f"Epoch {epoch:2d}: "
          f"Train={current.train_acc:5.2f}% "
          f"Val={current.val_acc:5.2f}% "
          f"Gap={current.generalization_gap:+5.2f}% "
          f"Trend={monitor.get_trend('val_acc')}")
    
    # Check overfitting every 5 epochs
    if epoch % 5 == 0:
        status = monitor.get_overfitting_severity()
        print(f"  Status: {status.value.replace('_', ' ').title()}")
        
        if status in [OverfittingStatus.MODERATE_OVERFITTING, 
                     OverfittingStatus.SEVERE_OVERFITTING]:
            report = monitor.get_overfitting_report()
            print(f"    Action needed! Top recommendation:")
            print(f"     {report.recommendations[0]}")
    
    # Early stopping check
    if epoch >= 15 and monitor.should_stop_early(patience=8, min_delta=0.1):
        print()
        print(f"Early stopping at epoch {epoch}")
        best_epoch, best_acc = monitor.get_best_epoch('val_acc')
        print(f"Best model was at epoch {best_epoch} with val_acc={best_acc:.2f}%")
        break

print()


print("=" * 80)
print("Example 7: Best Practices Guide")
print("=" * 80)
print()

best_practices = """
Best Practices for Training Monitoring:

1. **Always Track Both Train and Val Metrics**
   ```python
   monitor = TrainingMonitor()
   for epoch in range(epochs):
       train_metrics = trainer.train_epoch(train_loader, epoch)
       val_metrics = trainer.evaluate(val_loader)
       monitor.update(epoch, train_metrics, val_metrics)
   ```

2. **Check Overfitting Regularly**
   - Every 5-10 epochs during training
   - Look for generalization gap > 5%
   - Use strict mode for sustained overfitting detection
   
   ```python
   if monitor.is_overfitting(strict=True):
       report = monitor.get_overfitting_report()
       print(report)
   ```

3. **Set Appropriate Thresholds**
   - CIFAR-10/100: threshold=5-8% is reasonable
   - ImageNet: threshold=3-5% due to larger dataset
   - Small datasets: threshold=10-15% may be acceptable
   
   ```python
   monitor = TrainingMonitor(overfitting_threshold=7.0)
   ```

4. **Use Early Stopping**
   - Prevents wasting computation on overfitting
   - Patience: 10-20 epochs for CIFAR, 5-10 for ImageNet
   - min_delta: 0.1-0.5% depending on task
   
   ```python
   if monitor.should_stop_early(patience=10, min_delta=0.1):
       print("Stopping early!")
       break
   ```

5. **Track Trends, Not Just Current Values**
   - Improving trend is good even if current accuracy is low
   - Degrading trend is bad even if accuracy is high
   
   ```python
   val_trend = monitor.get_trend('val_acc')
   if val_trend == 'degrading':
       print("Validation performance degrading!")
   ```

6. **Save Best Model, Not Latest**
   ```python
   best_epoch, best_acc = monitor.get_best_epoch('val_acc')
   # Load checkpoint from best_epoch for deployment
   ```

7. **Act on Recommendations**
   - Severe overfitting: Add regularization, reduce capacity
   - Moderate overfitting: Monitor closely, consider early stopping
   - Underfitting: Increase capacity, train longer
   
   ```python
   report = monitor.get_overfitting_report()
   if report.status == OverfittingStatus.SEVERE_OVERFITTING:
       # Implement recommendations
       for rec in report.recommendations:
           print(f"TODO: {rec}")
   ```

8. **Use Visualization**
   ```python
   from pydml.analysis import plot_training_history
   plot_training_history(monitor.history, save_path='training.png')
   ```

9. **Per-Model Tracking for DML**
   ```python
   monitor = TrainingMonitor(track_per_model=True)
   # Tracks train_acc_model_0, train_acc_model_1, etc.
   ```

10. **Integrate with Checkpointing**
    ```python
    from pydml.utils import CheckpointManager
    
    manager = CheckpointManager('checkpoints', monitor='val_acc', mode='max')
    monitor = TrainingMonitor()
    
    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.evaluate(val_loader)
        
        monitor.update(epoch, train_metrics, val_metrics)
        manager.save(trainer, epoch, {**train_metrics, **val_metrics})
        
        if monitor.is_overfitting(strict=True):
            print("Overfitting detected - consider stopping")
    ```

Common Pitfalls to Avoid:

 Only looking at validation accuracy (miss overfitting)
 Not checking train accuracy (miss underfitting)
 Training too long without early stopping
 Ignoring generalization gap
 Not saving best model checkpoint
 Using latest model instead of best for deployment
 Not acting on overfitting warnings
 Inappropriate thresholds for dataset size

Interpreting Results:

 **Healthy Training:**
   - Train and val accuracy both improving
   - Gap < 5%
   - Val loss decreasing
   
 **Mild Overfitting (OK):**
   - Gap 3-5%
   - Val accuracy still improving
   - Continue training, monitor closely
   
 **Moderate Overfitting:**
   - Gap 5-10%
   - Val accuracy plateaued
   - Consider early stopping soon
   
 **Severe Overfitting:**
   - Gap > 10%
   - Val accuracy degrading
   - Stop training, add regularization, restart
   
 **Underfitting:**
   - Both train and val accuracy low (< 60%)
   - Not improving
   - Need more capacity or longer training
"""

print(best_practices)

print()
print("=" * 80)
print("All examples completed!")
print("=" * 80)
print()
print("Key Takeaways:")
print("  1. Always monitor both train and validation metrics")
print("  2. Generalization gap is the key indicator of overfitting")
print("  3. Use automatic detection to catch problems early")
print("  4. Act on recommendations promptly")
print("  5. Save best model, not latest model")
print("  6. Use early stopping to avoid wasting compute")
print()
