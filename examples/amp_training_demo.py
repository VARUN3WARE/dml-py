"""
Automatic Mixed Precision (AMP) Training Examples

This script demonstrates how to use PyDML's AMP support for faster training
with lower memory usage while maintaining accuracy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

from pydml.trainers.dml import DMLTrainer, DMLConfig
from pydml.utils.amp import AMPConfig, AMPManager, apply_amp_to_trainer
from pydml.models.cifar import resnet32
from pydml.utils.data import get_cifar10_loaders


def example_1_basic_amp():
    """Example 1: Basic AMP usage with DML trainer."""
    print("=" * 70)
    print("Example 1: Basic AMP Usage")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping this example.")
        return
    
    # Create simple models
    models = [
        nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ).cuda()
        for _ in range(2)
    ]
    
    # Create trainer with AMP enabled
    trainer = DMLTrainer(models, device='cuda', use_amp=True)
    
    print(f"AMP enabled: {trainer.use_amp}")
    print(f"AMP dtype: {trainer.amp_manager.config.dtype}")
    
    # Create dummy data
    X = torch.randn(1000, 100)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64)
    
    # Train for one epoch
    print("\nTraining with AMP...")
    trainer.train_epoch(loader, epoch=1)
    print("✓ Training completed successfully with AMP!")


def example_2_amp_config():
    """Example 2: Custom AMP configuration."""
    print("\n" + "=" * 70)
    print("Example 2: Custom AMP Configuration")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping this example.")
        return
    
    # Create custom AMP config
    amp_config = AMPConfig(
        enabled=True,
        dtype=torch.float16,
        init_scale=2.**10,  # Smaller initial scale
        growth_factor=1.5,  # Slower scale growth
        growth_interval=1000
    )
    
    print(f"AMP Configuration:")
    print(f"  Enabled: {amp_config.enabled}")
    print(f"  Dtype: {amp_config.dtype}")
    print(f"  Init scale: {amp_config.init_scale}")
    print(f"  Growth factor: {amp_config.growth_factor}")
    print(f"  Growth interval: {amp_config.growth_interval}")
    
    # Create AMP manager
    manager = AMPManager(amp_config, device='cuda')
    
    # Example usage
    model = nn.Linear(10, 5).cuda()
    x = torch.randn(2, 10, device='cuda')
    
    with manager.autocast():
        output = model(x)
        print(f"\nOutput dtype in autocast: {output.dtype}")


def example_3_performance_comparison():
    """Example 3: Compare training speed with and without AMP."""
    print("\n" + "=" * 70)
    print("Example 3: Performance Comparison (AMP vs FP32)")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping this example.")
        return
    
    # Create dataset
    X = torch.randn(5000, 100)
    y = torch.randint(0, 10, (5000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Test without AMP
    print("\n1. Training WITHOUT AMP (FP32):")
    models_fp32 = [
        nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        ).cuda()
        for _ in range(2)
    ]
    
    trainer_fp32 = DMLTrainer(models_fp32, device='cuda', use_amp=False)
    
    start_time = time.time()
    trainer_fp32.train_epoch(loader, epoch=1)
    fp32_time = time.time() - start_time
    
    print(f"   Time: {fp32_time:.2f}s")
    
    # Test with AMP
    print("\n2. Training WITH AMP (FP16):")
    models_amp = [
        nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        ).cuda()
        for _ in range(2)
    ]
    
    trainer_amp = DMLTrainer(models_amp, device='cuda', use_amp=True)
    
    start_time = time.time()
    trainer_amp.train_epoch(loader, epoch=1)
    amp_time = time.time() - start_time
    
    print(f"   Time: {amp_time:.2f}s")
    
    # Compare
    speedup = fp32_time / amp_time
    print(f"\nSpeedup with AMP: {speedup:.2f}x")
    if speedup > 1:
        print(f"AMP is {(speedup-1)*100:.1f}% faster!")
    else:
        print("AMP overhead detected (normal for small models)")


def example_4_cifar_with_amp():
    """Example 4: Train CIFAR models with AMP."""
    print("\n" + "=" * 70)
    print("Example 4: CIFAR-10 Training with AMP")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping this example.")
        return
    
    print("Loading CIFAR-10 dataset...")
    try:
        train_loader, val_loader, _ = get_cifar10_loaders(
            batch_size=128,
            download=False  # Don't download if not present
        )
    except:
        print("CIFAR-10 not available. Creating synthetic data instead.")
        X = torch.randn(1000, 3, 32, 32)
        y = torch.randint(0, 10, (1000,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=128)
        val_loader = DataLoader(dataset, batch_size=128)
    
    # Create ResNet models
    print("Creating ResNet-32 models...")
    models = [resnet32(num_classes=10).cuda() for _ in range(2)]
    
    # Create DML config
    config = DMLConfig(
        temperature=3.0,
        supervised_weight=1.0,
        mimicry_weight=1.0
    )
    
    # Create trainer with AMP
    print("Initializing DML trainer with AMP...")
    trainer = DMLTrainer(
        models=models,
        config=config,
        device='cuda',
        use_amp=True,  # Enable AMP
        amp_dtype=torch.float16
    )
    
    print(f"AMP enabled: {trainer.use_amp}")
    
    # Train for 2 epochs
    print("\nTraining for 2 epochs with AMP...")
    history = trainer.fit(train_loader, val_loader, epochs=2, verbose=True)
    
    print(f"\nFinal metrics:")
    print(f"  Train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Val accuracy: {history['val_acc'][-1]:.2f}%")


def example_5_bfloat16():
    """Example 5: Using BFloat16 instead of Float16."""
    print("\n" + "=" * 70)
    print("Example 5: BFloat16 Mixed Precision")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping this example.")
        return
    
    # Check if BF16 is supported
    if not torch.cuda.is_bf16_supported():
        print("BFloat16 not supported on this GPU (requires Ampere or newer)")
        return
    
    # Create models
    models = [
        nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ).cuda()
        for _ in range(2)
    ]
    
    # Use BF16 instead of FP16
    trainer = DMLTrainer(
        models=models,
        device='cuda',
        use_amp=True,
        amp_dtype=torch.bfloat16  # BFloat16
    )
    
    print(f"Using dtype: {trainer.amp_manager.config.dtype}")
    print("BFloat16 has:")
    print("  - Better numerical stability than FP16")
    print("  - Same dynamic range as FP32")
    print("  - Good for large language models")
    
    # Create data
    X = torch.randn(500, 100)
    y = torch.randint(0, 10, (500,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64)
    
    # Train
    print("\nTraining with BFloat16...")
    trainer.train_epoch(loader, epoch=1)
    print("✓ Training completed with BFloat16!")


def example_6_checkpoint_with_amp():
    """Example 6: Save and load checkpoints with AMP state."""
    print("\n" + "=" * 70)
    print("Example 6: Checkpointing with AMP")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping this example.")
        return
    
    import tempfile
    import os
    
    # Create models
    models = [
        nn.Sequential(nn.Linear(10, 5)).cuda()
        for _ in range(2)
    ]
    
    # Create trainer with AMP
    trainer = DMLTrainer(models, device='cuda', use_amp=True)
    
    # Train a bit
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)
    
    print("Training for 1 epoch...")
    trainer.train_epoch(loader, epoch=1)
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'amp_checkpoint.pt')
        
        print(f"\nSaving checkpoint to {checkpoint_path}...")
        trainer.save_checkpoint(checkpoint_path)
        
        # Create new trainer
        new_models = [
            nn.Sequential(nn.Linear(10, 5)).cuda()
            for _ in range(2)
        ]
        new_trainer = DMLTrainer(new_models, device='cuda', use_amp=True)
        
        # Load checkpoint
        print("Loading checkpoint...")
        new_trainer.load_checkpoint(checkpoint_path)
        
        print("✓ Checkpoint loaded successfully!")
        print(f"  Epoch: {new_trainer.current_epoch}")
        print(f"  Global step: {new_trainer.global_step}")


def example_7_best_practices():
    """Example 7: Best practices for AMP usage."""
    print("\n" + "=" * 70)
    print("Example 7: AMP Best Practices")
    print("=" * 70)
    
    best_practices = """
Best Practices for Automatic Mixed Precision Training:

1. Auto-Detection (Recommended):
   ```python
   # AMP auto-enables for GPUs with compute capability >= 7.0
   trainer = DMLTrainer(models, device='cuda', use_amp=None)
   ```

2. Explicit Control:
   ```python
   # Force enable/disable
   trainer = DMLTrainer(models, device='cuda', use_amp=True)
   ```

3. Choose Right Data Type:
   ```python
   # FP16 - Faster, more memory efficient (default)
   trainer = DMLTrainer(models, use_amp=True, amp_dtype=torch.float16)
   
   # BF16 - Better numerical stability (Ampere+ GPUs)
   trainer = DMLTrainer(models, use_amp=True, amp_dtype=torch.bfloat16)
   ```

4. When to Use AMP:
   ✓ Large models (ResNet, Transformers, etc.)
   ✓ High-resolution images
   ✓ GPUs with Tensor Cores (V100, A100, RTX 20/30/40 series)
   ✓ Batch size limited by memory
   
   ✗ Small models (overhead may outweigh benefits)
   ✗ Models with numerical instability
   ✗ CPUs (AMP is CUDA-only)

5. Memory Savings:
   - FP16/BF16 uses ~50% less memory than FP32
   - Allows 2x larger batch sizes
   - Faster training (1.5-3x on Tensor Core GPUs)

6. Gradient Scaling:
   - Automatically handled by PyDML
   - Prevents underflow in FP16 gradients
   - No manual intervention needed

7. Checkpointing:
   ```python
   # AMP state is automatically saved
   trainer.save_checkpoint('checkpoint.pt')
   
   # And restored
   trainer.load_checkpoint('checkpoint.pt')
   ```

8. Monitoring:
   ```python
   # Check if AMP is active
   if trainer.use_amp:
       print(f"Using {trainer.amp_manager.config.dtype}")
   ```

9. Debugging AMP Issues:
   - If training diverges, try BFloat16
   - Check for NaN/Inf in gradients
   - Reduce initial scale if needed:
     ```python
     from pydml.utils.amp import AMPConfig
     config = AMPConfig(init_scale=2.**10)  # Lower scale
     ```

10. Combine with Other Optimizations:
    ```python
    trainer = DMLTrainer(
        models,
        device='cuda',
        use_amp=True,  # Mixed precision
        # + gradient checkpointing in models
        # + larger batch size
        # + efficient data loading
    )
    ```

Performance Guidelines:
- V100/A100: Expect 2-3x speedup with FP16
- RTX 30/40: Expect 1.5-2x speedup with FP16
- GTX 10: Limited benefit (no Tensor Cores)
- CPU: AMP disabled automatically

Accuracy Impact:
- Usually <0.1% difference from FP32
- BFloat16 typically closer to FP32 than FP16
- For research: run with multiple seeds and report variance
    """
    
    print(best_practices)


if __name__ == "__main__":
    # Run all examples
    example_1_basic_amp()
    example_2_amp_config()
    example_3_performance_comparison()
    example_4_cifar_with_amp()
    example_5_bfloat16()
    example_6_checkpoint_with_amp()
    example_7_best_practices()
    
    print("\n" + "=" * 70)
    print("All AMP examples completed!")
    print("=" * 70)
