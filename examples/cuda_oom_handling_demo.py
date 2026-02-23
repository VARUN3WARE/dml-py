"""
CUDA Memory Management and OOM Handling Examples

This script demonstrates how to use PyDML's CUDA memory management utilities
to handle out-of-memory errors gracefully and monitor GPU memory usage.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pydml.utils.cuda_memory import (
    get_gpu_memory_info,
    clear_cuda_cache,
    print_memory_summary,
    handle_oom,
    safe_forward,
    AutoBatchSizeReducer,
    MemoryMonitor,
    CUDAOutOfMemoryError
)


def example_1_check_memory():
    """Example 1: Check GPU memory usage."""
    print("=" * 70)
    print("Example 1: Check GPU Memory Usage")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping GPU memory examples.")
        return
    
    # Get memory info
    info = get_gpu_memory_info()
    print(f"Allocated: {info['allocated']:.2f} GB")
    print(f"Reserved:  {info['reserved']:.2f} GB")
    print(f"Free:      {info['free']:.2f} GB")
    print(f"Total:     {info['total']:.2f} GB")
    print(f"Usage:     {(info['allocated']/info['total']*100):.1f}%")
    
    # Or use the pretty print function
    print_memory_summary()


def example_2_handle_oom_decorator():
    """Example 2: Use @handle_oom decorator."""
    print("=" * 70)
    print("Example 2: Handle OOM with Decorator")
    print("=" * 70)
    
    @handle_oom
    def train_step(model, data, target):
        """Training step with automatic OOM handling."""
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        return loss.item()
    
    # Create a simple model
    model = nn.Linear(100, 10)
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Create dummy data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.randn(32, 100, device=device)
    target = torch.randint(0, 10, (32,), device=device)
    
    # Train step will handle OOM gracefully
    try:
        loss = train_step(model, data, target)
        print(f"Loss: {loss:.4f}")
        print("Training step completed successfully!")
    except CUDAOutOfMemoryError as e:
        print(f"OOM Error caught: {e}")


def example_3_safe_forward():
    """Example 3: Use safe_forward for model inference."""
    print("=" * 70)
    print("Example 3: Safe Forward Pass")
    print("=" * 70)
    
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    data = torch.randn(16, 100, device=device)
    
    # safe_forward automatically handles OOM and retries after cache clear
    try:
        output = safe_forward(model, data)
        print(f"Output shape: {output.shape}")
        print("Forward pass completed successfully!")
    except CUDAOutOfMemoryError as e:
        print(f"OOM Error: {e}")


def example_4_auto_batch_size_reduction():
    """Example 4: Automatic batch size reduction on OOM."""
    print("=" * 70)
    print("Example 4: Automatic Batch Size Reduction")
    print("=" * 70)
    
    # Create a simple model
    model = nn.Linear(1000, 100)
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Create batch size reducer
    reducer = AutoBatchSizeReducer(
        initial_batch_size=1024,
        min_batch_size=32,
        reduction_factor=0.5
    )
    
    print(f"Starting batch size: {reducer.get_batch_size()}")
    
    # Simulate training loop with OOM handling
    success = False
    while reducer.can_reduce() and not success:
        batch_size = reducer.get_batch_size()
        
        try:
            # Try to train with current batch size
            data = torch.randn(batch_size, 1000, device=device)
            output = model(data)
            loss = output.sum()
            loss.backward()
            
            print(f" Training successful with batch size: {batch_size}")
            success = True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f" OOM with batch size {batch_size}, reducing...")
                reducer.reduce()
                clear_cuda_cache()
            else:
                raise
    
    if not success:
        print(f"Could not train even with minimum batch size {reducer.min_batch_size}")


def example_5_memory_monitor():
    """Example 5: Monitor memory during training."""
    print("=" * 70)
    print("Example 5: Memory Monitoring")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory monitoring example")
        return
    
    # Create memory monitor
    monitor = MemoryMonitor(warning_threshold=0.7)  # Warn at 70% usage
    
    # Create model
    model = nn.Sequential(
        nn.Linear(500, 200),
        nn.ReLU(),
        nn.Linear(200, 100)
    ).cuda()
    
    print("Training with memory monitoring...")
    
    for step in range(5):
        # Check memory before step
        info = monitor.check()
        usage_pct = (info['allocated'] / info['total'] * 100) if info['total'] > 0 else 0
        print(f"Step {step}: GPU usage {usage_pct:.1f}%")
        
        # Simulate training
        data = torch.randn(64, 500, device='cuda')
        output = model(data)
        loss = output.sum()
        loss.backward()
        
        # Clean up
        del data, output, loss
    
    print(f"\nPeak memory usage: {monitor.get_peak_memory():.2f} GB")
    
    # Reset for next training phase
    monitor.reset_peak()
    print("Peak memory counter reset")


def example_6_full_training_with_oom_handling():
    """Example 6: Complete training loop with OOM handling."""
    print("=" * 70)
    print("Example 6: Complete Training with OOM Handling")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create synthetic dataset
    X = torch.randn(1000, 100)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # Setup batch size reducer
    reducer = AutoBatchSizeReducer(initial_batch_size=128, min_batch_size=16)
    
    # Setup memory monitor
    if torch.cuda.is_available():
        monitor = MemoryMonitor(warning_threshold=0.8)
    
    # Try different batch sizes until one works
    dataloader = None
    while reducer.can_reduce():
        batch_size = reducer.get_batch_size()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        try:
            print(f"\nAttempting training with batch size: {batch_size}")
            
            # Train for one epoch
            model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                # Monitor memory if using CUDA
                if torch.cuda.is_available():
                    monitor.check()
                
                optimizer.zero_grad()
                output = safe_forward(model, data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx >= 5:  # Just demo with a few batches
                    break
            
            avg_loss = total_loss / min(6, len(dataloader))
            print(f" Training completed successfully!")
            print(f"  Average loss: {avg_loss:.4f}")
            
            if torch.cuda.is_available():
                print(f"  Peak GPU memory: {monitor.get_peak_memory():.2f} GB")
            
            break  # Success!
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f" OOM occurred, reducing batch size...")
                reducer.reduce()
                clear_cuda_cache()
            else:
                raise
        except CUDAOutOfMemoryError as e:
            print(f" OOM: {e}")
            reducer.reduce()
            clear_cuda_cache()



if __name__ == "__main__":
    # Run all examples
    example_1_check_memory()
    print("\n")
    
    example_2_handle_oom_decorator()
    print("\n")
    
    example_3_safe_forward()
    print("\n")
    
    example_4_auto_batch_size_reduction()
    print("\n")
    
    example_5_memory_monitor()
    print("\n")
    
    example_6_full_training_with_oom_handling()
    print("\n")
    print("All examples completed!")
    print("=" * 70)
