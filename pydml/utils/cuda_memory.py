"""
CUDA memory management and OOM error handling utilities.

This module provides tools for:
- Detecting and handling CUDA out-of-memory errors
- Memory monitoring and reporting
- Automatic gradient checkpointing
- Batch size reduction strategies
- Memory cleanup utilities
"""

import torch
import gc
import logging
from typing import Optional, Callable, Any, Dict
from functools import wraps

logger = logging.getLogger(__name__)


class CUDAOutOfMemoryError(Exception):
    """Custom exception for CUDA OOM errors with helpful recovery suggestions."""
    
    def __init__(self, message: str, suggestions: Optional[list] = None):
        self.suggestions = suggestions or []
        full_message = f"{message}\n\nRecovery suggestions:\n"
        for i, suggestion in enumerate(self.suggestions, 1):
            full_message += f"{i}. {suggestion}\n"
        super().__init__(full_message)


def get_gpu_memory_info(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get current GPU memory usage information.
    
    Args:
        device: CUDA device to query. If None, uses current device.
        
    Returns:
        Dictionary with memory info in GB:
        - allocated: Currently allocated memory
        - reserved: Reserved by PyTorch
        - free: Available memory
        - total: Total GPU memory
    """
    if not torch.cuda.is_available():
        return {
            'allocated': 0.0,
            'reserved': 0.0,
            'free': 0.0,
            'total': 0.0
        }
    
    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device = device.index if device.index is not None else 0
    
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    free = total - allocated
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': free,
        'total': total
    }


def clear_cuda_cache():
    """
    Clear CUDA cache and run garbage collection.
    
    This can help recover from OOM situations by freeing unused memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def print_memory_summary(device: Optional[torch.device] = None):
    """
    Print detailed GPU memory summary.
    
    Args:
        device: CUDA device to query. If None, uses current device.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    info = get_gpu_memory_info(device)
    print(f"\n{'='*60}")
    print("GPU Memory Summary")
    print(f"{'='*60}")
    print(f"Allocated: {info['allocated']:.2f} GB")
    print(f"Reserved:  {info['reserved']:.2f} GB")
    print(f"Free:      {info['free']:.2f} GB")
    print(f"Total:     {info['total']:.2f} GB")
    print(f"Usage:     {(info['allocated']/info['total']*100):.1f}%")
    print(f"{'='*60}\n")


def handle_oom(func: Callable) -> Callable:
    """
    Decorator to handle CUDA OOM errors gracefully.
    
    When OOM occurs:
    1. Clears CUDA cache
    2. Runs garbage collection
    3. Provides helpful error message with recovery suggestions
    
    Example:
        @handle_oom
        def train_step(model, batch):
            outputs = model(batch)
            loss = criterion(outputs)
            loss.backward()
            return loss
    
    Args:
        func: Function to wrap with OOM handling
        
    Returns:
        Wrapped function with OOM error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Clear cache and try to recover
                clear_cuda_cache()
                
                # Get memory info for error message
                mem_info = get_gpu_memory_info()
                
                suggestions = [
                    "Reduce batch size",
                    "Use gradient checkpointing (model.gradient_checkpointing_enable())",
                    "Reduce model size or use a smaller architecture",
                    "Use mixed precision training (torch.cuda.amp)",
                    "Reduce sequence length or input resolution",
                    f"Current GPU usage: {mem_info['allocated']:.2f}/{mem_info['total']:.2f} GB"
                ]
                
                raise CUDAOutOfMemoryError(
                    f"CUDA out of memory in {func.__name__}",
                    suggestions=suggestions
                ) from e
            else:
                raise
    
    return wrapper


class AutoBatchSizeReducer:
    """
    Automatically reduce batch size when OOM occurs.
    
    Example:
        reducer = AutoBatchSizeReducer(initial_batch_size=64, min_batch_size=4)
        
        while reducer.can_reduce():
            try:
                batch_size = reducer.get_batch_size()
                # Train with current batch size
                train(model, batch_size)
                break  # Success!
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    reducer.reduce()
                    clear_cuda_cache()
                else:
                    raise
    """
    
    def __init__(self, initial_batch_size: int, min_batch_size: int = 1, reduction_factor: float = 0.5):
        """
        Initialize batch size reducer.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            reduction_factor: Factor to multiply batch size by (e.g., 0.5 = halve)
        """
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.reduction_factor = reduction_factor
        self.attempts = 0
    
    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch_size
    
    def reduce(self):
        """Reduce batch size by reduction factor."""
        new_size = max(
            self.min_batch_size,
            int(self.current_batch_size * self.reduction_factor)
        )
        logger.warning(
            f"Reducing batch size from {self.current_batch_size} to {new_size} "
            f"due to OOM (attempt {self.attempts + 1})"
        )
        self.current_batch_size = new_size
        self.attempts += 1
    
    def can_reduce(self) -> bool:
        """Check if batch size can be reduced further."""
        return self.current_batch_size >= self.min_batch_size
    
    def reset(self):
        """Reset to initial batch size."""
        self.current_batch_size = self.initial_batch_size
        self.attempts = 0


def safe_forward(model: torch.nn.Module, *args, **kwargs) -> Any:
    """
    Safe forward pass with automatic OOM handling and recovery.
    
    Args:
        model: PyTorch model
        *args: Positional arguments for model forward
        **kwargs: Keyword arguments for model forward
        
    Returns:
        Model output
        
    Raises:
        CUDAOutOfMemoryError: If OOM cannot be recovered
    """
    try:
        return model(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            clear_cuda_cache()
            
            # Try again after clearing cache
            try:
                return model(*args, **kwargs)
            except RuntimeError as e2:
                if "out of memory" in str(e2).lower():
                    mem_info = get_gpu_memory_info()
                    suggestions = [
                        "Reduce batch size",
                        "Enable gradient checkpointing",
                        "Use smaller model architecture",
                        "Use mixed precision training",
                        f"GPU: {mem_info['allocated']:.2f}/{mem_info['total']:.2f} GB used"
                    ]
                    raise CUDAOutOfMemoryError(
                        "CUDA out of memory during forward pass",
                        suggestions=suggestions
                    ) from e2
                else:
                    raise
        else:
            raise


class MemoryMonitor:
    """
    Monitor GPU memory usage during training.
    
    Example:
        monitor = MemoryMonitor(warning_threshold=0.9)
        
        for batch in dataloader:
            monitor.check()  # Warns if memory usage > 90%
            outputs = model(batch)
    """
    
    def __init__(self, warning_threshold: float = 0.9, device: Optional[torch.device] = None):
        """
        Initialize memory monitor.
        
        Args:
            warning_threshold: Warn when memory usage exceeds this fraction (0-1)
            device: CUDA device to monitor
        """
        self.warning_threshold = warning_threshold
        self.device = device
        self.peak_memory = 0.0
    
    def check(self) -> Dict[str, float]:
        """
        Check current memory usage and warn if threshold exceeded.
        
        Returns:
            Dictionary with current memory info
        """
        info = get_gpu_memory_info(self.device)
        
        usage_fraction = info['allocated'] / info['total'] if info['total'] > 0 else 0
        
        if usage_fraction > self.warning_threshold:
            logger.warning(
                f"High GPU memory usage: {usage_fraction*100:.1f}% "
                f"({info['allocated']:.2f}/{info['total']:.2f} GB)"
            )
        
        self.peak_memory = max(self.peak_memory, info['allocated'])
        
        return info
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in GB."""
        return self.peak_memory
    
    def reset_peak(self):
        """Reset peak memory counter."""
        self.peak_memory = 0.0
