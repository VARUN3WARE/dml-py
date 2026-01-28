"""
Reproducibility utilities for PyDML.

This module provides tools to ensure reproducible results across different runs.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    This function sets random seeds for Python's random module, NumPy, and PyTorch
    (both CPU and CUDA). It also configures PyTorch's CUDA backend for deterministic
    behavior if requested.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed to use across all libraries.
    deterministic : bool, default=True
        If True, sets PyTorch to use deterministic algorithms where possible.
        This may reduce performance but ensures reproducibility.
        
    Notes
    -----
    When deterministic=True, some PyTorch operations may be slower or throw errors
    if deterministic algorithms are not available. Set to False if you encounter
    issues or need maximum performance.
    
    Examples
    --------
    >>> from pydml.utils import set_seed
    >>> set_seed(42)  # All experiments will use seed 42
    >>> # Train your models - results will be reproducible
    
    >>> set_seed(123, deterministic=False)  # Faster but less reproducible
    """
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        if deterministic:
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Allow non-deterministic operations for better performance
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def get_random_state() -> dict:
    """
    Get current random state from all libraries.
    
    Returns
    -------
    dict
        Dictionary containing random states for Python, NumPy, and PyTorch.
        Can be used with set_random_state() to restore state later.
        
    Examples
    --------
    >>> state = get_random_state()
    >>> # Do some random operations
    >>> set_random_state(state)  # Restore previous state
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict):
    """
    Restore random state from a saved state dictionary.
    
    Parameters
    ----------
    state : dict
        Random state dictionary from get_random_state().
        
    Examples
    --------
    >>> state = get_random_state()
    >>> # Random operations here
    >>> set_random_state(state)  # Back to saved state
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


class ReproducibleContext:
    """
    Context manager for reproducible code blocks.
    
    Parameters
    ----------
    seed : int
        Random seed to use within the context.
    restore : bool, default=True
        If True, restores previous random state after exiting context.
        
    Examples
    --------
    >>> with ReproducibleContext(seed=42):
    ...     # This block will always produce same results
    ...     data = torch.randn(10, 10)
    ...     # Random state is restored after this block
    """
    
    def __init__(self, seed: int, restore: bool = True):
        self.seed = seed
        self.restore = restore
        self.saved_state = None
    
    def __enter__(self):
        if self.restore:
            self.saved_state = get_random_state()
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.restore and self.saved_state is not None:
            set_random_state(self.saved_state)
