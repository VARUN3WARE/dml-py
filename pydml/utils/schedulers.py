"""
Learning Rate Scheduler utilities for PyTorch-DML.

This module provides helper functions to create and manage learning rate schedulers
for collaborative learning trainers.
"""

from typing import List, Optional, Union, Dict, Any
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    OneCycleLR,
    LambdaLR,
)


def create_step_schedulers(
    optimizers: List[optim.Optimizer],
    step_size: int = 30,
    gamma: float = 0.1,
) -> List[StepLR]:
    """
    Create StepLR schedulers for all optimizers.
    
    Decays the learning rate by gamma every step_size epochs.
    
    Args:
        optimizers: List of optimizers
        step_size: Period of learning rate decay (default: 30)
        gamma: Multiplicative factor of learning rate decay (default: 0.1)
    
    Returns:
        List of StepLR schedulers
    
    Example:
        >>> optimizers = [Adam(model.parameters()) for model in models]
        >>> schedulers = create_step_schedulers(optimizers, step_size=30, gamma=0.1)
        >>> trainer = DMLTrainer(models, optimizers=optimizers, schedulers=schedulers)
    """
    return [StepLR(opt, step_size=step_size, gamma=gamma) for opt in optimizers]


def create_multistep_schedulers(
    optimizers: List[optim.Optimizer],
    milestones: List[int],
    gamma: float = 0.1,
) -> List[MultiStepLR]:
    """
    Create MultiStepLR schedulers for all optimizers.
    
    Decays the learning rate by gamma at specified milestones.
    
    Args:
        optimizers: List of optimizers
        milestones: List of epoch indices where LR is decayed
        gamma: Multiplicative factor of learning rate decay (default: 0.1)
    
    Returns:
        List of MultiStepLR schedulers
    
    Example:
        >>> schedulers = create_multistep_schedulers(optimizers, milestones=[50, 100, 150])
    """
    return [MultiStepLR(opt, milestones=milestones, gamma=gamma) for opt in optimizers]


def create_cosine_schedulers(
    optimizers: List[optim.Optimizer],
    T_max: int,
    eta_min: float = 0,
) -> List[CosineAnnealingLR]:
    """
    Create CosineAnnealingLR schedulers for all optimizers.
    
    Sets the learning rate using a cosine annealing schedule.
    
    Args:
        optimizers: List of optimizers
        T_max: Maximum number of iterations (usually total epochs)
        eta_min: Minimum learning rate (default: 0)
    
    Returns:
        List of CosineAnnealingLR schedulers
    
    Example:
        >>> # For 200 epochs of training
        >>> schedulers = create_cosine_schedulers(optimizers, T_max=200)
    """
    return [CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min) for opt in optimizers]


def create_cosine_warmrestart_schedulers(
    optimizers: List[optim.Optimizer],
    T_0: int,
    T_mult: int = 1,
    eta_min: float = 0,
) -> List[CosineAnnealingWarmRestarts]:
    """
    Create CosineAnnealingWarmRestarts schedulers for all optimizers.
    
    Sets the learning rate using a cosine annealing schedule with warm restarts.
    
    Args:
        optimizers: List of optimizers
        T_0: Number of iterations for the first restart
        T_mult: Factor to increase T_i after each restart (default: 1)
        eta_min: Minimum learning rate (default: 0)
    
    Returns:
        List of CosineAnnealingWarmRestarts schedulers
    
    Example:
        >>> # Restart every 10 epochs
        >>> schedulers = create_cosine_warmrestart_schedulers(optimizers, T_0=10)
    """
    return [
        CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        for opt in optimizers
    ]


def create_exponential_schedulers(
    optimizers: List[optim.Optimizer],
    gamma: float = 0.95,
) -> List[ExponentialLR]:
    """
    Create ExponentialLR schedulers for all optimizers.
    
    Decays the learning rate by gamma every epoch.
    
    Args:
        optimizers: List of optimizers
        gamma: Multiplicative factor of learning rate decay (default: 0.95)
    
    Returns:
        List of ExponentialLR schedulers
    
    Example:
        >>> schedulers = create_exponential_schedulers(optimizers, gamma=0.95)
    """
    return [ExponentialLR(opt, gamma=gamma) for opt in optimizers]


def create_reduce_on_plateau_schedulers(
    optimizers: List[optim.Optimizer],
    mode: str = 'min',
    factor: float = 0.1,
    patience: int = 10,
    threshold: float = 1e-4,
) -> List[ReduceLROnPlateau]:
    """
    Create ReduceLROnPlateau schedulers for all optimizers.
    
    Reduces learning rate when a metric has stopped improving.
    Note: This scheduler requires manual stepping with the metric value.
    
    Args:
        optimizers: List of optimizers
        mode: 'min' for loss, 'max' for accuracy (default: 'min')
        factor: Factor by which LR is reduced (default: 0.1)
        patience: Number of epochs with no improvement before reducing LR (default: 10)
        threshold: Threshold for measuring improvement (default: 1e-4)
    
    Returns:
        List of ReduceLROnPlateau schedulers
    
    Example:
        >>> schedulers = create_reduce_on_plateau_schedulers(optimizers, patience=10)
        >>> # During training, manually step with validation loss:
        >>> for scheduler in schedulers:
        >>>     scheduler.step(val_loss)
    """
    return [
        ReduceLROnPlateau(
            opt,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
        )
        for opt in optimizers
    ]


def get_scheduler_info(schedulers: List[Any]) -> Dict[str, Any]:
    """
    Get information about schedulers.
    
    Args:
        schedulers: List of schedulers
    
    Returns:
        Dictionary with scheduler information
    """
    if not schedulers:
        return {'num_schedulers': 0, 'types': []}
    
    info = {
        'num_schedulers': len(schedulers),
        'types': [type(s).__name__ for s in schedulers],
        'current_lrs': []
    }
    
    # Get current learning rates
    for scheduler in schedulers:
        if hasattr(scheduler, 'get_last_lr'):
            info['current_lrs'].append(scheduler.get_last_lr())
        elif hasattr(scheduler, 'optimizer'):
            info['current_lrs'].append([group['lr'] for group in scheduler.optimizer.param_groups])
    
    return info


def validate_schedulers(
    schedulers: List[Any],
    optimizers: List[optim.Optimizer],
) -> bool:
    """
    Validate that schedulers are compatible with optimizers.
    
    Args:
        schedulers: List of schedulers
        optimizers: List of optimizers
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not schedulers:
        return True
    
    if len(schedulers) != len(optimizers):
        raise ValueError(
            f"Number of schedulers ({len(schedulers)}) must match "
            f"number of optimizers ({len(optimizers)})"
        )
    
    # Check each scheduler has corresponding optimizer
    for i, (scheduler, optimizer) in enumerate(zip(schedulers, optimizers)):
        if hasattr(scheduler, 'optimizer'):
            if scheduler.optimizer is not optimizer:
                raise ValueError(
                    f"Scheduler {i} is not associated with optimizer {i}. "
                    f"Make sure to create schedulers with the correct optimizers."
                )
    
    return True
