"""
Advanced Learning Rate Scheduling for PyTorch-DML.

This module provides high-level scheduling configurations and automatic
scheduler creation with warmup, best practices, and common training recipes.
"""

from typing import List, Optional, Union, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
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
    SequentialLR,
    ChainedScheduler,
)
import math


class SchedulerType(Enum):
    """Supported scheduler types."""
    STEP = "step"
    MULTISTEP = "multistep"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    COSINE_WARMRESTART = "cosine_warmrestart"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    ONE_CYCLE = "one_cycle"
    POLYNOMIAL = "polynomial"
    LINEAR = "linear"
    CONSTANT = "constant"


@dataclass
class WarmupConfig:
    """Configuration for learning rate warmup."""
    
    warmup_epochs: int = 5
    warmup_start_lr: float = 1e-6
    warmup_method: str = 'linear'  # 'linear', 'exponential', 'cosine'
    
    def __post_init__(self):
        """Validate configuration."""
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {self.warmup_epochs}")
        if self.warmup_start_lr <= 0:
            raise ValueError(f"warmup_start_lr must be positive, got {self.warmup_start_lr}")
        if self.warmup_method not in ['linear', 'exponential', 'cosine']:
            raise ValueError(f"warmup_method must be one of ['linear', 'exponential', 'cosine'], got '{self.warmup_method}'")


@dataclass
class SchedulerConfig:
    """
    High-level configuration for learning rate scheduling.
    
    This provides an easy way to configure schedulers without manually
    creating them, following best practices from research.
    
    Attributes:
        scheduler_type: Type of scheduler to use
        base_lr: Base learning rate (initial LR after warmup)
        warmup: Optional warmup configuration
        
        # StepLR parameters
        step_size: Epoch interval for StepLR
        gamma: Multiplicative factor for LR reduction
        
        # MultiStepLR parameters
        milestones: Epochs at which to reduce LR
        
        # CosineAnnealing parameters
        T_max: Maximum epochs for cosine annealing
        eta_min: Minimum learning rate for cosine
        
        # CosineWarmRestarts parameters
        T_0: Restart period for warm restarts
        T_mult: Period multiplier after restart
        
        # ReduceLROnPlateau parameters
        mode: 'min' for loss, 'max' for accuracy
        factor: LR reduction factor
        patience: Epochs to wait before reducing
        threshold: Threshold for measuring improvement
        
        # OneCycleLR parameters
        max_lr: Maximum LR for OneCycle
        total_steps: Total training steps
        pct_start: Percentage of cycle for LR increase
        
        # Polynomial/Linear parameters
        power: Power for polynomial decay
        final_lr: Final learning rate
    """
    
    scheduler_type: SchedulerType = SchedulerType.COSINE
    base_lr: float = 0.1
    warmup: Optional[WarmupConfig] = None
    
    # StepLR
    step_size: int = 30
    gamma: float = 0.1
    
    # MultiStepLR
    milestones: Optional[List[int]] = None
    
    # CosineAnnealing
    T_max: int = 200
    eta_min: float = 0.0
    
    # CosineWarmRestarts
    T_0: int = 10
    T_mult: int = 2
    
    # ReduceLROnPlateau
    mode: str = 'min'
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    
    # OneCycleLR
    max_lr: Optional[float] = None
    total_steps: Optional[int] = None
    pct_start: float = 0.3
    
    # Polynomial/Linear
    power: float = 1.0
    final_lr: float = 0.0
    
    def __post_init__(self):
        """Convert string to enum if needed."""
        if isinstance(self.scheduler_type, str):
            self.scheduler_type = SchedulerType(self.scheduler_type)
        
        # Set max_lr default
        if self.max_lr is None:
            self.max_lr = self.base_lr


def create_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_config: WarmupConfig,
    base_lr: float,
) -> LambdaLR:
    """
    Create a warmup scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_config: Warmup configuration
        base_lr: Target learning rate after warmup
        
    Returns:
        LambdaLR scheduler for warmup
    """
    warmup_epochs = warmup_config.warmup_epochs
    start_lr = warmup_config.warmup_start_lr
    method = warmup_config.warmup_method
    
    if method == 'linear':
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return start_lr / base_lr + (1.0 - start_lr / base_lr) * epoch / warmup_epochs
            return 1.0
    
    elif method == 'exponential':
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (start_lr / base_lr) ** (1.0 - epoch / warmup_epochs)
            return 1.0
    
    elif method == 'cosine':
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return start_lr / base_lr + (1.0 - start_lr / base_lr) * (
                    1.0 - math.cos(epoch / warmup_epochs * math.pi)
                ) / 2.0
            return 1.0
    
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def create_polynomial_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    power: float = 1.0,
    final_lr: float = 0.0,
) -> LambdaLR:
    """
    Create a polynomial learning rate scheduler.
    
    LR decreases from base_lr to final_lr using polynomial decay:
    lr = (base_lr - final_lr) * (1 - epoch/total_epochs)^power + final_lr
    
    Args:
        optimizer: Optimizer to schedule
        total_epochs: Total number of training epochs
        power: Power of polynomial (1.0 = linear)
        final_lr: Final learning rate
        
    Returns:
        LambdaLR scheduler
    """
    base_lr = optimizer.param_groups[0]['lr']
    
    def lr_lambda(epoch):
        if epoch >= total_epochs:
            return final_lr / base_lr
        return (1.0 - epoch / total_epochs) ** power * (1.0 - final_lr / base_lr) + final_lr / base_lr
    
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def create_scheduler_from_config(
    optimizer: optim.Optimizer,
    config: SchedulerConfig,
) -> Union[Any, SequentialLR]:
    """
    Create a scheduler from configuration.
    
    Automatically handles warmup if specified.
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        
    Returns:
        Configured scheduler (may be SequentialLR if warmup is used)
        
    Example:
        >>> config = SchedulerConfig(
        ...     scheduler_type=SchedulerType.COSINE,
        ...     base_lr=0.1,
        ...     T_max=200,
        ...     warmup=WarmupConfig(warmup_epochs=5)
        ... )
        >>> optimizer = optim.SGD(model.parameters(), lr=config.base_lr)
        >>> scheduler = create_scheduler_from_config(optimizer, config)
    """
    # Set the base learning rate
    for param_group in optimizer.param_groups:
        if config.warmup:
            param_group['lr'] = config.warmup.warmup_start_lr
        else:
            param_group['lr'] = config.base_lr
    
    # Create main scheduler
    if config.scheduler_type == SchedulerType.STEP:
        main_scheduler = StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    
    elif config.scheduler_type == SchedulerType.MULTISTEP:
        if config.milestones is None:
            raise ValueError("milestones must be specified for MultiStepLR scheduler, got None")
        main_scheduler = MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
        )
    
    elif config.scheduler_type == SchedulerType.EXPONENTIAL:
        main_scheduler = ExponentialLR(
            optimizer,
            gamma=config.gamma,
        )
    
    elif config.scheduler_type == SchedulerType.COSINE:
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
        )
    
    elif config.scheduler_type == SchedulerType.COSINE_WARMRESTART:
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
        )
    
    elif config.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
        main_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
        )
    
    elif config.scheduler_type == SchedulerType.ONE_CYCLE:
        if config.total_steps is None:
            raise ValueError("total_steps must be specified for OneCycleLR scheduler, got None")
        main_scheduler = OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            total_steps=config.total_steps,
            pct_start=config.pct_start,
        )
    
    elif config.scheduler_type == SchedulerType.POLYNOMIAL:
        main_scheduler = create_polynomial_scheduler(
            optimizer,
            total_epochs=config.T_max,
            power=config.power,
            final_lr=config.final_lr,
        )
    
    elif config.scheduler_type == SchedulerType.LINEAR:
        main_scheduler = create_polynomial_scheduler(
            optimizer,
            total_epochs=config.T_max,
            power=1.0,
            final_lr=config.final_lr,
        )
    
    elif config.scheduler_type == SchedulerType.CONSTANT:
        # No scheduling, constant LR
        main_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    else:
        raise ValueError(f"unsupported scheduler type '{config.scheduler_type}', must be one of {[t.value for t in SchedulerType]}")
    
    # Add warmup if specified
    if config.warmup and config.warmup.warmup_epochs > 0:
        warmup_scheduler = create_warmup_scheduler(
            optimizer,
            config.warmup,
            config.base_lr,
        )
        
        # Combine warmup and main scheduler
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.warmup.warmup_epochs],
        )
    
    return main_scheduler


def create_schedulers_from_config(
    optimizers: List[optim.Optimizer],
    config: SchedulerConfig,
) -> List[Any]:
    """
    Create schedulers for all optimizers from a single configuration.
    
    Args:
        optimizers: List of optimizers
        config: Scheduler configuration (applied to all optimizers)
        
    Returns:
        List of configured schedulers
        
    Example:
        >>> optimizers = [optim.SGD(m.parameters(), lr=0.1) for m in models]
        >>> config = SchedulerConfig(scheduler_type='cosine', T_max=200)
        >>> schedulers = create_schedulers_from_config(optimizers, config)
    """
    return [create_scheduler_from_config(opt, config) for opt in optimizers]


# Common training recipes
def get_cifar_schedule(
    optimizers: List[optim.Optimizer],
    total_epochs: int = 200,
    warmup_epochs: int = 5,
) -> List[Any]:
    """
    Get standard CIFAR training schedule.
    
    Uses cosine annealing with warmup, following best practices
    from recent research.
    
    Args:
        optimizers: List of optimizers (should use SGD with lr=0.1)
        total_epochs: Total training epochs
        warmup_epochs: Warmup epochs
        
    Returns:
        List of configured schedulers
    """
    config = SchedulerConfig(
        scheduler_type=SchedulerType.COSINE,
        base_lr=0.1,
        T_max=total_epochs,
        eta_min=0.0,
        warmup=WarmupConfig(
            warmup_epochs=warmup_epochs,
            warmup_start_lr=1e-6,
            warmup_method='linear',
        ),
    )
    return create_schedulers_from_config(optimizers, config)


def get_imagenet_schedule(
    optimizers: List[optim.Optimizer],
    total_epochs: int = 90,
) -> List[Any]:
    """
    Get standard ImageNet training schedule.
    
    Uses multistep LR with drops at 30, 60, and 80 epochs.
    
    Args:
        optimizers: List of optimizers (should use SGD with lr=0.1)
        total_epochs: Total training epochs
        
    Returns:
        List of configured schedulers
    """
    # Calculate milestones as fractions of total epochs
    milestones = [int(total_epochs * 0.33), int(total_epochs * 0.67), int(total_epochs * 0.89)]
    
    config = SchedulerConfig(
        scheduler_type=SchedulerType.MULTISTEP,
        base_lr=0.1,
        milestones=milestones,
        gamma=0.1,
        warmup=WarmupConfig(warmup_epochs=5),
    )
    return create_schedulers_from_config(optimizers, config)


def get_fine_tuning_schedule(
    optimizers: List[optim.Optimizer],
    total_epochs: int = 50,
) -> List[Any]:
    """
    Get schedule for fine-tuning pretrained models.
    
    Uses smaller learning rate with gentle cosine decay.
    
    Args:
        optimizers: List of optimizers (should use Adam/AdamW with lr=1e-4)
        total_epochs: Total fine-tuning epochs
        
    Returns:
        List of configured schedulers
    """
    config = SchedulerConfig(
        scheduler_type=SchedulerType.COSINE,
        base_lr=1e-4,
        T_max=total_epochs,
        eta_min=1e-6,
        warmup=WarmupConfig(
            warmup_epochs=3,
            warmup_start_lr=1e-6,
            warmup_method='linear',
        ),
    )
    return create_schedulers_from_config(optimizers, config)


__all__ = [
    'SchedulerType',
    'WarmupConfig',
    'SchedulerConfig',
    'create_warmup_scheduler',
    'create_polynomial_scheduler',
    'create_scheduler_from_config',
    'create_schedulers_from_config',
    'get_cifar_schedule',
    'get_imagenet_schedule',
    'get_fine_tuning_schedule',
]
