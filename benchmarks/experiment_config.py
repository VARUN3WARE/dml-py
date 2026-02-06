"""
Experiment configuration management for reproducible benchmarks.

This module provides a clean interface for defining, saving, and loading
experiment configurations to ensure reproducibility across runs.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch


@dataclass
class ExperimentConfig:
    """
    Configuration for a single benchmark experiment.
    
    This class encapsulates all hyperparameters and settings needed to
    reproduce a training run, including model architecture, training
    parameters, and hardware settings.
    
    Args:
        name: Unique identifier for this experiment
        dataset: Dataset name (e.g., 'cifar10', 'cifar100', 'imagenet')
        model_type: Model architecture (e.g., 'resnet32', 'mobilenet', 'wrn')
        num_models: Number of models for DML (1 for baseline)
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization coefficient
        momentum: SGD momentum parameter
        temperature: Temperature for distillation (if applicable)
        seed: Random seed for reproducibility
        device: Device to run on ('cuda' or 'cpu')
        num_workers: Number of data loading workers
        use_augmentation: Whether to apply data augmentation
        optimizer: Optimizer type ('sgd', 'adam', 'adamw')
        scheduler: Learning rate scheduler type
        scheduler_params: Parameters for the scheduler
        notes: Optional description or notes about this experiment
        
    Example:
        >>> config = ExperimentConfig(
        ...     name='baseline_resnet32_cifar10',
        ...     dataset='cifar10',
        ...     model_type='resnet32',
        ...     num_models=1,
        ...     epochs=200
        ... )
        >>> config.save('configs/baseline.json')
        >>> loaded = ExperimentConfig.load('configs/baseline.json')
    """
    
    name: str
    dataset: str
    model_type: str
    num_models: int = 1
    batch_size: int = 128
    epochs: int = 200
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    temperature: float = 3.0
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    use_augmentation: bool = True
    optimizer: str = 'sgd'
    scheduler: str = 'multistep'
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'milestones': [60, 120, 160],
        'gamma': 0.2
    })
    notes: str = ''
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_models < 1:
            raise ValueError(f"num_models must be >= 1, got {self.num_models}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
            
    def save(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path where config will be saved
            
        Example:
            >>> config.save('configs/experiment_001.json')
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to config file
            
        Returns:
            Loaded ExperimentConfig instance
            
        Example:
            >>> config = ExperimentConfig.load('configs/experiment_001.json')
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return asdict(self)
        
    def __str__(self) -> str:
        """Return formatted string representation."""
        lines = [f"Experiment: {self.name}"]
        lines.append(f"  Dataset: {self.dataset}")
        lines.append(f"  Model: {self.model_type} (x{self.num_models})")
        lines.append(f"  Training: {self.epochs} epochs, batch_size={self.batch_size}")
        lines.append(f"  Optimizer: {self.optimizer}, lr={self.learning_rate}")
        lines.append(f"  Device: {self.device}, seed={self.seed}")
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return '\n'.join(lines)


def create_baseline_configs(output_dir: str = 'benchmarks/configs') -> List[ExperimentConfig]:
    """
    Create standard baseline configurations for common experiments.
    
    This function generates a set of baseline configurations for
    benchmarking on CIFAR-10, useful for reproducible comparisons.
    
    Args:
        output_dir: Directory where configs will be saved
        
    Returns:
        List of created configurations
        
    Example:
        >>> configs = create_baseline_configs()
        >>> for config in configs:
        ...     print(config.name)
    """
    configs = [
        ExperimentConfig(
            name='baseline_resnet32_cifar10',
            dataset='cifar10',
            model_type='resnet32',
            num_models=1,
            notes='Single ResNet32 baseline for CIFAR-10'
        ),
        ExperimentConfig(
            name='baseline_mobilenet_cifar10',
            dataset='cifar10',
            model_type='mobilenet',
            num_models=1,
            notes='Single MobileNet baseline for CIFAR-10'
        ),
        ExperimentConfig(
            name='baseline_wrn_cifar10',
            dataset='cifar10',
            model_type='wrn-28-2',
            num_models=1,
            notes='Single WideResNet-28-2 baseline for CIFAR-10'
        ),
        ExperimentConfig(
            name='dml_2resnet_cifar10',
            dataset='cifar10',
            model_type='resnet32',
            num_models=2,
            notes='DML with 2 ResNet32 models on CIFAR-10'
        ),
        ExperimentConfig(
            name='dml_3mixed_cifar10',
            dataset='cifar10',
            model_type='mixed',
            num_models=3,
            notes='DML with 3 mixed models (ResNet32, MobileNet, WRN) on CIFAR-10'
        ),
    ]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for config in configs:
        config.save(output_path / f"{config.name}.json")
        
    return configs
