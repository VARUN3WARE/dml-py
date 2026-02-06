"""
Data loading utilities for benchmarking experiments.

This module provides standardized data loading for CIFAR-10, CIFAR-100,
and other datasets used in benchmarking.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional
import numpy as np


def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    use_augmentation: bool = True,
    val_split: Optional[float] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Get CIFAR-10 data loaders with standard preprocessing.
    
    Args:
        data_dir: Directory to store/load dataset
        batch_size: Batch size for training and validation
        num_workers: Number of data loading workers
        use_augmentation: Whether to use data augmentation for training
        val_split: If provided, split training data (e.g., 0.1 for 10% validation)
        
    Returns:
        Tuple of (train_loader, test_loader, val_loader)
        If val_split is None, val_loader will be None
        
    Example:
        >>> train_loader, test_loader, _ = get_cifar10_loaders(
        ...     batch_size=128,
        ...     use_augmentation=True
        ... )
    """
    # CIFAR-10 normalization values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    # Training transforms
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Handle validation split if requested
    val_loader = None
    if val_split is not None:
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        
        split_idx = int(np.floor(val_split * num_train))
        train_idx, val_idx = indices[split_idx:], indices[:split_idx]
        
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, val_loader


def get_cifar100_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    use_augmentation: bool = True,
    val_split: Optional[float] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Get CIFAR-100 data loaders with standard preprocessing.
    
    Args:
        data_dir: Directory to store/load dataset
        batch_size: Batch size for training and validation
        num_workers: Number of data loading workers
        use_augmentation: Whether to use data augmentation for training
        val_split: If provided, split training data (e.g., 0.1 for 10% validation)
        
    Returns:
        Tuple of (train_loader, test_loader, val_loader)
        If val_split is None, val_loader will be None
        
    Example:
        >>> train_loader, test_loader, _ = get_cifar100_loaders(
        ...     batch_size=128,
        ...     use_augmentation=True
        ... )
    """
    # CIFAR-100 normalization values
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    # Training transforms
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # Test transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Handle validation split if requested
    val_loader = None
    if val_split is not None:
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        
        split_idx = int(np.floor(val_split * num_train))
        train_idx, val_idx = indices[split_idx:], indices[:split_idx]
        
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, val_loader


def get_dataset_info(dataset_name: str) -> dict:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of dataset ('cifar10', 'cifar100', etc.)
        
    Returns:
        Dictionary with dataset information
        
    Example:
        >>> info = get_dataset_info('cifar10')
        >>> print(info['num_classes'])
        10
    """
    dataset_info = {
        'cifar10': {
            'num_classes': 10,
            'input_size': (3, 32, 32),
            'train_samples': 50000,
            'test_samples': 10000,
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010)
        },
        'cifar100': {
            'num_classes': 100,
            'input_size': (3, 32, 32),
            'train_samples': 50000,
            'test_samples': 10000,
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761)
        }
    }
    
    return dataset_info.get(dataset_name.lower(), {})
