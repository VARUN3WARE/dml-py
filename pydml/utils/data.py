"""
Data loading utilities for DML-PY.

This module provides convenient functions for loading common datasets.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Optional

from pydml.utils.validation import (
    validate_positive_int,
    validate_probability,
    validate_num_workers,
    validate_batch_size,
)


def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.1,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-10 data loaders.
    
    Args:
        data_dir: Directory to store/load the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        val_split: Fraction of training data to use for validation (0.0 to 1.0)
        download: Whether to download the dataset if not found
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Raises:
        TypeError: If arguments have wrong types
        ValueError: If arguments have invalid values
    """
    # Validate inputs
    if not isinstance(data_dir, str):
        raise TypeError(f"data_dir must be a string, got {type(data_dir).__name__}")
    
    batch_size = validate_batch_size(batch_size)
    num_workers = validate_num_workers(num_workers)
    val_split = validate_probability(val_split, "val_split")
    
    if not isinstance(download, bool):
        raise TypeError(f"download must be a boolean, got {type(download).__name__}")
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for validation/test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=transform_test
    )
    
    # Split training set into train and validation
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        # Apply test transform to validation set
        val_dataset.dataset.transform = transform_test
    else:
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    else:
        val_loader = None
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_cifar100_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.1,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-100 data loaders.
    
    Args:
        data_dir: Directory to store/load the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        val_split: Fraction of training data to use for validation (0.0 to 1.0)
        download: Whether to download the dataset if not found
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Raises:
        TypeError: If arguments have wrong types
        ValueError: If arguments have invalid values
    """
    # Validate inputs
    if not isinstance(data_dir, str):
        raise TypeError(f"data_dir must be a string, got {type(data_dir).__name__}")
    
    batch_size = validate_batch_size(batch_size)
    num_workers = validate_num_workers(num_workers)
    val_split = validate_probability(val_split, "val_split")
    
    if not isinstance(download, bool):
        raise TypeError(f"download must be a boolean, got {type(download).__name__}")
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # No augmentation for validation/test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=download, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=download, transform=transform_test
    )
    
    # Split training set into train and validation
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        # Apply test transform to validation set
        val_dataset.dataset.transform = transform_test
    else:
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    else:
        val_loader = None
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_mnist_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST data loaders.
    
    Args:
        data_dir: Directory to store/load the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        download: Whether to download the dataset if not found
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=download, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=download, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader
