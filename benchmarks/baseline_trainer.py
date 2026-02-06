"""
Baseline trainer for single-model benchmarking.

This module provides a clean, efficient trainer for single model training
to establish baseline performance metrics for comparison with DML methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from tqdm import tqdm

from .experiment_config import ExperimentConfig
from .metrics_logger import MetricsLogger


class BaselineTrainer:
    """
    Trainer for single model baseline experiments.
    
    This class provides a clean interface for training a single model
    with standard supervised learning, serving as a baseline for
    comparing mutual learning methods.
    
    Args:
        model: PyTorch model to train
        config: Experiment configuration
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        logger: Optional metrics logger (created if not provided)
        
    Example:
        >>> from pydml.models.cifar import resnet32
        >>> model = resnet32(num_classes=10)
        >>> config = ExperimentConfig(
        ...     name='baseline_test',
        ...     dataset='cifar10',
        ...     model_type='resnet32'
        ... )
        >>> trainer = BaselineTrainer(model, config, train_loader, val_loader)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: Optional[MetricsLogger] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup logger
        self.logger = logger or MetricsLogger(
            experiment_name=config.name,
            output_dir='benchmarks/results'
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
    def _create_optimizer(self) -> optim.Optimizer:
        """
        Create optimizer based on configuration.
        
        Returns:
            Configured optimizer instance
        """
        if self.config.optimizer.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"unsupported optimizer: {self.config.optimizer}")
            
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler based on configuration.
        
        Returns:
            Configured scheduler instance
        """
        if self.config.scheduler.lower() == 'multistep':
            milestones = self.config.scheduler_params.get('milestones', [60, 120, 160])
            gamma = self.config.scheduler_params.get('gamma', 0.2)
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma
            )
        elif self.config.scheduler.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler.lower() == 'none':
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        else:
            raise ValueError(f"unsupported scheduler: {self.config.scheduler}")
            
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics (loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.epochs} [Train]",
            leave=False
        )
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/total:.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary with validation metrics (loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
        
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dictionary with final training results and best metrics
            
        Example:
            >>> results = trainer.train()
            >>> print(f"Best accuracy: {results['best_val_acc']:.2%}")
        """
        print(f"\nStarting training: {self.config.name}")
        print(self.config)
        print()
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_metrics['loss'],
                train_acc=train_metrics['accuracy'],
                val_loss=val_metrics['loss'],
                val_acc=val_metrics['accuracy'],
                learning_rate=current_lr
            )
            
            # Print progress
            print(f"Epoch {epoch:3d}/{self.config.epochs} | "
                  f"Train: {train_metrics['accuracy']:.2%} ({train_metrics['loss']:.3f}) | "
                  f"Val: {val_metrics['accuracy']:.2%} ({val_metrics['loss']:.3f}) | "
                  f"LR: {current_lr:.6f}")
                  
        # Final save
        self.logger.save()
        
        # Print summary
        print("\nTraining complete")
        print(self.logger)
        print()
        
        return self.logger.get_summary()
        
    def save_model(self, filepath: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path where model will be saved
            
        Example:
            >>> trainer.save_model('checkpoints/best_model.pth')
        """
        torch.save({
            'epoch': self.config.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'best_val_acc': self.logger.best_val_acc,
        }, filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Example:
            >>> trainer.load_model('checkpoints/best_model.pth')
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
