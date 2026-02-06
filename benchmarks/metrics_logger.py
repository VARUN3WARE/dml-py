"""
Metrics logging and tracking for benchmark experiments.

This module provides clean interfaces for tracking training metrics,
saving results, and generating comparison reports.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np


class MetricsLogger:
    """
    Logger for tracking and persisting experiment metrics.
    
    This class provides a clean interface for logging metrics during
    training and saving them in multiple formats (JSON, CSV) for
    analysis and comparison.
    
    Args:
        experiment_name: Unique identifier for this experiment
        output_dir: Directory where metrics will be saved
        save_frequency: How often to save metrics (in epochs)
        
    Attributes:
        metrics: Dictionary storing all logged metrics
        start_time: Timestamp when logging began
        
    Example:
        >>> logger = MetricsLogger('baseline_resnet32', 'results/')
        >>> logger.log_epoch(epoch=1, train_loss=2.3, val_acc=0.45)
        >>> logger.log_epoch(epoch=2, train_loss=1.8, val_acc=0.58)
        >>> logger.save()
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = 'benchmarks/results',
        save_frequency: int = 10
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.save_frequency = save_frequency
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: Dict[str, List[Any]] = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'time_elapsed': []
        }
        
        self.start_time = datetime.now()
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def log_epoch(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        learning_rate: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Log metrics for a single epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            learning_rate: Current learning rate
            **kwargs: Additional custom metrics to log
            
        Example:
            >>> logger.log_epoch(
            ...     epoch=5,
            ...     train_loss=1.5,
            ...     train_acc=0.65,
            ...     val_loss=1.8,
            ...     val_acc=0.62,
            ...     learning_rate=0.1
            ... )
        """
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss if train_loss is not None else np.nan)
        self.metrics['train_acc'].append(train_acc if train_acc is not None else np.nan)
        self.metrics['val_loss'].append(val_loss if val_loss is not None else np.nan)
        self.metrics['val_acc'].append(val_acc if val_acc is not None else np.nan)
        self.metrics['learning_rate'].append(learning_rate if learning_rate is not None else np.nan)
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.metrics['time_elapsed'].append(elapsed)
        
        # Track additional custom metrics
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = [np.nan] * (len(self.metrics['epoch']) - 1)
            self.metrics[key].append(value)
            
        # Update best accuracy
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            
        # Auto-save periodically
        if epoch % self.save_frequency == 0:
            self.save()
            
    def save(self) -> None:
        """
        Save all metrics to disk in JSON and CSV formats.
        
        Creates two files:
        - {experiment_name}_metrics.json: Complete metrics with metadata
        - {experiment_name}_metrics.csv: Tabular format for easy analysis
        
        Example:
            >>> logger.save()
        """
        timestamp = datetime.now().isoformat()
        
        # Prepare summary statistics
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': timestamp,
            'total_epochs': len(self.metrics['epoch']),
            'best_val_acc': float(self.best_val_acc),
            'best_epoch': int(self.best_epoch),
            'total_time_seconds': self.metrics['time_elapsed'][-1] if self.metrics['time_elapsed'] else 0,
            'metrics': self.metrics
        }
        
        # Save JSON
        json_path = self.output_dir / f"{self.experiment_name}_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Save CSV
        csv_path = self.output_dir / f"{self.experiment_name}_metrics.csv"
        if self.metrics['epoch']:
            self._save_csv(csv_path)
            
    def _save_csv(self, filepath: Path) -> None:
        """Save metrics in CSV format."""
        metric_names = list(self.metrics.keys())
        num_rows = len(self.metrics['epoch'])
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metric_names)
            
            for i in range(num_rows):
                row = [self.metrics[name][i] for name in metric_names]
                writer.writerow(row)
                
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for this experiment.
        
        Returns:
            Dictionary containing key performance metrics
            
        Example:
            >>> summary = logger.get_summary()
            >>> print(f"Best accuracy: {summary['best_val_acc']:.2%}")
        """
        if not self.metrics['epoch']:
            return {}
            
        val_accs = [a for a in self.metrics['val_acc'] if not np.isnan(a)]
        train_accs = [a for a in self.metrics['train_acc'] if not np.isnan(a)]
        
        return {
            'experiment_name': self.experiment_name,
            'total_epochs': len(self.metrics['epoch']),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'final_val_acc': val_accs[-1] if val_accs else 0.0,
            'final_train_acc': train_accs[-1] if train_accs else 0.0,
            'total_time_seconds': self.metrics['time_elapsed'][-1] if self.metrics['time_elapsed'] else 0,
            'avg_epoch_time': self.metrics['time_elapsed'][-1] / len(self.metrics['epoch']) if self.metrics['epoch'] else 0
        }
        
    def __str__(self) -> str:
        """Return formatted summary string."""
        summary = self.get_summary()
        if not summary:
            return f"MetricsLogger({self.experiment_name}): No data logged yet"
            
        lines = [f"Experiment: {self.experiment_name}"]
        lines.append(f"  Epochs: {summary['total_epochs']}")
        lines.append(f"  Best Val Acc: {summary['best_val_acc']:.2%} (epoch {summary['best_epoch']})")
        lines.append(f"  Final Val Acc: {summary['final_val_acc']:.2%}")
        lines.append(f"  Total Time: {summary['total_time_seconds']:.1f}s")
        lines.append(f"  Avg Time/Epoch: {summary['avg_epoch_time']:.1f}s")
        return '\n'.join(lines)


def compare_experiments(experiment_names: List[str], results_dir: str = 'benchmarks/results') -> Dict[str, Any]:
    """
    Compare results from multiple experiments.
    
    Args:
        experiment_names: List of experiment names to compare
        results_dir: Directory containing experiment results
        
    Returns:
        Dictionary with comparison statistics
        
    Example:
        >>> comparison = compare_experiments([
        ...     'baseline_resnet32',
        ...     'dml_2resnet',
        ...     'dml_3mixed'
        ... ])
        >>> print(comparison['ranking'])
    """
    results_path = Path(results_dir)
    experiments = []
    
    for name in experiment_names:
        json_path = results_path / f"{name}_metrics.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                experiments.append({
                    'name': name,
                    'best_val_acc': data['best_val_acc'],
                    'best_epoch': data['best_epoch'],
                    'total_time': data['total_time_seconds']
                })
                
    if not experiments:
        return {}
        
    # Sort by best validation accuracy
    experiments.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    return {
        'ranking': experiments,
        'best_experiment': experiments[0]['name'],
        'best_accuracy': experiments[0]['best_val_acc'],
        'accuracy_spread': experiments[0]['best_val_acc'] - experiments[-1]['best_val_acc']
    }
