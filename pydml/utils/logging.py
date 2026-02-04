"""
Logging utilities for DML-PY.

This module provides logging and experiment tracking utilities.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class ExperimentLogger:
    """
    Logger for tracking experiments and their results.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to save logs (default: 'experiments')
    """
    
    def __init__(self, experiment_name: str, log_dir: str = 'experiments'):
        """
        Initialize experiment logger for tracking training progress.
        
        Creates a timestamped directory for the experiment and initializes tracking
        for configuration, metrics, and checkpoints. All experiment data will be
        saved in a structured format for easy analysis.
        
        Args:
            experiment_name: Name of the experiment (used in directory naming)
            log_dir: Base directory for all experiments (default: 'experiments')
            
        Example:
            >>> logger = ExperimentLogger('my_dml_experiment')
            >>> logger.log_config({'learning_rate': 0.1, 'batch_size': 128})
            >>> for epoch in range(100):
            ...     logger.log_metrics(epoch, {'train_loss': 0.5, 'val_acc': 85.2})
            >>> logger.finalize()
        
        Note:
            The experiment directory will be named as: {log_dir}/{experiment_name}_{timestamp}
            This ensures each run has a unique directory even for the same experiment name.
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.start_time = time.time()
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.metrics_history = {}
        self.config = {}
        
        print(f"Experiment logger initialized: {self.exp_dir}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log experiment configuration to a JSON file.
        
        Saves all hyperparameters and configuration settings for the experiment.
        This is crucial for reproducibility and comparing different runs.
        
        Args:
            config: Dictionary containing all configuration parameters such as
                   learning rate, batch size, model architecture, etc.
                   
        Example:
            >>> config = {
            ...     'model': 'ResNet32',
            ...     'learning_rate': 0.1,
            ...     'batch_size': 128,
            ...     'temperature': 3.0,
            ...     'num_models': 3
            ... }
            >>> logger.log_config(config)
        
        Note:
            Configuration is saved as {exp_dir}/config.json
        """
        self.config = config
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log training metrics for a specific epoch.
        
        Records metrics like loss, accuracy, learning rate, etc. for later analysis.
        Metrics are accumulated in memory and can be saved to disk with save_metrics().
        
        Args:
            epoch: Current epoch number (0-indexed or 1-indexed based on your preference)
            metrics: Dictionary of metric names to values, e.g.:
                    {'train_loss': 0.45, 'val_loss': 0.52, 'val_acc': 85.3,
                     'learning_rate': 0.01}
                     
        Example:
            >>> for epoch in range(100):
            ...     train_loss = train_one_epoch(model, train_loader)
            ...     val_acc = evaluate(model, val_loader)
            ...     logger.log_metrics(epoch, {
            ...         'train_loss': train_loss,
            ...         'val_acc': val_acc,
            ...         'learning_rate': optimizer.param_groups[0]['lr']
            ...     })
        
        Note:
            All metric names should be consistent across epochs for proper tracking.
            Metrics are stored in memory until save_metrics() or finalize() is called.
        """
        if 'epochs' not in self.metrics_history:
            self.metrics_history['epochs'] = []
        
        self.metrics_history['epochs'].append(epoch)
        
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def log_model(self, model: torch.nn.Module, name: str = 'model') -> None:
        """
        Save model state dict as a checkpoint.
        
        Saves only the model parameters (state_dict), not the full model object.
        This is the recommended PyTorch approach for model persistence.
        
        Args:
            model: PyTorch model to save
            name: Name for the checkpoint file (default: 'model')
                 The .pt extension will be added automatically
                 
        Example:
            >>> # Save best model
            >>> if val_acc > best_val_acc:
            ...     logger.log_model(model, name='best_model')
            ...     
            >>> # Save checkpoint at specific epochs
            >>> if epoch % 10 == 0:
            ...     logger.log_model(model, name=f'checkpoint_epoch_{epoch}')
        
        Note:
            To load the model later:
            >>> model = MyModel()
            >>> model.load_state_dict(torch.load('path/to/model.pt'))
        """
        model_path = os.path.join(self.exp_dir, f'{name}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")
    
    def log_text(self, text: str, filename: str = 'notes.txt') -> None:
        """
        Log arbitrary text notes to a file.
        
        Appends timestamped text entries to a log file. Useful for recording
        observations, hyperparameter changes, or any other notes during training.
        
        Args:
            text: The text message to log
            filename: Name of the log file (default: 'notes.txt')
                     File is created in the experiment directory
                     
        Example:
            >>> logger.log_text("Started training with increased learning rate")
            >>> logger.log_text("Model seems to be overfitting after epoch 50")
            >>> logger.log_text("Checkpoint saved", filename='checkpoints.log')
            
        Note:
            Each entry is prefixed with the current timestamp.
            The file is appended to, so multiple calls add entries sequentially.
        """
        text_path = os.path.join(self.exp_dir, filename)
        with open(text_path, 'a') as f:
            f.write(f"[{datetime.now()}] {text}\n")
    
    def save_metrics(self) -> None:
        """
        Save all accumulated metrics to a JSON file.
        
        Writes the complete metrics history to disk. The metrics file contains
        all logged metrics organized by epoch, making it easy to analyze training
        progress or create custom visualizations.
        
        Example:
            >>> logger.save_metrics()
            >>> # Metrics saved to: {exp_dir}/metrics.json
            >>> 
            >>> # Load and analyze later:
            >>> import json
            >>> with open('experiments/my_exp_20260205_120000/metrics.json') as f:
            ...     metrics = json.load(f)
            >>> print(metrics['val_acc'])  # List of validation accuracies per epoch
            
        Note:
            Metrics are saved in the format:
            {
                'epochs': [0, 1, 2, ...],
                'train_loss': [0.8, 0.6, 0.5, ...],
                'val_acc': [70.1, 75.3, 78.2, ...],
                ...
            }
        """
        metrics_path = os.path.join(self.exp_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def finalize(self) -> None:
        """
        Finalize the experiment by saving all data and creating a summary.
        
        This method should be called at the end of training. It saves all metrics,
        creates a summary file with experiment statistics, and records the total
        training time.
        
        Example:
            >>> logger = ExperimentLogger('my_experiment')
            >>> logger.log_config(config)
            >>> # ... training loop ...
            >>> logger.finalize()
            >>> 
            >>> # Creates:
            >>> # - metrics.json (all training metrics)
            >>> # - summary.json (experiment statistics and timing)
            
        Note:
            The summary file includes:
            - Experiment name and directory
            - Total training time
            - Configuration used
            - Best performance metrics (if applicable)
        """
        elapsed_time = time.time() - self.start_time
        
        summary = {
            'experiment_name': self.experiment_name,
            'elapsed_time': elapsed_time,
            'config': self.config,
            'final_metrics': {
                key: values[-1] if values else None
                for key, values in self.metrics_history.items()
                if key != 'epochs'
            }
        }
        
        summary_path = os.path.join(self.exp_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.save_metrics()
        print(f"Experiment finalized. Total time: {elapsed_time/60:.2f} minutes")
        print(f"Results saved to: {self.exp_dir}")


class ConsoleLogger:
    """Simple console logger with formatting."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize console logger for printing training progress to terminal.
        
        Provides formatted console output during training with optional verbosity control.
        Less feature-rich than ExperimentLogger but useful for quick experiments.
        
        Args:
            verbose: If True, prints detailed progress information (default: True)
                    If False, only prints essential information
                    
        Example:
            >>> logger = ConsoleLogger(verbose=True)
            >>> logger.log_config({'learning_rate': 0.1, 'epochs': 100})
            >>> for epoch in range(100):
            ...     logger.log_metrics(epoch, {'train_loss': 0.5, 'val_acc': 85.2})
                    
        Note:
            This logger only prints to console and does not save any files.
            For persistent logging, use ExperimentLogger instead.
        """
        self.verbose = verbose
    
    def info(self, message: str) -> None:
        """Log info message."""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        if self.verbose:
            print(f"[WARNING] {message}")
    
    def error(self, message: str) -> None:
        """Log error message."""
        print(f"[ERROR] {message}")
    
    def success(self, message: str) -> None:
        """Log success message."""
        if self.verbose:
            print(f"[SUCCESS] {message}")
    
    def section(self, title: str, width: int = 60) -> None:
        """Print a section header."""
        if self.verbose:
            print("\n" + "=" * width)
            print(title.center(width))
            print("=" * width)
    
    def subsection(self, title: str, width: int = 60) -> None:
        """Print a subsection header."""
        if self.verbose:
            print("\n" + "-" * width)
            print(title)
            print("-" * width)


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 32, 32)):
    """
    Print a comprehensive summary of the model architecture.
    
    Displays the total number of parameters, trainable parameters, and model size
    in megabytes. Useful for understanding model complexity and memory requirements.
    
    Args:
        model: PyTorch model to summarize
        input_size: Expected input tensor shape including batch dimension
                   Format: (batch_size, channels, height, width) for images
                   or (batch_size, features) for vectors
                   Default: (1, 3, 32, 32) for CIFAR-sized images
                   
    Example:
        >>> from pydml.models.cifar import resnet32
        >>> model = resnet32(num_classes=10)
        >>> print_model_summary(model, input_size=(1, 3, 32, 32))
        
        Model Summary:
        ============================================================
        Total parameters: 464,154
        Trainable parameters: 464,154
        Non-trainable parameters: 0
        Model size: 1.77 MB
        ============================================================
    
    Note:
        Model size assumes float32 (4 bytes per parameter). If using float16/bfloat16,
        actual memory usage will be approximately half.
    """
    print("\nModel Summary:")
    print("=" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model size in MB
    param_size = total_params * 4 / (1024 ** 2)  # Assuming float32
    print(f"Model size: {param_size:.2f} MB")
    
    print("=" * 60)
