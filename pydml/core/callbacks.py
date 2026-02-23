"""
Callback system for training hooks.

This module provides a callback system for extending trainer functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Callback(ABC):
    """
    Abstract base class for callbacks.
    
    Callbacks can be used to execute custom code at different points during training.
    """
    
    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer: Any, batch: int) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, trainer: Any, batch: int, loss: float) -> None:
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when validation metric stops improving.
    
    Args:
        monitor: Metric to monitor (e.g., 'val_loss', 'val_acc')
        patience: Number of epochs to wait before stopping
        mode: 'min' for metrics to minimize, 'max' for metrics to maximize
        min_delta: Minimum change to qualify as improvement
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0,
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Check if training should stop."""
        if self.monitor not in metrics:
            return
        
        current_value = metrics[self.monitor]
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best {self.monitor}: {self.best_value:.4f}")
                # Note: Actual stopping would require trainer support
                # This is a simplified version


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    Args:
        filepath: Path template for checkpoint files (can include {epoch}, {val_loss}, etc.)
        monitor: Metric to monitor for saving best model
        mode: 'min' for metrics to minimize, 'max' for metrics to maximize
        save_best_only: If True, only save when monitored metric improves
        save_freq: Save every N epochs (if save_best_only is False)
        verbose: Print messages when saving
    """
    
    def __init__(
        self,
        filepath: str = 'checkpoint_epoch_{epoch}.pt',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1,
        verbose: bool = True,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = None
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Save checkpoint if conditions are met."""
        should_save = False
        is_best = False
        
        if self.save_best_only:
            if self.monitor not in metrics:
                if self.verbose:
                    print(f"Warning: Metric '{self.monitor}' not found in metrics")
                return
            
            current_value = metrics[self.monitor]
            
            if self.mode == 'min':
                is_best = current_value < self.best_value
            else:
                is_best = current_value > self.best_value
            
            if is_best:
                self.best_value = current_value
                self.best_epoch = epoch
                should_save = True
        else:
            if epoch % self.save_freq == 0:
                should_save = True
        
        if should_save:
            filepath = self.filepath.format(epoch=epoch, **metrics)
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            trainer.save_checkpoint(filepath)
            
            if self.verbose:
                if is_best:
                    print(f" Saved best model to {filepath} ({self.monitor}: {self.best_value:.4f})")
                else:
                    print(f" Saved checkpoint to {filepath}")


class LearningRateLogger(Callback):
    """
    Log learning rates during training.
    """
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Log current learning rates."""
        lrs = [opt.param_groups[0]['lr'] for opt in trainer.optimizers]
        print(f"Learning rates: {lrs}")


class TensorBoardLogger(Callback):
    """
    Log metrics to TensorBoard.
    
    Args:
        log_dir: Directory to save TensorBoard logs
    """
    
    def __init__(self, log_dir: str = 'runs'):
        """
        Initialize TensorBoard callback for real-time training visualization.
        
        Creates a TensorBoard callback that logs training metrics in real-time.
        Metrics can be viewed in TensorBoard by running:
        tensorboard --logdir=runs
        
        Args:
            log_dir: Directory where TensorBoard logs will be saved (default: 'runs')
                    Each training run creates a subdirectory with a timestamp
                    
        Example:
            >>> from pydml.core.callbacks import TensorBoardCallback
            >>> callback = TensorBoardCallback(log_dir='tensorboard_logs')
            >>> trainer = DMLTrainer(models, callbacks=[callback])
            >>> trainer.fit(train_loader, val_loader, epochs=100)
            >>> # View in browser: tensorboard --logdir=tensorboard_logs
            
        Note:
            Requires tensorboard package: pip install tensorboard
            If tensorboard is not installed, the callback will print a warning
            and training will continue without logging.
        """
        self.log_dir = log_dir
        self.writer = None
    
    def on_train_begin(self, trainer: Any) -> None:
        """
        Initialize TensorBoard writer at the start of training.
        
        Creates a SummaryWriter that will log metrics to the specified directory.
        If tensorboard is not installed, prints a helpful error message.
        
        Args:
            trainer: The trainer instance (provides access to training state)
            
        Note:
            This method is called automatically by the trainer when training starts.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
            print(f"TensorBoard logging to {self.log_dir}")
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log training metrics to TensorBoard at the end of each epoch.
        
        Records all metrics (loss, accuracy, learning rate, etc.) as scalar values
        that can be visualized in TensorBoard's web interface.
        
        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            metrics: Dictionary of metric names to values, e.g.:
                    {'train_loss': 0.45, 'val_acc': 85.3, 'learning_rate': 0.01}
                    
        Note:
            This method is called automatically by the trainer at the end of each epoch.
            All metrics in the dictionary will appear as separate graphs in TensorBoard.
        """
        if self.writer is None:
            return
        
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)
    
    def on_train_end(self, trainer: Any) -> None:
        """
        Close TensorBoard writer and flush any remaining data.
        
        Ensures all logged data is written to disk before training ends.
        
        Args:
            trainer: The trainer instance
            
        Note:
            This method is called automatically by the trainer when training completes.
            Always flush and close the writer to ensure all data is saved properly.
        """
        if self.writer is not None:
            self.writer.close()
