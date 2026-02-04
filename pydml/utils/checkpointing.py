"""
Advanced checkpointing utilities for training.

Provides comprehensive checkpoint management including:
- Automatic periodic saving
- Best model tracking
- Resume training support
- Checkpoint rotation and cleanup
"""

import os
import glob
import shutil
from typing import Optional, Dict, Any, List
from pathlib import Path
import torch


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and cleanup.
    
    Features:
    - Automatic periodic checkpointing
    - Best model tracking
    - Keep only N best checkpoints
    - Resume training from latest or best checkpoint
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        max_to_keep: Maximum number of checkpoints to keep (None = keep all)
        keep_best: Whether to always keep the best checkpoint
        monitor: Metric to monitor for best model ('val_loss', 'val_acc', etc.)
        mode: 'min' to minimize monitor, 'max' to maximize
        
    Example:
        >>> manager = CheckpointManager(checkpoint_dir='checkpoints', max_to_keep=3)
        >>> manager.save(trainer, epoch=10, metrics={'val_loss': 0.5, 'val_acc': 92.0})
        >>> trainer = manager.load_best(trainer)
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        max_to_keep: Optional[int] = 5,
        keep_best: bool = True,
        monitor: str = 'val_loss',
        mode: str = 'min',
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoint = None
        self.checkpoints: List[Dict[str, Any]] = []
    
    def save(
        self,
        trainer: Any,
        epoch: int,
        metrics: Dict[str, float],
        is_best: Optional[bool] = None,
        prefix: str = 'checkpoint',
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            trainer: Trainer instance to save
            epoch: Current epoch number
            metrics: Dictionary of metrics
            is_best: Manually mark as best (overrides auto-detection)
            prefix: Filename prefix
            
        Returns:
            Path to saved checkpoint
        """
        # Determine if this is the best checkpoint
        if is_best is None and self.monitor in metrics:
            current_value = metrics[self.monitor]
            if self.mode == 'min':
                is_best = current_value < self.best_value
            else:
                is_best = current_value > self.best_value
            
            if is_best:
                self.best_value = current_value
        
        # Create checkpoint filename
        monitor_value = metrics.get(self.monitor, 0)
        filename = f"{prefix}_epoch{epoch:04d}_{self.monitor}{monitor_value:.4f}.pt"
        filepath = self.checkpoint_dir / filename
        
        # Update trainer epoch before saving
        trainer.current_epoch = epoch
        
        # Save checkpoint
        trainer.save_checkpoint(str(filepath))
        
        # Track checkpoint
        checkpoint_info = {
            'path': str(filepath),
            'epoch': epoch,
            'metrics': metrics.copy(),
            'is_best': is_best,
        }
        self.checkpoints.append(checkpoint_info)
        
        # Save as best if needed
        if is_best and self.keep_best:
            best_path = self.checkpoint_dir / f"{prefix}_best.pt"
            shutil.copy2(filepath, best_path)
            self.best_checkpoint = str(best_path)
            print(f"✓ Saved best model: {self.monitor} = {monitor_value:.4f}")
        
        # Cleanup old checkpoints
        if self.max_to_keep is not None:
            self._cleanup_checkpoints()
        
        return str(filepath)
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only max_to_keep recent ones."""
        if len(self.checkpoints) <= self.max_to_keep:
            return
        
        # Sort by metric (keep best ones)
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x['metrics'].get(self.monitor, float('inf')),
            reverse=(self.mode == 'max')
        )
        
        # Determine which to keep
        to_keep = set()
        
        # Always keep the best if enabled
        if self.keep_best and self.best_checkpoint:
            to_keep.add(self.best_checkpoint)
        
        # Keep the max_to_keep best checkpoints
        for ckpt in sorted_checkpoints[:self.max_to_keep]:
            to_keep.add(ckpt['path'])
        
        # Remove checkpoints not in keep set
        for ckpt in self.checkpoints[:]:
            if ckpt['path'] not in to_keep:
                try:
                    if os.path.exists(ckpt['path']):
                        os.remove(ckpt['path'])
                    self.checkpoints.remove(ckpt)
                except OSError as e:
                    print(f"Warning: Could not remove {ckpt['path']}: {e}")
    
    def load_latest(self, trainer: Any) -> Optional[int]:
        """
        Load the most recent checkpoint.
        
        Args:
            trainer: Trainer instance to load into
            
        Returns:
            Epoch number of loaded checkpoint, or None if no checkpoints found
        """
        checkpoints = self._find_checkpoints()
        if not checkpoints:
            print("No checkpoints found")
            return None
        
        # Get latest by modification time
        latest = max(checkpoints, key=lambda x: os.path.getmtime(x))
        
        print(f"Loading latest checkpoint: {latest}")
        trainer.load_checkpoint(latest)
        
        # Return the loaded epoch
        return trainer.current_epoch
    
    def load_best(self, trainer: Any) -> Optional[int]:
        """
        Load the best checkpoint.
        
        Args:
            trainer: Trainer instance to load into
            
        Returns:
            Epoch number of loaded checkpoint, or None if no best checkpoint
        """
        best_path = self.checkpoint_dir / 'checkpoint_best.pt'
        
        if not best_path.exists():
            # Try to find best from tracked checkpoints
            if not self.checkpoints:
                print("No best checkpoint found")
                return None
            
            sorted_ckpts = sorted(
                self.checkpoints,
                key=lambda x: x['metrics'].get(self.monitor, float('inf')),
                reverse=(self.mode == 'max')
            )
            best_path = sorted_ckpts[0]['path']
        
        print(f"Loading best checkpoint: {best_path}")
        trainer.load_checkpoint(str(best_path))
        
        # Return the loaded epoch
        return trainer.current_epoch
    
    def load_checkpoint(self, trainer: Any, checkpoint_path: str) -> Optional[int]:
        """
        Load a specific checkpoint.
        
        Args:
            trainer: Trainer instance to load into
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number of loaded checkpoint
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        print(f"Loading checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            return checkpoint.get('epoch', 0)
        except:
            return None
    
    def _find_checkpoints(self) -> List[str]:
        """Find all checkpoint files in the checkpoint directory."""
        return glob.glob(str(self.checkpoint_dir / 'checkpoint_*.pt'))
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with their information.
        
        Returns:
            List of dictionaries containing checkpoint information
        """
        checkpoints = []
        for filepath in self._find_checkpoints():
            try:
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                info = {
                    'path': filepath,
                    'epoch': checkpoint.get('epoch', 0),
                    'metrics': {
                        'train_loss': checkpoint.get('history', {}).get('train_loss', [])[-1] if checkpoint.get('history', {}).get('train_loss') else None,
                        'val_loss': checkpoint.get('history', {}).get('val_loss', [])[-1] if checkpoint.get('history', {}).get('val_loss') else None,
                        'val_acc': checkpoint.get('history', {}).get('val_acc', [])[-1] if checkpoint.get('history', {}).get('val_acc') else None,
                    },
                    'size_mb': os.path.getsize(filepath) / (1024 * 1024),
                }
                checkpoints.append(info)
            except:
                continue
        
        return sorted(checkpoints, key=lambda x: x['epoch'])
    
    def get_summary(self) -> str:
        """Get a summary of checkpoint status."""
        checkpoints = self.list_checkpoints()
        
        summary = f"\n{'='*60}\n"
        summary += "Checkpoint Summary\n"
        summary += f"{'='*60}\n"
        summary += f"Directory: {self.checkpoint_dir}\n"
        summary += f"Total checkpoints: {len(checkpoints)}\n"
        summary += f"Monitor metric: {self.monitor} ({self.mode})\n"
        
        if self.best_checkpoint and os.path.exists(self.best_checkpoint):
            summary += f"Best checkpoint: {os.path.basename(self.best_checkpoint)}\n"
            summary += f"Best {self.monitor}: {self.best_value:.4f}\n"
        
        if checkpoints:
            summary += f"\nRecent checkpoints:\n"
            for ckpt in checkpoints[-5:]:
                epoch = ckpt['epoch']
                val_loss = ckpt['metrics'].get('val_loss', 'N/A')
                val_acc = ckpt['metrics'].get('val_acc', 'N/A')
                size = ckpt.get('size_mb', 0)
                
                val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
                val_acc_str = f"{val_acc:.2f}%" if isinstance(val_acc, float) else str(val_acc)
                size_mb = size if size is not None else 0.0
                
                summary += f"  Epoch {epoch:3d}: val_loss={val_loss_str:8s} val_acc={val_acc_str:8s} ({size_mb:.1f} MB)\n"
        
        summary += f"{'='*60}\n"
        return summary


def auto_resume(trainer: Any, checkpoint_dir: str = 'checkpoints', resume_mode: str = 'latest') -> int:
    """
    Automatically resume training from a checkpoint if available.
    
    Args:
        trainer: Trainer instance
        checkpoint_dir: Directory containing checkpoints
        resume_mode: 'latest' or 'best'
        
    Returns:
        Starting epoch number (0 if no checkpoint found)
        
    Example:
        >>> trainer = DMLTrainer(models, device='cuda')
        >>> start_epoch = auto_resume(trainer, checkpoint_dir='checkpoints')
        >>> trainer.fit(train_loader, val_loader, epochs=100, start_epoch=start_epoch)
    """
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    if resume_mode == 'best':
        epoch = manager.load_best(trainer)
    else:
        epoch = manager.load_latest(trainer)
    
    if epoch is not None:
        print(f"✓ Resumed training from epoch {epoch}")
        return epoch + 1  # Start from next epoch
    else:
        print("No checkpoint found, starting from scratch")
        return 0


__all__ = [
    'CheckpointManager',
    'auto_resume',
]
