"""Tests for checkpoint and resume functionality."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
import shutil

from pydml.trainers.dml import DMLTrainer
from pydml.utils.checkpointing import CheckpointManager, auto_resume
from pydml.core.callbacks import ModelCheckpoint


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)


class TestBasicCheckpoint:
    """Test basic checkpoint save/load."""
    
    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        models = [SimpleModel() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
            trainer.save_checkpoint(checkpoint_path)
            
            assert os.path.exists(checkpoint_path)
    
    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        models = [SimpleModel() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        
        # Train a bit to change model weights
        X = torch.randn(50, 10)
        y = torch.randint(0, 5, (50,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)
        trainer.train_epoch(loader, epoch=1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
            
            # Save checkpoint
            trainer.save_checkpoint(checkpoint_path)
            original_epoch = trainer.current_epoch
            
            # Create new trainer and load
            new_models = [SimpleModel() for _ in range(2)]
            new_trainer = DMLTrainer(new_models, device='cpu')
            new_trainer.load_checkpoint(checkpoint_path)
            
            # Check epoch was restored
            assert new_trainer.current_epoch == original_epoch
    
    def test_checkpoint_contains_all_state(self):
        """Test that checkpoint contains all necessary state."""
        models = [SimpleModel() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
            trainer.save_checkpoint(checkpoint_path)
            
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            
            assert 'epoch' in checkpoint
            assert 'global_step' in checkpoint
            assert 'history' in checkpoint
            assert 'models' in checkpoint
            assert 'optimizers' in checkpoint
            assert 'schedulers' in checkpoint


class TestCheckpointManager:
    """Test CheckpointManager functionality."""
    
    def test_manager_initialization(self):
        """Test CheckpointManager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir, max_to_keep=3)
            
            assert manager.checkpoint_dir.exists()
            assert manager.max_to_keep == 3
            assert manager.monitor == 'val_loss'
            assert manager.mode == 'min'
    
    def test_save_checkpoint_with_manager(self):
        """Test saving checkpoints with manager."""
        models = [SimpleModel() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir, max_to_keep=3)
            
            metrics = {'val_loss': 1.5, 'val_acc': 75.0}
            filepath = manager.save(trainer, epoch=1, metrics=metrics)
            
            assert os.path.exists(filepath)
            assert 'checkpoint' in filepath
    
    def test_best_checkpoint_tracking(self):
        """Test that best checkpoint is tracked."""
        models = [SimpleModel() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                max_to_keep=3,
                keep_best=True,
                monitor='val_loss',
                mode='min'
            )
            
            # Save checkpoints with improving metric
            manager.save(trainer, epoch=1, metrics={'val_loss': 1.5, 'val_acc': 70.0})
            manager.save(trainer, epoch=2, metrics={'val_loss': 1.2, 'val_acc': 75.0})
            manager.save(trainer, epoch=3, metrics={'val_loss': 1.0, 'val_acc': 80.0})
            
            # Check best checkpoint exists
            best_path = os.path.join(tmpdir, 'checkpoint_best.pt')
            assert os.path.exists(best_path)
            assert manager.best_value == 1.0
    
    def test_checkpoint_cleanup(self):
        """Test that old checkpoints are cleaned up."""
        models = [SimpleModel() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir, max_to_keep=2)
            
            # Save more checkpoints than max_to_keep
            for i in range(5):
                manager.save(trainer, epoch=i+1, metrics={'val_loss': 1.5 - i*0.1, 'val_acc': 70.0 + i})
            
            # Count checkpoint files (excluding best)
            checkpoints = [f for f in os.listdir(tmpdir) if 'checkpoint_epoch' in f]
            assert len(checkpoints) <= 2  # Should only keep max_to_keep
    
    def test_load_latest(self):
        """Test loading latest checkpoint."""
        models = [SimpleModel() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            
            # Save multiple checkpoints
            manager.save(trainer, epoch=1, metrics={'val_loss': 1.5})
            manager.save(trainer, epoch=2, metrics={'val_loss': 1.3})
            manager.save(trainer, epoch=3, metrics={'val_loss': 1.1})
            
            # Load latest
            new_models = [SimpleModel() for _ in range(2)]
            new_trainer = DMLTrainer(new_models, device='cpu')
            epoch = manager.load_latest(new_trainer)
            
            assert epoch == 3
    
    def test_load_best(self):
        """Test loading best checkpoint."""
        models = [SimpleModel() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                keep_best=True,
                monitor='val_loss',
                mode='min'
            )
            
            # Save checkpoints (best is epoch 2)
            manager.save(trainer, epoch=1, metrics={'val_loss': 1.5})
            manager.save(trainer, epoch=2, metrics={'val_loss': 1.0})
            manager.save(trainer, epoch=3, metrics={'val_loss': 1.2})
            
            # Load best
            new_models = [SimpleModel() for _ in range(2)]
            new_trainer = DMLTrainer(new_models, device='cpu')
            epoch = manager.load_best(new_trainer)
            
            # Best checkpoint exists
            assert epoch is not None
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        models = [SimpleModel() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            
            # Save checkpoints
            for i in range(3):
                manager.save(trainer, epoch=i+1, metrics={'val_loss': 1.5 - i*0.1})
            
            # List checkpoints
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) >= 3
            assert all('epoch' in ckpt for ckpt in checkpoints)
    
    def test_get_summary(self):
        """Test checkpoint summary."""
        models = [SimpleModel() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            
            manager.save(trainer, epoch=1, metrics={'val_loss': 1.5})
            
            summary = manager.get_summary()
            assert 'Checkpoint Summary' in summary
            assert 'Total checkpoints' in summary


class TestAutoResume:
    """Test auto_resume functionality."""
    
    def test_auto_resume_latest(self):
        """Test auto-resuming from latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save checkpoint
            models = [SimpleModel() for _ in range(2)]
            trainer = DMLTrainer(models, device='cpu')
            
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            manager.save(trainer, epoch=5, metrics={'val_loss': 1.0})
            
            # Resume
            new_models = [SimpleModel() for _ in range(2)]
            new_trainer = DMLTrainer(new_models, device='cpu')
            start_epoch = auto_resume(new_trainer, checkpoint_dir=tmpdir, resume_mode='latest')
            
            assert start_epoch == 6  # Should start from next epoch
    
    def test_auto_resume_no_checkpoint(self):
        """Test auto-resume when no checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models = [SimpleModel() for _ in range(2)]
            trainer = DMLTrainer(models, device='cpu')
            
            start_epoch = auto_resume(trainer, checkpoint_dir=tmpdir)
            assert start_epoch == 0


class TestModelCheckpointCallback:
    """Test ModelCheckpoint callback."""
    
    def test_callback_save_best_only(self):
        """Test callback saves only best models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(
                filepath=os.path.join(tmpdir, 'best_model.pt'),
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=False
            )
            
            models = [SimpleModel() for _ in range(2)]
            trainer = DMLTrainer(models, device='cpu', callbacks=[callback])
            
            # Simulate epoch end with improving metric
            callback.on_epoch_end(trainer, epoch=1, metrics={'val_loss': 1.5})
            assert os.path.exists(os.path.join(tmpdir, 'best_model.pt'))
            
            # Worse metric shouldn't save
            os.remove(os.path.join(tmpdir, 'best_model.pt'))
            callback.on_epoch_end(trainer, epoch=2, metrics={'val_loss': 2.0})
            assert not os.path.exists(os.path.join(tmpdir, 'best_model.pt'))
    
    def test_callback_save_periodic(self):
        """Test callback saves periodically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(
                filepath=os.path.join(tmpdir, 'checkpoint_epoch_{epoch}.pt'),
                save_best_only=False,
                save_freq=2,  # Save every 2 epochs
                verbose=False
            )
            
            models = [SimpleModel() for _ in range(2)]
            trainer = DMLTrainer(models, device='cpu', callbacks=[callback])
            
            # Epoch 1 - shouldn't save
            callback.on_epoch_end(trainer, epoch=1, metrics={'val_loss': 1.0})
            assert not os.path.exists(os.path.join(tmpdir, 'checkpoint_epoch_1.pt'))
            
            # Epoch 2 - should save
            callback.on_epoch_end(trainer, epoch=2, metrics={'val_loss': 1.0})
            assert os.path.exists(os.path.join(tmpdir, 'checkpoint_epoch_2.pt'))


class TestResumeTraining:
    """Test resuming training from checkpoint."""
    
    def test_resume_training_continues_correctly(self):
        """Test that resumed training continues from correct epoch."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train for 3 epochs and save
            models = [SimpleModel() for _ in range(2)]
            trainer = DMLTrainer(models, device='cpu')
            trainer.fit(train_loader, epochs=3, verbose=False)
            
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
            trainer.save_checkpoint(checkpoint_path)
            
            # Load and continue training
            new_models = [SimpleModel() for _ in range(2)]
            new_trainer = DMLTrainer(new_models, device='cpu')
            new_trainer.load_checkpoint(checkpoint_path)
            
            assert new_trainer.current_epoch == 3
            
            # Continue training
            new_trainer.fit(train_loader, epochs=5, start_epoch=3, verbose=False)
            assert new_trainer.current_epoch == 5


class TestIntegration:
    """Integration tests for checkpoint system."""
    
    def test_full_workflow_with_checkpoint_manager(self):
        """Test complete workflow with CheckpointManager."""
        X = torch.randn(200, 10)
        y = torch.randint(0, 5, (200,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32)
        val_loader = DataLoader(dataset, batch_size=32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trainer and manager
            models = [SimpleModel() for _ in range(2)]
            trainer = DMLTrainer(models, device='cpu')
            manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                max_to_keep=2,
                monitor='val_loss',
                mode='min'
            )
            
            # Train for a few epochs with manual checkpointing
            for epoch in range(1, 4):
                trainer.train_epoch(train_loader, epoch=epoch)
                val_metrics = trainer.evaluate(val_loader)
                metrics = {'val_loss': val_metrics['val_loss'], 'val_acc': val_metrics['val_acc']}
                manager.save(trainer, epoch=epoch, metrics=metrics)
            
            # Resume from best
            new_models = [SimpleModel() for _ in range(2)]
            new_trainer = DMLTrainer(new_models, device='cpu')
            start_epoch = manager.load_best(new_trainer)
            
            assert start_epoch is not None
            
            # Get summary
            summary = manager.get_summary()
            assert len(summary) > 0
