"""
Unit tests for learning rate schedulers.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, MultiStepLR
from pydml import DMLTrainer
from pydml.utils.schedulers import (
    create_step_schedulers,
    create_cosine_schedulers,
    create_multistep_schedulers,
    create_exponential_schedulers,
    create_cosine_warmrestart_schedulers,
    get_scheduler_info,
    validate_schedulers,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.fc(x)


class TestSchedulerUtils:
    """Test scheduler utility functions."""
    
    def test_create_step_schedulers(self):
        """Test creating StepLR schedulers."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.Adam(m.parameters(), lr=0.1) for m in models]
        
        schedulers = create_step_schedulers(optimizers, step_size=30, gamma=0.1)
        
        assert len(schedulers) == 2
        assert all(isinstance(s, StepLR) for s in schedulers)
        
        # Test scheduler steps
        initial_lr = optimizers[0].param_groups[0]['lr']
        
        # Simulate training for 29 epochs
        for _ in range(29):
            # Dummy training step
            loss = torch.tensor([1.0], requires_grad=True)
            optimizers[0].zero_grad()
            loss.backward()
            optimizers[0].step()
            schedulers[0].step()
        assert optimizers[0].param_groups[0]['lr'] == initial_lr
        
        # 30th epoch - LR should change
        loss = torch.tensor([1.0], requires_grad=True)
        optimizers[0].zero_grad()
        loss.backward()
        optimizers[0].step()
        schedulers[0].step()
        assert optimizers[0].param_groups[0]['lr'] == pytest.approx(initial_lr * 0.1)
    
    def test_create_cosine_schedulers(self):
        """Test creating CosineAnnealingLR schedulers."""
        models = [SimpleModel() for _ in range(3)]
        optimizers = [optim.Adam(m.parameters(), lr=0.1) for m in models]
        
        schedulers = create_cosine_schedulers(optimizers, T_max=100)
        
        assert len(schedulers) == 3
        assert all(isinstance(s, CosineAnnealingLR) for s in schedulers)
    
    def test_create_multistep_schedulers(self):
        """Test creating MultiStepLR schedulers."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.SGD(m.parameters(), lr=0.1) for m in models]
        
        schedulers = create_multistep_schedulers(optimizers, milestones=[50, 100, 150])
        
        assert len(schedulers) == 2
        assert all(isinstance(s, MultiStepLR) for s in schedulers)
    
    def test_validate_schedulers(self):
        """Test scheduler validation."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.Adam(m.parameters()) for m in models]
        schedulers = create_step_schedulers(optimizers)
        
        # Should pass with matching counts
        assert validate_schedulers(schedulers, optimizers)
        
        # Should fail with mismatched counts
        with pytest.raises(ValueError, match="must match"):
            validate_schedulers(schedulers[:1], optimizers)
    
    def test_get_scheduler_info(self):
        """Test getting scheduler information."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.Adam(m.parameters(), lr=0.01) for m in models]
        schedulers = create_step_schedulers(optimizers)
        
        info = get_scheduler_info(schedulers)
        
        assert info['num_schedulers'] == 2
        assert info['types'] == ['StepLR', 'StepLR']
        assert len(info['current_lrs']) == 2


class TestDMLWithSchedulers:
    """Test DML trainer with various schedulers."""
    
    def test_dml_with_step_scheduler(self):
        """Test DML with StepLR scheduler."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.Adam(m.parameters(), lr=0.1) for m in models]
        schedulers = create_step_schedulers(optimizers, step_size=5, gamma=0.5)
        
        trainer = DMLTrainer(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            device='cpu'
        )
        
        assert len(trainer.schedulers) == 2
        
        # Check initial learning rate
        initial_lrs = trainer.get_learning_rates()
        assert all(lr == 0.1 for lr in initial_lrs)
    
    def test_dml_with_cosine_scheduler(self):
        """Test DML with CosineAnnealingLR scheduler."""
        models = [SimpleModel() for _ in range(3)]
        optimizers = [optim.SGD(m.parameters(), lr=0.1) for m in models]
        schedulers = create_cosine_schedulers(optimizers, T_max=100)
        
        trainer = DMLTrainer(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            device='cpu'
        )
        
        assert len(trainer.schedulers) == 3
    
    def test_scheduler_validation_error(self):
        """Test that mismatched schedulers raise error."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.Adam(m.parameters()) for m in models]
        schedulers = create_step_schedulers(optimizers[:1])  # Only 1 scheduler
        
        with pytest.raises(ValueError, match="must match"):
            DMLTrainer(
                models=models,
                optimizers=optimizers,
                schedulers=schedulers,
                device='cpu'
            )
    
    def test_scheduler_stepping_during_training(self):
        """Test that schedulers step during training."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.Adam(m.parameters(), lr=0.1) for m in models]
        schedulers = create_step_schedulers(optimizers, step_size=2, gamma=0.5)
        
        trainer = DMLTrainer(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            device='cpu'
        )
        
        # Create dummy data
        train_data = [(torch.randn(4, 10), torch.randint(0, 10, (4,))) for _ in range(5)]
        
        # Get initial LR
        initial_lrs = trainer.get_learning_rates()
        
        # Train for 3 epochs (scheduler steps at epoch 2)
        for epoch in range(1, 4):
            trainer.train_epoch(train_data, epoch)
            for scheduler in trainer.schedulers:
                scheduler.step()
        
        # After 2 steps, LR should be reduced
        final_lrs = trainer.get_learning_rates()
        assert final_lrs[0] == pytest.approx(initial_lrs[0] * 0.5)
    
    def test_checkpoint_save_load_with_schedulers(self):
        """Test saving and loading checkpoints with schedulers."""
        import tempfile
        import os
        
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.Adam(m.parameters(), lr=0.1) for m in models]
        schedulers = create_step_schedulers(optimizers, step_size=5)
        
        trainer = DMLTrainer(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            device='cpu'
        )
        
        # Simulate training for 3 epochs
        for _ in range(3):
            # Dummy training step for each optimizer
            for optimizer in trainer.optimizers:
                loss = torch.tensor([1.0], requires_grad=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Now step schedulers
            for scheduler in trainer.schedulers:
                scheduler.step()
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            checkpoint_path = f.name
        
        try:
            trainer.save_checkpoint(checkpoint_path)
            
            # Create new trainer and load checkpoint
            new_models = [SimpleModel() for _ in range(2)]
            new_optimizers = [optim.Adam(m.parameters(), lr=0.1) for m in new_models]
            new_schedulers = create_step_schedulers(new_optimizers, step_size=5)
            
            new_trainer = DMLTrainer(
                models=new_models,
                optimizers=new_optimizers,
                schedulers=new_schedulers,
                device='cpu'
            )
            
            new_trainer.load_checkpoint(checkpoint_path)
            
            # Check scheduler state is restored
            assert new_trainer.schedulers[0].last_epoch == trainer.schedulers[0].last_epoch
        
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
    
    def test_get_learning_rates(self):
        """Test getting current learning rates."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.Adam(m.parameters(), lr=0.01) for m in models]
        schedulers = create_step_schedulers(optimizers)
        
        trainer = DMLTrainer(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            device='cpu'
        )
        
        lrs = trainer.get_learning_rates()
        assert len(lrs) == 2
        assert all(lr == 0.01 for lr in lrs)


class TestSchedulerCreationFunctions:
    """Test all scheduler creation functions."""
    
    def test_exponential_schedulers(self):
        """Test exponential scheduler creation."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.Adam(m.parameters(), lr=0.1) for m in models]
        
        schedulers = create_exponential_schedulers(optimizers, gamma=0.95)
        
        assert len(schedulers) == 2
        
        # Test decay with proper optimizer step first
        initial_lr = optimizers[0].param_groups[0]['lr']
        
        # Dummy training step
        loss = torch.tensor([1.0], requires_grad=True)
        optimizers[0].zero_grad()
        loss.backward()
        optimizers[0].step()
        schedulers[0].step()
        
        assert optimizers[0].param_groups[0]['lr'] == pytest.approx(initial_lr * 0.95)
    
    def test_cosine_warmrestart_schedulers(self):
        """Test cosine warm restart scheduler creation."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.SGD(m.parameters(), lr=0.1) for m in models]
        
        schedulers = create_cosine_warmrestart_schedulers(optimizers, T_0=10)
        
        assert len(schedulers) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
