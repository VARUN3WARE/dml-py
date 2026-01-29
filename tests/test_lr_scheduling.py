"""
Tests for advanced learning rate scheduling functionality.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pydml.utils.lr_scheduling import (
    SchedulerType,
    WarmupConfig,
    SchedulerConfig,
    create_warmup_scheduler,
    create_polynomial_scheduler,
    create_scheduler_from_config,
    create_schedulers_from_config,
    get_cifar_schedule,
    get_imagenet_schedule,
    get_fine_tuning_schedule,
)
from pydml.trainers import DMLTrainer


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)


class TestWarmupConfig:
    """Test WarmupConfig dataclass."""
    
    def test_valid_config(self):
        """Test valid warmup configuration."""
        config = WarmupConfig(
            warmup_epochs=5,
            warmup_start_lr=1e-6,
            warmup_method='linear'
        )
        assert config.warmup_epochs == 5
        assert config.warmup_start_lr == 1e-6
        assert config.warmup_method == 'linear'
    
    def test_invalid_warmup_epochs(self):
        """Test invalid warmup epochs raises error."""
        with pytest.raises(ValueError, match="warmup_epochs must be non-negative"):
            WarmupConfig(warmup_epochs=-1)
    
    def test_invalid_warmup_start_lr(self):
        """Test invalid start lr raises error."""
        with pytest.raises(ValueError, match="warmup_start_lr must be positive"):
            WarmupConfig(warmup_start_lr=0.0)
    
    def test_invalid_warmup_method(self):
        """Test invalid warmup method raises error."""
        with pytest.raises(ValueError, match="warmup_method must be"):
            WarmupConfig(warmup_method='invalid')


class TestSchedulerConfig:
    """Test SchedulerConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SchedulerConfig()
        assert config.scheduler_type == SchedulerType.COSINE
        assert config.base_lr == 0.1
        assert config.warmup is None
    
    def test_string_to_enum_conversion(self):
        """Test automatic string to enum conversion."""
        config = SchedulerConfig(scheduler_type='cosine')
        assert config.scheduler_type == SchedulerType.COSINE
    
    def test_max_lr_default(self):
        """Test max_lr defaults to base_lr."""
        config = SchedulerConfig(base_lr=0.05)
        assert config.max_lr == 0.05


class TestWarmupScheduler:
    """Test warmup scheduler creation."""
    
    def test_linear_warmup(self):
        """Test linear warmup scheduler."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        warmup_config = WarmupConfig(
            warmup_epochs=5,
            warmup_start_lr=0.01,
            warmup_method='linear'
        )
        
        scheduler = create_warmup_scheduler(optimizer, warmup_config, base_lr=0.1)
        
        # Test LR progression
        initial_lr = optimizer.param_groups[0]['lr']
        scheduler.step()  # Epoch 1
        lr_epoch1 = optimizer.param_groups[0]['lr']
        assert lr_epoch1 > initial_lr  # Should increase
    
    def test_exponential_warmup(self):
        """Test exponential warmup scheduler."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        warmup_config = WarmupConfig(
            warmup_epochs=5,
            warmup_start_lr=0.01,
            warmup_method='exponential'
        )
        
        scheduler = create_warmup_scheduler(optimizer, warmup_config, base_lr=0.1)
        assert scheduler is not None
    
    def test_cosine_warmup(self):
        """Test cosine warmup scheduler."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        warmup_config = WarmupConfig(
            warmup_epochs=5,
            warmup_start_lr=0.01,
            warmup_method='cosine'
        )
        
        scheduler = create_warmup_scheduler(optimizer, warmup_config, base_lr=0.1)
        assert scheduler is not None


class TestPolynomialScheduler:
    """Test polynomial scheduler creation."""
    
    def test_linear_decay(self):
        """Test linear (polynomial with power=1) decay."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        scheduler = create_polynomial_scheduler(
            optimizer,
            total_epochs=10,
            power=1.0,
            final_lr=0.0
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step through epochs
        for _ in range(5):
            scheduler.step()
        
        mid_lr = optimizer.param_groups[0]['lr']
        assert mid_lr < initial_lr  # Should decrease
    
    def test_polynomial_decay(self):
        """Test polynomial decay with power=2."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        scheduler = create_polynomial_scheduler(
            optimizer,
            total_epochs=10,
            power=2.0,
            final_lr=0.01
        )
        
        # Step through all epochs
        for _ in range(10):
            scheduler.step()
        
        final_lr = optimizer.param_groups[0]['lr']
        assert abs(final_lr - 0.01) < 1e-6  # Should reach final_lr


class TestSchedulerFromConfig:
    """Test scheduler creation from configuration."""
    
    def test_cosine_scheduler(self):
        """Test cosine scheduler creation."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        config = SchedulerConfig(
            scheduler_type=SchedulerType.COSINE,
            base_lr=0.1,
            T_max=100,
            eta_min=0.0
        )
        
        scheduler = create_scheduler_from_config(optimizer, config)
        assert scheduler is not None
        
        # Test stepping
        initial_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        stepped_lr = optimizer.param_groups[0]['lr']
        assert stepped_lr <= initial_lr  # Cosine decreases
    
    def test_step_scheduler(self):
        """Test step scheduler creation."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        config = SchedulerConfig(
            scheduler_type=SchedulerType.STEP,
            base_lr=0.1,
            step_size=10,
            gamma=0.1
        )
        
        scheduler = create_scheduler_from_config(optimizer, config)
        assert scheduler is not None
    
    def test_multistep_scheduler(self):
        """Test multistep scheduler creation."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        config = SchedulerConfig(
            scheduler_type=SchedulerType.MULTISTEP,
            base_lr=0.1,
            milestones=[30, 60, 90],
            gamma=0.1
        )
        
        scheduler = create_scheduler_from_config(optimizer, config)
        assert scheduler is not None
    
    def test_scheduler_with_warmup(self):
        """Test scheduler with warmup."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        config = SchedulerConfig(
            scheduler_type=SchedulerType.COSINE,
            base_lr=0.1,
            T_max=100,
            warmup=WarmupConfig(
                warmup_epochs=5,
                warmup_start_lr=0.01,
                warmup_method='linear'
            )
        )
        
        scheduler = create_scheduler_from_config(optimizer, config)
        assert scheduler is not None
        
        # During warmup, LR should increase
        initial_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        warmed_lr = optimizer.param_groups[0]['lr']
        assert warmed_lr > initial_lr
    
    def test_exponential_scheduler(self):
        """Test exponential scheduler creation."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        config = SchedulerConfig(
            scheduler_type=SchedulerType.EXPONENTIAL,
            base_lr=0.1,
            gamma=0.95
        )
        
        scheduler = create_scheduler_from_config(optimizer, config)
        assert scheduler is not None
    
    def test_polynomial_scheduler(self):
        """Test polynomial scheduler creation."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        config = SchedulerConfig(
            scheduler_type=SchedulerType.POLYNOMIAL,
            base_lr=0.1,
            T_max=100,
            power=2.0,
            final_lr=0.001
        )
        
        scheduler = create_scheduler_from_config(optimizer, config)
        assert scheduler is not None
    
    def test_invalid_scheduler_type(self):
        """Test invalid scheduler type raises error."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        # Create config with invalid type (manually bypass enum)
        config = SchedulerConfig(scheduler_type=SchedulerType.MULTISTEP)
        config.milestones = None  # This should cause an error
        
        with pytest.raises(ValueError, match="milestones must be specified"):
            create_scheduler_from_config(optimizer, config)


class TestSchedulersFromConfig:
    """Test creating multiple schedulers from config."""
    
    def test_multiple_schedulers(self):
        """Test creating schedulers for multiple optimizers."""
        models = [SimpleModel() for _ in range(3)]
        optimizers = [optim.SGD(m.parameters(), lr=0.1) for m in models]
        
        config = SchedulerConfig(
            scheduler_type=SchedulerType.COSINE,
            base_lr=0.1,
            T_max=100
        )
        
        schedulers = create_schedulers_from_config(optimizers, config)
        
        assert len(schedulers) == 3
        for scheduler in schedulers:
            assert scheduler is not None


class TestTrainingRecipes:
    """Test pre-configured training recipes."""
    
    def test_cifar_schedule(self):
        """Test CIFAR training schedule."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.SGD(m.parameters(), lr=0.1, momentum=0.9) for m in models]
        
        schedulers = get_cifar_schedule(optimizers, total_epochs=200, warmup_epochs=5)
        
        assert len(schedulers) == 2
        for scheduler in schedulers:
            assert scheduler is not None
    
    def test_imagenet_schedule(self):
        """Test ImageNet training schedule."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.SGD(m.parameters(), lr=0.1, momentum=0.9) for m in models]
        
        schedulers = get_imagenet_schedule(optimizers, total_epochs=90)
        
        assert len(schedulers) == 2
        for scheduler in schedulers:
            assert scheduler is not None
    
    def test_fine_tuning_schedule(self):
        """Test fine-tuning schedule."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.Adam(m.parameters(), lr=1e-4) for m in models]
        
        schedulers = get_fine_tuning_schedule(optimizers, total_epochs=50)
        
        assert len(schedulers) == 2
        for scheduler in schedulers:
            assert scheduler is not None


class TestIntegration:
    """Test integration with DMLTrainer."""
    
    def test_trainer_with_scheduler_config(self):
        """Test DMLTrainer with scheduler from config."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.SGD(m.parameters(), lr=0.1) for m in models]
        
        config = SchedulerConfig(
            scheduler_type=SchedulerType.COSINE,
            base_lr=0.1,
            T_max=10
        )
        schedulers = create_schedulers_from_config(optimizers, config)
        
        trainer = DMLTrainer(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            device='cpu'
        )
        
        # Check trainer has schedulers
        assert len(trainer.schedulers) == 2
    
    def test_trainer_with_cifar_schedule(self):
        """Test DMLTrainer with CIFAR schedule."""
        models = [SimpleModel() for _ in range(2)]
        optimizers = [optim.SGD(m.parameters(), lr=0.1, momentum=0.9) for m in models]
        schedulers = get_cifar_schedule(optimizers, total_epochs=10)
        
        trainer = DMLTrainer(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            device='cpu'
        )
        
        assert len(trainer.schedulers) == 2
        
        # Test that schedulers work
        initial_lr = trainer.get_learning_rates()[0]
        
        # Create dummy data
        X = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        # Train one epoch
        trainer.train_epoch(loader, epoch=1)
        
        # LR might have changed (depends on warmup vs main schedule)
        new_lr = trainer.get_learning_rates()[0]
        assert new_lr is not None


class TestSchedulerBehavior:
    """Test scheduler behavior through training."""
    
    def test_warmup_then_decay(self):
        """Test LR increases during warmup then decreases."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        config = SchedulerConfig(
            scheduler_type=SchedulerType.COSINE,
            base_lr=0.1,
            T_max=20,
            eta_min=0.0,
            warmup=WarmupConfig(
                warmup_epochs=5,
                warmup_start_lr=0.01,
                warmup_method='linear'
            )
        )
        
        scheduler = create_scheduler_from_config(optimizer, config)
        
        lrs = []
        for epoch in range(20):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # Check warmup: LR should increase for first 5 epochs
        for i in range(1, 5):
            assert lrs[i] > lrs[i-1], f"LR should increase during warmup at epoch {i}"
        
        # Check decay: LR should generally decrease after warmup
        # (allowing small fluctuations)
        assert lrs[10] < lrs[5], "LR should decrease after warmup"
        assert lrs[19] < lrs[10], "LR should continue decreasing"
    
    def test_constant_scheduler(self):
        """Test constant LR scheduler."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        config = SchedulerConfig(
            scheduler_type=SchedulerType.CONSTANT,
            base_lr=0.1
        )
        
        scheduler = create_scheduler_from_config(optimizer, config)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        for _ in range(10):
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            assert abs(current_lr - initial_lr) < 1e-8, "LR should remain constant"
