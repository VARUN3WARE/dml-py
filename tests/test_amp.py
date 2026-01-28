"""Tests for Automatic Mixed Precision (AMP) training."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pydml.utils.amp import AMPConfig, AMPManager, apply_amp_to_trainer
from pydml.trainers.dml import DMLTrainer


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)


class TestAMPConfig:
    """Test AMP configuration."""
    
    def test_default_config(self):
        """Test default AMP configuration."""
        config = AMPConfig()
        
        # Check auto-detection
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
            assert config.enabled is True
        else:
            assert config.enabled is False
        
        assert config.dtype == torch.float16
    
    def test_explicit_enabled(self):
        """Test explicit enabling/disabling."""
        config_enabled = AMPConfig(enabled=True)
        assert config_enabled.enabled is True
        
        config_disabled = AMPConfig(enabled=False)
        assert config_disabled.enabled is False
    
    def test_custom_dtype(self):
        """Test custom dtype configuration."""
        config = AMPConfig(dtype=torch.bfloat16)
        assert config.dtype == torch.bfloat16
    
    def test_custom_scaler_params(self):
        """Test custom gradient scaler parameters."""
        config = AMPConfig(
            init_scale=2.**10,
            growth_factor=1.5,
            backoff_factor=0.75,
            growth_interval=1000
        )
        
        assert config.init_scale == 2.**10
        assert config.growth_factor == 1.5
        assert config.backoff_factor == 0.75
        assert config.growth_interval == 1000


class TestAMPManager:
    """Test AMP manager."""
    
    def test_initialization_cpu(self):
        """Test AMP manager initialization on CPU."""
        config = AMPConfig(enabled=False)
        manager = AMPManager(config, device='cpu')
        
        assert manager.scaler is None
        assert manager.config.enabled is False
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_initialization_cuda_enabled(self):
        """Test AMP manager initialization on CUDA with AMP enabled."""
        config = AMPConfig(enabled=True)
        manager = AMPManager(config, device='cuda')
        
        assert manager.scaler is not None
        assert manager.config.enabled is True
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_initialization_cuda_disabled(self):
        """Test AMP manager initialization on CUDA with AMP disabled."""
        config = AMPConfig(enabled=False)
        manager = AMPManager(config, device='cuda')
        
        assert manager.scaler is None
    
    def test_autocast_context_cpu(self):
        """Test autocast context on CPU."""
        config = AMPConfig(enabled=False)
        manager = AMPManager(config, device='cpu')
        
        model = SimpleModel()
        x = torch.randn(2, 10)
        
        with manager.autocast():
            output = model(x)
        
        assert output.dtype == torch.float32
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autocast_context_cuda(self):
        """Test autocast context on CUDA."""
        config = AMPConfig(enabled=True, dtype=torch.float16)
        manager = AMPManager(config, device='cuda')
        
        model = SimpleModel().cuda()
        x = torch.randn(2, 10, device='cuda')
        
        with manager.autocast():
            output = model(x)
        
        # Output should be float16 in autocast context
        assert output.dtype == torch.float16
    
    def test_scale_loss_disabled(self):
        """Test loss scaling when AMP is disabled."""
        config = AMPConfig(enabled=False)
        manager = AMPManager(config, device='cpu')
        
        loss = torch.tensor(1.5)
        scaled_loss = manager.scale_loss(loss)
        
        assert scaled_loss == loss
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_scale_loss_enabled(self):
        """Test loss scaling when AMP is enabled."""
        config = AMPConfig(enabled=True)
        manager = AMPManager(config, device='cuda')
        
        loss = torch.tensor(1.5, device='cuda')
        scaled_loss = manager.scale_loss(loss)
        
        # Scaled loss should be different from original
        assert scaled_loss.item() != loss.item()
    
    def test_state_dict_disabled(self):
        """Test state dict when AMP is disabled."""
        config = AMPConfig(enabled=False)
        manager = AMPManager(config, device='cpu')
        
        state = manager.state_dict()
        assert state == {}
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_state_dict_enabled(self):
        """Test state dict when AMP is enabled."""
        config = AMPConfig(enabled=True)
        manager = AMPManager(config, device='cuda')
        
        state = manager.state_dict()
        assert 'scaler' in state
        assert state['scaler'] is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_state_dict(self):
        """Test loading state dict."""
        config = AMPConfig(enabled=True)
        manager1 = AMPManager(config, device='cuda')
        manager2 = AMPManager(config, device='cuda')
        
        # Get state from manager1
        state = manager1.state_dict()
        
        # Load into manager2
        manager2.load_state_dict(state)
        
        # Should not raise


class TestAMPTraining:
    """Test AMP integration with training."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dml_trainer_with_amp(self):
        """Test DML trainer with AMP enabled."""
        # Create simple models
        models = [SimpleModel().cuda() for _ in range(2)]
        
        # Create trainer with AMP
        trainer = DMLTrainer(models, device='cuda', use_amp=True)
        
        assert trainer.use_amp is True
        assert trainer.amp_manager is not None
    
    def test_dml_trainer_without_amp(self):
        """Test DML trainer without AMP."""
        # Create simple models
        models = [SimpleModel() for _ in range(2)]
        
        # Create trainer without AMP
        trainer = DMLTrainer(models, device='cpu', use_amp=False)
        
        assert trainer.use_amp is False
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_step_with_amp(self):
        """Test one training step with AMP."""
        # Create simple dataset
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)
        
        # Create models and trainer
        models = [SimpleModel().cuda() for _ in range(2)]
        trainer = DMLTrainer(models, device='cuda', use_amp=True)
        
        # Train for one epoch
        trainer.train_epoch(loader, epoch=1)
        
        # Should complete without errors
    
    def test_training_step_without_amp(self):
        """Test one training step without AMP."""
        # Create simple dataset
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)
        
        # Create models and trainer
        models = [SimpleModel() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu', use_amp=False)
        
        # Train for one epoch
        trainer.train_epoch(loader, epoch=1)
        
        # Should complete without errors
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_checkpoint_with_amp(self):
        """Test saving/loading checkpoint with AMP state."""
        import tempfile
        import os
        
        # Create models and trainer
        models = [SimpleModel().cuda() for _ in range(2)]
        trainer = DMLTrainer(models, device='cuda', use_amp=True)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
            trainer.save_checkpoint(checkpoint_path)
            
            # Create new trainer
            new_models = [SimpleModel().cuda() for _ in range(2)]
            new_trainer = DMLTrainer(new_models, device='cuda', use_amp=True)
            
            # Load checkpoint
            new_trainer.load_checkpoint(checkpoint_path)
            
            # Should complete without errors
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_amp_with_bfloat16(self):
        """Test AMP with bfloat16 dtype."""
        # Check if bfloat16 is supported
        if not torch.cuda.is_bf16_supported():
            pytest.skip("BFloat16 not supported on this GPU")
        
        models = [SimpleModel().cuda() for _ in range(2)]
        trainer = DMLTrainer(models, device='cuda', use_amp=True, amp_dtype=torch.bfloat16)
        
        assert trainer.amp_manager.config.dtype == torch.bfloat16


class TestApplyAMPToTrainer:
    """Test apply_amp_to_trainer utility function."""
    
    def test_apply_amp_to_existing_trainer(self):
        """Test that trainers already have AMP built-in."""
        models = [SimpleModel() for _ in range(2)]
        trainer = DMLTrainer(models, device='cpu', use_amp=False)
        
        # Trainers now have built-in AMP
        assert hasattr(trainer, 'amp_manager')
        assert hasattr(trainer, 'use_amp')
    
    def test_amp_already_integrated(self):
        """Test that AMP is integrated, not applied separately."""
        models = [SimpleModel() for _ in range(2)]
        
        # Can enable at construction
        trainer = DMLTrainer(models, device='cpu', use_amp=True)
        assert trainer.use_amp is True
        
        # Or disable
        trainer2 = DMLTrainer(models, device='cpu', use_amp=False)
        assert trainer2.use_amp is False


class TestAMPIntegration:
    """Integration tests for AMP."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_training_workflow_with_amp(self):
        """Test complete training workflow with AMP."""
        # Create dataset
        X = torch.randn(200, 10)
        y = torch.randint(0, 5, (200,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=32)
        
        # Create models
        models = [SimpleModel().cuda() for _ in range(2)]
        
        # Create trainer with AMP
        trainer = DMLTrainer(models, device='cuda', use_amp=True)
        
        # Train for 2 epochs
        history = trainer.fit(train_loader, val_loader, epochs=2, verbose=False)
        
        # Check history
        assert 'train_loss' in history
        assert len(history['train_loss']) == 2
        assert all(isinstance(loss, float) for loss in history['train_loss'])
    
    def test_amp_with_different_model_sizes(self):
        """Test AMP with models of different sizes."""
        # Create models of different sizes
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            def forward(self, x):
                return self.fc(x)
        
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(10, 100),
                    nn.ReLU(),
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 5)
                )
            def forward(self, x):
                return self.net(x)
        
        models = [SmallModel(), LargeModel()]
        
        if torch.cuda.is_available():
            models = [m.cuda() for m in models]
            device = 'cuda'
        else:
            device = 'cpu'
        
        # Should work with AMP
        trainer = DMLTrainer(models, device=device, use_amp=True)
        
        # Create simple data
        X = torch.randn(50, 10)
        y = torch.randint(0, 5, (50,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)
        
        # Train for one epoch
        trainer.train_epoch(loader, epoch=1)
