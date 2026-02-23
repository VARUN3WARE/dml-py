"""
Tests for validation errors that should always be raised.

These tests verify that validation logic cannot be bypassed by Python
optimization flags (-O or -OO), which would remove assert statements.
"""

import pytest
import torch
import torch.nn as nn

from pydml.trainers.dml import DMLTrainer
from pydml.models.cifar.wrn import WideResNet


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.fc(x)


class TestBaseTrainerValidation:
    """Test validation in BaseCollaborativeTrainer."""
    
    def test_optimizer_count_mismatch_raises_error(self):
        """Test that mismatched optimizer count raises ValueError."""
        models = [SimpleModel(), SimpleModel(), SimpleModel()]
        optimizers = [
            torch.optim.SGD(models[0].parameters(), lr=0.1),
            torch.optim.SGD(models[1].parameters(), lr=0.1),
            # Missing third optimizer
        ]
        
        with pytest.raises(ValueError, match="Number of optimizers.*must match number of models"):
            trainer = DMLTrainer(models, optimizers=optimizers)
    
    def test_too_many_optimizers_raises_error(self):
        """Test that too many optimizers raises ValueError."""
        models = [SimpleModel(), SimpleModel()]
        optimizers = [
            torch.optim.SGD(models[0].parameters(), lr=0.1),
            torch.optim.SGD(models[1].parameters(), lr=0.1),
            torch.optim.SGD(models[0].parameters(), lr=0.1),  # Extra optimizer
        ]
        
        with pytest.raises(ValueError, match="Number of optimizers.*must match number of models"):
            trainer = DMLTrainer(models, optimizers=optimizers)
    
    def test_correct_optimizer_count_succeeds(self):
        """Test that correct optimizer count works."""
        models = [SimpleModel(), SimpleModel()]
        optimizers = [
            torch.optim.SGD(models[0].parameters(), lr=0.1),
            torch.optim.SGD(models[1].parameters(), lr=0.1),
        ]
        
        # Should not raise
        trainer = DMLTrainer(models, optimizers=optimizers)
        assert len(trainer.optimizers) == 2


class TestMobileNetValidation:
    """Test validation in MobileNetV2."""
    
    def test_invalid_stride_raises_error(self):
        """Test that invalid stride raises ValueError."""
        from pydml.models.cifar.mobilenet import InvertedResidual
        
        # stride=3 is invalid
        with pytest.raises(ValueError, match="stride must be 1 or 2"):
            block = InvertedResidual(inp=32, oup=64, stride=3, expand_ratio=6)
    
    def test_stride_zero_raises_error(self):
        """Test that stride=0 raises ValueError."""
        from pydml.models.cifar.mobilenet import InvertedResidual
        
        with pytest.raises(ValueError, match="stride must be 1 or 2"):
            block = InvertedResidual(inp=32, oup=64, stride=0, expand_ratio=6)
    
    def test_negative_stride_raises_error(self):
        """Test that negative stride raises ValueError."""
        from pydml.models.cifar.mobilenet import InvertedResidual
        
        with pytest.raises(ValueError, match="stride must be 1 or 2"):
            block = InvertedResidual(inp=32, oup=64, stride=-1, expand_ratio=6)
    
    def test_valid_stride_1_succeeds(self):
        """Test that stride=1 works."""
        from pydml.models.cifar.mobilenet import InvertedResidual
        
        # Should not raise
        block = InvertedResidual(inp=32, oup=32, stride=1, expand_ratio=6)
        assert block.stride == 1
    
    def test_valid_stride_2_succeeds(self):
        """Test that stride=2 works."""
        from pydml.models.cifar.mobilenet import InvertedResidual
        
        # Should not raise
        block = InvertedResidual(inp=32, oup=64, stride=2, expand_ratio=6)
        assert block.stride == 2


class TestWideResNetValidation:
    """Test validation in WideResNet."""
    
    def test_invalid_depth_raises_error(self):
        """Test that invalid depth raises ValueError."""
        # depth=27 doesn't satisfy (depth - 4) % 6 == 0
        # (27 - 4) % 6 = 23 % 6 = 5 (not 0)
        with pytest.raises(ValueError, match="depth must satisfy.*% 6 == 0"):
            model = WideResNet(depth=27)
    
    def test_depth_10_invalid_raises_error(self):
        """Test that depth=10 raises ValueError."""
        # (10 - 4) % 6 = 6 % 6 = 0 (this should work)
        # Let's test depth=11 instead
        # (11 - 4) % 6 = 7 % 6 = 1 (not 0)
        with pytest.raises(ValueError, match="depth must satisfy.*% 6 == 0"):
            model = WideResNet(depth=11)
    
    def test_depth_5_invalid_raises_error(self):
        """Test that depth=5 raises ValueError."""
        # (5 - 4) % 6 = 1 % 6 = 1 (not 0)
        with pytest.raises(ValueError, match="depth must satisfy.*% 6 == 0"):
            model = WideResNet(depth=5)
    
    def test_valid_depth_10_succeeds(self):
        """Test that depth=10 works."""
        # (10 - 4) % 6 = 6 % 6 = 0 
        model = WideResNet(depth=10, num_classes=10)
        assert model is not None
    
    def test_valid_depth_16_succeeds(self):
        """Test that depth=16 works."""
        # (16 - 4) % 6 = 12 % 6 = 0 
        model = WideResNet(depth=16, num_classes=10)
        assert model is not None
    
    def test_valid_depth_28_succeeds(self):
        """Test that depth=28 (default) works."""
        # (28 - 4) % 6 = 24 % 6 = 0 
        model = WideResNet(depth=28, num_classes=10)
        assert model is not None
    
    def test_valid_depth_40_succeeds(self):
        """Test that depth=40 works."""
        # (40 - 4) % 6 = 36 % 6 = 0 
        model = WideResNet(depth=40, num_classes=10)
        assert model is not None


class TestOptimizationFlagResistance:
    """
    Test that validations work even with Python optimization.
    
    These tests verify that the validation logic cannot be bypassed
    when running Python with -O or -OO flags.
    """
    
    def test_optimizer_validation_not_bypassable(self):
        """Verify optimizer validation uses proper exceptions, not assertions."""
        models = [SimpleModel(), SimpleModel()]
        optimizers = [torch.optim.SGD(models[0].parameters(), lr=0.1)]
        
        # This should ALWAYS raise, even with python -O
        with pytest.raises(ValueError):
            trainer = DMLTrainer(models, optimizers=optimizers)
    
    def test_mobilenet_validation_not_bypassable(self):
        """Verify MobileNet validation uses proper exceptions, not assertions."""
        from pydml.models.cifar.mobilenet import InvertedResidual
        
        # This should ALWAYS raise, even with python -O
        with pytest.raises(ValueError):
            block = InvertedResidual(inp=32, oup=64, stride=5, expand_ratio=6)
    
    def test_wrn_validation_not_bypassable(self):
        """Verify WRN validation uses proper exceptions, not assertions."""
        # This should ALWAYS raise, even with python -O
        with pytest.raises(ValueError):
            model = WideResNet(depth=29)


class TestValidationMessages:
    """Test that validation error messages are informative."""
    
    def test_optimizer_error_message_contains_counts(self):
        """Test that optimizer error message includes actual counts."""
        models = [SimpleModel(), SimpleModel(), SimpleModel()]
        optimizers = [torch.optim.SGD(models[0].parameters(), lr=0.1)]
        
        with pytest.raises(ValueError) as exc_info:
            trainer = DMLTrainer(models, optimizers=optimizers)
        
        error_message = str(exc_info.value)
        assert "3" in error_message  # Number of models
        assert "1" in error_message  # Number of optimizers
    
    def test_mobilenet_error_message_contains_stride(self):
        """Test that MobileNet error message includes stride value."""
        from pydml.models.cifar.mobilenet import InvertedResidual
        
        with pytest.raises(ValueError) as exc_info:
            block = InvertedResidual(inp=32, oup=64, stride=4, expand_ratio=6)
        
        error_message = str(exc_info.value)
        assert "4" in error_message  # The invalid stride value
    
    def test_wrn_error_message_contains_depth(self):
        """Test that WRN error message includes depth value."""
        with pytest.raises(ValueError) as exc_info:
            model = WideResNet(depth=29)
        
        error_message = str(exc_info.value)
        assert "29" in error_message  # The invalid depth value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
