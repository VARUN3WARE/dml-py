"""
Unit tests for VGG models.
"""

import pytest
import torch
import torch.nn as nn
from pydml.models.cifar import vgg11, vgg13, vgg16, vgg19


class TestVGGModels:
    """Test suite for VGG models."""
    
    def test_vgg11_creation(self):
        """Test VGG11 model creation."""
        model = vgg11(num_classes=10)
        assert isinstance(model, nn.Module)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
    
    def test_vgg13_creation(self):
        """Test VGG13 model creation."""
        model = vgg13(num_classes=10)
        assert isinstance(model, nn.Module)
    
    def test_vgg16_creation(self):
        """Test VGG16 model creation."""
        model = vgg16(num_classes=10)
        assert isinstance(model, nn.Module)
    
    def test_vgg19_creation(self):
        """Test VGG19 model creation."""
        model = vgg19(num_classes=10)
        assert isinstance(model, nn.Module)
    
    def test_vgg11_forward_pass(self):
        """Test VGG11 forward pass with CIFAR-sized input."""
        model = vgg11(num_classes=10)
        model.eval()
        
        # CIFAR input size: [batch, 3, 32, 32]
        x = torch.randn(4, 3, 32, 32)
        
        with torch.no_grad():
            output = model(x)
        
        # Output should be [batch, num_classes]
        assert output.shape == (4, 10)
    
    def test_vgg16_forward_pass(self):
        """Test VGG16 forward pass with CIFAR-sized input."""
        model = vgg16(num_classes=10)
        model.eval()
        
        x = torch.randn(2, 3, 32, 32)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_vgg19_forward_pass(self):
        """Test VGG19 forward pass with CIFAR-sized input."""
        model = vgg19(num_classes=10)
        model.eval()
        
        x = torch.randn(8, 3, 32, 32)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (8, 10)
    
    def test_vgg_cifar100(self):
        """Test VGG models with CIFAR-100 (100 classes)."""
        model = vgg16(num_classes=100)
        model.eval()
        
        x = torch.randn(4, 3, 32, 32)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (4, 100)
    
    def test_vgg_batch_norm(self):
        """Test VGG with and without batch normalization."""
        # With batch norm (default)
        model_bn = vgg16(num_classes=10, batch_norm=True)
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model_bn.modules())
        assert has_bn, "Model should contain BatchNorm layers"
        
        # Without batch norm
        model_no_bn = vgg16(num_classes=10, batch_norm=False)
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model_no_bn.modules())
        assert not has_bn, "Model should not contain BatchNorm layers"
    
    def test_vgg_gradient_flow(self):
        """Test that gradients flow through VGG model."""
        model = vgg16(num_classes=10)
        model.train()
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = model(x)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_vgg_parameter_count(self):
        """Test that parameter counts are reasonable."""
        models_info = [
            (vgg11, 'VGG11'),
            (vgg13, 'VGG13'),
            (vgg16, 'VGG16'),
            (vgg19, 'VGG19'),
        ]
        
        param_counts = []
        for model_fn, name in models_info:
            model = model_fn(num_classes=10)
            num_params = sum(p.numel() for p in model.parameters())
            param_counts.append((name, num_params))
            print(f"{name}: {num_params:,} parameters")
        
        # VGG19 should have more parameters than VGG11
        assert param_counts[3][1] > param_counts[0][1]
    
    def test_vgg_different_batch_sizes(self):
        """Test VGG with different batch sizes."""
        model = vgg16(num_classes=10)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 32, 32)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 10)
    
    def test_vgg_training_mode(self):
        """Test switching between train and eval modes."""
        model = vgg16(num_classes=10)
        
        # Training mode
        model.train()
        assert model.training
        
        # Eval mode
        model.eval()
        assert not model.training
    
    def test_vgg_reproducibility(self):
        """Test that VGG produces consistent outputs with same seed."""
        torch.manual_seed(42)
        model = vgg16(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        
        model.eval()
        with torch.no_grad():
            output1 = model(x)
        
        # Reset and test again
        model.eval()
        with torch.no_grad():
            output2 = model(x)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
