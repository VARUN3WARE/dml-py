"""
Tests for model validation functionality.

Tests the validate_model_compatibility function which ensures:
- Models are valid PyTorch modules
- Models have trainable parameters
- Models produce same output dimensions
- Validation provides clear error messages
"""

import pytest
import torch
import torch.nn as nn
from pydml.utils.validation import validate_model_compatibility


class SimpleModel(nn.Module):
    """Simple model with configurable output dimension."""
    
    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)


class ConvModel(nn.Module):
    """Simple convolutional model."""
    
    def __init__(self, output_dim=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, output_dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class EmptyModel(nn.Module):
    """Model with no parameters."""
    
    def forward(self, x):
        return x


class TestValidateModelCompatibility:
    """Test suite for validate_model_compatibility function."""
    
    def test_valid_models_same_dimension(self):
        """Test validation passes for models with same output dimension."""
        models = [SimpleModel(10, 5), SimpleModel(10, 5), SimpleModel(10, 5)]
        output_dim = validate_model_compatibility(models, 'cpu')
        assert output_dim == 5
    
    def test_valid_conv_models(self):
        """Test validation works for convolutional models."""
        models = [ConvModel(10), ConvModel(10)]
        output_dim = validate_model_compatibility(models, 'cpu')
        assert output_dim == 10
    
    def test_mixed_architecture_same_output(self):
        """Test validation passes for different architectures with same output."""
        models = [SimpleModel(10, 5), ConvModel(5)]
        output_dim = validate_model_compatibility(models, 'cpu')
        assert output_dim == 5
    
    def test_invalid_not_list(self):
        """Test validation fails if models is not a list or tuple."""
        with pytest.raises(TypeError, match="models must be a list or tuple"):
            validate_model_compatibility("not a list", 'cpu')
    
    def test_invalid_empty_list(self):
        """Test validation fails for empty model list."""
        with pytest.raises(ValueError, match="models list cannot be empty"):
            validate_model_compatibility([], 'cpu')
    
    def test_invalid_not_module(self):
        """Test validation fails if model is not a PyTorch module."""
        models = [SimpleModel(), "not a module"]
        with pytest.raises(TypeError, match="models\\[1\\] must be a torch.nn.Module"):
            validate_model_compatibility(models, 'cpu')
    
    def test_invalid_no_parameters(self):
        """Test validation fails for models with no parameters."""
        models = [SimpleModel(), EmptyModel()]
        with pytest.raises(ValueError, match="models\\[1\\] has no parameters"):
            validate_model_compatibility(models, 'cpu')
    
    def test_invalid_mismatched_dimensions(self):
        """Test validation fails for models with different output dimensions."""
        models = [SimpleModel(10, 5), SimpleModel(10, 7), SimpleModel(10, 5)]
        with pytest.raises(ValueError) as exc_info:
            validate_model_compatibility(models, 'cpu')
        
        error_msg = str(exc_info.value)
        assert "all models must have the same output dimension" in error_msg
        assert "models[0]: output_dim=5" in error_msg
        assert "models[1]: output_dim=7" in error_msg
        assert "models[2]: output_dim=5" in error_msg
    
    def test_detailed_error_message(self):
        """Test that error messages include parameter counts and dimensions."""
        models = [SimpleModel(10, 3), SimpleModel(10, 5)]
        with pytest.raises(ValueError) as exc_info:
            validate_model_compatibility(models, 'cpu')
        
        error_msg = str(exc_info.value)
        assert "output_dim=3" in error_msg
        assert "output_dim=5" in error_msg
        assert "params=" in error_msg  # Shows parameter count
    
    def test_models_on_different_devices(self):
        """Test validation handles models on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model1 = SimpleModel(10, 5).to('cpu')
        model2 = SimpleModel(10, 5).to('cuda')
        models = [model1, model2]
        
        # Should still validate successfully
        output_dim = validate_model_compatibility(models, 'cpu')
        assert output_dim == 5
    
    def test_large_number_of_models(self):
        """Test validation works with many models."""
        models = [SimpleModel(10, 5) for _ in range(10)]
        output_dim = validate_model_compatibility(models, 'cpu')
        assert output_dim == 5
    
    def test_tuple_of_models(self):
        """Test validation accepts tuple of models."""
        models = (SimpleModel(10, 5), SimpleModel(10, 5))
        output_dim = validate_model_compatibility(models, 'cpu')
        assert output_dim == 5
    
    def test_models_with_different_input_shapes(self):
        """Test validation works when models expect different input shapes."""
        # Linear model expects (batch, 10)
        # Conv model expects (batch, 3, 32, 32)
        # They should both work if output dims match
        models = [SimpleModel(10, 5), ConvModel(5)]
        output_dim = validate_model_compatibility(models, 'cpu')
        assert output_dim == 5
    
    def test_non_tensor_output(self):
        """Test validation fails if model returns non-tensor."""
        class BadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)  # Add parameters
            
            def forward(self, x):
                return [1, 2, 3]  # Returns list instead of tensor
        
        models = [SimpleModel(), BadModel()]
        with pytest.raises(ValueError, match="models\\[1\\] returned list, expected torch.Tensor"):
            validate_model_compatibility(models, 'cpu')
