"""
Tests for input validation utilities.

These tests verify that validation functions properly check inputs
and provide clear, helpful error messages.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pydml.utils.validation import (
    validate_positive_int,
    validate_positive_float,
    validate_probability,
    validate_range,
    validate_string_choice,
    validate_device,
    validate_model_list,
    validate_optimizer_list,
    validate_data_loader,
    validate_batch_size,
    validate_num_workers,
    validate_epochs,
    validate_learning_rate,
    validate_temperature,
    validate_tensor_shape,
    validate_weights,
)


class TestPositiveInt:
    """Test validate_positive_int function."""
    
    def test_valid_positive_int(self):
        """Test that positive integers are accepted."""
        assert validate_positive_int(5, "test") == 5
        assert validate_positive_int(1, "test") == 1
        assert validate_positive_int(1000, "test") == 1000
    
    def test_zero_not_allowed_by_default(self):
        """Test that zero is rejected by default."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_int(0, "test")
    
    def test_zero_allowed_when_specified(self):
        """Test that zero is accepted when allow_zero=True."""
        assert validate_positive_int(0, "test", allow_zero=True) == 0
    
    def test_negative_rejected(self):
        """Test that negative integers are rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_int(-1, "test")
        
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_positive_int(-5, "test", allow_zero=True)
    
    def test_float_rejected(self):
        """Test that floats are rejected."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_positive_int(5.5, "test")
    
    def test_string_rejected(self):
        """Test that strings are rejected."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_positive_int("5", "test")
    
    def test_bool_rejected(self):
        """Test that booleans are rejected."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_positive_int(True, "test")


class TestPositiveFloat:
    """Test validate_positive_float function."""
    
    def test_valid_positive_float(self):
        """Test that positive floats are accepted."""
        assert validate_positive_float(5.5, "test") == 5.5
        assert validate_positive_float(0.1, "test") == 0.1
        assert validate_positive_float(1000.0, "test") == 1000.0
    
    def test_int_converted_to_float(self):
        """Test that integers are converted to float."""
        result = validate_positive_float(5, "test")
        assert result == 5.0
        assert isinstance(result, float)
    
    def test_zero_not_allowed_by_default(self):
        """Test that zero is rejected by default."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_float(0.0, "test")
    
    def test_zero_allowed_when_specified(self):
        """Test that zero is accepted when allow_zero=True."""
        assert validate_positive_float(0.0, "test", allow_zero=True) == 0.0
    
    def test_negative_rejected(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_float(-1.5, "test")


class TestProbability:
    """Test validate_probability function."""
    
    def test_valid_probabilities(self):
        """Test that valid probabilities are accepted."""
        assert validate_probability(0.0, "test") == 0.0
        assert validate_probability(0.5, "test") == 0.5
        assert validate_probability(1.0, "test") == 1.0
    
    def test_int_converted_to_float(self):
        """Test that integers are converted to float."""
        result = validate_probability(1, "test")
        assert result == 1.0
        assert isinstance(result, float)
    
    def test_below_zero_rejected(self):
        """Test that values below 0 are rejected."""
        with pytest.raises(ValueError, match="must be in range"):
            validate_probability(-0.1, "test")
    
    def test_above_one_rejected(self):
        """Test that values above 1 are rejected."""
        with pytest.raises(ValueError, match="must be in range"):
            validate_probability(1.1, "test")


class TestRange:
    """Test validate_range function."""
    
    def test_valid_range(self):
        """Test that values within range are accepted."""
        assert validate_range(5.0, "test", 0.0, 10.0) == 5.0
        assert validate_range(0.0, "test", 0.0, 10.0) == 0.0
        assert validate_range(10.0, "test", 0.0, 10.0) == 10.0
    
    def test_below_range_rejected(self):
        """Test that values below range are rejected."""
        with pytest.raises(ValueError, match="must be in range"):
            validate_range(-1.0, "test", 0.0, 10.0)
    
    def test_above_range_rejected(self):
        """Test that values above range are rejected."""
        with pytest.raises(ValueError, match="must be in range"):
            validate_range(11.0, "test", 0.0, 10.0)


class TestStringChoice:
    """Test validate_string_choice function."""
    
    def test_valid_choice(self):
        """Test that valid choices are accepted."""
        choices = ['a', 'b', 'c']
        assert validate_string_choice('a', "test", choices) == 'a'
        assert validate_string_choice('b', "test", choices) == 'b'
    
    def test_invalid_choice_rejected(self):
        """Test that invalid choices are rejected."""
        choices = ['a', 'b', 'c']
        with pytest.raises(ValueError, match="must be one of"):
            validate_string_choice('d', "test", choices)
    
    def test_non_string_rejected(self):
        """Test that non-strings are rejected."""
        choices = ['a', 'b', 'c']
        with pytest.raises(TypeError, match="must be a string"):
            validate_string_choice(1, "test", choices)


class TestDevice:
    """Test validate_device function."""
    
    def test_cpu_string(self):
        """Test that 'cpu' string is accepted."""
        device = validate_device('cpu')
        assert isinstance(device, torch.device)
        assert device.type == 'cpu'
    
    def test_cpu_uppercase(self):
        """Test that 'CPU' is normalized to lowercase."""
        device = validate_device('CPU')
        assert device.type == 'cpu'
    
    def test_torch_device_object(self):
        """Test that torch.device objects are accepted."""
        original = torch.device('cpu')
        device = validate_device(original)
        assert device == original
    
    def test_invalid_device_rejected(self):
        """Test that invalid device strings are rejected."""
        with pytest.raises(ValueError, match="Invalid device"):
            validate_device('gpu')
    
    def test_non_string_rejected(self):
        """Test that non-strings are rejected."""
        with pytest.raises(TypeError, match="must be str or torch.device"):
            validate_device(123)


class TestModelList:
    """Test validate_model_list function."""
    
    def test_valid_model_list(self):
        """Test that list of models is accepted."""
        models = [nn.Linear(10, 10), nn.Linear(10, 10)]
        result = validate_model_list(models)
        assert len(result) == 2
        assert all(isinstance(m, nn.Module) for m in result)
    
    def test_tuple_converted_to_list(self):
        """Test that tuples are converted to lists."""
        models = (nn.Linear(10, 10), nn.Linear(10, 10))
        result = validate_model_list(models)
        assert isinstance(result, list)
        assert len(result) == 2
    
    def test_too_few_models_rejected(self):
        """Test that too few models are rejected."""
        models = [nn.Linear(10, 10)]
        with pytest.raises(ValueError, match="At least 2 models required"):
            validate_model_list(models, min_count=2)
    
    def test_non_module_rejected(self):
        """Test that non-Module objects are rejected."""
        models = [nn.Linear(10, 10), "not a model"]
        with pytest.raises(TypeError, match="must be a torch.nn.Module"):
            validate_model_list(models)
    
    def test_non_list_rejected(self):
        """Test that non-lists are rejected."""
        with pytest.raises(TypeError, match="must be a list or tuple"):
            validate_model_list(nn.Linear(10, 10))


class TestOptimizerList:
    """Test validate_optimizer_list function."""
    
    def test_valid_optimizer_list(self):
        """Test that list of optimizers is accepted."""
        model = nn.Linear(10, 10)
        optimizers = [
            torch.optim.SGD(model.parameters(), lr=0.1),
            torch.optim.Adam(model.parameters(), lr=0.001),
        ]
        result = validate_optimizer_list(optimizers, num_models=2)
        assert len(result) == 2
    
    def test_wrong_count_rejected(self):
        """Test that wrong optimizer count is rejected."""
        model = nn.Linear(10, 10)
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.1)]
        with pytest.raises(ValueError, match="Number of optimizers.*must match"):
            validate_optimizer_list(optimizers, num_models=2)
    
    def test_non_optimizer_rejected(self):
        """Test that non-Optimizer objects are rejected."""
        optimizers = ["not an optimizer", "also not"]
        with pytest.raises(TypeError, match="must be a torch.optim.Optimizer"):
            validate_optimizer_list(optimizers, num_models=2)


class TestDataLoader:
    """Test validate_data_loader function."""
    
    def test_valid_data_loader(self):
        """Test that valid DataLoader is accepted."""
        dataset = TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,)))
        loader = DataLoader(dataset, batch_size=2)
        result = validate_data_loader(loader)
        assert result == loader
    
    def test_empty_loader_rejected(self):
        """Test that empty DataLoader is rejected."""
        dataset = TensorDataset(torch.randn(0, 5), torch.randint(0, 2, (0,)))
        loader = DataLoader(dataset, batch_size=2)
        with pytest.raises(ValueError, match="appears to be empty"):
            validate_data_loader(loader)
    
    def test_non_dataloader_rejected(self):
        """Test that non-DataLoader objects are rejected."""
        with pytest.raises(TypeError, match="must be a torch.utils.data.DataLoader"):
            validate_data_loader([1, 2, 3])


class TestBatchSize:
    """Test validate_batch_size function."""
    
    def test_valid_batch_size(self):
        """Test that valid batch sizes are accepted."""
        assert validate_batch_size(32) == 32
        assert validate_batch_size(128) == 128
    
    def test_larger_than_dataset_rejected(self):
        """Test that batch size larger than dataset is rejected."""
        with pytest.raises(ValueError, match="cannot be larger than dataset size"):
            validate_batch_size(100, dataset_size=50)


class TestNumWorkers:
    """Test validate_num_workers function."""
    
    def test_valid_num_workers(self):
        """Test that valid number of workers are accepted."""
        assert validate_num_workers(0) == 0
        assert validate_num_workers(4) == 4
    
    def test_negative_rejected(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_num_workers(-1)


class TestEpochs:
    """Test validate_epochs function."""
    
    def test_valid_epochs(self):
        """Test that valid epoch counts are accepted."""
        assert validate_epochs(100) == 100
        assert validate_epochs(10, start_epoch=1) == 10
    
    def test_start_greater_than_total_rejected(self):
        """Test that start_epoch > epochs is rejected."""
        with pytest.raises(ValueError, match="cannot be greater than total epochs"):
            validate_epochs(10, start_epoch=20)


class TestLearningRate:
    """Test validate_learning_rate function."""
    
    def test_valid_learning_rate(self):
        """Test that valid learning rates are accepted."""
        assert validate_learning_rate(0.1) == 0.1
        assert validate_learning_rate(0.001) == 0.001
    
    def test_large_lr_warns(self):
        """Test that large learning rates trigger warning."""
        with pytest.warns(UserWarning, match="unusually large"):
            validate_learning_rate(2.0)


class TestTemperature:
    """Test validate_temperature function."""
    
    def test_valid_temperature(self):
        """Test that valid temperatures are accepted."""
        assert validate_temperature(3.0) == 3.0
        assert validate_temperature(10.0) == 10.0
    
    def test_low_temperature_warns(self):
        """Test that low temperatures trigger warning."""
        with pytest.warns(UserWarning, match="less than 1.0"):
            validate_temperature(0.5)
    
    def test_high_temperature_warns(self):
        """Test that very high temperatures trigger warning."""
        with pytest.warns(UserWarning, match="greater than 20.0"):
            validate_temperature(50.0)


class TestTensorShape:
    """Test validate_tensor_shape function."""
    
    def test_valid_tensor_shape(self):
        """Test that correct tensor dimensions are accepted."""
        tensor = torch.randn(10, 5)
        result = validate_tensor_shape(tensor, expected_ndim=2)
        assert result.shape == (10, 5)
    
    def test_wrong_dimensions_rejected(self):
        """Test that wrong dimensions are rejected."""
        tensor = torch.randn(10, 5, 3)
        with pytest.raises(ValueError, match="must be 2D tensor"):
            validate_tensor_shape(tensor, expected_ndim=2)
    
    def test_non_tensor_rejected(self):
        """Test that non-tensors are rejected."""
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            validate_tensor_shape([1, 2, 3], expected_ndim=1)


class TestWeights:
    """Test validate_weights function."""
    
    def test_valid_weights(self):
        """Test that valid weights are accepted."""
        weights = [0.5, 0.3, 0.2]
        result = validate_weights(weights, num_models=3)
        assert result == [0.5, 0.3, 0.2]
    
    def test_tuple_converted_to_list(self):
        """Test that tuples are converted to lists."""
        weights = (0.5, 0.5)
        result = validate_weights(weights, num_models=2)
        assert isinstance(result, list)
    
    def test_wrong_count_rejected(self):
        """Test that wrong weight count is rejected."""
        weights = [0.5, 0.5]
        with pytest.raises(ValueError, match="Number of weights.*must match"):
            validate_weights(weights, num_models=3)
    
    def test_negative_weight_rejected(self):
        """Test that negative weights are rejected."""
        weights = [0.5, -0.1]
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_weights(weights, num_models=2)
    
    def test_all_zero_rejected(self):
        """Test that all-zero weights are rejected."""
        weights = [0.0, 0.0]
        with pytest.raises(ValueError, match="at least one weight must be positive"):
            validate_weights(weights, num_models=2)
    
    def test_non_numeric_rejected(self):
        """Test that non-numeric weights are rejected."""
        weights = [0.5, "not a number"]
        with pytest.raises(TypeError, match="must be a number"):
            validate_weights(weights, num_models=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
