"""
Input validation utilities for PyDML.

This module provides comprehensive validation functions to ensure:
1. Early error detection with clear messages
2. Type safety
3. Value range checks
4. Consistency checks

All validation functions raise ValueError or TypeError with descriptive messages.
"""

from typing import Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def validate_positive_int(value: Any, name: str, allow_zero: bool = False) -> int:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        allow_zero: Whether to allow zero as valid value
        
    Returns:
        Validated integer value
        
    Raises:
        TypeError: If value is not an integer
        ValueError: If value is not positive (or >= 0 if allow_zero=True)
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    
    if allow_zero and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    elif not allow_zero and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    
    return value


def validate_positive_float(value: Any, name: str, allow_zero: bool = False) -> float:
    """
    Validate that a value is a positive float.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        allow_zero: Whether to allow zero as valid value
        
    Returns:
        Validated float value
        
    Raises:
        TypeError: If value is not a number
        ValueError: If value is not positive (or >= 0 if allow_zero=True)
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    
    value = float(value)
    
    if allow_zero and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    elif not allow_zero and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    
    return value


def validate_probability(value: Any, name: str) -> float:
    """
    Validate that a value is a valid probability (0 <= value <= 1).
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        
    Returns:
        Validated probability value
        
    Raises:
        TypeError: If value is not a number
        ValueError: If value is not in [0, 1]
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    
    value = float(value)
    
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in range [0, 1], got {value}")
    
    return value


def validate_range(value: Any, name: str, min_val: float, max_val: float) -> float:
    """
    Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        
    Returns:
        Validated value
        
    Raises:
        TypeError: If value is not a number
        ValueError: If value is outside the specified range
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    
    value = float(value)
    
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be in range [{min_val}, {max_val}], got {value}")
    
    return value


def validate_string_choice(value: Any, name: str, choices: List[str]) -> str:
    """
    Validate that a string value is in a list of allowed choices.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        choices: List of allowed string values
        
    Returns:
        Validated string value
        
    Raises:
        TypeError: If value is not a string
        ValueError: If value is not in choices
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    
    if value not in choices:
        raise ValueError(
            f"{name} must be one of {choices}, got '{value}'"
        )
    
    return value


def validate_device(device: Any) -> torch.device:
    """
    Validate and normalize a device specification.
    
    Args:
        device: Device specification (str or torch.device)
        
    Returns:
        torch.device object
        
    Raises:
        TypeError: If device is not a string or torch.device
        ValueError: If device string is invalid
    """
    if isinstance(device, torch.device):
        return device
    
    if not isinstance(device, str):
        raise TypeError(f"device must be str or torch.device, got {type(device).__name__}")
    
    device = device.lower()
    
    # Validate device string
    if not (device == 'cpu' or device.startswith('cuda') or device.startswith('mps')):
        raise ValueError(
            f"Invalid device '{device}'. Must be 'cpu', 'cuda', 'cuda:N', or 'mps'"
        )
    
    # Check CUDA availability if specified
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError(
            f"CUDA device '{device}' specified but CUDA is not available. "
            f"Use 'cpu' instead or install CUDA."
        )
    
    return torch.device(device)


def validate_model_list(models: Any, min_count: int = 2) -> List[nn.Module]:
    """
    Validate a list of PyTorch models.
    
    Args:
        models: List of models to validate
        min_count: Minimum number of models required
        
    Returns:
        Validated list of models
        
    Raises:
        TypeError: If models is not a list or contains non-Module objects
        ValueError: If there are too few models
    """
    if not isinstance(models, (list, tuple)):
        raise TypeError(f"models must be a list or tuple, got {type(models).__name__}")
    
    if len(models) < min_count:
        raise ValueError(
            f"At least {min_count} models required, got {len(models)}"
        )
    
    for i, model in enumerate(models):
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"models[{i}] must be a torch.nn.Module, "
                f"got {type(model).__name__}"
            )
    
    return list(models)


def validate_optimizer_list(
    optimizers: Any,
    num_models: int
) -> List[torch.optim.Optimizer]:
    """
    Validate a list of optimizers.
    
    Args:
        optimizers: List of optimizers to validate
        num_models: Expected number of optimizers
        
    Returns:
        Validated list of optimizers
        
    Raises:
        TypeError: If optimizers is not a list or contains non-Optimizer objects
        ValueError: If the number of optimizers doesn't match num_models
    """
    if not isinstance(optimizers, (list, tuple)):
        raise TypeError(
            f"optimizers must be a list or tuple, got {type(optimizers).__name__}"
        )
    
    if len(optimizers) != num_models:
        raise ValueError(
            f"Number of optimizers ({len(optimizers)}) must match "
            f"number of models ({num_models})"
        )
    
    for i, optimizer in enumerate(optimizers):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"optimizers[{i}] must be a torch.optim.Optimizer, "
                f"got {type(optimizer).__name__}"
            )
    
    return list(optimizers)


def validate_data_loader(data_loader: Any, name: str = "data_loader") -> DataLoader:
    """
    Validate a PyTorch DataLoader.
    
    Args:
        data_loader: DataLoader to validate
        name: Parameter name for error messages
        
    Returns:
        Validated DataLoader
        
    Raises:
        TypeError: If data_loader is not a DataLoader
        ValueError: If DataLoader appears to be empty
    """
    if not isinstance(data_loader, DataLoader):
        raise TypeError(
            f"{name} must be a torch.utils.data.DataLoader, "
            f"got {type(data_loader).__name__}"
        )
    
    # Check if DataLoader has data
    if len(data_loader) == 0:
        raise ValueError(f"{name} appears to be empty (length 0)")
    
    return data_loader


def validate_batch_size(batch_size: Any, dataset_size: Optional[int] = None) -> int:
    """
    Validate batch size.
    
    Args:
        batch_size: Batch size to validate
        dataset_size: Optional dataset size to check against
        
    Returns:
        Validated batch size
        
    Raises:
        TypeError: If batch_size is not an integer
        ValueError: If batch_size is invalid
    """
    batch_size = validate_positive_int(batch_size, "batch_size")
    
    if dataset_size is not None and batch_size > dataset_size:
        raise ValueError(
            f"batch_size ({batch_size}) cannot be larger than "
            f"dataset size ({dataset_size})"
        )
    
    return batch_size


def validate_num_workers(num_workers: Any) -> int:
    """
    Validate number of worker processes.
    
    Args:
        num_workers: Number of workers to validate
        
    Returns:
        Validated number of workers
        
    Raises:
        TypeError: If num_workers is not an integer
        ValueError: If num_workers is negative
    """
    return validate_positive_int(num_workers, "num_workers", allow_zero=True)


def validate_epochs(epochs: Any, start_epoch: int = 1) -> int:
    """
    Validate number of training epochs.
    
    Args:
        epochs: Number of epochs to validate
        start_epoch: Starting epoch number
        
    Returns:
        Validated number of epochs
        
    Raises:
        TypeError: If epochs is not an integer
        ValueError: If epochs is invalid
    """
    epochs = validate_positive_int(epochs, "epochs")
    
    if start_epoch > epochs:
        raise ValueError(
            f"start_epoch ({start_epoch}) cannot be greater than "
            f"total epochs ({epochs})"
        )
    
    return epochs


def validate_learning_rate(lr: Any) -> float:
    """
    Validate learning rate.
    
    Args:
        lr: Learning rate to validate
        
    Returns:
        Validated learning rate
        
    Raises:
        TypeError: If lr is not a number
        ValueError: If lr is not positive
    """
    lr = validate_positive_float(lr, "learning_rate")
    
    # Warn if learning rate is unusually large
    if lr > 1.0:
        import warnings
        warnings.warn(
            f"Learning rate {lr} is unusually large (> 1.0). "
            f"This may cause training instability.",
            UserWarning
        )
    
    return lr


def validate_temperature(temperature: Any) -> float:
    """
    Validate temperature parameter for distillation.
    
    Args:
        temperature: Temperature to validate
        
    Returns:
        Validated temperature
        
    Raises:
        TypeError: If temperature is not a number
        ValueError: If temperature is not positive
    """
    temperature = validate_positive_float(temperature, "temperature")
    
    # Warn if temperature is outside typical range
    if temperature < 1.0:
        import warnings
        warnings.warn(
            f"Temperature {temperature} is less than 1.0. "
            f"Typical values are in range [1.0, 20.0].",
            UserWarning
        )
    elif temperature > 20.0:
        import warnings
        warnings.warn(
            f"Temperature {temperature} is greater than 20.0. "
            f"Very high temperatures may reduce effectiveness.",
            UserWarning
        )
    
    return temperature


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_ndim: int,
    name: str = "tensor"
) -> torch.Tensor:
    """
    Validate tensor dimensionality.
    
    Args:
        tensor: Tensor to validate
        expected_ndim: Expected number of dimensions
        name: Tensor name for error messages
        
    Returns:
        Validated tensor
        
    Raises:
        TypeError: If tensor is not a torch.Tensor
        ValueError: If tensor has wrong number of dimensions
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    
    if tensor.ndim != expected_ndim:
        raise ValueError(
            f"{name} must be {expected_ndim}D tensor, "
            f"got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}"
        )
    
    return tensor


def validate_weights(weights: Any, num_models: int) -> List[float]:
    """
    Validate a list of weights for model ensemble.
    
    Args:
        weights: List of weights to validate
        num_models: Expected number of weights
        
    Returns:
        Validated and normalized list of weights
        
    Raises:
        TypeError: If weights is not a list or contains non-numeric values
        ValueError: If number of weights doesn't match num_models or weights are invalid
    """
    if not isinstance(weights, (list, tuple)):
        raise TypeError(f"weights must be a list or tuple, got {type(weights).__name__}")
    
    if len(weights) != num_models:
        raise ValueError(
            f"Number of weights ({len(weights)}) must match "
            f"number of models ({num_models})"
        )
    
    validated_weights = []
    for i, w in enumerate(weights):
        if not isinstance(w, (int, float)) or isinstance(w, bool):
            raise TypeError(
                f"weights[{i}] must be a number, got {type(w).__name__}"
            )
        if w < 0:
            raise ValueError(f"weights[{i}] must be non-negative, got {w}")
        validated_weights.append(float(w))
    
    # Check that at least one weight is positive
    if sum(validated_weights) == 0:
        raise ValueError("at least one weight must be positive, got all zeros")
    
    return validated_weights


def validate_model_compatibility(models, device='cpu'):
    """
    Validate that all models are compatible for mutual learning.
    
    This function performs comprehensive validation:
    - Checks all models are torch.nn.Module instances
    - Verifies models have trainable parameters
    - Validates all models produce same output dimension
    - Provides detailed error messages showing exact mismatches
    
    Args:
        models: List of PyTorch models
        device: Device to use for validation ('cpu' or 'cuda')
        
    Returns:
        int: The validated output dimension shared by all models
        
    Raises:
        TypeError: If any model is not a torch.nn.Module
        ValueError: If models have no parameters or mismatched output dimensions
    """
    import torch
    
    if not isinstance(models, (list, tuple)):
        raise TypeError(f"models must be a list or tuple, got {type(models).__name__}")
    
    if len(models) == 0:
        raise ValueError("models list cannot be empty")
    
    # Validate all are torch modules
    for i, model in enumerate(models):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"models[{i}] must be a torch.nn.Module instance, "
                f"got {type(model).__name__}"
            )
    
    # Check each model has parameters
    model_param_counts = []
    for i, model in enumerate(models):
        param_count = sum(p.numel() for p in model.parameters())
        model_param_counts.append(param_count)
        if param_count == 0:
            raise ValueError(
                f"models[{i}] has no parameters. "
                f"All models must have trainable parameters for mutual learning."
            )
    
    # Determine input shape by inspecting first model's first layer
    # This is more reliable than guessing common shapes
    first_model = models[0]
    input_shape = None
    
    # Try to infer input shape from first layer
    for module in first_model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Convolutional input: batch_size=2, channels, height, width
            input_shape = (2, module.in_channels, 32, 32)
            break
        elif isinstance(module, torch.nn.Linear):
            # Linear input: batch_size=2, features
            input_shape = (2, module.in_features)
            break
    
    # Fallback: try common shapes if inspection failed
    if input_shape is None:
        test_shapes = [(2, 3, 32, 32), (2, 1, 28, 28), (2, 784), (2, 10)]
    else:
        test_shapes = [input_shape]
    
    # For mixed architectures, we need to try different shapes per model
    # Store validated output dimension for each model
    model_outputs = []
    
    for i, model in enumerate(models):
        model_validated = False
        model_error = None
        
        # Try to infer input shape for this specific model
        model_shape = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                model_shape = (2, module.in_channels, 32, 32)
                break
            elif isinstance(module, torch.nn.Linear):
                model_shape = (2, module.in_features)
                break
        
        # Use model-specific shape or try all common shapes
        if model_shape is not None:
            shapes_to_try = [model_shape]
        else:
            shapes_to_try = [(2, 3, 32, 32), (2, 1, 28, 28), (2, 784), (2, 10)]
        
        for shape in shapes_to_try:
            try:
                with torch.no_grad():
                    dummy_input = torch.randn(*shape).to(device)
                    
                    # Move input to model's device
                    try:
                        model_device = next(model.parameters()).device
                        model_input = dummy_input.to(model_device)
                    except StopIteration:
                        # Should not happen due to parameter count check above
                        raise ValueError(f"models[{i}] has no parameters")
                    
                    # Get output dimension
                    output = model(model_input)
                    if not isinstance(output, torch.Tensor):
                        raise ValueError(
                            f"models[{i}] returned {type(output).__name__}, "
                            f"expected torch.Tensor"
                        )
                    
                    output_dim = output.shape[-1]
                    model_outputs.append((i, output_dim))
                    model_validated = True
                    break  # Successfully validated this model
                    
            except (RuntimeError, ValueError) as e:
                # Shape mismatch or non-tensor output, try next shape
                model_error = str(e)
                # If it's a non-tensor error, re-raise immediately
                if "expected torch.Tensor" in str(e):
                    raise
                continue
        
        if not model_validated:
            raise ValueError(
                f"Could not validate models[{i}] with test inputs. "
                f"Last error: {model_error}"
            )
    
    # Check all output dimensions match
    unique_dims = set(dim for _, dim in model_outputs)
    if len(unique_dims) > 1:
        # Build detailed error message showing each model's dimension
        dim_details = []
        for model_idx, dim in model_outputs:
            param_count = model_param_counts[model_idx]
            dim_details.append(
                f"  models[{model_idx}]: output_dim={dim}, params={param_count:,}"
            )
        
        raise ValueError(
            f"all models must have the same output dimension for mutual learning.\n"
            f"Found {len(unique_dims)} different output dimensions:\n" +
            "\n".join(dim_details)
        )
    
    # Return the validated output dimension
    return model_outputs[0][1]
