# Utilities API

Utility functions and helpers for data loading, metrics, and more.

## Data Loading

CIFAR dataset loaders with validation splits.

```{eval-rst}
.. automodule:: pydml.utils.data
   :members:
   :undoc-members:
```

### get_cifar10_loaders

```{eval-rst}
.. autofunction:: pydml.utils.data.get_cifar10_loaders
```

### get_cifar100_loaders

```{eval-rst}
.. autofunction:: pydml.utils.data.get_cifar100_loaders
```

## Metrics

Evaluation metrics for model performance.

```{eval-rst}
.. automodule:: pydml.utils.metrics
   :members:
   :undoc-members:
```

### Metric Functions

```{eval-rst}
.. autofunction:: pydml.utils.metrics.accuracy

.. autofunction:: pydml.utils.metrics.ensemble_accuracy

.. autofunction:: pydml.utils.metrics.ensemble_diversity

.. autofunction:: pydml.utils.metrics.expected_calibration_error

.. autofunction:: pydml.utils.metrics.confidence_entropy
```

## Logging

Logging utilities for experiment tracking.

```{eval-rst}
.. automodule:: pydml.utils.logging
   :members:
   :undoc-members:
```

## Input Validation

Comprehensive input validation functions.

```{eval-rst}
.. automodule:: pydml.utils.validation
   :members:
   :undoc-members:
```

### Validation Functions

```{eval-rst}
.. autofunction:: pydml.utils.validation.validate_positive_int

.. autofunction:: pydml.utils.validation.validate_positive_float

.. autofunction:: pydml.utils.validation.validate_probability

.. autofunction:: pydml.utils.validation.validate_range

.. autofunction:: pydml.utils.validation.validate_string_choice

.. autofunction:: pydml.utils.validation.validate_device

.. autofunction:: pydml.utils.validation.validate_model_list

.. autofunction:: pydml.utils.validation.validate_optimizer_list

.. autofunction:: pydml.utils.validation.validate_data_loader

.. autofunction:: pydml.utils.validation.validate_batch_size

.. autofunction:: pydml.utils.validation.validate_num_workers

.. autofunction:: pydml.utils.validation.validate_epochs

.. autofunction:: pydml.utils.validation.validate_learning_rate

.. autofunction:: pydml.utils.validation.validate_temperature

.. autofunction:: pydml.utils.validation.validate_tensor_shape

.. autofunction:: pydml.utils.validation.validate_weights
```
