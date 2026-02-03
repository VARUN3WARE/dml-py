# Analysis API

Tools for analyzing and visualizing collaborative learning.

## Training Monitor

Monitor training progress and detect overfitting.

```{eval-rst}
.. automodule:: pydml.analysis
   :members:
   :undoc-members:
```

### TrainingMonitor

```{eval-rst}
.. autoclass:: pydml.analysis.TrainingMonitor
   :members:
   :show-inheritance:
```

### OverfittingStatus

```{eval-rst}
.. autoclass:: pydml.analysis.OverfittingStatus
   :members:
   :undoc-members:
```

### OverfittingReport

```{eval-rst}
.. autoclass:: pydml.analysis.OverfittingReport
   :members:
   :undoc-members:
```

## Robustness Testing

Evaluate model robustness to noise and adversarial perturbations.

```{eval-rst}
.. automodule:: pydml.analysis.robustness
   :members:
   :undoc-members:
```

### test_robustness

```{eval-rst}
.. autofunction:: pydml.analysis.robustness.test_robustness
```

## Visualization

Visualization utilities for training analysis.

```{eval-rst}
.. automodule:: pydml.analysis.visualization
   :members:
   :undoc-members:
```

### Plotting Functions

```{eval-rst}
.. autofunction:: pydml.analysis.visualization.plot_training_history

.. autofunction:: pydml.analysis.visualization.plot_model_comparison

.. autofunction:: pydml.analysis.visualization.plot_loss_landscape

.. autofunction:: pydml.analysis.visualization.plot_ensemble_diversity

.. autofunction:: pydml.analysis.visualization.plot_confidence_distribution

.. autofunction:: pydml.analysis.visualization.plot_peer_influence
```
