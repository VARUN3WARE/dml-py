# Core API

The core module contains the base classes and fundamental components for collaborative learning.

## Base Trainer

```{eval-rst}
.. autoclass:: pydml.core.BaseCollaborativeTrainer
   :members:
   :undoc-members:
   :show-inheritance:
```

## Losses

```{eval-rst}
.. automodule:: pydml.core.losses
   :members:
   :undoc-members:
   :show-inheritance:
```

### CrossEntropyLoss

```{eval-rst}
.. autoclass:: pydml.core.losses.CrossEntropyLoss
   :members:
   :show-inheritance:
```

### KLDivergenceLoss

```{eval-rst}
.. autoclass:: pydml.core.losses.KLDivergenceLoss
   :members:
   :show-inheritance:
```

### DMLLoss

```{eval-rst}
.. autoclass:: pydml.core.losses.DMLLoss
   :members:
   :show-inheritance:
```

## Callbacks

```{eval-rst}
.. automodule:: pydml.core.callbacks
   :members:
   :undoc-members:
   :show-inheritance:
```

### EarlyStopping

```{eval-rst}
.. autoclass:: pydml.core.callbacks.EarlyStopping
   :members:
   :show-inheritance:
```

### ModelCheckpoint

```{eval-rst}
.. autoclass:: pydml.core.callbacks.ModelCheckpoint
   :members:
   :show-inheritance:
```

### TensorBoard

```{eval-rst}
.. autoclass:: pydml.core.callbacks.TensorBoard
   :members:
   :show-inheritance:
```
