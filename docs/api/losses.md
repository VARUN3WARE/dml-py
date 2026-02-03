# Losses API

Loss functions for collaborative learning.

## Attention Transfer Loss

Loss function for attention transfer between networks.

```{eval-rst}
.. automodule:: pydml.losses.attention_transfer
   :members:
   :undoc-members:
   :show-inheritance:
```

### AttentionTransferLoss

```{eval-rst}
.. autoclass:: pydml.losses.attention_transfer.AttentionTransferLoss
   :members:
   :show-inheritance:
```

## Core Losses

See [Core API - Losses](core.md#losses) for:

- `CrossEntropyLoss`: Standard classification loss
- `KLDivergenceLoss`: Knowledge distillation loss
- `DMLLoss`: Deep mutual learning loss

## Custom Losses

Create custom loss functions by inheriting from `torch.nn.Module`:

```python
import torch.nn as nn

class MyCustomLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, outputs, targets):
        # Implement custom loss logic
        loss = ...
        return self.weight * loss
```
