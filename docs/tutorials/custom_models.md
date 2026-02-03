# Custom Models Tutorial

Learn how to create custom models compatible with PyDML.

## Model Requirements

PyDML works with any `torch.nn.Module`:

```python
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ... more layers
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

## Using Custom Models with DML

```python
from pydml import DMLTrainer

# Create custom models
models = [MyCustomModel(num_classes=10) for _ in range(3)]

# Use with DML
trainer = DMLTrainer(models, device='cuda')
trainer.fit(train_loader, val_loader, epochs=200)
```

That's it! No special modifications needed.
