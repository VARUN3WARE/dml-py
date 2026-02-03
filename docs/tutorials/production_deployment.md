# Production Deployment Tutorial

Deploy PyDML models in production environments.

## Model Export

### Saving Trained Models

```python
# Save individual model
torch.save(trainer.models[0].state_dict(), 'model_0.pth')

# Save entire checkpoint
trainer.save_checkpoint('checkpoint.pth', epoch=200)
```

### Loading Models

```python
# Load model
model = resnet32(num_classes=10)
model.load_state_dict(torch.load('model_0.pth'))
model.eval()
```

## Inference

### Single Model Inference

```python
import torch

model.eval()
with torch.no_grad():
    outputs = model(images)
    predictions = outputs.argmax(dim=1)
```

### Ensemble Inference

```python
# Load all models
models = []
for i in range(3):
    model = resnet32(num_classes=10)
    model.load_state_dict(torch.load(f'model_{i}.pth'))
    model.eval()
    models.append(model)

# Ensemble prediction
with torch.no_grad():
    outputs = [model(images) for model in models]
    ensemble_output = torch.stack(outputs).mean(dim=0)
    predictions = ensemble_output.argmax(dim=1)
```

## Optimization

Tips for production deployment coming soon.
