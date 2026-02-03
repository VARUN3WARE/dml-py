# Trainers API

PyDML provides several trainer classes for different collaborative learning scenarios.

## DML Trainer

Deep Mutual Learning trainer for collaborative training of multiple networks.

```{eval-rst}
.. autoclass:: pydml.trainers.DMLTrainer
   :members:
   :undoc-members:
   :show-inheritance:
```

### DML Configuration

```{eval-rst}
.. autoclass:: pydml.trainers.dml.DMLConfig
   :members:
   :undoc-members:
```

## Distillation Trainer

Knowledge distillation from teacher to student network.

```{eval-rst}
.. autoclass:: pydml.trainers.DistillationTrainer
   :members:
   :undoc-members:
   :show-inheritance:
```

### Distillation Configuration

```{eval-rst}
.. autoclass:: pydml.trainers.distillation.DistillationConfig
   :members:
   :undoc-members:
```

## Co-Distillation Trainer

Combines teacher distillation with peer learning among students.

```{eval-rst}
.. autoclass:: pydml.trainers.CoDistillationTrainer
   :members:
   :undoc-members:
   :show-inheritance:
```

### Co-Distillation Configuration

```{eval-rst}
.. autoclass:: pydml.trainers.co_distillation.CoDistillationConfig
   :members:
   :undoc-members:
```

## Feature-Based DML Trainer

Deep Mutual Learning using intermediate feature representations.

```{eval-rst}
.. autoclass:: pydml.trainers.FeatureDMLTrainer
   :members:
   :undoc-members:
   :show-inheritance:
```
