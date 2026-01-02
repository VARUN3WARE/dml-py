"""
Tests for DMLTrainer input validation.
"""

import pytest
import torch
import torch.nn as nn
from pydml import DMLTrainer


# Helper test models
class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x):
        return self.fc(x)


class InvalidModel:
    """Not an nn.Module"""

    pass


# Validation tests


def test_models_must_be_list_or_tuple():
    """models must be list or tuple"""
    with pytest.raises(TypeError):
        DMLTrainer(models=SimpleModel(), device="cpu")


def test_at_least_two_models_required():
    """DML requires at least two models"""
    models = [SimpleModel()]
    with pytest.raises(ValueError):
        DMLTrainer(models=models, device="cpu")


def test_all_models_must_be_nn_module():
    """All items must be nn.Module"""
    models = [SimpleModel(), InvalidModel()]
    with pytest.raises(TypeError):
        DMLTrainer(models=models, device="cpu")


def test_output_dimension_mismatch_raises_error():
    """All models must have same output dimension"""
    models = [
        SimpleModel(num_classes=10),
        SimpleModel(num_classes=20),
    ]
    with pytest.raises(ValueError):
        DMLTrainer(models=models, device="cpu")


def test_valid_models_pass_validation():
    """Valid configuration should initialize successfully"""
    models = [
        SimpleModel(num_classes=10),
        SimpleModel(num_classes=10),
    ]
    trainer = DMLTrainer(models=models, device="cpu")
    assert trainer.num_models == 2
