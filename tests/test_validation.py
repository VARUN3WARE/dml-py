"""
Tests for DMLTrainer input validation.
"""

import pytest
import torch
import torch.nn as nn
from pydml.trainers import DMLTrainer


def test_dml_requires_list():
    """Test that DMLTrainer rejects non-list input"""
    model = nn.Linear(10, 5)
    with pytest.raises(TypeError, match="must be a list or tuple"):
        trainer = DMLTrainer(model, device='cpu')


def test_dml_requires_at_least_2_models():
    """Test that DMLTrainer requires at least 2 models"""
    models = [nn.Linear(10, 5)]
    with pytest.raises(ValueError, match="at least 2 models"):
        trainer = DMLTrainer(models, device='cpu')


def test_dml_requires_nn_modules():
    """Test that all models must be nn.Module instances"""
    models = [nn.Linear(10, 5), "not a model"]
    with pytest.raises(TypeError, match="torch.nn.Module"):
        trainer = DMLTrainer(models, device='cpu')


def test_dml_requires_matching_output_dims():
    """Test that all models must have same output dimension"""
    models = [
        nn.Sequential(nn.Flatten(), nn.Linear(3072, 5)),
        nn.Sequential(nn.Flatten(), nn.Linear(3072, 3))
    ]
    with pytest.raises(ValueError, match="same output dimension"):
        trainer = DMLTrainer(models, device='cpu')


def test_dml_accepts_valid_models():
    """Test that valid models are accepted"""
    models = [
        nn.Sequential(nn.Flatten(), nn.Linear(3072, 5)),
        nn.Sequential(nn.Flatten(), nn.Linear(3072, 5))
    ]
    trainer = DMLTrainer(models, device='cpu')
    assert len(trainer.models) == 2
