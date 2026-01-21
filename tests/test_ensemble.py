"""
Unit tests for ensemble prediction utilities.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pydml.utils.ensemble import (
    ensemble_predict,
    average_predictions,
    voting_predictions,
    weighted_predictions,
    max_confidence_predictions,
    ensemble_accuracy,
    calibrate_ensemble_weights,
    get_prediction_diversity,
    EnsembleModel,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, num_classes=10, bias_class=None):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
        # Optionally bias towards a specific class for testing
        if bias_class is not None:
            with torch.no_grad():
                self.fc.weight[bias_class] *= 2
                self.fc.bias[bias_class] += 1
    
    def forward(self, x):
        return self.fc(x)


class TestEnsemblePredictions:
    """Test ensemble prediction functions."""
    
    def test_average_predictions(self):
        """Test averaging predictions from multiple models."""
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        inputs = torch.randn(4, 10)
        
        predictions = average_predictions(models, inputs)
        
        assert predictions.shape == (4, 10)
        # Probabilities should sum to 1
        assert torch.allclose(predictions.sum(dim=1), torch.ones(4), atol=1e-5)
        # All values should be positive
        assert (predictions >= 0).all()
    
    def test_voting_predictions(self):
        """Test majority voting predictions."""
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        inputs = torch.randn(4, 10)
        
        predictions = voting_predictions(models, inputs)
        
        assert predictions.shape == (4, 10)
        # Should be one-hot encoded
        assert (predictions.sum(dim=1) == 1).all()
        # Only 0s and 1s
        assert ((predictions == 0) | (predictions == 1)).all()
    
    def test_weighted_predictions(self):
        """Test weighted ensemble predictions."""
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        inputs = torch.randn(4, 10)
        weights = [0.5, 0.3, 0.2]
        
        predictions = weighted_predictions(models, inputs, weights)
        
        assert predictions.shape == (4, 10)
        # Probabilities should sum to 1
        assert torch.allclose(predictions.sum(dim=1), torch.ones(4), atol=1e-5)
    
    def test_weighted_predictions_validation(self):
        """Test that weighted predictions validates inputs."""
        models = [SimpleModel() for _ in range(3)]
        inputs = torch.randn(4, 10)
        
        # Wrong number of weights
        with pytest.raises(ValueError, match="must match"):
            weighted_predictions(models, inputs, [0.5, 0.5])
        
        # Weights don't sum to 1
        with pytest.raises(ValueError, match="sum to 1.0"):
            weighted_predictions(models, inputs, [0.3, 0.3, 0.3])
    
    def test_max_confidence_predictions(self):
        """Test max confidence ensemble."""
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        inputs = torch.randn(4, 10)
        
        predictions = max_confidence_predictions(models, inputs)
        
        assert predictions.shape == (4, 10)
        # Should be valid probabilities
        assert torch.allclose(predictions.sum(dim=1), torch.ones(4), atol=1e-5)
    
    def test_ensemble_predict_methods(self):
        """Test ensemble_predict with different methods."""
        models = [SimpleModel() for _ in range(3)]
        inputs = torch.randn(4, 10)
        
        # Test all methods
        for method in ['average', 'vote', 'max']:
            predictions = ensemble_predict(models, inputs, method=method)
            assert predictions.shape == (4, 10)
        
        # Test weighted method
        predictions = ensemble_predict(
            models, inputs, method='weighted', weights=[0.5, 0.3, 0.2]
        )
        assert predictions.shape == (4, 10)
    
    def test_ensemble_predict_invalid_method(self):
        """Test that invalid method raises error."""
        models = [SimpleModel() for _ in range(2)]
        inputs = torch.randn(4, 10)
        
        with pytest.raises(ValueError, match="Unknown ensemble method"):
            ensemble_predict(models, inputs, method='invalid')
    
    def test_ensemble_predict_weighted_no_weights(self):
        """Test that weighted method requires weights."""
        models = [SimpleModel() for _ in range(2)]
        inputs = torch.randn(4, 10)
        
        with pytest.raises(ValueError, match="weights must be provided"):
            ensemble_predict(models, inputs, method='weighted')


class TestEnsembleAccuracy:
    """Test ensemble accuracy computation."""
    
    def test_ensemble_accuracy(self):
        """Test computing ensemble accuracy."""
        # Create simple dataset
        inputs = torch.randn(20, 10)
        targets = torch.randint(0, 10, (20,))
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=4)
        
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        
        accuracy = ensemble_accuracy(
            models, dataloader, method='average', device='cpu'
        )
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100
    
    def test_ensemble_accuracy_different_methods(self):
        """Test accuracy with different ensemble methods."""
        inputs = torch.randn(20, 10)
        targets = torch.randint(0, 10, (20,))
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=4)
        
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        
        for method in ['average', 'vote', 'max']:
            accuracy = ensemble_accuracy(models, dataloader, method=method, device='cpu')
            assert 0 <= accuracy <= 100


class TestWeightCalibration:
    """Test weight calibration utilities."""
    
    def test_calibrate_ensemble_weights(self):
        """Test automatic weight calibration."""
        # Create validation data
        inputs = torch.randn(20, 10)
        targets = torch.randint(0, 10, (20,))
        dataset = TensorDataset(inputs, targets)
        val_loader = DataLoader(dataset, batch_size=4)
        
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        
        weights = calibrate_ensemble_weights(models, val_loader, device='cpu')
        
        assert len(weights) == 3
        # Weights should sum to 1
        assert abs(sum(weights) - 1.0) < 1e-5
        # All weights should be positive
        assert all(w > 0 for w in weights)
    
    def test_calibrate_with_different_performance(self):
        """Test that better models get higher weights."""
        # Create data where class 0 is most common
        inputs = torch.randn(40, 10)
        targets = torch.cat([torch.zeros(30, dtype=torch.long), 
                            torch.randint(1, 10, (10,))])
        dataset = TensorDataset(inputs, targets)
        val_loader = DataLoader(dataset, batch_size=8)
        
        # Model biased towards class 0 should perform better
        models = [
            SimpleModel(num_classes=10, bias_class=0),  # Should get higher weight
            SimpleModel(num_classes=10),
            SimpleModel(num_classes=10),
        ]
        
        weights = calibrate_ensemble_weights(models, val_loader, device='cpu')
        
        # First model should have highest weight
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]


class TestPredictionDiversity:
    """Test prediction diversity metrics."""
    
    def test_get_prediction_diversity(self):
        """Test prediction diversity measurement."""
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        inputs = torch.randn(20, 10)
        
        diversity = get_prediction_diversity(models, inputs)
        
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1
    
    def test_diversity_identical_models(self):
        """Test that identical models have low diversity."""
        # Create identical models
        model = SimpleModel(num_classes=10)
        models = [model, model, model]
        inputs = torch.randn(20, 10)
        
        diversity = get_prediction_diversity(models, inputs)
        
        # Identical models should have 0 diversity
        assert diversity == 0.0
    
    def test_diversity_different_models(self):
        """Test that different models have non-zero diversity."""
        # Create models with different biases
        models = [
            SimpleModel(num_classes=10, bias_class=0),
            SimpleModel(num_classes=10, bias_class=5),
            SimpleModel(num_classes=10, bias_class=9),
        ]
        inputs = torch.randn(20, 10)
        
        diversity = get_prediction_diversity(models, inputs)
        
        # Different models should have some diversity
        assert diversity > 0


class TestEnsembleModel:
    """Test EnsembleModel wrapper class."""
    
    def test_ensemble_model_creation(self):
        """Test creating EnsembleModel."""
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        ensemble = EnsembleModel(models, method='average')
        
        assert len(ensemble.models) == 3
        assert ensemble.method == 'average'
    
    def test_ensemble_model_forward(self):
        """Test forward pass through ensemble."""
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        ensemble = EnsembleModel(models, method='average')
        
        inputs = torch.randn(4, 10)
        outputs = ensemble(inputs)
        
        assert outputs.shape == (4, 10)
    
    def test_ensemble_model_predict(self):
        """Test getting class predictions."""
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        ensemble = EnsembleModel(models, method='average')
        
        inputs = torch.randn(4, 10)
        predictions = ensemble.predict(inputs)
        
        assert predictions.shape == (4,)
        assert (predictions >= 0).all()
        assert (predictions < 10).all()
    
    def test_ensemble_model_predict_proba(self):
        """Test getting probability predictions."""
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        ensemble = EnsembleModel(models, method='average')
        
        inputs = torch.randn(4, 10)
        probabilities = ensemble.predict_proba(inputs)
        
        assert probabilities.shape == (4, 10)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-5)
    
    def test_ensemble_model_with_weights(self):
        """Test ensemble with custom weights."""
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        weights = [0.5, 0.3, 0.2]
        ensemble = EnsembleModel(models, method='weighted', weights=weights)
        
        inputs = torch.randn(4, 10)
        outputs = ensemble(inputs)
        
        assert outputs.shape == (4, 10)
    
    def test_ensemble_model_different_methods(self):
        """Test ensemble with different methods."""
        models = [SimpleModel(num_classes=10) for _ in range(3)]
        
        for method in ['average', 'vote', 'max']:
            ensemble = EnsembleModel(models, method=method)
            inputs = torch.randn(4, 10)
            outputs = ensemble(inputs)
            assert outputs.shape == (4, 10)


class TestTemperatureScaling:
    """Test temperature parameter effects."""
    
    def test_temperature_in_average(self):
        """Test temperature scaling in averaging."""
        models = [SimpleModel(num_classes=10) for _ in range(2)]
        inputs = torch.randn(4, 10)
        
        # Different temperatures
        pred_t1 = average_predictions(models, inputs, temperature=1.0)
        pred_t2 = average_predictions(models, inputs, temperature=2.0)
        
        # Should produce different results
        assert not torch.allclose(pred_t1, pred_t2)
        
        # Both should be valid probabilities
        assert torch.allclose(pred_t1.sum(dim=1), torch.ones(4), atol=1e-5)
        assert torch.allclose(pred_t2.sum(dim=1), torch.ones(4), atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
