"""
Ensemble prediction utilities for combining predictions from multiple trained models.

This module provides various strategies for combining predictions from an ensemble
of models trained with Deep Mutual Learning or other collaborative methods.
"""

from typing import List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ensemble_predict(
    models: List[nn.Module],
    inputs: torch.Tensor,
    method: str = 'average',
    weights: Optional[List[float]] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Combine predictions from multiple models using various ensemble methods.
    
    Args:
        models: List of trained PyTorch models
        inputs: Input tensor [batch_size, ...]
        method: Ensemble method ('average', 'vote', 'weighted', 'max')
        weights: Optional weights for weighted ensemble (must sum to 1)
        temperature: Temperature for softmax (default: 1.0)
    
    Returns:
        Combined predictions [batch_size, num_classes]
    
    Example:
        >>> models = [model1, model2, model3]
        >>> predictions = ensemble_predict(models, inputs, method='average')
    """
    if method == 'average':
        return average_predictions(models, inputs, temperature)
    elif method == 'vote':
        return voting_predictions(models, inputs)
    elif method == 'weighted':
        if weights is None:
            raise ValueError("weights must be provided for weighted ensemble")
        return weighted_predictions(models, inputs, weights, temperature)
    elif method == 'max':
        return max_confidence_predictions(models, inputs)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def average_predictions(
    models: List[nn.Module],
    inputs: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Average the softmax probabilities from all models.
    
    Args:
        models: List of trained models
        inputs: Input tensor
        temperature: Temperature for softmax (default: 1.0)
    
    Returns:
        Averaged probability predictions
    """
    predictions = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            logits = model(inputs)
            probs = F.softmax(logits / temperature, dim=1)
            predictions.append(probs)
    
    # Average probabilities
    avg_probs = torch.stack(predictions).mean(dim=0)
    return avg_probs


def voting_predictions(
    models: List[nn.Module],
    inputs: torch.Tensor,
) -> torch.Tensor:
    """
    Use majority voting to combine predictions (hard voting).
    
    Each model votes for one class, and the class with most votes wins.
    
    Args:
        models: List of trained models
        inputs: Input tensor
    
    Returns:
        One-hot encoded predictions based on majority vote
    """
    batch_size = inputs.size(0)
    votes = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            logits = model(inputs)
            predictions = logits.argmax(dim=1)
            votes.append(predictions)
    
    # Stack votes: [num_models, batch_size]
    votes = torch.stack(votes)
    
    # Find majority vote for each sample
    num_classes = models[0](inputs[:1]).size(1)
    final_predictions = torch.zeros(batch_size, num_classes, device=inputs.device)
    
    for i in range(batch_size):
        sample_votes = votes[:, i]
        # Count votes for each class
        vote_counts = torch.bincount(sample_votes, minlength=num_classes)
        winning_class = vote_counts.argmax()
        final_predictions[i, winning_class] = 1.0
    
    return final_predictions


def weighted_predictions(
    models: List[nn.Module],
    inputs: torch.Tensor,
    weights: List[float],
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Combine predictions using weighted average of probabilities.
    
    Useful when some models are known to perform better than others.
    
    Args:
        models: List of trained models
        inputs: Input tensor
        weights: List of weights (must sum to 1.0)
        temperature: Temperature for softmax
    
    Returns:
        Weighted averaged predictions
    """
    if len(models) != len(weights):
        raise ValueError(f"Number of models ({len(models)}) must match number of weights ({len(weights)})")
    
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
    
    predictions = []
    
    with torch.no_grad():
        for model, weight in zip(models, weights):
            model.eval()
            logits = model(inputs)
            probs = F.softmax(logits / temperature, dim=1)
            predictions.append(probs * weight)
    
    # Sum weighted probabilities
    weighted_probs = torch.stack(predictions).sum(dim=0)
    return weighted_probs


def max_confidence_predictions(
    models: List[nn.Module],
    inputs: torch.Tensor,
) -> torch.Tensor:
    """
    Select prediction from the model with highest confidence for each sample.
    
    Args:
        models: List of trained models
        inputs: Input tensor
    
    Returns:
        Predictions from most confident model for each sample
    """
    batch_size = inputs.size(0)
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            max_probs, _ = probs.max(dim=1)
            all_predictions.append(probs)
            all_confidences.append(max_probs)
    
    # Stack: [num_models, batch_size, num_classes]
    all_predictions = torch.stack(all_predictions)
    # Stack: [num_models, batch_size]
    all_confidences = torch.stack(all_confidences)
    
    # Find most confident model for each sample
    most_confident_models = all_confidences.argmax(dim=0)
    
    # Select predictions from most confident model
    final_predictions = torch.zeros_like(all_predictions[0])
    for i in range(batch_size):
        model_idx = most_confident_models[i]
        final_predictions[i] = all_predictions[model_idx, i]
    
    return final_predictions


def ensemble_accuracy(
    models: List[nn.Module],
    dataloader: torch.utils.data.DataLoader,
    method: str = 'average',
    weights: Optional[List[float]] = None,
    device: str = 'cuda',
) -> float:
    """
    Compute ensemble accuracy on a dataset.
    
    Args:
        models: List of trained models
        dataloader: DataLoader with test data
        method: Ensemble method to use
        weights: Optional weights for weighted ensemble
        device: Device to run inference on
    
    Returns:
        Accuracy as a percentage
    """
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        predictions = ensemble_predict(models, inputs, method, weights)
        predicted_classes = predictions.argmax(dim=1)
        
        correct += (predicted_classes == targets).sum().item()
        total += targets.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy


def calibrate_ensemble_weights(
    models: List[nn.Module],
    val_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
) -> List[float]:
    """
    Calibrate ensemble weights based on validation set performance.
    
    Weights are proportional to each model's accuracy on the validation set.
    
    Args:
        models: List of trained models
        val_loader: Validation data loader
        device: Device to run inference on
    
    Returns:
        List of calibrated weights that sum to 1.0
    """
    accuracies = []
    
    # Compute accuracy for each model
    for model in models:
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)
                
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        accuracy = correct / total
        accuracies.append(accuracy)
    
    # Convert accuracies to weights (softmax for smooth distribution)
    accuracies = torch.tensor(accuracies)
    weights = F.softmax(accuracies * 10, dim=0)  # Scale by 10 for sharper distribution
    
    return weights.tolist()


def get_prediction_diversity(
    models: List[nn.Module],
    inputs: torch.Tensor,
) -> float:
    """
    Measure prediction diversity among models.
    
    Higher diversity indicates models are making different predictions,
    which can lead to better ensemble performance.
    
    Args:
        models: List of trained models
        inputs: Input tensor
    
    Returns:
        Diversity score (0 to 1, higher is more diverse)
    """
    predictions = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            logits = model(inputs)
            pred_classes = logits.argmax(dim=1)
            predictions.append(pred_classes)
    
    # Stack predictions: [num_models, batch_size]
    predictions = torch.stack(predictions)
    
    # Calculate disagreement rate
    total_pairs = 0
    disagreements = 0
    
    num_models = len(models)
    for i in range(num_models):
        for j in range(i + 1, num_models):
            total_pairs += 1
            disagreements += (predictions[i] != predictions[j]).float().mean().item()
    
    diversity = disagreements / total_pairs if total_pairs > 0 else 0.0
    return diversity


class EnsembleModel(nn.Module):
    """
    Wrapper class for ensemble of models with built-in prediction combination.
    
    Example:
        >>> models = [model1, model2, model3]
        >>> ensemble = EnsembleModel(models, method='average')
        >>> predictions = ensemble(inputs)
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        method: str = 'average',
        weights: Optional[List[float]] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        self.weights = weights
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        return ensemble_predict(
            list(self.models),
            x,
            method=self.method,
            weights=self.weights,
            temperature=self.temperature,
        )
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        probs = self.forward(x)
        return probs.argmax(dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        return self.forward(x)
