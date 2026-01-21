"""
Ensemble Prediction Utilities Demo

This example demonstrates how to combine predictions from multiple trained models
using various ensemble methods after Deep Mutual Learning training.

Ensemble methods can improve performance by leveraging the diversity of predictions
from different models.
"""

import torch
import torch.nn as nn
from pydml.models.cifar import resnet32, mobilenet_v2, vgg16
from pydml.utils.ensemble import (
    ensemble_predict,
    average_predictions,
    voting_predictions,
    weighted_predictions,
    calibrate_ensemble_weights,
    get_prediction_diversity,
    EnsembleModel,
    ensemble_accuracy,
)


def demonstrate_ensemble_methods():
    """Demonstrate various ensemble prediction methods."""
    print("=" * 80)
    print("ENSEMBLE PREDICTION UTILITIES DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create ensemble of different architectures
    print("1. Creating Model Ensemble")
    print("-" * 80)
    
    models = [
        resnet32(num_classes=10),
        mobilenet_v2(num_classes=10),
        vgg16(num_classes=10),
    ]
    
    print(f"Created ensemble with {len(models)} models:")
    print("  - ResNet32")
    print("  - MobileNetV2")
    print("  - VGG16")
    print()
    
    # Create dummy input
    batch_size = 8
    inputs = torch.randn(batch_size, 3, 32, 32)
    print(f"Input shape: {inputs.shape}")
    print()
    
    # 2. Average Predictions (Soft Voting)
    print("2. Average Predictions (Soft Voting)")
    print("-" * 80)
    print("Averages probability distributions from all models")
    print()
    
    avg_predictions = average_predictions(models, inputs)
    print(f"Output shape: {avg_predictions.shape}")
    print(f"Sum of probabilities per sample: {avg_predictions.sum(dim=1)[0]:.4f}")
    print(f"Predicted classes: {avg_predictions.argmax(dim=1).tolist()}")
    print(f"Max confidences: {avg_predictions.max(dim=1)[0].tolist()[:3]}")
    print()
    
    # 3. Majority Voting (Hard Voting)
    print("3. Majority Voting (Hard Voting)")
    print("-" * 80)
    print("Each model votes for one class, majority wins")
    print()
    
    vote_predictions = voting_predictions(models, inputs)
    print(f"Output shape: {vote_predictions.shape}")
    print(f"Predicted classes: {vote_predictions.argmax(dim=1).tolist()}")
    print(f"One-hot encoded: {vote_predictions[0].tolist()}")
    print()
    
    # 4. Weighted Ensemble
    print("4. Weighted Ensemble")
    print("-" * 80)
    print("Assign different weights to models based on their performance")
    print()
    
    weights = [0.5, 0.3, 0.2]  # ResNet gets highest weight
    print(f"Weights: {weights}")
    
    weighted_preds = weighted_predictions(models, inputs, weights)
    print(f"Output shape: {weighted_preds.shape}")
    print(f"Predicted classes: {weighted_preds.argmax(dim=1).tolist()}")
    print()
    
    # 5. ensemble_predict() - Unified Interface
    print("5. Unified Ensemble Interface")
    print("-" * 80)
    print("Use ensemble_predict() with different methods")
    print()
    
    methods = ['average', 'vote', 'weighted', 'max']
    for method in methods:
        if method == 'weighted':
            preds = ensemble_predict(models, inputs, method=method, weights=weights)
        else:
            preds = ensemble_predict(models, inputs, method=method)
        
        predicted_classes = preds.argmax(dim=1)
        print(f"  {method:12s}: {predicted_classes.tolist()}")
    print()
    
    # 6. Automatic Weight Calibration
    print("6. Automatic Weight Calibration")
    print("-" * 80)
    print("Calibrate weights based on validation performance")
    print()
    
    # Create dummy validation data
    val_inputs = torch.randn(40, 3, 32, 32)
    val_targets = torch.randint(0, 10, (40,))
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
    
    calibrated_weights = calibrate_ensemble_weights(models, val_loader, device='cpu')
    print(f"Calibrated weights: {[f'{w:.4f}' for w in calibrated_weights]}")
    print(f"Sum of weights: {sum(calibrated_weights):.4f}")
    print()
    
    # 7. Prediction Diversity
    print("7. Prediction Diversity Analysis")
    print("-" * 80)
    print("Measure how diverse the model predictions are")
    print()
    
    diversity = get_prediction_diversity(models, inputs)
    print(f"Diversity score: {diversity:.4f}")
    print(f"Interpretation: {diversity*100:.1f}% of predictions differ between models")
    print()
    
    if diversity > 0.3:
        print("✓ High diversity - good for ensemble performance!")
    elif diversity > 0.1:
        print("○ Moderate diversity - ensemble may help")
    else:
        print("✗ Low diversity - models are too similar")
    print()
    
    # 8. EnsembleModel Wrapper
    print("8. EnsembleModel Wrapper Class")
    print("-" * 80)
    print("Convenient nn.Module wrapper for ensemble inference")
    print()
    
    # Create ensemble model
    ensemble_avg = EnsembleModel(models, method='average')
    ensemble_weighted = EnsembleModel(models, method='weighted', weights=weights)
    
    print("Created two ensemble models:")
    print(f"  - Average ensemble: {len(ensemble_avg.models)} models")
    print(f"  - Weighted ensemble: {len(ensemble_weighted.models)} models")
    print()
    
    # Forward pass
    outputs_avg = ensemble_avg(inputs)
    outputs_weighted = ensemble_weighted(inputs)
    
    print(f"Average ensemble output shape: {outputs_avg.shape}")
    print(f"Weighted ensemble output shape: {outputs_weighted.shape}")
    print()
    
    # Get predictions
    pred_classes = ensemble_avg.predict(inputs)
    pred_probs = ensemble_avg.predict_proba(inputs)
    
    print(f"Predicted classes: {pred_classes.tolist()}")
    print(f"Probability shape: {pred_probs.shape}")
    print()
    
    # 9. Ensemble Accuracy Computation
    print("9. Computing Ensemble Accuracy")
    print("-" * 80)
    
    # Create test dataset
    test_inputs = torch.randn(100, 3, 32, 32)
    test_targets = torch.randint(0, 10, (100,))
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
    
    print("Computing accuracy on test set (random data for demo)...")
    
    for method in ['average', 'vote', 'max']:
        acc = ensemble_accuracy(models, test_loader, method=method, device='cpu')
        print(f"  {method:12s} ensemble: {acc:.2f}%")
    
    # With calibrated weights
    acc_weighted = ensemble_accuracy(
        models, test_loader, method='weighted', 
        weights=calibrated_weights, device='cpu'
    )
    print(f"  {'weighted':12s} ensemble: {acc_weighted:.2f}%")
    print()
    
    # 10. Temperature Scaling
    print("10. Temperature Scaling")
    print("-" * 80)
    print("Control prediction confidence with temperature")
    print()
    
    temps = [0.5, 1.0, 2.0]
    print("Temperature effect on prediction confidence:")
    for temp in temps:
        preds = average_predictions(models, inputs[:1], temperature=temp)
        max_prob = preds.max().item()
        print(f"  T={temp:.1f}: max confidence = {max_prob:.4f}")
    
    print()
    print("Lower temperature → sharper (more confident) predictions")
    print("Higher temperature → softer (less confident) predictions")
    print()
    
    # 11. Comparison of Methods
    print("11. Method Comparison Summary")
    print("-" * 80)
    
    comparison = {
        'Average (Soft Voting)': {
            'Speed': 'Fast',
            'Accuracy': 'High',
            'Use Case': 'General purpose, best overall',
        },
        'Voting (Hard Voting)': {
            'Speed': 'Fastest',
            'Accuracy': 'Moderate',
            'Use Case': 'When probability calibration is poor',
        },
        'Weighted Ensemble': {
            'Speed': 'Fast',
            'Accuracy': 'Highest',
            'Use Case': 'When model performance varies significantly',
        },
        'Max Confidence': {
            'Speed': 'Moderate',
            'Accuracy': 'Variable',
            'Use Case': 'When one model is expert at each sample',
        },
    }
    
    for method, props in comparison.items():
        print(f"\n{method}:")
        for key, value in props.items():
            print(f"  {key:12s}: {value}")
    
    print()
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("  ✓ Average predictions works best in most cases")
    print("  ✓ Use weighted ensemble when models have different accuracies")
    print("  ✓ Calibrate weights automatically with validation data")
    print("  ✓ Higher diversity → better ensemble performance")
    print("  ✓ EnsembleModel provides convenient nn.Module interface")
    print()
    print("Usage Pattern:")
    print("  1. Train models with DML: trainer.fit(train_loader, val_loader)")
    print("  2. Get trained models: models = trainer.models")
    print("  3. Create ensemble: ensemble = EnsembleModel(models, method='average')")
    print("  4. Make predictions: predictions = ensemble(inputs)")
    print()


if __name__ == '__main__':
    demonstrate_ensemble_methods()
