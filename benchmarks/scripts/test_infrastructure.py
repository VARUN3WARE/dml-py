"""
Test script to verify benchmark infrastructure works correctly.

This script tests the core components without requiring full training.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from benchmarks import ExperimentConfig, BaselineTrainer, MetricsLogger


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(32 * 32 * 3, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_experiment_config():
    """Test ExperimentConfig creation and serialization."""
    print("\nTesting ExperimentConfig")
    
    # Create config
    config = ExperimentConfig(
        name='test_experiment',
        dataset='cifar10',
        model_type='simple',
        epochs=2,
        batch_size=32
    )
    
    print("Created config:")
    print(config)
    
    # Save and load
    config.save('benchmarks/configs/test_config.json')
    loaded = ExperimentConfig.load('benchmarks/configs/test_config.json')
    
    assert loaded.name == config.name
    assert loaded.epochs == config.epochs
    print("Config save/load verified")
    
    return config


def test_metrics_logger():
    """Test MetricsLogger functionality."""
    print("\nTesting MetricsLogger")
    
    logger = MetricsLogger('test_logger', output_dir='benchmarks/results')
    
    # Log some fake metrics
    for epoch in range(1, 6):
        logger.log_epoch(
            epoch=epoch,
            train_loss=2.0 / epoch,
            train_acc=0.3 * epoch / 5,
            val_loss=2.2 / epoch,
            val_acc=0.28 * epoch / 5,
            learning_rate=0.1 * (0.9 ** epoch)
        )
    
    print("Logged 5 epochs of metrics")
    
    # Save metrics
    logger.save()
    print("Saved metrics to JSON and CSV")
    
    # Get summary
    summary = logger.get_summary()
    print("\nSummary:")
    print(logger)
    
    assert summary['total_epochs'] == 5
    assert summary['best_epoch'] > 0
    print("MetricsLogger verified")
    
    return logger


def test_baseline_trainer():
    """Test BaselineTrainer with minimal training."""
    print("\nTesting BaselineTrainer")
    
    # Create simple model and config
    model = SimpleModel(num_classes=10)
    config = ExperimentConfig(
        name='test_trainer',
        dataset='cifar10',
        model_type='simple',
        epochs=2,
        batch_size=8,
        learning_rate=0.01
    )
    
    print("Created model and config")
    
    # Create dummy data loaders
    dummy_data = torch.randn(16, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (16,))
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(dummy_data, dummy_labels)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    print("Created dummy data loaders")
    
    # Create trainer
    trainer = BaselineTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    print("Created BaselineTrainer")
    
    # Run short training
    print("\nRunning 2 epochs of training...")
    results = trainer.train()
    
    print("\nTraining completed")
    print(f"Final val accuracy: {results['final_val_acc']:.2%}")
    print(f"Best val accuracy: {results['best_val_acc']:.2%}")
    
    # Test model save/load
    trainer.save_model('benchmarks/results/test_model.pth')
    print("Model saved")
    
    return trainer


def test_create_baseline_configs():
    """Test baseline config generation."""
    print("\nTesting Baseline Config Generation")
    
    from benchmarks.experiment_config import create_baseline_configs
    
    configs = create_baseline_configs(output_dir='benchmarks/configs')
    
    print(f"Created {len(configs)} baseline configurations:")
    for config in configs:
        print(f"  {config.name}")
    
    assert len(configs) == 5
    print("All baseline configs generated")
    
    return configs


def main():
    """Run all tests."""
    print("\nBenchmark Infrastructure Test Suite")
    
    try:
        # Test each component
        test_experiment_config()
        test_metrics_logger()
        test_create_baseline_configs()
        test_baseline_trainer()
        
        # Final summary
        print("\nAll tests passed")
        print("\nBenchmark infrastructure is ready for use")
        print("\nNext steps:")
        print("  1. Run full CIFAR-10 baseline experiments")
        print("  2. Compare with DML training")
        print("  3. Generate benchmark report\n")
        
    except Exception as e:
        print("\nTest failed")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
