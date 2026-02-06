"""
Run a single baseline experiment on CIFAR-10.

This script trains a single model using the baseline trainer and saves
all results for comparison with DML methods.

Usage:
    python run_baseline.py --config configs/baseline_resnet32_cifar10.json
    python run_baseline.py --model resnet32 --epochs 200
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from pydml.models.cifar import resnet32, mobilenet, wrn
from pydml.utils.reproducibility import set_seed
from benchmarks import ExperimentConfig, BaselineTrainer
from benchmarks.data_utils import get_cifar10_loaders, get_dataset_info


def get_model(model_type: str, num_classes: int) -> torch.nn.Module:
    """
    Create model based on type string.
    
    Args:
        model_type: Model architecture name
        num_classes: Number of output classes
        
    Returns:
        PyTorch model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'resnet32':
        return resnet32(num_classes=num_classes)
    elif model_type == 'mobilenet':
        return mobilenet(num_classes=num_classes)
    elif model_type.startswith('wrn'):
        # Parse WRN-depth-width format
        if '-' in model_type:
            parts = model_type.split('-')
            depth = int(parts[1])
            width = int(parts[2])
        else:
            depth, width = 28, 2
        return wrn(depth=depth, width=width, num_classes=num_classes)
    else:
        raise ValueError(f"unknown model type: {model_type}")


def run_experiment(config: ExperimentConfig) -> dict:
    """
    Run a complete baseline experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with experiment results
    """
    print(f"\nExperiment: {config.name}")
    print(f"Model: {config.model_type}")
    print(f"Dataset: {config.dataset}")
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Get dataset info
    dataset_info = get_dataset_info(config.dataset)
    num_classes = dataset_info['num_classes']
    
    # Create data loaders
    print("\nLoading data...")
    if config.dataset == 'cifar10':
        train_loader, test_loader, _ = get_cifar10_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            use_augmentation=config.use_augmentation
        )
    else:
        raise ValueError(f"unsupported dataset: {config.dataset}")
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nCreating model: {config.model_type}")
    model = get_model(config.model_type, num_classes)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = BaselineTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=test_loader
    )
    
    # Run training
    results = trainer.train()
    
    # Save model checkpoint
    checkpoint_dir = Path('benchmarks/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config.name}_best.pth"
    trainer.save_model(str(checkpoint_path))
    print(f"\nModel saved to {checkpoint_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run baseline experiment on CIFAR-10')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model', type=str, default='resnet32', 
                       help='Model type (resnet32, mobilenet, wrn-28-2)')
    parser.add_argument('--epochs', type=int, default=200, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, 
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, 
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--name', type=str, 
                       help='Experiment name (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = ExperimentConfig.load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        # Create config from arguments
        name = args.name or f"baseline_{args.model}_cifar10"
        config = ExperimentConfig(
            name=name,
            dataset='cifar10',
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed
        )
        
        # Save config
        config_dir = Path('benchmarks/configs')
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{config.name}.json"
        config.save(str(config_path))
        print(f"Saved config to {config_path}")
    
    # Run experiment
    results = run_experiment(config)
    
    # Print final summary
    print("\nFinal Results:")
    print(f"  Best accuracy: {results['best_val_acc']:.2%} (epoch {results['best_epoch']})")
    print(f"  Final accuracy: {results['final_val_acc']:.2%}")
    print(f"  Total time: {results['total_time_seconds']:.1f}s")
    print(f"  Avg time/epoch: {results['avg_epoch_time']:.1f}s")


if __name__ == '__main__':
    main()
