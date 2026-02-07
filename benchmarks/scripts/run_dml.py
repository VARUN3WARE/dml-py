"""
DML Experiment Runner

Script for running Deep Mutual Learning experiments with multiple models.
Supports 2+ model configurations and comprehensive metrics tracking.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydml.models.cifar import resnet32, mobilenet_v2, wrn_28_10
from pydml.trainers.dml import DMLTrainer, DMLConfig
from pydml.utils.reproducibility import set_seed
from benchmarks.experiment_config import ExperimentConfig
from benchmarks.metrics_logger import MetricsLogger
from benchmarks.data_utils import get_cifar10_loaders, get_cifar100_loaders


def create_models(model_type: str, num_models: int, num_classes: int):
    """Create multiple models of the same or mixed types."""
    models = []
    
    model_factory = {
        'resnet32': lambda: resnet32(num_classes=num_classes),
        'mobilenet': lambda: mobilenet_v2(num_classes=num_classes),
        'wrn': lambda: wrn_28_10(num_classes=num_classes),
    }
    
    if model_type not in model_factory:
        raise ValueError(f"Unknown model type: {model_type}")
    
    for i in range(num_models):
        model = model_factory[model_type]()
        models.append(model)
    
    return models


def get_data_loaders(config: ExperimentConfig):
    """Get data loaders based on dataset."""
    if config.dataset == 'cifar10':
        return get_cifar10_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            use_augmentation=config.use_augmentation,
        )
    elif config.dataset == 'cifar100':
        return get_cifar100_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            use_augmentation=config.use_augmentation,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")





def run_dml_experiment(config_path: str):
    """Run a DML experiment from configuration file."""
    
    # Load configuration
    config = ExperimentConfig.load(config_path)
    print(f"Running DML experiment: {config.name}")
    print(f"Models: {config.num_models}x {config.model_type}")
    print(f"Dataset: {config.dataset}")
    
    # Set random seed
    set_seed(config.seed)
    
    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders(config)
    num_classes = 100 if config.dataset == 'cifar100' else 10
    
    # Create models
    models = create_models(config.model_type, config.num_models, num_classes)
    total_params = sum(sum(p.numel() for p in m.parameters()) for m in models)
    print(f"Total parameters across {config.num_models} models: {total_params:,}")
    
    # Create optimizers
    optimizers = []
    for model in models:
        if config.optimizer == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        optimizers.append(optimizer)
    
    # Create schedulers
    schedulers = []
    for optimizer in optimizers:
        if config.scheduler == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=config.scheduler_params['milestones'],
                gamma=config.scheduler_params['gamma']
            )
        elif config.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs
            )
        else:
            raise ValueError(f"Unknown scheduler: {config.scheduler}")
        schedulers.append(scheduler)
    
    # Create DML config
    dml_config = DMLConfig(
        temperature=config.temperature,
        supervised_weight=1.0,
        mimicry_weight=1.0,
        peer_selection='all'
    )
    
    # Create DML trainer
    trainer = DMLTrainer(
        models=models,
        optimizers=optimizers,
        config=dml_config,
        device=config.device
    )
    
    # Setup metrics logger
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    logger = MetricsLogger(config.name, str(results_dir))
    
    # Setup checkpoints directory
    checkpoints_dir = Path(__file__).parent.parent / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_avg_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    print(f"\nStarting training for {config.epochs} epochs...")
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        
        # Train using trainer's built-in method
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate using trainer's built-in method
        val_metrics = trainer.evaluate(val_loader)
        
        # Update learning rate
        for scheduler in schedulers:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Extract average metrics
        avg_train_acc = train_metrics['train_acc']
        avg_val_acc = val_metrics['val_acc']
        train_loss = train_metrics['train_loss']
        val_loss = val_metrics['val_loss']
        
        # Extract per-model accuracies
        train_accs = [train_metrics[f'train_acc_model_{i}'] for i in range(config.num_models)]
        val_accs = [val_metrics[f'val_acc_model_{i}'] for i in range(config.num_models)]
        
        # Log metrics
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'avg_train_acc': avg_train_acc,
            'avg_val_acc': avg_val_acc,
            'epoch_time': epoch_time,
            'learning_rate': optimizers[0].param_groups[0]['lr']
        }
        
        # Add per-model metrics
        for i in range(config.num_models):
            metrics[f'model_{i}_train_acc'] = train_accs[i]
            metrics[f'model_{i}_val_acc'] = val_accs[i]
        
        logger.log_epoch(epoch, metrics)
        
        # Print progress
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"  Train Loss: {train_loss:.4f} - Train Acc: {avg_train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} - Val Acc: {avg_val_acc:.2f}%")
        print(f"  Per-model Val Acc: {', '.join([f'{acc:.2f}%' for acc in val_accs])}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Per-model Val Acc: {', '.join([f'{acc:.2f}%' for acc in val_accs])}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            best_epoch = epoch
            
            # Save all models
            checkpoint_path = checkpoints_dir / f"{config.name}_best.pth"
            torch.save({
                'epoch': epoch,
                'models_state_dict': [m.state_dict() for m in models],
                'optimizers_state_dict': [o.state_dict() for o in optimizers],
                'best_avg_acc': best_avg_acc,
                'config': config.to_dict(),
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
        
        print()
    
    total_time = time.time() - start_time
    
    # Final summary
    print("Training Complete")
    print(f"Best Average Accuracy: {best_avg_acc:.2f}% (epoch {best_epoch})")
    print(f"Total Training Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Save final metrics
    logger.save()
    
    return best_avg_acc, best_epoch, total_time


def main():
    parser = argparse.ArgumentParser(description='Run DML experiments')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment config JSON file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        best_acc, best_epoch, total_time = run_dml_experiment(args.config)
        print(f"\nExperiment completed successfully!")
        print(f"Best Accuracy: {best_acc:.2f}%")
        print(f"Best Epoch: {best_epoch}")
        print(f"Total Time: {total_time/60:.1f} minutes")
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
