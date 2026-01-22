"""
Architecture Performance Comparison

This script benchmarks different neural network architectures (ResNet, MobileNet, WRN)
using Deep Mutual Learning on CIFAR-10. It compares:
- Training time
- Final accuracy
- Model size
- Computational efficiency

Run this to understand which architecture combinations work best with DML.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
from pydml.trainers import DMLTrainer
from pydml.models.cifar import resnet32, mobilenet_v2, wrn_28_10, vgg16


def prepare_data(batch_size=128, subset_size=None):
    """Prepare CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Use subset if specified (for faster testing)
    if subset_size:
        train_dataset = torch.utils.data.Subset(train_dataset, range(subset_size))
        test_dataset = torch.utils.data.Subset(test_dataset, range(subset_size // 5))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    return train_loader, test_loader


def get_model_info(model):
    """Get model parameter count and size."""
    params = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    size_mb = param_size / (1024 ** 2)
    return params, size_mb


def benchmark_architecture(models, model_names, train_loader, test_loader, epochs=20, device='cuda'):
    """Benchmark a specific architecture ensemble."""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {' + '.join(model_names)}")
    print(f"{'='*80}")
    
    # Model info
    total_params = 0
    total_size = 0
    for i, (model, name) in enumerate(zip(models, model_names)):
        params, size_mb = get_model_info(model)
        total_params += params
        total_size += size_mb
        print(f"  {name:15} - {params:,} params ({size_mb:.2f} MB)")
    
    print(f"\n  Total: {total_params:,} params ({total_size:.2f} MB)")
    
    # Create trainer
    trainer = DMLTrainer(models=models, device=device)
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        verbose=False
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    results = trainer.evaluate(test_loader)
    
    # Compile results
    benchmark_results = {
        'ensemble_name': ' + '.join(model_names),
        'model_names': model_names,
        'num_models': len(models),
        'total_params': total_params,
        'total_size_mb': total_size,
        'training_time': training_time,
        'time_per_epoch': training_time / epochs,
        'ensemble_accuracy': results['val_acc'],
        'ensemble_loss': results['val_loss'],
        'individual_accuracies': [results[f'val_acc_model_{i}'] for i in range(len(models))],
        'history': history,
    }
    
    print(f"\n{'Results':^80}")
    print(f"{'-'*80}")
    print(f"  Training time: {training_time:.1f}s ({training_time/epochs:.1f}s/epoch)")
    print(f"  Ensemble accuracy: {results['val_acc']:.2f}%")
    print(f"  Ensemble loss: {results['val_loss']:.4f}")
    print(f"\n  Individual model accuracies:")
    for i, (name, acc) in enumerate(zip(model_names, benchmark_results['individual_accuracies'])):
        print(f"    {name:15}: {acc:.2f}%")
    
    # Clean up
    del trainer
    del models
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return benchmark_results


def run_benchmarks(train_loader, test_loader, epochs=20, device='cuda'):
    """Run benchmarks on different architecture combinations."""
    
    print(f"\n{'='*80}")
    print(f"ARCHITECTURE BENCHMARK - Deep Mutual Learning")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    all_results = []
    
    # Benchmark 1: Three ResNet32 (same architecture, different seeds)
    models = [resnet32(num_classes=10) for _ in range(3)]
    results = benchmark_architecture(
        models, ['ResNet32', 'ResNet32', 'ResNet32'],
        train_loader, test_loader, epochs, device
    )
    all_results.append(results)
    
    # Benchmark 2: Three MobileNetV2 (same architecture, different seeds)
    models = [mobilenet_v2(num_classes=10) for _ in range(3)]
    results = benchmark_architecture(
        models, ['MobileNetV2', 'MobileNetV2', 'MobileNetV2'],
        train_loader, test_loader, epochs, device
    )
    all_results.append(results)
    
    # Benchmark 3: Three WRN-28-10 (same architecture, different seeds)
    models = [wrn_28_10(num_classes=10) for _ in range(3)]
    results = benchmark_architecture(
        models, ['WRN-28-10', 'WRN-28-10', 'WRN-28-10'],
        train_loader, test_loader, epochs, device
    )
    all_results.append(results)
    
    # Benchmark 4: Mixed - ResNet + MobileNet + WRN (diverse)
    models = [
        resnet32(num_classes=10),
        mobilenet_v2(num_classes=10),
        wrn_28_10(num_classes=10),
    ]
    results = benchmark_architecture(
        models, ['ResNet32', 'MobileNetV2', 'WRN-28-10'],
        train_loader, test_loader, epochs, device
    )
    all_results.append(results)
    
    # Benchmark 5: ResNet + MobileNet + VGG (classic mix)
    models = [
        resnet32(num_classes=10),
        mobilenet_v2(num_classes=10),
        vgg16(num_classes=10),
    ]
    results = benchmark_architecture(
        models, ['ResNet32', 'MobileNetV2', 'VGG16'],
        train_loader, test_loader, epochs, device
    )
    all_results.append(results)
    
    return all_results


def print_comparison_table(results):
    """Print comparison table of all benchmarks."""
    print(f"\n{'='*120}")
    print(f"{'PERFORMANCE COMPARISON TABLE':^120}")
    print(f"{'='*120}")
    
    # Header
    print(f"{'Architecture':<40} | {'Params':>12} | {'Size':>10} | {'Time':>10} | {'Ens Acc':>10} | {'Avg Ind':>10} | {'Best Ind':>10}")
    print(f"{'-'*120}")
    
    # Rows
    for r in results:
        avg_individual = sum(r['individual_accuracies']) / len(r['individual_accuracies'])
        best_individual = max(r['individual_accuracies'])
        
        print(f"{r['ensemble_name']:<40} | {r['total_params']:>12,} | {r['total_size_mb']:>9.1f}M | "
              f"{r['training_time']:>9.1f}s | {r['ensemble_accuracy']:>9.2f}% | "
              f"{avg_individual:>9.2f}% | {best_individual:>9.2f}%")
    
    print(f"{'='*120}")
    
    # Analysis
    print(f"\n{'Key Insights':^120}")
    print(f"{'-'*120}")
    
    # Best ensemble accuracy
    best_ensemble = max(results, key=lambda x: x['ensemble_accuracy'])
    print(f"🏆 Best Ensemble Accuracy: {best_ensemble['ensemble_name']} ({best_ensemble['ensemble_accuracy']:.2f}%)")
    
    # Fastest training
    fastest = min(results, key=lambda x: x['training_time'])
    print(f"⚡ Fastest Training: {fastest['ensemble_name']} ({fastest['training_time']:.1f}s)")
    
    # Most efficient (accuracy per MB)
    for r in results:
        r['efficiency'] = r['ensemble_accuracy'] / r['total_size_mb']
    most_efficient = max(results, key=lambda x: x['efficiency'])
    print(f"💡 Most Efficient: {most_efficient['ensemble_name']} ({most_efficient['efficiency']:.3f} acc/MB)")
    
    # Most improvement from DML (ensemble vs best individual)
    for r in results:
        best_ind = max(r['individual_accuracies'])
        r['dml_boost'] = r['ensemble_accuracy'] - best_ind
    best_boost = max(results, key=lambda x: x['dml_boost'])
    print(f"📈 Best DML Boost: {best_boost['ensemble_name']} (+{best_boost['dml_boost']:.2f}%)")


def plot_comparisons(results):
    """Plot comparison charts."""
    fig = plt.figure(figsize=(16, 10))
    
    # Extract data
    names = [r['ensemble_name'] for r in results]
    ensemble_accs = [r['ensemble_accuracy'] for r in results]
    avg_ind_accs = [sum(r['individual_accuracies']) / len(r['individual_accuracies']) for r in results]
    training_times = [r['training_time'] for r in results]
    sizes_mb = [r['total_size_mb'] for r in results]
    
    # 1. Accuracy Comparison
    ax1 = plt.subplot(2, 3, 1)
    x = range(len(names))
    width = 0.35
    ax1.bar([i - width/2 for i in x], ensemble_accs, width, label='Ensemble', color='#2ecc71')
    ax1.bar([i + width/2 for i in x], avg_ind_accs, width, label='Avg Individual', color='#3498db')
    ax1.set_xlabel('Architecture', fontsize=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=10)
    ax1.set_title('Ensemble vs Individual Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace(' + ', '\n') for n in names], fontsize=8, rotation=0)
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    
    # 2. Training Time
    ax2 = plt.subplot(2, 3, 2)
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
    ax2.barh(names, training_times, color=colors[:len(names)])
    ax2.set_xlabel('Training Time (seconds)', fontsize=10)
    ax2.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')
    
    # 3. Model Size
    ax3 = plt.subplot(2, 3, 3)
    ax3.barh(names, sizes_mb, color=colors[:len(names)])
    ax3.set_xlabel('Total Size (MB)', fontsize=10)
    ax3.set_title('Model Size Comparison', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='x')
    
    # 4. Accuracy vs Size
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(sizes_mb, ensemble_accs, s=[t*2 for t in training_times], 
                         c=range(len(names)), cmap='viridis', alpha=0.6, edgecolors='black')
    for i, name in enumerate(names):
        ax4.annotate(name.split(' + ')[0], (sizes_mb[i], ensemble_accs[i]), 
                    fontsize=8, ha='center')
    ax4.set_xlabel('Total Size (MB)', fontsize=10)
    ax4.set_ylabel('Ensemble Accuracy (%)', fontsize=10)
    ax4.set_title('Accuracy vs Size (bubble size = training time)', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # 5. DML Boost (Ensemble - Best Individual)
    ax5 = plt.subplot(2, 3, 5)
    dml_boosts = [r['ensemble_accuracy'] - max(r['individual_accuracies']) for r in results]
    bars = ax5.bar(range(len(names)), dml_boosts, color=['#2ecc71' if b > 0 else '#e74c3c' for b in dml_boosts])
    ax5.set_xlabel('Architecture', fontsize=10)
    ax5.set_ylabel('Accuracy Gain (%)', fontsize=10)
    ax5.set_title('DML Boost (Ensemble - Best Individual)', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(len(names)))
    ax5.set_xticklabels([n.replace(' + ', '\n') for n in names], fontsize=8, rotation=0)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.grid(alpha=0.3, axis='y')
    
    # 6. Training Curves (last benchmark)
    ax6 = plt.subplot(2, 3, 6)
    for i, r in enumerate(results[-2:]):  # Show last 2 benchmarks
        ax6.plot(r['history']['val_acc'], label=r['ensemble_name'], linewidth=2, marker='o', markersize=3)
    ax6.set_xlabel('Epoch', fontsize=10)
    ax6.set_ylabel('Validation Accuracy (%)', fontsize=10)
    ax6.set_title('Training Curves (Selected)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('architecture_benchmark.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plots saved to: architecture_benchmark.png")
    plt.show()


def main():
    """Run the architecture benchmark."""
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 20  # Adjust for longer/shorter runs
    subset_size = 10000  # Use subset for faster testing (None for full dataset)
    
    print(f"\n{'='*80}")
    print(f"{'DEEP MUTUAL LEARNING - ARCHITECTURE BENCHMARK':^80}")
    print(f"{'='*80}")
    
    # Prepare data
    print("\nPreparing CIFAR-10 dataset...")
    train_loader, test_loader = prepare_data(batch_size=128, subset_size=subset_size)
    
    # Run benchmarks
    results = run_benchmarks(train_loader, test_loader, epochs=epochs, device=device)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Plot comparisons
    plot_comparisons(results)
    
    print(f"\n{'='*80}")
    print(f"{'BENCHMARK COMPLETE':^80}")
    print(f"{'='*80}")
    print("\nRecommendations:")
    print("  • For best accuracy: Use diverse architectures (ResNet + MobileNet + WRN)")
    print("  • For efficiency: Use MobileNet ensembles (faster, smaller)")
    print("  • For production: Balance accuracy and size based on your needs")
    print("  • DML works best with diverse architectures vs identical ones")


if __name__ == '__main__':
    main()
