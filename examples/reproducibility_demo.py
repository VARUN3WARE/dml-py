"""
Reproducibility Example for PyDML

This example demonstrates how to ensure reproducible results across different runs.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pydml import DMLTrainer, set_seed
from pydml.models.cifar import resnet32
from pydml.utils import get_cifar10_loaders


def example_basic_reproducibility():
    """Basic example of using set_seed."""
    print("=" * 70)
    print("Example 1: Basic Reproducibility")
    print("=" * 70)
    
    # Set seed for reproducible results
    set_seed(42)
    
    # Create models
    models = [resnet32(num_classes=10) for _ in range(3)]
    
    # Create trainer
    trainer = DMLTrainer(models=models, device='cpu', seed=42)
    
    # Generate random data
    x = torch.randn(8, 3, 32, 32)
    
    print(f"Random tensor sum: {x.sum().item():.6f}")
    print("Run this script multiple times - the sum will always be the same!")
    print()


def example_reproducible_training():
    """Example of reproducible training."""
    print("=" * 70)
    print("Example 2: Reproducible Training")
    print("=" * 70)
    
    # Set seed before creating anything
    set_seed(42)
    
    # Prepare data (small subset for demo)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use only first 100 samples for quick demo
    train_dataset = torch.utils.data.Subset(train_dataset, range(100))
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    
    # Create models
    models = [resnet32(num_classes=10) for _ in range(2)]
    
    # Create trainer with seed
    trainer = DMLTrainer(models=models, device='cpu', seed=42)
    
    # Train for 1 epoch
    print("Training for 1 epoch...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=train_loader,  # Using train as val for demo
        epochs=1,
        verbose=True
    )
    
    print(f"\nFinal training loss: {history['train_loss'][-1]:.6f}")
    print(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
    print("\nRun this script multiple times - results will be identical!")
    print()


def example_compare_with_without_seed():
    """Compare results with and without seed."""
    print("=" * 70)
    print("Example 3: With vs Without Seed")
    print("=" * 70)
    
    print("Run 1 (no seed):")
    result1_no_seed = torch.rand(5)
    print(f"  Random values: {result1_no_seed}")
    
    print("\nRun 2 (no seed):")
    result2_no_seed = torch.rand(5)
    print(f"  Random values: {result2_no_seed}")
    print(f"  Same as run 1? {torch.allclose(result1_no_seed, result2_no_seed)}")
    
    print("\nRun 3 (seed=42):")
    set_seed(42)
    result1_with_seed = torch.rand(5)
    print(f"  Random values: {result1_with_seed}")
    
    print("\nRun 4 (seed=42):")
    set_seed(42)
    result2_with_seed = torch.rand(5)
    print(f"  Random values: {result2_with_seed}")
    print(f"  Same as run 3? {torch.allclose(result1_with_seed, result2_with_seed)}")
    print()


def example_reproducible_context():
    """Example using ReproducibleContext."""
    print("=" * 70)
    print("Example 4: ReproducibleContext")
    print("=" * 70)
    
    from pydml.utils import ReproducibleContext
    
    # Set global seed
    set_seed(42)
    print("Global seed set to 42")
    
    # Generate some random data
    data1 = torch.rand(3)
    print(f"Data before context: {data1}")
    
    # Use different seed in context
    with ReproducibleContext(seed=999):
        data_in_context = torch.rand(3)
        print(f"Data inside context (seed=999): {data_in_context}")
    
    # After context, global seed is restored
    data2 = torch.rand(3)
    print(f"Data after context: {data2}")
    
    # Verify restoration
    set_seed(42)
    torch.rand(3)  # Skip first
    expected = torch.rand(3)
    print(f"Expected (seed=42 continuation): {expected}")
    print(f"Matches? {torch.allclose(data2, expected)}")
    print()


def example_best_practices():
    """Best practices for reproducibility."""
    print("=" * 70)
    print("Example 5: Best Practices")
    print("=" * 70)
    
    print("""
Best Practices for Reproducible Results:

1. Set seed at the start of your script:
   ```python
   from pydml import set_seed
   set_seed(42)
   ```

2. Use seed parameter in trainers:
   ```python
   trainer = DMLTrainer(models=models, seed=42)
   ```

3. Set num_workers=0 for DataLoader on Windows:
   ```python
   loader = DataLoader(dataset, num_workers=0)  # Reproducible
   ```

4. Disable CUDA benchmark for full reproducibility:
   ```python
   set_seed(42, deterministic=True)  # Slower but reproducible
   ```

5. Document your seed in papers/reports:
   - "All experiments use random seed 42 for reproducibility"

6. For multiple experiments with different seeds:
   ```python
   results = []
   for seed in [42, 123, 456]:
       set_seed(seed)
       result = train_model()
       results.append(result)
   
   mean_acc = np.mean([r['accuracy'] for r in results])
   std_acc = np.std([r['accuracy'] for r in results])
   print(f"Accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
   ```

7. Save seed with checkpoints:
   ```python
   checkpoint = {
       'seed': 42,
       'model': model.state_dict(),
       # ... other data
   }
   ```
    """)


if __name__ == '__main__':
    # Run all examples
    example_basic_reproducibility()
    example_compare_with_without_seed()
    example_reproducible_context()
    example_best_practices()
    
    # Uncomment to run training example (takes a few minutes)
    # example_reproducible_training()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
