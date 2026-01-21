"""
VGG Models for CIFAR Demo

This example demonstrates how to use VGG architectures (VGG11, VGG13, VGG16, VGG19)
with the PyTorch-DML library for CIFAR datasets.

VGG models are deep convolutional networks that use very small (3x3) convolution filters.
This implementation is adapted for CIFAR's 32x32 images.
"""

import torch
import torch.nn as nn
from pydml.models.cifar import vgg11, vgg13, vgg16, vgg19
from pydml.trainers import DMLTrainer


def demonstrate_vgg_models():
    """Demonstrate VGG model creation and usage."""
    print("=" * 80)
    print("VGG MODELS FOR CIFAR DEMONSTRATION")
    print("=" * 80)
    print()
    
    # 1. Create different VGG variants
    print("1. Creating VGG Models")
    print("-" * 80)
    
    models_info = [
        ("VGG-11", vgg11(num_classes=10)),
        ("VGG-13", vgg13(num_classes=10)),
        ("VGG-16", vgg16(num_classes=10)),
        ("VGG-19", vgg19(num_classes=10)),
    ]
    
    for name, model in models_info:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name:8} - Parameters: {num_params:,}")
    
    print()
    
    # 2. Test with sample input
    print("2. Testing Forward Pass")
    print("-" * 80)
    
    # Create sample CIFAR batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    print(f"Input shape: {x.shape}")
    
    # Test VGG16
    model = vgg16(num_classes=10)
    model.eval()
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Predictions (logits):\n{output[0]}")
    print()
    
    # 3. Deep Mutual Learning with VGG models
    print("3. Deep Mutual Learning with Mixed VGG Models")
    print("-" * 80)
    
    # Create ensemble of different VGG variants
    dml_models = [
        vgg11(num_classes=10),
        vgg13(num_classes=10),
        vgg16(num_classes=10),
    ]
    
    print(f"Created ensemble with {len(dml_models)} VGG variants:")
    for i, model in enumerate(dml_models):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model {i+1}: {num_params:,} parameters")
    
    # Initialize DML trainer
    trainer = DMLTrainer(
        models=dml_models,
        device='cpu'
    )
    
    print(f"\n✓ DML Trainer initialized with {len(dml_models)} models")
    print()
    
    # 4. VGG for CIFAR-100
    print("4. VGG for CIFAR-100 (100 classes)")
    print("-" * 80)
    
    model_cifar100 = vgg16(num_classes=100)
    x_cifar100 = torch.randn(4, 3, 32, 32)
    
    model_cifar100.eval()
    with torch.no_grad():
        output_cifar100 = model_cifar100(x_cifar100)
    
    print(f"Input shape: {x_cifar100.shape}")
    print(f"Output shape: {output_cifar100.shape}")
    print(f"✓ Successfully created VGG16 for CIFAR-100")
    print()
    
    # 5. Batch Normalization comparison
    print("5. Batch Normalization Comparison")
    print("-" * 80)
    
    model_with_bn = vgg16(num_classes=10, batch_norm=True)
    model_without_bn = vgg16(num_classes=10, batch_norm=False)
    
    params_with_bn = sum(p.numel() for p in model_with_bn.parameters())
    params_without_bn = sum(p.numel() for p in model_without_bn.parameters())
    
    print(f"VGG16 with BatchNorm:    {params_with_bn:,} parameters")
    print(f"VGG16 without BatchNorm: {params_without_bn:,} parameters")
    print(f"Difference: {params_with_bn - params_without_bn:,} parameters")
    print()
    
    # 6. Model architecture summary
    print("6. VGG16 Architecture Summary")
    print("-" * 80)
    
    model = vgg16(num_classes=10)
    
    # Count layers
    conv_layers = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    fc_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    bn_layers = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    
    print(f"Convolutional layers: {conv_layers}")
    print(f"Fully connected layers: {fc_layers}")
    print(f"Batch normalization layers: {bn_layers}")
    print()
    
    # 7. Memory footprint
    print("7. Memory Footprint Comparison")
    print("-" * 80)
    
    for name, model in [("VGG-11", vgg11()), ("VGG-16", vgg16()), ("VGG-19", vgg19())]:
        # Estimate memory (parameters only)
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        param_size_mb = param_size / (1024 ** 2)
        print(f"{name:8} - Model size: {param_size_mb:.2f} MB")
    
    print()
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features:")
    print("  ✓ VGG11, VGG13, VGG16, VGG19 available")
    print("  ✓ Adapted for CIFAR-10/100 (32x32 images)")
    print("  ✓ Supports both CIFAR-10 and CIFAR-100")
    print("  ✓ Optional batch normalization")
    print("  ✓ Compatible with DML training")
    print()
    print("Next Steps:")
    print("  1. Import: from pydml.models.cifar import vgg16")
    print("  2. Create: model = vgg16(num_classes=10)")
    print("  3. Train with DML or standard training")


if __name__ == '__main__':
    demonstrate_vgg_models()
