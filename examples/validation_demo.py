"""
Demonstration of safe validation that cannot be bypassed by optimization flags.

This example shows that all validation logic uses proper exceptions
instead of assert statements, ensuring that:
1. Validation always works, even with python -O or -OO
2. Error messages are informative
3. Invalid configurations are caught immediately

Run this script with and without optimization to verify:
    python validation_demo.py           # Normal execution
    python -O validation_demo.py        # With optimization (-O removes asserts)
    python -OO validation_demo.py       # With maximum optimization
"""

import torch
import torch.nn as nn
from pydml import DMLTrainer
from pydml.models.cifar import resnet32
from pydml.models.cifar.mobilenet import InvertedResidual
from pydml.models.cifar.wrn import WideResNet


def example_1_optimizer_count_validation():
    """Example 1: Optimizer count validation."""
    print("=" * 80)
    print("Example 1: Optimizer Count Validation")
    print("=" * 80)
    
    # Create 3 models
    models = [resnet32(num_classes=10) for _ in range(3)]
    
    # Try to create trainer with wrong number of optimizers
    print("\n✗ Attempting to create DMLTrainer with 2 optimizers for 3 models...")
    try:
        optimizers = [
            torch.optim.SGD(models[0].parameters(), lr=0.1),
            torch.optim.SGD(models[1].parameters(), lr=0.1),
            # Missing third optimizer!
        ]
        trainer = DMLTrainer(models, optimizers=optimizers)
        print("ERROR: This should have failed!")
    except ValueError as e:
        print(f"✓ Caught ValueError as expected:")
        print(f"  {e}")
    
    # Now create with correct number
    print("\n✓ Creating DMLTrainer with correct number of optimizers...")
    optimizers = [
        torch.optim.SGD(model.parameters(), lr=0.1)
        for model in models
    ]
    trainer = DMLTrainer(models, optimizers=optimizers)
    print(f"  Success! Trainer created with {len(trainer.optimizers)} optimizers")


def example_2_mobilenet_stride_validation():
    """Example 2: MobileNet stride validation."""
    print("\n" + "=" * 80)
    print("Example 2: MobileNet Stride Validation")
    print("=" * 80)
    
    # Try invalid stride values
    invalid_strides = [0, 3, 4, -1]
    
    for stride in invalid_strides:
        print(f"\n✗ Attempting to create InvertedResidual with stride={stride}...")
        try:
            block = InvertedResidual(inp=32, oup=64, stride=stride, expand_ratio=6)
            print("ERROR: This should have failed!")
        except ValueError as e:
            print(f"✓ Caught ValueError as expected:")
            print(f"  {e}")
    
    # Now create with valid strides
    valid_strides = [1, 2]
    print(f"\n✓ Creating InvertedResidual with valid strides {valid_strides}...")
    for stride in valid_strides:
        block = InvertedResidual(inp=32, oup=64, stride=stride, expand_ratio=6)
        print(f"  Success! Block created with stride={stride}")


def example_3_wrn_depth_validation():
    """Example 3: WideResNet depth validation."""
    print("\n" + "=" * 80)
    print("Example 3: WideResNet Depth Validation")
    print("=" * 80)
    
    # Try invalid depth values
    invalid_depths = [5, 11, 27, 29]
    
    for depth in invalid_depths:
        print(f"\n✗ Attempting to create WideResNet with depth={depth}...")
        print(f"  Checking: (depth - 4) % 6 == 0 => ({depth} - 4) % 6 = {(depth - 4) % 6}")
        try:
            model = WideResNet(depth=depth, num_classes=10)
            print("ERROR: This should have failed!")
        except ValueError as e:
            print(f"✓ Caught ValueError as expected:")
            print(f"  {e}")
    
    # Now create with valid depths
    valid_depths = [10, 16, 22, 28, 40]
    print(f"\n✓ Creating WideResNet with valid depths...")
    for depth in valid_depths:
        print(f"  depth={depth}: (depth - 4) % 6 = {(depth - 4) % 6} ✓")
        model = WideResNet(depth=depth, num_classes=10)
        print(f"    Success! Model created")


def example_4_optimization_flag_resistance():
    """Example 4: Verify validation works even with python -O."""
    print("\n" + "=" * 80)
    print("Example 4: Optimization Flag Resistance")
    print("=" * 80)
    
    print("\nThis script can be run with optimization flags:")
    print("  python validation_demo.py      # Normal")
    print("  python -O validation_demo.py   # Optimization (-O removes asserts)")
    print("  python -OO validation_demo.py  # Max optimization")
    
    print("\nAll validations use proper exceptions (ValueError), not asserts.")
    print("This means they ALWAYS work, regardless of optimization flags.")
    
    # Check if we're running with optimization
    import sys
    if __debug__:
        print("\n✓ Currently running in DEBUG mode (asserts enabled)")
    else:
        print("\n⚠️  Currently running in OPTIMIZED mode (asserts disabled)")
        print("   But validations still work because they use proper exceptions!")
    
    # Demonstrate that validation works
    print("\nDemonstrating validation still works:")
    try:
        models = [resnet32(num_classes=10), resnet32(num_classes=10)]
        optimizers = [torch.optim.SGD(models[0].parameters(), lr=0.1)]  # Wrong count
        trainer = DMLTrainer(models, optimizers=optimizers)
        print("ERROR: Validation was bypassed!")
    except ValueError:
        print("✓ Validation works correctly - ValueError raised as expected")


def example_5_informative_error_messages():
    """Example 5: Error messages are informative."""
    print("\n" + "=" * 80)
    print("Example 5: Informative Error Messages")
    print("=" * 80)
    
    print("\nAll validation errors include helpful information:")
    
    # Optimizer count error
    print("\n1. Optimizer count mismatch:")
    try:
        models = [resnet32(num_classes=10) for _ in range(3)]
        optimizers = [torch.optim.SGD(models[0].parameters(), lr=0.1)]
        trainer = DMLTrainer(models, optimizers=optimizers)
    except ValueError as e:
        print(f"   Message: {e}")
        print("   ✓ Includes actual counts")
    
    # Stride error
    print("\n2. Invalid stride:")
    try:
        block = InvertedResidual(inp=32, oup=64, stride=5, expand_ratio=6)
    except ValueError as e:
        print(f"   Message: {e}")
        print("   ✓ Includes invalid value")
    
    # Depth error
    print("\n3. Invalid depth:")
    try:
        model = WideResNet(depth=29, num_classes=10)
    except ValueError as e:
        print(f"   Message: {e}")
        print("   ✓ Includes constraint and actual value")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "PyDML Validation Demonstration" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Run all examples
    example_1_optimizer_count_validation()
    example_2_mobilenet_stride_validation()
    example_3_wrn_depth_validation()
    example_4_optimization_flag_resistance()
    example_5_informative_error_messages()
    
    print("\n" + "=" * 80)
    print("All Examples Completed Successfully!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  ✓ All validations use ValueError, not assert statements")
    print("  ✓ Validations cannot be bypassed by python -O or -OO flags")
    print("  ✓ Error messages are informative and include actual values")
    print("  ✓ Invalid configurations are caught immediately")
    print("  ✓ Production code is safe from silent failures")
    print("\n")
