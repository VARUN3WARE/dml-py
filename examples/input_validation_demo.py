"""
Demonstration of comprehensive input validation.

This example shows how PyDML validates inputs and provides clear,
helpful error messages to make debugging easier.
"""

import torch
import torch.nn as nn
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32
from pydml.utils import get_cifar10_loaders


def example_1_invalid_batch_size():
    """Example 1: Invalid batch size."""
    print("=" * 80)
    print("Example 1: Invalid Batch Size")
    print("=" * 80)
    
    print("\n Attempting to create data loader with negative batch size...")
    try:
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            batch_size=-32,  # Invalid: negative
            download=False
        )
        print("ERROR: This should have failed!")
    except (ValueError, TypeError) as e:
        print(f" Caught error as expected:")
        print(f"  {type(e).__name__}: {e}")
    
    print("\n Attempting to create data loader with non-integer batch size...")
    try:
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            batch_size=32.5,  # Invalid: float
            download=False
        )
        print("ERROR: This should have failed!")
    except (ValueError, TypeError) as e:
        print(f" Caught error as expected:")
        print(f"  {type(e).__name__}: {e}")


def example_2_invalid_val_split():
    """Example 2: Invalid validation split."""
    print("\n" + "=" * 80)
    print("Example 2: Invalid Validation Split")
    print("=" * 80)
    
    print("\n Attempting to create data loader with val_split > 1.0...")
    try:
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            val_split=1.5,  # Invalid: > 1.0
            download=False
        )
        print("ERROR: This should have failed!")
    except ValueError as e:
        print(f" Caught ValueError as expected:")
        print(f"  {e}")
    
    print("\n Attempting to create data loader with negative val_split...")
    try:
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            val_split=-0.1,  # Invalid: negative
            download=False
        )
        print("ERROR: This should have failed!")
    except ValueError as e:
        print(f" Caught ValueError as expected:")
        print(f"  {e}")


def example_3_invalid_num_workers():
    """Example 3: Invalid number of workers."""
    print("\n" + "=" * 80)
    print("Example 3: Invalid Number of Workers")
    print("=" * 80)
    
    print("\n Attempting to create data loader with negative num_workers...")
    try:
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            num_workers=-1,  # Invalid: negative
            download=False
        )
        print("ERROR: This should have failed!")
    except ValueError as e:
        print(f" Caught ValueError as expected:")
        print(f"  {e}")


def example_4_invalid_dml_config():
    """Example 4: Invalid DML configuration."""
    print("\n" + "=" * 80)
    print("Example 4: Invalid DML Configuration")
    print("=" * 80)
    
    print("\n Attempting to create DMLConfig with negative temperature...")
    try:
        config = DMLConfig(temperature=-1.0)  # Invalid: negative
        print("ERROR: This should have failed!")
    except ValueError as e:
        print(f" Caught ValueError as expected:")
        print(f"  {e}")
    
    print("\n Attempting to create DMLConfig with invalid peer_selection...")
    try:
        config = DMLConfig(peer_selection='invalid')  # Invalid: not in choices
        print("ERROR: This should have failed!")
    except ValueError as e:
        print(f" Caught ValueError as expected:")
        print(f"  {e}")
    
    print("\n Attempting to create DMLConfig with both weights zero...")
    try:
        config = DMLConfig(
            supervised_weight=0.0,
            mimicry_weight=0.0  # Invalid: both zero
        )
        print("ERROR: This should have failed!")
    except ValueError as e:
        print(f" Caught ValueError as expected:")
        print(f"  {e}")


def example_5_invalid_model_count():
    """Example 5: Invalid model count."""
    print("\n" + "=" * 80)
    print("Example 5: Invalid Model Count")
    print("=" * 80)
    
    print("\n Attempting to create DMLTrainer with only 1 model...")
    try:
        models = [resnet32(num_classes=10)]  # Invalid: need at least 2
        trainer = DMLTrainer(models, device='cpu')
        print("ERROR: This should have failed!")
    except ValueError as e:
        print(f" Caught ValueError as expected:")
        print(f"  {e}")
    
    print("\n Attempting to create DMLTrainer with non-Module object...")
    try:
        models = [resnet32(num_classes=10), "not a model"]  # Invalid: not a Module
        trainer = DMLTrainer(models, device='cpu')
        print("ERROR: This should have failed!")
    except TypeError as e:
        print(f" Caught TypeError as expected:")
        print(f"  {e}")


def example_6_helpful_error_messages():
    """Example 6: Helpful error messages."""
    print("\n" + "=" * 80)
    print("Example 6: Helpful Error Messages")
    print("=" * 80)
    
    print("\nAll validation errors include:")
    print("  • What went wrong")
    print("  • What was expected")
    print("  • What was actually provided")
    print("  • How to fix it")
    
    print("\nExamples:")
    
    # Example: batch size
    print("\n1. Invalid batch size:")
    try:
        get_cifar10_loaders(batch_size="invalid", download=False)
    except TypeError as e:
        print(f"   {e}")
    
    # Example: temperature
    print("\n2. Invalid temperature:")
    try:
        config = DMLConfig(temperature="3.0")
    except TypeError as e:
        print(f"   {e}")
    
    # Example: peer selection
    print("\n3. Invalid peer selection:")
    try:
        config = DMLConfig(peer_selection="invalid_option")
    except ValueError as e:
        print(f"   {e}")


def example_7_valid_configurations():
    """Example 7: Valid configurations."""
    print("\n" + "=" * 80)
    print("Example 7: Valid Configurations Work Seamlessly")
    print("=" * 80)
    
    print("\n Creating data loaders with valid parameters...")
    try:
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            batch_size=128,
            num_workers=0,
            val_split=0.1,
            download=True
        )
        print("  Success! Created train, val, and test loaders")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n Creating DMLConfig with valid parameters...")
    try:
        config = DMLConfig(
            temperature=3.0,
            supervised_weight=1.0,
            mimicry_weight=1.0,
            peer_selection='all'
        )
        print(f"  Success! Config created: {config}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n Creating DMLTrainer with valid models...")
    try:
        models = [resnet32(num_classes=10) for _ in range(2)]
        trainer = DMLTrainer(models, config=config, device='cpu')
        print(f"  Success! Trainer created with {len(trainer.models)} models")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "PyDML Input Validation Demo" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Run all examples
    example_1_invalid_batch_size()
    example_2_invalid_val_split()
    example_3_invalid_num_workers()
    example_4_invalid_dml_config()
    example_5_invalid_model_count()
    example_6_helpful_error_messages()
    example_7_valid_configurations()
    
    print("\n" + "=" * 80)
    print("All Examples Completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("   All inputs are validated before use")
    print("   Error messages are clear and actionable")
    print("   Type errors and value errors are caught early")
    print("   Debugging is easier with helpful messages")
    print("   Valid configurations work without any issues")
    print("\n")
