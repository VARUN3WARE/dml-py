"""
Tests for reproducibility utilities.
"""

import pytest
import torch
import numpy as np
import random

from pydml.utils.reproducibility import (
    set_seed,
    get_random_state,
    set_random_state,
    ReproducibleContext
)


class TestSetSeed:
    """Test set_seed function."""
    
    def test_set_seed_basic(self):
        """Test that set_seed makes operations reproducible."""
        set_seed(42)
        
        # Generate random numbers
        py_rand1 = random.random()
        np_rand1 = np.random.rand()
        torch_rand1 = torch.rand(1).item()
        
        # Reset seed and regenerate
        set_seed(42)
        py_rand2 = random.random()
        np_rand2 = np.random.rand()
        torch_rand2 = torch.rand(1).item()
        
        # Should be identical
        assert py_rand1 == py_rand2
        assert np_rand1 == np_rand2
        assert torch_rand1 == torch_rand2
    
    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        result1 = torch.rand(10)
        
        set_seed(123)
        result2 = torch.rand(10)
        
        # Should be different
        assert not torch.allclose(result1, result2)
    
    def test_set_seed_deterministic_mode(self):
        """Test deterministic mode setting."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        set_seed(42, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        
        set_seed(42, deterministic=False)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
    
    def test_set_seed_cuda(self):
        """Test CUDA seed setting if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        set_seed(42)
        cuda_rand1 = torch.cuda.FloatTensor(1).normal_().item()
        
        set_seed(42)
        cuda_rand2 = torch.cuda.FloatTensor(1).normal_().item()
        
        assert cuda_rand1 == cuda_rand2


class TestRandomState:
    """Test random state save/restore."""
    
    def test_save_restore_state(self):
        """Test that random state can be saved and restored."""
        set_seed(42)
        
        # Save state
        state = get_random_state()
        
        # Generate some random numbers
        random.random()
        np.random.rand()
        torch.rand(1)
        
        # Restore state
        set_random_state(state)
        
        # These should match what would have been generated after set_seed(42)
        expected_py = random.random()
        expected_np = np.random.rand()
        expected_torch = torch.rand(1)
        
        # Reset and verify
        set_seed(42)
        actual_py = random.random()
        actual_np = np.random.rand()
        actual_torch = torch.rand(1)
        
        assert expected_py == actual_py
        assert expected_np == actual_np
        assert torch.allclose(expected_torch, actual_torch)


class TestReproducibleContext:
    """Test ReproducibleContext context manager."""
    
    def test_context_manager_basic(self):
        """Test that context manager works."""
        set_seed(42)
        before = torch.rand(5)
        
        with ReproducibleContext(seed=123):
            inside = torch.rand(5)
        
        after = torch.rand(5)
        
        # Verify results
        set_seed(42)
        torch.rand(5)  # Skip the 'before' tensor
        
        set_seed(123)
        expected_inside = torch.rand(5)
        
        assert torch.allclose(inside, expected_inside)
    
    def test_context_manager_restore(self):
        """Test that state is restored after context."""
        set_seed(42)
        
        # Save expected continuation
        state_before = get_random_state()
        expected = torch.rand(5)
        set_random_state(state_before)
        
        # Use context
        with ReproducibleContext(seed=999, restore=True):
            torch.rand(100)  # Do lots of random ops
        
        # Should continue from before context
        actual = torch.rand(5)
        assert torch.allclose(expected, actual)
    
    def test_context_manager_no_restore(self):
        """Test context without state restoration."""
        set_seed(42)
        
        with ReproducibleContext(seed=123, restore=False):
            inside = torch.rand(5)
        
        # State should not be restored - we continue with seed 123
        after = torch.rand(5)
        
        # Verify
        set_seed(123)
        expected_inside = torch.rand(5)
        expected_after = torch.rand(5)
        
        assert torch.allclose(inside, expected_inside)
        assert torch.allclose(after, expected_after)


class TestReproducibilityIntegration:
    """Integration tests for reproducibility."""
    
    def test_model_training_reproducible(self):
        """Test that model training is reproducible with seed."""
        from pydml.models.cifar import resnet32
        
        # Train 1
        set_seed(42)
        model1 = resnet32(num_classes=10)
        x1 = torch.randn(8, 3, 32, 32)
        y1 = model1(x1)
        
        # Train 2
        set_seed(42)
        model2 = resnet32(num_classes=10)
        x2 = torch.randn(8, 3, 32, 32)
        y2 = model2(x2)
        
        # Should be identical
        assert torch.allclose(x1, x2)
        assert torch.allclose(y1, y2)
        
        # Model weights should also be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_data_loading_reproducible(self):
        """Test that data augmentation is reproducible."""
        from torchvision import transforms
        from PIL import Image
        
        # Create a simple image
        img = Image.new('RGB', (32, 32), color='red')
        
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        # Transform 1
        set_seed(42)
        tensor1 = transform(img)
        
        # Transform 2
        set_seed(42)
        tensor2 = transform(img)
        
        assert torch.allclose(tensor1, tensor2)


def test_seed_in_trainer():
    """Test that trainers accept and use seed parameter."""
    from pydml.trainers import DMLTrainer
    from pydml.models.cifar import resnet32
    
    # Create trainer with seed
    models = [resnet32(num_classes=10) for _ in range(2)]
    trainer = DMLTrainer(models=models, device='cpu', seed=42)
    
    assert trainer.seed == 42
    
    # Verify reproducibility
    set_seed(42)
    x1 = torch.randn(4, 3, 32, 32)
    
    trainer2 = DMLTrainer(
        models=[resnet32(num_classes=10) for _ in range(2)],
        device='cpu',
        seed=42
    )
    x2 = torch.randn(4, 3, 32, 32)
    
    assert torch.allclose(x1, x2)
