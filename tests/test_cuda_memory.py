"""Tests for CUDA memory management and OOM handling."""

import pytest
import torch
import torch.nn as nn
from pydml.utils.cuda_memory import (
    get_gpu_memory_info,
    clear_cuda_cache,
    handle_oom,
    safe_forward,
    AutoBatchSizeReducer,
    MemoryMonitor,
    CUDAOutOfMemoryError
)


class TestMemoryInfo:
    """Test memory information utilities."""
    
    def test_get_gpu_memory_info_no_cuda(self):
        """Test memory info when CUDA is not available."""
        if not torch.cuda.is_available():
            info = get_gpu_memory_info()
            assert info['allocated'] == 0.0
            assert info['reserved'] == 0.0
            assert info['free'] == 0.0
            assert info['total'] == 0.0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_gpu_memory_info_cuda(self):
        """Test memory info with CUDA."""
        info = get_gpu_memory_info()
        assert info['allocated'] >= 0.0
        assert info['reserved'] >= 0.0
        assert info['free'] >= 0.0
        assert info['total'] > 0.0
        assert info['allocated'] <= info['total']
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_gpu_memory_info_with_allocation(self):
        """Test memory info changes with tensor allocation."""
        initial_info = get_gpu_memory_info()
        
        # Allocate some memory
        tensor = torch.randn(1000, 1000, device='cuda')
        
        after_info = get_gpu_memory_info()
        assert after_info['allocated'] > initial_info['allocated']
        
        # Clean up
        del tensor
        clear_cuda_cache()


class TestClearCache:
    """Test cache clearing."""
    
    def test_clear_cuda_cache_no_cuda(self):
        """Test cache clearing without CUDA (should not error)."""
        clear_cuda_cache()  # Should not raise
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_clear_cuda_cache_with_cuda(self):
        """Test cache clearing with CUDA."""
        # Allocate and deallocate
        tensor = torch.randn(1000, 1000, device='cuda')
        del tensor
        
        # Clear cache
        clear_cuda_cache()  # Should not raise


class TestHandleOOM:
    """Test OOM error handling decorator."""
    
    def test_handle_oom_normal_function(self):
        """Test decorator with normal function execution."""
        @handle_oom
        def normal_func(x):
            return x * 2
        
        result = normal_func(5)
        assert result == 10
    
    def test_handle_oom_non_oom_error(self):
        """Test decorator with non-OOM error."""
        @handle_oom
        def error_func():
            raise ValueError("Not OOM")
        
        with pytest.raises(ValueError, match="Not OOM"):
            error_func()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_handle_oom_with_oom_error(self):
        """Test decorator with simulated OOM error."""
        @handle_oom
        def oom_func():
            raise RuntimeError("CUDA out of memory. Tried to allocate...")
        
        with pytest.raises(CUDAOutOfMemoryError):
            oom_func()


class TestSafeForward:
    """Test safe forward pass."""
    
    def test_safe_forward_normal(self):
        """Test safe forward with normal execution."""
        model = nn.Linear(10, 5)
        x = torch.randn(2, 10)
        
        output = safe_forward(model, x)
        assert output.shape == (2, 5)
    
    def test_safe_forward_with_error(self):
        """Test safe forward with non-OOM error."""
        model = nn.Linear(10, 5)
        x = torch.randn(2, 5)  # Wrong size
        
        with pytest.raises(RuntimeError):
            safe_forward(model, x)


class TestAutoBatchSizeReducer:
    """Test automatic batch size reduction."""
    
    def test_initialization(self):
        """Test reducer initialization."""
        reducer = AutoBatchSizeReducer(initial_batch_size=64, min_batch_size=4)
        assert reducer.get_batch_size() == 64
        assert reducer.can_reduce()
        assert reducer.attempts == 0
    
    def test_reduce(self):
        """Test batch size reduction."""
        reducer = AutoBatchSizeReducer(initial_batch_size=64, min_batch_size=4, reduction_factor=0.5)
        
        assert reducer.get_batch_size() == 64
        
        reducer.reduce()
        assert reducer.get_batch_size() == 32
        assert reducer.attempts == 1
        
        reducer.reduce()
        assert reducer.get_batch_size() == 16
        assert reducer.attempts == 2
        
        reducer.reduce()
        assert reducer.get_batch_size() == 8
        
        reducer.reduce()
        assert reducer.get_batch_size() == 4
    
    def test_min_batch_size_limit(self):
        """Test that reduction stops at minimum."""
        reducer = AutoBatchSizeReducer(initial_batch_size=8, min_batch_size=4, reduction_factor=0.5)
        
        reducer.reduce()
        assert reducer.get_batch_size() == 4
        
        # Try to reduce below minimum
        reducer.reduce()
        assert reducer.get_batch_size() == 4  # Should stay at minimum
    
    def test_can_reduce(self):
        """Test can_reduce logic."""
        reducer = AutoBatchSizeReducer(initial_batch_size=4, min_batch_size=4)
        assert reducer.can_reduce()  # At minimum but still valid
        
        reducer = AutoBatchSizeReducer(initial_batch_size=2, min_batch_size=4)
        assert reducer.can_reduce() is False  # Below minimum
    
    def test_reset(self):
        """Test reset functionality."""
        reducer = AutoBatchSizeReducer(initial_batch_size=64, min_batch_size=4)
        
        reducer.reduce()
        reducer.reduce()
        assert reducer.get_batch_size() == 16
        assert reducer.attempts == 2
        
        reducer.reset()
        assert reducer.get_batch_size() == 64
        assert reducer.attempts == 0


class TestMemoryMonitor:
    """Test memory monitoring."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = MemoryMonitor(warning_threshold=0.9)
        assert monitor.warning_threshold == 0.9
        assert monitor.peak_memory == 0.0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_check_memory(self):
        """Test memory checking."""
        monitor = MemoryMonitor(warning_threshold=0.9)
        
        info = monitor.check()
        assert 'allocated' in info
        assert 'total' in info
        assert info['allocated'] >= 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_peak_memory_tracking(self):
        """Test peak memory tracking."""
        monitor = MemoryMonitor()
        
        # Initial check
        monitor.check()
        initial_peak = monitor.get_peak_memory()
        
        # Allocate memory
        tensor = torch.randn(1000, 1000, device='cuda')
        monitor.check()
        after_peak = monitor.get_peak_memory()
        
        assert after_peak >= initial_peak
        
        # Clean up
        del tensor
        clear_cuda_cache()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reset_peak(self):
        """Test peak memory reset."""
        monitor = MemoryMonitor()
        
        tensor = torch.randn(1000, 1000, device='cuda')
        monitor.check()
        assert monitor.get_peak_memory() > 0
        
        monitor.reset_peak()
        assert monitor.peak_memory == 0.0
        
        del tensor
        clear_cuda_cache()


class TestCUDAOutOfMemoryError:
    """Test custom OOM exception."""
    
    def test_error_creation(self):
        """Test error with suggestions."""
        suggestions = ["Use smaller batch", "Enable gradient checkpointing"]
        error = CUDAOutOfMemoryError("OOM occurred", suggestions=suggestions)
        
        error_str = str(error)
        assert "OOM occurred" in error_str
        assert "Use smaller batch" in error_str
        assert "Enable gradient checkpointing" in error_str
    
    def test_error_without_suggestions(self):
        """Test error without suggestions."""
        error = CUDAOutOfMemoryError("OOM occurred")
        error_str = str(error)
        assert "OOM occurred" in error_str


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_oom_workflow(self):
        """Test complete OOM handling workflow."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ).cuda()
        
        # Create batch size reducer
        reducer = AutoBatchSizeReducer(initial_batch_size=1024, min_batch_size=32)
        
        # Create memory monitor
        monitor = MemoryMonitor(warning_threshold=0.8)
        
        # Simulate training with automatic batch size reduction
        batch_size = reducer.get_batch_size()
        
        try:
            # Create input
            x = torch.randn(batch_size, 100, device='cuda')
            
            # Monitor memory
            monitor.check()
            
            # Safe forward
            output = safe_forward(model, x)
            
            assert output.shape == (batch_size, 10)
            
        finally:
            # Cleanup
            clear_cuda_cache()
    
    def test_handle_oom_decorator_with_model(self):
        """Test OOM decorator with model forward pass."""
        model = nn.Linear(10, 5)
        
        @handle_oom
        def forward_pass(x):
            return model(x)
        
        x = torch.randn(2, 10)
        output = forward_pass(x)
        assert output.shape == (2, 5)
