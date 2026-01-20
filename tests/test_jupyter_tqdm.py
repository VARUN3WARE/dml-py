"""
Test tqdm Jupyter environment detection.
"""

import sys
from unittest.mock import MagicMock


def test_jupyter_detection_in_terminal():
    """Test that tqdm detection works in terminal (non-Jupyter) environment."""
    # In terminal, get_ipython should not be defined
    # Import should use regular tqdm
    from pydml.core.base_trainer import tqdm
    
    # Check that tqdm is imported (will be regular tqdm in terminal)
    assert tqdm is not None
    assert callable(tqdm)


def test_jupyter_detection_mock():
    """Test Jupyter detection with mocked environment."""
    # Save original builtins
    import builtins
    original_get_ipython = getattr(builtins, 'get_ipython', None)
    
    try:
        # Mock Jupyter environment
        mock_ipython = MagicMock()
        mock_ipython.__class__.__name__ = 'ZMQInteractiveShell'
        builtins.get_ipython = lambda: mock_ipython
        
        # Re-import to trigger detection
        import importlib
        from pydml.core import base_trainer
        importlib.reload(base_trainer)
        
        # Verify detection function works
        assert base_trainer._is_jupyter() == True
        
    finally:
        # Restore original state
        if original_get_ipython is None:
            if hasattr(builtins, 'get_ipython'):
                delattr(builtins, 'get_ipython')
        else:
            builtins.get_ipython = original_get_ipython
        
        # Reload to restore normal state
        import importlib
        from pydml.core import base_trainer
        importlib.reload(base_trainer)


def test_tqdm_import_works():
    """Test that tqdm import doesn't raise errors."""
    try:
        from pydml.core.base_trainer import tqdm, _is_jupyter
        
        # Should not raise any errors
        is_jupyter = _is_jupyter()
        assert isinstance(is_jupyter, bool)
        
    except ImportError as e:
        pytest.fail(f"Failed to import tqdm: {e}")


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
