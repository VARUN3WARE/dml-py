"""
DML-PY - A Collaborative Deep Learning Library

Main package for collaborative neural network training.
"""

__version__ = "0.1.0"
__author__ = "DML-PY Contributors"
__license__ = "MIT"

from pydml.trainers.dml import DMLTrainer, DMLConfig
from pydml.core.base_trainer import BaseCollaborativeTrainer
from pydml.utils.reproducibility import set_seed
from pydml.utils.cuda_memory import (
    handle_oom,
    clear_cuda_cache,
    get_gpu_memory_info,
    AutoBatchSizeReducer,
    MemoryMonitor
)

__all__ = [
    "DMLTrainer",
    "DMLConfig",
    "BaseCollaborativeTrainer",
    "set_seed",
    "handle_oom",
    "clear_cuda_cache",
    "get_gpu_memory_info",
    "AutoBatchSizeReducer",
    "MemoryMonitor",
]
