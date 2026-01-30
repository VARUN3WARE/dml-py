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
from pydml.utils.amp import AMPConfig, AMPManager
from pydml.utils.checkpointing import CheckpointManager, auto_resume
from pydml.utils.lr_scheduling import (
    SchedulerType,
    SchedulerConfig,
    WarmupConfig,
    create_schedulers_from_config,
    get_cifar_schedule,
    get_imagenet_schedule,
    get_fine_tuning_schedule,
)
from pydml.analysis import (
    TrainingMonitor,
    OverfittingStatus,
    TrainingMetrics,
    OverfittingReport,
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
    "AMPConfig",
    "AMPManager",
    "CheckpointManager",
    "auto_resume",
    "SchedulerType",
    "SchedulerConfig",
    "WarmupConfig",
    "create_schedulers_from_config",
    "get_cifar_schedule",
    "get_imagenet_schedule",
    "get_fine_tuning_schedule",
    "TrainingMonitor",
    "OverfittingStatus",
    "TrainingMetrics",
    "OverfittingReport",
]
