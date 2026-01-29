"""Utils module for DML-PY."""

from .data import get_cifar10_loaders, get_cifar100_loaders
from .metrics import accuracy
from .logging import ExperimentLogger, ConsoleLogger
from .reproducibility import set_seed, get_random_state, set_random_state, ReproducibleContext
from .cuda_memory import (
    get_gpu_memory_info,
    clear_cuda_cache,
    print_memory_summary,
    handle_oom,
    safe_forward,
    AutoBatchSizeReducer,
    MemoryMonitor,
    CUDAOutOfMemoryError
)
from .checkpointing import CheckpointManager, auto_resume
from .amp import AMPConfig, AMPManager, apply_amp_to_trainer
from .distributed import DistributedConfig, DistributedManager, launch_distributed, apply_distributed_to_trainer
from .export import ExportConfig, ModelExporter, export_ensemble, quick_export
from .hyperparameter_search import (
    HyperparameterSpace,
    HyperparameterSearcher,
    GridSearcher,
    RandomSearcher,
    OptunaSearcher,
    create_dml_search_space,
    quick_search
)
from .schedulers import (
    create_step_schedulers,
    create_multistep_schedulers,
    create_cosine_schedulers,
    create_cosine_warmrestart_schedulers,
    create_exponential_schedulers,
    create_reduce_on_plateau_schedulers,
    get_scheduler_info,
    validate_schedulers,
)
from .lr_scheduling import (
    SchedulerType,
    WarmupConfig,
    SchedulerConfig,
    create_warmup_scheduler,
    create_polynomial_scheduler,
    create_scheduler_from_config,
    create_schedulers_from_config,
    get_cifar_schedule,
    get_imagenet_schedule,
    get_fine_tuning_schedule,
)
from .ensemble import (
    ensemble_predict,
    average_predictions,
    voting_predictions,
    weighted_predictions,
    max_confidence_predictions,
    ensemble_accuracy,
    calibrate_ensemble_weights,
    get_prediction_diversity,
    EnsembleModel,
)

__all__ = [
    'get_cifar10_loaders',
    'get_cifar100_loaders',
    'accuracy',
    'ExperimentLogger',
    'ConsoleLogger',
    'set_seed',
    'get_random_state',
    'set_random_state',
    'ReproducibleContext',
    'get_gpu_memory_info',
    'clear_cuda_cache',
    'print_memory_summary',
    'handle_oom',
    'safe_forward',
    'AutoBatchSizeReducer',
    'MemoryMonitor',
    'CUDAOutOfMemoryError',
    'CheckpointManager',
    'auto_resume',
    'AMPConfig',
    'AMPManager',
    'apply_amp_to_trainer',
    'DistributedConfig',
    'DistributedManager',
    'launch_distributed',
    'apply_distributed_to_trainer',
    'ExportConfig',
    'ModelExporter',
    'export_ensemble',
    'quick_export',
    'HyperparameterSpace',
    'HyperparameterSearcher',
    'GridSearcher',
    'RandomSearcher',
    'OptunaSearcher',
    'create_dml_search_space',
    'quick_search',
    'create_step_schedulers',
    'create_multistep_schedulers',
    'create_cosine_schedulers',
    'create_cosine_warmrestart_schedulers',
    'create_exponential_schedulers',
    'create_reduce_on_plateau_schedulers',
    'get_scheduler_info',
    'validate_schedulers',
    'SchedulerType',
    'WarmupConfig',
    'SchedulerConfig',
    'create_warmup_scheduler',
    'create_polynomial_scheduler',
    'create_scheduler_from_config',
    'create_schedulers_from_config',
    'get_cifar_schedule',
    'get_imagenet_schedule',
    'get_fine_tuning_schedule',
    'ensemble_predict',
    'average_predictions',
    'voting_predictions',
    'weighted_predictions',
    'max_confidence_predictions',
    'ensemble_accuracy',
    'calibrate_ensemble_weights',
    'get_prediction_diversity',
    'EnsembleModel',
]
