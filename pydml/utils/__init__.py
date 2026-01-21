"""Utils module for DML-PY."""

from .data import get_cifar10_loaders, get_cifar100_loaders
from .metrics import accuracy
from .logging import ExperimentLogger, ConsoleLogger
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
