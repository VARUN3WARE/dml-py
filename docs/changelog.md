# Changelog

All notable changes to PyDML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-31

### Added

- **Comprehensive Documentation**: Full Sphinx documentation with API reference, tutorials, and user guides
- **Input Validation**: 18 validation functions for type and value checking
  - Clear error messages with actual vs. expected values
  - Validation in data loaders, configurations, and core utilities
  - 58 comprehensive test cases
- **Training Monitor**: Automatic overfitting detection with actionable recommendations
  - 5 severity levels: NO_OVERFITTING, MILD, MODERATE, SEVERE, UNDERFITTING
  - Trend analysis for metrics
  - Early stopping support
  - 22 new tests
- **Production-Safe Validation**: Replaced assert statements with proper exceptions
  - Works even with `python -O` optimization flag
  - 21 new tests for edge cases

### Changed

- Improved error messages throughout codebase
- Enhanced type safety with comprehensive validation

### Fixed

- Assert statements replaced with ValueError to prevent silent failures in optimized mode
- Syntax errors in data loading utilities

## [0.1.0] - 2025-12-28

### Added

- Core DML implementation
- Knowledge Distillation Trainer
- Co-Distillation Trainer
- Feature-Based DML Trainer
- CIFAR models (ResNet, MobileNet, WideResNet)
- Attention Transfer mechanisms
- Curriculum Learning strategies
- Visualization tools (6 plot types)
- Robustness analysis
- PyPI package publication
- 40+ unit tests
- 17 working examples
- Checkpoint management
- LR scheduling with warmup
- Mixed precision training support

### Initial Release

- BaseCollaborativeTrainer framework
- Loss functions (CE, KL, DML, Attention Transfer)
- Callbacks (EarlyStopping, ModelCheckpoint, TensorBoard)
- CIFAR-10/100 data loaders
- Metrics (accuracy, ECE, entropy, diversity)
- Experiment logging

## [Unreleased]

### Planned Features

- Multi-GPU distributed training (DDP)
- Additional model architectures (VGG, DenseNet, EfficientNet)
- Jupyter notebook tutorials
- Advanced adversarial robustness testing
- Integration with Weights & Biases
- TorchScript export support
- ONNX model export

---

[0.2.0]: https://github.com/VARUN3WARE/dml-py/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/VARUN3WARE/dml-py/releases/tag/v0.1.0
