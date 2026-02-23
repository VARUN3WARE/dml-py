# Future Work

This document outlines open research directions, engineering tasks, and contribution opportunities for PyDML. It is intended for new contributors, researchers, and anyone interested in taking this project forward.

## Current State

PyDML is a collaborative deep learning library implementing Deep Mutual Learning and related methods. The codebase includes:

- **Trainers:** DML, Knowledge Distillation, Feature-based DML, Co-Distillation, Confidence-Weighted DML
- **Strategies:** Curriculum learning, Peer selection, Temperature scaling
- **Analysis:** Visualization, Robustness testing, Loss landscape, Training monitor
- **Models:** ResNet32, MobileNetV2, WRN-28-10 (CIFAR variants)
- **Infrastructure:** Reproducible experiment configs, metrics logging, baseline trainer

### Benchmark Results (CIFAR-10)

| Method                 | Accuracy | Parameters | Training Time |
| ---------------------- | -------- | ---------- | ------------- |
| ResNet32 (baseline)    | 93.50%   | 467K       | 59 min        |
| MobileNetV2 (baseline) | 92.50%   | 2.2M       | 224 min       |
| WRN-28-10 (baseline)   | 96.05%   | 36.5M      | 1141 min      |
| DML 2x ResNet32        | 93.86%   | 933K       | 84 min        |

DML shows a +0.36% accuracy improvement over single ResNet32 with modest training overhead.

---

## Research Directions

### 1. Scaling DML to More Models

**Priority:** High
**Difficulty:** Medium

The 2-model DML experiment showed promising results. Key questions remain:

- Does 3-model DML improve over 2-model? (initial GPU memory issues with mixed architectures)
- What is the optimal number of peer models?
- How does performance scale with increasing model count?

**Suggested experiments:**

- 3x ResNet32 on CIFAR-10 (fits in single GPU memory)
- 4x ResNet32 on CIFAR-10
- Asymmetric configurations (e.g., 2x ResNet32 + 1x MobileNet with reduced batch size)

### 2. Cross-Architecture DML

**Priority:** High
**Difficulty:** Medium-High

DML between heterogeneous architectures (e.g., ResNet + MobileNet + WRN) is underexplored. This requires:

- Memory-efficient training strategies for mixed-size model groups
- Gradient accumulation for large models in multi-model setups
- Analysis of knowledge transfer dynamics between architectures of different capacity

### 3. CIFAR-100 and ImageNet Validation

**Priority:** High
**Difficulty:** Medium

Current benchmarks are limited to CIFAR-10. Extending to harder datasets is critical for research credibility:

- CIFAR-100 baselines and DML experiments
- ImageNet subset experiments (e.g., ImageNet-100)
- Full ImageNet validation with standard ResNet-50 / EfficientNet models

### 4. Statistical Rigor

**Priority:** High
**Difficulty:** Low-Medium

Current results are from single runs. For publication-quality results:

- Run each configuration 5+ times with different seeds
- Report mean and standard deviation
- Conduct paired statistical tests (e.g., paired t-test or Wilcoxon signed-rank)
- Confidence intervals for accuracy improvements

### 5. Knowledge Distillation Comparison

**Priority:** Medium-High
**Difficulty:** Medium

A direct comparison between DML and classical knowledge distillation would strengthen the case for DML:

- Use WRN-28-10 (96.05%) as teacher, ResNet32 as student
- Compare: single model vs KD-trained model vs DML-trained model
- Analyze when DML outperforms KD and vice versa

### 6. Ensemble vs DML Analysis

**Priority:** Medium-High
**Difficulty:** Low

Compare DML against independently trained ensembles:

- Train 2 ResNet32 models independently
- Combine predictions by averaging logits
- Compare: single model vs independent ensemble vs DML ensemble
- This directly demonstrates whether collaborative training adds value beyond ensembling

### 7. Ablation Studies

**Priority:** Medium
**Difficulty:** Low-Medium

Systematic analysis of DML hyperparameters:

- Temperature sensitivity (T = 1, 2, 3, 5, 10, 20)
- Loss weight ratios (supervised_weight vs mimicry_weight)
- Effect of peer selection strategy (all, best, random, dynamic)
- Learning rate schedule interaction with DML
- Batch size sensitivity

### 8. Advanced DML Variants

**Priority:** Medium
**Difficulty:** High

Novel research directions building on the existing codebase:

- **Adaptive temperature:** Learn the temperature parameter during training
- **Asymmetric DML:** Different learning rates or loss weights per model
- **Progressive DML:** Start with independent training, gradually introduce mutual learning
- **Selective mimicry:** Only learn from peers on samples where the peer is more confident
- **Curriculum DML:** Order training samples by difficulty, combine with mutual learning

### 9. Transfer Learning with DML

**Priority:** Medium
**Difficulty:** Medium

- Pre-train models with DML on source domain
- Fine-tune on target domain
- Compare transfer learning performance: DML pre-trained vs independently pre-trained
- Test across domain shifts (e.g., CIFAR to STL-10, or ImageNet to specialized datasets)

### 10. Federated DML

**Priority:** Low-Medium
**Difficulty:** High

Combine DML with federated learning for privacy-preserving collaborative training:

- Each client trains a local model
- Share soft predictions instead of gradients (privacy benefit)
- Analyze communication efficiency vs standard federated averaging

---

## Engineering Tasks

### Infrastructure

- [ ] **Multi-GPU support for DML:** Enable training large model groups across multiple GPUs
- [ ] **Gradient accumulation in DML trainer:** Allow larger effective batch sizes on limited memory
- [ ] **Automated experiment sweeps:** Script to run hyperparameter sweeps and aggregate results
- [ ] **TensorBoard / W&B integration:** Real-time training visualization and experiment tracking
- [ ] **CI/CD pipeline:** Automated testing on GPU instances for training correctness

### Code Quality

- [ ] **Type annotations:** Complete type hints across all modules
- [ ] **Docstring coverage:** Ensure every public method has a complete docstring
- [ ] **Integration tests:** End-to-end training tests with small models and few epochs
- [ ] **Performance profiling:** Identify and optimize training bottlenecks
- [ ] **Memory profiling:** Track GPU memory usage across different configurations

### Models and Datasets

- [ ] **Add more model architectures:** EfficientNet, Vision Transformer (ViT), DenseNet
- [ ] **Add more datasets:** STL-10, Tiny ImageNet, SVHN
- [ ] **Pre-trained model zoo:** Provide downloadable checkpoints for common configurations
- [ ] **Data loading optimization:** Prefetching, caching, and faster augmentation pipelines

### Documentation

- [ ] **Tutorial notebooks:** Jupyter notebooks for common workflows
- [ ] **Benchmark reproduction guide:** Step-by-step instructions to reproduce all results
- [ ] **Architecture decision records:** Document why certain design choices were made
- [ ] **API changelog:** Track breaking changes across versions

---

## Contribution Opportunities

### Good First Issues

These tasks are well-scoped and suitable for new contributors:

1. **Add accuracy per class logging:** Track per-class accuracy during evaluation
2. **Add early stopping:** Implement patience-based early stopping in the training loop
3. **Add learning rate warmup:** Gradual warmup schedule option for DML training
4. **Add model parameter count to logs:** Print parameter count at training start
5. **Add training resumption:** Load checkpoint and resume training from last epoch

### Medium Complexity

6. **Implement DML with gradient accumulation:** Enable larger effective batch sizes
7. **Add cosine similarity loss:** Alternative to KL divergence for peer learning
8. **Add mixup / cutmix augmentation:** Data augmentation strategies compatible with DML
9. **W&B experiment tracking integration:** Log metrics to Weights & Biases
10. **Add CIFAR-100 benchmark configs:** Create and validate configurations for CIFAR-100

### Advanced

11. **Multi-GPU DML training:** Distribute models across GPUs
12. **Adaptive temperature scheduling:** Learn or schedule temperature during training
13. **Neural architecture search for peer selection:** Automate model pairing
14. **DML with self-supervised learning:** Combine contrastive learning with mutual learning
15. **Benchmark against recent papers:** Reproduce and compare with state-of-the-art methods

---

## How to Get Started

1. Read the [CONTRIBUTING.md](CONTRIBUTING.md) guide
2. Set up your development environment following [GETTING_STARTED.md](GETTING_STARTED.md)
3. Run the existing tests: `pytest tests/ -v`
4. Pick a task from the "Good First Issues" section above
5. Open an issue to discuss your approach before starting
6. Submit a pull request with tests and documentation

For questions, open a GitHub issue or discussion.

---

## Maintainer Notes

This repository is actively maintained. The maintainer reviews PRs weekly and will respond to issues within a few days. Major changes should be discussed in an issue before implementation.

**Last Updated:** February 23, 2026
