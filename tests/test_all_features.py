"""
Comprehensive test script for all DML-PY features.
Tests all modules implemented in Phases 3-5 and novel research.
"""

import sys
import traceback

def test_phase3_imports():
    """Test Phase 3: AMP, DDP, Export"""
    from pydml.utils.amp import AMPConfig, AMPManager
    from pydml.utils.distributed import DistributedConfig, DistributedManager
    from pydml.utils.export import ExportConfig, ModelExporter

def test_phase4_imports():
    """Test Phase 4: Loss Landscape, Hyperparameter Search"""
    from pydml.analysis.loss_landscape import LossLandscape
    from pydml.utils.hyperparameter_search import (
        HyperparameterSearcher, HyperparameterSpace
    )

def test_confidence_weighted_imports():
    """Test Novel Research: Confidence-Weighted DML"""
    from pydml.trainers.confidence_weighted import (
        ConfidenceWeightedDML, ConfidenceWeightedConfig
    )

def test_existing_trainers():
    """Test existing trainer imports"""
    from pydml.trainers import DMLTrainer, DistillationTrainer
    from pydml.trainers.feature_dml import FeatureDMLTrainer
    from pydml.trainers.co_distillation import CoDistillationTrainer

def test_strategies():
    """Test strategy modules"""
    from pydml.strategies.curriculum import CurriculumStrategy
    from pydml.strategies.peer_selection import PeerSelector, PeerSelectionConfig
    from pydml.strategies.temperature_scaling import TemperatureScheduler

def test_losses():
    """Test loss modules"""
    from pydml.core.losses import KLDivergenceLoss, DMLLoss
    from pydml.losses.attention_transfer import AttentionTransferLoss

def test_analysis():
    """Test analysis modules"""
    from pydml.analysis.visualization import plot_training_history
    from pydml.analysis.robustness import add_noise_to_model, test_robustness_to_noise

def test_models():
    """Test model imports"""
    from pydml.models.cifar import resnet32, resnet110, wrn_28_10, mobilenet_v2

def test_amp_functionality():
    """Test AMP basic functionality"""
    import torch
    from pydml.utils.amp import AMPConfig, AMPManager
    
    config = AMPConfig(enabled=False)  # Disabled for CPU testing
    manager = AMPManager(config)
    
    # Test that manager is created
    assert manager is not None

def test_export_functionality():
    """Test model export basic functionality"""
    import torch
    import torch.nn as nn
    from pydml.utils.export import ExportConfig, ModelExporter
    
    # Create simple model and config
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    config = ExportConfig()
    exporter = ModelExporter(model, config)
    
    # Test that exporter is created
    assert exporter is not None

def test_loss_landscape_functionality():
    """Test loss landscape basic functionality"""
    import torch
    import torch.nn as nn
    from pydml.analysis.loss_landscape import LossLandscape
    
    # Create simple model and data
    model = nn.Linear(5, 2)
    data_loader = [(torch.randn(4, 5), torch.randint(0, 2, (4,)))]
    criterion = nn.CrossEntropyLoss()
    
    landscape = LossLandscape(model, criterion, data_loader, device='cpu')

def test_hyperparameter_search_functionality():
    """Test hyperparameter search basic functionality"""
    from pydml.utils.hyperparameter_search import (
        HyperparameterSearcher, HyperparameterSpace, create_dml_search_space
    )
    
    # Test search space creation using helper function
    space = create_dml_search_space()
    assert space is not None

def test_confidence_weighted_functionality():
    """Test Confidence-Weighted DML basic functionality"""
    import torch
    import torch.nn as nn
    from pydml.trainers.confidence_weighted import (
        ConfidenceWeightedDML, ConfidenceWeightedConfig
    )
    
    # Create simple models
    models = [nn.Linear(10, 5) for _ in range(2)]
    config = ConfidenceWeightedConfig(confidence_threshold=0.5)
    
    trainer = ConfidenceWeightedDML(
        models=models,
        config=config,
        device='cpu'
    )
    
    # Test confidence computation
    logits = torch.randn(4, 5)
    confidence = trainer.compute_confidence(logits)
    assert confidence.shape == (4,)


def run_all_tests():
    """Run all tests manually and print results (legacy function - use pytest instead)"""
    print("=" * 80)
    print("DML-PY COMPREHENSIVE FEATURE TEST")
    print("=" * 80)
    print("Note: This is a legacy test runner. Use 'pytest tests/' for proper testing.")
    print()
    
    tests = [
        ("Phase 3 Imports", test_phase3_imports),
        ("Phase 4 Imports", test_phase4_imports),
        ("Confidence-Weighted Imports", test_confidence_weighted_imports),
        ("Existing Trainers", test_existing_trainers),
        ("Strategies", test_strategies),
        ("Losses", test_losses),
        ("Analysis", test_analysis),
        ("Models", test_models),
        ("AMP Functionality", test_amp_functionality),
        ("Export Functionality", test_export_functionality),
        ("Loss Landscape Functionality", test_loss_landscape_functionality),
        ("HPO Functionality", test_hyperparameter_search_functionality),
        ("Confidence-Weighted Functionality", test_confidence_weighted_functionality),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, f"✓ {name} passed"))
            print(f"✓ {name} passed")
        except Exception as e:
            results.append((name, False, f"✗ {name} failed: {str(e)}"))
            print(f"✗ {name} failed: {str(e)}")
            traceback.print_exc()
    
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! DML-PY is fully operational.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("\nFailed tests:")
        for name, success, message in results:
            if not success:
                print(f"  - {name}: {message}")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
