"""
Tests for training monitoring and overfitting detection.
"""

import pytest
from pydml.analysis.training_monitor import (
    OverfittingStatus,
    TrainingMetrics,
    OverfittingReport,
    TrainingMonitor,
)


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""
    
    def test_creation(self):
        """Test creating TrainingMetrics."""
        metrics = TrainingMetrics(
            epoch=10,
            train_loss=0.5,
            val_loss=0.7,
            train_acc=85.0,
            val_acc=80.0,
        )
        assert metrics.epoch == 10
        assert metrics.train_loss == 0.5
        assert metrics.val_loss == 0.7
        assert metrics.train_acc == 85.0
        assert metrics.val_acc == 80.0
    
    def test_generalization_gap(self):
        """Test generalization gap calculation."""
        metrics = TrainingMetrics(
            epoch=10,
            train_loss=0.5,
            val_loss=0.7,
            train_acc=85.0,
            val_acc=80.0,
        )
        assert metrics.generalization_gap == 5.0
    
    def test_loss_gap(self):
        """Test loss gap calculation."""
        metrics = TrainingMetrics(
            epoch=10,
            train_loss=0.5,
            val_loss=0.7,
            train_acc=85.0,
            val_acc=80.0,
        )
        assert abs(metrics.loss_gap - 0.2) < 1e-10
    
    def test_is_overfitting(self):
        """Test overfitting detection."""
        # Not overfitting
        metrics1 = TrainingMetrics(
            epoch=10,
            train_loss=0.5,
            val_loss=0.6,
            train_acc=82.0,
            val_acc=80.0,
        )
        assert not metrics1.is_overfitting(threshold=5.0)
        
        # Overfitting
        metrics2 = TrainingMetrics(
            epoch=10,
            train_loss=0.3,
            val_loss=0.8,
            train_acc=95.0,
            val_acc=75.0,
        )
        assert metrics2.is_overfitting(threshold=5.0)


class TestTrainingMonitor:
    """Test TrainingMonitor class."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = TrainingMonitor()
        assert monitor.window_size == 5
        assert monitor.overfitting_threshold == 5.0
        assert len(monitor.history['epoch']) == 0
    
    def test_custom_initialization(self):
        """Test monitor with custom parameters."""
        monitor = TrainingMonitor(
            window_size=10,
            overfitting_threshold=10.0,
            track_per_model=True
        )
        assert monitor.window_size == 10
        assert monitor.overfitting_threshold == 10.0
        assert monitor.track_per_model is True
    
    def test_update(self):
        """Test updating monitor with metrics."""
        monitor = TrainingMonitor()
        
        train_metrics = {
            'train_loss': 0.5,
            'train_acc': 85.0,
        }
        val_metrics = {
            'val_loss': 0.6,
            'val_acc': 82.0,
        }
        
        monitor.update(1, train_metrics, val_metrics)
        
        assert len(monitor.history['epoch']) == 1
        assert monitor.history['train_loss'][0] == 0.5
        assert monitor.history['train_acc'][0] == 85.0
        assert monitor.history['val_loss'][0] == 0.6
        assert monitor.history['val_acc'][0] == 82.0
    
    def test_multiple_updates(self):
        """Test multiple metric updates."""
        monitor = TrainingMonitor()
        
        for epoch in range(1, 6):
            train_metrics = {
                'train_loss': 1.0 - epoch * 0.1,
                'train_acc': 50.0 + epoch * 5,
            }
            val_metrics = {
                'val_loss': 1.1 - epoch * 0.08,
                'val_acc': 48.0 + epoch * 4,
            }
            monitor.update(epoch, train_metrics, val_metrics)
        
        assert len(monitor.history['epoch']) == 5
        assert monitor.history['epoch'][-1] == 5
    
    def test_get_current_metrics(self):
        """Test getting current metrics."""
        monitor = TrainingMonitor()
        
        # No data yet
        assert monitor.get_current_metrics() is None
        
        # Add data
        monitor.update(1, {'train_loss': 0.5, 'train_acc': 80.0},
                      {'val_loss': 0.6, 'val_acc': 75.0})
        
        current = monitor.get_current_metrics()
        assert current is not None
        assert current.epoch == 1
        assert current.train_acc == 80.0
        assert current.val_acc == 75.0
    
    def test_get_generalization_gap(self):
        """Test generalization gap calculation."""
        monitor = TrainingMonitor()
        
        # No data
        assert monitor.get_generalization_gap() == 0.0
        
        # Add data
        monitor.update(1, {'train_loss': 0.5, 'train_acc': 85.0},
                      {'val_loss': 0.6, 'val_acc': 80.0})
        
        assert monitor.get_generalization_gap() == 5.0
    
    def test_is_overfitting_simple(self):
        """Test simple overfitting detection."""
        monitor = TrainingMonitor(overfitting_threshold=5.0)
        
        # Not overfitting
        monitor.update(1, {'train_loss': 0.5, 'train_acc': 82.0},
                      {'val_loss': 0.6, 'val_acc': 80.0})
        assert not monitor.is_overfitting()
        
        # Overfitting
        monitor.update(2, {'train_loss': 0.3, 'train_acc': 95.0},
                      {'val_loss': 0.8, 'val_acc': 75.0})
        assert monitor.is_overfitting()
    
    def test_is_overfitting_strict(self):
        """Test strict overfitting detection (sustained)."""
        monitor = TrainingMonitor(window_size=3, overfitting_threshold=5.0)
        
        # Build up history with sustained overfitting
        for i in range(5):
            monitor.update(
                i + 1,
                {'train_loss': 0.3, 'train_acc': 90.0},
                {'val_loss': 0.7, 'val_acc': 75.0}
            )
        
        # Strict check requires sustained overfitting
        assert monitor.is_overfitting(strict=True)
    
    def test_is_underfitting(self):
        """Test underfitting detection."""
        monitor = TrainingMonitor(window_size=3)
        
        # Build history with low, non-improving performance
        for i in range(5):
            monitor.update(
                i + 1,
                {'train_loss': 2.0, 'train_acc': 40.0},
                {'val_loss': 2.1, 'val_acc': 38.0}
            )
        
        assert monitor.is_underfitting()
    
    def test_get_overfitting_severity(self):
        """Test overfitting severity classification."""
        monitor = TrainingMonitor()
        
        # No overfitting - add multiple epochs for sufficient data
        for i in range(5):
            monitor.update(i + 1, {'train_loss': 0.5, 'train_acc': 82.0},
                          {'val_loss': 0.6, 'val_acc': 80.0})
        assert monitor.get_overfitting_severity() == OverfittingStatus.NO_OVERFITTING
        
        # Mild overfitting (gap ~4%)
        monitor = TrainingMonitor()
        for i in range(5):
            monitor.update(i + 1, {'train_loss': 0.5, 'train_acc': 84.0},
                          {'val_loss': 0.6, 'val_acc': 80.0})
        assert monitor.get_overfitting_severity() == OverfittingStatus.MILD_OVERFITTING
        
        # Moderate overfitting (gap ~8%)
        monitor = TrainingMonitor()
        for i in range(5):
            monitor.update(i + 1, {'train_loss': 0.3, 'train_acc': 88.0},
                          {'val_loss': 0.7, 'val_acc': 80.0})
        assert monitor.get_overfitting_severity() == OverfittingStatus.MODERATE_OVERFITTING
        
        # Severe overfitting (gap ~20%)
        monitor = TrainingMonitor()
        for i in range(5):
            monitor.update(i + 1, {'train_loss': 0.1, 'train_acc': 100.0},
                          {'val_loss': 1.0, 'val_acc': 80.0})
        assert monitor.get_overfitting_severity() == OverfittingStatus.SEVERE_OVERFITTING
    
    def test_get_overfitting_report(self):
        """Test overfitting report generation."""
        monitor = TrainingMonitor()
        
        # Add data showing consistent severe overfitting pattern
        for i in range(5):
            monitor.update(i + 1, {'train_loss': 0.2, 'train_acc': 95.0},
                          {'val_loss': 0.8, 'val_acc': 75.0})
        
        report = monitor.get_overfitting_report()
        
        assert isinstance(report, OverfittingReport)
        assert report.status == OverfittingStatus.SEVERE_OVERFITTING
        assert report.generalization_gap == 20.0
        assert len(report.recommendations) > 0
        assert report.confidence > 0
        
        # Check string formatting works
        report_str = str(report)
        assert "Overfitting Analysis Report" in report_str
        assert "Recommendations:" in report_str
    
    def test_get_trend(self):
        """Test trend analysis."""
        monitor = TrainingMonitor(window_size=5)
        
        # Improving trend
        for i in range(10):
            monitor.update(
                i + 1,
                {'train_loss': 1.0 - i * 0.08, 'train_acc': 50.0 + i * 4},
                {'val_loss': 1.1 - i * 0.07, 'val_acc': 48.0 + i * 3.5}
            )
        
        assert monitor.get_trend('val_acc') == 'improving'
        assert monitor.get_trend('val_loss') == 'improving'
        
        # Degrading trend
        monitor = TrainingMonitor(window_size=5)
        for i in range(10):
            monitor.update(
                i + 1,
                {'train_loss': 0.5 + i * 0.05, 'train_acc': 80.0 - i * 2},
                {'val_loss': 0.6 + i * 0.06, 'val_acc': 75.0 - i * 2.5}
            )
        
        assert monitor.get_trend('val_acc') == 'degrading'
    
    def test_should_stop_early(self):
        """Test early stopping logic."""
        monitor = TrainingMonitor()
        
        # Not enough data
        monitor.update(1, {'train_loss': 0.5, 'train_acc': 80.0},
                      {'val_loss': 0.6, 'val_acc': 75.0})
        assert not monitor.should_stop_early(patience=5)
        
        # Improving - should not stop
        for i in range(5):
            monitor.update(
                i + 2,
                {'train_loss': 0.5 - i * 0.02, 'train_acc': 80.0 + i * 1.0},
                {'val_loss': 0.6 - i * 0.02, 'val_acc': 75.0 + i * 1.0}
            )
        
        assert not monitor.should_stop_early(patience=3, min_delta=0.5)
        
        # Plateau - should stop
        for i in range(10):
            monitor.update(
                i + 7,
                {'train_loss': 0.4, 'train_acc': 85.0},
                {'val_loss': 0.5, 'val_acc': 80.0}
            )
        
        assert monitor.should_stop_early(patience=5, min_delta=0.1)
        
        # Plateau - should stop
        for i in range(10):
            monitor.update(
                i + 17,
                {'train_loss': 0.3, 'train_acc': 85.0},
                {'val_loss': 0.4, 'val_acc': 82.0}
            )
        
        assert monitor.should_stop_early(patience=5, min_delta=0.1)
    
    def test_get_best_epoch(self):
        """Test finding best epoch."""
        monitor = TrainingMonitor()
        
        # Build history
        accuracies = [70.0, 75.0, 80.0, 85.0, 83.0, 82.0]
        for i, acc in enumerate(accuracies):
            monitor.update(
                i + 1,
                {'train_loss': 0.5, 'train_acc': acc + 5},
                {'val_loss': 0.6, 'val_acc': acc}
            )
        
        best_epoch, best_acc = monitor.get_best_epoch('val_acc')
        assert best_epoch == 4  # Epoch 4 has highest val_acc (85.0)
        assert best_acc == 85.0
    
    def test_get_summary(self):
        """Test summary generation."""
        monitor = TrainingMonitor()
        
        # No data
        summary = monitor.get_summary()
        assert "No training data available" in summary
        
        # With data
        for i in range(10):
            monitor.update(
                i + 1,
                {'train_loss': 1.0 - i * 0.08, 'train_acc': 50.0 + i * 4},
                {'val_loss': 1.1 - i * 0.07, 'val_acc': 48.0 + i * 3.5}
            )
        
        summary = monitor.get_summary()
        assert "Training Progress Summary" in summary
        assert "Current Epoch:" in summary
        assert "Best Performance:" in summary
        assert "Trends:" in summary
    
    def test_per_model_tracking(self):
        """Test per-model metric tracking."""
        monitor = TrainingMonitor(track_per_model=True)
        
        train_metrics = {
            'train_loss': 0.5,
            'train_acc': 85.0,
            'train_acc_model_0': 83.0,
            'train_acc_model_1': 87.0,
        }
        val_metrics = {
            'val_loss': 0.6,
            'val_acc': 80.0,
        }
        
        monitor.update(1, train_metrics, val_metrics)
        
        assert 'train_acc_model_0' in monitor.history
        assert 'train_acc_model_1' in monitor.history
        assert monitor.history['train_acc_model_0'][0] == 83.0
        assert monitor.history['train_acc_model_1'][0] == 87.0


class TestIntegration:
    """Test integration scenarios."""
    
    def test_typical_training_workflow(self):
        """Test typical training monitoring workflow."""
        monitor = TrainingMonitor(overfitting_threshold=5.0)
        
        # Simulate training for 20 epochs
        for epoch in range(1, 21):
            # Simulate improving then overfitting
            if epoch <= 10:
                train_acc = 50.0 + epoch * 4
                val_acc = 48.0 + epoch * 3.5
            else:
                train_acc = 90.0 + (epoch - 10) * 1
                val_acc = 83.0 - (epoch - 10) * 0.5
            
            train_metrics = {
                'train_loss': max(0.1, 2.0 - epoch * 0.1),
                'train_acc': train_acc,
            }
            val_metrics = {
                'val_loss': max(0.2, 2.1 - epoch * 0.09),
                'val_acc': val_acc,
            }
            
            monitor.update(epoch, train_metrics, val_metrics)
        
        # Check overfitting was detected
        assert monitor.is_overfitting()
        
        # Get report
        report = monitor.get_overfitting_report()
        assert report.status in [
            OverfittingStatus.MODERATE_OVERFITTING,
            OverfittingStatus.SEVERE_OVERFITTING
        ]
        
        # Check best epoch was identified
        best_epoch, _ = monitor.get_best_epoch('val_acc')
        assert best_epoch <= 12  # Should be before overfitting got bad
    
    def test_healthy_training(self):
        """Test monitoring healthy training (no overfitting)."""
        monitor = TrainingMonitor()
        
        # Simulate healthy training
        for epoch in range(1, 21):
            train_acc = 50.0 + epoch * 2
            val_acc = 48.0 + epoch * 1.9  # Close to train acc
            
            monitor.update(
                epoch,
                {'train_loss': 2.0 - epoch * 0.08, 'train_acc': train_acc},
                {'val_loss': 2.1 - epoch * 0.07, 'val_acc': val_acc}
            )
        
        # Should not detect overfitting
        assert not monitor.is_overfitting()
        assert monitor.get_overfitting_severity() in [
            OverfittingStatus.NO_OVERFITTING,
            OverfittingStatus.MILD_OVERFITTING
        ]
