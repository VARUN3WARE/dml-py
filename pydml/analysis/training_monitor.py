"""
Training Monitoring and Overfitting Detection.

This module provides tools for monitoring training progress, detecting
overfitting, and analyzing model performance during training.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings


class OverfittingStatus(Enum):
    """Status of overfitting detection."""
    NO_OVERFITTING = "no_overfitting"
    MILD_OVERFITTING = "mild_overfitting"
    MODERATE_OVERFITTING = "moderate_overfitting"
    SEVERE_OVERFITTING = "severe_overfitting"
    UNDERFITTING = "underfitting"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class TrainingMetrics:
    """Container for training metrics at a specific epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    
    @property
    def generalization_gap(self) -> float:
        """Calculate generalization gap (train_acc - val_acc)."""
        return self.train_acc - self.val_acc
    
    @property
    def loss_gap(self) -> float:
        """Calculate loss gap (val_loss - train_loss)."""
        return self.val_loss - self.train_loss
    
    def is_overfitting(self, threshold: float = 5.0) -> bool:
        """
        Check if overfitting based on generalization gap.
        
        Args:
            threshold: Accuracy gap threshold (default: 5%)
            
        Returns:
            True if generalization gap exceeds threshold
        """
        return self.generalization_gap > threshold


@dataclass
class OverfittingReport:
    """Detailed overfitting analysis report."""
    status: OverfittingStatus
    generalization_gap: float
    loss_gap: float
    train_acc: float
    val_acc: float
    train_loss: float
    val_loss: float
    recommendations: List[str]
    confidence: float
    
    def __str__(self) -> str:
        """Format report as string."""
        lines = [
            "=" * 60,
            "Overfitting Analysis Report",
            "=" * 60,
            f"Status: {self.status.value.replace('_', ' ').title()}",
            f"Confidence: {self.confidence:.1f}%",
            "",
            "Metrics:",
            f"  Train Accuracy: {self.train_acc:.2f}%",
            f"  Val Accuracy:   {self.val_acc:.2f}%",
            f"  Generalization Gap: {self.generalization_gap:+.2f}%",
            "",
            f"  Train Loss: {self.train_loss:.4f}",
            f"  Val Loss:   {self.val_loss:.4f}",
            f"  Loss Gap:   {self.loss_gap:+.4f}",
            "",
        ]
        
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class TrainingMonitor:
    """
    Monitor training progress and detect overfitting.
    
    This class tracks training and validation metrics over epochs,
    analyzes trends, and provides recommendations for improving training.
    
    Attributes:
        history: Dictionary of metric lists
        window_size: Window size for trend analysis
        overfitting_threshold: Generalization gap threshold for overfitting
        
    Example:
        >>> monitor = TrainingMonitor()
        >>> for epoch in range(epochs):
        ...     train_metrics = trainer.train_epoch(train_loader, epoch)
        ...     val_metrics = trainer.evaluate(val_loader)
        ...     monitor.update(epoch, train_metrics, val_metrics)
        ...     
        ...     # Check for overfitting
        ...     if monitor.is_overfitting():
        ...         print("Overfitting detected!")
        ...         report = monitor.get_overfitting_report()
        ...         print(report)
    """
    
    def __init__(
        self,
        window_size: int = 5,
        overfitting_threshold: float = 5.0,
        track_per_model: bool = False,
    ):
        """
        Initialize training monitor.
        
        Args:
            window_size: Window size for trend analysis (default: 5)
            overfitting_threshold: Generalization gap threshold % (default: 5.0)
            track_per_model: Track per-model metrics for DML (default: False)
        """
        self.window_size = window_size
        self.overfitting_threshold = overfitting_threshold
        self.track_per_model = track_per_model
        
        self.history: Dict[str, List[float]] = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        
        self._overfitting_detected = False
        self._last_check_epoch = -1
    
    def update(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """
        Update monitor with new metrics.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
        """
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_metrics.get('train_loss', 0.0))
        self.history['val_loss'].append(val_metrics.get('val_loss', 0.0))
        self.history['train_acc'].append(train_metrics.get('train_acc', 0.0))
        self.history['val_acc'].append(val_metrics.get('val_acc', 0.0))
        
        # Track per-model metrics if enabled
        if self.track_per_model:
            for key, value in train_metrics.items():
                if key.startswith('train_acc_model_'):
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(value)
    
    def get_current_metrics(self) -> Optional[TrainingMetrics]:
        """
        Get current (latest) training metrics.
        
        Returns:
            TrainingMetrics object or None if no data
        """
        if not self.history['epoch']:
            return None
        
        return TrainingMetrics(
            epoch=self.history['epoch'][-1],
            train_loss=self.history['train_loss'][-1],
            val_loss=self.history['val_loss'][-1],
            train_acc=self.history['train_acc'][-1],
            val_acc=self.history['val_acc'][-1],
        )
    
    def get_generalization_gap(self) -> float:
        """
        Get current generalization gap (train_acc - val_acc).
        
        Returns:
            Generalization gap in percentage points
        """
        if not self.history['epoch']:
            return 0.0
        
        return self.history['train_acc'][-1] - self.history['val_acc'][-1]
    
    def is_overfitting(self, strict: bool = False) -> bool:
        """
        Check if overfitting is occurring.
        
        Args:
            strict: If True, requires sustained overfitting over window
            
        Returns:
            True if overfitting detected
        """
        if len(self.history['epoch']) < 2:
            return False
        
        current_gap = self.get_generalization_gap()
        
        if strict and len(self.history['epoch']) >= self.window_size:
            # Check if overfitting is sustained over window
            recent_gaps = [
                self.history['train_acc'][i] - self.history['val_acc'][i]
                for i in range(-self.window_size, 0)
            ]
            return all(gap > self.overfitting_threshold for gap in recent_gaps)
        
        return current_gap > self.overfitting_threshold
    
    def is_underfitting(self) -> bool:
        """
        Check if underfitting is occurring.
        
        Underfitting is detected when both train and val accuracy are low
        and not improving.
        
        Returns:
            True if underfitting detected
        """
        if len(self.history['epoch']) < self.window_size:
            return False
        
        # Check if both accuracies are low
        recent_train_acc = self.history['train_acc'][-self.window_size:]
        recent_val_acc = self.history['val_acc'][-self.window_size:]
        
        avg_train = sum(recent_train_acc) / len(recent_train_acc)
        avg_val = sum(recent_val_acc) / len(recent_val_acc)
        
        # Low performance threshold
        low_threshold = 60.0  # Adjust based on task
        
        # Check if not improving
        train_improving = recent_train_acc[-1] > recent_train_acc[0]
        val_improving = recent_val_acc[-1] > recent_val_acc[0]
        
        return (avg_train < low_threshold and avg_val < low_threshold and
                not train_improving and not val_improving)
    
    def get_overfitting_severity(self) -> OverfittingStatus:
        """
        Classify overfitting severity.
        
        Returns:
            OverfittingStatus enum value
        """
        if len(self.history['epoch']) < 2:
            return OverfittingStatus.INSUFFICIENT_DATA
        
        gap = self.get_generalization_gap()
        
        # Check for underfitting first
        if self.is_underfitting():
            return OverfittingStatus.UNDERFITTING
        
        # Classify overfitting severity
        if gap < 3.0:
            return OverfittingStatus.NO_OVERFITTING
        elif gap < 5.0:
            return OverfittingStatus.MILD_OVERFITTING
        elif gap < 10.0:
            return OverfittingStatus.MODERATE_OVERFITTING
        else:
            return OverfittingStatus.SEVERE_OVERFITTING
    
    def get_overfitting_report(self) -> OverfittingReport:
        """
        Generate detailed overfitting analysis report.
        
        Returns:
            OverfittingReport with analysis and recommendations
        """
        current = self.get_current_metrics()
        if current is None:
            return OverfittingReport(
                status=OverfittingStatus.INSUFFICIENT_DATA,
                generalization_gap=0.0,
                loss_gap=0.0,
                train_acc=0.0,
                val_acc=0.0,
                train_loss=0.0,
                val_loss=0.0,
                recommendations=["Insufficient training data to analyze"],
                confidence=0.0,
            )
        
        status = self.get_overfitting_severity()
        gap = current.generalization_gap
        loss_gap = current.loss_gap
        
        # Generate recommendations
        recommendations = []
        confidence = 100.0
        
        if status == OverfittingStatus.SEVERE_OVERFITTING:
            recommendations = [
                "Add stronger regularization (increase weight decay)",
                "Add dropout layers (0.3-0.5)",
                "Reduce model capacity (fewer layers/units)",
                "Increase data augmentation",
                "Consider early stopping",
                "Add batch normalization if not present",
            ]
            confidence = min(100.0, 60.0 + gap * 2)
        
        elif status == OverfittingStatus.MODERATE_OVERFITTING:
            recommendations = [
                "Increase regularization (weight decay: 1e-4 to 5e-4)",
                "Add/increase dropout (0.2-0.3)",
                "Apply data augmentation",
                "Consider reducing learning rate",
                "Monitor validation metrics more closely",
            ]
            confidence = min(100.0, 50.0 + gap * 3)
        
        elif status == OverfittingStatus.MILD_OVERFITTING:
            recommendations = [
                "Monitor for increasing gap",
                "Consider light regularization",
                "Current performance is acceptable",
                "May improve with longer training",
            ]
            confidence = 70.0
        
        elif status == OverfittingStatus.UNDERFITTING:
            recommendations = [
                "Increase model capacity (more layers/units)",
                "Train for more epochs",
                "Increase learning rate",
                "Reduce regularization",
                "Check data quality and preprocessing",
                "Verify loss function is appropriate",
            ]
            confidence = 80.0
        
        elif status == OverfittingStatus.NO_OVERFITTING:
            recommendations = [
                "Training is progressing well",
                "Continue monitoring metrics",
                "Consider training longer if val accuracy improving",
            ]
            confidence = 90.0
        
        return OverfittingReport(
            status=status,
            generalization_gap=gap,
            loss_gap=loss_gap,
            train_acc=current.train_acc,
            val_acc=current.val_acc,
            train_loss=current.train_loss,
            val_loss=current.val_loss,
            recommendations=recommendations,
            confidence=confidence,
        )
    
    def get_trend(self, metric: str = 'val_acc', window: Optional[int] = None) -> str:
        """
        Analyze trend of a metric.
        
        Args:
            metric: Metric name to analyze
            window: Window size for trend (default: self.window_size)
            
        Returns:
            Trend description: 'improving', 'degrading', 'stable', 'insufficient_data'
        """
        if window is None:
            window = self.window_size
        
        if metric not in self.history or len(self.history[metric]) < window:
            return 'insufficient_data'
        
        recent_values = self.history[metric][-window:]
        
        # Simple linear trend
        first_half = sum(recent_values[:window//2]) / (window//2)
        second_half = sum(recent_values[window//2:]) / (len(recent_values) - window//2)
        
        diff = second_half - first_half
        
        # For loss metrics, lower is better
        if 'loss' in metric:
            if diff < -0.01:
                return 'improving'
            elif diff > 0.01:
                return 'degrading'
        else:  # For accuracy metrics, higher is better
            if diff > 0.5:
                return 'improving'
            elif diff < -0.5:
                return 'degrading'
        
        return 'stable'
    
    def should_stop_early(
        self,
        patience: int = 10,
        min_delta: float = 0.1,
        monitor: str = 'val_acc',
    ) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
            
        Returns:
            True if training should stop
        """
        if len(self.history[monitor]) < patience + 1:
            return False
        
        recent = self.history[monitor][-patience-1:]
        baseline = recent[0]  # Value from patience epochs ago
        current = recent[-1]
        
        # For accuracy, check if we haven't improved from baseline
        if 'acc' in monitor:
            improvement = current - baseline
            return improvement < min_delta
        else:  # For loss
            improvement = baseline - current
            return improvement < min_delta
    
    def get_best_epoch(self, metric: str = 'val_acc') -> Tuple[int, float]:
        """
        Get epoch with best metric value.
        
        Args:
            metric: Metric name
            
        Returns:
            Tuple of (epoch, value)
        """
        if metric not in self.history or not self.history[metric]:
            return -1, 0.0
        
        values = self.history[metric]
        
        if 'acc' in metric:
            best_idx = values.index(max(values))
        else:  # loss metrics
            best_idx = values.index(min(values))
        
        return self.history['epoch'][best_idx], values[best_idx]
    
    def get_summary(self) -> str:
        """
        Get formatted summary of training progress.
        
        Returns:
            Multi-line summary string
        """
        if not self.history['epoch']:
            return "No training data available"
        
        current = self.get_current_metrics()
        best_val_epoch, best_val_acc = self.get_best_epoch('val_acc')
        
        lines = [
            "=" * 60,
            "Training Progress Summary",
            "=" * 60,
            f"Current Epoch: {current.epoch}",
            f"Total Epochs Trained: {len(self.history['epoch'])}",
            "",
            "Current Metrics:",
            f"  Train Loss: {current.train_loss:.4f} | Train Acc: {current.train_acc:.2f}%",
            f"  Val Loss:   {current.val_loss:.4f} | Val Acc:   {current.val_acc:.2f}%",
            f"  Generalization Gap: {current.generalization_gap:+.2f}%",
            "",
            "Best Performance:",
            f"  Best Val Acc: {best_val_acc:.2f}% (epoch {best_val_epoch})",
            "",
            "Trends:",
            f"  Train Acc: {self.get_trend('train_acc')}",
            f"  Val Acc:   {self.get_trend('val_acc')}",
            f"  Val Loss:  {self.get_trend('val_loss')}",
            "",
        ]
        
        # Add overfitting status
        status = self.get_overfitting_severity()
        lines.append(f"Status: {status.value.replace('_', ' ').title()}")
        
        if status in [OverfittingStatus.MODERATE_OVERFITTING, OverfittingStatus.SEVERE_OVERFITTING]:
            lines.append("⚠️  Warning: Overfitting detected!")
        
        lines.append("=" * 60)
        return "\n".join(lines)


__all__ = [
    'OverfittingStatus',
    'TrainingMetrics',
    'OverfittingReport',
    'TrainingMonitor',
]
