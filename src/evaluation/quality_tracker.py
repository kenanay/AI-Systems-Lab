"""
src/evaluation/quality_tracker.py

Quality Tracking System for Training

Model kalite izleme ve regression detection sistemi:
- Continuous Evaluation: Training sırasında otomatik değerlendirme
- Regression Detection: Performance düşüşlerini tespit etme
- Performance Monitoring: Metric history tracking
- Alert System: Threshold-based uyarı sistemi
- Checkpoint Integration: Checkpoint manager entegrasyonu

Bu modül production-ready quality tracking sağlar:
- Real-time monitoring: Training sırasında continuous tracking
- Historical analysis: Metric trend analysis
- Regression alerts: Otomatik regression detection
- Best model tracking: En iyi model'i takip
- Integration: Checkpoint manager ile entegrasyon

Quality Metrics:
    Regression Detection:
        - Compare current vs best performance
        - Threshold-based alerts (e.g., >5% drop)
        - Multiple consecutive degradations
    
    Trend Analysis:
        - Moving averages for smoothing
        - Slope computation for trends
        - Improvement/degradation detection
    
    Alert Levels:
        - INFO: Performance improvement
        - WARNING: Minor degradation (<5%)
        - ERROR: Major regression (>5%)
        - CRITICAL: Severe regression (>10%)

Kaynaklar:
    - ML Model Monitoring Best Practices
    - Continuous Evaluation in Production
    - https://github.com/evidentlyai/evidently

Usage:
    >>> from src.evaluation.quality_tracker import QualityTracker
    >>> 
    >>> # Create tracker
    >>> tracker = QualityTracker(
    ...     metrics_to_track=["accuracy", "perplexity"],
    ...     regression_threshold=0.05
    ... )
    >>> 
    >>> # Add evaluation
    >>> tracker.add_evaluation(epoch=1, results=eval_results)
    >>> 
    >>> # Check for regressions
    >>> alerts = tracker.check_regressions()
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import EvaluationResults, MetricResult

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# Alert System
# =========================

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityAlert:
    """
    Quality alert.
    
    Attributes:
        level: Alert severity level
        metric_name: Metric that triggered alert
        message: Alert message
        current_value: Current metric value
        best_value: Best observed value
        degradation_pct: Degradation percentage
        timestamp: When alert was created
    """
    level: AlertLevel
    metric_name: str
    message: str
    current_value: float
    best_value: float
    degradation_pct: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"[{self.level.value.upper()}] {self.metric_name}: {self.message} "
            f"(current={self.current_value:.4f}, best={self.best_value:.4f}, "
            f"degradation={self.degradation_pct:.2f}%)"
        )


# =========================
# Metric History
# =========================

@dataclass
class MetricHistory:
    """
    History for a single metric.
    
    Attributes:
        metric_name: Name of metric
        values: List of (epoch, value) tuples
        higher_is_better: True if higher values are better
        best_value: Best value observed
        best_epoch: Epoch with best value
    """
    metric_name: str
    values: List[Tuple[int, float]] = field(default_factory=list)
    higher_is_better: bool = True
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    
    def add_value(self, epoch: int, value: float) -> None:
        """
        Add a value.
        
        Args:
            epoch: Training epoch
            value: Metric value
        """
        self.values.append((epoch, value))
        
        # Update best
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
        else:
            if self.higher_is_better:
                if value > self.best_value:
                    self.best_value = value
                    self.best_epoch = epoch
            else:
                if value < self.best_value:
                    self.best_value = value
                    self.best_epoch = epoch
    
    def get_current_value(self) -> Optional[float]:
        """Get most recent value."""
        if not self.values:
            return None
        return self.values[-1][1]
    
    def get_moving_average(self, window: int = 3) -> List[float]:
        """
        Compute moving average.
        
        Args:
            window: Window size
            
        Returns:
            List of moving averages
        """
        if len(self.values) < window:
            return [v for _, v in self.values]
        
        values_only = [v for _, v in self.values]
        
        moving_avg = []
        for i in range(len(values_only)):
            start = max(0, i - window + 1)
            window_values = values_only[start:i+1]
            moving_avg.append(np.mean(window_values))
        
        return moving_avg
    
    def compute_trend(self, window: int = 5) -> float:
        """
        Compute trend (slope of recent values).
        
        Args:
            window: Number of recent values to consider
            
        Returns:
            float: Slope (positive = improving if higher_is_better)
        """
        if len(self.values) < 2:
            return 0.0
        
        # Get recent values
        recent = self.values[-window:]
        
        if len(recent) < 2:
            return 0.0
        
        # Linear regression
        epochs = np.array([e for e, _ in recent])
        values = np.array([v for _, v in recent])
        
        # Normalize epochs to 0-based
        epochs = epochs - epochs[0]
        
        # Slope
        if len(epochs) > 1:
            slope = np.polyfit(epochs, values, 1)[0]
        else:
            slope = 0.0
        
        return float(slope)
    
    def detect_regression(
        self,
        threshold: float = 0.05
    ) -> Optional[QualityAlert]:
        """
        Detect regression.
        
        Args:
            threshold: Degradation threshold (0.05 = 5%)
            
        Returns:
            QualityAlert if regression detected, None otherwise
        """
        current = self.get_current_value()
        
        if current is None or self.best_value is None:
            return None
        
        # Calculate degradation
        if self.higher_is_better:
            # For higher is better: degradation if current < best
            degradation = (self.best_value - current) / abs(self.best_value)
        else:
            # For lower is better: degradation if current > best
            degradation = (current - self.best_value) / abs(self.best_value)
        
        # Check threshold
        if degradation <= threshold:
            return None
        
        # Determine alert level
        if degradation > 0.10:  # >10%
            level = AlertLevel.CRITICAL
            msg = f"Severe regression detected"
        elif degradation > 0.05:  # >5%
            level = AlertLevel.ERROR
            msg = f"Major regression detected"
        else:  # >threshold but <5%
            level = AlertLevel.WARNING
            msg = f"Minor degradation detected"
        
        return QualityAlert(
            level=level,
            metric_name=self.metric_name,
            message=msg,
            current_value=current,
            best_value=self.best_value,
            degradation_pct=degradation * 100
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "values": self.values,
            "higher_is_better": self.higher_is_better,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "current_value": self.get_current_value(),
            "trend": self.compute_trend()
        }


# =========================
# Quality Tracker
# =========================

class QualityTracker:
    """
    Quality tracking system.
    
    Tracks model quality during training with regression detection.
    """
    
    def __init__(
        self,
        metrics_to_track: Optional[List[str]] = None,
        regression_threshold: float = 0.05,
        alert_on_improvement: bool = False,
        tracking_dir: Optional[str] = None
    ):
        """
        Initialize quality tracker.
        
        Args:
            metrics_to_track: List of metric names to track (None = all)
            regression_threshold: Threshold for regression detection (0.05 = 5%)
            alert_on_improvement: Whether to alert on improvements
            tracking_dir: Directory for saving tracking data
        """
        self.metrics_to_track = metrics_to_track
        self.regression_threshold = regression_threshold
        self.alert_on_improvement = alert_on_improvement
        self.tracking_dir = Path(tracking_dir) if tracking_dir else None
        
        self.metric_histories: Dict[str, MetricHistory] = {}
        self.alerts: List[QualityAlert] = []
        self.evaluations: List[Tuple[int, EvaluationResults]] = []
        
        if self.tracking_dir:
            self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"QualityTracker initialized "
            f"(threshold={regression_threshold}, metrics={metrics_to_track or 'all'})"
        )
    
    def add_evaluation(
        self,
        epoch: int,
        results: EvaluationResults
    ) -> List[QualityAlert]:
        """
        Add evaluation results.
        
        Args:
            epoch: Training epoch
            results: Evaluation results
            
        Returns:
            List[QualityAlert]: Alerts generated
        """
        logger.info(f"Adding evaluation for epoch {epoch}")
        
        # Store evaluation
        self.evaluations.append((epoch, results))
        
        # Track each metric
        new_alerts = []
        
        for metric in results.metrics:
            # Check if should track
            if self.metrics_to_track and metric.name not in self.metrics_to_track:
                continue
            
            # Get or create history
            if metric.name not in self.metric_histories:
                self.metric_histories[metric.name] = MetricHistory(
                    metric_name=metric.name,
                    higher_is_better=metric.higher_is_better
                )
            
            history = self.metric_histories[metric.name]
            
            # Add value
            prev_best = history.best_value
            history.add_value(epoch, metric.value)
            
            # Check for regression
            alert = history.detect_regression(self.regression_threshold)
            if alert:
                new_alerts.append(alert)
                self.alerts.append(alert)
                logger.warning(str(alert))
            
            # Check for improvement
            if self.alert_on_improvement and prev_best is not None:
                if history.best_value != prev_best:
                    improvement_alert = QualityAlert(
                        level=AlertLevel.INFO,
                        metric_name=metric.name,
                        message="New best value achieved",
                        current_value=metric.value,
                        best_value=float(history.best_value) if history.best_value is not None else 0.0,
                        degradation_pct=0.0
                    )
                    new_alerts.append(improvement_alert)
                    self.alerts.append(improvement_alert)
                    logger.info(str(improvement_alert))
        
        # Save if tracking directory set
        if self.tracking_dir:
            self._save_checkpoint(epoch)
        
        return new_alerts
    
    def check_regressions(self) -> List[QualityAlert]:
        """
        Check all metrics for regressions.
        
        Returns:
            List[QualityAlert]: Current regression alerts
        """
        alerts = []
        
        for history in self.metric_histories.values():
            alert = history.detect_regression(self.regression_threshold)
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def get_metric_history(self, metric_name: str) -> Optional[MetricHistory]:
        """
        Get history for a metric.
        
        Args:
            metric_name: Metric name
            
        Returns:
            MetricHistory or None
        """
        return self.metric_histories.get(metric_name)
    
    def get_summary(self) -> Dict:
        """
        Get tracking summary.
        
        Returns:
            Dict with summary statistics
        """
        summary = {
            "num_evaluations": len(self.evaluations),
            "num_alerts": len(self.alerts),
            "metrics": {}
        }
        
        for metric_name, history in self.metric_histories.items():
            summary["metrics"][metric_name] = {
                "best_value": history.best_value,
                "best_epoch": history.best_epoch,
                "current_value": history.get_current_value(),
                "trend": history.compute_trend(),
                "num_values": len(history.values)
            }
        
        # Alert breakdown
        alert_counts = {level: 0 for level in AlertLevel}
        for alert in self.alerts:
            alert_counts[alert.level] += 1
        
        summary["alert_breakdown"] = {
            level.value: count
            for level, count in alert_counts.items()
        }
        
        return summary
    
    def _save_checkpoint(self, epoch: int) -> None:
        """
        Save tracking checkpoint.
        
        Args:
            epoch: Current epoch
        """
        if self.tracking_dir is None:
            return
            
        checkpoint_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "metric_histories": {
                name: history.to_dict()
                for name, history in self.metric_histories.items()
            }
        }
        
        checkpoint_path = self.tracking_dir / f"quality_tracking_epoch_{epoch}.json"
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Quality tracking checkpoint saved: {checkpoint_path}")
    
    def save_report(self, path: Union[str, Path]) -> None:
        """
        Save complete tracking report.
        
        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "summary": self.get_summary(),
            "metric_histories": {
                name: history.to_dict()
                for name, history in self.metric_histories.items()
            },
            "alerts": [
                {
                    "level": alert.level.value,
                    "metric": alert.metric_name,
                    "message": alert.message,
                    "current_value": alert.current_value,
                    "best_value": alert.best_value,
                    "degradation_pct": alert.degradation_pct,
                    "timestamp": alert.timestamp
                }
                for alert in self.alerts
            ]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Quality tracking report saved: {path}")


# =========================
# Main (for testing)
# =========================

def main():
    """Test quality tracker."""
    print("\n" + "=" * 80)
    print("QUALITY TRACKER TEST")
    print("=" * 80)
    
    from src.evaluation.metrics import EvaluationResults
    
    # Create tracker
    tracker = QualityTracker(
        metrics_to_track=["accuracy", "perplexity"],
        regression_threshold=0.05,
        alert_on_improvement=True
    )
    
    print("\n" + "-" * 80)
    print("Simulating Training with Quality Tracking")
    print("-" * 80)
    
    # Simulate training epochs
    epochs_data = [
        # Epoch 1: Initial
        {"accuracy": 0.70, "perplexity": 65.0},
        # Epoch 2: Improvement
        {"accuracy": 0.75, "perplexity": 58.0},
        # Epoch 3: More improvement
        {"accuracy": 0.80, "perplexity": 52.0},
        # Epoch 4: Best
        {"accuracy": 0.85, "perplexity": 48.0},
        # Epoch 5: Minor regression
        {"accuracy": 0.83, "perplexity": 50.0},
        # Epoch 6: Major regression
        {"accuracy": 0.78, "perplexity": 55.0},
    ]
    
    for epoch, data in enumerate(epochs_data, 1):
        print(f"\n--- Epoch {epoch} ---")
        
        # Create results
        results = EvaluationResults()
        results.add_metric("accuracy", data["accuracy"], "%", higher_is_better=True)
        results.add_metric("perplexity", data["perplexity"], "", higher_is_better=False)
        
        # Add to tracker
        alerts = tracker.add_evaluation(epoch, results)
        
        # Print metrics
        print(f"Metrics: accuracy={data['accuracy']:.2f}, perplexity={data['perplexity']:.1f}")
        
        # Print alerts
        if alerts:
            print("Alerts:")
            for alert in alerts:
                print(f"  {alert}")
        else:
            print("No alerts")
    
    # Summary
    print("\n" + "-" * 80)
    print("Tracking Summary")
    print("-" * 80)
    
    summary = tracker.get_summary()
    print(f"Total evaluations: {summary['num_evaluations']}")
    print(f"Total alerts: {summary['num_alerts']}")
    
    print("\nMetric Best Values:")
    for metric_name, data in summary['metrics'].items():
        print(f"  {metric_name}:")
        print(f"    Best: {data['best_value']:.4f} (epoch {data['best_epoch']})")
        print(f"    Current: {data['current_value']:.4f}")
        print(f"    Trend: {data['trend']:.6f}")
    
    print("\nAlert Breakdown:")
    for level, count in summary['alert_breakdown'].items():
        if count > 0:
            print(f"  {level}: {count}")
    
    # Check current regressions
    print("\n" + "-" * 80)
    print("Current Regressions")
    print("-" * 80)
    
    regressions = tracker.check_regressions()
    if regressions:
        for alert in regressions:
            print(f"  {alert}")
    else:
        print("  No active regressions")
    
    # Save report
    tracker.save_report("quality_tracking_test.json")
    
    print("\n" + "=" * 80)
    print("✓ Quality tracker test completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
