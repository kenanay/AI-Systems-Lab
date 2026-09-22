"""
src/training/metrics_logger.py

Training Metrics Logger

Bu modül training sırasında tüm metrics'leri toplar ve saklar:
- Structured metric storage (timestamped)
- JSON/CSV export
- Metric history tracking
- Aggregation ve statistics
- Query interface

Logger, training sürecinin her anını kaydeder ve sonradan
analiz için structured format'ta saklar.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricEntry:
    """
    Single metric entry with timestamp.
    
    Bir training step'indeki tüm metrics'leri içerir:
    - step: Global training step
    - epoch: Training epoch
    - timestamp: ISO format timestamp
    - metrics: Dict of metric name → value
    - phase: 'train' or 'val'
    
    Attributes:
        step: Global training step number
        epoch: Training epoch number
        phase: Training phase ('train', 'val', 'test')
        timestamp: ISO 8601 timestamp string
        metrics: Dictionary of metric names to values
    
    Example:
        >>> entry = MetricEntry(
        ...     step=100,
        ...     epoch=0,
        ...     phase='train',
        ...     timestamp='2024-01-01T12:00:00',
        ...     metrics={'loss': 0.5, 'lr': 0.001}
        ... )
    """
    step: int
    epoch: int
    phase: str
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricEntry":
        """Create from dictionary."""
        return cls(**data)


class TrainingLogger:
    """
    Training metrics logger.
    
    Collects and stores all training metrics with timestamps.
    Supports multiple output formats (JSON, CSV) and query interface.
    
    Features:
    - Automatic timestamping
    - Metric history tracking
    - JSON/CSV export
    - Statistics calculation
    - Query by step/phase
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of experiment/run
        auto_save: Auto-save after each log (slower but safer)
    
    Example:
        >>> logger = TrainingLogger(log_dir='logs', experiment_name='exp1')
        >>> logger.log_metrics(
        ...     step=0, epoch=0, phase='train',
        ...     metrics={'loss': 2.5, 'lr': 0.001}
        ... )
        >>> logger.save_json()
        >>> stats = logger.get_statistics('loss')
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str = "experiment",
        auto_save: bool = False
    ):
        """Initialize training logger."""
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.auto_save = auto_save
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metric history
        # List of MetricEntry objects
        self.history: List[MetricEntry] = []
        
        # Run metadata
        self.metadata: Dict[str, Any] = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'log_dir': str(log_dir)
        }
        
        logger.info(f"TrainingLogger initialized: {self.log_dir}/{experiment_name}")
    
    def log_metrics(
        self,
        step: int,
        epoch: int,
        phase: str,
        metrics: Dict[str, float]
    ):
        """
        Log metrics for a training step.
        
        Args:
            step: Global training step
            epoch: Current epoch
            phase: Training phase ('train', 'val', 'test')
            metrics: Dictionary of metric name → value
        
        Example:
            >>> logger.log_metrics(
            ...     step=100,
            ...     epoch=0,
            ...     phase='train',
            ...     metrics={
            ...         'loss': 0.5,
            ...         'lr': 0.001,
            ...         'grad_norm': 1.2
            ...     }
            ... )
        """
        entry = MetricEntry(
            step=step,
            epoch=epoch,
            phase=phase,
            timestamp=datetime.now().isoformat(),
            metrics=metrics
        )
        
        self.history.append(entry)
        
        if self.auto_save:
            self.save_json()
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameter name → value
        """
        self.metadata['hyperparameters'] = hparams
        logger.info(f"Hyperparameters logged: {len(hparams)} params")
    
    def log_metadata(self, **kwargs):
        """
        Log additional metadata.
        
        Args:
            **kwargs: Arbitrary metadata key-value pairs
        """
        self.metadata.update(kwargs)
    
    def get_metric_history(
        self,
        metric_name: str,
        phase: Optional[str] = None
    ) -> List[tuple]:
        """
        Get history of a specific metric.
        
        Args:
            metric_name: Name of metric
            phase: Filter by phase (optional)
        
        Returns:
            List of (step, value) tuples
        
        Example:
            >>> history = logger.get_metric_history('loss', phase='train')
            >>> steps, values = zip(*history)
        """
        result = []
        for entry in self.history:
            if phase and entry.phase != phase:
                continue
            if metric_name in entry.metrics:
                result.append((entry.step, entry.metrics[metric_name]))
        return result
    
    def get_statistics(
        self,
        metric_name: str,
        phase: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Name of metric
            phase: Filter by phase (optional)
        
        Returns:
            Dictionary with min, max, mean, std, count
        
        Example:
            >>> stats = logger.get_statistics('loss', phase='train')
            >>> print(f"Min: {stats['min']:.4f}, Mean: {stats['mean']:.4f}")
        """
        history = self.get_metric_history(metric_name, phase)
        
        if not history:
            return {}
        
        values = [v for _, v in history]
        
        import statistics
        
        return {
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'count': len(values),
            'last': values[-1]
        }
    
    def get_latest_metrics(self, phase: Optional[str] = None) -> Optional[Dict[str, float]]:
        """
        Get latest metrics.
        
        Args:
            phase: Filter by phase (optional)
        
        Returns:
            Latest metrics dictionary or None
        """
        for entry in reversed(self.history):
            if phase is None or entry.phase == phase:
                return entry.metrics
        return None
    
    def get_step_metrics(self, step: int) -> List[MetricEntry]:
        """
        Get all metrics for a specific step.
        
        Args:
            step: Training step
        
        Returns:
            List of MetricEntry objects for that step
        """
        return [e for e in self.history if e.step == step]
    
    def save_json(self, filename: Optional[str] = None):
        """
        Save metrics to JSON file.
        
        Args:
            filename: Output filename (default: experiment_name_metrics.json)
        
        Example:
            >>> logger.save_json('my_metrics.json')
        """
        if filename is None:
            filename = f"{self.experiment_name}_metrics.json"
        
        filepath = self.log_dir / filename
        
        data = {
            'metadata': self.metadata,
            'history': [entry.to_dict() for entry in self.history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics saved to JSON: {filepath}")
    
    def save_csv(self, filename: Optional[str] = None):
        """
        Save metrics to CSV file.
        
        Args:
            filename: Output filename (default: experiment_name_metrics.csv)
        
        Note:
            CSV format flattens the metrics dictionary into columns.
        
        Example:
            >>> logger.save_csv('my_metrics.csv')
        """
        if filename is None:
            filename = f"{self.experiment_name}_metrics.csv"
        
        filepath = self.log_dir / filename
        
        if not self.history:
            logger.warning("No metrics to save")
            return
        
        # Collect all unique metric names
        all_metrics = set()
        for entry in self.history:
            all_metrics.update(entry.metrics.keys())
        
        # CSV header: step, epoch, phase, timestamp, metric1, metric2, ...
        fieldnames = ['step', 'epoch', 'phase', 'timestamp'] + sorted(all_metrics)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in self.history:
                row = {
                    'step': entry.step,
                    'epoch': entry.epoch,
                    'phase': entry.phase,
                    'timestamp': entry.timestamp
                }
                row.update(entry.metrics)
                writer.writerow(row)
        
        logger.info(f"Metrics saved to CSV: {filepath}")
    
    def load_json(self, filename: str):
        """
        Load metrics from JSON file.
        
        Args:
            filename: Input filename
        
        Example:
            >>> logger.load_json('previous_run.json')
        """
        filepath = self.log_dir / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metadata = data.get('metadata', {})
        self.history = [MetricEntry.from_dict(e) for e in data.get('history', [])]
        
        logger.info(f"Metrics loaded from JSON: {filepath} ({len(self.history)} entries)")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of logged metrics.
        
        Returns:
            Dictionary with summary information
        """
        # Collect unique metrics and phases
        all_metrics = set()
        all_phases = set()
        
        for entry in self.history:
            all_metrics.update(entry.metrics.keys())
            all_phases.add(entry.phase)
        
        # Get latest step
        latest_step = max((e.step for e in self.history), default=0)
        
        # Statistics for key metrics
        stats = {}
        for metric in all_metrics:
            stats[metric] = {}
            for phase in all_phases:
                phase_stats = self.get_statistics(metric, phase)
                if phase_stats:
                    stats[metric][phase] = phase_stats
        
        return {
            'experiment_name': self.experiment_name,
            'total_entries': len(self.history),
            'latest_step': latest_step,
            'metrics': sorted(all_metrics),
            'phases': sorted(all_phases),
            'statistics': stats,
            'metadata': self.metadata
        }
    
    def print_summary(self):
        """Print summary of logged metrics."""
        summary = self.get_summary()
        
        print("=" * 80)
        print(f"Training Metrics Summary: {summary['experiment_name']}")
        print("=" * 80)
        print(f"\nTotal entries: {summary['total_entries']}")
        print(f"Latest step: {summary['latest_step']}")
        print(f"Metrics tracked: {', '.join(summary['metrics'])}")
        print(f"Phases: {', '.join(summary['phases'])}")
        
        print("\nMetric Statistics:")
        for metric, phase_stats in summary['statistics'].items():
            print(f"\n  {metric}:")
            for phase, stats in phase_stats.items():
                print(f"    {phase}:")
                print(f"      Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
                print(f"      Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                print(f"      Count: {stats['count']}, Last: {stats['last']:.4f}")


if __name__ == "__main__":
    # Test metrics logger
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Metrics Logger Test")
    print("=" * 80)
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create logger
        print("\n1. Creating logger...")
        metrics_logger = TrainingLogger(
            log_dir=tmpdir,
            experiment_name='test_run'
        )
        
        # Log hyperparameters
        print("\n2. Logging hyperparameters...")
        metrics_logger.log_hyperparameters({
            'learning_rate': 0.001,
            'batch_size': 32,
            'model': 'GPT'
        })
        
        # Simulate training
        print("\n3. Simulating training (20 steps)...")
        for step in range(20):
            # Training metrics
            metrics_logger.log_metrics(
                step=step,
                epoch=0,
                phase='train',
                metrics={
                    'loss': 2.0 - step * 0.05,
                    'lr': 0.001 * (1 - step / 20),
                    'grad_norm': 1.5 - step * 0.02
                }
            )
            
            # Validation every 5 steps
            if step % 5 == 0 and step > 0:
                metrics_logger.log_metrics(
                    step=step,
                    epoch=0,
                    phase='val',
                    metrics={
                        'loss': 2.1 - step * 0.04,
                        'accuracy': 0.5 + step * 0.02
                    }
                )
        
        # Get statistics
        print("\n4. Computing statistics...")
        train_loss_stats = metrics_logger.get_statistics('loss', phase='train')
        print(f"   Train loss: min={train_loss_stats['min']:.3f}, "
              f"mean={train_loss_stats['mean']:.3f}, "
              f"last={train_loss_stats['last']:.3f}")
        
        # Get history
        print("\n5. Getting metric history...")
        loss_history = metrics_logger.get_metric_history('loss', phase='train')
        print(f"   Train loss history: {len(loss_history)} entries")
        print(f"   First 3: {loss_history[:3]}")
        
        # Save files
        print("\n6. Saving to files...")
        metrics_logger.save_json()
        metrics_logger.save_csv()
        
        # Load back
        print("\n7. Loading from JSON...")
        new_logger = TrainingLogger(log_dir=tmpdir, experiment_name='loaded')
        new_logger.load_json('test_run_metrics.json')
        print(f"   Loaded {len(new_logger.history)} entries")
        
        # Print summary
        print("\n8. Summary:")
        metrics_logger.print_summary()
    
    print("\n" + "=" * 80)
    print("✅ Metrics logger tests completed!")
    print("=" * 80)
