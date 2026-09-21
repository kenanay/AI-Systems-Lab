"""
src/training/training_monitor.py

Real-Time Training Monitor

Bu modül training sürecini gerçek zamanlı izler:
- Training status (PENDING/RUNNING/PAUSED/COMPLETED/FAILED)
- Progress tracking (step/epoch/percentage)
- ETA (estimated time of arrival) calculation
- Performance metrics (steps/sec, samples/sec)
- Resource monitoring (memory, GPU usage optional)
- Live updates for UI/logging

Training monitor, training pipeline'ının durumunu
dışarıdan query edilebilir biçimde tutar ve
progress bar'lar için gerekli bilgiyi sağlar.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path
import time
import json
import logging

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training status enum."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressSnapshot:
    """
    Training progress snapshot.
    
    Attributes:
        step: Current training step
        epoch: Current epoch
        total_steps: Total steps planned
        total_epochs: Total epochs planned
        progress_pct: Progress percentage (0-100)
        timestamp: Snapshot timestamp
    """
    step: int
    epoch: int
    total_steps: int
    total_epochs: int
    progress_pct: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step': self.step,
            'epoch': self.epoch,
            'total_steps': self.total_steps,
            'total_epochs': self.total_epochs,
            'progress_pct': self.progress_pct,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PerformanceMetrics:
    """
    Training performance metrics.
    
    Attributes:
        steps_per_sec: Steps processed per second
        samples_per_sec: Samples processed per second
        avg_step_time: Average time per step (seconds)
        memory_used_mb: Memory used (MB, optional)
        gpu_utilization_pct: GPU utilization % (optional)
    """
    steps_per_sec: float
    samples_per_sec: float
    avg_step_time: float
    memory_used_mb: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'steps_per_sec': round(self.steps_per_sec, 2),
            'samples_per_sec': round(self.samples_per_sec, 2),
            'avg_step_time': round(self.avg_step_time, 4),
            'memory_used_mb': round(self.memory_used_mb, 2) if self.memory_used_mb else None,
            'gpu_utilization_pct': round(self.gpu_utilization_pct, 1) if self.gpu_utilization_pct else None
        }


class TrainingMonitor:
    """
    Real-time training monitor.
    
    Tracks training progress, estimates completion time,
    and provides performance metrics.
    
    Features:
    - Status management (PENDING/RUNNING/COMPLETED/etc.)
    - Progress tracking (steps, epochs, percentage)
    - ETA calculation based on recent performance
    - Performance metrics (throughput, avg time)
    - Resource monitoring hooks
    - JSON export for external monitoring
    
    Args:
        total_steps: Total training steps
        total_epochs: Total epochs
        batch_size: Batch size for samples/sec calculation
        monitor_file: Optional file to save monitor state
        eta_window: Number of recent steps to use for ETA (default: 100)
    
    Example:
        >>> monitor = TrainingMonitor(total_steps=1000, total_epochs=10)
        >>> monitor.start()
        >>> for step in range(1000):
        ...     monitor.update(step=step, epoch=step//100)
        ...     eta = monitor.get_eta()
        ...     print(f"Step {step}, ETA: {eta}")
        >>> monitor.complete()
    """
    
    def __init__(
        self,
        total_steps: int,
        total_epochs: int,
        batch_size: int = 32,
        monitor_file: Optional[Path] = None,
        eta_window: int = 100
    ):
        """Initialize training monitor."""
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.monitor_file = Path(monitor_file) if monitor_file else None
        self.eta_window = eta_window
        
        # Status
        self.status = TrainingStatus.PENDING
        self.current_step = 0
        self.current_epoch = 0
        
        # Timing
        self.start_time: Optional[datetime] = None
        self.pause_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.total_paused_duration = timedelta(0)
        
        # Performance tracking
        self.step_times: List[float] = []  # Recent step times for ETA
        self.last_step_time: Optional[float] = None
        
        # Metrics history
        self.metrics_history: List[Dict[str, Any]] = []
        
        logger.info(f"TrainingMonitor initialized: {total_steps} steps, {total_epochs} epochs")
    
    def start(self):
        """Start training monitoring."""
        if self.status == TrainingStatus.PENDING:
            self.status = TrainingStatus.RUNNING
            self.start_time = datetime.now()
            self.last_step_time = time.time()
            logger.info("Training started")
            self._save_state()
    
    def pause(self):
        """Pause training."""
        if self.status == TrainingStatus.RUNNING:
            self.status = TrainingStatus.PAUSED
            self.pause_time = datetime.now()
            logger.info("Training paused")
            self._save_state()
    
    def resume(self):
        """Resume training from pause."""
        if self.status == TrainingStatus.PAUSED and self.pause_time:
            self.status = TrainingStatus.RUNNING
            paused_duration = datetime.now() - self.pause_time
            self.total_paused_duration += paused_duration
            self.pause_time = None
            self.last_step_time = time.time()  # Reset step timer
            logger.info(f"Training resumed (was paused for {paused_duration})")
            self._save_state()
    
    def complete(self):
        """Mark training as completed."""
        self.status = TrainingStatus.COMPLETED
        self.end_time = datetime.now()
        logger.info("Training completed")
        self._save_state()
    
    def fail(self, error_message: str):
        """Mark training as failed."""
        self.status = TrainingStatus.FAILED
        self.end_time = datetime.now()
        logger.error(f"Training failed: {error_message}")
        self._save_state()
    
    def cancel(self):
        """Cancel training."""
        self.status = TrainingStatus.CANCELLED
        self.end_time = datetime.now()
        logger.info("Training cancelled")
        self._save_state()
    
    def update(
        self,
        step: int,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Update training progress.
        
        Args:
            step: Current step
            epoch: Current epoch
            metrics: Optional metrics dict (loss, accuracy, etc.)
        """
        self.current_step = step
        self.current_epoch = epoch
        
        # Track step time
        current_time = time.time()
        if self.last_step_time is not None:
            step_duration = current_time - self.last_step_time
            self.step_times.append(step_duration)
            
            # Keep only recent history for ETA
            if len(self.step_times) > self.eta_window:
                self.step_times.pop(0)
        
        self.last_step_time = current_time
        
        # Store metrics
        if metrics:
            self.metrics_history.append({
                'step': step,
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                **metrics
            })
    
    def get_progress(self) -> ProgressSnapshot:
        """
        Get current progress snapshot.
        
        Returns:
            ProgressSnapshot with current progress
        """
        progress_pct = (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0.0
        
        return ProgressSnapshot(
            step=self.current_step,
            epoch=self.current_epoch,
            total_steps=self.total_steps,
            total_epochs=self.total_epochs,
            progress_pct=progress_pct
        )
    
    def get_eta(self) -> Optional[timedelta]:
        """
        Estimate time to completion.
        
        Returns:
            Estimated time remaining, or None if not enough data
        
        Example:
            >>> eta = monitor.get_eta()
            >>> if eta:
            ...     print(f"ETA: {eta}")
        """
        if not self.step_times or self.current_step >= self.total_steps:
            return None
        
        # Average step time from recent history
        avg_step_time = sum(self.step_times) / len(self.step_times)
        
        # Remaining steps
        remaining_steps = self.total_steps - self.current_step
        
        # Estimated seconds
        estimated_seconds = avg_step_time * remaining_steps
        
        return timedelta(seconds=estimated_seconds)
    
    def get_performance(self) -> Optional[PerformanceMetrics]:
        """
        Get current performance metrics.
        
        Returns:
            PerformanceMetrics or None if not enough data
        """
        if not self.step_times:
            return None
        
        avg_step_time = sum(self.step_times) / len(self.step_times)
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
        samples_per_sec = steps_per_sec * self.batch_size
        
        return PerformanceMetrics(
            steps_per_sec=steps_per_sec,
            samples_per_sec=samples_per_sec,
            avg_step_time=avg_step_time
        )
    
    def get_elapsed_time(self) -> Optional[timedelta]:
        """
        Get elapsed training time (excluding pauses).
        
        Returns:
            Time elapsed since start, or None if not started
        """
        if not self.start_time:
            return None
        
        if self.end_time:
            # Training finished
            total_time = self.end_time - self.start_time
        elif self.pause_time:
            # Currently paused
            total_time = self.pause_time - self.start_time
        else:
            # Currently running
            total_time = datetime.now() - self.start_time
        
        # Subtract paused duration
        return total_time - self.total_paused_duration
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary with all monitoring information
        """
        progress = self.get_progress()
        performance = self.get_performance()
        eta = self.get_eta()
        elapsed = self.get_elapsed_time()
        
        summary = {
            'status': self.status.value,
            'progress': progress.to_dict(),
            'performance': performance.to_dict() if performance else None,
            'eta_seconds': eta.total_seconds() if eta else None,
            'eta_formatted': str(eta).split('.')[0] if eta else None,  # HH:MM:SS
            'elapsed_seconds': elapsed.total_seconds() if elapsed else None,
            'elapsed_formatted': str(elapsed).split('.')[0] if elapsed else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_paused_seconds': self.total_paused_duration.total_seconds()
        }
        
        return summary
    
    def print_summary(self):
        """Print formatted training summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("Training Monitor Summary")
        print("=" * 60)
        
        print(f"\nStatus: {summary['status'].upper()}")
        
        progress = summary['progress']
        print(f"\nProgress:")
        print(f"  Step: {progress['step']:,} / {progress['total_steps']:,}")
        print(f"  Epoch: {progress['epoch']} / {progress['total_epochs']}")
        print(f"  Completion: {progress['progress_pct']:.1f}%")
        
        if summary['performance']:
            perf = summary['performance']
            print(f"\nPerformance:")
            print(f"  Steps/sec: {perf['steps_per_sec']:.2f}")
            print(f"  Samples/sec: {perf['samples_per_sec']:.1f}")
            print(f"  Avg step time: {perf['avg_step_time']:.4f}s")
        
        if summary['eta_formatted']:
            print(f"\nETA: {summary['eta_formatted']}")
        
        if summary['elapsed_formatted']:
            print(f"Elapsed: {summary['elapsed_formatted']}")
        
        print("=" * 60 + "\n")
    
    def _save_state(self):
        """Save monitor state to file."""
        if not self.monitor_file:
            return
        
        try:
            summary = self.get_summary()
            with open(self.monitor_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save monitor state: {e}")
    
    def load_state(self):
        """Load monitor state from file."""
        if not self.monitor_file or not self.monitor_file.exists():
            return
        
        try:
            with open(self.monitor_file, 'r') as f:
                state = json.load(f)
            
            self.status = TrainingStatus(state['status'])
            self.current_step = state['progress']['step']
            self.current_epoch = state['progress']['epoch']
            
            if state['start_time']:
                self.start_time = datetime.fromisoformat(state['start_time'])
            if state['end_time']:
                self.end_time = datetime.fromisoformat(state['end_time'])
            
            logger.info(f"Monitor state loaded from {self.monitor_file}")
        except Exception as e:
            logger.warning(f"Failed to load monitor state: {e}")


if __name__ == "__main__":
    # Test training monitor
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Training Monitor Test")
    print("=" * 80)
    
    # Simulate training
    total_steps = 100
    total_epochs = 5
    batch_size = 32
    
    monitor = TrainingMonitor(
        total_steps=total_steps,
        total_epochs=total_epochs,
        batch_size=batch_size
    )
    
    print("\n1. Starting training...")
    monitor.start()
    assert monitor.status == TrainingStatus.RUNNING
    
    print("\n2. Simulating training steps...")
    for step in range(total_steps):
        epoch = step // (total_steps // total_epochs)
        
        # Simulate work
        time.sleep(0.01)
        
        # Update monitor
        metrics = {
            'loss': 2.0 * (1 - step / total_steps),  # Decreasing loss
            'accuracy': step / total_steps  # Increasing accuracy
        }
        monitor.update(step=step, epoch=epoch, metrics=metrics)
        
        # Print progress every 20 steps
        if step % 20 == 0 or step == total_steps - 1:
            progress = monitor.get_progress()
            eta = monitor.get_eta()
            perf = monitor.get_performance()
            
            print(f"\n  Step {step}/{total_steps} ({progress.progress_pct:.1f}%)")
            if eta:
                print(f"    ETA: {str(eta).split('.')[0]}")
            if perf:
                print(f"    Speed: {perf.steps_per_sec:.2f} steps/sec, {perf.samples_per_sec:.1f} samples/sec")
    
    print("\n3. Testing pause/resume...")
    monitor.pause()
    assert monitor.status == TrainingStatus.PAUSED
    time.sleep(0.5)
    monitor.resume()
    assert monitor.status == TrainingStatus.RUNNING
    
    print("\n4. Completing training...")
    monitor.complete()
    assert monitor.status == TrainingStatus.COMPLETED
    
    print("\n5. Final summary...")
    monitor.print_summary()
    
    # Test summary data
    summary = monitor.get_summary()
    assert summary['progress']['step'] == total_steps - 1
    assert summary['status'] == 'completed'
    assert summary['elapsed_seconds'] is not None
    assert summary['performance'] is not None
    
    print("\n✓ Monitor test completed!")
    print(f"  - Total steps: {summary['progress']['step'] + 1}")
    print(f"  - Elapsed time: {summary['elapsed_formatted']}")
    print(f"  - Average speed: {summary['performance']['steps_per_sec']:.2f} steps/sec")
    print(f"  - Paused duration: {summary['total_paused_seconds']:.2f}s")
    
    print("\n" + "=" * 80)
    print("✅ Training Monitor tests completed!")
    print("=" * 80)
