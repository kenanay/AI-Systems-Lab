"""
src/training/tensorboard_logger.py

TensorBoard Integration for Training Visualization

Bu modül TensorBoard ile entegrasyon sağlar:
- SummaryWriter wrapper
- Automatic scalar logging (loss, lr, etc.)
- Histogram logging (weights, gradients)
- Model graph visualization
- Text/image logging support

TensorBoard, training sürecini real-time görselleştirmek için
industry-standard bir araçtır. Bu wrapper kullanımı kolaylaştırır.

Kullanım:
    tensorboard --logdir=logs
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

# TensorBoard optional dependency
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning(
        "TensorBoard not available. Install with: pip install tensorboard\n"
        "TensorBoardLogger will be disabled."
    )


class TensorBoardLogger:
    """
    TensorBoard logger wrapper.
    
    Wraps torch.utils.tensorboard.SummaryWriter with convenient methods
    for logging training metrics, model parameters, and visualizations.
    
    Features:
    - Scalar logging (loss, lr, metrics)
    - Histogram logging (weights, gradients)
    - Model graph visualization
    - Text logging
    - Graceful degradation if TensorBoard unavailable
    
    Args:
        log_dir: Directory for TensorBoard logs
        experiment_name: Name of experiment (creates subdirectory)
        enabled: Enable/disable logging (useful for debugging)
        flush_secs: How often to flush to disk (seconds)
    
    Example:
        >>> tb_logger = TensorBoardLogger(
        ...     log_dir='runs',
        ...     experiment_name='exp1'
        ... )
        >>> tb_logger.log_scalar('train/loss', 0.5, step=100)
        >>> tb_logger.log_model_parameters(model, step=100)
        >>> tb_logger.close()
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str = "experiment",
        enabled: bool = True,
        flush_secs: int = 30
    ):
        """Initialize TensorBoard logger."""
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        
        if not self.enabled:
            if not TENSORBOARD_AVAILABLE:
                logger.warning("TensorBoard not available, logging disabled")
            else:
                logger.info("TensorBoard logging disabled")
            self.writer = None
            return
        
        # Create log directory
        full_log_dir = self.log_dir / experiment_name
        full_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create SummaryWriter
        self.writer = SummaryWriter(
            log_dir=str(full_log_dir),
            flush_secs=flush_secs
        )
        
        logger.info(f"TensorBoard logging enabled: {full_log_dir}")
        logger.info(f"  Start TensorBoard with: tensorboard --logdir={self.log_dir}")
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int
    ):
        """
        Log scalar value.
        
        Args:
            tag: Name/tag for the scalar (e.g., 'train/loss', 'val/accuracy')
            value: Scalar value to log
            step: Training step (x-axis in TensorBoard)
        
        Example:
            >>> tb_logger.log_scalar('train/loss', 0.5, step=100)
            >>> tb_logger.log_scalar('train/lr', 0.001, step=100)
        """
        if not self.enabled:
            return
        
        self.writer.add_scalar(tag, value, global_step=step)
    
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: int
    ):
        """
        Log multiple scalars in a group.
        
        Args:
            main_tag: Main tag/group name
            tag_scalar_dict: Dictionary of tag → value
            step: Training step
        
        Example:
            >>> tb_logger.log_scalars(
            ...     'losses',
            ...     {'train': 0.5, 'val': 0.6},
            ...     step=100
            ... )
        """
        if not self.enabled:
            return
        
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step=step)
    
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: int
    ):
        """
        Log histogram of tensor values.
        
        Useful for visualizing weight distributions, gradient magnitudes, etc.
        
        Args:
            tag: Name/tag for the histogram
            values: Tensor containing values to histogram
            step: Training step
        
        Example:
            >>> # Log weight distribution
            >>> tb_logger.log_histogram(
            ...     'weights/layer1',
            ...     model.layer1.weight,
            ...     step=100
            ... )
        """
        if not self.enabled:
            return
        
        # values shape: arbitrary [...]
        # TensorBoard will create histogram from all values
        self.writer.add_histogram(tag, values, global_step=step)
    
    def log_model_parameters(
        self,
        model: nn.Module,
        step: int,
        log_gradients: bool = True
    ):
        """
        Log all model parameters as histograms.
        
        Automatically logs weight distributions for all model parameters.
        Optionally logs gradient distributions.
        
        Args:
            model: PyTorch model
            step: Training step
            log_gradients: Also log gradient histograms
        
        Example:
            >>> tb_logger.log_model_parameters(
            ...     model,
            ...     step=100,
            ...     log_gradients=True
            ... )
        """
        if not self.enabled:
            return
        
        for name, param in model.named_parameters():
            # Log parameter values
            # param.data shape: varies by layer
            self.writer.add_histogram(
                f'parameters/{name}',
                param.data,
                global_step=step
            )
            
            # Log gradients if available
            if log_gradients and param.grad is not None:
                # param.grad shape: same as param.data
                self.writer.add_histogram(
                    f'gradients/{name}',
                    param.grad,
                    global_step=step
                )
    
    def log_model_graph(
        self,
        model: nn.Module,
        input_shape: tuple
    ):
        """
        Log model computational graph.
        
        Visualizes model architecture in TensorBoard.
        
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor (without batch dimension)
                e.g., (seq_len,) for language models
        
        Example:
            >>> tb_logger.log_model_graph(
            ...     model,
            ...     input_shape=(128,)  # seq_len=128
            ... )
        """
        if not self.enabled:
            return
        
        try:
            # Create dummy input
            # dummy_input shape: [1, *input_shape]
            dummy_input = torch.zeros(1, *input_shape, dtype=torch.long)
            
            # Log model graph
            self.writer.add_graph(model, dummy_input)
            logger.info("Model graph logged to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: int
    ):
        """
        Log text output.
        
        Useful for logging generated text samples during training.
        
        Args:
            tag: Name/tag for the text
            text: Text string to log
            step: Training step
        
        Example:
            >>> tb_logger.log_text(
            ...     'generated_samples',
            ...     'Once upon a time...',
            ...     step=100
            ... )
        """
        if not self.enabled:
            return
        
        self.writer.add_text(tag, text, global_step=step)
    
    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameter name → value
            metrics: Optional final metrics dictionary
        
        Example:
            >>> tb_logger.log_hyperparameters(
            ...     {'lr': 0.001, 'batch_size': 32},
            ...     {'final_loss': 0.5}
            ... )
        """
        if not self.enabled:
            return
        
        # Convert non-serializable values to strings
        clean_hparams = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                clean_hparams[k] = v
            else:
                clean_hparams[k] = str(v)
        
        self.writer.add_hparams(
            clean_hparams,
            metrics or {}
        )
    
    def log_metrics_dict(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ):
        """
        Log dictionary of metrics as scalars.
        
        Convenience method to log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric name → value
            step: Training step
            prefix: Optional prefix for all tags (e.g., 'train/', 'val/')
        
        Example:
            >>> metrics = {
            ...     'loss': 0.5,
            ...     'lr': 0.001,
            ...     'grad_norm': 1.2
            ... }
            >>> tb_logger.log_metrics_dict(metrics, step=100, prefix='train/')
        """
        if not self.enabled:
            return
        
        for name, value in metrics.items():
            tag = f"{prefix}{name}" if prefix else name
            self.log_scalar(tag, value, step)
    
    def flush(self):
        """Force flush to disk."""
        if self.enabled and self.writer:
            self.writer.flush()
    
    def close(self):
        """Close TensorBoard writer."""
        if self.enabled and self.writer:
            self.writer.close()
            logger.info("TensorBoard writer closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class DummyTensorBoardLogger:
    """
    Dummy logger when TensorBoard is not available.
    
    Provides same interface as TensorBoardLogger but does nothing.
    Prevents code from breaking when TensorBoard not installed.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize dummy logger."""
        self.enabled = False
    
    def log_scalar(self, *args, **kwargs):
        """Dummy method."""
        pass
    
    def log_scalars(self, *args, **kwargs):
        """Dummy method."""
        pass
    
    def log_histogram(self, *args, **kwargs):
        """Dummy method."""
        pass
    
    def log_model_parameters(self, *args, **kwargs):
        """Dummy method."""
        pass
    
    def log_model_graph(self, *args, **kwargs):
        """Dummy method."""
        pass
    
    def log_text(self, *args, **kwargs):
        """Dummy method."""
        pass
    
    def log_hyperparameters(self, *args, **kwargs):
        """Dummy method."""
        pass
    
    def log_metrics_dict(self, *args, **kwargs):
        """Dummy method."""
        pass
    
    def flush(self):
        """Dummy method."""
        pass
    
    def close(self):
        """Dummy method."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        pass


# Factory function
def create_tensorboard_logger(
    log_dir: Union[str, Path],
    experiment_name: str = "experiment",
    enabled: bool = True,
    **kwargs
) -> Union[TensorBoardLogger, DummyTensorBoardLogger]:
    """
    Create TensorBoard logger (or dummy if unavailable).
    
    Args:
        log_dir: Directory for logs
        experiment_name: Experiment name
        enabled: Enable logging
        **kwargs: Additional arguments for TensorBoardLogger
    
    Returns:
        TensorBoardLogger if available, DummyTensorBoardLogger otherwise
    
    Example:
        >>> tb_logger = create_tensorboard_logger('runs', 'exp1')
        >>> # Works regardless of TensorBoard installation
        >>> tb_logger.log_scalar('loss', 0.5, step=0)
    """
    if TENSORBOARD_AVAILABLE and enabled:
        return TensorBoardLogger(log_dir, experiment_name, enabled, **kwargs)
    else:
        return DummyTensorBoardLogger(log_dir, experiment_name, enabled, **kwargs)


if __name__ == "__main__":
    # Test TensorBoard logger
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("TensorBoard Logger Test")
    print("=" * 80)
    
    if not TENSORBOARD_AVAILABLE:
        print("\n⚠️  TensorBoard not available")
        print("Install with: pip install tensorboard")
        print("Testing with DummyLogger instead...\n")
    
    import tempfile
    from src.model.gpt import GPTModel, GPTConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create logger
        print("\n1. Creating TensorBoard logger...")
        tb_logger = create_tensorboard_logger(
            log_dir=tmpdir,
            experiment_name='test_run'
        )
        print(f"   Enabled: {tb_logger.enabled}")
        
        # Log scalars
        print("\n2. Logging scalars...")
        for step in range(10):
            tb_logger.log_scalar('train/loss', 2.0 - step * 0.1, step)
            tb_logger.log_scalar('train/lr', 0.001 * (1 - step/10), step)
        print("   ✓ 10 steps logged")
        
        # Log multiple scalars
        print("\n3. Logging scalar groups...")
        tb_logger.log_scalars(
            'losses',
            {'train': 0.5, 'val': 0.6},
            step=10
        )
        print("   ✓ Scalar group logged")
        
        # Log metrics dict
        print("\n4. Logging metrics dict...")
        metrics = {
            'loss': 0.5,
            'perplexity': 1.6,
            'accuracy': 0.75
        }
        tb_logger.log_metrics_dict(metrics, step=10, prefix='train/')
        print("   ✓ Metrics dict logged")
        
        # Create test model
        print("\n5. Creating test model...")
        config = GPTConfig(
            vocab_size=100,
            max_seq_len=32,
            d_model=64,
            n_layers=2,
            n_heads=2
        )
        model = GPTModel(config)
        print(f"   ✓ Model created: {model.get_num_params():,} params")
        
        # Log model parameters
        print("\n6. Logging model parameters...")
        tb_logger.log_model_parameters(model, step=0, log_gradients=False)
        print("   ✓ Parameters logged")
        
        # Log histograms
        print("\n7. Logging histograms...")
        weights = torch.randn(100, 50)
        tb_logger.log_histogram('test/weights', weights, step=0)
        print("   ✓ Histogram logged")
        
        # Log text
        print("\n8. Logging text...")
        tb_logger.log_text(
            'samples/generation',
            'This is a test generation sample.',
            step=0
        )
        print("   ✓ Text logged")
        
        # Log hyperparameters
        print("\n9. Logging hyperparameters...")
        tb_logger.log_hyperparameters(
            hparams={'lr': 0.001, 'batch_size': 32},
            metrics={'final_loss': 0.5}
        )
        print("   ✓ Hyperparameters logged")
        
        # Flush and close
        print("\n10. Closing logger...")
        tb_logger.flush()
        tb_logger.close()
        print("   ✓ Closed")
        
        if TENSORBOARD_AVAILABLE:
            print(f"\n📊 View logs with:")
            print(f"   tensorboard --logdir={tmpdir}")
    
    print("\n" + "=" * 80)
    print("✅ TensorBoard logger tests completed!")
    print("=" * 80)
