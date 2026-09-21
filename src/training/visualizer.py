"""
src/training/visualizer.py

Training Metrics Visualization

Bu modül training metrics'leri için matplotlib plots oluşturur:
- Loss curves (train/val)
- Learning rate schedule
- Accuracy curves
- Gradient norm progression
- Multi-metric comparison
- Custom plot generation

Matplotlib plots, metrics'leri statik görselleştirmek ve
raporlar oluşturmak için kullanılır.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """
    Training metrics visualizer.
    
    Creates matplotlib plots for training metrics including
    loss curves, learning rate schedules, and custom visualizations.
    
    Features:
    - Loss curves (train/val with smoothing)
    - Learning rate schedule
    - Multi-metric comparison
    - Gradient norm tracking
    - Customizable plots
    - Automatic save to files
    
    Args:
        save_dir: Directory to save plots
        style: Matplotlib style ('default', 'seaborn', 'ggplot')
        figsize: Default figure size (width, height)
        dpi: Resolution for saved plots
    
    Example:
        >>> viz = TrainingVisualizer(save_dir='plots')
        >>> viz.plot_loss_curves(
        ...     train_losses=[(0, 2.5), (10, 1.5), (20, 0.8)],
        ...     val_losses=[(0, 2.6), (10, 1.6), (20, 0.9)]
        ... )
        >>> viz.save_all_plots()
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        style: str = 'seaborn-v0_8',
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100
    ):
        """Initialize visualizer."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
        
        # Set style
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default")
            plt.style.use('default')
        
        # Store figures
        self.figures: Dict[str, plt.Figure] = {}
        
        logger.info(f"TrainingVisualizer initialized: {save_dir}")
    
    def plot_loss_curves(
        self,
        train_losses: List[Tuple[int, float]],
        val_losses: Optional[List[Tuple[int, float]]] = None,
        title: str = "Training Loss",
        smooth_window: int = 0
    ) -> plt.Figure:
        """
        Plot loss curves.
        
        Args:
            train_losses: List of (step, loss) tuples for training
            val_losses: List of (step, loss) tuples for validation (optional)
            title: Plot title
            smooth_window: Moving average window size (0 = no smoothing)
        
        Returns:
            Matplotlib figure
        
        Example:
            >>> viz.plot_loss_curves(
            ...     train_losses=[(0, 2.5), (100, 1.2), (200, 0.8)],
            ...     val_losses=[(0, 2.6), (100, 1.3), (200, 0.9)],
            ...     smooth_window=5
            ... )
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract steps and losses
        if train_losses:
            train_steps, train_vals = zip(*train_losses)
            
            # Apply smoothing if requested
            if smooth_window > 1 and len(train_vals) > smooth_window:
                train_vals_smooth = self._moving_average(train_vals, smooth_window)
                ax.plot(train_steps, train_vals, alpha=0.3, color='blue', label='Train (raw)')
                ax.plot(train_steps, train_vals_smooth, color='blue', linewidth=2, label='Train (smoothed)')
            else:
                ax.plot(train_steps, train_vals, color='blue', linewidth=2, label='Train')
        
        # Plot validation losses
        if val_losses:
            val_steps, val_vals = zip(*val_losses)
            ax.plot(val_steps, val_vals, color='red', linewidth=2, marker='o', 
                   markersize=5, label='Validation')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        self.figures['loss_curves'] = fig
        return fig
    
    def plot_lr_schedule(
        self,
        lr_history: List[Tuple[int, float]],
        title: str = "Learning Rate Schedule"
    ) -> plt.Figure:
        """
        Plot learning rate schedule.
        
        Args:
            lr_history: List of (step, lr) tuples
            title: Plot title
        
        Returns:
            Matplotlib figure
        
        Example:
            >>> viz.plot_lr_schedule(
            ...     lr_history=[(0, 0.0), (100, 0.001), (1000, 0.0001)]
            ... )
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if lr_history:
            steps, lrs = zip(*lr_history)
            ax.plot(steps, lrs, color='green', linewidth=2)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for LR
        
        plt.tight_layout()
        
        self.figures['lr_schedule'] = fig
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics: Dict[str, List[Tuple[int, float]]],
        title: str = "Metrics Comparison",
        ylabel: str = "Value"
    ) -> plt.Figure:
        """
        Plot multiple metrics on same axes.
        
        Args:
            metrics: Dictionary of metric_name -> [(step, value), ...]
            title: Plot title
            ylabel: Y-axis label
        
        Returns:
            Matplotlib figure
        
        Example:
            >>> viz.plot_metrics_comparison({
            ...     'perplexity': [(0, 150), (100, 50), (200, 20)],
            ...     'accuracy': [(0, 0.3), (100, 0.6), (200, 0.8)]
            ... })
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        colors = plt.cm.tab10.colors  # Color palette
        
        for idx, (metric_name, history) in enumerate(metrics.items()):
            if history:
                steps, values = zip(*history)
                color = colors[idx % len(colors)]
                ax.plot(steps, values, color=color, linewidth=2, 
                       marker='o', markersize=4, label=metric_name)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        self.figures['metrics_comparison'] = fig
        return fig
    
    def plot_gradient_norm(
        self,
        grad_norm_history: List[Tuple[int, float]],
        title: str = "Gradient Norm"
    ) -> plt.Figure:
        """
        Plot gradient norm progression.
        
        Args:
            grad_norm_history: List of (step, grad_norm) tuples
            title: Plot title
        
        Returns:
            Matplotlib figure
        
        Example:
            >>> viz.plot_gradient_norm(
            ...     grad_norm_history=[(0, 5.0), (100, 2.0), (200, 1.0)]
            ... )
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if grad_norm_history:
            steps, norms = zip(*grad_norm_history)
            ax.plot(steps, norms, color='purple', linewidth=2)
            
            # Add threshold line at common clipping value
            if max(norms) > 1.0:
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, 
                          label='Common clip threshold')
                ax.legend()
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for gradient norm
        
        plt.tight_layout()
        
        self.figures['gradient_norm'] = fig
        return fig
    
    def plot_training_summary(
        self,
        train_losses: List[Tuple[int, float]],
        val_losses: Optional[List[Tuple[int, float]]],
        lr_history: List[Tuple[int, float]],
        grad_norm_history: List[Tuple[int, float]]
    ) -> plt.Figure:
        """
        Create comprehensive training summary with subplots.
        
        Args:
            train_losses: Training loss history
            val_losses: Validation loss history
            lr_history: Learning rate history
            grad_norm_history: Gradient norm history
        
        Returns:
            Matplotlib figure with 4 subplots
        
        Example:
            >>> viz.plot_training_summary(
            ...     train_losses=train_data,
            ...     val_losses=val_data,
            ...     lr_history=lr_data,
            ...     grad_norm_history=grad_data
            ... )
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        # Plot 1: Loss curves
        ax = axes[0, 0]
        if train_losses:
            steps, losses = zip(*train_losses)
            ax.plot(steps, losses, color='blue', linewidth=2, label='Train')
        if val_losses:
            steps, losses = zip(*val_losses)
            ax.plot(steps, losses, color='red', linewidth=2, marker='o', label='Val')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Learning rate
        ax = axes[0, 1]
        if lr_history:
            steps, lrs = zip(*lr_history)
            ax.plot(steps, lrs, color='green', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('LR Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Gradient norm
        ax = axes[1, 0]
        if grad_norm_history:
            steps, norms = zip(*grad_norm_history)
            ax.plot(steps, norms, color='purple', linewidth=2)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Loss histogram
        ax = axes[1, 1]
        if train_losses:
            _, losses = zip(*train_losses)
            ax.hist(losses, bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Loss Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Loss Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Training Summary', fontsize=16, y=0.995)
        plt.tight_layout()
        
        self.figures['training_summary'] = fig
        return fig
    
    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """
        Compute moving average.
        
        Args:
            data: List of values
            window: Window size
        
        Returns:
            Smoothed values
        """
        if len(data) < window:
            return list(data)
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            end = i + 1
            smoothed.append(sum(data[start:end]) / (end - start))
        
        return smoothed
    
    def save_figure(self, name: str, filename: Optional[str] = None):
        """
        Save a specific figure.
        
        Args:
            name: Figure name (key in self.figures)
            filename: Output filename (default: {name}.png)
        """
        if name not in self.figures:
            logger.warning(f"Figure '{name}' not found")
            return
        
        filename = filename or f"{name}.png"
        filepath = self.save_dir / filename
        
        self.figures[name].savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Figure saved: {filepath}")
    
    def save_all_plots(self):
        """Save all generated plots."""
        for name in self.figures.keys():
            self.save_figure(name)
        
        logger.info(f"All plots saved to {self.save_dir}")
    
    def close_all(self):
        """Close all figures to free memory."""
        for fig in self.figures.values():
            plt.close(fig)
        
        self.figures.clear()
        logger.info("All figures closed")


if __name__ == "__main__":
    # Test visualizer
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Training Visualizer Test")
    print("=" * 80)
    
    import tempfile
    import numpy as np
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create visualizer
        viz = TrainingVisualizer(save_dir=tmpdir)
        
        # Generate sample data
        steps = list(range(0, 100, 5))
        
        # Simulated training with decay
        train_losses = [(s, 2.0 * np.exp(-s/30) + 0.1) for s in steps]
        val_losses = [(s, 2.1 * np.exp(-s/30) + 0.15) for s in steps[::2]]  # Less frequent
        
        # LR schedule: warmup + decay
        lr_history = []
        for s in steps:
            if s < 20:
                lr = 0.001 * (s / 20)  # Warmup
            else:
                lr = 0.001 * (1 - (s - 20) / 100)  # Decay
            lr_history.append((s, max(lr, 0.0001)))
        
        # Gradient norms
        grad_norm_history = [(s, 3.0 * np.exp(-s/50) + 0.5) for s in steps]
        
        print("\n1. Plotting loss curves...")
        viz.plot_loss_curves(train_losses, val_losses, smooth_window=3)
        
        print("\n2. Plotting LR schedule...")
        viz.plot_lr_schedule(lr_history)
        
        print("\n3. Plotting gradient norm...")
        viz.plot_gradient_norm(grad_norm_history)
        
        print("\n4. Plotting metrics comparison...")
        perplexity = [(s, np.exp(loss)) for s, loss in train_losses]
        accuracy = [(s, 1 - loss/2) for s, loss in train_losses]
        viz.plot_metrics_comparison({
            'perplexity': perplexity,
            'accuracy': accuracy
        })
        
        print("\n5. Creating training summary...")
        viz.plot_training_summary(
            train_losses, val_losses,
            lr_history, grad_norm_history
        )
        
        print("\n6. Saving all plots...")
        viz.save_all_plots()
        
        print(f"\n✓ Generated {len(viz.figures)} plots:")
        for name in viz.figures.keys():
            filepath = Path(tmpdir) / f"{name}.png"
            size_kb = filepath.stat().st_size / 1024
            print(f"  - {name}.png: {size_kb:.1f} KB")
        
        viz.close_all()
    
    print("\n" + "=" * 80)
    print("✅ Visualizer tests completed!")
    print("=" * 80)
