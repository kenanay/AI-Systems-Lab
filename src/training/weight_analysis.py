"""
src/training/weight_analysis.py

Model Weight and Gradient Analysis

Bu modül model weights ve gradients'ları analiz eder:
- Parameter statistics (mean, std, min, max, norm)
- Gradient flow analysis (vanishing/exploding gradient detection)
- Dead neuron detection (neurons with zero gradients)
- Weight distribution analysis (histogram, outliers)
- Layer-wise analysis (which layers learn fast/slow)
- Health metrics (training stability indicators)

Weight analysis, model training sağlığını izlemek,
problematic layers'ı tespit etmek ve optimization
stratejisini ayarlamak için kritik bilgi sağlar.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParameterStats:
    """
    Statistics for a single parameter tensor.
    
    Attributes:
        name: Parameter name
        shape: Parameter shape
        mean: Mean value
        std: Standard deviation
        min: Minimum value
        max: Maximum value
        norm: L2 norm
        num_zeros: Number of zero elements
        num_elements: Total number of elements
    """
    name: str
    shape: Tuple[int, ...]
    mean: float
    std: float
    min: float
    max: float
    norm: float
    num_zeros: int
    num_elements: int
    
    @property
    def sparsity(self) -> float:
        """Sparsity ratio (0-1)."""
        return self.num_zeros / self.num_elements if self.num_elements > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'shape': list(self.shape),
            'mean': round(self.mean, 6),
            'std': round(self.std, 6),
            'min': round(self.min, 6),
            'max': round(self.max, 6),
            'norm': round(self.norm, 6),
            'sparsity': round(self.sparsity, 4),
            'num_elements': self.num_elements
        }


@dataclass
class GradientStats:
    """
    Gradient statistics for a parameter.
    
    Attributes:
        name: Parameter name
        grad_mean: Gradient mean
        grad_std: Gradient std
        grad_norm: Gradient L2 norm
        grad_min: Gradient min
        grad_max: Gradient max
        has_nan: Whether gradient contains NaN
        has_inf: Whether gradient contains Inf
        is_dead: Whether neuron is dead (zero gradient)
    """
    name: str
    grad_mean: float
    grad_std: float
    grad_norm: float
    grad_min: float
    grad_max: float
    has_nan: bool
    has_inf: bool
    is_dead: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'grad_mean': round(self.grad_mean, 8),
            'grad_std': round(self.grad_std, 8),
            'grad_norm': round(self.grad_norm, 6),
            'grad_min': round(self.grad_min, 8),
            'grad_max': round(self.grad_max, 8),
            'has_nan': self.has_nan,
            'has_inf': self.has_inf,
            'is_dead': self.is_dead
        }


@dataclass
class HealthMetrics:
    """
    Overall model health metrics.
    
    Attributes:
        total_params: Total parameter count
        total_dead_neurons: Count of dead neurons
        vanishing_gradient_layers: Layers with vanishing gradients
        exploding_gradient_layers: Layers with exploding gradients
        nan_gradient_layers: Layers with NaN gradients
        avg_grad_norm: Average gradient norm across layers
        max_grad_norm: Maximum gradient norm
        min_grad_norm: Minimum gradient norm
    """
    total_params: int
    total_dead_neurons: int
    vanishing_gradient_layers: List[str]
    exploding_gradient_layers: List[str]
    nan_gradient_layers: List[str]
    avg_grad_norm: float
    max_grad_norm: float
    min_grad_norm: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_params': self.total_params,
            'total_dead_neurons': self.total_dead_neurons,
            'vanishing_gradient_count': len(self.vanishing_gradient_layers),
            'exploding_gradient_count': len(self.exploding_gradient_layers),
            'nan_gradient_count': len(self.nan_gradient_layers),
            'vanishing_gradient_layers': self.vanishing_gradient_layers,
            'exploding_gradient_layers': self.exploding_gradient_layers,
            'nan_gradient_layers': self.nan_gradient_layers,
            'avg_grad_norm': round(self.avg_grad_norm, 6),
            'max_grad_norm': round(self.max_grad_norm, 6),
            'min_grad_norm': round(self.min_grad_norm, 6)
        }


class WeightAnalyzer:
    """
    Analyzer for model weights and gradients.
    
    Analyzes parameter statistics, gradient flow, and identifies
    training health issues like vanishing/exploding gradients
    and dead neurons.
    
    Features:
    - Parameter statistics (mean, std, norm, sparsity)
    - Gradient flow analysis
    - Dead neuron detection
    - Vanishing/exploding gradient detection
    - NaN/Inf detection
    - Layer-wise analysis
    - Health metrics summary
    
    Args:
        vanishing_threshold: Gradient norm below this is vanishing (default: 1e-7)
        exploding_threshold: Gradient norm above this is exploding (default: 100.0)
        dead_threshold: Gradient norm below this means dead (default: 1e-10)
    
    Example:
        >>> analyzer = WeightAnalyzer()
        >>> param_stats = analyzer.analyze_parameters(model)
        >>> grad_stats = analyzer.analyze_gradients(model)
        >>> health = analyzer.compute_health_metrics(grad_stats)
        >>> analyzer.print_summary(param_stats, grad_stats, health)
    """
    
    def __init__(
        self,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 100.0,
        dead_threshold: float = 1e-10
    ):
        """Initialize weight analyzer."""
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.dead_threshold = dead_threshold
        
        logger.info(f"WeightAnalyzer initialized (vanishing<{vanishing_threshold}, exploding>{exploding_threshold})")
    
    def analyze_parameters(self, model: nn.Module) -> List[ParameterStats]:
        """
        Analyze model parameters.
        
        Args:
            model: PyTorch model
        
        Returns:
            List of ParameterStats for each parameter
        
        Example:
            >>> stats = analyzer.analyze_parameters(model)
            >>> for stat in stats:
            ...     print(f"{stat.name}: norm={stat.norm:.4f}")
        """
        param_stats = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # param.data shape: [*param.shape]
            data = param.data
            
            stats = ParameterStats(
                name=name,
                shape=tuple(data.shape),
                mean=float(data.mean()),
                std=float(data.std()),
                min=float(data.min()),
                max=float(data.max()),
                norm=float(torch.norm(data)),  # L2 norm of flattened tensor
                num_zeros=int((data == 0).sum()),
                num_elements=int(data.numel())
            )
            
            param_stats.append(stats)
        
        logger.info(f"Analyzed {len(param_stats)} parameters")
        return param_stats
    
    def analyze_gradients(self, model: nn.Module) -> List[GradientStats]:
        """
        Analyze model gradients.
        
        Args:
            model: PyTorch model (after backward pass)
        
        Returns:
            List of GradientStats for each parameter with gradients
        
        Example:
            >>> loss.backward()
            >>> grad_stats = analyzer.analyze_gradients(model)
            >>> dead = [s for s in grad_stats if s.is_dead]
            >>> print(f"Dead neurons: {len(dead)}")
        """
        grad_stats = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            
            # param.grad shape: [*param.shape] (same as param.data)
            grad = param.grad
            
            # Check for NaN/Inf
            has_nan = bool(torch.isnan(grad).any())
            has_inf = bool(torch.isinf(grad).any())
            
            # Gradient norm
            grad_norm = float(torch.norm(grad))  # L2 norm of flattened tensor
            
            # Dead neuron detection
            is_dead = grad_norm < self.dead_threshold
            
            stats = GradientStats(
                name=name,
                grad_mean=float(grad.mean()),
                grad_std=float(grad.std()),
                grad_norm=grad_norm,
                grad_min=float(grad.min()),
                grad_max=float(grad.max()),
                has_nan=has_nan,
                has_inf=has_inf,
                is_dead=is_dead
            )
            
            grad_stats.append(stats)
        
        logger.info(f"Analyzed {len(grad_stats)} gradients")
        return grad_stats
    
    def compute_health_metrics(self, grad_stats: List[GradientStats]) -> HealthMetrics:
        """
        Compute overall model health metrics.
        
        Args:
            grad_stats: List of gradient statistics
        
        Returns:
            HealthMetrics with health indicators
        
        Example:
            >>> health = analyzer.compute_health_metrics(grad_stats)
            >>> if health.total_dead_neurons > 0:
            ...     print(f"Warning: {health.total_dead_neurons} dead neurons")
        """
        vanishing_layers = []
        exploding_layers = []
        nan_layers = []
        dead_count = 0
        grad_norms = []
        
        for stats in grad_stats:
            # Check for issues
            if stats.has_nan:
                nan_layers.append(stats.name)
            
            if stats.is_dead:
                dead_count += 1
            
            if stats.grad_norm < self.vanishing_threshold:
                vanishing_layers.append(stats.name)
            
            if stats.grad_norm > self.exploding_threshold:
                exploding_layers.append(stats.name)
            
            grad_norms.append(stats.grad_norm)
        
        # Aggregate metrics
        total_params = sum(1 for _ in grad_stats)
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        max_grad_norm = max(grad_norms) if grad_norms else 0.0
        min_grad_norm = min(grad_norms) if grad_norms else 0.0
        
        health = HealthMetrics(
            total_params=total_params,
            total_dead_neurons=dead_count,
            vanishing_gradient_layers=vanishing_layers,
            exploding_gradient_layers=exploding_layers,
            nan_gradient_layers=nan_layers,
            avg_grad_norm=avg_grad_norm,
            max_grad_norm=max_grad_norm,
            min_grad_norm=min_grad_norm
        )
        
        logger.info(f"Health metrics: {dead_count} dead, {len(vanishing_layers)} vanishing, {len(exploding_layers)} exploding")
        return health
    
    def print_summary(
        self,
        param_stats: List[ParameterStats],
        grad_stats: List[GradientStats],
        health: HealthMetrics
    ):
        """
        Print formatted analysis summary.
        
        Args:
            param_stats: Parameter statistics
            grad_stats: Gradient statistics
            health: Health metrics
        """
        print("\n" + "=" * 70)
        print("Weight Analysis Summary")
        print("=" * 70)
        
        # Parameters
        print(f"\nParameters ({len(param_stats)} layers):")
        total_params = sum(s.num_elements for s in param_stats)
        print(f"  Total parameters: {total_params:,}")
        
        if param_stats:
            avg_norm = sum(s.norm for s in param_stats) / len(param_stats)
            avg_sparsity = sum(s.sparsity for s in param_stats) / len(param_stats)
            print(f"  Average norm: {avg_norm:.4f}")
            print(f"  Average sparsity: {avg_sparsity:.4f}")
        
        # Gradients
        print(f"\nGradients ({len(grad_stats)} layers):")
        print(f"  Average grad norm: {health.avg_grad_norm:.6f}")
        print(f"  Min grad norm: {health.min_grad_norm:.6f}")
        print(f"  Max grad norm: {health.max_grad_norm:.6f}")
        
        # Health issues
        print(f"\nHealth Metrics:")
        print(f"  Dead neurons: {health.total_dead_neurons}")
        print(f"  Vanishing gradients: {len(health.vanishing_gradient_layers)}")
        print(f"  Exploding gradients: {len(health.exploding_gradient_layers)}")
        print(f"  NaN gradients: {len(health.nan_gradient_layers)}")
        
        # Warnings
        if health.total_dead_neurons > 0:
            print(f"\n⚠️  WARNING: {health.total_dead_neurons} dead neurons detected")
        
        if health.vanishing_gradient_layers:
            print(f"\n⚠️  WARNING: Vanishing gradients in {len(health.vanishing_gradient_layers)} layers:")
            for layer in health.vanishing_gradient_layers[:5]:  # Show first 5
                print(f"     - {layer}")
        
        if health.exploding_gradient_layers:
            print(f"\n⚠️  WARNING: Exploding gradients in {len(health.exploding_gradient_layers)} layers:")
            for layer in health.exploding_gradient_layers[:5]:
                print(f"     - {layer}")
        
        if health.nan_gradient_layers:
            print(f"\n🚨 CRITICAL: NaN gradients in {len(health.nan_gradient_layers)} layers:")
            for layer in health.nan_gradient_layers:
                print(f"     - {layer}")
        
        print("=" * 70 + "\n")
    
    def save_analysis(
        self,
        param_stats: List[ParameterStats],
        grad_stats: List[GradientStats],
        health: HealthMetrics,
        filepath: Path
    ):
        """
        Save analysis to JSON file.
        
        Args:
            param_stats: Parameter statistics
            grad_stats: Gradient statistics
            health: Health metrics
            filepath: Output file path
        """
        data = {
            'parameters': [s.to_dict() for s in param_stats],
            'gradients': [s.to_dict() for s in grad_stats],
            'health': health.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Analysis saved to {filepath}")


if __name__ == "__main__":
    # Test weight analyzer
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Weight Analyzer Test")
    print("=" * 80)
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 5)
        
        def forward(self, x):
            # x shape: [batch, 10]
            x = torch.relu(self.fc1(x))  # [batch, 20]
            x = torch.relu(self.fc2(x))  # [batch, 10]
            x = self.fc3(x)  # [batch, 5]
            return x
    
    model = TestModel()
    
    print("\n1. Analyzing parameters...")
    analyzer = WeightAnalyzer()
    param_stats = analyzer.analyze_parameters(model)
    
    print(f"\n✓ Found {len(param_stats)} parameter tensors:")
    for stat in param_stats:
        print(f"  - {stat.name}: shape={stat.shape}, norm={stat.norm:.4f}, sparsity={stat.sparsity:.4f}")
    
    print("\n2. Running forward + backward pass...")
    # x shape: [batch=4, input=10]
    x = torch.randn(4, 10)
    # output shape: [batch=4, output=5]
    output = model(x)
    # target shape: [batch=4, output=5]
    target = torch.randn(4, 5)
    # loss shape: scalar
    loss = ((output - target) ** 2).mean()
    loss.backward()
    
    print("\n3. Analyzing gradients...")
    grad_stats = analyzer.analyze_gradients(model)
    
    print(f"\n✓ Found {len(grad_stats)} gradient tensors:")
    for stat in grad_stats:
        status = "DEAD" if stat.is_dead else "OK"
        print(f"  - {stat.name}: norm={stat.grad_norm:.6f} [{status}]")
    
    print("\n4. Computing health metrics...")
    health = analyzer.compute_health_metrics(grad_stats)
    
    print("\n5. Final summary...")
    analyzer.print_summary(param_stats, grad_stats, health)
    
    # Test edge cases
    print("\n6. Testing edge cases...")
    
    # Create model with dead neurons (zero gradients)
    print("  - Testing dead neuron detection...")
    model.fc1.weight.grad.zero_()  # Kill gradients
    grad_stats_dead = analyzer.analyze_gradients(model)
    health_dead = analyzer.compute_health_metrics(grad_stats_dead)
    assert health_dead.total_dead_neurons > 0, "Dead neuron detection failed"
    print(f"    ✓ Detected {health_dead.total_dead_neurons} dead neurons")
    
    # Test NaN detection
    print("  - Testing NaN detection...")
    model.fc2.weight.grad[0, 0] = float('nan')
    grad_stats_nan = analyzer.analyze_gradients(model)
    assert any(s.has_nan for s in grad_stats_nan), "NaN detection failed"
    print("    ✓ NaN detection working")
    
    print("\n" + "=" * 80)
    print("✅ Weight Analyzer tests completed!")
    print("=" * 80)
