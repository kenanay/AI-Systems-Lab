"""
src/evaluation/metrics.py

Core Evaluation Metrics for Language Models

Language model evaluation için temel metrikler:
- Perplexity: Model'in prediction kalitesi
- Accuracy: Token-level doğruluk
- Loss: Cross-entropy loss hesaplama
- Top-K Accuracy: Top-k predictions içinde doğru token
- Confidence: Prediction confidence scores
- Entropy: Prediction entropy (model certainty)

Bu modül production-ready metric computation sağlar:
- Batch processing: Verimli batch-wise hesaplama
- Numerical stability: Log-space hesaplamalar
- Multiple metrics: Comprehensive evaluation
- Aggregation: Batch results aggregation
- Statistical analysis: Mean, std, confidence intervals

Formüller:
    Perplexity = exp(cross_entropy_loss)
    Accuracy = correct_predictions / total_predictions
    Top-K Accuracy = predictions_in_top_k / total_predictions
    Entropy = -sum(p * log(p))

Kaynaklar:
    - Language Modeling Evaluation Metrics
    - Perplexity and Cross-Entropy
    - https://huggingface.co/docs/transformers/perplexity

Usage:
    >>> from src.evaluation.metrics import compute_perplexity, compute_accuracy
    >>> 
    >>> # Compute perplexity
    >>> perplexity = compute_perplexity(logits, targets)
    >>> 
    >>> # Compute accuracy
    >>> accuracy = compute_accuracy(predictions, targets)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# Metric Results
# =========================

@dataclass
class MetricResult:
    """
    Single metric result.
    
    Attributes:
        name: Metric name
        value: Metric value
        unit: Unit (e.g., '%', 'bits')
        higher_is_better: True if higher values are better
        metadata: Additional metadata
    """
    name: str
    value: float
    unit: str = ""
    higher_is_better: bool = True
    metadata: Dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation."""
        direction = "↑" if self.higher_is_better else "↓"
        return f"{self.name}: {self.value:.4f}{self.unit} {direction}"


@dataclass
class EvaluationResults:
    """
    Complete evaluation results.
    
    Attributes:
        metrics: List of metric results
        num_samples: Number of samples evaluated
        num_tokens: Number of tokens evaluated
        metadata: Additional metadata
    """
    metrics: List[MetricResult] = field(default_factory=list)
    num_samples: int = 0
    num_tokens: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def add_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        higher_is_better: bool = True,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add a metric result."""
        self.metrics.append(MetricResult(
            name=name,
            value=value,
            unit=unit,
            higher_is_better=higher_is_better,
            metadata=metadata or {}
        ))
    
    def get_metric(self, name: str) -> Optional[MetricResult]:
        """Get metric by name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "metrics": {
                m.name: {
                    "value": m.value,
                    "unit": m.unit,
                    "higher_is_better": m.higher_is_better,
                    "metadata": m.metadata
                }
                for m in self.metrics
            },
            "num_samples": self.num_samples,
            "num_tokens": self.num_tokens,
            "metadata": self.metadata
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {path}")
    
    def __str__(self) -> str:
        """String representation."""
        lines = ["Evaluation Results:"]
        lines.append(f"  Samples: {self.num_samples:,}")
        lines.append(f"  Tokens: {self.num_tokens:,}")
        lines.append("  Metrics:")
        for metric in self.metrics:
            lines.append(f"    {metric}")
        return "\n".join(lines)


# =========================
# Core Metrics
# =========================

def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute perplexity from logits and targets.
    
    Perplexity = exp(cross_entropy_loss)
    
    Lower perplexity = better model (daha iyi tahmin)
    
    Args:
        logits: Model predictions, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        ignore_index: Index to ignore (e.g., padding)
        
    Returns:
        float: Perplexity value
        
    Example:
        >>> logits = torch.randn(2, 10, 1000)  # [B, T, V]
        >>> targets = torch.randint(0, 1000, (2, 10))  # [B, T]
        >>> perplexity = compute_perplexity(logits, targets)
    """
    # logits shape: [B, T, V]
    # targets shape: [B, T]
    
    # Flatten for loss computation
    # logits_flat shape: [B*T, V]
    logits_flat = logits.reshape(-1, logits.size(-1))
    
    # targets_flat shape: [B*T]
    targets_flat = targets.reshape(-1)
    
    # Compute cross-entropy loss
    # loss is scalar
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()
    
    return perplexity


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute token-level accuracy.
    
    Accuracy = correct_predictions / total_predictions
    
    Args:
        predictions: Predicted token IDs, shape [batch, seq_len]
        targets: Target token IDs, shape [batch, seq_len]
        ignore_index: Index to ignore (e.g., padding)
        
    Returns:
        float: Accuracy (0.0 to 1.0)
    """
    # predictions shape: [B, T]
    # targets shape: [B, T]
    
    # Create mask for valid positions
    # mask shape: [B, T]
    mask = (targets != ignore_index)
    
    # Check correct predictions
    # correct shape: [B, T]
    correct = (predictions == targets) & mask
    
    # Calculate accuracy
    accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
    
    return accuracy


def compute_top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
    ignore_index: int = -100
) -> float:
    """
    Compute top-k accuracy.
    
    Top-K Accuracy = predictions_in_top_k / total_predictions
    
    Args:
        logits: Model predictions, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        k: Top-k value
        ignore_index: Index to ignore
        
    Returns:
        float: Top-k accuracy (0.0 to 1.0)
    """
    # logits shape: [B, T, V]
    # targets shape: [B, T]
    
    # Get top-k predictions
    # top_k_preds shape: [B, T, k]
    _, top_k_preds = torch.topk(logits, k, dim=-1)
    
    # Create mask for valid positions
    # mask shape: [B, T]
    mask = (targets != ignore_index)
    
    # Expand targets for comparison
    # targets_expanded shape: [B, T, 1]
    targets_expanded = targets.unsqueeze(-1)
    
    # Check if target is in top-k
    # in_top_k shape: [B, T, k]
    in_top_k = (top_k_preds == targets_expanded)
    
    # Any match in top-k?
    # matches shape: [B, T]
    matches = in_top_k.any(dim=-1) & mask
    
    # Calculate accuracy
    accuracy = matches.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
    
    return accuracy


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Model predictions, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        ignore_index: Index to ignore
        
    Returns:
        float: Loss value
    """
    # logits shape: [B, T, V]
    # targets shape: [B, T]
    
    # Flatten
    logits_flat = logits.reshape(-1, logits.size(-1))  # [B*T, V]
    targets_flat = targets.reshape(-1)  # [B*T]
    
    # Compute loss
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    return loss.item()


def compute_confidence_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Compute confidence-related metrics.
    
    Returns:
        Dict with:
            - mean_confidence: Average confidence for correct predictions
            - mean_max_prob: Average max probability
            - calibration_error: Simple calibration error
    
    Args:
        logits: Model predictions, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        ignore_index: Index to ignore
    """
    # logits shape: [B, T, V]
    # targets shape: [B, T]
    
    # Get probabilities
    # probs shape: [B, T, V]
    probs = F.softmax(logits, dim=-1)
    
    # Get max probabilities
    # max_probs shape: [B, T]
    max_probs, predictions = torch.max(probs, dim=-1)
    
    # Create mask
    mask = (targets != ignore_index)
    
    # Get target probabilities (confidence for correct class)
    # target_probs shape: [B, T]
    target_probs = torch.gather(probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    
    # Calculate metrics on valid positions
    valid_max_probs = max_probs[mask]
    valid_target_probs = target_probs[mask]
    valid_correct = (predictions == targets)[mask]
    
    if mask.sum().item() == 0:
        return {
            "mean_confidence": 0.0,
            "mean_max_prob": 0.0,
            "calibration_error": 0.0
        }
    
    # Mean confidence for correct predictions
    mean_confidence = valid_target_probs.mean().item()
    
    # Mean max probability
    mean_max_prob = valid_max_probs.mean().item()
    
    # Simple calibration error (difference between confidence and accuracy)
    accuracy = valid_correct.float().mean().item()
    calibration_error = abs(mean_max_prob - accuracy)
    
    return {
        "mean_confidence": mean_confidence,
        "mean_max_prob": mean_max_prob,
        "calibration_error": calibration_error
    }


def compute_entropy(
    logits: torch.Tensor,
    ignore_index: int = -100,
    targets: Optional[torch.Tensor] = None
) -> float:
    """
    Compute average prediction entropy.
    
    Entropy = -sum(p * log(p))
    
    Lower entropy = more confident predictions
    
    Args:
        logits: Model predictions, shape [batch, seq_len, vocab_size]
        ignore_index: Index to ignore
        targets: Optional targets for masking, shape [batch, seq_len]
        
    Returns:
        float: Average entropy
    """
    # logits shape: [B, T, V]
    
    # Get probabilities
    # probs shape: [B, T, V]
    probs = F.softmax(logits, dim=-1)
    
    # Compute entropy: -sum(p * log(p))
    # entropy shape: [B, T]
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    
    if targets is not None:
        # Mask invalid positions
        mask = (targets != ignore_index)
        entropy = entropy[mask]
    
    # Average entropy
    avg_entropy = entropy.mean().item()
    
    return avg_entropy


# =========================
# Batch Evaluation
# =========================

def evaluate_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    compute_all: bool = True
) -> EvaluationResults:
    """
    Evaluate a single batch with all metrics.
    
    Args:
        logits: Model predictions, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        ignore_index: Index to ignore
        compute_all: Compute all metrics (slower but comprehensive)
        
    Returns:
        EvaluationResults with all metrics
    """
    results = EvaluationResults()
    
    # Count samples and tokens
    results.num_samples = targets.size(0)
    
    mask = (targets != ignore_index)
    results.num_tokens = mask.sum().item()
    
    # Get predictions
    # predictions shape: [B, T]
    predictions = torch.argmax(logits, dim=-1)
    
    # Core metrics
    loss = compute_loss(logits, targets, ignore_index)
    perplexity = compute_perplexity(logits, targets, ignore_index)
    accuracy = compute_accuracy(predictions, targets, ignore_index)
    
    results.add_metric("loss", loss, "", higher_is_better=False)
    results.add_metric("perplexity", perplexity, "", higher_is_better=False)
    results.add_metric("accuracy", accuracy, "%", higher_is_better=True)
    
    if compute_all:
        # Top-k accuracy
        top5_acc = compute_top_k_accuracy(logits, targets, k=5, ignore_index=ignore_index)
        top10_acc = compute_top_k_accuracy(logits, targets, k=10, ignore_index=ignore_index)
        
        results.add_metric("top5_accuracy", top5_acc, "%", higher_is_better=True)
        results.add_metric("top10_accuracy", top10_acc, "%", higher_is_better=True)
        
        # Confidence scores
        confidence_scores = compute_confidence_scores(logits, targets, ignore_index)
        results.add_metric("mean_confidence", confidence_scores["mean_confidence"], "", higher_is_better=True)
        results.add_metric("mean_max_prob", confidence_scores["mean_max_prob"], "", higher_is_better=True)
        results.add_metric("calibration_error", confidence_scores["calibration_error"], "", higher_is_better=False)
        
        # Entropy
        entropy = compute_entropy(logits, ignore_index, targets)
        results.add_metric("entropy", entropy, "bits", higher_is_better=False)
    
    return results


def aggregate_results(results_list: List[EvaluationResults]) -> EvaluationResults:
    """
    Aggregate multiple evaluation results.
    
    Args:
        results_list: List of EvaluationResults
        
    Returns:
        Aggregated EvaluationResults (weighted by num_tokens)
    """
    if not results_list:
        return EvaluationResults()
    
    # Collect all metric names
    metric_names = set()
    for results in results_list:
        for metric in results.metrics:
            metric_names.add(metric.name)
    
    # Aggregate each metric (weighted by num_tokens)
    aggregated = EvaluationResults()
    
    total_samples = sum(r.num_samples for r in results_list)
    total_tokens = sum(r.num_tokens for r in results_list)
    
    aggregated.num_samples = total_samples
    aggregated.num_tokens = total_tokens
    
    for metric_name in metric_names:
        # Collect values and weights
        values = []
        weights = []
        
        for results in results_list:
            metric = results.get_metric(metric_name)
            if metric is not None:
                values.append(metric.value)
                weights.append(results.num_tokens)
        
        if values:
            # Weighted average
            weighted_value = np.average(values, weights=weights)
            
            # Get metadata from first occurrence
            first_metric = next(
                r.get_metric(metric_name)
                for r in results_list
                if r.get_metric(metric_name) is not None
            )
            
            aggregated.add_metric(
                metric_name,
                weighted_value,
                first_metric.unit,
                first_metric.higher_is_better,
                {"std": float(np.std(values))}
            )
    
    return aggregated


# =========================
# Main (for testing)
# =========================

def main():
    """Test metrics computation."""
    print("\n" + "=" * 80)
    print("METRICS COMPUTATION TEST")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 4
    seq_len = 10
    vocab_size = 100
    
    print(f"\nTest Data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocab size: {vocab_size}")
    
    # Random logits and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("\n" + "-" * 80)
    print("Computing Metrics...")
    print("-" * 80)
    
    # Evaluate batch
    results = evaluate_batch(logits, targets, compute_all=True)
    
    print("\n" + str(results))
    
    # Save results
    results.save("evaluation_results_test.json")
    
    print("\n" + "=" * 80)
    print("✓ Metrics computation test completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
