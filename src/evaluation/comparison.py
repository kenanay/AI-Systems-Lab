"""
src/evaluation/comparison.py

Model Comparison Framework

Multiple model karşılaştırma sistemi:
- Side-by-side Evaluation: Birden fazla model paralel değerlendirme
- Metric Aggregation: Tüm metrikler üzerinde karşılaştırma
- Statistical Significance: İstatistiksel anlamlılık testleri
- Comparison Reports: Detaylı karşılaştırma raporları
- Win/Loss Analysis: Hangi model hangi metrikte daha iyi

Bu modül production-ready model comparison sağlar:
- Multiple models: N modeli paralel evaluate etme
- Statistical tests: T-test, bootstrap CI, effect size
- Ranking: Model ranking by metrics
- Reporting: JSON, Markdown, HTML export
- Visualization: Comparison charts (future)

Statistical Tests:
    T-Test:
        - Paired t-test for metric differences
        - H0: No difference between models
        - p < 0.05: Statistically significant difference
    
    Bootstrap Confidence Intervals:
        - Resampling-based CI estimation
        - 95% CI for metric differences
        - Robust to distribution assumptions
    
    Effect Size (Cohen's d):
        - d = (mean1 - mean2) / pooled_std
        - |d| < 0.2: Small effect
        - |d| < 0.5: Medium effect
        - |d| >= 0.8: Large effect

Kaynaklar:
    - Statistical Significance Testing for NLP
    - Bootstrap Methods for Standard Errors and Confidence Intervals
    - https://aclanthology.org/

Usage:
    >>> from src.evaluation.comparison import ModelComparison
    >>> 
    >>> # Create comparison
    >>> comparison = ModelComparison()
    >>> comparison.add_model("model-A", results_A)
    >>> comparison.add_model("model-B", results_B)
    >>> 
    >>> # Compare models
    >>> report = comparison.compare()
    >>> print(report)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
from scipy import stats
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
# Statistical Tests
# =========================

def paired_t_test(
    scores1: List[float],
    scores2: List[float]
) -> Tuple[float, float]:
    """
    Paired t-test for two sets of scores.
    
    H0: No difference between the two models
    H1: There is a difference
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        
    Returns:
        Tuple[float, float]: (t_statistic, p_value)
    """
    if len(scores1) != len(scores2):
        raise ValueError("Score lists must have same length")
    
    if len(scores1) < 2:
        return 0.0, 1.0
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    return float(t_stat), float(p_value)


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for mean.
    
    Args:
        scores: List of scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Tuple[float, float, float]: (mean, lower_bound, upper_bound)
    """
    scores_array = np.array(scores)
    n = len(scores_array)
    
    if n < 2:
        mean = scores_array[0] if n == 1 else 0.0
        return mean, mean, mean
    
    # Bootstrap resampling
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(scores_array, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean = np.mean(scores_array)
    lower = np.percentile(bootstrap_means, lower_percentile)
    upper = np.percentile(bootstrap_means, upper_percentile)
    
    return float(mean), float(lower), float(upper)


def cohen_d(
    scores1: List[float],
    scores2: List[float]
) -> float:
    """
    Compute Cohen's d effect size.
    
    Effect size interpretation:
        |d| < 0.2: Small
        |d| < 0.5: Medium
        |d| >= 0.8: Large
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        
    Returns:
        float: Cohen's d
    """
    scores1_array = np.array(scores1)
    scores2_array = np.array(scores2)
    
    n1, n2 = len(scores1_array), len(scores2_array)
    
    if n1 < 2 or n2 < 2:
        return 0.0
    
    # Means
    mean1 = np.mean(scores1_array)
    mean2 = np.mean(scores2_array)
    
    # Standard deviations
    std1 = np.std(scores1_array, ddof=1)
    std2 = np.std(scores2_array, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    return float(d)


# =========================
# Comparison Results
# =========================

@dataclass
class MetricComparison:
    """
    Comparison for a single metric.
    
    Attributes:
        metric_name: Name of metric
        model_scores: Dict[model_name, List[scores]]
        winner: Model with best score
        statistical_significance: Dict of pairwise p-values
    """
    metric_name: str
    model_scores: Dict[str, List[float]] = field(default_factory=dict)
    winner: Optional[str] = None
    statistical_significance: Dict[Tuple[str, str], float] = field(default_factory=dict)
    effect_sizes: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    def add_score(self, model_name: str, score: float) -> None:
        """Add a score for a model."""
        if model_name not in self.model_scores:
            self.model_scores[model_name] = []
        self.model_scores[model_name].append(score)
    
    def compute_statistics(self, higher_is_better: bool = True) -> None:
        """
        Compute statistical tests.
        
        Args:
            higher_is_better: True if higher scores are better
        """
        if len(self.model_scores) < 2:
            return
        
        # Determine winner (by mean score)
        mean_scores = {
            model: np.mean(scores)
            for model, scores in self.model_scores.items()
        }
        
        if higher_is_better:
            self.winner = max(mean_scores, key=lambda k: float(mean_scores[k]))
        else:
            self.winner = min(mean_scores, key=lambda k: float(mean_scores[k]))
        
        # Pairwise comparisons
        model_names = list(self.model_scores.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                scores1 = self.model_scores[model1]
                scores2 = self.model_scores[model2]
                
                # T-test
                if len(scores1) >= 2 and len(scores2) >= 2:
                    try:
                        _, p_value = paired_t_test(scores1, scores2)
                        self.statistical_significance[(model1, model2)] = p_value
                        
                        # Effect size
                        d = cohen_d(scores1, scores2)
                        self.effect_sizes[(model1, model2)] = d
                    except Exception as e:
                        logger.warning(f"Statistical test failed for {model1} vs {model2}: {e}")
    
    def get_summary(self) -> Dict:
        """Get comparison summary."""
        summary = {
            "metric": self.metric_name,
            "winner": self.winner,
            "mean_scores": {
                model: float(np.mean(scores))
                for model, scores in self.model_scores.items()
            },
            "std_scores": {
                model: float(np.std(scores))
                for model, scores in self.model_scores.items()
            }
        }
        
        # Add significance
        if self.statistical_significance:
            summary["pairwise_significance"] = {
                f"{m1}_vs_{m2}": {
                    "p_value": p,
                    "significant": p < 0.05,
                    "effect_size": self.effect_sizes.get((m1, m2), 0.0)
                }
                for (m1, m2), p in self.statistical_significance.items()
            }
        
        return summary


@dataclass
class ComparisonReport:
    """
    Complete comparison report.
    
    Attributes:
        model_names: List of model names
        metric_comparisons: Dict[metric_name, MetricComparison]
        overall_winner: Overall best model
        metadata: Report metadata
    """
    model_names: List[str] = field(default_factory=list)
    metric_comparisons: Dict[str, MetricComparison] = field(default_factory=dict)
    overall_winner: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def compute_overall_winner(self) -> None:
        """Compute overall winner based on win counts."""
        if not self.metric_comparisons:
            return
        
        # Count wins per model
        win_counts = {model: 0 for model in self.model_names}
        
        for comparison in self.metric_comparisons.values():
            if comparison.winner:
                win_counts[comparison.winner] += 1
        
        # Model with most wins
        self.overall_winner = max(win_counts, key=lambda k: win_counts[k]) if win_counts else None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_names": self.model_names,
            "overall_winner": self.overall_winner,
            "metric_comparisons": {
                name: comp.get_summary()
                for name, comp in self.metric_comparisons.items()
            },
            "metadata": self.metadata
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save report to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison report saved: {path}")
    
    def to_markdown(self) -> str:
        """Generate Markdown report."""
        lines = ["# Model Comparison Report\n"]
        
        lines.append(f"## Models: {', '.join(self.model_names)}\n")
        lines.append(f"**Overall Winner**: {self.overall_winner}\n")
        
        lines.append("\n## Metric Comparisons\n")
        
        for metric_name, comparison in self.metric_comparisons.items():
            lines.append(f"\n### {metric_name}\n")
            lines.append(f"**Winner**: {comparison.winner}\n")
            
            # Table
            lines.append("\n| Model | Mean | Std |")
            lines.append("|-------|------|-----|")
            
            for model in self.model_names:
                if model in comparison.model_scores:
                    mean = np.mean(comparison.model_scores[model])
                    std = np.std(comparison.model_scores[model])
                    lines.append(f"| {model} | {mean:.4f} | {std:.4f} |")
            
            # Statistical significance
            if comparison.statistical_significance:
                lines.append("\n**Statistical Significance**:\n")
                for (m1, m2), p in comparison.statistical_significance.items():
                    sig = "✓" if p < 0.05 else "✗"
                    d = comparison.effect_sizes.get((m1, m2), 0.0)
                    lines.append(f"- {m1} vs {m2}: p={p:.4f} {sig}, d={d:.2f}")
        
        return "\n".join(lines)


# =========================
# Model Comparison
# =========================

class ModelComparison:
    """
    Model comparison framework.
    
    Compare multiple models across metrics.
    """
    
    def __init__(self):
        """Initialize comparison."""
        self.models: Dict[str, EvaluationResults] = {}
        logger.info("ModelComparison initialized")
    
    def add_model(
        self,
        model_name: str,
        results: EvaluationResults
    ) -> None:
        """
        Add a model's evaluation results.
        
        Args:
            model_name: Name of the model
            results: Evaluation results
        """
        self.models[model_name] = results
        logger.info(f"Added model: {model_name} ({len(results.metrics)} metrics)")
    
    def compare(self) -> ComparisonReport:
        """
        Compare all models.
        
        Returns:
            ComparisonReport with detailed comparison
        """
        if len(self.models) < 2:
            raise ValueError("Need at least 2 models to compare")
        
        logger.info(f"Comparing {len(self.models)} models...")
        
        report = ComparisonReport(
            model_names=list(self.models.keys()),
            metadata={
                "num_models": len(self.models)
            }
        )
        
        # Collect all metric names
        all_metrics = set()
        for results in self.models.values():
            for metric in results.metrics:
                all_metrics.add(metric.name)
        
        # Compare each metric
        for metric_name in all_metrics:
            comparison = MetricComparison(metric_name=metric_name)
            
            # Collect scores from each model
            higher_is_better = True
            
            for model_name, results in self.models.items():
                metric = results.get_metric(metric_name)
                if metric is not None:
                    comparison.add_score(model_name, metric.value)
                    higher_is_better = metric.higher_is_better
            
            # Compute statistics
            comparison.compute_statistics(higher_is_better)
            
            report.metric_comparisons[metric_name] = comparison
        
        # Compute overall winner
        report.compute_overall_winner()
        
        logger.info(f"✓ Comparison complete. Winner: {report.overall_winner}")
        
        return report
    
    def compare_pair(
        self,
        model1_name: str,
        model2_name: str
    ) -> Dict:
        """
        Compare two models in detail.
        
        Args:
            model1_name: First model name
            model2_name: Second model name
            
        Returns:
            Dict with detailed pairwise comparison
        """
        if model1_name not in self.models or model2_name not in self.models:
            raise ValueError(f"Models not found: {model1_name}, {model2_name}")
        
        results1 = self.models[model1_name]
        results2 = self.models[model2_name]
        
        comparison = {
            "model1": model1_name,
            "model2": model2_name,
            "metrics": {}
        }
        
        # Compare each metric
        for metric1 in results1.metrics:
            metric2 = results2.get_metric(metric1.name)
            
            if metric2 is not None:
                # Difference
                diff = metric1.value - metric2.value
                
                # Percent difference
                if metric2.value != 0:
                    pct_diff = (diff / abs(metric2.value)) * 100
                else:
                    pct_diff = 0.0
                
                # Winner
                if metric1.higher_is_better:
                    winner = model1_name if diff > 0 else model2_name
                else:
                    winner = model1_name if diff < 0 else model2_name
                
                comparison["metrics"][metric1.name] = {
                    "model1_value": metric1.value,
                    "model2_value": metric2.value,
                    "difference": diff,
                    "percent_difference": pct_diff,
                    "winner": winner
                }
        
        return comparison


# =========================
# Main (for testing)
# =========================

def main():
    """Test model comparison."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TEST")
    print("=" * 80)
    
    # Create mock evaluation results for 3 models
    from src.evaluation.metrics import EvaluationResults
    
    # Model A (best)
    results_A = EvaluationResults()
    results_A.add_metric("accuracy", 0.85, "%", higher_is_better=True)
    results_A.add_metric("perplexity", 45.2, "", higher_is_better=False)
    results_A.add_metric("bleu-4", 68.5, "", higher_is_better=True)
    
    # Model B (medium)
    results_B = EvaluationResults()
    results_B.add_metric("accuracy", 0.78, "%", higher_is_better=True)
    results_B.add_metric("perplexity", 52.1, "", higher_is_better=False)
    results_B.add_metric("bleu-4", 61.3, "", higher_is_better=True)
    
    # Model C (worst)
    results_C = EvaluationResults()
    results_C.add_metric("accuracy", 0.72, "%", higher_is_better=True)
    results_C.add_metric("perplexity", 58.9, "", higher_is_better=False)
    results_C.add_metric("bleu-4", 55.7, "", higher_is_better=True)
    
    # Create comparison
    comparison = ModelComparison()
    comparison.add_model("Model-A", results_A)
    comparison.add_model("Model-B", results_B)
    comparison.add_model("Model-C", results_C)
    
    # Compare
    print("\n" + "-" * 80)
    print("Comparing Models...")
    print("-" * 80)
    
    report = comparison.compare()
    
    print(f"\nOverall Winner: {report.overall_winner}")
    
    print("\nMetric Winners:")
    for metric_name, comp in report.metric_comparisons.items():
        mean_scores = {
            model: np.mean(scores)
            for model, scores in comp.model_scores.items()
        }
        winner_str = str(comp.winner)
        winner_score = f"{mean_scores[comp.winner]:.4f}" if comp.winner and comp.winner in mean_scores else "N/A"
        print(f"  {metric_name}: {winner_str} ({winner_score})")
    
    # Pairwise comparison
    print("\n" + "-" * 80)
    print("Pairwise Comparison: Model-A vs Model-B")
    print("-" * 80)
    
    pair_comp = comparison.compare_pair("Model-A", "Model-B")
    
    for metric_name, data in pair_comp["metrics"].items():
        print(f"\n{metric_name}:")
        print(f"  Model-A: {data['model1_value']:.4f}")
        print(f"  Model-B: {data['model2_value']:.4f}")
        print(f"  Difference: {data['difference']:.4f} ({data['percent_difference']:.2f}%)")
        print(f"  Winner: {data['winner']}")
    
    # Save report
    report.save("comparison_report_test.json")
    
    # Generate Markdown
    print("\n" + "-" * 80)
    print("Markdown Report")
    print("-" * 80)
    print(report.to_markdown())
    
    print("\n" + "=" * 80)
    print("✓ Model comparison test completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
