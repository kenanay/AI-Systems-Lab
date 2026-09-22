"""
src/evaluation/__init__.py

Model evaluation, metrics and benchmarks module.
"""

from src.evaluation.metrics import (
    MetricResult,
    EvaluationResults,
    compute_loss,
    compute_perplexity,
    compute_accuracy,
    compute_top_k_accuracy,
    compute_entropy,
    evaluate_batch,
    compute_bleu,
    compute_rouge,
    evaluate_generation,
    aggregate_results,
)

from src.evaluation.benchmarks import (
    BenchmarkResult,
    BenchmarkRunner,
)

__all__ = [
    "MetricResult",
    "EvaluationResults",
    "compute_loss",
    "compute_perplexity",
    "compute_accuracy",
    "compute_top_k_accuracy",
    "compute_entropy",
    "evaluate_batch",
    "compute_bleu",
    "compute_rouge",
    "evaluate_generation",
    "aggregate_results",
    "BenchmarkResult",
    "BenchmarkRunner",
]
