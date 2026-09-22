"""Simulators module for AI Systems Lab."""

from src.simulators.tensor_math import (
    TensorShapeAnalyzer,
    BroadcastingEngine,
    MatMulVisualizer,
    ActivationSimulator,
    AutogradGraphSimulator,
)
from src.simulators.transformer_architecture import (
    PositionalEncodingEngine,
    AttentionVariantsEngine,
    TransformerBlockEngine,
)

__all__ = [
    "TensorShapeAnalyzer",
    "BroadcastingEngine",
    "MatMulVisualizer",
    "ActivationSimulator",
    "AutogradGraphSimulator",
    "PositionalEncodingEngine",
    "AttentionVariantsEngine",
    "TransformerBlockEngine",
]
