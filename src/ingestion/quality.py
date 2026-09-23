"""
src/ingestion/quality.py

Ingestion quality scoring utilities.
"""
from typing import Dict, Any, Optional
from src.normalization.text_normalizer import QualityScorer


def calculate_quality_score(text: str, metadata: Optional[Dict[str, Any]] = None) -> float:
    """
    Calculate text quality score (0.0 to 1.0) for ingested text.

    Args:
        text: Raw or normalized text string.
        metadata: Optional metadata dictionary from parser.

    Returns:
        float: Quality score between 0.0 and 1.0.
    """
    if metadata is None:
        metadata = {}
    return QualityScorer.score(text, metadata)
