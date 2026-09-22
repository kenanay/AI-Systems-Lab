"""
PII Detection Module
"""

from src.pii.turkish_detector import (
    TurkishPIIDetector,
    PIIType,
    PIIMatch,
    scan_and_mask_text,
)

scan_and_mask = scan_and_mask_text

__all__ = [
    "TurkishPIIDetector",
    "PIIType",
    "PIIMatch",
    "scan_and_mask_text",
    "scan_and_mask",
]
