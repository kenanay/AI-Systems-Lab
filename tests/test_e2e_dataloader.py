"""
tests/test_e2e_dataloader.py

End-to-End DataLoader Test
Compiled Parquet dataset'ten batch loading test eder.
"""

from pathlib import Path
import sys
import pytest

from src.training.dataloader import create_dataloader, DataLoaderStats

PARQUET_PATH = Path("datasets/compiled_datasets/turkish_education_v1.0.0/dataset.parquet")
BATCH_SIZE = 2
SEQUENCE_LENGTH = 256


def test_e2e_dataloader():
    """E2E DataLoader test - executes if parquet dataset exists, skips cleanly otherwise."""
    if not PARQUET_PATH.exists():
        pytest.skip(f"Parquet file not found: {PARQUET_PATH}")

    dataloader = create_dataloader(
        parquet_path=PARQUET_PATH,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        stride=None,
        pack_sequences=True,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    assert dataloader.dataset is not None
    assert len(dataloader) > 0

    stats = DataLoaderStats()
    for batch_idx, batch in enumerate(dataloader):
        stats.update(batch)
        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch

        B, T = batch["input_ids"].shape
        assert B == BATCH_SIZE
        assert T == SEQUENCE_LENGTH

        if batch_idx >= 2:
            break

    summary = stats.summary()
    assert summary["total_batches"] > 0
    assert summary["total_tokens"] > 0
