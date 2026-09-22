"""
tests/test_evaluation.py

Unit and integration tests for evaluation metrics and benchmark runner:
- BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Batch generation evaluation
- Perplexity and accuracy metrics
- BenchmarkRunner and evaluation API router
"""

import pytest
import torch
from fastapi.testclient import TestClient

from src.evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    evaluate_generation,
    compute_perplexity,
    compute_accuracy,
    tokenize_text,
)
from src.evaluation.benchmarks import BenchmarkRunner, BenchmarkResult
from backend.main import app


# ============================================================================
# BLEU Metric Tests
# ============================================================================

def test_tokenize_text():
    """Test text tokenization for metrics."""
    tokens = tokenize_text("Merhaba Dünya! Bu bir test cümlesidir.")
    assert "merhaba" in tokens
    assert "dünya" in tokens
    assert "test" in tokens
    assert tokenize_text("") == []


def test_compute_bleu_exact_match():
    """Test BLEU score with exact match."""
    pred = "türkiye nin başkenti ankara dır"
    ref = "türkiye nin başkenti ankara dır"
    scores = compute_bleu(pred, ref)
    assert scores["bleu-1"] == 100.0
    assert scores["bleu-2"] == 100.0
    assert scores["bleu-3"] == 100.0
    assert scores["bleu-4"] == 100.0
    assert scores["bleu"] == 100.0


def test_compute_bleu_no_overlap():
    """Test BLEU score with completely different texts."""
    pred = "elma armut muz çilek"
    ref = "araba tren uçak gemi"
    scores = compute_bleu(pred, ref)
    assert scores["bleu-1"] == 0.0
    assert scores["bleu"] == 0.0


def test_compute_bleu_partial_match():
    """Test BLEU score with partial overlap."""
    pred = "yapay zeka sistemleri yerel bilgisayarda çalışır"
    ref = "yapay zeka modelleri yerel sunucularda çalışır"
    scores = compute_bleu(pred, ref)
    assert scores["bleu-1"] > 0.0
    assert scores["bleu-1"] <= 100.0


def test_compute_bleu_empty_inputs():
    """Test BLEU score with empty strings."""
    assert compute_bleu("", "referans")["bleu"] == 0.0
    assert compute_bleu("tahmin", "")["bleu"] == 0.0
    assert compute_bleu("", "")["bleu"] == 0.0


def test_compute_bleu_multiple_references():
    """Test BLEU score with multiple valid references."""
    pred = "türkiye nin başkenti ankara dır"
    refs = [
        "başkent ankara dır",
        "türkiye nin başkenti ankara dır",
    ]
    scores = compute_bleu(pred, refs)
    assert scores["bleu"] == 100.0


# ============================================================================
# ROUGE Metric Tests
# ============================================================================

def test_compute_rouge_exact_match():
    """Test ROUGE with exact match."""
    pred = "derin öğrenme modelleri büyük verilerle eğitilir"
    ref = "derin öğrenme modelleri büyük verilerle eğitilir"
    scores = compute_rouge(pred, ref)
    assert scores["rouge-1"] == 100.0
    assert scores["rouge-2"] == 100.0
    assert scores["rouge-l"] == 100.0


def test_compute_rouge_no_overlap():
    """Test ROUGE with no word overlap."""
    pred = "kedi köpek kuş balık"
    ref = "masa sandalye dolap halı"
    scores = compute_rouge(pred, ref)
    assert scores["rouge-1"] == 0.0
    assert scores["rouge-2"] == 0.0
    assert scores["rouge-l"] == 0.0


def test_compute_rouge_partial_match():
    """Test ROUGE with partial overlap."""
    pred = "yapay zeka ve makine öğrenmesi"
    ref = "derin öğrenme ve yapay zeka sistemleri"
    scores = compute_rouge(pred, ref)
    assert scores["rouge-1"] > 0.0
    assert scores["rouge-l"] > 0.0


def test_compute_rouge_empty_inputs():
    """Test ROUGE with empty inputs."""
    scores = compute_rouge("", "")
    assert scores["rouge-1"] == 0.0
    assert scores["rouge-2"] == 0.0
    assert scores["rouge-l"] == 0.0


# ============================================================================
# Generation Evaluation Batch Tests
# ============================================================================

def test_evaluate_generation_batch():
    """Test batch generation evaluation."""
    preds = [
        "türkiye nin başkenti ankara",
        "python yüksek seviyeli dildir",
    ]
    refs = [
        "türkiye nin başkenti ankara",
        "python nesne yönelimli dildir",
    ]
    summary = evaluate_generation(preds, refs)
    assert "bleu" in summary
    assert "rouge-1" in summary
    assert "rouge-l" in summary
    assert summary["bleu-1"] > 50.0
    assert summary["rouge-1"] > 50.0


# ============================================================================
# Perplexity and Accuracy Tests
# ============================================================================

def test_compute_perplexity_and_accuracy():
    """Test perplexity and accuracy on known logits."""
    batch_size = 2
    seq_len = 4
    vocab_size = 10
    
    # Perfect predictions: logits very high at target index
    targets = torch.tensor([[1, 2, 3, 4], [0, 5, 2, 1]], dtype=torch.long)
    logits = torch.zeros(batch_size, seq_len, vocab_size)
    for b in range(batch_size):
        for s in range(seq_len):
            logits[b, s, targets[b, s]] = 20.0
            
    perp = compute_perplexity(logits, targets)
    predictions = logits.argmax(dim=-1)
    acc = compute_accuracy(predictions, targets)
    
    assert perp >= 1.0
    assert perp < 1.05  # Almost 1.0 for confident correct logits
    assert acc == 1.0  # 100% accuracy (1.0)


# ============================================================================
# BenchmarkRunner Tests
# ============================================================================

def test_benchmark_runner_unsupported():
    """Test unsupported benchmark name raises ValueError."""
    runner = BenchmarkRunner("dummy-model")
    with pytest.raises(ValueError):
        runner.run_benchmark("invalid_metric_name")


# ============================================================================
# FastAPI Evaluation Router Tests
# ============================================================================

def test_evaluation_api_list_results():
    """Test GET /api/v1/evaluation/results endpoint."""
    client = TestClient(app)
    response = client.get("/api/v1/evaluation/results")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_evaluation_api_model_not_found():
    """Test POST /api/v1/evaluation/run for nonexistent model."""
    client = TestClient(app)
    response = client.post(
        "/api/v1/evaluation/run",
        json={
            "model_name": "nonexistent-model-xyz",
            "benchmark_name": "perplexity",
            "max_samples": 5,
            "batch_size": 2
        }
    )
    # Model not found returns 400 or 404
    assert response.status_code in [400, 404]
