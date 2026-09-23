"""
tests/test_evaluation_extended.py

Extended unit tests for Model Evaluation & Benchmark Lab endpoints:
- Inspect text BLEU / ROUGE n-gram analysis
- List benchmarks metadata
- BenchmarkRecord database persistence & CRUD (GET, DELETE)
- Model comparison and error handling
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import SessionLocal
from backend.models import BenchmarkRecord


@pytest.fixture
def client():
    return TestClient(app)


def test_inspect_text_exact_match(client):
    """Test text inspection with identical candidate and reference."""
    payload = {
        "candidate": "Yapay zeka dil modelleri hızla gelişiyor.",
        "reference": "Yapay zeka dil modelleri hızla gelişiyor.",
        "max_n": 4
    }
    response = client.post("/api/v1/evaluation/inspect-text", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["brevity_penalty"] == 1.0
    assert data["candidate_len"] == data["reference_len"]
    assert "yapay" in data["matched_unigrams"]
    assert any("yapay zeka" in bg for bg in data["matched_bigrams"])
    assert data["bleu"]["bleu-1"] > 90.0
    assert data["rouge"]["rouge-1"] > 90.0
    assert data["rouge"]["rouge-l"] > 90.0
    assert data["chrf"] == 100.0
    assert data["exact_match"] == 100.0
    assert data["token_f1"]["f1"] == 100.0


def test_inspect_text_partial_overlap(client):
    """Test text inspection with partial overlap and brevity penalty."""
    payload = {
        "candidate": "Bugün hava çok güzel.",
        "reference": "Bugün hava çok güzel ve güneşli bir gün.",
        "max_n": 3
    }
    response = client.post("/api/v1/evaluation/inspect-text", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["candidate_len"] < data["reference_len"]
    assert data["brevity_penalty"] < 1.0
    assert "hava" in data["matched_unigrams"]
    assert "bugün hava" in data["matched_bigrams"]
    assert data["exact_match"] == 0.0
    assert data["token_f1"]["f1"] > 0.0
    assert data["chrf"] > 0.0


def test_inspect_text_empty_mismatch(client):
    """Test text inspection with no matching tokens."""
    payload = {
        "candidate": "Elma armut çilek",
        "reference": "Mavi araba uçak",
        "max_n": 2
    }
    response = client.post("/api/v1/evaluation/inspect-text", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert len(data["matched_unigrams"]) == 0
    assert len(data["matched_bigrams"]) == 0
    assert data["bleu"]["bleu-1"] == 0.0
    assert data["rouge"]["rouge-1"] == 0.0
    assert data["exact_match"] == 0.0
    assert data["token_f1"]["f1"] == 0.0
    # ChrF character unigrams may have incidental overlap ('a', 'm'), but score is very low
    assert data["chrf"] < 20.0


def test_list_available_benchmarks(client):
    """Test GET /api/v1/evaluation/benchmarks endpoint."""
    response = client.get("/api/v1/evaluation/benchmarks")
    assert response.status_code == 200
    data = response.json()

    assert "benchmarks" in data
    assert "perplexity" in data["benchmarks"]
    assert "bleu" in data["benchmarks"]
    assert "rouge" in data["benchmarks"]
    assert "accuracy" in data["benchmarks"]
    assert "turkish_summarization" in data["benchmarks"]
    assert "turkish_qa" in data["benchmarks"]
    assert data["benchmarks"]["perplexity"]["lower_is_better"] is True
    assert data["benchmarks"]["bleu"]["lower_is_better"] is False
    assert data["benchmarks"]["turkish_qa"]["metric"] == "token_f1"


def test_benchmark_database_crud(client):
    """Test BenchmarkRecord database persistence, query, retrieval by ID, and deletion."""
    db = SessionLocal()
    bench_id = f"TEST-BENCH-{int(datetime.utcnow().timestamp() * 1000)}"

    try:
        # Create record directly in DB
        record = BenchmarkRecord(
            benchmark_id=bench_id,
            model_name="test-llm-model",
            benchmark_name="bleu",
            score=78.5,
            metrics={"bleu-1": 85.0, "bleu-2": 78.5, "bleu-4": 65.2},
            samples_evaluated=40,
            dataset_path="datasets/test.jsonl",
            created_at=datetime.utcnow()
        )
        db.add(record)
        db.commit()

        # 1. List results filtered by model_name
        list_resp = client.get(f"/api/v1/evaluation/results?model_name=test-llm-model")
        assert list_resp.status_code == 200
        items = list_resp.json()
        assert len(items) >= 1
        found = next((item for item in items if item["benchmark_id"] == bench_id), None)
        assert found is not None
        assert found["score"] == 78.5
        assert found["benchmark_name"] == "bleu"

        # 2. Get specific result by ID
        get_resp = client.get(f"/api/v1/evaluation/results/{bench_id}")
        assert get_resp.status_code == 200
        item_data = get_resp.json()
        assert item_data["benchmark_id"] == bench_id
        assert item_data["model_name"] == "test-llm-model"
        assert item_data["metrics"]["bleu-1"] == 85.0

        # 3. Delete result
        del_resp = client.delete(f"/api/v1/evaluation/results/{bench_id}")
        assert del_resp.status_code == 200
        assert "deleted successfully" in del_resp.json()["message"]

        # 4. Verify 404 after deletion
        get_after_del = client.get(f"/api/v1/evaluation/results/{bench_id}")
        assert get_after_del.status_code == 404

        # 5. Verify deleting nonexistent ID gives 404
        del_nonexistent = client.delete("/api/v1/evaluation/results/NON-EXISTENT-ID")
        assert del_nonexistent.status_code == 404

    finally:
        # Clean up in case test failed midway
        stale = db.query(BenchmarkRecord).filter(BenchmarkRecord.benchmark_id == bench_id).first()
        if stale:
            db.delete(stale)
            db.commit()
        db.close()


def test_gsm8k_and_turkish_knowledge_benchmarks():
    """Test GSM8K CoT and Turkish Knowledge dataset creators and BenchmarkRunner."""
    from src.evaluation.benchmarks import (
        create_gsm8k_cot_benchmark,
        create_turkish_knowledge_benchmark,
        BenchmarkRunner
    )

    # 1. Test GSM8K CoT dataset
    gsm8k_ds = create_gsm8k_cot_benchmark()
    assert len(gsm8k_ds) >= 8
    ex0 = gsm8k_ds.examples[0]
    assert "####" in ex0.target
    assert ex0.metadata["domain"] == "aritmetik"
    assert "numeric_answer" in ex0.metadata

    # 2. Test Turkish Knowledge dataset
    tr_ds = create_turkish_knowledge_benchmark()
    assert len(tr_ds) >= 8
    tr_ex0 = tr_ds.examples[0]
    assert "keywords" in tr_ex0.metadata
    assert "ankara" in tr_ex0.metadata["keywords"]


def test_sample_questions_api(client):
    """Test GET /api/v1/evaluation/benchmarks/sample-questions endpoint."""
    resp_gsm8k = client.get("/api/v1/evaluation/benchmarks/sample-questions?benchmark_name=gsm8k_cot")
    assert resp_gsm8k.status_code == 200
    data_gsm8k = resp_gsm8k.json()
    assert len(data_gsm8k) >= 8
    assert any("####" in q["target"] for q in data_gsm8k)
    assert data_gsm8k[0]["numeric_answer"] is not None

    resp_tr = client.get("/api/v1/evaluation/benchmarks/sample-questions?benchmark_name=turkish_knowledge")
    assert resp_tr.status_code == 200
    data_tr = resp_tr.json()
    assert len(data_tr) >= 8
    assert any("ankara" in (q["keywords"] or []) for q in data_tr)

    resp_sm = client.get("/api/v1/evaluation/benchmarks/sample-questions?benchmark_name=turkish_summarization")
    assert resp_sm.status_code == 200
    data_sm = resp_sm.json()
    assert len(data_sm) >= 6
    assert data_sm[0]["category"] == "summarization"
    assert len(data_sm[0]["target"]) > 0

    resp_qa = client.get("/api/v1/evaluation/benchmarks/sample-questions?benchmark_name=turkish_qa")
    assert resp_qa.status_code == 200
    data_qa = resp_qa.json()
    assert len(data_qa) >= 8
    assert data_qa[0]["context"] is not None
    assert data_qa[0]["question"] is not None


def test_turkish_benchmarks_and_metrics():
    """Test standalone metrics: normalize, exact_match, token_f1, chrf, and dataset builders."""
    from src.evaluation.metrics import (
        normalize_text_for_eval,
        compute_exact_match,
        compute_token_f1,
        compute_chrf,
    )
    from src.evaluation.benchmarks import (
        create_turkish_summarization_benchmark,
        create_turkish_qa_benchmark,
    )

    # 1. Normalization
    assert normalize_text_for_eval("  Örnek, Metin! Çığlık?  ") == "örnek metin çığlık"

    # 2. Exact match
    assert compute_exact_match("Ankara", ["İstanbul", "ankara"]) == 100.0
    assert compute_exact_match("İzmir", "Ankara") == 0.0

    # 3. Token F1
    f1_exact = compute_token_f1("Mustafa Kemal Atatürk", "Mustafa Kemal Atatürk")
    assert f1_exact["f1"] == 100.0
    f1_part = compute_token_f1("Atatürk liderdir", "Mustafa Kemal Atatürk")
    assert 0.0 < f1_part["f1"] < 100.0

    # 4. ChrF
    chrf_exact = compute_chrf("Türkiye Cumhuriyeti", "Türkiye Cumhuriyeti")
    assert chrf_exact == 100.0
    chrf_part = compute_chrf("Türkiye", "Türkiye Cumhuriyeti")
    assert 0.0 < chrf_part < 100.0

    # 5. Datasets
    sm_ds = create_turkish_summarization_benchmark()
    assert len(sm_ds) >= 6
    assert sm_ds.name == "turkish_summarization"

    qa_ds = create_turkish_qa_benchmark()
    assert len(qa_ds) >= 8
    assert qa_ds.name == "turkish_qa"


def test_radar_comparison_api(client, monkeypatch):
    """Test POST /api/v1/evaluation/radar-comparison endpoint with 6 dimensions."""
    from src.evaluation.benchmarks import BenchmarkResult, BenchmarkRunner

    def fake_run_benchmark(self, name, **kwargs):
        return BenchmarkResult(
            benchmark_id="test_bm_123",
            model_name=self.model_name,
            benchmark_name=name,
            score=82.5 if "small" in self.model_name else 74.0,
            metrics={"score": 80.0},
            timestamp=datetime.now(),
            samples_evaluated=8
        )

    import backend.routers.evaluation as eval_router
    monkeypatch.setattr(eval_router.BenchmarkRunner, "__init__", lambda self, model_name, device="cpu": setattr(self, "model_name", model_name))
    monkeypatch.setattr(eval_router.BenchmarkRunner, "run_benchmark", fake_run_benchmark)

    payload = {
        "model_names": ["nano-gpt-v1", "turkish-gpt-small"]
    }
    response = client.post("/api/v1/evaluation/radar-comparison", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "models" in data
    assert len(data["models"]) == 2
    assert len(data["dimensions"]) == 6
    assert "overall_winner" in data
    assert "winner_by_dimension" in data

    for model_res in data["models"]:
        assert len(model_res["dimensions"]) == 6
        assert 0.0 <= model_res["overall_average"] <= 100.0
        for dim in model_res["dimensions"]:
            assert 0.0 <= dim["score"] <= 100.0
            assert dim["dimension_key"] in ["reasoning", "knowledge", "qa", "summarization", "fluency", "stability"]

