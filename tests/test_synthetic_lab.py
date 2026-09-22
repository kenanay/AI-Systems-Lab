"""
tests/test_synthetic_lab.py

Unit and API tests for Synthetic Data Generation & Filtering Lab
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.quality.synthetic_curation import (
    SyntheticDataGenerator,
    QualityScoringEngine,
    QualityFilterPipeline,
    FilterThresholds,
    SyntheticSample,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_generator_paradigms():
    """Verify synthetic data generation across all 4 paradigms."""
    gen = SyntheticDataGenerator()
    paradigms = ["self_instruct", "chain_of_thought", "code_synthesis", "textbook_qa"]

    for p in paradigms:
        samples = gen.generate_batch(paradigm=p, domain="computer_science", count=2, include_edge_cases=False)
        assert len(samples) == 2
        for s in samples:
            assert s.id.startswith(f"SYNTH-{p[:4].upper()}")
            assert len(s.instruction) > 5
            assert len(s.response) > 10
            assert "composite_score" in s.metrics
            assert s.metrics["composite_score"] >= 60.0


def test_quality_scorer_clean_and_degenerate():
    """Verify scoring behavior on clean text vs repetitive loop, low perplexity, and PII."""
    scorer = QualityScoringEngine()

    # 1. Clean pedagogical text
    clean_instr = "İkili arama algoritmasını açıklayın."
    clean_resp = (
        "İkili arama (Binary Search), sıralı bir dizi üzerinde hedef elemanı bulmak için "
        "böl ve fethet mantığıyla çalışır. Her adımda arama uzayını ikiye bölerek "
        "O(log n) zamanda sonuca ulaşır."
    )
    clean_score = scorer.score_sample(clean_instr, clean_resp)
    assert clean_score["composite_score"] >= 75.0
    assert clean_score["verdict"] in ["ACCEPT", "NEEDS_REVISION"]
    assert clean_score["pii_count"] == 0
    assert clean_score["repetition_ratio_2g"] < 0.15
    assert 10.0 <= clean_score["perplexity"] <= 85.0

    # 2. Degenerate repetition loop
    rep_resp = "yapay zeka " * 40
    rep_score = scorer.score_sample("AI nedir?", rep_resp)
    assert rep_score["repetition_ratio_2g"] > 0.40
    assert rep_score["perplexity"] < 6.0
    assert rep_score["composite_score"] < clean_score["composite_score"]

    # 3. PII leakage
    pii_resp = "Öğrenci Ali Veli, TC Kimlik No: 10000000146 ve telefon: 0555 123 45 67 olarak kayıtlıdır."
    pii_score = scorer.score_sample("Kayıt bilgisi ver.", pii_resp)
    assert pii_score["pii_count"] >= 1
    assert any("PII Tespiti" in r for r in pii_score["verdict_reasons"])


def test_filter_pipeline_stages_and_funnel():
    """Verify that multi-stage filter pipeline correctly filters out flawed samples and generates funnel stats."""
    gen = SyntheticDataGenerator()
    pipe = QualityFilterPipeline()

    # Generate 3 valid + 4 edge cases (total 7)
    samples = gen.generate_batch(count=3, include_edge_cases=True)
    assert len(samples) >= 7

    thresholds = FilterThresholds(
        min_perplexity=5.0,
        max_perplexity=120.0,
        min_words=12,
        max_repetition_ratio=0.18,
        min_quality_score=65.0,
        pii_action="reject",
        max_jaccard_similarity=0.85
    )

    result = pipe.run_pipeline(samples, thresholds)
    assert result["total_input_count"] >= 7
    assert len(result["passed_samples"]) >= 1
    assert len(result["rejected_samples"]) >= 3
    assert result["final_yield_pct"] > 0.0
    assert result["avg_final_quality"] >= result["avg_initial_quality"]

    # Check that funnel stages exist
    stage_names = [st["stage_name"] for st in result["funnel_stages"]]
    assert any("Uzunluk" in name for name in stage_names)
    assert any("Tekrarlama" in name for name in stage_names)
    assert any("PII" in name for name in stage_names)
    assert any("Perplexity" in name for name in stage_names)
    assert any("Kalite" in name for name in stage_names)
    assert any("Tekilleştirme" in name for name in stage_names)


def test_pii_masking_action():
    """Verify that pii_action='mask' masks sensitive numbers and keeps sample alive if other metrics pass."""
    pipe = QualityFilterPipeline()
    scorer = QualityScoringEngine()

    full_resp = (
        "Kullanıcı profil kaydı oluşturuldu. "
        "TC Kimlik: 10000000146, Telefon: 0555 123 45 67, E-posta: test.user@ornek.com. "
        "Bu kullanıcı sisteme başarıyla giriş yaptı ve doğrulandı."
    )
    pii_sample = SyntheticSample(
        id="SYNTH-PII-TEST",
        instruction="Kullanıcı profili örneği oluştur.",
        input_context="Kayıt sistemi",
        response=full_resp,
        paradigm="self_instruct",
        domain="turkish_knowledge",
        complexity="intermediate",
        metrics=scorer.score_sample("Kullanıcı profili örneği oluştur.", full_resp)
    )

    # 1. Action = reject
    res_reject = pipe.run_pipeline([pii_sample], FilterThresholds(pii_action="reject"))
    assert len(res_reject["passed_samples"]) == 0
    assert len(res_reject["rejected_samples"]) == 1
    assert "PII" in res_reject["rejected_samples"][0]["rejection_stage"]

    # 2. Action = mask
    res_mask = pipe.run_pipeline([pii_sample], FilterThresholds(pii_action="mask", min_quality_score=40.0))
    assert len(res_mask["passed_samples"]) == 1
    masked_resp = res_mask["passed_samples"][0]["response"]
    assert "12345678901" not in masked_resp  # Must be masked like 123****8901 or similar


def test_synthetic_lab_api_endpoints(client):
    """Verify all REST API endpoints for Synthetic Data Lab."""
    # 1. GET /api/v1/synthetic-lab/templates
    r_templates = client.get("/api/v1/synthetic-lab/templates")
    assert r_templates.status_code == 200
    tpl_data = r_templates.json()
    assert "paradigms" in tpl_data
    assert "preset_profiles" in tpl_data
    assert "strict" in tpl_data["preset_profiles"]
    assert "balanced" in tpl_data["preset_profiles"]

    # 2. POST /api/v1/synthetic-lab/generate
    r_gen = client.post(
        "/api/v1/synthetic-lab/generate",
        json={
            "paradigm": "self_instruct",
            "domain": "computer_science",
            "complexity": "intermediate",
            "count": 2,
            "include_edge_cases": True
        }
    )
    assert r_gen.status_code == 200
    gen_data = r_gen.json()
    assert gen_data["total_generated"] >= 2
    assert len(gen_data["samples"]) >= 2
    sample_list = gen_data["samples"]

    # 3. POST /api/v1/synthetic-lab/filter
    r_filter = client.post(
        "/api/v1/synthetic-lab/filter",
        json={
            "samples": sample_list,
            "thresholds": {
                "min_perplexity": 5.0,
                "max_perplexity": 120.0,
                "min_words": 10,
                "max_words": 1000,
                "max_repetition_ratio": 0.20,
                "min_quality_score": 60.0,
                "pii_action": "reject",
                "max_jaccard_similarity": 0.85
            }
        }
    )
    assert r_filter.status_code == 200
    filter_data = r_filter.json()
    assert "funnel_stages" in filter_data
    assert "passed_samples" in filter_data
    assert "final_yield_pct" in filter_data

    # 4. POST /api/v1/synthetic-lab/score-sample
    r_score = client.post(
        "/api/v1/synthetic-lab/score-sample",
        json={
            "instruction": "Kırmızı dev yıldızların evrimi nasıldır?",
            "response": (
                "Kırmızı devler, çekirdeğindeki hidrojeni tüketen orta kütleli yıldızların genişlemesiyle oluşur. "
                "Helyum füzyonu başlar ve yıldızın dış katmanları uzaya yayılarak gezegenimsi bulutsu oluşturur."
            ),
            "input_context": "Astrofizik"
        }
    )
    assert r_score.status_code == 200
    score_data = r_score.json()
    assert "composite_score" in score_data
    assert "perplexity" in score_data
    assert "subscores" in score_data

    # 5. POST /api/v1/synthetic-lab/export
    r_export = client.post(
        "/api/v1/synthetic-lab/export",
        json={
            "samples": filter_data["passed_samples"],
            "dataset_name": "astrophysics_synthetic_v1",
            "export_format": "jsonl"
        }
    )
    assert r_export.status_code == 200
    export_data = r_export.json()
    assert export_data["success"] is True
    assert export_data["total_samples"] == len(filter_data["passed_samples"])
    assert "preview_jsonl" in export_data

    # 6. POST /api/v1/synthetic-lab/ingest-to-dataset
    r_ingest = client.post(
        "/api/v1/synthetic-lab/ingest-to-dataset",
        json={
            "samples": filter_data["passed_samples"][:2],
            "dataset_name": "astrophysics_synthetic_curated",
            "domain": "natural_sciences",
            "paradigm": "textbook_qa",
            "training_allowed": True
        }
    )
    assert r_ingest.status_code == 200
    ingest_data = r_ingest.json()
    assert ingest_data["success"] is True
    assert ingest_data["file_id"].startswith("file_synth_")
    assert ingest_data["document_count"] == 2
    assert "avg_quality_score" in ingest_data

    # 7. POST /api/v1/synthetic-lab/ingest-to-dataset duplicate handling
    r_duplicate = client.post(
        "/api/v1/synthetic-lab/ingest-to-dataset",
        json={
            "samples": filter_data["passed_samples"][:2],
            "dataset_name": "astrophysics_synthetic_curated",
        }
    )
    assert r_duplicate.status_code == 200
    dup_data = r_duplicate.json()
    assert dup_data["file_id"] == ingest_data["file_id"]

    # 8. POST /api/v1/synthetic-lab/ingest-to-dataset empty validation
    r_empty = client.post(
        "/api/v1/synthetic-lab/ingest-to-dataset",
        json={
            "samples": [],
            "dataset_name": "empty_set"
        }
    )
    assert r_empty.status_code == 400

