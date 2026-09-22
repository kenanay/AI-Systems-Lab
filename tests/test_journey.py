"""
tests/test_journey.py

Unit tests for Guided Learning Journey, Knowledge Graph, and AI Glossary
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.learning.journey_engine import JourneyEngine


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def engine():
    return JourneyEngine()


def test_journey_engine_curriculum_integrity(engine):
    """Verify all 13 stages are present, ordered, and have questions."""
    curriculum = engine.get_curriculum()
    assert len(curriculum) == 13

    for idx, stage in enumerate(curriculum):
        assert stage["order"] == idx
        assert stage["id"] == f"stage_{idx}"
        assert len(stage["concepts"]) > 0
        assert len(stage["objectives"]) > 0
        assert len(stage["questions"]) >= 2
        assert stage["title"].startswith(f"Aşama {idx}")


def test_journey_engine_knowledge_graph_validity(engine):
    """Verify graph nodes and edges have no dangling references."""
    graph = engine.get_knowledge_graph()
    nodes = graph["nodes"]
    edges = graph["edges"]

    assert len(nodes) >= 10
    assert len(edges) >= 10

    node_ids = {n["id"] for n in nodes}
    for edge in edges:
        assert edge["source"] in node_ids, f"Dangling edge source: {edge['source']}"
        assert edge["target"] in node_ids, f"Dangling edge target: {edge['target']}"


def test_journey_engine_glossary_filtering(engine):
    """Verify glossary searching and category filtering."""
    all_terms = engine.get_glossary()
    assert len(all_terms) >= 15

    # Filter by category
    mimari_terms = engine.get_glossary(category="Mimari")
    assert len(mimari_terms) > 0
    assert all(t["category"] == "Mimari" for t in mimari_terms)

    # Search by keyword
    rope_search = engine.get_glossary(search="RoPE")
    assert len(rope_search) >= 1
    assert any("RoPE" in t["term"] for t in rope_search)


def test_journey_engine_check_question(engine):
    """Verify question answer checking and pedagogical feedback."""
    # Test correct answer
    res_correct = engine.check_question("q_stage_0_1", 1)
    assert res_correct["is_correct"] is True
    assert res_correct["correct_index"] == 1
    assert "iç boyutlar" in res_correct["explanation"]

    # Test wrong answer
    res_wrong = engine.check_question("q_stage_0_1", 0)
    assert res_wrong["is_correct"] is False
    assert res_wrong["correct_index"] == 1

    # Test non-existent question
    res_none = engine.check_question("non_existent_q", 0)
    assert res_none["is_correct"] is False


def test_api_curriculum_endpoints(client):
    """Verify GET /api/v1/journey/curriculum endpoints."""
    resp = client.get("/api/v1/journey/curriculum")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 13

    # Specific stage
    resp_stage = client.get("/api/v1/journey/curriculum/stage_0")
    assert resp_stage.status_code == 200
    assert resp_stage.json()["id"] == "stage_0"

    # Non-existent stage
    resp_404 = client.get("/api/v1/journey/curriculum/stage_999")
    assert resp_404.status_code == 404


def test_api_graph_endpoint(client):
    """Verify GET /api/v1/journey/graph endpoint."""
    resp = client.get("/api/v1/journey/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) > 0


def test_api_glossary_endpoint(client):
    """Verify GET /api/v1/journey/glossary endpoint."""
    resp = client.get("/api/v1/journey/glossary?category=De%C4%9Ferlendirme")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) > 0
    assert any("BLEU" in item["term"] for item in data)


def test_api_check_question_endpoint(client):
    """Verify POST /api/v1/journey/check-question endpoint."""
    payload = {
        "question_id": "q_stage_6_1",
        "selected_option": 1
    }
    resp = client.post("/api/v1/journey/check-question", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_correct"] is True
    assert "Ölçekleme" in data["explanation"] or "Softmax" in data["explanation"]
