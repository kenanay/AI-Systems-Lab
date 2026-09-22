"""
tests/test_training_and_inference_api.py

Test Model, Training, and Inference API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_models_list_api() -> None:
    """Test GET /api/v1/models"""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_inference_status_api() -> None:
    """Test GET /api/v1/inference/status"""
    response = client.get("/api/v1/inference/status")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "model_loaded" in data


def test_training_jobs_list_api() -> None:
    """Test GET /api/v1/training/jobs"""
    response = client.get("/api/v1/training/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_start_training_api_validation() -> None:
    """Test POST /api/v1/training/start validation"""
    payload = {
        "job_name": "Test Mini Run",
        "model_name": "gpt-test-tiny",
        "job_type": "PRETRAIN",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 2,
        "max_seq_len": 32
    }
    response = client.post("/api/v1/training/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["job_name"] == "Test Mini Run"
    assert data["status"] in ["PENDING", "RUNNING", "COMPLETED"]

    # Test job detail
    job_id = data["job_id"]
    detail_res = client.get(f"/api/v1/training/jobs/{job_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["job_id"] == job_id


def test_delete_nonexistent_model_api() -> None:
    """Test DELETE /api/v1/models/{model_name} for nonexistent model"""
    response = client.delete("/api/v1/models/definitely_nonexistent_model_xyz")
    assert response.status_code == 404


def test_verify_nonexistent_model_api() -> None:
    """Test POST /api/v1/models/{model_name}/verify for nonexistent model"""
    response = client.post("/api/v1/models/nonexistent_model_xyz/verify")
    assert response.status_code == 404


def test_verify_existing_model_if_present() -> None:
    """Test POST /api/v1/models/{model_name}/verify for available model in registry."""
    list_res = client.get("/api/v1/models")
    assert list_res.status_code == 200
    models = list_res.json()
    if models:
        first_model = models[0]
        model_name = first_model["model_name"]
        verify_res = client.post(f"/api/v1/models/{model_name}/verify")
        assert verify_res.status_code == 200
        data = verify_res.json()
        assert "verified" in data
        assert "actual_hash" in data
        assert "expected_hash" in data


def test_beam_search_api() -> None:
    """Test POST /api/v1/inference/beam-search"""
    response = client.post(
        "/api/v1/inference/beam-search",
        json={
            "prompt": "Yapay zeka ve derin öğrenme",
            "beam_width": 3,
            "max_length": 8,
            "length_penalty": 1.0,
            "num_return_sequences": 2
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "hypotheses" in data
    assert len(data["hypotheses"]) >= 1
    assert "score" in data["hypotheses"][0]
    assert "tokens" in data["hypotheses"][0]
    assert "execution_time_ms" in data


def test_next_token_probs_api() -> None:
    """Test POST /api/v1/inference/next-token-probs"""
    response = client.post(
        "/api/v1/inference/next-token-probs",
        json={
            "prompt": "Transformer mimarisinde attention",
            "top_k": 5,
            "temperature": 1.0
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert len(data["candidates"]) == 5
    assert data["candidates"][0]["rank"] == 1
    assert "probability" in data["candidates"][0]
    assert "raw_logit" in data["candidates"][0]


def test_streaming_generation_endpoint() -> None:
    """Test POST /api/v1/inference/generate/stream SSE streaming"""
    response = client.post(
        "/api/v1/inference/generate/stream",
        json={
            "prompt": "Test streaming",
            "max_length": 5,
            "temperature": 0.8
        }
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

