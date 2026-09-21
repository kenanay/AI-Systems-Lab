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
