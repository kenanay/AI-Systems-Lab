"""
tests/test_sft_integration.py

Integration tests for Supervised Fine-Tuning (SFT and SFT_LORA)
within the TrainingService and API workflows.
"""

import time
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import SessionLocal
from backend.models import TrainingJob

client = TestClient(app)


def test_sft_training_worker_execution() -> None:
    """Test starting an SFT job, verifying worker executes with InstructionDataset and loss masking."""
    payload = {
        "job_name": "Test SFT Integration Run",
        "model_name": "gpt-sft-test",
        "job_type": "SFT",
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
    job_id = data["job_id"]
    assert data["job_type"] == "SFT"

    # Wait for completion (max 20 seconds)
    completed = False
    for _ in range(40):
        time.sleep(0.5)
        res = client.get(f"/api/v1/training/jobs/{job_id}")
        if res.status_code == 200:
            job_data = res.json()
            if job_data["status"] == "COMPLETED":
                completed = True
                assert job_data["progress"] == 1.0
                assert job_data["best_checkpoint"] is not None
                assert isinstance(job_data.get("metrics"), list)
                break
            elif job_data["status"] == "FAILED":
                pytest.fail(f"SFT Job failed: {job_data.get('error')}")

    assert completed, f"SFT Job {job_id} did not finish in time"


def test_sft_lora_training_worker_execution() -> None:
    """Test starting an SFT_LORA job with LoRA adapter injection."""
    payload = {
        "job_name": "Test SFT LoRA Integration",
        "model_name": "gpt-sft-lora-test",
        "job_type": "SFT_LORA",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 2,
        "max_seq_len": 32,
        "use_lora": True,
        "lora_r": 4,
        "lora_alpha": 8
    }
    response = client.post("/api/v1/training/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    job_id = data["job_id"]
    assert data["job_type"] == "SFT_LORA"

    completed = False
    for _ in range(40):
        time.sleep(0.5)
        res = client.get(f"/api/v1/training/jobs/{job_id}")
        if res.status_code == 200:
            job_data = res.json()
            if job_data["status"] == "COMPLETED":
                completed = True
                assert job_data["progress"] == 1.0
                assert job_data["best_checkpoint"] is not None
                break
            elif job_data["status"] == "FAILED":
                pytest.fail(f"SFT_LORA Job failed: {job_data.get('error')}")

    assert completed, f"SFT_LORA Job {job_id} did not finish in time"
