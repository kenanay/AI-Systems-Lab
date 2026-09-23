"""
tests/test_sft_integration.py

Integration tests for Supervised Fine-Tuning (SFT and SFT_LORA)
within the TrainingService and API workflows.
"""

import time
import pytest
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import SessionLocal, init_db
from backend.models import FileRecord, TokenizerRecord, DatasetVersion, TrainingJob
from src.tokenizer.bpe import BPETokenizer
from src.tokenizer.loading import artifact_hash
from src.model.gpt import GPTModel, GPTConfig
from src.registry.model_registry import ModelRegistry

client = TestClient(app)


@pytest.fixture(scope="module")
def sft_env(tmp_path_factory):
    init_db()
    tmp_dir = tmp_path_factory.mktemp("sft_integration")

    # 1. Train and save BPE tokenizer
    tok_dir = tmp_dir / "tokenizer"
    tok = BPETokenizer(vocab_size=260)
    tok.train([
        "Merhaba dünya, nasılsın?",
        "Yapay zeka modelleri metin üretir.",
        "Instruction: Bu nedir? Response: Bu bir cevaptır."
    ])
    tok.save(tok_dir)
    tok_hash = artifact_hash(tok_dir)

    # 2. Save base GPT model checkpoint
    model_dir = tmp_dir / "base_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    cp_path = model_dir / "model.pt"

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        dropout=0.0
    )
    base_model = GPTModel(cfg)
    torch.save({
        "model_state_dict": base_model.state_dict(),
        "config": {
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_ff": cfg.d_ff,
            "dropout": cfg.dropout
        },
        "tokenizer_sha256": tok_hash
    }, cp_path)

    # Register in ModelRegistry("models")
    registry = ModelRegistry("models")
    base_model_name = f"gpt-sft-base-{int(time.time())}"
    base_version = "1.0.0"
    registry.register_model(
        model_name=base_model_name,
        version=base_version,
        checkpoint_path=cp_path,
        tokenizer_path=tok_dir
    )

    # 3. Create instruction dataset parquet
    data_dir = tmp_dir / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "dataset.parquet"

    table = pa.Table.from_pydict({
        "instruction": [
            "Türkiye'nin başkenti neresidir?",
            "Yapay zeka nedir?",
            "Python nedir?",
            "Makine öğrenmesi nedir?"
        ],
        "response": [
            "Ankara'dır.",
            "Yapay zeka akıllı sistemlerdir.",
            "Python bir programlama dilidir.",
            "Veriden öğrenen algoritmalar bütünüdür."
        ],
        "split": ["train", "train", "validation", "validation"]
    })
    pq.write_table(table, parquet_path)
    ds_hash = artifact_hash(parquet_path)

    # 4. Insert records into DB
    db = SessionLocal()
    try:
        file_id = f"FILE-SFT-{int(time.time())}"
        file_rec = FileRecord(
            file_id=file_id,
            original_name="sft_data.txt",
            relative_path=str(parquet_path),
            mime_type="application/octet-stream",
            size_bytes=100,
            sha256="a" * 64,
            training_allowed=True
        )
        db.add(file_rec)

        tok_id = f"TOK-SFT-{int(time.time())}"
        tok_rec = TokenizerRecord(
            tokenizer_id=tok_id,
            name="sft-bpe",
            tokenizer_type="BPE",
            vocab_size=tok.vocab_size,
            storage_path=str(tok_dir),
            is_active=True
        )
        db.add(tok_rec)

        ds_id = f"DS-SFT-{int(time.time())}"
        ds_rec = DatasetVersion(
            dataset_id=ds_id,
            name="sft-dataset",
            version="1.0.0",
            compiler_version="1.0.0",
            num_documents=4,
            tokenizer_id=tok_id,
            storage_path=str(parquet_path),
            is_active=True,
            source_file_ids=[file_id],
            custom_metadata={
                "dataset_sha256": ds_hash,
                "tokenizer_sha256": tok_hash
            }
        )
        db.add(ds_rec)
        db.commit()

        yield {
            "dataset_id": ds_id,
            "tokenizer_id": tok_id,
            "base_model": base_model_name,
            "base_version": base_version,
            "vocab_size": tok.vocab_size
        }
    finally:
        db.close()


def test_sft_training_worker_execution(sft_env) -> None:
    """Test starting an SFT job, verifying worker executes with InstructionDataset and loss masking."""
    payload = {
        "job_name": "Test SFT Integration Run",
        "model_name": "gpt-sft-test",
        "job_type": "SFT",
        "dataset_id": sft_env["dataset_id"],
        "tokenizer_id": sft_env["tokenizer_id"],
        "base_model": sft_env["base_model"],
        "base_version": sft_env["base_version"],
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 2,
        "max_seq_len": 32
    }
    response = client.post("/api/v1/training/start", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    job_id = data["job_id"]
    assert data["job_type"] in {"SFT", "FULL_SFT"}

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


def test_sft_lora_training_worker_execution(sft_env) -> None:
    """Test starting an SFT_LORA job with LoRA adapter injection."""
    payload = {
        "job_name": "Test SFT LoRA Integration",
        "model_name": "gpt-sft-lora-test",
        "job_type": "SFT_LORA",
        "dataset_id": sft_env["dataset_id"],
        "tokenizer_id": sft_env["tokenizer_id"],
        "base_model": sft_env["base_model"],
        "base_version": sft_env["base_version"],
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 2,
        "max_seq_len": 32,
        "lora_r": 4,
        "lora_alpha": 8
    }
    response = client.post("/api/v1/training/start", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    job_id = data["job_id"]
    assert data["job_type"] in {"SFT_LORA", "LORA_SFT"}

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
