import pytest
from pathlib import Path
import tempfile
import torch

from src.registry.model_registry import ModelRegistry, ModelMetadata


def test_model_registry_integrity_verification():
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(registry_dir=temp_dir)

        # Create dummy checkpoint and tokenizer
        chk_path = Path(temp_dir) / "dummy.pt"
        tok_path = Path(temp_dir) / "dummy.model"
        torch.save({"dummy": torch.tensor([1, 2, 3])}, chk_path)
        tok_path.write_text("dummy tokenizer")

        # Register model
        metadata = registry.register_model(
            model_name="test-llm",
            version="1.0.0",
            checkpoint_path=chk_path,
            tokenizer_path=tok_path,
            description="Unit test model",
            parameters=1000
        )

        assert metadata.model_hash is not None
        assert len(metadata.model_hash) == 64

        # 1. Verification of valid model
        verify_result = registry.verify_model("test-llm", version="1.0.0")
        assert verify_result["verified"] is True
        assert verify_result["status"] == "VALID"
        assert verify_result["actual_hash"] == metadata.model_hash

        # 2. Loading with verify_integrity=True should succeed
        loaded = registry.load_model("test-llm", version="1.0.0", verify_integrity=True)
        assert loaded["metadata"]["model_name"] == "test-llm"

        # 3. Tamper with the checkpoint file
        saved_chk = Path(registry.models_dir) / "test-llm" / "1.0.0" / "model.pt"
        with open(saved_chk, "ab") as f:
            f.write(b"TAMPERED_CONTENT")

        # 4. Verification should now report CORRUPTED
        corrupted_result = registry.verify_model("test-llm", version="1.0.0")
        assert corrupted_result["verified"] is False
        assert corrupted_result["status"] == "CORRUPTED"
        assert corrupted_result["actual_hash"] != metadata.model_hash

        # 5. Loading with verify_integrity=True should raise ValueError
        with pytest.raises(ValueError, match="Model bütünlük doğrulaması başarısız"):
            registry.load_model("test-llm", version="1.0.0", verify_integrity=True)


def test_artifact_compatibility_endpoint():
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.security.jwt import create_access_token

    client = TestClient(app)
    token = create_access_token({"sub": "admin", "role": "admin"})
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Compatible scenario: model vocab >= tokenizer vocab, matching tokenizer
    res_comp = client.post("/api/v1/models/validate-compatibility", json={
        "model_vocab_size": 1000,
        "tokenizer_vocab_size": 800,
        "model_tokenizer_id": "tok_bpe_1",
        "tokenizer_id": "tok_bpe_1",
        "dataset_version": "v1.0",
        "tokenizer_dataset_version": "v1.0"
    }, headers=headers)
    assert res_comp.status_code == 200
    data_comp = res_comp.json()
    assert data_comp["compatible"] is True
    assert data_comp["status"] == "compatible"
    assert len(data_comp["warnings"]) == 0

    # 2. Incompatible scenario: model vocab < tokenizer vocab (index overflow risk)
    res_overflow = client.post("/api/v1/models/validate-compatibility", json={
        "model_vocab_size": 500,
        "tokenizer_vocab_size": 1000,
        "model_tokenizer_id": "tok_bpe_1",
        "tokenizer_id": "tok_bpe_1"
    }, headers=headers)
    assert res_overflow.status_code == 200
    data_overflow = res_overflow.json()
    assert data_overflow["compatible"] is False
    assert data_overflow["status"] == "incompatible"
    assert any("İndeks Taşması" in w for w in data_overflow["warnings"])

    # 3. Tokenizer mismatch warning
    res_mismatch = client.post("/api/v1/models/validate-compatibility", json={
        "model_vocab_size": 1000,
        "tokenizer_vocab_size": 1000,
        "model_tokenizer_id": "tok_bpe_1",
        "tokenizer_id": "tok_wordpiece_2"
    }, headers=headers)
    assert res_mismatch.status_code == 200
    data_mismatch = res_mismatch.json()
    assert data_mismatch["status"] == "warning"
    assert any("Tokenizer Uyuşmazlığı" in w for w in data_mismatch["warnings"])

