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
    from backend.database import SessionLocal
    from backend.models import TokenizerRecord
    import tempfile
    import shutil

    client = TestClient(app)
    token = create_access_token({"sub": "admin", "role": "admin"})
    headers = {"Authorization": f"Bearer {token}"}

    # Seed real Tokenizer in DB
    db = SessionLocal()
    try:
        tok_seed = db.query(TokenizerRecord).filter(TokenizerRecord.tokenizer_id == "tok_verified_1").first()
        if not tok_seed:
            tok_seed = TokenizerRecord(
                tokenizer_id="tok_verified_1",
                name="verified_bpe",
                tokenizer_type="BPE",
                vocab_size=800,
                storage_path="models/tokenizers/tok_verified_1"
            )
            db.add(tok_seed)
            db.commit()
    finally:
        db.close()

    # Seed real Model in ModelRegistry
    registry = ModelRegistry("models")
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_chk, tempfile.NamedTemporaryFile(suffix=".json") as tmp_tok:
        torch.save({"dummy": torch.tensor([1])}, tmp_chk.name)
        Path(tmp_tok.name).write_text("{}")
        try:
            registry.register_model(
                model_name="test-server-compat-model",
                version="1.0.0",
                checkpoint_path=tmp_chk.name,
                tokenizer_path=tmp_tok.name,
                training_config={
                    "vocab_size": 1000,
                    "tokenizer_id": "tok_verified_1"
                }
            )
        except ValueError:
            pass  # Already registered

    # 1. Server-authoritative compatible scenario: model & tokenizer exist and match
    res_server_comp = client.post("/api/v1/models/validate-compatibility", json={
        "model_name": "test-server-compat-model",
        "model_version": "1.0.0",
        "tokenizer_id": "tok_verified_1"
    }, headers=headers)
    assert res_server_comp.status_code == 200
    data_srv = res_server_comp.json()
    assert data_srv["compatible"] is True
    assert data_srv["status"] == "compatible"
    assert data_srv["checks"]["model_verified"] is True
    assert data_srv["checks"]["tokenizer_verified"] is True

    # 2. Simulated math scenario (no registered names): vocab 1000 vs 800
    res_comp = client.post("/api/v1/models/validate-compatibility", json={
        "model_vocab_size": 1000,
        "tokenizer_vocab_size": 800,
        "model_tokenizer_id": "tok_sim_1"
    }, headers=headers)
    assert res_comp.status_code == 200
    data_comp = res_comp.json()
    assert data_comp["compatible"] is True
    assert data_comp["status"] == "compatible"

    # 3. Incompatible scenario: model vocab < tokenizer vocab (index overflow risk)
    res_overflow = client.post("/api/v1/models/validate-compatibility", json={
        "model_vocab_size": 500,
        "tokenizer_vocab_size": 1000
    }, headers=headers)
    assert res_overflow.status_code == 200
    data_overflow = res_overflow.json()
    assert data_overflow["compatible"] is False
    assert data_overflow["status"] == "incompatible"
    assert any("İndeks Taşması" in w for w in data_overflow["warnings"])

    # 4. Tokenizer mismatch produces incompatible / error (Model was trained with tok_verified_1, but tok_different provided)
    # Seed a second tokenizer
    db = SessionLocal()
    try:
        if not db.query(TokenizerRecord).filter(TokenizerRecord.tokenizer_id == "tok_verified_2").first():
            db.add(TokenizerRecord(
                tokenizer_id="tok_verified_2",
                name="verified_bpe_2",
                tokenizer_type="BPE",
                vocab_size=800,
                storage_path="models/tokenizers/tok_verified_2"
            ))
            db.commit()
    finally:
        db.close()

    res_mismatch = client.post("/api/v1/models/validate-compatibility", json={
        "model_name": "test-server-compat-model",
        "model_version": "1.0.0",
        "tokenizer_id": "tok_verified_2"
    }, headers=headers)
    assert res_mismatch.status_code == 200
    data_mismatch = res_mismatch.json()
    assert data_mismatch["compatible"] is False
    assert data_mismatch["status"] == "incompatible"
    assert any("Tokenizer Uyuşmazlığı" in w for w in data_mismatch["warnings"])

    # 5. Server-authoritative: Non-existent model must return verification_failed (NOT compatible)
    res_missing_model = client.post("/api/v1/models/validate-compatibility", json={
        "model_name": "definitely_not_a_registered_model",
        "tokenizer_id": "tok_verified_1"
    }, headers=headers)
    assert res_missing_model.status_code == 200
    data_missing = res_missing_model.json()
    assert data_missing["compatible"] is False
    assert data_missing["status"] == "verification_failed"
    assert any("doğrulanamadı" in w or "bulunamadı" in w for w in data_missing["warnings"])



