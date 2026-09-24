import pytest
from pathlib import Path
import tempfile
import torch

from src.registry.model_registry import ModelRegistry, ModelMetadata
from src.model.gpt import GPTConfig, GPTModel


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


def test_inference_pipeline_rejects_executable_checkpoints(tmp_path):
    """The standalone inference loader must use the same safe boundary as the registry."""
    import torch
    from src.inference.pipeline import InferencePipeline

    flag_file = tmp_path / "pipeline-pwned.txt"

    class Exploit:
        def __reduce__(self):
            import os
            return (os.system, (f"touch {flag_file}",))

    checkpoint = tmp_path / "malicious.pt"
    torch.save({"payload": Exploit()}, checkpoint)

    with pytest.raises(ValueError, match="weights_only=True"):
        InferencePipeline.from_pretrained(checkpoint, tmp_path / "unused-tokenizer.json")

    assert not flag_file.exists()


def test_legacy_gpt_config_checkpoint_remains_loadable(tmp_path):
    """Legacy checkpoints storing GPTConfig objects still load through the safe allowlist."""
    config = GPTConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        dropout=0.0,
    )
    checkpoint = tmp_path / "legacy-gpt.pt"
    torch.save(
        {
            "model_state_dict": GPTModel(config).state_dict(),
            "config": config,
            "epoch": 3,
        },
        checkpoint,
    )

    registry = ModelRegistry(tmp_path / "registry")
    registry.register_model(
        model_name="legacy-gpt",
        version="1.0.0",
        checkpoint_path=checkpoint,
    )
    loaded = registry.load_model("legacy-gpt", version="1.0.0", load_weights=True)

    assert isinstance(loaded["full_checkpoint"]["config"], GPTConfig)
    assert loaded["full_checkpoint"]["epoch"] == 3
    assert loaded["state_dict"]


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

    # 6. Fail-closed: Model with missing vocab_size or missing tokenizer link returns verification_failed
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_chk2, tempfile.NamedTemporaryFile(suffix=".json") as tmp_tok2:
        torch.save({"dummy": torch.tensor([1])}, tmp_chk2.name)
        Path(tmp_tok2.name).write_text("{}")
        try:
            registry.register_model(
                model_name="test-missing-meta-model",
                version="1.0.0",
                checkpoint_path=tmp_chk2.name,
                tokenizer_path=tmp_tok2.name,
                training_config={}  # Missing vocab_size and tokenizer_id!
            )
        except ValueError:
            pass

    res_missing_meta = client.post("/api/v1/models/validate-compatibility", json={
        "model_name": "test-missing-meta-model",
        "model_version": "1.0.0",
        "tokenizer_id": "tok_verified_1"
    }, headers=headers)
    assert res_missing_meta.status_code == 200
    data_missing_meta = res_missing_meta.json()
    assert data_missing_meta["compatible"] is False
    assert data_missing_meta["status"] == "verification_failed"
    assert any("zorunlu sözlük boyutu" in w or "eksik" in w for w in data_missing_meta["warnings"])


