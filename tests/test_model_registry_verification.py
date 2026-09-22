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
