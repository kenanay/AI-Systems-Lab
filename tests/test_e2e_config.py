"""
tests/test_e2e_config.py

End-to-End Config Integration Test
Training config oluşturur ve validate eder.
"""

from pathlib import Path
import pytest

from src.training.config import (
    TrainingRunConfig,
    ModelArchitectureConfig,
    TrainingConfig,
    DataConfig
)

DATASET_PATH = "datasets/compiled_datasets/turkish_education_v1.0.0/dataset.parquet"
TOKENIZER_ID = "TOK-D026F1DF"
VOCAB_SIZE = 942
CONTEXT_LENGTH = 512


def test_e2e_config(tmp_path: Path):
    """End-to-end config creation, validation, YAML save/load and serialization."""
    # 1. Model config
    model_config = ModelArchitectureConfig(
        architecture="transformer_decoder",
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
        activation="gelu",
        use_bias=True,
        tie_embeddings=True
    )
    model_config.validate()
    assert model_config.d_head == 32
    assert model_config.num_parameters > 0

    # 2. Training config
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=0.0003,
        max_steps=100000,
        warmup_steps=1000,
        gradient_clip=1.0,
        weight_decay=0.01,
        eval_interval=1000,
        save_interval=5000,
        scheduler="cosine",
        min_lr=1e-5
    )
    training_config.validate()
    assert training_config.batch_size == 32

    # 3. Data config
    dataset_file = Path(DATASET_PATH)
    if not dataset_file.exists():
        # Use dummy file in tmp_path for validation if actual dataset not on disk
        dummy_dataset = tmp_path / "dummy.parquet"
        dummy_dataset.write_text("dummy")
        used_dataset_path = str(dummy_dataset)
    else:
        used_dataset_path = DATASET_PATH

    data_config = DataConfig(
        dataset_path=used_dataset_path,
        tokenizer_id=TOKENIZER_ID,
        sequence_length=512,
        stride=None,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )
    data_config.validate()
    assert data_config.sequence_length == 512

    # 4. Complete TrainingRunConfig
    config = TrainingRunConfig(
        version="1.0.0",
        name="turkish_education_training",
        model=model_config,
        training=training_config,
        data=data_config,
        metadata={
            "description": "Turkish education content training",
            "dataset_version": "1.0.0",
            "tags": ["turkish", "education", "test"]
        }
    )
    config.validate()
    assert config.name == "turkish_education_training"

    # 5. Save and load from YAML
    yaml_path = tmp_path / "test_config.yaml"
    config.to_yaml(yaml_path)
    assert yaml_path.exists()

    loaded_config = TrainingRunConfig.from_yaml(yaml_path)
    assert loaded_config.name == config.name
    assert loaded_config.model.vocab_size == config.model.vocab_size
    assert loaded_config.training.batch_size == config.training.batch_size

    # 6. Serialization to dict
    config_dict = config.to_dict()
    assert "version" in config_dict
    assert "model" in config_dict
    assert "training" in config_dict
    assert "data" in config_dict
