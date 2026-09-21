"""
tests/test_e2e_config.py

End-to-End Config Integration Test

Training config oluşturur ve validate eder.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config import (
    TrainingRunConfig,
    ModelArchitectureConfig,
    TrainingConfig,
    DataConfig
)

print("=" * 80)
print("E2E Config Integration Test")
print("=" * 80)

# Test parameters - real data from E2E test
DATASET_PATH = "datasets/compiled_datasets/turkish_education_v1.0.0/dataset.parquet"
TOKENIZER_ID = "TOK-D026F1DF"
VOCAB_SIZE = 942  # Actual vocab size from tokenizer
CONTEXT_LENGTH = 512

# Test 1: Create model config
print("\n[Test 1] Model Architecture Config")
print(f"  Creating config with vocab_size={VOCAB_SIZE}, context_length={CONTEXT_LENGTH}")

try:
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
    
    # Validate
    model_config.validate()
    
    print(f"  ✅ Model config created and validated")
    print(f"  Architecture: {model_config.architecture}")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_layers: {model_config.n_layers}")
    print(f"  n_heads: {model_config.n_heads}")
    print(f"  d_head: {model_config.d_head}")
    print(f"  Estimated parameters: ~{model_config.num_parameters / 1e6:.1f}M")
    
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    sys.exit(1)

# Test 2: Create training config
print("\n[Test 2] Training Config")

try:
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
    
    # Validate
    training_config.validate()
    
    print(f"  ✅ Training config created and validated")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Max steps: {training_config.max_steps:,}")
    print(f"  Warmup steps: {training_config.warmup_steps}")
    print(f"  Scheduler: {training_config.scheduler}")
    
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    sys.exit(1)

# Test 3: Create data config
print("\n[Test 3] Data Config")
print(f"  Dataset path: {DATASET_PATH}")
print(f"  Tokenizer ID: {TOKENIZER_ID}")

try:
    data_config = DataConfig(
        dataset_path=DATASET_PATH,
        tokenizer_id=TOKENIZER_ID,
        sequence_length=512,
        stride=None,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Validate
    data_config.validate()
    
    print(f"  ✅ Data config created and validated")
    print(f"  Sequence length: {data_config.sequence_length}")
    print(f"  Shuffle: {data_config.shuffle}")
    print(f"  Num workers: {data_config.num_workers}")
    
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    sys.exit(1)

# Test 4: Create complete training run config
print("\n[Test 4] Complete Training Run Config")

try:
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
    
    # Validate (full validation including cross-config checks)
    config.validate()
    
    print(f"  ✅ Complete config created and validated")
    print(f"  Config name: {config.name}")
    print(f"  Config version: {config.version}")
    print(f"  Model parameters: ~{config.model.num_parameters / 1e6:.1f}M")
    
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Save and load config
print("\n[Test 5] Save and Load Config")
output_path = Path("configs/training/test_turkish_education.yaml")
output_path.parent.mkdir(parents=True, exist_ok=True)

try:
    # Save
    config.to_yaml(output_path)
    print(f"  ✅ Config saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size} bytes")
    
    # Load
    loaded_config = TrainingRunConfig.from_yaml(output_path)
    print(f"  ✅ Config loaded from YAML")
    
    # Verify
    assert loaded_config.name == config.name
    assert loaded_config.model.vocab_size == config.model.vocab_size
    assert loaded_config.training.batch_size == config.training.batch_size
    assert loaded_config.data.dataset_path == config.data.dataset_path
    
    print(f"  ✅ Loaded config matches original")
    
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Config to dict
print("\n[Test 6] Config Serialization")

try:
    config_dict = config.to_dict()
    
    print(f"  ✅ Config serialized to dict")
    print(f"  Dict keys: {list(config_dict.keys())}")
    print(f"  Model config keys: {list(config_dict['model'].keys())}")
    
    # Verify dict structure
    assert "version" in config_dict
    assert "name" in config_dict
    assert "model" in config_dict
    assert "training" in config_dict
    assert "data" in config_dict
    assert "metadata" in config_dict
    
    print(f"  ✅ Dict structure valid")
    
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("E2E Config Integration Test Summary")
print("=" * 80)
print("✅ Model config: PASSED")
print("✅ Training config: PASSED")
print("✅ Data config: PASSED")
print("✅ Complete config: PASSED")
print("✅ Save/load: PASSED")
print("✅ Serialization: PASSED")
print("\n🎉 All tests PASSED!")
print("=" * 80)
print(f"\n📝 Config file created: {output_path}")
print(f"   Ready for training with ~{config.model.num_parameters / 1e6:.1f}M parameter model")
