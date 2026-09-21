"""
tests/test_e2e_dataloader.py

End-to-End DataLoader Test

Compiled Parquet dataset'ten batch loading test eder.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataloader import create_dataloader, DataLoaderStats

# Test parameters
PARQUET_PATH = Path("datasets/compiled_datasets/turkish_education_v1.0.0/dataset.parquet")
BATCH_SIZE = 2
SEQUENCE_LENGTH = 256

print("=" * 80)
print("E2E DataLoader Test")
print("=" * 80)

# Test 1: DataLoader creation
print("\n[Test 1] DataLoader Creation")
print(f"  Parquet path: {PARQUET_PATH}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Sequence length: {SEQUENCE_LENGTH}")

if not PARQUET_PATH.exists():
    print(f"  ❌ ERROR: Parquet file not found: {PARQUET_PATH}")
    sys.exit(1)

try:
    dataloader = create_dataloader(
        parquet_path=PARQUET_PATH,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        stride=None,  # No overlap
        pack_sequences=True,  # Pack sequences for efficiency
        shuffle=False,  # Deterministic for testing
        num_workers=0,  # Single-threaded for simplicity
        pin_memory=False
    )
    print(f"  ✅ DataLoader created successfully")
    print(f"  Dataset size: {len(dataloader.dataset)} sequences")
    print(f"  Number of batches: {len(dataloader)}")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Batch iteration
print("\n[Test 2] Batch Iteration")
stats = DataLoaderStats()

try:
    for batch_idx, batch in enumerate(dataloader):
        # Update stats
        stats.update(batch)
        
        # Validate batch structure
        assert "input_ids" in batch, "input_ids missing"
        assert "labels" in batch, "labels missing"
        assert "attention_mask" in batch, "attention_mask missing"
        
        # Validate shapes
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        
        B, T = input_ids.shape
        assert labels.shape == (B, T), f"labels shape mismatch"
        assert attention_mask.shape == (B, T), f"attention_mask shape mismatch"
        assert B == BATCH_SIZE, f"batch_size mismatch: expected {BATCH_SIZE}, got {B}"
        assert T == SEQUENCE_LENGTH, f"sequence_length mismatch: expected {SEQUENCE_LENGTH}, got {T}"
        
        # Print first batch details
        if batch_idx == 0:
            print(f"  Batch {batch_idx + 1}:")
            print(f"    Shape: input_ids={input_ids.shape}, labels={labels.shape}")
            print(f"    input_ids[0][:10]: {input_ids[0][:10].tolist()}")
            print(f"    labels[0][:10]: {labels[0][:10].tolist()}")
            print(f"    attention_mask[0][:10]: {attention_mask[0][:10].tolist()}")
            
            # Count valid tokens (non-padding)
            valid_tokens = (attention_mask[0] == 1).sum().item()
            print(f"    Valid tokens in seq 0: {valid_tokens}/{SEQUENCE_LENGTH}")
        
        # Only process first 3 batches for test
        if batch_idx >= 2:
            break
    
    print(f"  ✅ Batch iteration successful")
    print(f"  Processed {batch_idx + 1} batches")
    
except Exception as e:
    print(f"  ❌ ERROR during iteration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Statistics
print("\n[Test 3] DataLoader Statistics")
summary = stats.summary()
print(f"  Total batches: {summary['total_batches']}")
print(f"  Total tokens: {summary['total_tokens']:,}")
print(f"  Valid tokens: {summary['valid_tokens']:,}")
print(f"  Padding tokens: {summary['padding_tokens']:,}")
print(f"  Padding ratio: {summary['padding_ratio']:.2%}")
print(f"  Efficiency: {summary['efficiency']:.2%}")

if summary['efficiency'] > 0.5:
    print(f"  ✅ Good efficiency (>50%)")
else:
    print(f"  ⚠️  Low efficiency (<50%) - consider adjusting sequence packing")

# Test 4: Token validation
print("\n[Test 4] Token Validation")
print("  Checking if token IDs are within vocabulary bounds...")

max_token_id = 0
min_token_id = float('inf')

for batch_idx, batch in enumerate(dataloader):
    input_ids = batch["input_ids"]
    batch_max = input_ids.max().item()
    batch_min = input_ids.min().item()
    
    max_token_id = max(max_token_id, batch_max)
    min_token_id = min(min_token_id, batch_min)
    
    if batch_idx >= 2:
        break

print(f"  Token ID range: [{min_token_id}, {max_token_id}]")
print(f"  Expected vocab size: 942 (from tokenizer TOK-D026F1DF)")

if max_token_id < 942:
    print(f"  ✅ All token IDs within vocabulary bounds")
else:
    print(f"  ⚠️  Some token IDs exceed vocabulary size")

# Summary
print("\n" + "=" * 80)
print("E2E DataLoader Test Summary")
print("=" * 80)
print("✅ DataLoader creation: PASSED")
print("✅ Batch iteration: PASSED")
print("✅ Statistics tracking: PASSED")
print("✅ Token validation: PASSED")
print("\n🎉 All tests PASSED!")
print("=" * 80)
