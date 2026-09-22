"""
tests/test_embedder.py

Comprehensive Tests for RAG Embedder Module:
- DummyEmbedder (deterministic hash-based)
- LocalEmbedder (sentence-transformers)
- CachedEmbedder (disk + memory cache)
- Factory function (get_embedder)
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.rag.embedder import (
    DummyEmbedder,
    LocalEmbedder,
    CachedEmbedder,
    get_embedder,
    HAS_SENTENCE_TRANSFORMERS
)


# ============================================================================
# 1. DummyEmbedder Tests
# ============================================================================

def test_dummy_embedder_initialization():
    """Test DummyEmbedder initialization with different dimensions."""
    embedder = DummyEmbedder(d_model=64)
    assert embedder.d_model == 64
    
    embedder2 = DummyEmbedder(d_model=128)
    assert embedder2.d_model == 128


def test_dummy_embedder_single_text():
    """Test single text embedding generation."""
    embedder = DummyEmbedder(d_model=64)
    
    text = "Transformer mimarisi attention kullanır"
    emb = embedder.embed(text)
    
    # Check shape
    assert emb.shape == (64,)
    
    # Check normalized (L2 norm should be ~1.0)
    norm = torch.norm(emb, p=2)
    assert abs(norm.item() - 1.0) < 1e-5


def test_dummy_embedder_deterministic():
    """Test that same text produces same embedding (deterministic)."""
    embedder = DummyEmbedder(d_model=64)
    
    text = "test text"
    emb1 = embedder.embed(text)
    emb2 = embedder.embed(text)
    
    # Should be exactly same
    assert torch.allclose(emb1, emb2, atol=1e-8)


def test_dummy_embedder_different_texts():
    """Test that different texts produce different embeddings."""
    embedder = DummyEmbedder(d_model=64)
    
    emb1 = embedder.embed("text one")
    emb2 = embedder.embed("text two")
    
    # Should be different
    assert not torch.allclose(emb1, emb2, atol=1e-3)


def test_dummy_embedder_batch():
    """Test batch embedding generation."""
    embedder = DummyEmbedder(d_model=64)
    
    texts = ["text1", "text2", "text3"]
    embs = embedder.embed_batch(texts)
    
    # Check shape
    assert embs.shape == (3, 64)
    
    # Check each embedding is normalized
    for i in range(3):
        norm = torch.norm(embs[i], p=2)
        assert abs(norm.item() - 1.0) < 1e-5


# ============================================================================
# 2. LocalEmbedder Tests (if sentence-transformers available)
# ============================================================================

@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_local_embedder_initialization():
    """Test LocalEmbedder initialization."""
    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
    
    # Check dimension (all-MiniLM-L6-v2 should be 384-dim)
    assert embedder.d_model == 384
    assert embedder.device == "cpu"


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_local_embedder_single_text():
    """Test single text embedding with real model."""
    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
    
    text = "Yapay zeka ve derin öğrenme"
    emb = embedder.embed(text)
    
    # Check shape
    assert emb.shape == (384,)
    
    # Check normalized
    norm = torch.norm(emb, p=2)
    assert abs(norm.item() - 1.0) < 1e-5


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_local_embedder_semantic_similarity():
    """Test that semantically similar texts have higher cosine similarity."""
    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
    
    # Similar texts
    emb1 = embedder.embed("kedi evde oturuyor")
    emb2 = embedder.embed("kedi evde uyuyor")
    
    # Dissimilar text
    emb3 = embedder.embed("matematik formülü türev integral")
    
    # Calculate cosine similarities
    sim_similar = torch.dot(emb1, emb2).item()
    sim_dissimilar = torch.dot(emb1, emb3).item()
    
    # Similar texts should have higher similarity
    assert sim_similar > sim_dissimilar
    assert sim_similar > 0.5  # Should be reasonably high


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_local_embedder_batch():
    """Test batch embedding generation."""
    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
    
    texts = [
        "Transformer attention mechanism",
        "Neural network backpropagation",
        "Machine learning optimization"
    ]
    
    embs = embedder.embed_batch(texts, batch_size=2)
    
    # Check shape
    assert embs.shape == (3, 384)
    
    # Check all normalized
    for i in range(3):
        norm = torch.norm(embs[i], p=2)
        assert abs(norm.item() - 1.0) < 1e-5


# ============================================================================
# 3. CachedEmbedder Tests
# ============================================================================

def test_cached_embedder_with_dummy():
    """Test CachedEmbedder wrapping DummyEmbedder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = DummyEmbedder(d_model=64)
        cached = CachedEmbedder(base, cache_dir=tmpdir)
        
        text = "test caching"
        
        # First call - cache miss
        emb1 = cached.embed(text)
        
        # Second call - cache hit
        emb2 = cached.embed(text)
        
        # Should be same
        assert torch.allclose(emb1, emb2)
        
        # Check cache exists
        cache_dir = Path(tmpdir)
        assert len(list(cache_dir.rglob("*.pt"))) > 0


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_cached_embedder_speedup():
    """Test that cache provides speedup."""
    import time
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base = LocalEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
        cached = CachedEmbedder(base, cache_dir=tmpdir)
        
        text = "This is a test sentence for cache speedup"
        
        # First call - cache miss
        start = time.time()
        emb1 = cached.embed(text)
        time_miss = time.time() - start
        
        # Second call - cache hit
        start = time.time()
        emb2 = cached.embed(text)
        time_hit = time.time() - start
        
        # Check results match
        assert torch.allclose(emb1, emb2)
        
        # Cache hit should be much faster
        assert time_hit < time_miss / 10  # At least 10x faster


def test_cached_embedder_batch():
    """Test batch embedding with cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = DummyEmbedder(d_model=64)
        cached = CachedEmbedder(base, cache_dir=tmpdir)
        
        texts = ["text1", "text2", "text3", "text1"]  # text1 repeated
        
        # First batch
        embs1 = cached.embed_batch(texts)
        
        # Second batch with same texts
        embs2 = cached.embed_batch(texts)
        
        # Should match
        assert torch.allclose(embs1, embs2)
        
        # Check cache has 3 unique entries (text1, text2, text3)
        cache_dir = Path(tmpdir)
        cache_files = list(cache_dir.rglob("*.pt"))
        assert len(cache_files) == 3


def test_cached_embedder_clear():
    """Test cache clearing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = DummyEmbedder(d_model=64)
        cached = CachedEmbedder(base, cache_dir=tmpdir)
        
        # Add some cache
        cached.embed("test1")
        cached.embed("test2")
        
        cache_dir = Path(tmpdir)
        assert len(list(cache_dir.rglob("*.pt"))) > 0
        
        # Clear cache
        cached.clear_cache()
        
        # Cache should be empty
        assert len(list(cache_dir.rglob("*.pt"))) == 0


# ============================================================================
# 4. Factory Function Tests
# ============================================================================

def test_get_embedder_dummy():
    """Test factory function for DummyEmbedder."""
    embedder = get_embedder(embedder_type="dummy", d_model=128, use_cache=False)
    
    assert isinstance(embedder, DummyEmbedder)
    assert embedder.d_model == 128


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_get_embedder_local():
    """Test factory function for LocalEmbedder."""
    embedder = get_embedder(
        embedder_type="local",
        model_name="all-MiniLM-L6-v2",
        use_cache=False
    )
    
    assert isinstance(embedder, LocalEmbedder)
    assert embedder.d_model == 384


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_get_embedder_with_cache():
    """Test factory function with cache enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = get_embedder(
            embedder_type="local",
            model_name="all-MiniLM-L6-v2",
            use_cache=True,
            cache_dir=tmpdir
        )
        
        assert isinstance(embedder, CachedEmbedder)
        assert embedder.d_model == 384


def test_get_embedder_invalid_type():
    """Test factory function with invalid embedder type."""
    with pytest.raises(ValueError, match="Unknown embedder type"):
        get_embedder(embedder_type="invalid_type")


# ============================================================================
# 5. Edge Cases and Error Handling
# ============================================================================

def test_dummy_embedder_empty_string():
    """Test DummyEmbedder with empty string."""
    embedder = DummyEmbedder(d_model=64)
    
    emb = embedder.embed("")
    
    # Should still return normalized vector
    assert emb.shape == (64,)
    norm = torch.norm(emb, p=2)
    assert abs(norm.item() - 1.0) < 1e-5


def test_dummy_embedder_unicode():
    """Test DummyEmbedder with unicode characters."""
    embedder = DummyEmbedder(d_model=64)
    
    text = "Türkçe karakterler: ğüşıöç ĞÜŞIÖÇ"
    emb = embedder.embed(text)
    
    assert emb.shape == (64,)
    norm = torch.norm(emb, p=2)
    assert abs(norm.item() - 1.0) < 1e-5


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_local_embedder_long_text():
    """Test LocalEmbedder with very long text."""
    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
    
    # Create long text (>512 tokens)
    long_text = " ".join(["word"] * 600)
    emb = embedder.embed(long_text)
    
    # Should still work (model will truncate internally)
    assert emb.shape == (384,)
    norm = torch.norm(emb, p=2)
    assert abs(norm.item() - 1.0) < 1e-5


# ============================================================================
# Summary Statistics
# ============================================================================

def test_embedder_performance_summary():
    """
    Summary test showing embedder performance characteristics.
    This is informational, not a strict test.
    """
    embedder = DummyEmbedder(d_model=64)
    
    # Test various text lengths
    texts = [
        "short",
        "medium length text with multiple words",
        "very long text " * 50
    ]
    
    for text in texts:
        emb = embedder.embed(text)
        assert emb.shape == (64,)
        # All produce valid embeddings regardless of length
    
    # Batch performance
    batch = ["text"] * 100
    batch_embs = embedder.embed_batch(batch)
    assert batch_embs.shape == (100, 64)
