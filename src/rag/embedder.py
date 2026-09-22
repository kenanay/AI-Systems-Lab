"""
src/rag/embedder.py

Text Embedding Generation for RAG
Local AI Research Lab - Developed by Kenan AY

Bu modül metin embedding'lerini üretir:
- LocalEmbedder: sentence-transformers ile local embedding (CPU/GPU)
- CachedEmbedder: Disk cache ile hızlandırılmış embedding
- DummyEmbedder: Test ve fallback için deterministik embedding
"""

import torch
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Optional sentence-transformers import
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Using DummyEmbedder fallback.")


class BaseEmbedder:
    """Base class for all embedders."""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
    
    def embed(self, text: str) -> torch.Tensor:
        """
        Generate embedding for single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding tensor, shape [d_model]
        """
        raise NotImplementedError
    
    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Generate embeddings for batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Embedding tensor, shape [batch_size, d_model]
        """
        # Shape: [batch_size, d_model]
        return torch.stack([self.embed(text) for text in texts])


class DummyEmbedder(BaseEmbedder):
    """
    Deterministik hash-based embedding generator.
    Test ve fallback için kullanılır.
    """
    
    def __init__(self, d_model: int = 64):
        super().__init__(d_model)
        logger.info(f"DummyEmbedder initialized (d_model={d_model})")
    
    def embed(self, text: str) -> torch.Tensor:
        """
        Hash-based deterministik vektör üretir.
        
        Args:
            text: Input text
            
        Returns:
            Normalized embedding, shape [d_model]
        """
        clean = text.lower().strip()
        h = abs(hash(clean)) % (2**31 - 1)
        generator = torch.Generator().manual_seed(h)
        # vec shape: [d_model]
        vec = torch.randn(self.d_model, generator=generator)
        # L2 normalize: vec / ||vec||
        return vec / torch.norm(vec, p=2).clamp_min(1e-12)


class LocalEmbedder(BaseEmbedder):
    """
    sentence-transformers kullanarak local embedding üretir.
    GPU varsa otomatik kullanır, yoksa CPU'da çalışır.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        # Device seçimi
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.normalize = normalize
        self.model_name = model_name
        
        logger.info(f"Loading sentence-transformers model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Model dimension'ı otomatik belirle
        test_embedding = self.model.encode("test", convert_to_tensor=True)
        super().__init__(d_model=test_embedding.shape[0])
        
        logger.info(f"LocalEmbedder initialized: {model_name} (d_model={self.d_model}, device={device})")
    
    def embed(self, text: str) -> torch.Tensor:
        """
        Tek metin için embedding üretir.
        
        Args:
            text: Input text
            
        Returns:
            Embedding tensor, shape [d_model]
        """
        # embedding shape: [d_model]
        embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            device=self.device
        )
        return embedding.cpu()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Batch metinler için embedding üretir (daha hızlı).
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            
        Returns:
            Embeddings tensor, shape [batch_size, d_model]
        """
        # embeddings shape: [len(texts), d_model]
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            device=self.device,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.cpu()


class CachedEmbedder(BaseEmbedder):
    """
    Disk cache ile hızlandırılmış embedder.
    Aynı metni tekrar embed etmek yerine cache'den okur.
    """
    
    def __init__(
        self,
        base_embedder: BaseEmbedder,
        cache_dir: Union[str, Path] = ".embeddings_cache",
        max_cache_size_mb: int = 500
    ):
        super().__init__(base_embedder.d_model)
        self.base_embedder = base_embedder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_mb = max_cache_size_mb
        
        # In-memory cache (LRU-like)
        self.memory_cache: Dict[str, torch.Tensor] = {}
        self.max_memory_items = 1000
        
        logger.info(f"CachedEmbedder initialized (cache_dir={cache_dir})")
    
    def _text_hash(self, text: str) -> str:
        """Text için cache key oluşturur."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, text_hash: str) -> Path:
        """Cache dosya path'i."""
        # İlk 2 karakter ile subdirectory oluştur (daha iyi file system performance)
        subdir = self.cache_dir / text_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{text_hash}.pt"
    
    def embed(self, text: str) -> torch.Tensor:
        """
        Cache'li embedding üretir.
        
        Args:
            text: Input text
            
        Returns:
            Cached or newly generated embedding, shape [d_model]
        """
        text_hash = self._text_hash(text)
        
        # 1. Memory cache check
        if text_hash in self.memory_cache:
            return self.memory_cache[text_hash]
        
        # 2. Disk cache check
        cache_path = self._get_cache_path(text_hash)
        if cache_path.exists():
            try:
                embedding = torch.load(cache_path, map_location='cpu')
                # Memory cache'e ekle
                if len(self.memory_cache) >= self.max_memory_items:
                    # LRU-like: İlk item'ı çıkar
                    self.memory_cache.pop(next(iter(self.memory_cache)))
                self.memory_cache[text_hash] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Cache load error: {e}")
        
        # 3. Generate new embedding
        embedding = self.base_embedder.embed(text)
        
        # 4. Save to disk cache
        try:
            torch.save(embedding, cache_path)
        except Exception as e:
            logger.warning(f"Cache save error: {e}")
        
        # 5. Add to memory cache
        if len(self.memory_cache) >= self.max_memory_items:
            self.memory_cache.pop(next(iter(self.memory_cache)))
        self.memory_cache[text_hash] = embedding
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Batch embedding with cache.
        
        Args:
            texts: List of input texts
            
        Returns:
            Embeddings tensor, shape [batch_size, d_model]
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for all texts
        for i, text in enumerate(texts):
            text_hash = self._text_hash(text)
            
            # Memory cache
            if text_hash in self.memory_cache:
                embeddings.append((i, self.memory_cache[text_hash]))
                continue
            
            # Disk cache
            cache_path = self._get_cache_path(text_hash)
            if cache_path.exists():
                try:
                    embedding = torch.load(cache_path, map_location='cpu')
                    embeddings.append((i, embedding))
                    self.memory_cache[text_hash] = embedding
                    continue
                except:
                    pass
            
            # Not cached
            uncached_texts.append(text)
            uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.base_embedder.embed_batch(uncached_texts)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                text_hash = self._text_hash(text)
                cache_path = self._get_cache_path(text_hash)
                try:
                    torch.save(embedding, cache_path)
                except:
                    pass
                self.memory_cache[text_hash] = embedding
            
            # Add to results
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, embedding))
        
        # Sort by original index and stack
        # Result shape: [len(texts), d_model]
        embeddings.sort(key=lambda x: x[0])
        return torch.stack([emb for _, emb in embeddings])
    
    def clear_cache(self):
        """Clear disk and memory cache."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True)
        self.memory_cache.clear()
        logger.info("Cache cleared")


def get_embedder(
    embedder_type: str = "local",
    model_name: str = "all-MiniLM-L6-v2",
    d_model: int = 64,
    use_cache: bool = True,
    cache_dir: str = ".embeddings_cache",
    **kwargs
) -> BaseEmbedder:
    """
    Embedder factory function.
    
    Args:
        embedder_type: 'local' (sentence-transformers) or 'dummy' (hash-based)
        model_name: sentence-transformers model name
        d_model: Embedding dimension (only for dummy)
        use_cache: Enable disk cache
        cache_dir: Cache directory path
        
    Returns:
        BaseEmbedder instance
    """
    embedder_type = embedder_type.lower().strip()
    
    if embedder_type == "dummy":
        embedder = DummyEmbedder(d_model=d_model)
    elif embedder_type == "local":
        if not HAS_SENTENCE_TRANSFORMERS:
            logger.warning("sentence-transformers not available, falling back to DummyEmbedder")
            embedder = DummyEmbedder(d_model=d_model)
        else:
            embedder = LocalEmbedder(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")
    
    # Wrap with cache if requested
    if use_cache and embedder_type != "dummy":
        embedder = CachedEmbedder(embedder, cache_dir=cache_dir)
    
    return embedder


# Convenience function for quick embedding
def embed_text(
    text: str,
    embedder_type: str = "local",
    model_name: str = "all-MiniLM-L6-v2"
) -> torch.Tensor:
    """
    Quick embedding generation without creating embedder instance.
    
    Args:
        text: Input text
        embedder_type: 'local' or 'dummy'
        model_name: sentence-transformers model name
        
    Returns:
        Embedding tensor, shape [d_model]
    """
    embedder = get_embedder(embedder_type=embedder_type, model_name=model_name, use_cache=True)
    return embedder.embed(text)
