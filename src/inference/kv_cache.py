"""
src/inference/kv_cache.py

Key-Value Cache for Autoregressive Generation

Bu modül autoregressive generation'da key/value state'leri cache'leyerek
recomputation'ı önler ve inference'ı hızlandırır.

KV Cache Optimization:
-----------------------
Autoregressive generation'da her step:
1. Tüm önceki token'ları tekrar process etmek → O(n²) complexity
2. Veya sadece yeni token'ı process et + önceki K,V'leri cache'den al → O(n)

Cache Strategy:
- Her layer için separate K,V cache
- Her new token için sadece yeni K,V compute et
- Önceki K,V'leri cache'den concat et

Memory Trade-off:
- Extra memory: [batch, n_heads, seq_len, d_k] × 2 (K and V) × n_layers
- Speed gain: ~10-50x for long sequences

Real-world usage:
- GPT, Claude, ChatGPT hepsi KV cache kullanır
- "Prefill" phase: İlk prompt → tüm K,V compute
- "Decode" phase: Her yeni token → sadece 1 token compute + cache concat
"""

import torch
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KVCacheConfig:
    """
    KV Cache configuration.
    
    Attributes:
        max_batch_size: Maximum batch size to cache
        max_seq_len: Maximum sequence length to cache
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_k: Dimension per head
        dtype: Data type for cache
        device: Device to store cache
    """
    max_batch_size: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    d_k: int
    dtype: torch.dtype = torch.float32
    device: str = 'cpu'


class LayerKVCache:
    """
    Key-Value cache for a single transformer layer.
    
    Stores K and V tensors for reuse in autoregressive generation.
    
    Attributes:
        k_cache: Cached key states, shape [batch, n_heads, seq_len, d_k]
        v_cache: Cached value states, shape [batch, n_heads, seq_len, d_k]
        current_length: Current cached sequence length
    
    Example:
        >>> cache = LayerKVCache(
        ...     max_batch=4, max_seq_len=512, 
        ...     n_heads=8, d_k=64
        ... )
        >>> # First step: cache initial K,V
        >>> cache.update(k_new, v_new, start_pos=0)
        >>> # Next steps: append new K,V
        >>> cache.update(k_new, v_new, start_pos=5)
    """
    
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_heads: int,
        d_k: int,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu'
    ):
        """Initialize layer cache."""
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.d_k = d_k
        self.dtype = dtype
        self.device = device
        
        # Preallocate cache tensors
        # k_cache shape: [max_batch, n_heads, max_seq_len, d_k]
        self.k_cache = torch.zeros(
            max_batch_size, n_heads, max_seq_len, d_k,
            dtype=dtype, device=device
        )
        
        # v_cache shape: [max_batch, n_heads, max_seq_len, d_k]
        self.v_cache = torch.zeros(
            max_batch_size, n_heads, max_seq_len, d_k,
            dtype=dtype, device=device
        )
        
        # Track current length
        self.current_length = 0
        
        logger.debug(f"LayerKVCache initialized: [{max_batch_size}, {n_heads}, {max_seq_len}, {d_k}]")
    
    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        start_pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K,V and return full K,V.
        
        Args:
            k_new: New key states, shape [batch, n_heads, new_seq_len, d_k]
            v_new: New value states, shape [batch, n_heads, new_seq_len, d_k]
            start_pos: Position to insert new K,V
        
        Returns:
            k_full: Full key states including cache, shape [batch, n_heads, total_len, d_k]
            v_full: Full value states including cache, shape [batch, n_heads, total_len, d_k]
        
        Example:
            >>> # k_new shape: [2, 8, 1, 64] (batch=2, heads=8, new_len=1, d_k=64)
            >>> # v_new shape: [2, 8, 1, 64]
            >>> k_full, v_full = cache.update(k_new, v_new, start_pos=5)
            >>> # k_full shape: [2, 8, 6, 64] (includes 5 cached + 1 new)
        """
        # k_new shape: [batch, n_heads, new_seq_len, d_k]
        # v_new shape: [batch, n_heads, new_seq_len, d_k]
        batch_size, n_heads, new_seq_len, d_k = k_new.shape
        
        # Validate
        assert batch_size <= self.max_batch_size, f"Batch size {batch_size} exceeds max {self.max_batch_size}"
        assert start_pos + new_seq_len <= self.max_seq_len, (
            f"Sequence length {start_pos + new_seq_len} exceeds max {self.max_seq_len}"
        )
        
        # Insert new K,V into cache
        end_pos = start_pos + new_seq_len
        
        # Update cache tensors
        # k_cache[:batch, :, start_pos:end_pos, :] shape: [batch, n_heads, new_seq_len, d_k]
        self.k_cache[:batch_size, :, start_pos:end_pos, :] = k_new
        self.v_cache[:batch_size, :, start_pos:end_pos, :] = v_new
        
        # Update current length
        self.current_length = end_pos
        
        # Return full K,V up to current length
        # k_full shape: [batch, n_heads, current_length, d_k]
        # v_full shape: [batch, n_heads, current_length, d_k]
        k_full = self.k_cache[:batch_size, :, :end_pos, :]
        v_full = self.v_cache[:batch_size, :, :end_pos, :]
        
        return k_full, v_full
    
    def get(self, batch_size: int, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached K,V.
        
        Args:
            batch_size: Batch size to retrieve
            seq_len: Sequence length to retrieve (default: current_length)
        
        Returns:
            k: Cached keys, shape [batch, n_heads, seq_len, d_k]
            v: Cached values, shape [batch, n_heads, seq_len, d_k]
        """
        if seq_len is None:
            seq_len = self.current_length
        
        # k shape: [batch, n_heads, seq_len, d_k]
        # v shape: [batch, n_heads, seq_len, d_k]
        k = self.k_cache[:batch_size, :, :seq_len, :]
        v = self.v_cache[:batch_size, :, :seq_len, :]
        
        return k, v
    
    def clear(self):
        """Clear cache."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_length = 0
        logger.debug("LayerKVCache cleared")
    
    def memory_usage_mb(self) -> float:
        """Calculate memory usage in MB."""
        # Each cache: [max_batch, n_heads, max_seq_len, d_k]
        # Two caches (K and V)
        num_elements = (
            self.max_batch_size * self.n_heads * 
            self.max_seq_len * self.d_k * 2
        )
        bytes_per_element = torch.finfo(self.dtype).bits // 8
        bytes_total = num_elements * bytes_per_element
        mb = bytes_total / (1024 * 1024)
        return mb


class KVCache:
    """
    Multi-layer Key-Value cache for transformer.
    
    Manages separate K,V caches for each transformer layer.
    
    Features:
    - Per-layer cache management
    - Automatic device handling
    - Memory tracking
    - Clear/reset functionality
    
    Args:
        config: KV cache configuration
    
    Example:
        >>> config = KVCacheConfig(
        ...     max_batch_size=4,
        ...     max_seq_len=512,
        ...     n_layers=6,
        ...     n_heads=8,
        ...     d_k=64
        ... )
        >>> cache = KVCache(config)
        >>> # Use in generation
        >>> for layer_idx in range(n_layers):
        ...     k_full, v_full = cache.update_layer(
        ...         layer_idx, k_new, v_new, start_pos
        ...     )
    """
    
    def __init__(self, config: KVCacheConfig):
        """Initialize multi-layer cache."""
        self.config = config
        
        # Create cache for each layer
        self.layer_caches: List[LayerKVCache] = []
        
        for _ in range(config.n_layers):
            layer_cache = LayerKVCache(
                max_batch_size=config.max_batch_size,
                max_seq_len=config.max_seq_len,
                n_heads=config.n_heads,
                d_k=config.d_k,
                dtype=config.dtype,
                device=config.device
            )
            self.layer_caches.append(layer_cache)
        
        logger.info(
            f"KVCache initialized: {config.n_layers} layers, "
            f"[{config.max_batch_size}, {config.n_heads}, {config.max_seq_len}, {config.d_k}], "
            f"Total memory: {self.total_memory_mb():.2f} MB"
        )
    
    def update_layer(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        start_pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for specific layer.
        
        Args:
            layer_idx: Layer index (0-indexed)
            k_new: New key states, shape [batch, n_heads, new_len, d_k]
            v_new: New value states, shape [batch, n_heads, new_len, d_k]
            start_pos: Position to insert
        
        Returns:
            k_full: Full keys including cache
            v_full: Full values including cache
        """
        return self.layer_caches[layer_idx].update(k_new, v_new, start_pos)
    
    def get_layer(
        self,
        layer_idx: int,
        batch_size: int,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached K,V for specific layer."""
        return self.layer_caches[layer_idx].get(batch_size, seq_len)
    
    def clear(self):
        """Clear all layer caches."""
        for cache in self.layer_caches:
            cache.clear()
        logger.info("KVCache cleared")
    
    def total_memory_mb(self) -> float:
        """Calculate total memory usage in MB."""
        return sum(cache.memory_usage_mb() for cache in self.layer_caches)
    
    def current_length(self, layer_idx: int = 0) -> int:
        """Get current cached sequence length."""
        return self.layer_caches[layer_idx].current_length


if __name__ == "__main__":
    # Test KV cache
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("KV Cache Test")
    print("=" * 80)
    
    # Create cache config
    config = KVCacheConfig(
        max_batch_size=2,
        max_seq_len=128,
        n_layers=4,
        n_heads=8,
        d_k=64,
        dtype=torch.float32,
        device='cpu'
    )
    
    print(f"\n1. Cache config:")
    print(f"   Max batch: {config.max_batch_size}")
    print(f"   Max seq len: {config.max_seq_len}")
    print(f"   Layers: {config.n_layers}")
    print(f"   Heads: {config.n_heads}")
    print(f"   d_k: {config.d_k}")
    
    # Create cache
    cache = KVCache(config)
    print(f"   Total memory: {cache.total_memory_mb():.2f} MB")
    
    print("\n2. Testing cache update...")
    
    # Simulate prefill phase: process 10 tokens
    batch_size = 2
    n_heads = 8
    d_k = 64
    
    # Initial K,V (prefill)
    # k shape: [batch=2, heads=8, seq_len=10, d_k=64]
    k_initial = torch.randn(batch_size, n_heads, 10, d_k)
    v_initial = torch.randn(batch_size, n_heads, 10, d_k)
    
    # Update layer 0 cache
    k_full, v_full = cache.update_layer(0, k_initial, v_initial, start_pos=0)
    
    print(f"   Prefill: {k_initial.shape} -> {k_full.shape}")
    assert k_full.shape == (batch_size, n_heads, 10, d_k)
    assert cache.current_length(0) == 10
    
    print("\n3. Testing decode phase (autoregressive)...")
    
    # Decode phase: add 1 token at a time
    for step in range(5):
        # k_new shape: [batch=2, heads=8, new_len=1, d_k=64]
        k_new = torch.randn(batch_size, n_heads, 1, d_k)
        v_new = torch.randn(batch_size, n_heads, 1, d_k)
        
        start_pos = 10 + step
        k_full, v_full = cache.update_layer(0, k_new, v_new, start_pos=start_pos)
        
        expected_len = start_pos + 1
        print(f"   Step {step}: pos={start_pos}, k_full.shape={k_full.shape}")
        assert k_full.shape == (batch_size, n_heads, expected_len, d_k)
        assert cache.current_length(0) == expected_len
    
    print("\n4. Testing multi-layer cache...")
    
    for layer_idx in range(config.n_layers):
        # k shape: [2, 8, 5, 64]
        k = torch.randn(batch_size, n_heads, 5, d_k)
        v = torch.randn(batch_size, n_heads, 5, d_k)
        
        k_full, v_full = cache.update_layer(layer_idx, k, v, start_pos=0)
        print(f"   Layer {layer_idx}: {k.shape} -> {k_full.shape}")
    
    print("\n5. Testing cache retrieval...")
    
    k_retrieved, v_retrieved = cache.get_layer(0, batch_size=2)
    print(f"   Retrieved: k.shape={k_retrieved.shape}, v.shape={v_retrieved.shape}")
    assert k_retrieved.shape[2] == cache.current_length(0)
    
    print("\n6. Testing cache clear...")
    
    cache.clear()
    assert cache.current_length(0) == 0
    print("   ✓ Cache cleared")
    
    print("\n7. Memory usage analysis...")
    
    # Calculate theoretical memory
    per_layer_mb = cache.layer_caches[0].memory_usage_mb()
    total_mb = cache.total_memory_mb()
    
    print(f"   Per layer: {per_layer_mb:.2f} MB")
    print(f"   Total ({config.n_layers} layers): {total_mb:.2f} MB")
    
    # Test with different configs
    print("\n8. Testing different cache sizes...")
    
    configs_test = [
        (1, 512, 12, 8, 64),   # Small model
        (4, 2048, 24, 16, 64), # Medium model
        (8, 4096, 32, 32, 128) # Large model
    ]
    
    for batch, seq_len, n_layers, n_heads, d_k in configs_test:
        test_config = KVCacheConfig(
            max_batch_size=batch,
            max_seq_len=seq_len,
            n_layers=n_layers,
            n_heads=n_heads,
            d_k=d_k
        )
        test_cache = KVCache(test_config)
        mem_mb = test_cache.total_memory_mb()
        print(f"   [{batch}, {seq_len}, {n_layers}, {n_heads}, {d_k}] → {mem_mb:.1f} MB")
    
    print("\n" + "=" * 80)
    print("✅ KV Cache tests completed!")
    print("=" * 80)
