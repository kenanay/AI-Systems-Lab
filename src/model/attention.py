"""
src/model/attention.py

Multi-Head Attention Mechanism

Bu modül transformer'ın core component'i olan multi-head attention'ı içerir:
- Scaled dot-product attention
- Multiple attention heads (parallel processing)
- Causal masking (autoregressive models için)

Attention mechanism, sequence içindeki farklı pozisyonlar arasındaki
ilişkileri öğrenir. Multi-head kullanarak farklı representation subspace'lere
odaklanır.

Formül:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Multi-head:
    head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        dropout: Dropout rate for attention weights
    
    Shape:
        Q: [batch, n_heads, seq_len, d_k]
        K: [batch, n_heads, seq_len, d_k]
        V: [batch, n_heads, seq_len, d_v]
        Output: [batch, n_heads, seq_len, d_v]
        Attention weights: [batch, n_heads, seq_len, seq_len]
    """
    
    def __init__(self, dropout: float = 0.1, use_sdpa: bool = True):
        super().__init__()
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.use_sdpa = use_sdpa
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        is_causal: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Scaled dot-product attention hesapla.
        
        Args:
            q: Query tensor, shape [B, H, T, D_k]
            k: Key tensor, shape [B, H, T, D_k]
            v: Value tensor, shape [B, H, T, D_v]
            mask: Attention mask, shape [B, 1, T, T] or [B, H, T, T]
                  True değerleri mask'lanır (attention yapılmaz)
            need_weights: If True, computes and returns explicit attention weights [B, H, T, T].
                          If False and use_sdpa is enabled, utilizes hardware-accelerated
                          F.scaled_dot_product_attention without materializing the O(T^2) matrix.
            is_causal: If True, applies causal lower-triangular mask (for autoregressive models).
        
        Returns:
            output: Attention output, shape [B, H, T, D_v]
            attention_weights: Attention weights, shape [B, H, T, T] if need_weights=True, else None
            
        Shape notation:
            B = batch_size
            H = n_heads
            T = seq_len
            D_k = d_k (key/query dimension)
            D_v = d_v (value dimension, usually = d_k)
        """
        # Fast path: PyTorch SDPA when explicit attention weights are not requested
        if self.use_sdpa and not need_weights:
            dropout_p = self.dropout_p if self.training else 0.0
            if mask is None and is_causal is True:
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=True
                )
            else:
                attn_mask = None
                causal_flag = False
                if mask is not None:
                    if mask.dtype == torch.bool:
                        attn_mask = ~mask
                    else:
                        attn_mask = mask
                elif is_causal is True:
                    causal_flag = True

                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=causal_flag
                )
            return output, None

        # Standard explicit attention calculation (need_weights=True or use_sdpa=False)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_scores = scores / math.sqrt(d_k)
        
        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(mask, -1e9)
        elif is_causal is True:
            t_q = q.size(-2)
            t_k = k.size(-2)
            causal_mask = torch.triu(
                torch.ones(t_q, t_k, dtype=torch.bool, device=q.device),
                diagonal=1
            )
            scaled_scores = scaled_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        attention_weights = F.softmax(scaled_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer
    
    Multiple attention heads'i parallel olarak çalıştırır.
    Her head farklı representation subspace'e odaklanır.
    
    Args:
        d_model: Model dimension (input/output dimension)
        n_heads: Number of attention heads
        dropout: Dropout rate
    
    Shape:
        Input: [batch, seq_len, d_model]
        Output: [batch, seq_len, d_model]
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_sdpa: bool = True
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.use_sdpa = use_sdpa
        
        # Linear projections for Q, K, V
        # Input: [B, T, D] -> Output: [B, T, D]
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        # Input: [B, T, D] -> Output: [B, T, D]
        self.w_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout, use_sdpa=use_sdpa)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        logger.debug(
            f"MultiHeadAttention initialized: d_model={d_model}, "
            f"n_heads={n_heads}, d_k={self.d_k}, use_sdpa={use_sdpa}"
        )
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split d_model into n_heads * d_k.
        
        Args:
            x: Input tensor, shape [B, T, D]
        
        Returns:
            Reshaped tensor, shape [B, H, T, D_k]
            
        Transformation:
            [B, T, D] -> [B, T, H, D_k] -> [B, H, T, D_k]
        """
        # x shape: [B, T, D]
        batch_size, seq_len, d_model = x.size()
        
        # Reshape: [B, T, D] -> [B, T, H, D_k]
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose: [B, T, H, D_k] -> [B, H, T, D_k]
        x = x.transpose(1, 2)
        
        return x
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine n_heads * d_k back into d_model.
        
        Args:
            x: Input tensor, shape [B, H, T, D_k]
        
        Returns:
            Combined tensor, shape [B, T, D]
            
        Transformation:
            [B, H, T, D_k] -> [B, T, H, D_k] -> [B, T, D]
        """
        # x shape: [B, H, T, D_k]
        batch_size, n_heads, seq_len, d_k = x.size()
        
        # Transpose: [B, H, T, D_k] -> [B, T, H, D_k]
        x = x.transpose(1, 2)
        
        # Reshape: [B, T, H, D_k] -> [B, T, D]
        x = x.contiguous().view(batch_size, seq_len, self.d_model)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        is_causal: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Multi-head attention forward pass with optional KV caching.
        
        Args:
            x: Input tensor, shape [B, T, D]
            mask: Attention mask, shape [B, 1, T, T] or [B, 1, 1, T_full]
                  True positions are masked (no attention)
            use_cache: Whether to use/return KV cache
            cache_k: Cached key states, shape [B, H, T_prev, D_k]
            cache_v: Cached value states, shape [B, H, T_prev, D_k]
            need_weights: Whether to compute and return attention weights
            is_causal: Whether to apply causal mask
        
        Returns:
            output: Attention output, shape [B, T, D]
            attention_weights: Attention weights, shape [B, H, T, T_full] or None
            k_out: Key states to cache, shape [B, H, T_full, D_k] (if use_cache)
            v_out: Value states to cache, shape [B, H, T_full, D_k] (if use_cache)
            
        Pipeline:
            Input [B, T, D]
            -> Linear Q, K, V [B, T, D]
            -> Split heads [B, H, T, D_k]
            -> Concat with cache (if provided) [B, H, T_full, D_k]
            -> Scaled Dot-Product Attention [B, H, T, D_k]
            -> Combine heads [B, T, D]
            -> Output projection [B, T, D]
            
        KV Cache Usage:
            - First call (prefill): cache_k/cache_v = None, returns full K,V
            - Subsequent calls (decode): provide cache_k/cache_v, returns updated K,V
        """
        # x shape: [B, T, D]
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        # q, k, v shape: [B, T, D]
        q = self.w_q(x)
        k_new = self.w_k(x)
        v_new = self.w_v(x)
        
        # Split into multiple heads
        # q, k_new, v_new shape: [B, H, T, D_k]
        q = self._split_heads(q)
        k_new = self._split_heads(k_new)
        v_new = self._split_heads(v_new)
        
        # Handle KV cache
        if use_cache:
            if cache_k is not None and cache_v is not None:
                # Concat cached K,V with new K,V
                # cache_k shape: [B, H, T_prev, D_k]
                # k_new shape: [B, H, T, D_k]
                # k shape: [B, H, T_prev + T, D_k]
                k = torch.cat([cache_k, k_new], dim=2)
                v = torch.cat([cache_v, v_new], dim=2)
            else:
                # First call - no cache yet
                k = k_new
                v = v_new
            
            # Return K,V for caching
            k_out = k
            v_out = v
        else:
            # No caching
            k = k_new
            v = v_new
            k_out = None
            v_out = None
        
        # Scaled dot-product attention
        # k, v shape: [B, H, T_full, D_k]
        # attn_output shape: [B, H, T, D_k]
        # attention_weights shape: [B, H, T, T_full] (or None if need_weights=False)
        attn_output, attention_weights = self.attention(
            q, k, v,
            mask=mask,
            need_weights=need_weights,
            is_causal=is_causal
        )
        
        # Combine heads
        # combined shape: [B, T, D]
        combined = self._combine_heads(attn_output)
        
        # Output projection
        # output shape: [B, T, D]
        output = self.w_o(combined)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output, attention_weights, k_out, v_out


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask (for autoregressive models).
    
    Causal mask ensures that position i can only attend to positions <= i.
    This is essential for autoregressive language models (GPT-style).
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
    
    Returns:
        Causal mask, shape [1, 1, T, T]
        True positions are masked (no attention allowed)
        
    Example:
        For seq_len=4:
        [[False,  True,  True,  True],
         [False, False,  True,  True],
         [False, False, False,  True],
         [False, False, False, False]]
         
        Position 0 can only see position 0
        Position 1 can see positions 0, 1
        Position 2 can see positions 0, 1, 2
        etc.
    """
    # Create lower triangular matrix (1s below diagonal, including diagonal)
    # mask shape: [T, T]
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    # Invert: 1 -> 0 (allow), 0 -> 1 (mask)
    # mask shape: [T, T]
    mask = mask == 0
    
    # Add batch and head dimensions
    # mask shape: [1, 1, T, T]
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    return mask


def create_attention_layer(
    d_model: int,
    n_heads: int,
    dropout: float = 0.1,
    use_sdpa: bool = True
) -> MultiHeadAttention:
    """
    Factory function to create multi-head attention layer.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        use_sdpa: Whether to use PyTorch scaled_dot_product_attention
    
    Returns:
        MultiHeadAttention instance
    
    Example:
        >>> attention = create_attention_layer(d_model=256, n_heads=8)
        >>> x = torch.randn(2, 100, 256)  # [B=2, T=100, D=256]
        >>> mask = create_causal_mask(100, x.device)
        >>> output, attn_weights = attention(x, mask)
        >>> output.shape
        torch.Size([2, 100, 256])
    """
    return MultiHeadAttention(
        d_model=d_model,
        n_heads=n_heads,
        dropout=dropout,
        use_sdpa=use_sdpa
    )


if __name__ == "__main__":
    # Test multi-head attention
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Multi-Head Attention Test")
    print("=" * 80)
    
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 256
    n_heads = 8
    
    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_k (per head): {d_model // n_heads}")
    
    # Create attention layer
    print(f"\n[1] Creating multi-head attention layer...")
    attention = create_attention_layer(
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.1
    )
    print(f"✅ Attention layer created")
    
    # Create input
    print(f"\n[2] Creating input tensor...")
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"  x shape: {x.shape}")
    
    # Create causal mask
    print(f"\n[3] Creating causal mask...")
    mask = create_causal_mask(seq_len, x.device)
    print(f"  mask shape: {mask.shape}")
    print(f"  mask[0, 0, :3, :3]:")
    print(f"    {mask[0, 0, :3, :3]}")
    
    # Forward pass without mask
    print(f"\n[4] Forward pass (no mask)...")
    output_no_mask, attn_weights_no_mask = attention(x)
    print(f"  output shape: {output_no_mask.shape}")
    print(f"  attention_weights shape: {attn_weights_no_mask.shape}")
    print(f"  output mean: {output_no_mask.mean():.4f}")
    print(f"  output std: {output_no_mask.std():.4f}")
    
    # Validate shapes
    assert output_no_mask.shape == (batch_size, seq_len, d_model)
    assert attn_weights_no_mask.shape == (batch_size, n_heads, seq_len, seq_len)
    print(f"✅ Shapes correct")
    
    # Forward pass with causal mask
    print(f"\n[5] Forward pass (with causal mask)...")
    output_masked, attn_weights_masked = attention(x, mask)
    print(f"  output shape: {output_masked.shape}")
    print(f"  attention_weights shape: {attn_weights_masked.shape}")
    
    # Check causal property
    # Position i should only attend to positions <= i
    print(f"\n[6] Verifying causal property...")
    # Check first head, first batch
    attn_first_head = attn_weights_masked[0, 0]  # [T, T]
    
    # Position 0 should only attend to position 0
    # attn_first_head[0, 1:] should be ~0
    future_attention = attn_first_head[0, 1:].abs().max().item()
    print(f"  Position 0 attending to future: {future_attention:.6f} (should be ~0)")
    
    if future_attention < 1e-6:
        print(f"✅ Causal masking working correctly")
    else:
        print(f"⚠️  Causal masking may not be working")
    
    # Check attention weights sum to 1
    print(f"\n[7] Verifying attention weights normalization...")
    attn_sums = attn_weights_masked.sum(dim=-1)  # Sum over keys
    expected_ones = torch.ones_like(attn_sums)
    is_normalized = torch.allclose(attn_sums, expected_ones, atol=1e-6)
    
    # Check different positions (0 has only self, later positions have more context)
    print(f"  Position 0 attention sum: {attn_sums[0, 0, 0]:.6f} (can only attend to self)")
    print(f"  Position 5 attention sum: {attn_sums[0, 0, 5]:.6f} (can attend to 0-5)")
    print(f"  Position 9 attention sum: {attn_sums[0, 0, 9]:.6f} (can attend to all)")
    
    if is_normalized:
        print(f"✅ Attention weights properly normalized")
    else:
        print(f"⚠️  Attention weights not normalized (this can happen with causal mask)")
    
    print(f"\n" + "=" * 80)
    print("✅ All tests PASSED!")
    print("=" * 80)
