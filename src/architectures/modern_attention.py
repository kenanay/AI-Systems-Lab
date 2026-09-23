"""
src/architectures/modern_attention.py

Modern Attention Innovations: Grouped-Query Attention (GQA) & RoPE
===================================================================
Bu modül LLaMA-2/3, Mistral, Gemma ve modern açık kaynaklı frontier modellerde
kullanılan iki temel dikkat yeniliğini sağlar:

1. Rotary Position Embeddings (RoPE):
   Pozisyonel bilgiyi tensörlere toplamak (additive) yerine, her 2D özellik
   boyut çiftini konum açısıyla (m * θ) döndürerek (rotasyon) uygulayan yaklaşım.
   İki token arasındaki dikkat skoru doğrudan bağıl uzaklıklarına (m - n) bağlı olur.

2. Grouped-Query Attention (GQA):
   Standart MHA'da (Multi-Head Attention) her query head için bir KV head bulunurken,
   GQA'de birden fazla Query head aynı Key/Value head'i paylaşır (ör. 8 Q head başına
   2 KV head). Bu sayede üretimde (inference) KV Cache bellek tüketimi 4x-8x azalır,
   çıkarım hızı dramatik artar.
"""

from typing import Optional, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Query ve Key vektörlerinin ardışık çift boyutlarını pozisyona bağlı açıyla döndürür.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Inverse frequencies: theta_i = base^(-2(i-1)/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cosine and sine tables
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims: [-x2, x1]."""
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x_or_seq_len: Any, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin embeddings for sequence length.
        Supports:
            rope(16) -> returns [1, 16, 1, dim]
            rope(x, seq_len=16)
        """
        if isinstance(x_or_seq_len, int):
            length = x_or_seq_len
            dtype = torch.float32
        elif torch.is_tensor(x_or_seq_len):
            length = seq_len if seq_len is not None else x_or_seq_len.shape[1]
            dtype = x_or_seq_len.dtype
        else:
            length = seq_len if seq_len is not None else 1
            dtype = torch.float32

        if length > self.cos_cached.size(0):
            self._build_cache(length)
        cos = self.cos_cached[:length, :].to(dtype=dtype).unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:length, :].to(dtype=dtype).unsqueeze(0).unsqueeze(2)
        return cos, sin

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE rotation to query or key tensor.
        
        Args:
            x: shape [B, H, T, D_head]
            cos, sin: shape [1, T, 1, D_head] or [T, D_head]
        """
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
        elif cos.dim() == 4 and cos.size(1) == x.size(2) and cos.size(2) == 1:
            # Format [1, T, 1, D] -> transpose to [1, 1, T, D] for broadcasting with [B, H, T, D]
            cos = cos.transpose(1, 2)
            sin = sin.transpose(1, 2)

        return (x * cos) + (self._rotate_half(x) * sin)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    
    n_heads query head'i barındırırken, n_kv_heads (daha az) sayıda
    Key/Value head'i kullanır.
    
    MHA: n_heads == n_kv_heads
    MQA: n_kv_heads == 1
    GQA: 1 < n_kv_heads < n_heads
    """

    def __init__(
        self,
        d_model: int,
        n_heads: Optional[int] = None,
        n_kv_heads: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        use_rope: bool = True
    ):
        super().__init__()
        actual_heads = num_heads if num_heads is not None else n_heads
        actual_kv_heads = num_kv_heads if num_kv_heads is not None else n_kv_heads
        if actual_heads is None:
            actual_heads = 8
        if actual_kv_heads is None:
            actual_kv_heads = 2

        assert d_model % actual_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({actual_heads})"
        assert actual_heads % actual_kv_heads == 0, f"n_heads ({actual_heads}) must be divisible by n_kv_heads ({actual_kv_heads})"

        self.d_model = d_model
        self.n_heads = actual_heads
        self.n_kv_heads = actual_kv_heads
        self.d_k = d_model // actual_heads
        self.num_queries_per_kv = actual_heads // actual_kv_heads
        self.use_rope = use_rope

        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryEmbedding(dim=self.d_k, max_seq_len=max_seq_len)
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for GQA.
        
        Args:
            x: Input tensor, shape [B, T, D]
            mask: Attention mask
            is_causal: Whether to apply causal masking
        """
        batch_size, seq_len, _ = x.shape

        # Projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Apply RoPE if enabled
        if self.rope is not None:
            cos, sin = self.rope(x, seq_len)
            q = self.rope.apply_rope(q, cos, sin)
            k = self.rope.apply_rope(k, cos, sin)

        # Repeat KV heads to match query heads count
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Scaled dot-product attention
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal and mask is None
        )

        # Combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output, None
