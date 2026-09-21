"""
src/model/embeddings.py

Embeddings & Positional Encoding

Bu modül transformer model için embedding katmanlarını içerir:
- Token embeddings: Vocabulary'deki her token için learned vector
- Positional encoding: Sequence pozisyon bilgisi için learned encoding
- Dropout regularization

Embeddings, token ID'lerini continuous vector space'e map eder.
Positional encoding, sequence içindeki pozisyon bilgisini ekler.
"""

import torch
import torch.nn as nn
import math
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer
    
    Her token ID'yi d_model boyutunda bir vektöre map eder.
    
    Args:
        vocab_size: Vocabulary boyutu (token sayısı)
        d_model: Embedding dimension (model dimension)
        padding_idx: Padding token ID (optional)
    
    Shape:
        Input: [batch_size, seq_len] - Token IDs
        Output: [batch_size, seq_len, d_model] - Token embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        # Embedding lookup table
        # Shape: [vocab_size, d_model]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )
        
        # Embedding scaling factor (GPT/BERT style)
        # Scale embeddings by sqrt(d_model) for better gradient flow
        self.scale = math.sqrt(d_model)
        
        logger.debug(
            f"TokenEmbedding initialized: vocab_size={vocab_size}, "
            f"d_model={d_model}, scale={self.scale:.3f}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Token IDs'i embeddings'lere map et.
        
        Args:
            x: Token IDs, shape [B, T]
        
        Returns:
            Token embeddings, shape [B, T, D]
            
        Shape notation:
            B = batch_size
            T = sequence_length
            D = d_model
        """
        # x shape: [B, T]
        # embedding(x) shape: [B, T, D]
        embeddings = self.embedding(x)
        
        # Scale embeddings
        # scaled shape: [B, T, D]
        scaled_embeddings = embeddings * self.scale
        
        return scaled_embeddings
    
    def get_embedding_weight(self) -> torch.Tensor:
        """
        Embedding weight matrix'i döndür (output projection için).
        
        Returns:
            Embedding weights, shape [vocab_size, d_model]
        """
        return self.embedding.weight


class PositionalEncoding(nn.Module):
    """
    Learnable Positional Encoding
    
    Her pozisyon için learned encoding. Sinusoidal yerine learnable kullanıyoruz
    çünkü daha flexible ve genelde daha iyi sonuç veriyor.
    
    Args:
        max_seq_len: Maximum sequence length
        d_model: Embedding dimension
        dropout: Dropout rate
    
    Shape:
        Input: [batch_size, seq_len, d_model] - Token embeddings
        Output: [batch_size, seq_len, d_model] - Embeddings + positional info
    """
    
    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Learnable positional embeddings
        # Shape: [max_seq_len, d_model]
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)
        
        logger.debug(
            f"PositionalEncoding initialized: max_seq_len={max_seq_len}, "
            f"d_model={d_model}, dropout={dropout}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Token embeddings'lere positional information ekle.
        
        Args:
            x: Token embeddings, shape [B, T, D]
        
        Returns:
            Embeddings with positional info, shape [B, T, D]
            
        Shape notation:
            B = batch_size
            T = sequence_length
            D = d_model
        """
        # x shape: [B, T, D]
        batch_size, seq_len, d_model = x.shape
        
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
        )
        
        # Position indices: [0, 1, 2, ..., T-1]
        # positions shape: [T]
        positions = torch.arange(seq_len, device=x.device)
        
        # Position embeddings
        # pos_emb shape: [T, D]
        pos_emb = self.position_embedding(positions)
        
        # Add positional embeddings to token embeddings
        # Broadcasting: [B, T, D] + [T, D] -> [B, T, D]
        x_with_pos = x + pos_emb
        
        # Apply dropout
        # output shape: [B, T, D]
        output = self.dropout(x_with_pos)
        
        return output


class TransformerEmbedding(nn.Module):
    """
    Complete Transformer Embedding Layer
    
    Token embeddings + Positional encoding + Dropout
    
    Bu modül, transformer model'in input embedding katmanıdır.
    Token ID'lerini alır, embedding'lere map eder, positional bilgi ekler.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        padding_idx: Padding token ID (optional)
    
    Shape:
        Input: [batch_size, seq_len] - Token IDs
        Output: [batch_size, seq_len, d_model] - Complete embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
        padding_idx: Optional[int] = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=padding_idx
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            max_seq_len=max_seq_len,
            d_model=d_model,
            dropout=dropout
        )
        
        logger.info(
            f"TransformerEmbedding initialized: vocab={vocab_size}, "
            f"d_model={d_model}, max_seq_len={max_seq_len}"
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Token IDs'i complete embeddings'e dönüştür.
        
        Args:
            token_ids: Input token IDs, shape [B, T]
        
        Returns:
            Complete embeddings, shape [B, T, D]
            
        Pipeline:
            Token IDs [B, T]
            -> Token Embeddings [B, T, D]
            -> + Positional Encoding [B, T, D]
            -> Dropout [B, T, D]
        """
        # token_ids shape: [B, T]
        
        # Step 1: Token embeddings
        # token_emb shape: [B, T, D]
        token_emb = self.token_embedding(token_ids)
        
        # Step 2: Add positional encoding + dropout
        # output shape: [B, T, D]
        output = self.positional_encoding(token_emb)
        
        return output
    
    def get_token_embedding_weight(self) -> torch.Tensor:
        """
        Token embedding weight matrix (for tied embeddings).
        
        Returns:
            Token embedding weights, shape [vocab_size, d_model]
        """
        return self.token_embedding.get_embedding_weight()


def create_embedding_layer(
    vocab_size: int,
    d_model: int,
    max_seq_len: int,
    dropout: float = 0.1,
    padding_idx: Optional[int] = 0
) -> TransformerEmbedding:
    """
    Factory function to create embedding layer.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        padding_idx: Padding token ID
    
    Returns:
        TransformerEmbedding instance
    
    Example:
        >>> embedding = create_embedding_layer(
        ...     vocab_size=8000,
        ...     d_model=256,
        ...     max_seq_len=512
        ... )
        >>> token_ids = torch.randint(0, 8000, (2, 100))  # [B=2, T=100]
        >>> embeddings = embedding(token_ids)  # [2, 100, 256]
    """
    return TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        dropout=dropout,
        padding_idx=padding_idx
    )


if __name__ == "__main__":
    # Test embedding layers
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Embedding Layers Test")
    print("=" * 80)
    
    # Parameters
    vocab_size = 1000
    d_model = 256
    max_seq_len = 512
    batch_size = 2
    seq_len = 100
    
    print(f"\nParameters:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    
    # Create embedding layer
    print(f"\n[1] Creating embedding layer...")
    embedding = create_embedding_layer(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        dropout=0.1
    )
    print(f"✅ Embedding layer created")
    
    # Create dummy input
    print(f"\n[2] Creating dummy token IDs...")
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"  token_ids shape: {token_ids.shape}")
    print(f"  token_ids range: [{token_ids.min()}, {token_ids.max()}]")
    
    # Forward pass
    print(f"\n[3] Forward pass...")
    embeddings = embedding(token_ids)
    print(f"  embeddings shape: {embeddings.shape}")
    print(f"  embeddings mean: {embeddings.mean():.4f}")
    print(f"  embeddings std: {embeddings.std():.4f}")
    
    # Validate shape
    expected_shape = (batch_size, seq_len, d_model)
    assert embeddings.shape == expected_shape, (
        f"Shape mismatch: expected {expected_shape}, got {embeddings.shape}"
    )
    print(f"✅ Shape validation passed")
    
    # Test token embedding weight
    print(f"\n[4] Token embedding weight...")
    token_weight = embedding.get_token_embedding_weight()
    print(f"  token_weight shape: {token_weight.shape}")
    print(f"  Expected: [{vocab_size}, {d_model}]")
    assert token_weight.shape == (vocab_size, d_model)
    print(f"✅ Token weight shape correct")
    
    print(f"\n" + "=" * 80)
    print("✅ All tests PASSED!")
    print("=" * 80)
