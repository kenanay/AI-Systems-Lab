"""
src/model/gpt.py

GPT-style Decoder-only Transformer Model

Bu modül complete GPT-style language model'i içerir.
GPT (Generative Pre-trained Transformer) decoder-only bir mimaridır.

Architecture Pipeline:
    Token IDs [batch, seq_len]
    -> Token + Positional Embeddings [batch, seq_len, d_model]
    -> Transformer Decoder (N layers) [batch, seq_len, d_model]
    -> Output Projection (to vocab) [batch, seq_len, vocab_size]
    -> Logits [batch, seq_len, vocab_size]

Model autoregressive: Her token sadece kendinden öncekilere attend edebilir.
Causal masking ile sağlanır.

Kaynaklar:
    - Language Models are Unsupervised Multitask Learners (GPT-2)
    - Improving Language Understanding by Generative Pre-Training (GPT-1)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import logging

from src.model.embeddings import TransformerEmbedding, create_embedding_layer
from src.model.transformer_block import TransformerDecoder, create_transformer_decoder
from src.model.attention import create_causal_mask

logger = logging.getLogger(__name__)


@dataclass
class GPTConfig:
    """
    GPT Model Configuration
    
    Bu dataclass model'in tüm hyperparameter'larını tutar.
    
    Attributes:
        vocab_size: Vocabulary size (number of tokens)
        max_seq_len: Maximum sequence length (context window)
        d_model: Model dimension (embedding dimension)
        n_layers: Number of transformer decoder layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension (typically 4 * d_model)
        dropout: Dropout rate
        activation: Activation function for FFN
        use_gated_ffn: Use gated FFN (GLU-style)
        tie_embeddings: Tie input and output embeddings (weight sharing)
    """
    vocab_size: int = 8000
    max_seq_len: int = 512
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    activation: str = 'gelu'
    use_gated_ffn: bool = False
    tie_embeddings: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
    
    @property
    def d_k(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'activation': self.activation,
            'use_gated_ffn': self.use_gated_ffn,
            'tie_embeddings': self.tie_embeddings,
        }


class GPTModel(nn.Module):
    """
    GPT-style Decoder-only Transformer Language Model
    
    Complete autoregressive language model. Token sequence alır,
    her pozisyon için next-token prediction yapar.
    
    Architecture:
        1. Token + Positional Embeddings
        2. Transformer Decoder (N layers)
        3. Output Projection (Linear: d_model → vocab_size)
        4. (Optional) Tie embeddings: output projection uses embedding weights
    
    Args:
        config: GPTConfig instance
    
    Shape:
        Input: token_ids [batch, seq_len]
        Output: logits [batch, seq_len, vocab_size]
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model
        
        # Token + Positional Embeddings
        self.embeddings = create_embedding_layer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            padding_idx=0  # Assuming 0 is padding token
        )
        
        # Transformer Decoder (stack of N blocks)
        self.decoder = create_transformer_decoder(
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            use_gated_ffn=config.use_gated_ffn
        )
        
        # Output projection: d_model → vocab_size
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Tie embeddings (share weights between input embeddings and output projection)
        if config.tie_embeddings:
            # Output projection uses embedding weights
            # This reduces parameters and often improves performance
            self.output_projection.weight = self.embeddings.get_token_embedding_weight()
            logger.info("Embeddings tied: input and output share weights")
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"GPT Model initialized: {config.n_layers} layers, "
            f"{config.d_model} dim, {config.n_heads} heads, "
            f"{n_params:,} parameters"
        )
    
    def _init_weights(self):
        """
        Initialize model weights.
        
        - Linear layers: Xavier/Glorot uniform
        - Embeddings: Normal(0, 0.02)
        - LayerNorm: weight=1, bias=0 (default)
        """
        def _init_module(module):
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Normal initialization for embeddings
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    # Zero out padding embedding
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm default init (weight=1, bias=0)
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        self.apply(_init_module)
        logger.debug("Weights initialized: Xavier for Linear, Normal(0,0.02) for Embeddings")
    
    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        GPT forward pass.
        
        Args:
            token_ids: Input token IDs, shape [B, T]
            mask: Causal attention mask, shape [1, 1, T, T]
                  If None, created automatically
        
        Returns:
            logits: Output logits, shape [B, T, V]
            attention_weights: List of attention weights from each layer
                              Each: [B, H, T, T]
        
        Pipeline:
            Token IDs [B, T]
            -> Embeddings [B, T, D]
            -> Transformer Decoder [B, T, D]
            -> Output Projection [B, T, V]
            -> Logits [B, T, V]
            
        Shape notation:
            B = batch_size
            T = seq_len
            D = d_model
            V = vocab_size
            H = n_heads
        """
        # token_ids shape: [B, T]
        batch_size, seq_len = token_ids.shape
        
        # Validate sequence length
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
        )
        
        # Create causal mask if not provided
        if mask is None:
            # mask shape: [1, 1, T, T]
            mask = create_causal_mask(seq_len, token_ids.device)
        
        # Step 1: Token + Positional Embeddings
        # embeddings shape: [B, T, D]
        embeddings = self.embeddings(token_ids)
        
        # Step 2: Transformer Decoder
        # decoder_output shape: [B, T, D]
        # attention_weights: List of [B, H, T, T], length = n_layers
        decoder_output, attention_weights = self.decoder(embeddings, mask)
        
        # Step 3: Output Projection
        # decoder_output shape: [B, T, D]
        # output_projection: [D, V]
        # logits shape: [B, T, V]
        logits = self.output_projection(decoder_output)
        
        return logits, attention_weights
    
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Prompt'tan başlayarak token-by-token generation yapar.
        
        Args:
            prompt_ids: Initial token IDs, shape [B, T_prompt]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k logits (None = no filtering)
            top_p: Keep top logits with cumulative prob >= p (nucleus sampling)
        
        Returns:
            Generated token IDs, shape [B, T_prompt + max_new_tokens]
        
        Example:
            >>> prompt = torch.tensor([[1, 2, 3]])  # [1, 3]
            >>> output = model.generate(prompt, max_new_tokens=10)
            >>> output.shape
            torch.Size([1, 13])
        """
        self.eval()
        
        # generated shape: [B, T]
        generated = prompt_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get sequence length
                seq_len = generated.size(1)
                
                # Truncate if exceeds max_seq_len (use last max_seq_len tokens)
                if seq_len > self.max_seq_len:
                    input_ids = generated[:, -self.max_seq_len:]
                else:
                    input_ids = generated
                
                # Forward pass
                # logits shape: [B, T, V]
                logits, _ = self.forward(input_ids)
                
                # Get logits for last position
                # next_token_logits shape: [B, V]
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    # Keep only top k logits
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    # Set other logits to -inf
                    next_token_logits[next_token_logits < top_k_logits[:, -1:]] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative prob > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Set removed logits to -inf (batch-aware indexing)
                    # Create batch indices
                    batch_indices = torch.arange(next_token_logits.size(0), device=next_token_logits.device)
                    batch_indices = batch_indices.unsqueeze(1).expand_as(sorted_indices)
                    
                    # Use advanced indexing
                    next_token_logits[batch_indices[sorted_indices_to_remove], sorted_indices[sorted_indices_to_remove]] = float('-inf')
                
                # Sample from distribution
                # probs shape: [B, V]
                probs = torch.softmax(next_token_logits, dim=-1)
                # next_token shape: [B, 1]
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                # generated shape: [B, T+1]
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get number of parameters.
        
        Args:
            non_embedding: If True, exclude embedding parameters
        
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if non_embedding:
            # Subtract embedding parameters
            n_params -= self.embeddings.token_embedding.embedding.weight.numel()
            n_params -= self.embeddings.positional_encoding.position_embedding.weight.numel()
        
        return n_params


def create_gpt_model(config: GPTConfig) -> GPTModel:
    """
    Factory function to create GPT model.
    
    Args:
        config: GPTConfig instance
    
    Returns:
        GPTModel instance
    
    Example:
        >>> config = GPTConfig(
        ...     vocab_size=8000,
        ...     max_seq_len=512,
        ...     d_model=256,
        ...     n_layers=6,
        ...     n_heads=8
        ... )
        >>> model = create_gpt_model(config)
        >>> token_ids = torch.randint(0, 8000, (2, 100))
        >>> logits, attn_weights = model(token_ids)
        >>> logits.shape
        torch.Size([2, 100, 8000])
    """
    return GPTModel(config)


if __name__ == "__main__":
    # Test GPT model
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("GPT Model Test")
    print("=" * 80)
    
    # Create configuration
    config = GPTConfig(
        vocab_size=1000,
        max_seq_len=128,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
        tie_embeddings=True
    )
    
    print(f"\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print(f"  d_k (per head): {config.d_k}")
    
    # Create model
    print(f"\n{'='*80}")
    print("[1] Model Creation")
    print(f"{'='*80}")
    
    model = create_gpt_model(config)
    print(f"✅ Model created")
    
    # Count parameters
    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)
    
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Non-embedding parameters: {non_emb_params:,}")
    print(f"  Embedding parameters: {total_params - non_emb_params:,}")
    
    # Test forward pass
    print(f"\n{'='*80}")
    print("[2] Forward Pass Test")
    print(f"{'='*80}")
    
    batch_size = 2
    seq_len = 50
    
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"  Input token_ids shape: {token_ids.shape}")
    
    logits, attention_weights = model(token_ids)
    
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Number of attention weight tensors: {len(attention_weights)}")
    print(f"  Each attention weight shape: {attention_weights[0].shape}")
    print(f"  Logits mean: {logits.mean():.4f}")
    print(f"  Logits std: {logits.std():.4f}")
    
    # Validate shapes
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert len(attention_weights) == config.n_layers
    print(f"✅ Shape validation passed")
    
    # Test generation
    print(f"\n{'='*80}")
    print("[3] Text Generation Test")
    print(f"{'='*80}")
    
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    print(f"  Prompt shape: {prompt.shape}")
    
    generated = model.generate(
        prompt_ids=prompt,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50
    )
    
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated tokens: {generated[0].tolist()[:15]}...")
    
    assert generated.shape == (1, 30)  # 10 prompt + 20 new
    print(f"✅ Generation test passed")
    
    # Test batch generation
    print(f"\n{'='*80}")
    print("[4] Batch Generation Test")
    print(f"{'='*80}")
    
    batch_prompt = torch.randint(0, config.vocab_size, (3, 5))
    print(f"  Batch prompt shape: {batch_prompt.shape}")
    
    batch_generated = model.generate(
        prompt_ids=batch_prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_p=0.9
    )
    
    print(f"  Batch generated shape: {batch_generated.shape}")
    assert batch_generated.shape == (3, 15)  # 5 prompt + 10 new
    print(f"✅ Batch generation test passed")
    
    # Test with different temperatures
    print(f"\n{'='*80}")
    print("[5] Temperature Comparison")
    print(f"{'='*80}")
    
    test_prompt = torch.randint(0, config.vocab_size, (1, 5))
    
    for temp in [0.5, 1.0, 1.5]:
        gen = model.generate(
            prompt_ids=test_prompt,
            max_new_tokens=10,
            temperature=temp
        )
        print(f"  Temperature {temp}: {gen[0].tolist()}")
    
    print(f"\n" + "=" * 80)
    print("✅ All tests PASSED!")
    print("=" * 80)
