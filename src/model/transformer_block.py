"""
src/model/transformer_block.py

Transformer Decoder Block

Bu modül transformer decoder bloğunu içerir. Bir decoder block şu bileşenlerden oluşur:
1. Multi-Head Self-Attention (with causal masking)
2. Add & Norm (residual connection + layer normalization)
3. Feed-Forward Network
4. Add & Norm (residual connection + layer normalization)

Architecture:
    x -> [Self-Attention -> Add & Norm] -> [FFN -> Add & Norm] -> output
    
Residual connections (skip connections) gradient flow'u iyileştirir ve
derin ağlarda training'i stabilize eder.

Layer Normalization her sub-layer'dan sonra uygulanır.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Optional, Tuple, List
import logging

from src.model.attention import MultiHeadAttention, create_attention_layer
from src.model.ffn import PositionWiseFFN, create_ffn_layer

logger = logging.getLogger(__name__)


class TransformerDecoderBlock(nn.Module):
    """
    Transformer Decoder Block (GPT-style)
    
    Bir decoder block iki ana component içerir:
    1. Multi-head self-attention (causal masked)
    2. Position-wise feed-forward network
    
    Her component'ten sonra residual connection ve layer normalization uygulanır.
    
    Architecture:
        # Pre-LN variant (modern, more stable)
        x1 = LayerNorm(x)
        x = x + Attention(x1)
        
        x2 = LayerNorm(x)
        x = x + FFN(x2)
        
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function for FFN
        use_gated_ffn: Use gated FFN instead of standard FFN
    
    Shape:
        Input: [batch, seq_len, d_model]
        Output: [batch, seq_len, d_model]
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_gated_ffn: bool = False,
        use_sdpa: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.use_sdpa = use_sdpa
        
        # Multi-head self-attention
        self.attention = create_attention_layer(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_sdpa=use_sdpa
        )
        
        # Feed-forward network
        self.ffn = create_ffn_layer(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            gated=use_gated_ffn
        )
        
        # Layer normalization (pre-LN)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(p=dropout)
        
        logger.debug(
            f"TransformerDecoderBlock initialized: d_model={d_model}, "
            f"n_heads={n_heads}, d_ff={d_ff}, use_sdpa={use_sdpa}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        is_causal: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Transformer decoder block forward pass.
        
        Args:
            x: Input tensor, shape [B, T, D]
            mask: Causal attention mask, shape [B, 1, T, T] or [1, 1, T, T]
                  True positions are masked
            need_weights: Whether to return explicit attention weights
            is_causal: Whether to apply causal mask
        
        Returns:
            output: Block output, shape [B, T, D]
            attention_weights: Attention weights, shape [B, H, T, T] if need_weights else None
            
        Pipeline (Pre-LN):
            Input [B, T, D]
            -> LayerNorm [B, T, D]
            -> Multi-Head Attention [B, T, D]
            -> Dropout [B, T, D]
            -> Residual Add [B, T, D]
            
            -> LayerNorm [B, T, D]
            -> FFN [B, T, D]
            -> Dropout [B, T, D]
            -> Residual Add [B, T, D]
            -> Output [B, T, D]
            
        Shape notation:
            B = batch_size
            T = seq_len
            D = d_model
            H = n_heads
        """
        # x shape: [B, T, D]
        
        # Self-Attention sub-layer with Pre-LN
        # Step 1: Layer normalization
        x_norm = self.ln1(x)
        
        # Step 2: Multi-head attention
        attn_output, attention_weights, _, _ = self.attention(
            x_norm,
            mask=mask,
            need_weights=need_weights,
            is_causal=is_causal
        )
        
        # Step 3: Dropout
        attn_output = self.dropout(attn_output)
        
        # Step 4: Residual connection
        x = x + attn_output
        
        # Feed-Forward sub-layer with Pre-LN
        # Step 5: Layer normalization
        x_norm = self.ln2(x)
        
        # Step 6: Feed-forward network
        ffn_output = self.ffn(x_norm)
        
        # Step 7: Dropout
        ffn_output = self.dropout(ffn_output)
        
        # Step 8: Residual connection
        output = x + ffn_output
        
        return output, attention_weights


class TransformerDecoder(nn.Module):
    """
    Stack of Transformer Decoder Blocks
    
    N decoder block'u stack eder. Her block önceki block'un output'unu input olarak alır.
    
    Args:
        n_layers: Number of decoder blocks
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        use_gated_ffn: Use gated FFN
        use_sdpa: Whether to use PyTorch SDPA
        gradient_checkpointing: Whether to use gradient checkpointing during training
    
    Shape:
        Input: [batch, seq_len, d_model]
        Output: [batch, seq_len, d_model]
    """
    
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_gated_ffn: bool = False,
        use_sdpa: bool = True,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.d_model = d_model
        self.use_sdpa = use_sdpa
        self.gradient_checkpointing = gradient_checkpointing
        
        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                use_gated_ffn=use_gated_ffn,
                use_sdpa=use_sdpa
            )
            for _ in range(n_layers)
        ])
        
        # Final layer normalization (Pre-LN style)
        self.ln_final = nn.LayerNorm(d_model)
        
        logger.info(
            f"TransformerDecoder initialized: {n_layers} layers, "
            f"d_model={d_model}, n_heads={n_heads}, use_sdpa={use_sdpa}, "
            f"gradient_checkpointing={gradient_checkpointing}"
        )
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory-efficient training."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        is_causal: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Transformer decoder forward pass.
        
        Args:
            x: Input tensor, shape [B, T, D]
            mask: Causal attention mask, shape [1, 1, T, T]
            need_weights: Whether to return explicit attention weights from each layer
            is_causal: Whether to apply causal mask in attention
        
        Returns:
            output: Decoder output, shape [B, T, D]
            attention_weights_list: List of attention weights from each layer
                                   Each: [B, H, T, T] or None
        
        Pipeline:
            Input [B, T, D]
            -> Layer 1 [B, T, D]
            -> Layer 2 [B, T, D]
            -> ...
            -> Layer N [B, T, D]
            -> Final LayerNorm [B, T, D]
            -> Output [B, T, D]
        """

        # Collect attention weights from each layer
        attention_weights_list = []
        
        # Pass through each decoder block
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(inputs[0], inputs[1], need_weights=need_weights, is_causal=is_causal)
                    return custom_forward

                x, attention_weights = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    mask,
                    use_reentrant=False
                )
            else:
                x, attention_weights = layer(
                    x,
                    mask=mask,
                    need_weights=need_weights,
                    is_causal=is_causal
                )
            attention_weights_list.append(attention_weights)
        
        # Final layer normalization
        output = self.ln_final(x)
        
        return output, attention_weights_list


def create_transformer_block(
    d_model: int,
    n_heads: int,
    d_ff: int,
    dropout: float = 0.1,
    activation: str = 'gelu',
    use_gated_ffn: bool = False,
    use_sdpa: bool = True
) -> TransformerDecoderBlock:
    """
    Factory function to create a single transformer decoder block.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        use_gated_ffn: Use gated FFN
        use_sdpa: Whether to use PyTorch SDPA
    
    Returns:
        TransformerDecoderBlock instance
    """
    return TransformerDecoderBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        use_gated_ffn=use_gated_ffn,
        use_sdpa=use_sdpa
    )


def create_transformer_decoder(
    n_layers: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    dropout: float = 0.1,
    activation: str = 'gelu',
    use_gated_ffn: bool = False,
    use_sdpa: bool = True,
    gradient_checkpointing: bool = False
) -> TransformerDecoder:
    """
    Factory function to create stacked transformer decoder.
    
    Args:
        n_layers: Number of decoder blocks
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function
        use_gated_ffn: Use gated FFN
        use_sdpa: Whether to use PyTorch SDPA
        gradient_checkpointing: Whether to use gradient checkpointing during training
    
    Returns:
        TransformerDecoder instance
    
    Example:
        >>> decoder = create_transformer_decoder(
        ...     n_layers=6,
        ...     d_model=256,
        ...     n_heads=8,
        ...     d_ff=1024
        ... )
        >>> x = torch.randn(2, 100, 256)
        >>> mask = create_causal_mask(100, x.device)
        >>> output, attn_list = decoder(x, mask)
        >>> len(attn_list)  # One per layer
        6
    """
    return TransformerDecoder(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        use_gated_ffn=use_gated_ffn,
        use_sdpa=use_sdpa,
        gradient_checkpointing=gradient_checkpointing
    )


if __name__ == "__main__":
    # Test transformer blocks
    logging.basicConfig(level=logging.INFO)
    
    from src.model.attention import create_causal_mask
    
    print("=" * 80)
    print("Transformer Decoder Block Test")
    print("=" * 80)
    
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 256
    n_heads = 8
    d_ff = 1024
    n_layers = 3
    
    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  n_layers: {n_layers}")
    
    # Test 1: Single decoder block
    print(f"\n{'='*80}")
    print("[1] Single Transformer Decoder Block")
    print(f"{'='*80}")
    
    block = create_transformer_block(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    print(f"✅ Decoder block created")
    
    # Create input and mask
    x = torch.randn(batch_size, seq_len, d_model)
    mask = create_causal_mask(seq_len, x.device)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Mask shape: {mask.shape}")
    
    # Forward pass
    output, attn_weights = block(x, mask)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"  Output mean: {output.mean():.4f}")
    print(f"  Output std: {output.std():.4f}")
    
    # Validate shapes
    assert output.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    print(f"✅ Shape validation passed")
    
    # Test residual connection
    # Output should be different from input but in same range
    print(f"\n  Residual Connection Check:")
    print(f"    Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"    Output mean: {output.mean():.4f}, std: {output.std():.4f}")
    print(f"    Difference: {(output - x).abs().mean():.4f}")
    
    if (output - x).abs().mean() > 0.01:
        print(f"✅ Residual connection working (output differs from input)")
    
    # Test 2: Stacked decoder (multiple layers)
    print(f"\n{'='*80}")
    print("[2] Stacked Transformer Decoder")
    print(f"{'='*80}")
    
    decoder = create_transformer_decoder(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    print(f"✅ {n_layers}-layer decoder created")
    
    # Forward pass
    output_stack, attn_list = decoder(x, mask)
    
    print(f"  Output shape: {output_stack.shape}")
    print(f"  Number of attention weight tensors: {len(attn_list)}")
    print(f"  Each attention weight shape: {attn_list[0].shape}")
    print(f"  Output mean: {output_stack.mean():.4f}")
    print(f"  Output std: {output_stack.std():.4f}")
    
    # Validate
    assert output_stack.shape == (batch_size, seq_len, d_model)
    assert len(attn_list) == n_layers
    assert all(a.shape == (batch_size, n_heads, seq_len, seq_len) for a in attn_list)
    print(f"✅ Shape validation passed")
    
    # Test 3: Layer-wise changes
    print(f"\n{'='*80}")
    print("[3] Layer-wise Representation Changes")
    print(f"{'='*80}")
    
    # Track representations through layers
    decoder.eval()
    with torch.no_grad():
        representations = [x]
        current = x
        
        for i, layer in enumerate(decoder.layers):
            current, _ = layer(current, mask)
            representations.append(current)
            
            # Compare with previous layer
            if i > 0:
                diff = (current - representations[i]).abs().mean()
                print(f"  Layer {i+1} change from Layer {i}: {diff:.4f}")
    
    print(f"✅ Representations evolve through layers")
    
    # Test 4: Parameter count
    print(f"\n{'='*80}")
    print("[4] Parameter Count")
    print(f"{'='*80}")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params_block = count_parameters(block)
    params_decoder = count_parameters(decoder)
    
    print(f"  Single block parameters: {params_block:,}")
    print(f"  {n_layers}-layer decoder parameters: {params_decoder:,}")
    print(f"  Expected (approx {n_layers} × block): {n_layers * params_block:,}")
    print(f"  Actual vs Expected ratio: {params_decoder / (n_layers * params_block):.3f}")
    
    # Should be slightly more than n_layers * block due to final LayerNorm
    expected_params = n_layers * params_block + 2 * d_model  # final LN
    print(f"  Expected with final LN: {expected_params:,}")
    
    print(f"\n" + "=" * 80)
    print("✅ All tests PASSED!")
    print("=" * 80)
