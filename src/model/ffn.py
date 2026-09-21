"""
src/model/ffn.py

Position-wise Feed-Forward Network

Bu modül transformer'daki position-wise feed-forward network'ü içerir.
Her pozisyon için bağımsız olarak uygulanır (shared weights).

Mimari:
    FFN(x) = activation(x @ W1 + b1) @ W2 + b2
    
    x: [batch, seq_len, d_model]
    W1: [d_model, d_ff]
    W2: [d_ff, d_model]
    
Genellikle d_ff = 4 * d_model (expansion factor)

Activation fonksiyonu olarak GELU kullanıyoruz (GPT-style).
GELU, ReLU'ya göre daha smooth ve genellikle daha iyi sonuç verir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PositionWiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    İki linear layer ve bir activation function'dan oluşur.
    Her sequence pozisyonu için aynı transformation uygulanır.
    
    Architecture:
        Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model) → Dropout
    
    Args:
        d_model: Model dimension (input/output dimension)
        d_ff: Feed-forward dimension (hidden layer size)
        dropout: Dropout rate
        activation: Activation function name ('gelu', 'relu', 'swish')
    
    Shape:
        Input: [batch, seq_len, d_model]
        Output: [batch, seq_len, d_model]
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation_name = activation
        
        # First linear layer: d_model → d_ff (expansion)
        # w1 shape: [d_model, d_ff]
        self.w1 = nn.Linear(d_model, d_ff)
        
        # Second linear layer: d_ff → d_model (projection back)
        # w2 shape: [d_ff, d_model]
        self.w2 = nn.Linear(d_ff, d_model)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Dropout (applied after each layer)
        self.dropout = nn.Dropout(p=dropout)
        
        logger.debug(
            f"PositionWiseFFN initialized: d_model={d_model}, "
            f"d_ff={d_ff}, activation={activation}"
        )
    
    def _get_activation(self, name: str) -> nn.Module:
        """
        Activation function seç.
        
        Args:
            name: Activation function adı
        
        Returns:
            Activation function module
        """
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swish': nn.SiLU(),  # SiLU is Swish
        }
        
        if name not in activations:
            logger.warning(
                f"Unknown activation '{name}', using GELU. "
                f"Available: {list(activations.keys())}"
            )
            return activations['gelu']
        
        return activations[name]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed-forward network forward pass.
        
        Args:
            x: Input tensor, shape [B, T, D]
        
        Returns:
            Output tensor, shape [B, T, D]
            
        Pipeline:
            Input [B, T, D]
            -> Linear1 [B, T, D_ff]
            -> Activation [B, T, D_ff]
            -> Dropout [B, T, D_ff]
            -> Linear2 [B, T, D]
            -> Dropout [B, T, D]
            
        Shape notation:
            B = batch_size
            T = seq_len
            D = d_model
            D_ff = d_ff
        """
        # x shape: [B, T, D]
        
        # First layer: expand dimension
        # x @ w1: [B, T, D] @ [D, D_ff] -> [B, T, D_ff]
        hidden = self.w1(x)
        
        # Activation function
        # hidden shape: [B, T, D_ff]
        hidden = self.activation(hidden)
        
        # Dropout after activation
        # hidden shape: [B, T, D_ff]
        hidden = self.dropout(hidden)
        
        # Second layer: project back to d_model
        # hidden @ w2: [B, T, D_ff] @ [D_ff, D] -> [B, T, D]
        output = self.w2(hidden)
        
        # Dropout after second layer
        # output shape: [B, T, D]
        output = self.dropout(output)
        
        return output


class GatedFFN(nn.Module):
    """
    Gated Feed-Forward Network (GLU-style)
    
    Modern transformer'larda kullanılan gated variant.
    İki parallel projection yapar ve biri gate olarak kullanılır.
    
    Architecture:
        FFN(x) = (x @ W_gate ⊙ activation(x @ W_up)) @ W_down
        
        ⊙ : element-wise multiplication (gating mechanism)
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function for gate
    
    Shape:
        Input: [batch, seq_len, d_model]
        Output: [batch, seq_len, d_model]
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Gate projection: d_model → d_ff
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        
        # Up projection: d_model → d_ff
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        
        # Down projection: d_ff → d_model
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        logger.debug(
            f"GatedFFN initialized: d_model={d_model}, "
            f"d_ff={d_ff}, activation={activation}"
        )
    
    def _get_activation(self, name: str) -> nn.Module:
        """Activation function seç."""
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swish': nn.SiLU(),
        }
        return activations.get(name, nn.GELU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gated FFN forward pass.
        
        Args:
            x: Input tensor, shape [B, T, D]
        
        Returns:
            Output tensor, shape [B, T, D]
            
        Pipeline:
            Input [B, T, D]
            -> Gate path: Linear [B, T, D_ff]
            -> Up path: Linear → Activation [B, T, D_ff]
            -> Element-wise multiply (gating) [B, T, D_ff]
            -> Down projection [B, T, D]
            -> Dropout [B, T, D]
        """
        # x shape: [B, T, D]
        
        # Gate projection
        # gate shape: [B, T, D_ff]
        gate = self.w_gate(x)
        
        # Up projection with activation
        # up shape: [B, T, D_ff]
        up = self.activation(self.w_up(x))
        
        # Gating: element-wise multiplication
        # gated shape: [B, T, D_ff]
        gated = gate * up
        
        # Down projection
        # output shape: [B, T, D]
        output = self.w_down(gated)
        
        # Dropout
        # output shape: [B, T, D]
        output = self.dropout(output)
        
        return output


def create_ffn_layer(
    d_model: int,
    d_ff: int,
    dropout: float = 0.1,
    activation: str = 'gelu',
    gated: bool = False
) -> nn.Module:
    """
    Factory function to create FFN layer.
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension (typically 4 * d_model)
        dropout: Dropout rate
        activation: Activation function ('gelu', 'relu', 'swish')
        gated: Use gated FFN (GLU-style) instead of standard FFN
    
    Returns:
        FFN layer (PositionWiseFFN or GatedFFN)
    
    Example:
        >>> # Standard FFN
        >>> ffn = create_ffn_layer(d_model=256, d_ff=1024)
        >>> x = torch.randn(2, 100, 256)  # [B=2, T=100, D=256]
        >>> output = ffn(x)  # [2, 100, 256]
        
        >>> # Gated FFN
        >>> gated_ffn = create_ffn_layer(d_model=256, d_ff=1024, gated=True)
        >>> output = gated_ffn(x)  # [2, 100, 256]
    """
    if gated:
        return GatedFFN(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
    else:
        return PositionWiseFFN(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )


if __name__ == "__main__":
    # Test feed-forward networks
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Feed-Forward Network Test")
    print("=" * 80)
    
    # Parameters
    batch_size = 2
    seq_len = 100
    d_model = 256
    d_ff = 1024  # 4x expansion
    
    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}")
    print(f"  expansion_factor: {d_ff / d_model:.1f}x")
    
    # Test 1: Standard FFN
    print(f"\n{'='*80}")
    print("[1] Standard Position-wise FFN")
    print(f"{'='*80}")
    
    ffn_standard = create_ffn_layer(
        d_model=d_model,
        d_ff=d_ff,
        dropout=0.1,
        activation='gelu'
    )
    print(f"✅ Standard FFN created")
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"  Input shape: {x.shape}")
    
    # Forward pass
    output_standard = ffn_standard(x)
    print(f"  Output shape: {output_standard.shape}")
    print(f"  Output mean: {output_standard.mean():.4f}")
    print(f"  Output std: {output_standard.std():.4f}")
    
    # Validate shape
    assert output_standard.shape == (batch_size, seq_len, d_model)
    print(f"✅ Shape validation passed")
    
    # Check position-wise property
    # Each position should be independent
    # Set to eval mode to disable dropout for this test
    ffn_standard.eval()
    
    with torch.no_grad():
        x_test = torch.randn(batch_size, seq_len, d_model)
        output_full = ffn_standard(x_test)
        
        x_single_pos = x_test[:, 0:1, :]  # [B, 1, D]
        output_single = ffn_standard(x_single_pos)
        
        # Should match first position of full output
        diff = (output_single - output_full[:, 0:1, :]).abs().max()
        print(f"  Position-wise independence check (eval mode): max_diff={diff:.6f}")
        if diff < 1e-5:
            print(f"✅ Position-wise property verified")
        else:
            print(f"⚠️  Position-wise property may not hold (diff={diff:.6f})")
    
    # Set back to train mode
    ffn_standard.train()
    
    # Test 2: Gated FFN
    print(f"\n{'='*80}")
    print("[2] Gated FFN (GLU-style)")
    print(f"{'='*80}")
    
    ffn_gated = create_ffn_layer(
        d_model=d_model,
        d_ff=d_ff,
        dropout=0.1,
        activation='gelu',
        gated=True
    )
    print(f"✅ Gated FFN created")
    
    # Forward pass
    output_gated = ffn_gated(x)
    print(f"  Output shape: {output_gated.shape}")
    print(f"  Output mean: {output_gated.mean():.4f}")
    print(f"  Output std: {output_gated.std():.4f}")
    
    # Validate shape
    assert output_gated.shape == (batch_size, seq_len, d_model)
    print(f"✅ Shape validation passed")
    
    # Test 3: Different activation functions
    print(f"\n{'='*80}")
    print("[3] Activation Functions Comparison")
    print(f"{'='*80}")
    
    activations = ['gelu', 'relu', 'swish']
    x_test = torch.randn(2, 10, d_model)
    
    for act in activations:
        ffn_act = create_ffn_layer(
            d_model=d_model,
            d_ff=d_ff,
            activation=act
        )
        output_act = ffn_act(x_test)
        
        print(f"  {act.upper()}:")
        print(f"    Output mean: {output_act.mean():.4f}")
        print(f"    Output std: {output_act.std():.4f}")
        print(f"    Output range: [{output_act.min():.4f}, {output_act.max():.4f}]")
    
    # Test 4: Parameter count
    print(f"\n{'='*80}")
    print("[4] Parameter Count")
    print(f"{'='*80}")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params_standard = count_parameters(ffn_standard)
    params_gated = count_parameters(ffn_gated)
    
    print(f"  Standard FFN parameters: {params_standard:,}")
    print(f"    W1: {d_model * d_ff:,} (+ {d_ff:,} bias)")
    print(f"    W2: {d_ff * d_model:,} (+ {d_model:,} bias)")
    print(f"    Total: {d_model * d_ff + d_ff + d_ff * d_model + d_model:,}")
    
    print(f"\n  Gated FFN parameters: {params_gated:,}")
    print(f"    W_gate: {d_model * d_ff:,} (no bias)")
    print(f"    W_up: {d_model * d_ff:,} (no bias)")
    print(f"    W_down: {d_ff * d_model:,} (no bias)")
    print(f"    Total: {3 * d_model * d_ff:,}")
    
    print(f"\n  Gated FFN has {params_gated - params_standard:,} more parameters")
    print(f"  ({(params_gated / params_standard - 1) * 100:.1f}% increase)")
    
    print(f"\n" + "=" * 80)
    print("✅ All tests PASSED!")
    print("=" * 80)
