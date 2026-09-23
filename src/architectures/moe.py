"""
src/architectures/moe.py

Sparse Mixture of Experts (MoE) Architecture
===========================================
Bu modül modern açık kaynaklı LLM'lerde (Mixtral 8x7B, DeepSeek-V2/V3, Switch Transformer, Grok)
kullanılan Sparse Mixture of Experts (MoE) mekanizmasını sağlar.

Temel Prensipler:
1. Sparse Activation: Modelin toplam parametre sayısı büyük tutulurken, her token için yalnızca
   k adet uzman (ör. top-2) aktive edilir. Böylece parametre kapasitesi artarken token başına
   hesaplama maliyeti (FLOPs) düşük kalır.
2. Top-k Routing: Yönlendirici (Router/Gate) ağı, her token'ın gösterimini girdi olarak alıp
   uzmanlar üzerinde bir olasılık dağılımı (softmax) üretir ve en yüksek skora sahip k uzmanı seçer.
3. Auxiliary Load Balancing Loss: Uzmanların eşit oranda kullanılmasını sağlamak ve tüm token'ların
   tek bir uzmana çökmesini (expert collapse) önlemek amacıyla Switch Transformer / GShard tarzı
   dengeleme kaybı hesaplanır:
   L_aux = E * sum_i(f_i * P_i)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.ffn import PositionWiseFFN, create_ffn_layer
from src.model.attention import create_attention_layer

logger = logging.getLogger(__name__)


@dataclass
class MoEConfig:
    """
    MoE Layer Configuration.
    
    Attributes:
        d_model: Model dimension
        num_experts: Total number of expert networks (e.g. 4, 8, 16)
        top_k: Number of active experts per token (typically 1 or 2)
        d_ff: Hidden dimension inside each expert FFN
        dropout: Dropout probability
        activation: Activation function ('gelu', 'relu', 'swish')
        use_gated_ffn: Whether to use gated GLU-style FFN for experts
        aux_loss_coef: Weight multiplier for load balancing loss
    """
    d_model: int = 256
    num_experts: int = 4
    top_k: int = 2
    d_ff: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    use_gated_ffn: bool = False
    aux_loss_coef: float = 0.01

    def __post_init__(self):
        assert self.num_experts > 0, "num_experts must be positive"
        assert 1 <= self.top_k <= self.num_experts, "top_k must be between 1 and num_experts"
        assert self.d_model > 0, "d_model must be positive"
        assert self.d_ff > 0, "d_ff must be positive"


class MoERouter(nn.Module):
    """
    Top-k MoE Router / Gating Network.
    
    Her token için uzman skorlarını hesaplar, en yüksek k uzmanı seçer,
    olasılıkları normalize eder ve uzman yük dengeleme (load balancing) kaybını üretir.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, noisy_gating: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for router.
        
        Args:
            x: Input tensor, shape [B, T, D]
            
        Returns:
            topk_weights: Normalized routing weights for chosen experts, shape [B, T, top_k]
            topk_indices: Indices of chosen experts, shape [B, T, top_k]
            router_probs: Full softmax probabilities across all experts, shape [B, T, num_experts]
            aux_loss: Auxiliary load balancing loss scalar
        """
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len

        # Compute raw routing logits: [B, T, num_experts]
        logits = self.gate(x)

        # Full softmax probability distribution
        router_probs = F.softmax(logits, dim=-1)  # [B, T, E]

        # Top-k selection
        # topk_raw_weights, topk_indices: [B, T, top_k]
        topk_raw_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Renormalize weights so they sum to 1 over the selected top-k
        topk_weights = topk_raw_weights / (topk_raw_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # -------------------------------------------------------------
        # Auxiliary Load Balancing Loss (Switch Transformer / Mixtral)
        # f_i: fraction of tokens dispatched to expert i (across top-1 or top-k)
        # P_i: average probability allocated to expert i across all tokens
        # -------------------------------------------------------------
        flat_probs = router_probs.view(num_tokens, self.num_experts)  # [N, E]
        p_i = flat_probs.mean(dim=0)  # [E]

        # One-hot mask for selected experts: [N, top_k, E]
        flat_indices = topk_indices.view(num_tokens, self.top_k)  # [N, top_k]
        mask = F.one_hot(flat_indices, num_classes=self.num_experts).float()  # [N, top_k, E]
        tokens_per_expert = mask.sum(dim=(0, 1))  # [E]
        f_i = tokens_per_expert / (num_tokens * self.top_k)  # [E]

        # Auxiliary loss: E * sum_i(f_i * P_i)
        aux_loss = self.num_experts * torch.sum(f_i * p_i)

        return topk_weights, topk_indices, router_probs, aux_loss


class MoELayer(nn.Module):
    """
    Sparse Mixture of Experts Feed-Forward Layer.
    
    N adet bağımsız PositionWiseFFN uzmanından oluşur.
    Her token için yalnızca top_k uzman çalıştırılır ve çıktıları
    yönlendirici ağırlıklarıyla ağırlıklı toplanarak birleştirilir.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.aux_loss_coef = config.aux_loss_coef

        # Router network
        self.router = MoERouter(
            d_model=config.d_model,
            num_experts=config.num_experts,
            top_k=config.top_k
        )

        # Pool of Expert Networks
        self.experts = nn.ModuleList([
            create_ffn_layer(
                d_model=config.d_model,
                d_ff=config.d_ff,
                dropout=config.dropout,
                activation=config.activation,
                gated=config.use_gated_ffn
            )
            for _ in range(config.num_experts)
        ])

        logger.debug(
            f"MoELayer initialized: {config.num_experts} experts, "
            f"top_{config.top_k} routing, d_model={config.d_model}"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor, shape [B, T, D]
            
        Returns:
            output: MoE output tensor, shape [B, T, D]
            routing_info: Dictionary containing routing analytics:
                - aux_loss: Scaled auxiliary load balancing loss
                - unscaled_aux_loss: Raw auxiliary loss
                - expert_dispatch_counts: Number of tokens dispatched to each expert
                - router_probs: Mean softmax probability for each expert
                - topk_indices: Top-k expert assignments
                - topk_weights: Top-k routing weights
        """
        batch_size, seq_len, d_model = x.shape
        topk_weights, topk_indices, router_probs, raw_aux_loss = self.router(x)

        # Output accumulation tensor: [B, T, D]
        output = torch.zeros_like(x)

        # Reshape for efficient token-level routing
        x_flat = x.view(-1, d_model)  # [N, D]
        num_tokens = x_flat.size(0)

        # Dispatch counts for diagnostic monitoring
        dispatch_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)

        # Iterate over each expert and evaluate tokens assigned to it
        for expert_idx in range(self.num_experts):
            # Check which (token, k_slot) pairs routed to this expert
            expert_mask = (topk_indices == expert_idx)  # [B, T, top_k]
            if not expert_mask.any():
                continue

            dispatch_counts[expert_idx] = expert_mask.sum().item()

            # Execute expert on the full sequence or selected tokens
            # For numerical stability and simple vectorized execution:
            expert_out = self.experts[expert_idx](x)  # [B, T, D]

            # Accumulate weighted output for each k slot
            for k in range(self.top_k):
                k_mask = (topk_indices[:, :, k] == expert_idx).unsqueeze(-1)  # [B, T, 1]
                weight_k = topk_weights[:, :, k].unsqueeze(-1)  # [B, T, 1]
                output = output + (expert_out * weight_k) * k_mask.float()

        aux_loss = self.aux_loss_coef * raw_aux_loss

        routing_info: Dict[str, Any] = {
            "aux_loss": aux_loss,
            "unscaled_aux_loss": raw_aux_loss.item(),
            "expert_dispatch_counts": dispatch_counts.tolist(),
            "router_probs": router_probs.mean(dim=(0, 1)).detach().cpu().tolist(),
            "topk_indices": topk_indices.detach().cpu().tolist(),
            "topk_weights": topk_weights.detach().cpu().tolist(),
        }

        return output, routing_info


class MoETransformerBlock(nn.Module):
    """
    Transformer Decoder Block with MoE Feed-Forward Network.
    
    Pre-LN Transformer mimarisi:
    x -> LayerNorm -> Self-Attention -> Residual Add
      -> LayerNorm -> MoE FFN -> Residual Add
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        moe_config: MoEConfig,
        dropout: float = 0.1,
        use_sdpa: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-Head Attention
        self.attention = create_attention_layer(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_sdpa=use_sdpa
        )

        # Sparse MoE Layer
        self.moe = MoELayer(moe_config)

        # Layer Normalization (Pre-LN)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Forward pass for MoE Transformer block.
        
        Returns:
            output: [B, T, D]
            attn_weights: Attention weights (if requested)
            routing_info: MoE routing diagnostic dictionary
        """
        # 1. Attention sub-layer with Pre-LN
        x_norm = self.ln1(x)
        attn_out, attn_weights, _, _ = self.attention(
            x_norm,
            mask=mask,
            need_weights=need_weights,
            is_causal=is_causal
        )
        x = x + self.dropout(attn_out)

        # 2. MoE Feed-Forward sub-layer with Pre-LN
        x_norm = self.ln2(x)
        moe_out, routing_info = self.moe(x_norm)
        x = x + self.dropout(moe_out)

        return x, attn_weights, routing_info
