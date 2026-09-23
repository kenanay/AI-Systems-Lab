"""
src/architectures/mamba_ssm.py

Mamba: Selective State Space Model (SSM)
========================================
Bu modül Albert Gu ve Tri Dao tarafından geliştirilen Mamba (Selective SSM)
mimarisinin saf PyTorch implementasyonunu sağlar.

Temel Prensipler:
1. Doğrusal Zaman ve Bellek Karmaşıklığı: Transformer modellerindeki O(T^2)
   attention karmaşıklığı yerine, sekans uzunluğuna göre O(T) doğrusal hesaplama
   ve O(1) sabit çıkarım (inference) bellek gereksinimi sunar.
2. Seçici Mekanizma (Selection Mechanism): Geleneksel LTI (Linear Time-Invariant)
   durum uzayı modellerinde parametreler zaman boyunca sabitken, Mamba'da B, C ve
   Δ (zaman adımı) parametreleri anlık girdiye (x_t) bağlı dinamik olarak üretilir.
   Bu sayede model önemli bilgileri seçici olarak hafızasında tutarken gereksiz
   bilgileri anında unutabilir.
3. Çift Mod (Dual Mode): Eğitim sırasında taranabilir ardışık işlem, çıkarım (inference)
   sırasında ise sabit boyutlu durum vektörü (d_state) ile çalışan rekürsif hücre.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MambaConfig:
    """
    Configuration for Mamba Selective SSM.
    
    Attributes:
        d_model: Input and output embedding dimension
        d_state: SSM latent state dimension (typically 16)
        d_conv: 1D convolution window width (typically 4)
        expand: Dimension expansion factor (inner_dim = expand * d_model)
        dt_rank: Rank of Delta projection (defaults to ceil(d_model / 16))
    """
    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: Optional[int] = None

    def __post_init__(self):
        if self.dt_rank is None:
            self.dt_rank = math.ceil(self.d_model / 16)
        self.d_inner = int(self.expand * self.d_model)


class SelectiveSSM(nn.Module):
    """
    Continuous-to-discrete selective state space model.
    
    Equations:
        h_t = exp(Δ_t * A) * h_{t-1} + (Δ_t * B_t) * x_t
        y_t = C_t * h_t + D * x_t
    """

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Learnable skip connection parameter D: [d_inner]
        self.D = nn.Parameter(torch.ones(d_inner))

        # A parameter initialized with HiPPO heuristic (real part negative for stability)
        # Stored in log-space: A = -exp(A_log)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Projection from x to [dt, B, C]
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        # Projection from dt_rank to d_inner
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize dt_proj bias to ensure reasonable initial Δ values
        dt_init_std = dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Set initial dt to approximately 0.001 - 0.1
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective SSM forward scan.
        
        Args:
            u: Input tensor after conv and activation, shape [B, T, d_inner]
            
        Returns:
            y: Output tensor, shape [B, T, d_inner]
            final_state: Latent state at the end of the sequence, shape [B, d_inner, d_state]
        """
        batch_size, seq_len, d_inner = u.shape

        # A is negative: shape [d_inner, d_state]
        A = -torch.exp(self.A_log.float())

        # Project input to dt, B, C
        # x_dbl: [B, T, dt_rank + 2 * d_state]
        x_dbl = self.x_proj(u)
        dt_raw, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )

        # Delta computation: [B, T, d_inner]
        dt = F.softplus(self.dt_proj(dt_raw))

        # Discretization:
        # dA: [B, T, d_inner, d_state] = exp(dt[:, :, :, None] * A[None, None, :, :])
        dA = torch.exp(torch.einsum("btd,dn->btdn", dt, A))
        
        # dB: [B, T, d_inner, d_state] = dt[:, :, :, None] * B[:, :, None, :]
        dB = torch.einsum("btd,btn->btdn", dt, B)

        # Sequential scan along time dimension (O(T) linear pass)
        h = torch.zeros(batch_size, d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []

        for t in range(seq_len):
            u_t = u[:, t, :].unsqueeze(-1)  # [B, d_inner, 1]
            dA_t = dA[:, t, :, :]           # [B, d_inner, d_state]
            dB_t = dB[:, t, :, :]           # [B, d_inner, d_state]
            C_t = C[:, t, :]                # [B, d_state]

            # State update: h_t = dA * h_{t-1} + dB * u_t
            h = dA_t * h + dB_t * u_t

            # Output: y_t = C_t @ h_t
            y_t = torch.einsum("bdn,bn->bd", h, C_t)  # [B, d_inner]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # [B, T, d_inner]

        # Add skip connection: D * u
        y = y + u * self.D

        return y, h


class MambaBlock(nn.Module):
    """
    Complete Mamba Layer Block.
    
    Pipeline:
        x -> in_proj -> split into (u, residual_gate)
          -> 1D Convolution -> SiLU
          -> Selective SSM
          -> Multiply with SiLU(residual_gate)
          -> out_proj -> output
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_conv = config.d_conv

        # Linear projection to expand dimension for both branches
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=False)

        # 1D Depthwise Convolution
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=True,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # Selective State Space Model
        self.ssm = SelectiveSSM(
            d_inner=config.d_inner,
            d_state=config.d_state,
            dt_rank=config.dt_rank
        )

        # Output projection back to d_model
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Mamba block.
        
        Args:
            x: Input tensor, shape [B, T, d_model]
            
        Returns:
            output: Output tensor, shape [B, T, d_model]
            final_state: SSM latent state, shape [B, d_inner, d_state]
        """
        batch_size, seq_len, _ = x.shape

        # Expand projection: [B, T, 2 * d_inner]
        projected = self.in_proj(x)
        u, res_gate = torch.chunk(projected, 2, dim=-1)

        # 1D causal convolution on branch u
        # Permute for conv1d: [B, d_inner, T]
        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        u_act = F.silu(u_conv)

        # Selective SSM
        y_ssm, final_state = self.ssm(u_act)

        # Gated multiplication
        gate_act = F.silu(res_gate)
        y_gated = y_ssm * gate_act

        # Project back to d_model
        output = self.out_proj(y_gated)

        return output, final_state
