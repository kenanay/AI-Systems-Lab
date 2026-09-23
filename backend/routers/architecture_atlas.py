"""
backend/routers/architecture_atlas.py

Architecture Atlas API Endpoints
Local AI Research Lab - Developed by Kenan AY

Provides interactive REST endpoints for:
- 10 modern foundation model architectures catalog with analytical formulas and trade-offs
- Sparse Mixture of Experts (MoE) token routing visualizer with Top-k gating and auxiliary loss
- Selective State Space Models (Mamba / SSM) forward simulation and O(T) state tracking
- Grouped-Query Attention (GQA) & RoPE forward pass simulation
- Theoretical complexity scaling curves comparing Dense Transformer vs MoE vs Mamba SSM
"""

import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import torch
import logging

from src.architectures.atlas_registry import (
    get_architecture_catalog,
    get_architecture_by_id,
    simulate_scaling_curves,
)
from src.architectures.moe import (
    MoEConfig,
    MoERouter,
    MoELayer,
    MoETransformerBlock,
)
from src.architectures.mamba_ssm import (
    MambaConfig,
    SelectiveSSM,
    MambaBlock,
)
from src.architectures.modern_attention import (
    RotaryEmbedding,
    GroupedQueryAttention,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/architecture-atlas",
    tags=["architecture-atlas"]
)


# ============================================================================
# Request / Response Schemas
# ============================================================================

class MoERouteRequest(BaseModel):
    tokens: Optional[List[str]] = Field(
        default=["DeepSeek", "Sparse", "Mixture", "Of", "Experts", "Routing", "Gate", "Balance"],
        description="List of token strings to route"
    )
    d_model: int = Field(32, ge=8, le=256, description="Hidden dimension")
    num_experts: int = Field(8, ge=2, le=32, description="Number of experts in pool")
    top_k: int = Field(2, ge=1, le=4, description="Top-k experts activated per token")
    noisy_gating: bool = Field(False, description="Whether to add exploration noise to router logits")
    seed: Optional[int] = Field(42, description="Random seed for deterministic demonstration")


class TokenRouteDetail(BaseModel):
    token: str
    token_index: int
    selected_experts: List[int]
    routing_weights: List[float]
    all_probabilities: List[float]


class MoERouteResponse(BaseModel):
    num_tokens: int
    num_experts: int
    top_k: int
    routes: List[TokenRouteDetail]
    expert_load: Dict[str, int]
    expert_load_fractions: Dict[str, float]
    mean_expert_probabilities: Dict[str, float]
    aux_loss: float
    is_balanced: bool
    summary: str


class SimulateRequest(BaseModel):
    architecture: str = Field(
        ...,
        description="Architecture to simulate: 'moe', 'mamba_ssm', 'gqa'"
    )
    batch_size: int = Field(1, ge=1, le=16)
    seq_len: int = Field(8, ge=1, le=128)
    d_model: int = Field(32, ge=8, le=256)
    num_experts: Optional[int] = Field(8, ge=2, le=32)
    top_k: Optional[int] = Field(2, ge=1, le=4)
    d_state: Optional[int] = Field(16, ge=4, le=64)
    d_conv: Optional[int] = Field(4, ge=2, le=8)
    expand: Optional[int] = Field(2, ge=1, le=4)
    num_heads: Optional[int] = Field(8, ge=2, le=32)
    num_kv_heads: Optional[int] = Field(2, ge=1, le=16)


class SimulateResponse(BaseModel):
    architecture: str
    input_shape: List[int]
    output_shape: List[int]
    parameter_count: int
    forward_time_ms: float
    output_stats: Dict[str, float]
    architecture_metadata: Dict[str, Any]


class ScalingCurveRequest(BaseModel):
    seq_lengths: Optional[List[int]] = Field(
        default=[128, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        description="Sequence length sampling points"
    )
    d_model: int = Field(4096, ge=512, le=16384)
    num_layers: int = Field(32, ge=1, le=128)


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/catalog")
def get_catalog():
    """Returns the complete pedagogical catalog of 10 modern foundation model architectures."""
    catalog = get_architecture_catalog()
    return {
        "status": "success",
        "total_architectures": len(catalog),
        "architectures": catalog,
    }


@router.get("/catalog/{arch_id}")
def get_architecture_detail(arch_id: str):
    """Returns detailed specification, formulas, trade-offs, and reference implementations for a specific architecture."""
    arch = get_architecture_by_id(arch_id)
    if not arch:
        raise HTTPException(status_code=404, detail=f"Architecture '{arch_id}' not found in catalog")
    return {
        "status": "success",
        "architecture": arch,
    }


@router.post("/moe/route", response_model=MoERouteResponse)
def route_moe_tokens(req: MoERouteRequest):
    """
    Simulates token routing through a Sparse MoE gating layer.
    Computes top-k expert selection, softmax distribution, and auxiliary load-balancing loss.
    """
    if req.seed is not None:
        torch.manual_seed(req.seed)

    tokens = req.tokens or [f"T{i}" for i in range(8)]
    num_tokens = len(tokens)

    # Initialize MoE Router
    router_layer = MoERouter(
        d_model=req.d_model,
        num_experts=req.num_experts,
        top_k=req.top_k,
        noisy_gating=req.noisy_gating,
    )

    # Create synthetic token embeddings with rich representation
    embeddings = torch.randn(1, num_tokens, req.d_model)

    router_layer.eval()
    with torch.no_grad():
        router_weights, selected_experts, all_probs, aux_loss = router_layer(embeddings)

    # Extract arrays
    weights_np = router_weights.squeeze(0).cpu().numpy()
    experts_np = selected_experts.squeeze(0).cpu().numpy()
    all_probs_np = all_probs.squeeze(0).cpu().numpy()

    routes: List[TokenRouteDetail] = []
    expert_counts = [0] * req.num_experts

    for i, tok in enumerate(tokens):
        sel_exp = experts_np[i].tolist()
        r_weights = [round(float(w), 4) for w in weights_np[i].tolist()]
        probs = [round(float(p), 4) for p in all_probs_np[i].tolist()]

        for exp_id in sel_exp:
            expert_counts[exp_id] += 1

        routes.append(TokenRouteDetail(
            token=tok,
            token_index=i,
            selected_experts=sel_exp,
            routing_weights=r_weights,
            all_probabilities=probs,
        ))

    total_assignments = num_tokens * req.top_k
    expert_load = {f"Expert_{e}": expert_counts[e] for e in range(req.num_experts)}
    expert_load_fractions = {
        f"Expert_{e}": round(expert_counts[e] / max(1, total_assignments), 4)
        for e in range(req.num_experts)
    }

    mean_probs = all_probs_np.mean(axis=0)
    mean_expert_probabilities = {
        f"Expert_{e}": round(float(mean_probs[e]), 4)
        for e in range(req.num_experts)
    }

    ideal_load = total_assignments / req.num_experts
    max_imbalance = max(abs(cnt - ideal_load) for cnt in expert_counts)
    is_balanced = max_imbalance <= (total_assignments * 0.4)

    summary = (
        f"Routed {num_tokens} tokens across {req.num_experts} experts (Top-{req.top_k}). "
        f"Auxiliary load-balancing loss: {float(aux_loss):.4f}. "
        f"{'Load is reasonably balanced.' if is_balanced else 'Expert routing shows collapse/imbalance without further training.'}"
    )

    return MoERouteResponse(
        num_tokens=num_tokens,
        num_experts=req.num_experts,
        top_k=req.top_k,
        routes=routes,
        expert_load=expert_load,
        expert_load_fractions=expert_load_fractions,
        mean_expert_probabilities=mean_expert_probabilities,
        aux_loss=round(float(aux_loss), 6),
        is_balanced=is_balanced,
        summary=summary,
    )


@router.post("/simulate", response_model=SimulateResponse)
def simulate_architecture(req: SimulateRequest):
    """
    Executes a real PyTorch forward pass simulation on the chosen modern architecture
    (Sparse MoE Block, Mamba SSM Block, or Grouped-Query Attention).
    Returns execution latency, shape transforms, parameter count, and architectural diagnostics.
    """
    x = torch.randn(req.batch_size, req.seq_len, req.d_model)
    metadata: Dict[str, Any] = {}

    start_time = time.perf_counter()

    if req.architecture.lower() == "moe":
        config = MoEConfig(
            d_model=req.d_model,
            num_experts=req.num_experts or 8,
            top_k=req.top_k or 2,
            d_ff=req.d_model * 4,
        )
        block = MoETransformerBlock(
            d_model=config.d_model,
            n_heads=req.num_heads or 4,
            moe_config=config,
        )
        block.eval()
        with torch.no_grad():
            out, _, routing_info = block(x)

        aux_loss = routing_info["unscaled_aux_loss"]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        param_count = sum(p.numel() for p in block.parameters())
        metadata = {
            "type": "Sparse Mixture of Experts (MoE)",
            "num_experts": config.num_experts,
            "top_k": config.top_k,
            "aux_loss": round(float(aux_loss), 6),
            "expert_d_ff": config.d_ff,
            "complexity": "O(T^2 * d_model) attention + O(T * top_k * d_ff) expert FFN",
        }

    elif req.architecture.lower() in ("mamba_ssm", "mamba", "ssm"):
        config = MambaConfig(
            d_model=req.d_model,
            d_state=req.d_state or 16,
            d_conv=req.d_conv or 4,
            expand=req.expand or 2,
        )
        block = MambaBlock(config)
        block.eval()
        with torch.no_grad():
            out, final_state = block(x)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        param_count = sum(p.numel() for p in block.parameters())
        metadata = {
            "type": "Selective State Space Model (Mamba SSM)",
            "d_state": config.d_state,
            "d_inner": config.d_inner,
            "d_conv": config.d_conv,
            "expand_factor": config.expand,
            "complexity": "O(T * d_inner * d_state) STRICTLY LINEAR with sequence length!",
            "recurrent_state_dim": f"[{req.batch_size}, {config.d_inner}, {config.d_state}]",
        }

    elif req.architecture.lower() in ("gqa", "attention"):
        num_heads = req.num_heads or 8
        num_kv_heads = req.num_kv_heads or 2
        block = GroupedQueryAttention(
            d_model=req.d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        block.eval()
        with torch.no_grad():
            out, _ = block(x)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        param_count = sum(p.numel() for p in block.parameters())
        metadata = {
            "type": "Grouped-Query Attention (GQA)",
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": req.d_model // num_heads,
            "kv_cache_compression": f"{num_heads // num_kv_heads}x reduction vs Multi-Head Attention",
            "complexity": "O(T^2 * d_model) attention computation with compressed KV cache footprint",
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported architecture '{req.architecture}'. Supported: 'moe', 'mamba_ssm', 'gqa'"
        )

    out_flat = out.view(-1)
    stats = {
        "mean": round(float(out_flat.mean().item()), 5),
        "std": round(float(out_flat.std().item()), 5),
        "min": round(float(out_flat.min().item()), 5),
        "max": round(float(out_flat.max().item()), 5),
        "norm": round(float(torch.norm(out).item()), 5),
    }

    return SimulateResponse(
        architecture=req.architecture,
        input_shape=list(x.shape),
        output_shape=list(out.shape),
        parameter_count=param_count,
        forward_time_ms=round(elapsed_ms, 3),
        output_stats=stats,
        architecture_metadata=metadata,
    )


@router.post("/scaling-curve")
def get_scaling_curves(req: ScalingCurveRequest):
    """
    Computes theoretical scaling curves comparing:
    - Standard Dense Transformer (O(T^2))
    - Sparse Mixture of Experts (O(T^2) attn + O(k*T) FFN)
    - Mamba Selective SSM (Strictly O(T))
    across specified sequence lengths.
    """
    seq_lengths = req.seq_lengths or [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    data = simulate_scaling_curves(
        seq_lengths=seq_lengths,
        d_model=req.d_model,
        num_layers=req.num_layers,
    )
    return {
        "status": "success",
        "d_model": req.d_model,
        "num_layers": req.num_layers,
        "curve_points": data,
        "takeaway": (
            "At 32k context, Mamba SSM achieves >10x FLOPs reduction and O(1) inference memory compared to Dense Transformer."
        )
    }
