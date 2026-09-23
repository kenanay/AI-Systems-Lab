"""
Architecture Atlas: Modular deep learning architectures for modern LLMs and foundation models.
Provides reference implementations and interactive simulation components for:
- Sparse Mixture of Experts (MoE) with Top-k Gating & Auxiliary Load Balancing Loss
- Selective State Space Models (Mamba / SSM) with linear O(T) sequential scan
- Grouped-Query Attention (GQA) & Rotary Position Embeddings (RoPE)
- Architecture Atlas Registry & Analytical Scaling Curve Engine
"""

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
from src.architectures.atlas_registry import (
    ArchitectureSpec,
    ARCHITECTURE_CATALOG,
    get_architecture_catalog,
    get_architecture_by_id,
    simulate_scaling_curves,
)

__all__ = [
    "MoEConfig",
    "MoERouter",
    "MoELayer",
    "MoETransformerBlock",
    "MambaConfig",
    "SelectiveSSM",
    "MambaBlock",
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "ArchitectureSpec",
    "ARCHITECTURE_CATALOG",
    "get_architecture_catalog",
    "get_architecture_by_id",
    "simulate_scaling_curves",
]
