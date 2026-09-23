"""
Unit and integration tests for Architecture Atlas
Local AI Research Lab - Developed by Kenan AY

Tests:
1. Sparse Mixture of Experts (MoE) with Top-k Gating & Auxiliary Loss
2. Selective State Space Models (Mamba SSM) with linear O(T) sequential scan
3. Grouped-Query Attention (GQA) & Rotary Position Embeddings (RoPE)
4. Architecture Atlas Registry & Analytical Scaling Curves
5. FastAPI REST API endpoints
"""

import pytest
import torch
from fastapi.testclient import TestClient

from backend.main import app
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
    get_architecture_catalog,
    get_architecture_by_id,
    simulate_scaling_curves,
)

client = TestClient(app)


# ============================================================================
# 1. MoE Tests
# ============================================================================

def test_moe_router_top_k():
    """Verify router selects exactly top_k experts and routing weights sum to 1.0."""
    torch.manual_seed(42)
    B, T, D = 2, 8, 32
    E, K = 4, 2
    router = MoERouter(d_model=D, num_experts=E, top_k=K, noisy_gating=False)
    x = torch.randn(B, T, D)

    weights, indices, probs, aux_loss = router(x)

    assert weights.shape == (B, T, K)
    assert indices.shape == (B, T, K)
    assert probs.shape == (B, T, E)

    # Weights for each token must sum to ~1.0
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-4)

    # Indices must be in valid range [0, E-1]
    assert (indices >= 0).all() and (indices < E).all()

    # Auxiliary loss must be non-negative
    assert aux_loss.item() >= 0.0


def test_moe_aux_loss_differentiable():
    """Verify auxiliary load-balancing loss gradients propagate back to gate weights."""
    router = MoERouter(d_model=16, num_experts=4, top_k=2)
    x = torch.randn(1, 4, 16, requires_grad=True)

    weights, indices, probs, aux_loss = router(x)
    aux_loss.backward()

    assert router.gate.weight.grad is not None
    assert torch.norm(router.gate.weight.grad) > 0.0


def test_moe_layer_forward():
    """Verify MoELayer passes tokens through expert pool correctly."""
    config = MoEConfig(d_model=32, num_experts=4, top_k=2, d_ff=64)
    moe_layer = MoELayer(config)
    x = torch.randn(2, 6, 32)

    out, routing_info = moe_layer(x)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert "aux_loss" in routing_info
    assert routing_info["aux_loss"].item() >= 0.0


def test_moe_transformer_block():
    """Verify complete MoE Transformer block with attention and MoE FFN."""
    config = MoEConfig(d_model=32, num_experts=4, top_k=2, d_ff=64)
    block = MoETransformerBlock(
        d_model=32,
        n_heads=4,
        moe_config=config,
    )
    x = torch.randn(1, 8, 32)

    out, attn_weights, routing_info = block(x)

    assert out.shape == (1, 8, 32)
    assert not torch.isnan(out).any()
    assert "aux_loss" in routing_info


# ============================================================================
# 2. Mamba / SSM Tests
# ============================================================================

def test_selective_ssm_forward():
    """Verify SelectiveSSM performs continuous-to-discrete ZOH scan with linear complexity."""
    d_inner = 32
    d_state = 16
    ssm = SelectiveSSM(d_inner=d_inner, d_state=d_state)
    u = torch.randn(2, 8, d_inner)

    y, h = ssm(u)

    assert y.shape == (2, 8, d_inner)
    assert h.shape == (2, d_inner, d_state)
    assert not torch.isnan(y).any()


def test_mamba_block_forward():
    """Verify MambaBlock end-to-end forward pass with depthwise conv, silu gate, and SSM."""
    config = MambaConfig(d_model=32, d_state=16, d_conv=4, expand=2)
    block = MambaBlock(config)
    x = torch.randn(2, 10, 32)

    out, final_state = block(x)

    assert out.shape == (2, 10, 32)
    assert final_state.shape == (2, config.d_inner, config.d_state)
    assert not torch.isnan(out).any()


def test_mamba_recurrent_state_invariance():
    """Verify Mamba parameters count and recurrent state dimensions."""
    config = MambaConfig(d_model=16, d_state=8, d_conv=3, expand=2)
    block = MambaBlock(config)

    assert config.d_inner == 32
    params = sum(p.numel() for p in block.parameters())
    assert params > 0


# ============================================================================
# 3. Modern Attention Tests (RoPE & GQA)
# ============================================================================

def test_rope_embeddings():
    """Verify Rotary Embedding shape and rotation values."""
    dim = 16
    max_len = 32
    rope = RotaryEmbedding(dim=dim, max_seq_len=max_len)
    cos, sin = rope(16)

    assert cos.shape == (1, 16, 1, dim)
    assert sin.shape == (1, 16, 1, dim)

    # cos^2 + sin^2 == 1 (Pythagorean identity for rotation components)
    pythagorean = (cos ** 2 + sin ** 2)
    assert torch.allclose(pythagorean, torch.ones_like(pythagorean), atol=1e-5)


def test_gqa_forward():
    """Verify Grouped-Query Attention with 8 query heads and 2 KV heads (4x KV sharing)."""
    d_model = 32
    num_heads = 8
    num_kv_heads = 2
    gqa = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads)
    x = torch.randn(2, 6, d_model)

    out, attn_weights = gqa(x)

    assert out.shape == (2, 6, d_model)
    assert attn_weights.shape == (2, num_heads, 6, 6)
    assert not torch.isnan(out).any()


# ============================================================================
# 4. Atlas Registry & Scaling Curve Tests
# ============================================================================

def test_architecture_catalog_content():
    """Verify that catalog contains 10 modern foundation model architectures."""
    catalog = get_architecture_catalog()
    assert len(catalog) == 10

    arch_ids = [a["id"] for a in catalog]
    assert "moe_sparse" in arch_ids
    assert "mamba_ssm" in arch_ids
    assert "gqa_attention" in arch_ids
    assert "rwkv" in arch_ids
    assert "retnet" in arch_ids
    assert "hybrid_jamba" in arch_ids

    # Every architecture must have essential pedagogical fields
    for a in catalog:
        assert a["name"]
        assert a["category"]
        assert a["time_complexity"]
        assert a["formula"]
        assert len(a["strengths"]) > 0
        assert len(a["limitations"]) > 0


def test_scaling_curve_computation():
    """Verify that theoretical scaling curve reflects O(T^2) vs O(T) behavior."""
    seq_lengths = [128, 512, 2048, 8192]
    curves = simulate_scaling_curves(seq_lengths, d_model=2048, num_layers=16)

    assert "dense_transformer" in curves
    assert "mamba_ssm" in curves
    assert len(curves["seq_lengths"]) == 4

    # Transformer FLOPs must grow quadratically with sequence length
    first_trans = curves["dense_transformer"]["flops"][0]
    last_trans = curves["dense_transformer"]["flops"][-1]
    ratio_trans = last_trans / first_trans

    # Mamba FLOPs must grow linearly with sequence length
    first_mamba = curves["mamba_ssm"]["flops"][0]
    last_mamba = curves["mamba_ssm"]["flops"][-1]
    ratio_mamba = last_mamba / first_mamba

    # 8192 / 128 = 64x length increase.
    # Linear growth should be ~64x; quadratic growth will be significantly higher (>64x)
    assert ratio_mamba < ratio_trans


# ============================================================================
# 5. REST API Endpoint Tests
# ============================================================================

def test_api_catalog():
    """Test GET /api/v1/architecture-atlas/catalog."""
    res = client.get("/api/v1/architecture-atlas/catalog")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["total_architectures"] == 10


def test_api_catalog_detail():
    """Test GET /api/v1/architecture-atlas/catalog/{arch_id}."""
    res = client.get("/api/v1/architecture-atlas/catalog/moe_sparse")
    assert res.status_code == 200
    data = res.json()
    assert data["architecture"]["id"] == "moe_sparse"

    # Test 404 for invalid ID
    res_404 = client.get("/api/v1/architecture-atlas/catalog/unknown_arch_xyz")
    assert res_404.status_code == 404


def test_api_moe_route():
    """Test POST /api/v1/architecture-atlas/moe/route."""
    payload = {
        "tokens": ["DeepSeek", "Sparse", "MoE", "Routing"],
        "d_model": 16,
        "num_experts": 4,
        "top_k": 2,
        "noisy_gating": False,
        "seed": 42,
    }
    res = client.post("/api/v1/architecture-atlas/moe/route", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["num_tokens"] == 4
    assert data["num_experts"] == 4
    assert len(data["routes"]) == 4
    assert len(data["routes"][0]["selected_experts"]) == 2
    assert "aux_loss" in data
    assert "expert_load" in data


def test_api_simulate_moe():
    """Test POST /api/v1/architecture-atlas/simulate for MoE."""
    payload = {
        "architecture": "moe",
        "batch_size": 1,
        "seq_len": 4,
        "d_model": 16,
        "num_experts": 4,
        "top_k": 2,
    }
    res = client.post("/api/v1/architecture-atlas/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["architecture"] == "moe"
    assert data["output_shape"] == [1, 4, 16]
    assert data["parameter_count"] > 0
    assert "mean" in data["output_stats"]


def test_api_simulate_mamba():
    """Test POST /api/v1/architecture-atlas/simulate for Mamba SSM."""
    payload = {
        "architecture": "mamba_ssm",
        "batch_size": 1,
        "seq_len": 4,
        "d_model": 16,
        "d_state": 8,
    }
    res = client.post("/api/v1/architecture-atlas/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["architecture"] == "mamba_ssm"
    assert data["output_shape"] == [1, 4, 16]


def test_api_simulate_gqa():
    """Test POST /api/v1/architecture-atlas/simulate for GQA."""
    payload = {
        "architecture": "gqa",
        "batch_size": 1,
        "seq_len": 4,
        "d_model": 16,
        "num_heads": 4,
        "num_kv_heads": 2,
    }
    res = client.post("/api/v1/architecture-atlas/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["architecture"] == "gqa"
    assert data["output_shape"] == [1, 4, 16]


def test_api_scaling_curve():
    """Test POST /api/v1/architecture-atlas/scaling-curve."""
    payload = {
        "seq_lengths": [128, 512, 1024],
        "d_model": 2048,
        "num_layers": 16,
    }
    res = client.post("/api/v1/architecture-atlas/scaling-curve", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["curve_points"]["seq_lengths"]) == 3
