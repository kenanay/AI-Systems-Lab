"""
Tests for Positional Encoding & Transformer Architecture Lab simulation engine and FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

from src.simulators.transformer_architecture import (
    PositionalEncodingEngine,
    AttentionVariantsEngine,
    TransformerBlockEngine,
)

client = TestClient(app)


class TestPositionalEncodingEngine:
    def test_sinusoidal_pe_generation(self):
        res = PositionalEncodingEngine.compute_sinusoidal(seq_len=16, d_model=32)
        assert res["seq_len"] == 16
        assert res["d_model"] == 32
        assert len(res["pe_matrix"]) == 16
        assert len(res["pe_matrix"][0]) == 32
        # Position 0 sin dimension should be 0.0, cos dimension should be 1.0
        assert abs(res["pe_matrix"][0][0]) < 1e-4
        assert abs(res["pe_matrix"][0][1] - 1.0) < 1e-4
        assert len(res["similarity_matrix"]) == 16
        assert len(res["waves"]) > 0

    def test_rope_rotation_and_relative_distance(self):
        res = PositionalEncodingEngine.compute_rope(dim=16, max_seq_len=8, base=10000.0)
        assert res["dim"] == 16
        assert res["max_seq_len"] == 8
        assert len(res["positions_data"]) == 8
        assert len(res["theta_values"]) == 8

        # Position 0 angle is 0.0
        assert res["positions_data"][0]["angle_rad"] == 0.0
        # Check relative dot product matrix is 8x8
        assert len(res["relative_dot_matrix"]) == 8
        assert len(res["relative_dot_matrix"][0]) == 8

    def test_alibi_slopes_and_matrices(self):
        res = PositionalEncodingEngine.compute_alibi(num_heads=8, seq_len=6)
        assert res["num_heads"] == 8
        assert res["seq_len"] == 6
        assert len(res["slopes"]) == 8
        assert len(res["heads"]) == 8
        # Diagonal (i == j) distance is 0, so penalty is 0.0
        h0_mat = res["heads"][0]["bias_matrix"]
        assert h0_mat[0][0] == 0.0
        assert h0_mat[1][1] == 0.0
        # Off-diagonal should be negative
        assert h0_mat[0][1] < 0.0

    def test_compare_methods(self):
        res = PositionalEncodingEngine.compare_methods()
        assert len(res) == 4
        names = [m["name"] for m in res]
        assert any("RoPE" in n for n in names)
        assert any("Sinusoidal" in n for n in names)
        assert any("ALiBi" in n for n in names)


class TestAttentionVariantsEngine:
    def test_mha_classification(self):
        res = AttentionVariantsEngine.analyze_kv_cache(
            batch_size=1,
            seq_len=2048,
            num_query_heads=32,
            num_kv_heads=32,
            head_dim=128,
            num_layers=32,
            dtype="float16",
        )
        assert "MHA" in res["variant_type"]
        assert res["queries_per_kv_head"] == 1
        assert res["comparison"]["savings_vs_mha_pct"] == 0.0

    def test_gqa_classification_and_savings(self):
        # 32 Q heads, 8 KV heads -> 4:1 GQA -> 75% memory savings
        res = AttentionVariantsEngine.analyze_kv_cache(
            batch_size=2,
            seq_len=4096,
            num_query_heads=32,
            num_kv_heads=8,
            head_dim=128,
            num_layers=32,
            dtype="float16",
        )
        assert "GQA" in res["variant_type"]
        assert res["queries_per_kv_head"] == 4
        assert res["comparison"]["savings_vs_mha_pct"] == 75.0
        assert "4.0x" in res["comparison"]["savings_multiplier"]
        assert len(res["head_groups"]) == 8

    def test_mqa_classification(self):
        res = AttentionVariantsEngine.analyze_kv_cache(
            batch_size=1,
            seq_len=2048,
            num_query_heads=32,
            num_kv_heads=1,
            head_dim=128,
            num_layers=32,
            dtype="float16",
        )
        assert "MQA" in res["variant_type"]
        assert res["queries_per_kv_head"] == 32
        assert res["comparison"]["savings_vs_mha_pct"] > 90.0


class TestTransformerBlockEngine:
    def test_pre_ln_swiglu_simulation(self):
        res = TransformerBlockEngine.simulate_forward_pass(
            norm_type="rmsnorm",
            ffn_type="swiglu",
            norm_placement="pre_ln",
            batch_size=1,
            seq_len=4,
            d_model=8,
            d_ff=16,
        )
        assert res["norm_type"] == "rmsnorm"
        assert res["ffn_type"] == "swiglu"
        assert len(res["stages"]) >= 5
        stage_names = [s["sublayer_name"] for s in res["stages"]]
        assert "Block Input x" in stage_names
        assert any("Residual Addition" in s for s in stage_names)
        assert any("SwiGLU" in s for s in stage_names)

    def test_post_ln_mlp_simulation(self):
        res = TransformerBlockEngine.simulate_forward_pass(
            norm_type="layernorm",
            ffn_type="standard_mlp",
            norm_placement="post_ln",
            batch_size=1,
            seq_len=4,
            d_model=8,
            d_ff=16,
        )
        assert res["norm_type"] == "layernorm"
        assert res["ffn_type"] == "standard_mlp"
        stage_names = [s["sublayer_name"] for s in res["stages"]]
        assert any("Standard MLP" in s for s in stage_names)

    def test_calculate_architecture_params(self):
        # GPT-2 Small params: ~124M
        res = TransformerBlockEngine.calculate_architecture_params(
            vocab_size=50257,
            d_model=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=12,
            d_ff=3072,
            tie_word_embeddings=True,
            ffn_type="standard_mlp",
        )
        # Should be between 120M and 130M
        assert 120.0 <= res["total_millions"] <= 130.0
        assert "vram_inference" in res
        assert "fp16" in res["vram_inference"]


class TestTransformerLabAPIEndpoints:
    def test_api_sinusoidal(self):
        resp = client.post("/api/v1/transformer-lab/positional-encoding/sinusoidal", json={
            "seq_len": 12,
            "d_model": 16,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["seq_len"] == 12
        assert len(data["pe_matrix"]) == 12

    def test_api_rope(self):
        resp = client.post("/api/v1/transformer-lab/positional-encoding/rope", json={
            "dim": 8,
            "max_seq_len": 6,
            "base": 10000.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["positions_data"]) == 6

    def test_api_alibi(self):
        resp = client.post("/api/v1/transformer-lab/positional-encoding/alibi", json={
            "num_heads": 4,
            "seq_len": 4,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["heads"]) == 4

    def test_api_compare(self):
        resp = client.get("/api/v1/transformer-lab/positional-encoding/compare")
        assert resp.status_code == 200
        assert len(resp.json()) == 4

    def test_api_attention_variants(self):
        resp = client.post("/api/v1/transformer-lab/attention-variants/analyze", json={
            "batch_size": 1,
            "seq_len": 1024,
            "num_query_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 64,
            "num_layers": 16,
            "dtype": "float16",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "GQA" in data["variant_type"]
        assert data["queries_per_kv_head"] == 4

    def test_api_block_simulate(self):
        resp = client.post("/api/v1/transformer-lab/transformer-block/simulate", json={
            "norm_type": "rmsnorm",
            "ffn_type": "swiglu",
            "norm_placement": "pre_ln",
            "batch_size": 1,
            "seq_len": 4,
            "d_model": 8,
            "d_ff": 16,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "stages" in data

    def test_api_params_and_presets(self):
        resp_presets = client.get("/api/v1/transformer-lab/transformer-block/presets")
        assert resp_presets.status_code == 200
        presets = resp_presets.json()
        assert len(presets) >= 4

        p0 = presets[0]
        resp_calc = client.post("/api/v1/transformer-lab/transformer-block/params", json={
            "vocab_size": p0["vocab_size"],
            "d_model": p0["d_model"],
            "n_layers": p0["n_layers"],
            "n_heads": p0["n_heads"],
            "n_kv_heads": p0["n_kv_heads"],
            "d_ff": p0["d_ff"],
            "tie_word_embeddings": p0["tie_word_embeddings"],
            "ffn_type": p0["ffn_type"],
        })
        assert resp_calc.status_code == 200
        calc_data = resp_calc.json()
        assert calc_data["total_parameters"] > 0
