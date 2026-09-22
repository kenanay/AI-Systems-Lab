"""
Tests for Tensor & Math Lab simulation engine and FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

from src.simulators.tensor_math import (
    TensorShapeAnalyzer,
    BroadcastingEngine,
    MatMulVisualizer,
    ActivationSimulator,
    AutogradGraphSimulator,
)

client = TestClient(app)


class TestTensorShapeAnalyzer:
    def test_shape_analysis_fp32(self):
        shape = [2, 3, 4]
        res = TensorShapeAnalyzer.analyze_shape(shape, "float32")
        assert res["rank"] == 3
        assert res["total_elements"] == 24
        assert res["total_bytes"] == 24 * 4
        assert res["element_strides"] == [12, 4, 1]
        assert res["byte_strides"] == [48, 16, 4]
        assert "memory_comparison" in res
        assert res["memory_comparison"]["float16"]["bytes"] == 24 * 2

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            TensorShapeAnalyzer.analyze_shape([2, -1, 4])

    def test_reshape_infer_dimension(self):
        res = TensorShapeAnalyzer.simulate_reshape([2, 3, 4], [2, -1])
        assert res["success"] is True
        assert res["resolved_shape"] == [2, 12]
        assert res["new_strides"] == [12, 1]

    def test_reshape_invalid(self):
        res = TensorShapeAnalyzer.simulate_reshape([2, 3, 4], [5, 5])
        assert res["success"] is False
        assert "mismatch" in res["error"]

    def test_transpose_and_contiguity(self):
        # [2, 3] -> transposed [3, 2]
        res = TensorShapeAnalyzer.simulate_transpose([2, 3], [1, 0])
        assert res["success"] is True
        assert res["transposed_shape"] == [3, 2]
        # Transposed strides should be [1, 3] while contiguous would be [2, 1]
        assert res["is_contiguous"] is False
        assert res["requires_contiguous_call"] is True


class TestBroadcastingEngine:
    def test_broadcast_compatible(self):
        res = BroadcastingEngine.analyze_broadcast([3, 1], [1, 4])
        assert res["compatible"] is True
        assert res["result_shape"] == [3, 4]
        assert len(res["steps"]) == 2

    def test_broadcast_incompatible(self):
        res = BroadcastingEngine.analyze_broadcast([3, 2], [3, 3])
        assert res["compatible"] is False
        assert res["result_shape"] is None
        assert res["error"] is not None

    def test_simulate_2d_broadcast_addition(self):
        matrix_a = [[1.0], [2.0], [3.0]]  # 3x1
        matrix_b = [[10.0, 20.0]]           # 1x2
        res = BroadcastingEngine.simulate_2d_broadcast(matrix_a, matrix_b, "add")
        assert res["success"] is True
        assert res["result_matrix"] == [
            [11.0, 21.0],
            [12.0, 22.0],
            [13.0, 23.0],
        ]


class TestMatMulVisualizer:
    def test_matmul_calculation_and_breakdown(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        res = MatMulVisualizer.multiply(a, b, selected_row=0, selected_col=1)
        assert res["success"] is True
        # C[0, 1] = 1*6 + 2*8 = 6 + 16 = 22
        assert res["matrix_c"] == [[19.0, 22.0], [43.0, 50.0]]
        assert res["cell_value"] == 22.0
        assert len(res["pairwise_terms"]) == 2
        assert res["hardware_metrics"]["total_flops"] == 2 * 2 * 2 * 2  # 2 * M * N * K

    def test_matmul_dimension_mismatch(self):
        a = [[1.0, 2.0, 3.0]]  # 1x3
        b = [[1.0, 2.0], [3.0, 4.0]]  # 2x2
        res = MatMulVisualizer.multiply(a, b)
        assert res["success"] is False
        assert "Inner dimension mismatch" in res["error"]


class TestActivationSimulator:
    def test_supported_functions(self):
        funcs = ActivationSimulator.get_supported_functions()
        ids = [f["id"] for f in funcs]
        assert "gelu" in ids
        assert "silu" in ids
        assert "relu" in ids
        assert "softmax" in ids

    def test_gelu_curve(self):
        res = ActivationSimulator.compute_curve("gelu", x_min=-2.0, x_max=2.0, num_points=21)
        assert res["activation"] == "gelu"
        assert len(res["points"]) == 21
        # GELU(0) should be 0.0
        mid = [p for p in res["points"] if abs(p["x"]) < 1e-4][0]
        assert abs(mid["y"]) < 1e-4

    def test_temperature_softmax(self):
        logits = [2.0, 1.0, 0.0]
        # Low temp -> high certainty
        res_low = ActivationSimulator.simulate_temperature_softmax(logits, temperature=0.1)
        assert res_low["argmax_index"] == 0
        assert res_low["max_probability"] > 0.95

        # High temp -> flatter
        res_high = ActivationSimulator.simulate_temperature_softmax(logits, temperature=5.0)
        assert res_high["entropy_bits"] > res_low["entropy_bits"]


class TestAutogradGraphSimulator:
    def test_mlp_backpropagation(self):
        res = AutogradGraphSimulator.simulate(
            x=[1.0, -1.0],
            y_target=[1.0],
            hidden_dim=3,
            activation="relu",
            seed=42,
        )
        assert res["success"] is True
        assert res["loss"] >= 0.0
        assert len(res["nodes"]) == 9
        assert len(res["edges"]) == 8

        # Verify weights have computed gradients
        w1_node = next(n for n in res["nodes"] if n["id"] == "w1")
        assert w1_node["grad_val"] is not None
        assert len(w1_node["grad_val"]) == 2  # in_dim = 2


class TestTensorLabAPIEndpoints:
    def test_api_shape(self):
        resp = client.post("/api/v1/tensor-lab/shape", json={"shape": [1, 32, 512], "dtype": "bfloat16"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["rank"] == 3
        assert data["total_elements"] == 1 * 32 * 512
        assert data["dtype"] == "bfloat16"

    def test_api_broadcast(self):
        resp = client.post("/api/v1/tensor-lab/broadcast", json={"shape_a": [1, 10], "shape_b": [5, 1]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["compatible"] is True
        assert data["result_shape"] == [5, 10]

    def test_api_matmul(self):
        resp = client.post("/api/v1/tensor-lab/matmul", json={
            "matrix_a": [[1.0, 2.0], [3.0, 4.0]],
            "matrix_b": [[2.0, 0.0], [1.0, 2.0]],
            "selected_row": 1,
            "selected_col": 0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        # C[1, 0] = 3*2 + 4*1 = 10
        assert data["cell_value"] == 10.0

    def test_api_activation_curve(self):
        resp = client.post("/api/v1/tensor-lab/activation/curve", json={"activation": "silu", "num_points": 31})
        assert resp.status_code == 200
        data = resp.json()
        assert data["activation"] == "silu"
        assert len(data["points"]) == 31

    def test_api_softmax_temperature(self):
        resp = client.post("/api/v1/tensor-lab/activation/softmax", json={"logits": [3.0, 1.0, -1.0], "temperature": 0.5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["argmax_index"] == 0

    def test_api_backprop_simulate(self):
        resp = client.post("/api/v1/tensor-lab/backprop/simulate", json={
            "x": [0.5, 0.2],
            "y_target": [1.0],
            "hidden_dim": 2,
            "activation": "gelu",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "nodes" in data
