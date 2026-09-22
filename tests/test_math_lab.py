"""
tests/test_math_lab.py

Unit and integration test suite for Math Lab (Linear Algebra & Calculus Simulators).
Tests:
- LinearAlgebraSimulator: matrix multiply, transpose, inverse, determinant, eigenvalues, vector dot product
- CalculusSimulator: numerical derivatives, gradient descent, chain rule breakdown
- REST API Endpoints: /api/v1/math-lab/*
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np

from backend.main import app
from src.simulators.math_operations import (
    LinearAlgebraSimulator,
    CalculusSimulator,
)


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================================
# Linear Algebra Simulator Unit Tests
# ============================================================================

def test_matrix_multiply_basic_and_detailed():
    """Temel 2x2 ve detaylı hücre ayrıştırmalı matris çarpımı."""
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    
    res = LinearAlgebraSimulator.matrix_multiply(a, b, detailed_cell=(0, 1))
    assert res.success is True
    assert res.result == [[19.0, 22.0], [43.0, 50.0]]
    assert res.properties["result_shape"] == [2, 2]
    assert len(res.steps) >= 3
    
    # Detaylı hücre adımı kontrolü
    detailed_steps = [s for s in res.steps if s.get("step") == 2]
    assert len(detailed_steps) == 1
    assert "Hücre [0,1]" in detailed_steps[0]["title"]
    assert len(detailed_steps[0]["cell_calculation"]) == 2


def test_matrix_multiply_dimension_mismatch():
    """Uyumsuz boyutlu matris çarpımı hata döndürmelidir."""
    a = [[1.0, 2.0, 3.0]]  # 1x3
    b = [[1.0, 2.0], [3.0, 4.0]]  # 2x2
    res = LinearAlgebraSimulator.matrix_multiply(a, b)
    assert res.success is False
    assert "uyumsuz" in (res.error or "").lower()


def test_matrix_transpose_and_symmetry():
    """Matris transpozu ve simetri tespiti."""
    # Simetrik matris
    sym = [[2.0, 3.0], [3.0, 5.0]]
    res_sym = LinearAlgebraSimulator.matrix_transpose(sym)
    assert res_sym.success is True
    assert res_sym.result == sym
    assert res_sym.properties["is_symmetric"] is True

    # Asimetrik matris
    asym = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    res_asym = LinearAlgebraSimulator.matrix_transpose(asym)
    assert res_asym.success is True
    assert res_asym.properties["transposed_shape"] == [3, 2]
    assert res_asym.properties["is_symmetric"] is False


def test_matrix_inverse_and_singular():
    """Matris tersi ve tekil matris (singular matrix) tespiti."""
    # Tersinir 2x2 matris
    mat = [[4.0, 7.0], [2.0, 6.0]]
    res = LinearAlgebraSimulator.matrix_inverse(mat)
    assert res.success is True
    assert res.result is not None
    # M * M^(-1) == I
    m_inv = np.array(res.result)
    m = np.array(mat)
    assert np.allclose(np.dot(m, m_inv), np.eye(2), atol=1e-5)

    # Tekil matris (det = 0)
    sing = [[1.0, 2.0], [2.0, 4.0]]
    res_sing = LinearAlgebraSimulator.matrix_inverse(sing)
    assert res_sing.success is False
    assert "tekil" in (res_sing.error or "").lower()


def test_matrix_determinant():
    """Determinant hesabı (2x2 ve 3x3)."""
    # 2x2
    mat2 = [[3.0, 8.0], [4.0, 6.0]]
    det2 = LinearAlgebraSimulator.matrix_determinant(mat2)
    assert det2["success"] is True
    assert pytest.approx(det2["determinant"]) == -14.0

    # 3x3
    mat3 = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]
    det3 = LinearAlgebraSimulator.matrix_determinant(mat3)
    assert det3["success"] is True
    assert pytest.approx(det3["determinant"]) == 1.0


def test_matrix_eigenvalues():
    """Öz değer ve öz vektör hesaplaması."""
    # Simetrik 2x2 matris
    mat = [[2.0, 1.0], [1.0, 2.0]]
    res = LinearAlgebraSimulator.eigenvalues_eigenvectors(mat)
    assert res["success"] is True
    evals = res["eigenvalues"]
    assert len(evals) == 2
    # lambda_1 = 3, lambda_2 = 1
    assert pytest.approx(sorted(evals)) == [1.0, 3.0]
    assert pytest.approx(res["trace"]) == 4.0


def test_vector_dot_product_and_angle():
    """Vektör nokta çarpımı, açı ve kosinüs benzerliği."""
    # Dik (orthogonal) vektörler: [1, 0] ve [0, 1]
    res_ortho = LinearAlgebraSimulator.vector_dot_product([1.0, 0.0], [0.0, 1.0])
    assert res_ortho.success is True
    assert res_ortho.result is not None
    assert pytest.approx(res_ortho.result[0]) == 0.0
    assert pytest.approx(res_ortho.properties["angle_degrees"]) == 90.0
    assert pytest.approx(res_ortho.properties["cosine_similarity"]) == 0.0

    # Paralel vektörler: [2, 0] ve [4, 0]
    res_par = LinearAlgebraSimulator.vector_dot_product([2.0, 0.0], [4.0, 0.0])
    assert res_par.success is True
    assert res_par.result is not None
    assert pytest.approx(res_par.result[0]) == 8.0
    assert pytest.approx(res_par.properties["angle_degrees"]) == 0.0
    assert pytest.approx(res_par.properties["cosine_similarity"]) == 1.0


# ============================================================================
# Calculus Simulator Unit Tests
# ============================================================================

def test_calculus_compute_derivative():
    """Türev hesaplayıcı eğri üretimi."""
    res = CalculusSimulator.compute_derivative("quadratic", -3.0, 3.0, 61)
    assert res.success is True
    assert len(res.x_values) == 61
    assert len(res.dy_dx_values) == 61
    assert "x²" in res.formula
    assert "2x" in res.derivative_formula

    # x = 0 noktasında f'(0) == 0
    mid_idx = 30
    assert pytest.approx(res.x_values[mid_idx], abs=1e-5) == 0.0
    assert pytest.approx(res.dy_dx_values[mid_idx], abs=1e-5) == 0.0


def test_calculus_gradient_descent():
    """1D Gradiyen inişi algoritması optimizasyon testi."""
    res = CalculusSimulator.gradient_descent(
        function_name="quadratic",
        initial_x=3.0,
        learning_rate=0.1,
        max_iterations=100,
        tolerance=1e-5
    )
    assert res.success is True
    assert len(res.history) > 0
    assert res.converged is True
    assert abs(res.final_x) < 1e-4
    assert abs(res.final_y) < 1e-4


def test_calculus_chain_rule_breakdown():
    """Zincir kuralı adım adım ayrıştırması."""
    # f(u) = u^2, g(x) = 2x + 1, x = 1 -> g(1) = 3, f(3) = 9, df/dx = 2*u * 2 = 12
    res = CalculusSimulator.chain_rule_breakdown(
        outer_function="square",
        inner_function="linear",
        x_value=1.0
    )
    assert res["success"] is True
    assert pytest.approx(res["x_value"]) == 1.0
    assert len(res["steps"]) == 3
    # Step 1: inner derivative = 2
    assert pytest.approx(res["steps"][0]["derivative_at_x"]) == 2.0
    # Step 2: outer derivative = 2 * 3 = 6
    assert pytest.approx(res["steps"][1]["derivative_at_g_x"]) == 6.0
    # Step 3: total derivative = 6 * 2 = 12
    assert pytest.approx(res["steps"][2]["total_derivative"]) == 12.0


# ============================================================================
# Math Lab REST API Integration Tests
# ============================================================================

def test_api_matrix_multiply(client):
    """POST /api/v1/math-lab/matrix/multiply endpoint testi."""
    payload = {
        "matrix_a": [[1.0, 2.0], [3.0, 4.0]],
        "matrix_b": [[2.0, 0.0], [1.0, 2.0]],
        "detailed_cell": [0, 0]
    }
    response = client.post("/api/v1/math-lab/matrix/multiply", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["result"] == [[4.0, 4.0], [10.0, 8.0]]


def test_api_matrix_transpose(client):
    """POST /api/v1/math-lab/matrix/transpose endpoint testi."""
    payload = {
        "matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    }
    response = client.post("/api/v1/math-lab/matrix/transpose", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["result"] == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


def test_api_matrix_inverse(client):
    """POST /api/v1/math-lab/matrix/inverse endpoint testi."""
    payload = {
        "matrix": [[2.0, 0.0], [0.0, 4.0]]
    }
    response = client.post("/api/v1/math-lab/matrix/inverse", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["result"] == [[0.5, 0.0], [0.0, 0.25]]


def test_api_vector_dot_product(client):
    """POST /api/v1/math-lab/vector/dot endpoint testi."""
    payload = {
        "vector_a": [3.0, 4.0],
        "vector_b": [3.0, 4.0]
    }
    response = client.post("/api/v1/math-lab/vector/dot", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert pytest.approx(data["result"]) == 25.0


def test_api_calculus_derivative(client):
    """POST /api/v1/math-lab/calculus/derivative endpoint testi."""
    payload = {
        "function_name": "sin",
        "x_min": -3.14,
        "x_max": 3.14,
        "num_points": 50
    }
    response = client.post("/api/v1/math-lab/calculus/derivative", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["x_values"]) == 50


def test_api_calculus_gradient_descent(client):
    """POST /api/v1/math-lab/calculus/gradient-descent endpoint testi."""
    payload = {
        "function_name": "quadratic",
        "initial_x": 2.5,
        "learning_rate": 0.08,
        "max_iterations": 40
    }
    response = client.post("/api/v1/math-lab/calculus/gradient-descent", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["history"]) > 0


def test_api_calculus_chain_rule(client):
    """POST /api/v1/math-lab/calculus/chain-rule endpoint testi."""
    payload = {
        "outer_function": "square",
        "inner_function": "linear",
        "x_value": 2.0
    }
    response = client.post("/api/v1/math-lab/calculus/chain-rule", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "steps" in data


def test_api_presets(client):
    """GET /api/v1/math-lab/presets/* endpoint testi."""
    resp_m = client.get("/api/v1/math-lab/presets/matrices")
    assert resp_m.status_code == 200
    assert len(resp_m.json()) >= 3

    resp_v = client.get("/api/v1/math-lab/presets/vectors")
    assert resp_v.status_code == 200
    assert len(resp_v.json()) >= 2
