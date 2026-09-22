"""
tests/test_attention_api.py

Integration and unit tests for the Attention Inspection API endpoint (/api/v1/inference/attention).
Validates:
- Successful attention matrix generation
- Matrix shapes matching token sequence length (T x T)
- Causal masking integrity (upper triangle must be 0)
- Layer and Head specific selections
- Empty / validation handling
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_attention_inspect_basic():
    """Test basic attention inspection on input text."""
    response = client.post(
        "/api/v1/inference/attention",
        json={"text": "Yapay zeka sistemleri öğrenir."}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert "tokens" in data
    assert "matrix" in data
    assert "num_layers" in data
    assert "num_heads" in data
    
    tokens = data["tokens"]
    matrix = data["matrix"]
    n_tokens = len(tokens)
    
    assert n_tokens > 0
    assert len(matrix) == n_tokens
    assert len(matrix[0]) == n_tokens
    
    # Verify Causal Masking: upper triangle (future tokens) must have 0 attention
    for i in range(n_tokens):
        for j in range(i + 1, n_tokens):
            assert matrix[i][j] == 0.0, f"Causal mask violated at row {i}, col {j}"
            
    # Verify probability distribution: each row sums to <= 1.05 (allowing float precision)
    for i in range(n_tokens):
        row_sum = sum(matrix[i])
        assert row_sum <= 1.05, f"Attention row {i} sum exceeds 1: {row_sum}"


def test_attention_inspect_specific_layer_and_head():
    """Test inspection with explicit layer and head indices."""
    response = client.post(
        "/api/v1/inference/attention",
        json={
            "text": "Transformer mimarisi attention kullanır.",
            "layer_idx": 1,
            "head_idx": 1
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["selected_layer"] == 1
    assert data["selected_head"] == 1
    assert "all_heads_matrix" in data
    assert "Head 0" in data["all_heads_matrix"]
    assert "Head 1" in data["all_heads_matrix"]


def test_attention_inspect_empty_text_validation():
    """Test that empty string fails validation with 422."""
    response = client.post(
        "/api/v1/inference/attention",
        json={"text": ""}
    )
    assert response.status_code == 422
