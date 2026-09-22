"""
tests/test_embeddings_api.py

Unit and integration tests for the Embedding Lab API endpoints:
- POST /api/v1/embeddings/project (2D & 3D PCA projection)
- POST /api/v1/embeddings/similarity (Cosine similarity & angular distance)
- POST /api/v1/embeddings/analogy (Vector arithmetic A - B + C = ?)
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_embeddings_projection_2d():
    """Test 2D PCA projection of word embeddings."""
    words = ["bilgisayar", "yazılım", "model", "veri", "algoritma"]
    response = client.post(
        "/api/v1/embeddings/project",
        json={
            "words": words,
            "dimensions": 2,
            "normalize": True
        }
    )
    assert response.status_code == 200
    data = response.json()

    assert data["dimensions"] == 2
    assert "points" in data
    assert len(data["points"]) == len(words)
    assert "explained_variance_ratio" in data
    assert len(data["explained_variance_ratio"]) == 2

    # Check that variance ratio values are valid probabilities
    for r in data["explained_variance_ratio"]:
        assert 0.0 <= r <= 1.0
    assert sum(data["explained_variance_ratio"]) <= 1.05

    for pt in data["points"]:
        assert "text" in pt
        assert "x" in pt
        assert "y" in pt
        assert pt["z"] is None
        assert "norm" in pt
        assert pt["norm"] > 0


def test_embeddings_projection_3d():
    """Test 3D PCA projection of word embeddings."""
    words = ["sıcak", "soğuk", "güneş", "buz", "kar"]
    response = client.post(
        "/api/v1/embeddings/project",
        json={
            "words": words,
            "dimensions": 3,
            "normalize": True
        }
    )
    assert response.status_code == 200
    data = response.json()

    assert data["dimensions"] == 3
    assert len(data["points"]) == len(words)
    assert len(data["explained_variance_ratio"]) == 3

    for pt in data["points"]:
        assert pt["z"] is not None


def test_embeddings_projection_validation():
    """Test that fewer than 2 words returns validation error."""
    response = client.post(
        "/api/v1/embeddings/project",
        json={"words": ["tek"], "dimensions": 2}
    )
    assert response.status_code == 422


def test_cosine_similarity_identical():
    """Test cosine similarity between identical words is 1.0."""
    response = client.post(
        "/api/v1/embeddings/similarity",
        json={
            "word_a": "teknoloji",
            "word_b": "teknoloji"
        }
    )
    assert response.status_code == 200
    data = response.json()

    assert data["cosine_similarity"] == 1.0
    assert data["angle_degrees"] == 0.0
    assert data["euclidean_distance"] == 0.0


def test_cosine_similarity_different_words():
    """Test cosine similarity between different words."""
    response = client.post(
        "/api/v1/embeddings/similarity",
        json={
            "word_a": "öğrenci",
            "word_b": "öğretmen"
        }
    )
    assert response.status_code == 200
    data = response.json()

    assert -1.0 <= data["cosine_similarity"] <= 1.0
    assert 0.0 <= data["angle_degrees"] <= 180.0
    assert data["euclidean_distance"] >= 0.0
    assert "dot_product" in data


def test_vector_analogy():
    """Test vector arithmetic analogy (kral - erkek + kadın = kraliçe)."""
    response = client.post(
        "/api/v1/embeddings/analogy",
        json={
            "word_a": "kral",
            "word_b": "erkek",
            "word_c": "kadın",
            "candidates": ["kraliçe", "prenses", "saray", "kitap"]
        }
    )
    assert response.status_code == 200
    data = response.json()

    assert data["word_a"] == "kral"
    assert data["word_b"] == "erkek"
    assert data["word_c"] == "kadın"
    assert "formula" in data
    assert "top_matches" in data
    assert len(data["top_matches"]) > 0
    for match in data["top_matches"]:
        assert "word" in match
        assert "similarity" in match
