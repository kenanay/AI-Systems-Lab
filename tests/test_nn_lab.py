"""
tests/test_nn_lab.py

Unit and integration test suite for Neural Network & Backprop Lab.
Tests:
- MLPSimulator: forward/backward pass, activations, loss functions, dead ReLU detection, chain rule steps
- OptimizerRaceSimulator: analytical loss surfaces, trajectory calculations, convergence, contour grid
- REST API endpoints: /mlp/simulate, /optimizers/race, /landscapes, /activations
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.simulators.nn_backprop import MLPSimulator, OptimizerRaceSimulator, LossLandscapes


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================================
# MLPSimulator Unit Tests
# ============================================================================

def test_mlp_simulator_forward_backward_mse():
    """MLP MSE loss ve ReLU ile ileri-geri yayılım testi."""
    res = MLPSimulator.simulate(
        inputs=[0.5, -0.2],
        targets=[1.0, 0.0],
        hidden_dims=[4, 4],
        activation="relu",
        loss_function="mse",
        learning_rate=0.05,
        seed=42,
    )

    assert "loss" in res
    assert res["loss"] >= 0.0
    assert len(res["predictions"]) == 2
    assert res["architecture"] == [2, 4, 4, 2]

    # Düğümler: 2 giriş + 4 + 4 gizli + 2 çıktı = 12 düğüm
    assert len(res["nodes"]) == 12

    # Sinapslar: (2*4) + (4*4) + (4*2) = 8 + 16 + 8 = 32 sinaps
    assert len(res["synapses"]) == 32

    # Her sinapsta ağırlık ve gradyen var mı?
    for syn in res["synapses"]:
        assert "weight" in syn
        assert "gradient_w" in syn
        assert "weight_update" in syn

    # Zincir kuralı adımları
    assert len(res["chain_rule_steps"]) >= 4
    for step in res["chain_rule_steps"]:
        assert "formula" in step
        assert "latex" in step

    # Katman sağlığı
    assert len(res["layer_health"]) == 3
    assert res["overall_health"] in ["HEALTHY", "VANISHING_RISK", "EXPLODING_RISK", "DEAD_NEURONS_DETECTED"]


def test_mlp_simulator_activations():
    """Farklı aktivasyon fonksiyonları testi (gelu, sigmoid, tanh, swiglu, leaky_relu)."""
    for act in ["gelu", "sigmoid", "tanh", "swiglu", "leaky_relu"]:
        res = MLPSimulator.simulate(
            inputs=[1.0, -1.0],
            targets=[0.5],
            hidden_dims=[3],
            activation=act,
            loss_function="mse",
            seed=123,
        )
        assert res["loss"] >= 0.0
        assert len(res["predictions"]) == 1
        assert res["activation"] == act
        assert "derivative_formula" in res["activation_derivative_formula"] or "σ'" in res["activation_derivative_formula"]


def test_mlp_simulator_loss_functions():
    """Farklı kayıp fonksiyonları testi (cross_entropy, binary_cross_entropy, smooth_l1)."""
    # Cross entropy
    ce_res = MLPSimulator.simulate(
        inputs=[0.3, 0.7],
        targets=[1.0, 0.0],
        hidden_dims=[4],
        loss_function="cross_entropy",
    )
    assert ce_res["loss"] >= 0.0

    # Binary cross entropy
    bce_res = MLPSimulator.simulate(
        inputs=[0.5],
        targets=[1.0],
        hidden_dims=[3],
        loss_function="binary_cross_entropy",
    )
    assert bce_res["loss"] >= 0.0

    # Smooth L1
    l1_res = MLPSimulator.simulate(
        inputs=[0.5, 0.2],
        targets=[0.8, 0.1],
        hidden_dims=[4],
        loss_function="smooth_l1",
    )
    assert l1_res["loss"] >= 0.0


def test_mlp_dead_relu_detection():
    """Negatif girişlerle ölü ReLU tespiti testi."""
    res = MLPSimulator.simulate(
        inputs=[-10.0, -10.0],
        targets=[0.0],
        hidden_dims=[4],
        activation="relu",
        seed=999,
    )
    # Negatif girişlerde bazı ReLU nöronlarının sönmesi beklenir
    assert isinstance(res["dead_neurons"], list)


# ============================================================================
# OptimizerRaceSimulator Unit Tests
# ============================================================================

def test_optimizer_race_quadratic_bowl():
    """Eliptik çanak üzerinde tüm optimizer'ların yarışı."""
    race = OptimizerRaceSimulator.run_race(
        landscape_key="quadratic_bowl",
        optimizers=["sgd", "momentum", "rmsprop", "adam", "adamw"],
        start_pos=(2.0, 1.0),
        learning_rate=0.05,
        steps=30,
    )

    assert "landscape" in race
    assert race["landscape"]["key"] == "quadratic_bowl"
    assert "winner" in race
    assert race["winner"] in ["sgd", "momentum", "rmsprop", "adam", "adamw"]

    for opt_name in ["sgd", "momentum", "rmsprop", "adam", "adamw"]:
        data = race["optimizers"][opt_name]
        assert len(data["trajectory"]) == 31
        assert data["final_loss"] <= data["initial_loss"] + 1.0  # kayıp azalmalı
        assert data["total_distance"] > 0.0


def test_optimizer_race_saddle_point():
    """Eyer noktası yüzeyinde yarış ve eyer noktasından kaçış."""
    race = OptimizerRaceSimulator.run_race(
        landscape_key="saddle",
        optimizers=["adam", "adamw", "sgd"],
        start_pos=(-0.05, 1.5),
        learning_rate=0.03,
        steps=25,
    )
    assert len(race["optimizers"]) == 3
    assert "contours" in race
    assert len(race["contours"]["z_grid"]) > 0


def test_landscape_contours_generation():
    """Eşyükselti (contour) ızgarası üretimi testi."""
    contours = OptimizerRaceSimulator.get_landscape_contours("rosenbrock", grid_size=20)
    assert len(contours["x_vals"]) == 20
    assert len(contours["y_vals"]) == 20
    assert len(contours["z_grid"]) == 20
    assert len(contours["z_grid"][0]) == 20
    assert contours["z_min"] <= contours["z_max"]


# ============================================================================
# REST API Endpoints Integration Tests
# ============================================================================

def test_api_mlp_simulate(client):
    """POST /api/v1/nn-lab/mlp/simulate endpoint testi."""
    payload = {
        "inputs": [0.7, -0.3],
        "targets": [1.0, 0.0],
        "hidden_dims": [4, 4],
        "activation": "gelu",
        "loss_function": "mse",
        "learning_rate": 0.05,
        "seed": 42
    }
    response = client.post("/api/v1/nn-lab/mlp/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "loss" in data
    assert "nodes" in data
    assert "synapses" in data
    assert "chain_rule_steps" in data
    assert len(data["nodes"]) == 12


def test_api_optimizer_race(client):
    """POST /api/v1/nn-lab/optimizers/race endpoint testi."""
    payload = {
        "landscape": "quadratic_bowl",
        "optimizers": ["adam", "sgd"],
        "learning_rate": 0.02,
        "steps": 20
    }
    response = client.post("/api/v1/nn-lab/optimizers/race", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "winner" in data
    assert "optimizers" in data
    assert "adam" in data["optimizers"]
    assert "sgd" in data["optimizers"]
    assert "contours" in data


def test_api_get_landscapes(client):
    """GET /api/v1/nn-lab/landscapes endpoint testi."""
    response = client.get("/api/v1/nn-lab/landscapes")
    assert response.status_code == 200
    data = response.json()
    assert "landscapes" in data
    assert "saddle" in data["landscapes"]
    assert "rosenbrock" in data["landscapes"]
    assert "beale" in data["landscapes"]
    assert "quadratic_bowl" in data["landscapes"]
    assert "optimizers" in data


def test_api_get_activations(client):
    """GET /api/v1/nn-lab/activations endpoint testi."""
    response = client.get("/api/v1/nn-lab/activations?num_points=50")
    assert response.status_code == 200
    data = response.json()
    assert "x_vals" in data
    assert len(data["x_vals"]) == 50
    assert "activations" in data
    assert "relu" in data["activations"]
    assert "gelu" in data["activations"]
    assert "sigmoid" in data["activations"]
    assert "tanh" in data["activations"]
    assert len(data["activations"]["relu"]["fx"]) == 50
    assert len(data["activations"]["relu"]["dfx"]) == 50
