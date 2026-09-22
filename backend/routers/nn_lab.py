"""
backend/routers/nn_lab.py

Neural Network & Backprop Lab REST API Router
Local AI Research Lab - Kenan AY

Endpoints:
- POST /api/v1/nn-lab/mlp/simulate: Multi-layer perceptron forward & backward autograd simulation
- POST /api/v1/nn-lab/optimizers/race: Multi-optimizer convergence race on 2D loss landscapes
- GET  /api/v1/nn-lab/landscapes: Available 2D test surfaces & formulas
- GET  /api/v1/nn-lab/activations: Activation function curves & derivative dynamics
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import numpy as np
import torch
import logging

from src.simulators.nn_backprop import MLPSimulator, OptimizerRaceSimulator

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/nn-lab",
    tags=["nn-lab"]
)


# ============================================================================
# Request / Response Schemas
# ============================================================================

class MLPSimulateRequest(BaseModel):
    inputs: List[float] = Field(default=[0.8, -0.5], min_length=1, max_length=6, description="Giriş vektörü x")
    targets: List[float] = Field(default=[1.0, 0.0], min_length=1, max_length=4, description="Hedef vektör y")
    hidden_dims: Optional[List[int]] = Field(default=[4, 4], description="Gizli katman boyutları [h1, h2]")
    activation: str = Field(default="relu", description="Aktivasyon fonksiyonu (relu, gelu, swiglu, sigmoid, tanh, leaky_relu)")
    loss_function: str = Field(default="mse", description="Kayıp fonksiyonu (mse, cross_entropy, binary_cross_entropy, smooth_l1)")
    learning_rate: float = Field(default=0.05, ge=0.0001, le=2.0, description="Öğrenme oranı (eta)")
    seed: int = Field(default=42, description="Rastgele ağırlık tohumu")


class OptimizerRaceRequest(BaseModel):
    landscape: str = Field(default="quadratic_bowl", description="Yüzey anahtarı: saddle, rosenbrock, beale, quadratic_bowl")
    optimizers: Optional[List[str]] = Field(
        default=["sgd", "momentum", "rmsprop", "adam", "adamw"],
        description="Yarışacak optimizer'lar (sgd, momentum, rmsprop, adam, adamw)"
    )
    start_x: Optional[float] = Field(default=None, description="Başlangıç x koordinatı")
    start_y: Optional[float] = Field(default=None, description="Başlangıç y koordinatı")
    learning_rate: float = Field(default=0.02, ge=0.0001, le=1.0, description="Öğrenme oranı")
    momentum: float = Field(default=0.9, ge=0.0, le=0.99, description="Momentum katsayısı (beta1)")
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.5, description="Ağırlık sönümü (weight decay)")
    steps: int = Field(default=60, ge=10, le=120, description="Yarış adım sayısı")


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/mlp/simulate")
async def simulate_mlp(request: MLPSimulateRequest) -> Dict[str, Any]:
    """
    Çok katmanlı yapay sinir ağında tam ileri ve geri yayılımı simüle eder.
    Tüm nöronların pre/post aktivasyonlarını, sinaps ağırlıklarını, analitik
    gradiyenleri, zincir kuralı adımlarını ve ölü nöron teşhisini döner.
    """
    try:
        res = MLPSimulator.simulate(
            inputs=request.inputs,
            targets=request.targets,
            hidden_dims=request.hidden_dims,
            activation=request.activation,
            loss_function=request.loss_function,
            learning_rate=request.learning_rate,
            seed=request.seed,
        )
        return res
    except Exception as e:
        logger.error(f"MLP simulation error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"MLP simülasyonu başarısız oldu: {str(e)}")


@router.post("/optimizers/race")
async def run_optimizer_race(request: OptimizerRaceRequest) -> Dict[str, Any]:
    """
    2D kayıp yüzeyinde birden çok optimizasyon algoritmasını (SGD, Momentum, RMSprop, Adam, AdamW)
    aynı anda yarıştırır ve izledikleri yörüngeleri döner.
    """
    try:
        start_pos = (request.start_x, request.start_y) if request.start_x is not None and request.start_y is not None else None
        res = OptimizerRaceSimulator.run_race(
            landscape_key=request.landscape,
            optimizers=request.optimizers,
            start_pos=start_pos,
            learning_rate=request.learning_rate,
            momentum_beta=request.momentum,
            weight_decay=request.weight_decay,
            steps=request.steps,
        )
        return res
    except Exception as e:
        logger.error(f"Optimizer race error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Optimizer yarışı başarısız oldu: {str(e)}")


@router.get("/landscapes")
async def get_landscapes() -> Dict[str, Any]:
    """
    Desteklenen 2D test yüzeylerinin listesini ve parametrelerini döner.
    """
    landscapes_info = {}
    for key, land in OptimizerRaceSimulator.LANDSCAPES.items():
        landscapes_info[key] = {
            "key": key,
            "name": land["name"],
            "formula": land["formula"],
            "description": land["description"],
            "default_start": land["default_start"],
            "optimum": land["optimum"],
            "x_range": land["x_range"],
            "y_range": land["y_range"],
        }
    return {
        "landscapes": landscapes_info,
        "optimizers": OptimizerRaceSimulator.OPTIMIZER_CONFIGS,
    }


@router.get("/activations")
async def get_activation_curves(
    num_points: int = Query(80, ge=20, le=200, description="Eğri çözünürlüğü")
) -> Dict[str, Any]:
    """
    Aktivasyon fonksiyonlarının f(x) ve f'(x) değerlerini ve
    doygunluk (vanishing gradient) risk bölgelerini döner.
    """
    x_vals = np.linspace(-4.0, 4.0, num_points).tolist()
    activations = ["relu", "gelu", "sigmoid", "tanh", "leaky_relu", "swiglu"]

    results = {}
    for act in activations:
        fx = []
        dfx = []
        for x in x_vals:
            t_x = torch.tensor([x], dtype=torch.float32, requires_grad=True)
            y = MLPSimulator.apply_activation(t_x, act)
            y.backward()
            grad_val = float(t_x.grad.item()) if t_x.grad is not None else 0.0

            fx.append(round(float(y.item()), 4))
            dfx.append(round(grad_val, 4))

        # Doygunluk / Kaybolan gradiyen analizi (|f'(x)| < 0.05)
        vanishing_ranges = []
        if act == "sigmoid":
            vanishing_ranges.append("x < -3.0 (Alt doygunluk, f' ≈ 0)")
            vanishing_ranges.append("x > 3.0 (Üst doygunluk, f' ≈ 0)")
        elif act == "tanh":
            vanishing_ranges.append("x < -2.5 (Negatif doygunluk)")
            vanishing_ranges.append("x > 2.5 (Pozitif doygunluk)")
        elif act == "relu":
            vanishing_ranges.append("x < 0 (Ölü ReLU bölgesi, f' = 0)")
        elif act == "leaky_relu":
            vanishing_ranges.append("x < 0 (Sızıntı bölgesi, f' = 0.1)")

        results[act] = {
            "name": act.upper(),
            "derivative_formula": MLPSimulator.activation_derivative_formula(act),
            "fx": fx,
            "dfx": dfx,
            "max_derivative": round(float(max(dfx)), 4),
            "vanishing_notes": vanishing_ranges,
        }

    return {
        "x_vals": [round(x, 3) for x in x_vals],
        "activations": results,
    }
