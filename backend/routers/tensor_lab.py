"""
backend/routers/tensor_lab.py

Tensor & Math Lab API Endpoints
Local AI Research Lab - Developed by Kenan AY

Provides interactive REST endpoints for:
- Tensor shape, strides, and multi-precision memory footprint calculations
- Reshape and transpose (contiguity) simulations
- Broadcasting rules and 2D matrix broadcasting simulations
- General Matrix Multiplication (GEMM) with cell-level dot-product math breakdown
- Activation function curves, temperature-scaled Softmax, and Shannon entropy
- 2-layer MLP computation graph forward and backward autograd gradient flow
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import logging

from src.simulators.tensor_math import (
    TensorShapeAnalyzer,
    BroadcastingEngine,
    MatMulVisualizer,
    ActivationSimulator,
    AutogradGraphSimulator,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/tensor-lab",
    tags=["tensor-lab"]
)


# ============================================================================
# Request / Response Schemas
# ============================================================================

class ShapeAnalyzeRequest(BaseModel):
    shape: List[int] = Field(..., min_length=1, max_length=8, description="Tensör boyutları [d0, d1, ...]")
    dtype: str = Field("float32", description="Veri tipi (float32, float16, bfloat16, int8, int64)")


class ReshapeRequest(BaseModel):
    original_shape: List[int] = Field(..., min_length=1, max_length=8)
    target_shape: List[int] = Field(..., min_length=1, max_length=8)


class TransposeRequest(BaseModel):
    shape: List[int] = Field(..., min_length=1, max_length=8)
    permutation: List[int] = Field(..., min_length=1, max_length=8)


class BroadcastCheckRequest(BaseModel):
    shape_a: List[int] = Field(..., min_length=1, max_length=8)
    shape_b: List[int] = Field(..., min_length=1, max_length=8)


class BroadcastSimulateRequest(BaseModel):
    matrix_a: List[List[float]] = Field(..., description="2D Matris A")
    matrix_b: List[List[float]] = Field(..., description="2D Matris B")
    operation: str = Field("add", description="add, mul, veya sub")


class MatMulRequest(BaseModel):
    matrix_a: List[List[float]] = Field(..., description="A Matrisi (M x K)")
    matrix_b: List[List[float]] = Field(..., description="B Matrisi (K x N)")
    selected_row: Optional[int] = Field(0, description="Detaylı gösterilecek satır indisi i")
    selected_col: Optional[int] = Field(0, description="Detaylı gösterilecek sütun indisi j")


class ActivationCurveRequest(BaseModel):
    activation: str = Field("gelu", description="gelu, silu, relu, sigmoid, tanh, softmax")
    x_min: float = Field(-4.0, ge=-20.0, le=0.0)
    x_max: float = Field(4.0, ge=0.0, le=20.0)
    num_points: int = Field(81, ge=11, le=201)
    temperature: float = Field(1.0, ge=0.01, le=10.0)


class SoftmaxTemperatureRequest(BaseModel):
    logits: List[float] = Field(..., min_length=1, max_length=32)
    temperature: float = Field(1.0, ge=0.01, le=10.0)


class BackpropSimulateRequest(BaseModel):
    x: List[float] = Field(..., min_length=1, max_length=16, description="Giriş vektörü")
    y_target: List[float] = Field(..., min_length=1, max_length=16, description="Hedef vektör")
    hidden_dim: int = Field(3, ge=1, le=16, description="Gizli katman nöron sayısı")
    activation: str = Field("relu", description="relu, gelu, sigmoid, tanh")
    seed: int = Field(42, description="Rastgele ağırlık başlatma tohumu")


# ============================================================================
# Shape & Memory Endpoints
# ============================================================================

@router.post("/shape")
async def analyze_tensor_shape(request: ShapeAnalyzeRequest) -> Dict[str, Any]:
    """Tensör boyutlarını, stride'ları, veri tiplerine göre bellek kullanımını analiz eder."""
    try:
        return TensorShapeAnalyzer.analyze_shape(request.shape, request.dtype)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Shape analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Şekil analiz hatası: {str(e)}")


@router.post("/reshape")
async def simulate_reshape(request: ReshapeRequest) -> Dict[str, Any]:
    """Tensör yeniden şekillendirme (reshape/-1) kurallarını doğrular."""
    result = TensorShapeAnalyzer.simulate_reshape(request.original_shape, request.target_shape)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Geçersiz reshape işlemi."))
    return result


@router.post("/transpose")
async def simulate_transpose(request: TransposeRequest) -> Dict[str, Any]:
    """Tensör transpoze / permütasyon ve bellek sürekliliği (contiguity) durumunu hesaplar."""
    result = TensorShapeAnalyzer.simulate_transpose(request.shape, request.permutation)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Geçersiz transpoze işlemi."))
    return result


# ============================================================================
# Broadcasting Endpoints
# ============================================================================

@router.post("/broadcast")
async def analyze_broadcasting(request: BroadcastCheckRequest) -> Dict[str, Any]:
    """NumPy/PyTorch broadcasting kurallarını sağdan sola adım adım analiz eder."""
    return BroadcastingEngine.analyze_broadcast(request.shape_a, request.shape_b)


@router.post("/broadcast/simulate")
async def simulate_broadcasting_operation(request: BroadcastSimulateRequest) -> Dict[str, Any]:
    """İki 2D matrisi broadcast ederek belirtilen işlemi (add, mul, sub) uygular."""
    result = BroadcastingEngine.simulate_2d_broadcast(
        request.matrix_a, request.matrix_b, request.operation
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Broadcasting işlemi başarısız."))
    return result


# ============================================================================
# GEMM / Matrix Multiplication Endpoints
# ============================================================================

@router.post("/matmul")
async def multiply_matrices(request: MatMulRequest) -> Dict[str, Any]:
    """
    İki matrisi çarpar, FLOP ve bellek bant genişliği hesaplar,
    seçili hücre için nokta çarpımı adım adım döker.
    """
    result = MatMulVisualizer.multiply(
        request.matrix_a,
        request.matrix_b,
        selected_row=request.selected_row,
        selected_col=request.selected_col,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Matris çarpımı başarısız."))
    return result


@router.get("/matmul/sample")
async def get_matmul_sample(
    m: int = Query(3, ge=1, le=8),
    k: int = Query(3, ge=1, le=8),
    n: int = Query(3, ge=1, le=8),
    preset: str = Query("simple", description="simple, sparse, gaussian"),
) -> Dict[str, Any]:
    """Örnek test matrisleri üretir."""
    return MatMulVisualizer.generate_sample(m=m, k=k, n=n, preset=preset)


# ============================================================================
# Activation & Gradient Endpoints
# ============================================================================

@router.get("/activations")
async def list_activations() -> List[Dict[str, str]]:
    """Desteklenen aktivasyon fonksiyonlarını listeler."""
    return ActivationSimulator.get_supported_functions()


@router.post("/activation/curve")
async def get_activation_curve(request: ActivationCurveRequest) -> Dict[str, Any]:
    """Aktivasyon fonksiyonu eğrisini ve türevini (gradyanını) döner."""
    return ActivationSimulator.compute_curve(
        activation=request.activation,
        x_min=request.x_min,
        x_max=request.x_max,
        num_points=request.num_points,
        temperature=request.temperature,
    )


@router.post("/activation/softmax")
async def compute_temperature_softmax(request: SoftmaxTemperatureRequest) -> Dict[str, Any]:
    """Sıcaklık katsayısı (tau) ile Softmax ve Shannon entropisi hesaplar."""
    return ActivationSimulator.simulate_temperature_softmax(
        logits=request.logits,
        temperature=request.temperature,
    )


# ============================================================================
# Autograd & Computation Graph Endpoints
# ============================================================================

@router.post("/backprop/simulate")
async def simulate_autograd(request: BackpropSimulateRequest) -> Dict[str, Any]:
    """
    2-katmanlı MLP ağında ileri besleme (forward pass) ve geri yayılım (backward pass)
    hesaplayarak tüm düğüm tensörlerini ve gradyanlarını döner.
    """
    try:
        return AutogradGraphSimulator.simulate(
            x=request.x,
            y_target=request.y_target,
            hidden_dim=request.hidden_dim,
            activation=request.activation,
            seed=request.seed,
        )
    except Exception as e:
        logger.error(f"Autograd simulation error: {e}")
        raise HTTPException(status_code=500, detail=f"Autograd simülasyon hatası: {str(e)}")
