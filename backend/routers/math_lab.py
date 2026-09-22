"""
backend/routers/math_lab.py

Math Lab API Endpoints

Author: Kenan AY
Location: Kütahya, TÜRKİYE
Version: 1.0.0

Provides interactive REST endpoints for:
- Linear Algebra operations (matrix multiplication, transpose, inverse, determinant)
- Vector operations (dot product, magnitude, orthogonality)
- Calculus operations (derivatives, gradient descent, chain rule)
- Eigenvalue/eigenvector computation
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from src.simulators.math_operations import (
    LinearAlgebraSimulator,
    CalculusSimulator,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/math-lab",
    tags=["math-lab"]
)


# ============================================================================
# Request / Response Schemas
# ============================================================================

class MatrixMultiplyRequest(BaseModel):
    matrix_a: List[List[float]] = Field(..., description="A matrisi")
    matrix_b: List[List[float]] = Field(..., description="B matrisi")
    detailed_cell: Optional[List[int]] = Field(None, description="Detaylı gösterilecek hücre [row, col]")


class MatrixOperationRequest(BaseModel):
    matrix: List[List[float]] = Field(..., description="Girdi matrisi")


class VectorDotProductRequest(BaseModel):
    vector_a: List[float] = Field(..., description="Vektör A")
    vector_b: List[float] = Field(..., description="Vektör B")


class DerivativeRequest(BaseModel):
    function_name: str = Field(..., description="Fonksiyon adı (quadratic, cubic, sin, exp, sigmoid)")
    x_min: float = Field(-5.0, ge=-20.0, le=0.0)
    x_max: float = Field(5.0, ge=0.0, le=20.0)
    num_points: int = Field(100, ge=20, le=500)


class GradientDescentRequest(BaseModel):
    function_name: str = Field(..., description="Optimize edilecek fonksiyon")
    initial_x: float = Field(2.0, ge=-10.0, le=10.0)
    learning_rate: float = Field(0.1, ge=0.001, le=1.0)
    max_iterations: int = Field(50, ge=1, le=200)
    tolerance: float = Field(1e-6, ge=1e-10, le=0.01)


class ChainRuleRequest(BaseModel):
    outer_function: str = Field(..., description="Dış fonksiyon (square, sin, exp)")
    inner_function: str = Field(..., description="İç fonksiyon (linear, quadratic, cubic)")
    x_value: float = Field(1.0, description="Hesaplama noktası")


# ============================================================================
# Linear Algebra Endpoints
# ============================================================================

@router.post("/matrix/multiply")
async def multiply_matrices(request: MatrixMultiplyRequest) -> Dict[str, Any]:
    """
    İki matrisi çarpar ve adım adım açıklama sağlar.
    C = A × B hesaplaması yapar.
    """
    try:
        detailed_cell = tuple(request.detailed_cell) if request.detailed_cell else None
        result = LinearAlgebraSimulator.matrix_multiply(
            request.matrix_a,
            request.matrix_b,
            detailed_cell=detailed_cell
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "success": True,
            "result": result.result,
            "steps": result.steps,
            "properties": result.properties
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Matrix multiply error: {e}")
        raise HTTPException(status_code=500, detail=f"Matris çarpımı hatası: {str(e)}")


@router.post("/matrix/transpose")
async def transpose_matrix(request: MatrixOperationRequest) -> Dict[str, Any]:
    """
    Matris transpozisyonu: satırlar ve sütunlar yer değiştirir.
    M^T hesaplaması yapar.
    """
    try:
        result = LinearAlgebraSimulator.matrix_transpose(request.matrix)
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "success": True,
            "result": result.result,
            "steps": result.steps,
            "properties": result.properties
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Matrix transpose error: {e}")
        raise HTTPException(status_code=500, detail=f"Matris transpose hatası: {str(e)}")


@router.post("/matrix/inverse")
async def inverse_matrix(request: MatrixOperationRequest) -> Dict[str, Any]:
    """
    Matris tersini hesaplar: M^(-1).
    Kare matrisler için geçerlidir ve determinant ≠ 0 olmalıdır.
    """
    try:
        result = LinearAlgebraSimulator.matrix_inverse(request.matrix)
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "success": True,
            "result": result.result,
            "steps": result.steps,
            "properties": result.properties
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Matrix inverse error: {e}")
        raise HTTPException(status_code=500, detail=f"Matris ters hesaplama hatası: {str(e)}")


@router.post("/matrix/determinant")
async def calculate_determinant(request: MatrixOperationRequest) -> Dict[str, Any]:
    """
    Matris determinantını hesaplar.
    det(M) hesaplaması yapar.
    """
    try:
        result = LinearAlgebraSimulator.matrix_determinant(request.matrix)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Determinant calculation error: {e}")
        raise HTTPException(status_code=500, detail=f"Determinant hesaplama hatası: {str(e)}")


@router.post("/matrix/eigenvalues")
async def calculate_eigenvalues(request: MatrixOperationRequest) -> Dict[str, Any]:
    """
    Matrisin öz değerlerini (eigenvalues) ve öz vektörlerini (eigenvectors) hesaplar.
    """
    try:
        result = LinearAlgebraSimulator.eigenvalues_eigenvectors(request.matrix)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Eigenvalue calculation error: {e}")
        raise HTTPException(status_code=500, detail=f"Öz değer hesaplama hatası: {str(e)}")


# ============================================================================
# Vector Operations Endpoints
# ============================================================================

@router.post("/vector/dot")
async def vector_dot_product(request: VectorDotProductRequest) -> Dict[str, Any]:
    """
    İki vektörün nokta çarpımını (dot product) hesaplar.
    a · b = Σ(a[i] × b[i]) hesaplaması yapar.
    Geometrik yorumu da sağlar: a · b = |a| × |b| × cos(θ)
    """
    try:
        result = LinearAlgebraSimulator.vector_dot_product(
            request.vector_a,
            request.vector_b
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "success": True,
            "result": result.result[0],  # Scalar value
            "steps": result.steps,
            "properties": result.properties
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector dot product error: {e}")
        raise HTTPException(status_code=500, detail=f"Vektör nokta çarpımı hatası: {str(e)}")


# ============================================================================
# Calculus Endpoints
# ============================================================================

@router.get("/calculus/functions")
async def list_supported_functions() -> List[Dict[str, str]]:
    """Desteklenen matematiksel fonksiyonları listeler."""
    return CalculusSimulator.list_supported_functions()


@router.post("/calculus/derivative")
async def compute_derivative(request: DerivativeRequest) -> Dict[str, Any]:
    """
    Fonksiyonun türevini hesaplar ve grafik için veri döner.
    f(x) ve f'(x) değerlerini hesaplar.
    """
    try:
        result = CalculusSimulator.compute_derivative(
            function_name=request.function_name,
            x_min=request.x_min,
            x_max=request.x_max,
            num_points=request.num_points
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "success": True,
            "x_values": result.x_values,
            "y_values": result.y_values,
            "dy_dx_values": result.dy_dx_values,
            "formula": result.formula,
            "derivative_formula": result.derivative_formula
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Derivative computation error: {e}")
        raise HTTPException(status_code=500, detail=f"Türev hesaplama hatası: {str(e)}")


@router.post("/calculus/gradient-descent")
async def simulate_gradient_descent(request: GradientDescentRequest) -> Dict[str, Any]:
    """
    Gradient descent algoritmasını simüle eder.
    Fonksiyon minimumunu bulmak için iteratif optimizasyon yapar.
    x_(t+1) = x_t - α × ∇f(x_t) adımlarını gösterir.
    """
    try:
        result = CalculusSimulator.gradient_descent(
            function_name=request.function_name,
            initial_x=request.initial_x,
            learning_rate=request.learning_rate,
            max_iterations=request.max_iterations,
            tolerance=request.tolerance
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "success": True,
            "history": result.history,
            "final_x": result.final_x,
            "final_y": result.final_y,
            "converged": result.converged,
            "iterations": result.iterations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gradient descent error: {e}")
        raise HTTPException(status_code=500, detail=f"Gradient descent hatası: {str(e)}")


@router.post("/calculus/chain-rule")
async def demonstrate_chain_rule(request: ChainRuleRequest) -> Dict[str, Any]:
    """
    Chain rule'u adım adım gösterir.
    d/dx[f(g(x))] = f'(g(x)) × g'(x) hesaplamasını açıklar.
    """
    try:
        result = CalculusSimulator.chain_rule_breakdown(
            outer_function=request.outer_function,
            inner_function=request.inner_function,
            x_value=request.x_value
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chain rule error: {e}")
        raise HTTPException(status_code=500, detail=f"Chain rule hatası: {str(e)}")


# ============================================================================
# Utility Endpoints
# ============================================================================

@router.get("/presets/matrices")
async def get_matrix_presets() -> List[Dict[str, Any]]:
    """
    Örnek matris presetleri döner.
    Test ve öğrenme için hazır örnekler.
    """
    return [
        {
            "name": "2x2 Basit",
            "matrix_a": [[1, 2], [3, 4]],
            "matrix_b": [[5, 6], [7, 8]],
            "description": "Temel 2x2 matris çarpımı"
        },
        {
            "name": "3x3 Birim Matris",
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "description": "3x3 identity matrix (I)"
        },
        {
            "name": "Rotasyon Matrisi (90°)",
            "matrix": [[0, -1], [1, 0]],
            "description": "2D counter-clockwise 90° rotation"
        },
        {
            "name": "Simetrik Matris",
            "matrix": [[4, 1, 2], [1, 5, 3], [2, 3, 6]],
            "description": "Simetrik positive-definite matrix"
        }
    ]


@router.get("/presets/vectors")
async def get_vector_presets() -> List[Dict[str, Any]]:
    """
    Örnek vektör presetleri döner.
    """
    return [
        {
            "name": "Birim Vektörler (2D)",
            "vector_a": [1.0, 0.0],
            "vector_b": [0.0, 1.0],
            "description": "Ortogonal unit vectors (x, y)"
        },
        {
            "name": "Paralel Vektörler",
            "vector_a": [1.0, 2.0, 3.0],
            "vector_b": [2.0, 4.0, 6.0],
            "description": "Parallel vectors (angle = 0°)"
        },
        {
            "name": "Dik Vektörler (3D)",
            "vector_a": [1.0, 0.0, 0.0],
            "vector_b": [0.0, 1.0, 0.0],
            "description": "Orthogonal vectors (angle = 90°)"
        }
    ]
