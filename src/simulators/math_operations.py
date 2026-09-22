"""
src/simulators/math_operations.py

Math Lab Backend Simulators:
- Linear Algebra operations (matrix, vector operations)
- Calculus operations (derivatives, gradient descent, chain rule)
- Probability distributions (optional)

Author: Kenan AY
Location: Kütahya, TÜRKİYE
Version: 1.0.0
"""

from typing import List, Dict, Any, Tuple, Optional, Callable, TypedDict, Union, Sequence
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Linear Algebra Simulator
# ============================================================================

@dataclass
class MatrixOperationResult:
    """Matrix işlemi sonucu."""
    success: bool
    result: Optional[List[List[float]]]
    steps: List[Dict[str, Any]]
    properties: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class VectorOperationResult:
    """Vektör işlemi sonucu."""
    success: bool
    result: Optional[List[float]]
    steps: List[Dict[str, Any]]
    properties: Dict[str, Any]
    error: Optional[str] = None


class LinearAlgebraSimulator:
    """
    Lineer cebir işlemlerini simüle eder ve adım adım açıklar.
    
    Features:
    - Matrix multiplication with step-by-step breakdown
    - Matrix operations: transpose, inverse, determinant
    - Vector operations: dot product, cross product, magnitude
    - Eigenvalue/eigenvector computation
    """
    
    @staticmethod
    def matrix_multiply(
        matrix_a: List[List[float]],
        matrix_b: List[List[float]],
        detailed_cell: Optional[Union[Tuple[int, int], List[int], Sequence[int]]] = None
    ) -> MatrixOperationResult:
        """
        İki matrisi çarpar ve adım adım açıklama sağlar.
        
        Args:
            matrix_a: A matrisi (M x K), shape [M, K]
            matrix_b: B matrisi (K x N), shape [K, N]
            detailed_cell: Detaylı gösterilecek hücre (row, col)
            
        Returns:
            MatrixOperationResult with result matrix (M x N) and steps
        """
        try:
            A = np.array(matrix_a, dtype=float)
            B = np.array(matrix_b, dtype=float)
            
            # Shape validation
            if A.ndim != 2 or B.ndim != 2:
                return MatrixOperationResult(
                    success=False,
                    result=None,
                    steps=[],
                    properties={},
                    error="Matrisler 2D olmalı"
                )
            
            M, K_a = A.shape
            K_b, N = B.shape
            
            if K_a != K_b:
                return MatrixOperationResult(
                    success=False,
                    result=None,
                    steps=[],
                    properties={},
                    error=f"Boyut uyumsuzluğu: A ({M}x{K_a}) ve B ({K_b}x{N})"
                )
            
            K = K_a
            
            # Compute result
            C = A @ B  # shape: [M, N]
            
            # Generate step-by-step explanation
            steps = []
            
            # Step 1: Dimension check
            steps.append({
                "step": 1,
                "title": "Boyut Kontrolü",
                "description": f"A: ({M}×{K}), B: ({K}×{N}) → C: ({M}×{N})",
                "formula": "C[i,j] = Σ(k=0 to K-1) A[i,k] × B[k,j]",
                "valid": True
            })
            
            # Step 2: Detailed calculation for one cell
            if detailed_cell is not None:
                i, j = detailed_cell
                if 0 <= i < M and 0 <= j < N:
                    cell_steps = []
                    cell_sum = 0.0
                    
                    for k in range(K):
                        a_val = A[i, k]
                        b_val = B[k, j]
                        product = a_val * b_val
                        cell_sum += product
                        
                        cell_steps.append({
                            "k": k,
                            "a_value": float(a_val),
                            "b_value": float(b_val),
                            "product": float(product),
                            "running_sum": float(cell_sum)
                        })
                    
                    steps.append({
                        "step": 2,
                        "title": f"Hücre [{i},{j}] Hesaplama",
                        "description": f"C[{i},{j}] nokta çarpımı",
                        "cell_calculation": cell_steps,
                        "final_value": float(C[i, j])
                    })
            
            # Step 3: Result summary
            steps.append({
                "step": 3,
                "title": "Sonuç",
                "description": f"Matris çarpımı tamamlandı: ({M}×{K}) × ({K}×{N}) = ({M}×{N})",
                "flops": M * N * K * 2,  # multiply-add pairs
                "memory_reads": (M * K + K * N) * 4,  # float32 bytes
                "memory_writes": M * N * 4
            })
            
            # Properties
            properties = {
                "shape_a": [M, K],
                "shape_b": [K, N],
                "shape_result": [M, N],
                "result_shape": [M, N],
                "total_flops": M * N * K * 2,
                "result_min": float(np.min(C)),
                "result_max": float(np.max(C)),
                "result_mean": float(np.mean(C)),
                "frobenius_norm": float(np.linalg.norm(C, 'fro'))
            }

            return MatrixOperationResult(
                success=True,
                result=C.tolist(),
                steps=steps,
                properties=properties
            )
            
        except Exception as e:
            logger.error(f"Matrix multiply error: {e}")
            return MatrixOperationResult(
                success=False,
                result=None,
                steps=[],
                properties={},
                error=str(e)
            )
    
    @staticmethod
    def matrix_transpose(matrix: List[List[float]]) -> MatrixOperationResult:
        """
        Matris transpozisyonu (satırlar ↔ sütunlar).
        
        Args:
            matrix: Girdi matrisi, shape [M, N]
            
        Returns:
            MatrixOperationResult with transposed matrix, shape [N, M]
        """
        try:
            M = np.array(matrix, dtype=float)
            
            if M.ndim != 2:
                return MatrixOperationResult(
                    success=False,
                    result=None,
                    steps=[],
                    properties={},
                    error="Matris 2D olmalı"
                )
            
            M_T = M.T
            rows, cols = M.shape
            
            steps = [{
                "step": 1,
                "title": "Transpose İşlemi",
                "description": f"Satırlar ve sütunlar yer değiştirir: ({rows}×{cols}) → ({cols}×{rows})",
                "formula": "M^T[i,j] = M[j,i]"
            }]
            
            properties = {
                "original_shape": [rows, cols],
                "transposed_shape": [cols, rows],
                "is_square": rows == cols,
                "is_symmetric": np.allclose(M, M_T) if rows == cols else False
            }
            
            return MatrixOperationResult(
                success=True,
                result=M_T.tolist(),
                steps=steps,
                properties=properties
            )
            
        except Exception as e:
            return MatrixOperationResult(
                success=False,
                result=None,
                steps=[],
                properties={},
                error=str(e)
            )
    
    @staticmethod
    def matrix_inverse(matrix: List[List[float]]) -> MatrixOperationResult:
        """
        Matris tersi (inverse) hesaplar: A^(-1).
        
        Args:
            matrix: Kare matris, shape [N, N]
            
        Returns:
            MatrixOperationResult with inverse matrix
        """
        try:
            M = np.array(matrix, dtype=float)
            
            if M.ndim != 2:
                return MatrixOperationResult(
                    success=False,
                    result=None,
                    steps=[],
                    properties={},
                    error="Matris 2D olmalı"
                )
            
            rows, cols = M.shape
            
            if rows != cols:
                return MatrixOperationResult(
                    success=False,
                    result=None,
                    steps=[],
                    properties={},
                    error=f"Matris kare olmalı: ({rows}×{cols})"
                )
            
            # Calculate determinant first
            det = np.linalg.det(M)
            
            if abs(det) < 1e-10:
                return MatrixOperationResult(
                    success=False,
                    result=None,
                    steps=[],
                    properties={"determinant": float(det)},
                    error=f"Matris tekil (singular): determinant = {det:.2e}"
                )
            
            # Compute inverse
            M_inv = np.linalg.inv(M)
            
            # Verify: M × M^(-1) = I
            identity = M @ M_inv
            identity_check = np.allclose(identity, np.eye(rows))
            
            steps = [
                {
                    "step": 1,
                    "title": "Determinant Kontrolü",
                    "description": f"det(M) = {det:.6f} ≠ 0, matris tersine çevrilebilir",
                    "determinant": float(det)
                },
                {
                    "step": 2,
                    "title": "Ters Hesaplama",
                    "description": "Gauss-Jordan eliminasyonu ile M^(-1) hesaplandı",
                    "formula": "M × M^(-1) = I"
                },
                {
                    "step": 3,
                    "title": "Doğrulama",
                    "description": "M × M^(-1) = I kontrol edildi",
                    "verification_passed": identity_check
                }
            ]
            
            properties = {
                "shape": [rows, cols],
                "determinant": float(det),
                "condition_number": float(np.linalg.cond(M)),
                "is_well_conditioned": float(np.linalg.cond(M)) < 100,
                "verification_passed": identity_check
            }
            
            return MatrixOperationResult(
                success=True,
                result=M_inv.tolist(),
                steps=steps,
                properties=properties
            )
            
        except np.linalg.LinAlgError as e:
            return MatrixOperationResult(
                success=False,
                result=None,
                steps=[],
                properties={},
                error=f"Lineer cebir hatası: {str(e)}"
            )
        except Exception as e:
            return MatrixOperationResult(
                success=False,
                result=None,
                steps=[],
                properties={},
                error=str(e)
            )
    
    @staticmethod
    def matrix_determinant(matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Matris determinantını hesaplar.
        
        Args:
            matrix: Kare matris, shape [N, N]
            
        Returns:
            Dict with determinant value and properties
        """
        try:
            M = np.array(matrix, dtype=float)
            
            if M.ndim != 2:
                return {
                    "success": False,
                    "error": "Matris 2D olmalı"
                }
            
            rows, cols = M.shape
            
            if rows != cols:
                return {
                    "success": False,
                    "error": f"Matris kare olmalı: ({rows}×{cols})"
                }
            
            det = np.linalg.det(M)
            
            return {
                "success": True,
                "determinant": float(det),
                "shape": [rows, cols],
                "is_singular": abs(det) < 1e-10,
                "is_invertible": abs(det) >= 1e-10,
                "interpretation": (
                    "Matris tekil (singular)" if abs(det) < 1e-10
                    else "Matris tersine çevrilebilir"
                )
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def vector_dot_product(
        vec_a: List[float],
        vec_b: List[float]
    ) -> VectorOperationResult:
        """
        İki vektörün nokta çarpımı (dot product).
        
        Args:
            vec_a: Vektör A, shape [N]
            vec_b: Vektör B, shape [N]
            
        Returns:
            VectorOperationResult with scalar result
        """
        try:
            a = np.array(vec_a, dtype=float)
            b = np.array(vec_b, dtype=float)
            
            if a.shape != b.shape:
                return VectorOperationResult(
                    success=False,
                    result=None,
                    steps=[],
                    properties={},
                    error=f"Vektör boyutları eşleşmeli: {a.shape} vs {b.shape}"
                )
            
            dot_product = np.dot(a, b)
            
            # Step-by-step calculation
            steps = []
            partial_sum = 0.0
            element_products = []
            
            for i, (a_i, b_i) in enumerate(zip(a, b)):
                product = a_i * b_i
                partial_sum += product
                element_products.append({
                    "index": i,
                    "a_value": float(a_i),
                    "b_value": float(b_i),
                    "product": float(product),
                    "running_sum": float(partial_sum)
                })
            
            steps.append({
                "step": 1,
                "title": "Nokta Çarpımı",
                "description": f"a · b = Σ(a[i] × b[i]) for i=0 to {len(a)-1}",
                "element_products": element_products,
                "final_result": float(dot_product)
            })
            
            # Geometric interpretation
            magnitude_a = float(np.linalg.norm(a))
            magnitude_b = float(np.linalg.norm(b))
            
            cos_angle = 0.0
            angle_radians = 0.0
            angle_degrees = 0.0
            has_angle = magnitude_a > 1e-10 and magnitude_b > 1e-10
            
            if has_angle:
                cos_val = np.clip(dot_product / (magnitude_a * magnitude_b), -1.0, 1.0)
                cos_angle = float(cos_val)
                angle_radians = float(np.arccos(cos_angle))
                angle_degrees = float(np.degrees(angle_radians))
                
                steps.append({
                    "step": 2,
                    "title": "Geometrik Yorum",
                    "description": "a · b = |a| × |b| × cos(θ)",
                    "magnitude_a": magnitude_a,
                    "magnitude_b": magnitude_b,
                    "cos_angle": cos_angle,
                    "angle_radians": angle_radians,
                    "angle_degrees": angle_degrees
                })
            
            properties = {
                "dimension": len(a),
                "magnitude_a": magnitude_a,
                "magnitude_b": magnitude_b,
                "is_orthogonal": bool(abs(dot_product) < 1e-10),
                "cosine_similarity": cos_angle if has_angle else 0.0,
                "angle_degrees": angle_degrees if has_angle else 0.0,
                "angle_radians": angle_radians if has_angle else 0.0
            }
            
            return VectorOperationResult(
                success=True,
                result=[float(dot_product)],  # Scalar as list
                steps=steps,
                properties=properties
            )
            
        except Exception as e:
            return VectorOperationResult(
                success=False,
                result=None,
                steps=[],
                properties={},
                error=str(e)
            )
    
    @staticmethod
    def eigenvalues_eigenvectors(matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Matrisin öz değerlerini (eigenvalues) ve öz vektörlerini (eigenvectors) hesaplar.
        
        Args:
            matrix: Kare matris, shape [N, N]
            
        Returns:
            Dict with eigenvalues and eigenvectors
        """
        try:
            M = np.array(matrix, dtype=float)
            
            if M.ndim != 2:
                return {
                    "success": False,
                    "error": "Matris 2D olmalı"
                }
            
            rows, cols = M.shape
            
            if rows != cols:
                return {
                    "success": False,
                    "error": f"Matris kare olmalı: ({rows}×{cols})"
                }
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(M)
            
            # Sort by magnitude (descending)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            eigen_pairs = []
            for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
                eigen_pairs.append({
                    "index": i,
                    "eigenvalue": float(val.real) if np.isreal(val) else {"real": float(val.real), "imag": float(val.imag)},
                    "eigenvector": vec.real.tolist() if np.all(np.isreal(vec)) else [{"real": float(v.real), "imag": float(v.imag)} for v in vec],
                    "magnitude": float(np.abs(val))
                })
            
            return {
                "success": True,
                "shape": [rows, cols],
                "eigenvalues": eigenvalues.tolist(),
                "eigenvectors": eigenvectors.tolist(),
                "eigen_pairs": eigen_pairs,
                "trace": float(np.trace(M)),
                "determinant": float(np.linalg.det(M)),
                "is_real": bool(np.all(np.isreal(eigenvalues)))
            }
            
        except np.linalg.LinAlgError as e:
            return {
                "success": False,
                "error": f"Lineer cebir hatası: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# ============================================================================
# Calculus Simulator
# ============================================================================

@dataclass
class DerivativeResult:
    """Türev hesaplama sonucu."""
    success: bool
    x_values: List[float]
    y_values: List[float]
    dy_dx_values: List[float]
    formula: str
    derivative_formula: str
    error: Optional[str] = None


@dataclass
class GradientDescentResult:
    """Gradient descent simülasyonu sonucu."""
    success: bool
    history: List[Dict[str, Any]]
    final_x: float
    final_y: float
    converged: bool
    iterations: int
    error: Optional[str] = None


class FunctionDefinition(TypedDict):
    formula: str
    fn: Callable[..., Any]
    derivative: str
    derivative_fn: Callable[..., Any]


class OuterFunctionDefinition(TypedDict):
    f: Callable[..., Any]
    f_prime: Callable[..., Any]
    formula: str
    derivative: str


class InnerFunctionDefinition(TypedDict):
    g: Callable[..., Any]
    g_prime: Callable[..., Any]
    formula: str
    derivative: str


class CalculusSimulator:
    """
    Calculus (kalkülüs) işlemlerini simüle eder.
    
    Features:
    - Numerical derivatives
    - Function plotting with derivatives
    - Gradient descent visualization
    - Chain rule breakdown
    """
    
    # Supported functions
    FUNCTIONS: Dict[str, FunctionDefinition] = {
        "quadratic": {
            "formula": "f(x) = x²",
            "fn": lambda x: x**2,
            "derivative": "f'(x) = 2x",
            "derivative_fn": lambda x: 2*x
        },
        "cubic": {
            "formula": "f(x) = x³ - 3x",
            "fn": lambda x: x**3 - 3*x,
            "derivative": "f'(x) = 3x² - 3",
            "derivative_fn": lambda x: 3*x**2 - 3
        },
        "sin": {
            "formula": "f(x) = sin(x)",
            "fn": np.sin,
            "derivative": "f'(x) = cos(x)",
            "derivative_fn": np.cos
        },
        "exp": {
            "formula": "f(x) = e^x",
            "fn": np.exp,
            "derivative": "f'(x) = e^x",
            "derivative_fn": np.exp
        },
        "sigmoid": {
            "formula": "f(x) = 1/(1+e^(-x))",
            "fn": lambda x: 1 / (1 + np.exp(-x)),
            "derivative": "f'(x) = f(x)(1-f(x))",
            "derivative_fn": lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x)))
        }
    }
    
    @staticmethod
    def compute_derivative(
        function_name: str,
        x_min: float = -5.0,
        x_max: float = 5.0,
        num_points: int = 100
    ) -> DerivativeResult:
        """
        Fonksiyonun türevini hesaplar ve grafik için veri döner.
        
        Args:
            function_name: Fonksiyon adı (quadratic, cubic, sin, exp, sigmoid)
            x_min: X minimum değer
            x_max: X maximum değer
            num_points: Nokta sayısı
            
        Returns:
            DerivativeResult with function and derivative values
        """
        try:
            if function_name not in CalculusSimulator.FUNCTIONS:
                return DerivativeResult(
                    success=False,
                    x_values=[],
                    y_values=[],
                    dy_dx_values=[],
                    formula="",
                    derivative_formula="",
                    error=f"Bilinmeyen fonksiyon: {function_name}"
                )
            
            func_info = CalculusSimulator.FUNCTIONS[function_name]
            fn = func_info["fn"]
            derivative_fn = func_info["derivative_fn"]
            
            # Generate x values
            x = np.linspace(x_min, x_max, num_points)
            
            # Compute function and derivative values
            y = fn(x)
            dy_dx = derivative_fn(x)
            
            return DerivativeResult(
                success=True,
                x_values=x.tolist(),
                y_values=y.tolist(),
                dy_dx_values=dy_dx.tolist(),
                formula=func_info["formula"],
                derivative_formula=func_info["derivative"]
            )
            
        except Exception as e:
            logger.error(f"Derivative computation error: {e}")
            return DerivativeResult(
                success=False,
                x_values=[],
                y_values=[],
                dy_dx_values=[],
                formula="",
                derivative_formula="",
                error=str(e)
            )
    
    @staticmethod
    def gradient_descent(
        function_name: str,
        initial_x: float = 2.0,
        learning_rate: float = 0.1,
        max_iterations: int = 50,
        tolerance: float = 1e-6
    ) -> GradientDescentResult:
        """
        Gradient descent algoritmasını simüle eder.
        
        Args:
            function_name: Optimize edilecek fonksiyon
            initial_x: Başlangıç noktası
            learning_rate: Öğrenme oranı (α)
            max_iterations: Maksimum iterasyon
            tolerance: Yakınsama toleransı
            
        Returns:
            GradientDescentResult with optimization history
        """
        try:
            if function_name not in CalculusSimulator.FUNCTIONS:
                return GradientDescentResult(
                    success=False,
                    history=[],
                    final_x=0.0,
                    final_y=0.0,
                    converged=False,
                    iterations=0,
                    error=f"Bilinmeyen fonksiyon: {function_name}"
                )
            
            func_info = CalculusSimulator.FUNCTIONS[function_name]
            fn = func_info["fn"]
            derivative_fn = func_info["derivative_fn"]
            
            history = []
            x = initial_x
            
            for iteration in range(max_iterations):
                y = float(fn(x))
                gradient = float(derivative_fn(x))
                
                # Record step
                history.append({
                    "iteration": iteration,
                    "x": float(x),
                    "y": float(y),
                    "gradient": float(gradient),
                    "learning_rate": learning_rate,
                    "update": -learning_rate * gradient
                })
                
                # Check convergence
                if abs(gradient) < tolerance:
                    return GradientDescentResult(
                        success=True,
                        history=history,
                        final_x=float(x),
                        final_y=float(y),
                        converged=True,
                        iterations=iteration + 1
                    )
                
                # Update x
                x = x - learning_rate * gradient
            
            # Max iterations reached
            final_y = float(fn(x))
            
            return GradientDescentResult(
                success=True,
                history=history,
                final_x=float(x),
                final_y=final_y,
                converged=False,
                iterations=max_iterations
            )
            
        except Exception as e:
            logger.error(f"Gradient descent error: {e}")
            return GradientDescentResult(
                success=False,
                history=[],
                final_x=0.0,
                final_y=0.0,
                converged=False,
                iterations=0,
                error=str(e)
            )
    
    @staticmethod
    def chain_rule_breakdown(
        outer_function: str,
        inner_function: str,
        x_value: float
    ) -> Dict[str, Any]:
        """
        Chain rule'u adım adım açıklar: d/dx[f(g(x))] = f'(g(x)) × g'(x).
        
        Args:
            outer_function: Dış fonksiyon (örn: "square", "sin", "exp")
            inner_function: İç fonksiyon (örn: "linear", "quadratic")
            x_value: Hesaplama noktası
            
        Returns:
            Dict with chain rule breakdown
        """
        try:
            # Define outer functions: f(u)
            outer_fns: Dict[str, OuterFunctionDefinition] = {
                "square": {
                    "f": lambda u: u**2,
                    "f_prime": lambda u: 2*u,
                    "formula": "f(u) = u²",
                    "derivative": "f'(u) = 2u"
                },
                "sin": {
                    "f": np.sin,
                    "f_prime": np.cos,
                    "formula": "f(u) = sin(u)",
                    "derivative": "f'(u) = cos(u)"
                },
                "exp": {
                    "f": np.exp,
                    "f_prime": np.exp,
                    "formula": "f(u) = e^u",
                    "derivative": "f'(u) = e^u"
                }
            }
            
            # Define inner functions: g(x)
            inner_fns: Dict[str, InnerFunctionDefinition] = {
                "linear": {
                    "g": lambda x: 2*x + 1,
                    "g_prime": lambda x: 2,
                    "formula": "g(x) = 2x + 1",
                    "derivative": "g'(x) = 2"
                },
                "quadratic": {
                    "g": lambda x: x**2,
                    "g_prime": lambda x: 2*x,
                    "formula": "g(x) = x²",
                    "derivative": "g'(x) = 2x"
                },
                "cubic": {
                    "g": lambda x: x**3,
                    "g_prime": lambda x: 3*x**2,
                    "formula": "g(x) = x³",
                    "derivative": "g'(x) = 3x²"
                }
            }
            
            if outer_function not in outer_fns:
                return {
                    "success": False,
                    "error": f"Bilinmeyen dış fonksiyon: {outer_function}"
                }
            
            if inner_function not in inner_fns:
                return {
                    "success": False,
                    "error": f"Bilinmeyen iç fonksiyon: {inner_function}"
                }
            
            f_info = outer_fns[outer_function]
            g_info = inner_fns[inner_function]
            
            # Compute values
            g_x = float(g_info["g"](x_value))
            f_g_x = float(f_info["f"](g_x))
            g_prime_x = float(g_info["g_prime"](x_value))
            f_prime_g_x = float(f_info["f_prime"](g_x))
            
            # Chain rule result
            chain_rule_result = f_prime_g_x * g_prime_x
            
            g_form = g_info["formula"]
            comp_formula = f_info["formula"].replace("u", f"({g_form})")

            return {
                "success": True,
                "x_value": float(x_value),
                "composite_formula": comp_formula,
                "steps": [
                    {
                        "step": 1,
                        "title": "İç Fonksiyon",
                        "formula": g_info["formula"],
                        "derivative": g_info["derivative"],
                        "value_at_x": g_x,
                        "derivative_at_x": g_prime_x
                    },
                    {
                        "step": 2,
                        "title": "Dış Fonksiyon",
                        "formula": f_info["formula"],
                        "derivative": f_info["derivative"],
                        "value_at_g_x": f_g_x,
                        "derivative_at_g_x": f_prime_g_x
                    },
                    {
                        "step": 3,
                        "title": "Chain Rule",
                        "formula": "d/dx[f(g(x))] = f'(g(x)) × g'(x)",
                        "calculation": f"{f_prime_g_x:.4f} × {g_prime_x:.4f} = {chain_rule_result:.4f}",
                        "result": chain_rule_result,
                        "total_derivative": chain_rule_result
                    }
                ],
                "final_derivative": chain_rule_result
            }
            
        except Exception as e:
            logger.error(f"Chain rule error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def list_supported_functions() -> List[Dict[str, str]]:
        """Desteklenen fonksiyonları listeler."""
        return [
            {
                "name": name,
                "formula": info["formula"],
                "derivative": info["derivative"]
            }
            for name, info in CalculusSimulator.FUNCTIONS.items()
        ]
