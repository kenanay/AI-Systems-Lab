"""
Tensor & Math Lab Simulation Engine

Provides core mathematical algorithms, tensor shape analysis, broadcasting logic,
matrix multiplication (GEMM) cell-level breakdown, activation functions with temperature
scaling, and 2-layer MLP autograd backpropagation graph simulations.

Author: Kenan AY
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import torch


# Supported data types and their byte sizes
DTYPE_BYTES: Dict[str, int] = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
    "float64": 8,
}

DTYPE_DESCRIPTIONS: Dict[str, str] = {
    "float32": "Standard single-precision floating point (32-bit). Baseline for training & inference.",
    "float16": "Half-precision floating point (16-bit). 50% memory reduction, common on modern GPUs.",
    "bfloat16": "Brain floating point (16-bit). Same 8-bit dynamic range as FP32, ideal for LLMs.",
    "int8": "8-bit signed integer. Used in quantized inference (e.g., bitsandbytes, AWQ, GPTQ).",
    "uint8": "8-bit unsigned integer (0-255). Often used for raw image pixels or tokens.",
    "int32": "32-bit integer. Standard indexing and token ID representation.",
    "int64": "64-bit integer (Long). PyTorch default for embedding token indices.",
    "float64": "Double-precision floating point (64-bit). High precision scientific computing.",
}


def format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable string (B, KB, MB, GB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{num_bytes / (1024 ** 3):.2f} GB"


class TensorShapeAnalyzer:
    """Analyzes tensor shapes, strides, memory footprints, and transformation operations."""

    @staticmethod
    def compute_strides(shape: List[int], itemsize: int = 1) -> List[int]:
        """Compute C-contiguous (row-major) strides."""
        if not shape:
            return []
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return [s * itemsize for s in strides]

    @classmethod
    def analyze_shape(cls, shape: List[int], dtype: str = "float32") -> Dict[str, Any]:
        """
        Calculates total elements, memory across multiple dtypes, C-contiguous strides,
        and provides common NLP/Vision architectural semantic interpretations.
        """
        normalized_dtype = dtype.lower()
        if normalized_dtype not in DTYPE_BYTES:
            normalized_dtype = "float32"

        # Validate dimensions
        if any(d <= 0 for d in shape):
            raise ValueError("All dimensions must be positive integers.")

        total_elements = int(np.prod(shape)) if shape else 0
        element_bytes = DTYPE_BYTES[normalized_dtype]
        total_bytes = total_elements * element_bytes

        element_strides = cls.compute_strides(shape, itemsize=1)
        byte_strides = cls.compute_strides(shape, itemsize=element_bytes)

        # Memory footprints across different dtypes
        memory_comparison = {}
        for dt, b_size in [
            ("float32", 4),
            ("float16", 2),
            ("bfloat16", 2),
            ("int8", 1),
            ("int64", 8),
        ]:
            b_total = total_elements * b_size
            memory_comparison[dt] = {
                "bytes": b_total,
                "formatted": format_bytes(b_total),
                "ratio_vs_fp32": round(b_size / 4.0, 2),
            }

        # Semantic interpretation based on rank
        rank = len(shape)
        semantic_hint = "Arbitrary Tensor"
        if rank == 1:
            semantic_hint = "1D Vector (e.g. Bias, Logits, Token Embeddings of 1 token)"
        elif rank == 2:
            semantic_hint = f"2D Matrix (e.g. Batch x Dim [{shape[0]}, {shape[1]}] or Weight Matrix)"
        elif rank == 3:
            semantic_hint = f"3D Tensor (e.g. [Batch={shape[0]}, SeqLen={shape[1]}, HiddenDim={shape[2]}])"
        elif rank == 4:
            semantic_hint = (
                f"4D Attention Tensor [Batch={shape[0]}, Heads={shape[1]}, SeqLen={shape[2]}, HeadDim={shape[3]}] "
                f"or Vision [Batch={shape[0]}, Channels={shape[1]}, Height={shape[2]}, Width={shape[3]}]"
            )

        return {
            "shape": shape,
            "rank": rank,
            "total_elements": total_elements,
            "dtype": normalized_dtype,
            "element_bytes": element_bytes,
            "total_bytes": total_bytes,
            "formatted_memory": format_bytes(total_bytes),
            "element_strides": element_strides,
            "byte_strides": byte_strides,
            "is_contiguous": True,
            "semantic_interpretation": semantic_hint,
            "memory_comparison": memory_comparison,
            "dtype_description": DTYPE_DESCRIPTIONS.get(normalized_dtype, ""),
        }

    @classmethod
    def simulate_reshape(cls, original_shape: List[int], target_shape: List[int]) -> Dict[str, Any]:
        """Simulate tensor reshape, resolving any single -1 dimension."""
        orig_elements = int(np.prod(original_shape))
        minus_one_count = target_shape.count(-1)

        if minus_one_count > 1:
            return {
                "success": False,
                "error": "Only one dimension can be inferred (-1).",
            }

        resolved_shape = list(target_shape)
        if minus_one_count == 1:
            known_prod = 1
            idx_neg = -1
            for i, dim in enumerate(target_shape):
                if dim == -1:
                    idx_neg = i
                elif dim <= 0:
                    return {"success": False, "error": f"Invalid dimension value: {dim}"}
                else:
                    known_prod *= dim
            if orig_elements % known_prod != 0:
                return {
                    "success": False,
                    "error": f"Cannot reshape {original_shape} ({orig_elements} elements) into {target_shape}.",
                }
            resolved_shape[idx_neg] = orig_elements // known_prod
        else:
            new_elements = int(np.prod(resolved_shape))
            if new_elements != orig_elements:
                return {
                    "success": False,
                    "error": (
                        f"Shape mismatch: original has {orig_elements} elements, "
                        f"target requires {new_elements}."
                    ),
                }

        new_strides = cls.compute_strides(resolved_shape, itemsize=1)
        return {
            "success": True,
            "original_shape": original_shape,
            "target_shape": target_shape,
            "resolved_shape": resolved_shape,
            "total_elements": orig_elements,
            "new_strides": new_strides,
            "contiguous": True,
            "requires_copy": False,
            "explanation": f"Successfully reshaped {original_shape} to {resolved_shape} (preserves {orig_elements} elements).",
        }

    @classmethod
    def simulate_transpose(cls, shape: List[int], permutation: List[int]) -> Dict[str, Any]:
        """Simulate tensor transposition / axis permutation and stride calculation."""
        rank = len(shape)
        if sorted(permutation) != list(range(rank)):
            return {
                "success": False,
                "error": f"Permutation must contain all indices from 0 to {rank - 1}.",
            }

        orig_strides = cls.compute_strides(shape, itemsize=1)
        transposed_shape = [shape[p] for p in permutation]
        transposed_strides = [orig_strides[p] for p in permutation]

        # Check if resulting memory layout is still contiguous
        contiguous_strides = cls.compute_strides(transposed_shape, itemsize=1)
        is_contiguous = transposed_strides == contiguous_strides

        return {
            "success": True,
            "original_shape": shape,
            "permutation": permutation,
            "transposed_shape": transposed_shape,
            "transposed_strides": transposed_strides,
            "is_contiguous": is_contiguous,
            "requires_contiguous_call": not is_contiguous,
            "note": (
                "Tensor is contiguous."
                if is_contiguous
                else "Tensor is non-contiguous in memory! Calling .contiguous() will allocate new memory buffer."
            ),
        }


class BroadcastingEngine:
    """Simulates NumPy & PyTorch broadcasting rules with step-by-step expansion."""

    @staticmethod
    def analyze_broadcast(shape_a: List[int], shape_b: List[int]) -> Dict[str, Any]:
        """
        Evaluates broadcasting rules right-to-left.
        Returns compatibility, target shape, and dimensional alignment steps.
        """
        len_a = len(shape_a)
        len_b = len(shape_b)
        max_len = max(len_a, len_b)

        # Right-aligned padding with 1s
        pad_a = [1] * (max_len - len_a) + list(shape_a)
        pad_b = [1] * (max_len - len_b) + list(shape_b)

        steps = []
        out_shape = []
        compatible = True
        error_msg = None

        for idx in range(max_len):
            dim_idx = idx - max_len  # Negative index from -max_len to -1
            da = pad_a[idx]
            db = pad_b[idx]

            if da == db:
                res_dim = da
                action = f"Dimensions match ({da}). No expansion needed."
                exp_a = 1
                exp_b = 1
            elif da == 1:
                res_dim = db
                action = f"A has dimension 1; broadcast/replicated {db} times."
                exp_a = db
                exp_b = 1
            elif db == 1:
                res_dim = da
                action = f"B has dimension 1; broadcast/replicated {da} times."
                exp_a = 1
                exp_b = da
            else:
                compatible = False
                res_dim = -1
                action = f"Incompatible dimensions: {da} vs {db} (neither is 1)!"
                error_msg = f"Cannot broadcast dimensions at axis {dim_idx} ({da} != {db})."
                exp_a = 0
                exp_b = 0

            out_shape.append(res_dim)
            steps.append({
                "axis": dim_idx,
                "dim_a": da,
                "dim_b": db,
                "result_dim": res_dim,
                "action": action,
                "expansion_factor_a": exp_a,
                "expansion_factor_b": exp_b,
                "compatible": da == db or da == 1 or db == 1,
            })

        return {
            "compatible": compatible,
            "shape_a": shape_a,
            "shape_b": shape_b,
            "padded_a": pad_a,
            "padded_b": pad_b,
            "result_shape": out_shape if compatible else None,
            "steps": steps,
            "error": error_msg,
        }

    @staticmethod
    def simulate_2d_broadcast(
        matrix_a: List[List[float]],
        matrix_b: List[List[float]],
        operation: str = "add"
    ) -> Dict[str, Any]:
        """
        Executes an actual 2D broadcast operation with NumPy, returning
        the original matrices, the broadcasted views, and the computed result.
        """
        arr_a = np.array(matrix_a, dtype=float)
        arr_b = np.array(matrix_b, dtype=float)

        shape_a = list(arr_a.shape)
        shape_b = list(arr_b.shape)

        analysis = BroadcastingEngine.analyze_broadcast(shape_a, shape_b)
        if not analysis["compatible"]:
            return {
                "success": False,
                "analysis": analysis,
                "error": analysis["error"],
            }

        target_shape = analysis["result_shape"]
        # Use np.broadcast_to
        broad_a = np.broadcast_to(arr_a, target_shape)
        broad_b = np.broadcast_to(arr_b, target_shape)

        if operation == "add":
            res = broad_a + broad_b
        elif operation == "mul":
            res = broad_a * broad_b
        elif operation == "sub":
            res = broad_a - broad_b
        else:
            res = broad_a + broad_b

        return {
            "success": True,
            "analysis": analysis,
            "operation": operation,
            "matrix_a": arr_a.tolist(),
            "matrix_b": arr_b.tolist(),
            "broadcasted_a": broad_a.tolist(),
            "broadcasted_b": broad_b.tolist(),
            "result_matrix": res.tolist(),
        }


class MatMulVisualizer:
    """
    Simulates General Matrix Multiplication (GEMM): C = A @ B.
    Provides detailed cell-by-cell dot product math and hardware arithmetic intensity metrics.
    """

    @staticmethod
    def multiply(
        matrix_a: List[List[float]],
        matrix_b: List[List[float]],
        selected_row: Optional[int] = None,
        selected_col: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Multiplies A (M x K) by B (K x N), computes C (M x N),
        and returns detailed step-by-step dot product breakdown for a selected cell (i, j).
        """
        arr_a = np.array(matrix_a, dtype=float)
        arr_b = np.array(matrix_b, dtype=float)

        if arr_a.ndim != 2 or arr_b.ndim != 2:
            return {"success": False, "error": "Both inputs must be 2D matrices."}

        m, k_a = arr_a.shape
        k_b, n = arr_b.shape

        if k_a != k_b:
            return {
                "success": False,
                "error": f"Inner dimension mismatch: A is ({m} x {k_a}), but B is ({k_b} x {n}). Inner dimensions must match!",
            }

        k = k_a
        arr_c = np.matmul(arr_a, arr_b)

        # FLOP analysis
        # Total Multiply-Adds: M * N * K
        # In FLOPs (1 multiply + 1 add = 2 FLOPs): 2 * M * N * K
        flops = 2 * m * n * k
        read_elements = (m * k) + (k * n)
        write_elements = m * n
        total_data_bytes = (read_elements + write_elements) * 4  # FP32
        arithmetic_intensity = round(flops / max(1, total_data_bytes), 2)  # FLOPs per Byte

        # Default selected cell to (0, 0)
        row = selected_row if (selected_row is not None and 0 <= selected_row < m) else 0
        col = selected_col if (selected_col is not None and 0 <= selected_col < n) else 0

        row_a = arr_a[row, :].tolist()
        col_b = arr_b[:, col].tolist()

        terms = []
        formula_parts = []
        for idx in range(k):
            val_a = row_a[idx]
            val_b = col_b[idx]
            prod = round(val_a * val_b, 4)
            terms.append({
                "k_index": idx,
                "a_val": round(val_a, 4),
                "b_val": round(val_b, 4),
                "product": prod,
            })
            formula_parts.append(f"({val_a:.2f} × {val_b:.2f})")

        cell_value = float(arr_c[row, col])
        formula_str = f"C[{row},{col}] = " + " + ".join(formula_parts) + f" = {cell_value:.4f}"

        return {
            "success": True,
            "m": m,
            "k": k,
            "n": n,
            "shape_a": [m, k],
            "shape_b": [k, n],
            "shape_c": [m, n],
            "matrix_a": arr_a.tolist(),
            "matrix_b": arr_b.tolist(),
            "matrix_c": arr_c.tolist(),
            "selected_cell": {"row": row, "col": col},
            "cell_value": round(cell_value, 4),
            "row_vector": row_a,
            "col_vector": col_b,
            "pairwise_terms": terms,
            "formula_string": formula_str,
            "hardware_metrics": {
                "total_flops": flops,
                "read_bytes_fp32": read_elements * 4,
                "write_bytes_fp32": write_elements * 4,
                "total_memory_bytes": total_data_bytes,
                "formatted_memory": format_bytes(total_data_bytes),
                "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
            },
        }

    @staticmethod
    def generate_sample(m: int = 3, k: int = 3, n: int = 3, preset: str = "simple") -> Dict[str, Any]:
        """Generate clean, human-readable integer or float matrices for testing."""
        np.random.seed(42)
        if preset == "simple":
            a = np.random.randint(1, 6, size=(m, k)).astype(float)
            b = np.random.randint(1, 6, size=(k, n)).astype(float)
        elif preset == "sparse":
            a = (np.random.rand(m, k) > 0.5) * np.random.randint(1, 4, size=(m, k))
            b = (np.random.rand(k, n) > 0.5) * np.random.randint(1, 4, size=(k, n))
            a = a.astype(float)
            b = b.astype(float)
        else:
            a = np.round(np.random.randn(m, k), 2)
            b = np.round(np.random.randn(k, n), 2)

        return {
            "matrix_a": a.tolist(),
            "matrix_b": b.tolist(),
        }


class ActivationSimulator:
    """
    Simulates non-linear activations (GELU, ReLU, SiLU/Swish, Sigmoid, Tanh, Softmax)
    with temperature scaling and derivative curves.
    """

    @staticmethod
    def get_supported_functions() -> List[Dict[str, str]]:
        return [
            {"id": "gelu", "name": "GELU (Gaussian Error Linear Unit)", "llm_usage": "BERT, GPT-2, GPT-3, ViT"},
            {"id": "silu", "name": "SiLU / Swish (x * sigmoid(x))", "llm_usage": "LLaMA, Mistral, Gemma (SwiGLU)"},
            {"id": "relu", "name": "ReLU (Rectified Linear Unit)", "llm_usage": "Classic CNNs, ResNets, Early MLPs"},
            {"id": "sigmoid", "name": "Sigmoid (Logistic)", "llm_usage": "Binary gates, GRU/LSTM gates, Attention weights"},
            {"id": "tanh", "name": "Tanh (Hyperbolic Tangent)", "llm_usage": "RNN/LSTM hidden states, Normalization"},
            {"id": "softmax", "name": "Softmax with Temperature", "llm_usage": "Self-Attention probability distribution, Output Logits"},
        ]

    @staticmethod
    def compute_curve(
        activation: str = "gelu",
        x_min: float = -4.0,
        x_max: float = 4.0,
        num_points: int = 81,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Calculates 1D activation curve and analytical gradient over [x_min, x_max]."""
        xs = np.linspace(x_min, x_max, num_points)
        t_xs = torch.tensor(xs, dtype=torch.float32, requires_grad=True)

        func_key = activation.lower()
        if func_key == "relu":
            y = torch.relu(t_xs)
            formula = "f(x) = max(0, x)"
            latex = r"f(x) = \max(0, x)"
            desc = "Zero for negative values, identity for positive. Simple and fast, but suffers from dead neurons."
        elif func_key == "sigmoid":
            y = torch.sigmoid(t_xs)
            formula = "f(x) = 1 / (1 + exp(-x))"
            latex = r"\sigma(x) = \frac{1}{1 + e^{-x}}"
            desc = "S-shaped curve bounding outputs to (0, 1). Susceptible to vanishing gradient for large |x|."
        elif func_key == "tanh":
            y = torch.tanh(t_xs)
            formula = "f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"
            latex = r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}"
            desc = "Zero-centered S-shaped curve between -1 and +1. Stronger gradients than sigmoid near 0."
        elif func_key == "silu":
            y = torch.nn.functional.silu(t_xs)
            formula = "f(x) = x * sigmoid(x)"
            latex = r"\text{SiLU}(x) = x \cdot \sigma(x)"
            desc = "Smooth non-monotonic activation. Core component of SwiGLU FFN in modern LLaMA architectures."
        elif func_key == "gelu":
            y = torch.nn.functional.gelu(t_xs)
            formula = "f(x) = 0.5 * x * (1 + erf(x / sqrt(2)))"
            latex = r"\text{GELU}(x) = x \Phi(x) = x \cdot P(X \le x), X \sim \mathcal{N}(0, 1)"
            desc = "Standard activation in Transformer encoders/decoders. Weights inputs by their magnitude."
        elif func_key == "softmax":
            tau = max(1e-4, temperature)
            y = torch.softmax(t_xs / tau, dim=0)
            formula = f"Softmax(x, tau={tau:.2f})_i = exp(x_i / tau) / sum(exp(x_j / tau))"
            latex = r"P(x_i) = \frac{e^{x_i / \tau}}{\sum_j e^{x_j / \tau}}"
            desc = f"Converts scores into a probability distribution. Temperature tau={tau:.2f} controls sharpness."
        else:
            y = torch.relu(t_xs)
            formula = "f(x) = max(0, x)"
            latex = r"f(x) = \max(0, x)"
            desc = "Defaulting to ReLU."

        # Compute numerical or autograd derivative
        if func_key != "softmax":
            y_sum = y.sum()
            y_sum.backward()
            dys = t_xs.grad.numpy() if t_xs.grad is not None else np.zeros_like(xs)
        else:
            # Softmax derivative: S_i * (1 - S_i) along diagonal
            s_vals = y.detach().numpy()
            dys = s_vals * (1.0 - s_vals)

        points = []
        for x_val, y_val, dy_val in zip(xs, y.detach().numpy(), dys):
            points.append({
                "x": round(float(x_val), 4),
                "y": round(float(y_val), 4),
                "derivative": round(float(dy_val), 4),
            })

        return {
            "activation": func_key,
            "formula": formula,
            "latex": latex,
            "description": desc,
            "temperature": temperature,
            "points": points,
        }

    @staticmethod
    def simulate_temperature_softmax(logits: List[float], temperature: float = 1.0) -> Dict[str, Any]:
        """
        Demonstrates temperature scaling on a discrete logit vector (e.g. LLM next-token generation).
        Computes probabilities, Shannon entropy, and argmax confidence.
        """
        tau = max(1e-4, float(temperature))
        arr_logits = np.array(logits, dtype=float)

        scaled_logits = arr_logits / tau
        # Stabilize by subtracting max
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)

        # Shannon entropy: H(P) = -sum(p * log2(p))
        eps = 1e-12
        entropy = float(-np.sum(probs * np.log2(probs + eps)))
        max_idx = int(np.argmax(probs))

        elements = []
        for i, (l_val, sl_val, p_val) in enumerate(zip(arr_logits, scaled_logits, probs)):
            elements.append({
                "index": i,
                "raw_logit": round(float(l_val), 4),
                "scaled_logit": round(float(sl_val), 4),
                "probability": round(float(p_val), 4),
                "percentage": round(float(p_val * 100.0), 2),
            })

        interpretation = ""
        if tau < 0.5:
            interpretation = "Low temperature (Greedy / Sharp): Concentrates probability into the highest logit."
        elif tau <= 1.2:
            interpretation = "Balanced temperature (Standard): Natural balance between coherence and variety."
        else:
            interpretation = "High temperature (Creative / Flat): Flattens distribution toward uniform randomness."

        return {
            "temperature": tau,
            "logits": arr_logits.tolist(),
            "elements": elements,
            "entropy_bits": round(entropy, 4),
            "max_probability": round(float(probs[max_idx]), 4),
            "argmax_index": max_idx,
            "interpretation": interpretation,
        }


class AutogradGraphSimulator:
    """
    Simulates a 2-Layer MLP Forward and Backward Pass (Autograd Computation Graph).
    Produces visual nodes, tensors, gradients, and formulas.
    """

    @staticmethod
    def simulate(
        x: List[float],
        y_target: List[float],
        hidden_dim: int = 4,
        activation: str = "relu",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Forward:
            z1 = x @ W1 + b1
            a1 = act(z1)
            z2 = a1 @ W2 + b2
            y_pred = z2
            Loss = 0.5 * sum((y_pred - y_target)^2)
        Backward:
            Calculates exact PyTorch analytical gradients for W1, b1, W2, b2, z1, a1, z2.
        """
        torch.manual_seed(seed)
        in_dim = len(x)
        out_dim = len(y_target)

        # PyTorch Tensors
        t_x = torch.tensor([x], dtype=torch.float32, requires_grad=False)
        t_y = torch.tensor([y_target], dtype=torch.float32, requires_grad=False)

        w1 = torch.randn(in_dim, hidden_dim, requires_grad=True)
        b1 = torch.zeros(1, hidden_dim, requires_grad=True)

        w2 = torch.randn(hidden_dim, out_dim, requires_grad=True)
        b2 = torch.zeros(1, out_dim, requires_grad=True)

        # Forward Pass
        z1 = torch.matmul(t_x, w1) + b1
        z1.retain_grad()

        if activation.lower() == "gelu":
            a1 = torch.nn.functional.gelu(z1)
        elif activation.lower() == "sigmoid":
            a1 = torch.sigmoid(z1)
        elif activation.lower() == "tanh":
            a1 = torch.tanh(z1)
        else:
            a1 = torch.relu(z1)
        a1.retain_grad()

        z2 = torch.matmul(a1, w2) + b2
        z2.retain_grad()

        y_pred = z2
        loss = 0.5 * torch.sum((y_pred - t_y) ** 2)

        # Backward Pass
        loss.backward()

        def to_list(t: Optional[torch.Tensor]) -> Any:
            if t is None:
                return None
            arr = t.detach().cpu().numpy()
            return np.round(arr, 4).tolist()

        # Build interactive computation graph nodes and edges
        nodes = [
            {
                "id": "input_x",
                "label": "Input x",
                "type": "input",
                "shape": list(t_x.shape),
                "forward_val": to_list(t_x)[0],
                "grad_val": None,
                "layer": 0,
            },
            {
                "id": "w1",
                "label": "Weights W1",
                "type": "parameter",
                "shape": list(w1.shape),
                "forward_val": to_list(w1),
                "grad_val": to_list(w1.grad),
                "layer": 1,
            },
            {
                "id": "b1",
                "label": "Bias b1",
                "type": "parameter",
                "shape": list(b1.shape),
                "forward_val": to_list(b1)[0],
                "grad_val": to_list(b1.grad)[0] if b1.grad is not None else None,
                "layer": 1,
            },
            {
                "id": "z1",
                "label": f"Linear 1 (z1 = x·W1 + b1)",
                "type": "operation",
                "shape": list(z1.shape),
                "forward_val": to_list(z1)[0],
                "grad_val": to_list(z1.grad)[0] if z1.grad is not None else None,
                "layer": 2,
            },
            {
                "id": "a1",
                "label": f"Activation ({activation.upper()})",
                "type": "activation",
                "shape": list(a1.shape),
                "forward_val": to_list(a1)[0],
                "grad_val": to_list(a1.grad)[0] if a1.grad is not None else None,
                "layer": 3,
            },
            {
                "id": "w2",
                "label": "Weights W2",
                "type": "parameter",
                "shape": list(w2.shape),
                "forward_val": to_list(w2),
                "grad_val": to_list(w2.grad),
                "layer": 4,
            },
            {
                "id": "b2",
                "label": "Bias b2",
                "type": "parameter",
                "shape": list(b2.shape),
                "forward_val": to_list(b2)[0],
                "grad_val": to_list(b2.grad)[0] if b2.grad is not None else None,
                "layer": 4,
            },
            {
                "id": "z2",
                "label": "Linear 2 (z2 = a1·W2 + b2)",
                "type": "operation",
                "shape": list(z2.shape),
                "forward_val": to_list(z2)[0],
                "grad_val": to_list(z2.grad)[0] if z2.grad is not None else None,
                "layer": 5,
            },
            {
                "id": "loss",
                "label": "MSE Loss: 0.5 * ||y_pred - y||²",
                "type": "loss",
                "shape": [],
                "forward_val": round(float(loss.item()), 4),
                "grad_val": 1.0,  # dL/dL = 1.0
                "layer": 6,
            },
        ]

        edges = [
            {"from": "input_x", "to": "z1", "label": "x"},
            {"from": "w1", "to": "z1", "label": "W1"},
            {"from": "b1", "to": "z1", "label": "b1"},
            {"from": "z1", "to": "a1", "label": "z1"},
            {"from": "a1", "to": "z2", "label": "a1"},
            {"from": "w2", "to": "z2", "label": "W2"},
            {"from": "b2", "to": "z2", "label": "b2"},
            {"from": "z2", "to": "loss", "label": "y_pred"},
        ]

        return {
            "success": True,
            "architecture": f"MLP: {in_dim} -> {hidden_dim} -> {out_dim} ({activation.upper()})",
            "loss": round(float(loss.item()), 4),
            "y_target": y_target,
            "y_pred": to_list(y_pred)[0],
            "nodes": nodes,
            "edges": edges,
        }
