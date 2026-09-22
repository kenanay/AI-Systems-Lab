"""
src/simulators/nn_backprop.py

Neural Network Architecture & Backpropagation Simulator
Local AI Research Lab - Kenan AY

Features:
1. Multi-Layer Perceptron (MLP) Autograd & Chain-Rule Simulator:
   - Configurable architecture (Input -> Hidden Layers -> Output)
   - Multiple activations: ReLU, GELU, SwiGLU, Sigmoid, Tanh, LeakyReLU
   - Analytical chain-rule breakdown per layer
   - Gradient health diagnostics (Vanishing, Exploding, Dead Neurons)
   - Interactive SVG-ready node & edge graph representation

2. 2D Loss Landscape & Multi-Optimizer Race Simulator:
   - Benchmark surfaces: Saddle Point, Rosenbrock (Banana), Beale, Quadratic Bowl
   - Optimizers: SGD, Momentum, RMSprop, Adam, AdamW (decoupled weight decay)
   - Step-by-step trajectory generation with convergence metrics
   - 2D contour elevation grid generator for visualization
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 1. MLP Forward & Backpropagation Graph Simulator
# ============================================================================

@dataclass
class MLPNode:
    """Tekil bir nöron düğümü."""
    id: str
    layer_idx: int
    neuron_idx: int
    layer_type: str  # 'input', 'hidden', 'output'
    pre_activation_z: Optional[float] = None
    post_activation_a: Optional[float] = None
    gradient_z: Optional[float] = None
    gradient_a: Optional[float] = None
    is_dead: bool = False


@dataclass
class MLPSynapse:
    """İki nöron arasındaki ağırlık (bağlantı)."""
    id: str
    source_id: str
    target_id: str
    source_layer: int
    target_layer: int
    weight: float
    gradient_w: Optional[float] = None
    weight_update: Optional[float] = None


class MLPSimulator:
    """
    Çok katmanlı yapay sinir ağının (MLP) ileri ve geri geçişlerini
    tüm ara tensörler, türevler ve zincir kuralı adımlarıyla simüle eder.
    """

    SUPPORTED_ACTIVATIONS = ["relu", "gelu", "swiglu", "sigmoid", "tanh", "leaky_relu"]
    SUPPORTED_LOSSES = ["mse", "cross_entropy", "binary_cross_entropy", "smooth_l1"]

    @classmethod
    def apply_activation(cls, x: torch.Tensor, activation_name: str) -> torch.Tensor:
        act = activation_name.lower()
        if act == "gelu":
            return F.gelu(x)
        elif act == "sigmoid":
            return torch.sigmoid(x)
        elif act == "tanh":
            return torch.tanh(x)
        elif act == "leaky_relu":
            return F.leaky_relu(x, negative_slope=0.1)
        elif act == "swiglu":
            # SwiGLU: SiLU(x) * x (simplified 1-stream formulation)
            return F.silu(x) * x
        else:
            return F.relu(x)

    @classmethod
    def activation_derivative_formula(cls, activation_name: str) -> str:
        act = activation_name.lower()
        if act == "relu":
            return "σ'(z) = 1 if z > 0 else 0"
        elif act == "gelu":
            return "σ'(z) ≈ 0.5 * (1 + tanh(√(2/π) * (z + 0.044715 z³))) + z * sech²(...)"
        elif act == "sigmoid":
            return "σ'(z) = σ(z) * (1 - σ(z)) = a * (1 - a)"
        elif act == "tanh":
            return "σ'(z) = 1 - tanh²(z) = 1 - a²"
        elif act == "leaky_relu":
            return "σ'(z) = 1 if z > 0 else 0.1"
        elif act == "swiglu":
            return "σ'(z) = SiLU'(z) * z + SiLU(z)"
        return "σ'(z) = 1"

    @staticmethod
    def _to_numpy(t: Optional[torch.Tensor], default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Tensörü güvenli bir şekilde numpy array'e dönüştürür."""
        if t is not None:
            return t.detach().cpu().numpy()
        return default

    @classmethod
    def simulate(
        cls,
        inputs: List[float],
        targets: List[float],
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        loss_function: str = "mse",
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Tam bir ileri ve geri geçiş çalıştırır.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_dims = [4, 4]
        else:
            hidden_dims = [max(1, min(8, h)) for h in hidden_dims[:3]]

        in_dim = len(inputs)
        out_dim = len(targets)

        t_x = torch.tensor([inputs], dtype=torch.float32, requires_grad=False)
        t_y = torch.tensor([targets], dtype=torch.float32, requires_grad=False)

        layer_dims = [in_dim] + hidden_dims + [out_dim]
        num_layers = len(layer_dims) - 1

        # Ağırlık ve bias tensörleri
        weights: List[torch.Tensor] = []
        biases: List[torch.Tensor] = []

        for i in range(num_layers):
            d_in, d_out = layer_dims[i], layer_dims[i + 1]
            # Xavier/He initialization
            scale = math.sqrt(2.0 / (d_in + d_out))
            w = torch.randn(d_in, d_out, dtype=torch.float32) * scale
            w.requires_grad = True
            b = torch.zeros(1, d_out, dtype=torch.float32, requires_grad=True)
            weights.append(w)
            biases.append(b)

        # İleri Geçiş (Forward Pass)
        z_list: List[torch.Tensor] = []
        a_list: List[torch.Tensor] = [t_x]

        cur_a = t_x
        for i in range(num_layers):
            z = torch.matmul(cur_a, weights[i]) + biases[i]
            z.retain_grad()
            z_list.append(z)

            # Son katman doğrusal (linear) çıktıdır, gizli katmanlara aktivasyon uygulanır
            if i < num_layers - 1:
                a = cls.apply_activation(z, activation)
                a.retain_grad()
                a_list.append(a)
                cur_a = a
            else:
                a_list.append(z)  # output layer a == z
                cur_a = z

        y_pred = cur_a

        # Kayıp Fonksiyonu (Loss Computation)
        loss_name = loss_function.lower()
        if loss_name == "cross_entropy" and out_dim > 1:
            loss = F.cross_entropy(y_pred, torch.argmax(t_y, dim=-1))
        elif loss_name == "binary_cross_entropy":
            probs = torch.sigmoid(y_pred)
            loss = F.binary_cross_entropy(probs, t_y)
        elif loss_name == "smooth_l1":
            loss = F.smooth_l1_loss(y_pred, t_y)
        else:
            loss = 0.5 * torch.sum((y_pred - t_y) ** 2)

        # Geriye Yayılım (Backward Pass via PyTorch Autograd)
        loss.backward()

        # Zincir Kuralı Açıklamaları & Adımları (Chain-Rule Derivations)
        chain_rule_steps = []
        # Çıktı katmanı hatası: delta_L
        out_z = z_list[-1]
        out_grad = out_z.grad.detach().cpu().numpy()[0] if out_z.grad is not None else np.zeros(out_dim)
        chain_rule_steps.append({
            "step": 1,
            "layer": f"Çıktı Katmanı (Layer {num_layers})",
            "formula": "δ_L = ∂L / ∂z_L = y_pred - y_target",
            "latex": r"\delta_L = \frac{\partial \mathcal{L}}{\partial z_L} = \hat{y} - y",
            "values": [round(float(v), 4) for v in out_grad],
            "description": "Çıktı katmanındaki hata sinyali (gradient of loss with respect to output logits).",
        })

        for l_idx in reversed(range(num_layers)):
            w_grad = cls._to_numpy(weights[l_idx].grad)
            b_grad_raw = cls._to_numpy(biases[l_idx].grad)
            b_grad = b_grad_raw[0] if b_grad_raw is not None else None
            w_norm = float(np.linalg.norm(w_grad)) if w_grad is not None else 0.0

            chain_rule_steps.append({
                "step": len(chain_rule_steps) + 1,
                "layer": f"Ağırlık Gradiyeni W_{l_idx + 1}",
                "formula": f"∂L / ∂W_{l_idx + 1} = a_{l_idx}^T · δ_{l_idx + 1}",
                "latex": rf"\frac{{\partial \mathcal{{L}}}}{{\partial W_{{{l_idx + 1}}}}} = a_{{{l_idx}}}^T \delta_{{{l_idx + 1}}}",
                "grad_norm": round(w_norm, 5),
                "bias_grads": [round(float(b), 4) for b in b_grad] if b_grad is not None else [],
                "description": f"Katman {l_idx + 1} parametrelerinin loss'a göre türevi ve L2 gradiyen normu.",
            })

        # Gradiyen Sağlığı & Teşhis (Gradient Health Diagnostics)
        dead_neurons = []
        layer_health = []
        overall_health = "HEALTHY"

        for l_idx in range(num_layers):
            w_g = cls._to_numpy(weights[l_idx].grad, np.array([0.0]))
            assert w_g is not None
            max_abs_g = float(np.max(np.abs(w_g)))

            status = "HEALTHY"
            if max_abs_g < 1e-5:
                status = "VANISHING_RISK"
                if overall_health == "HEALTHY":
                    overall_health = "VANISHING_RISK"
            elif max_abs_g > 10.0:
                status = "EXPLODING_RISK"
                overall_health = "EXPLODING_RISK"

            layer_health.append({
                "layer_idx": l_idx + 1,
                "max_grad": round(max_abs_g, 6),
                "mean_grad": round(float(np.mean(np.abs(w_g))), 6),
                "status": status,
            })

            # Gizli katmanlarda ölü ReLU tespiti (a_i == 0 ve grad == 0)
            if l_idx < num_layers - 1:
                z_vals = z_list[l_idx].detach().cpu().numpy()[0]
                a_vals = a_list[l_idx + 1].detach().cpu().numpy()[0]
                z_g_raw = cls._to_numpy(z_list[l_idx].grad)
                z_grads = z_g_raw[0] if z_g_raw is not None else np.zeros_like(z_vals)

                for n_idx in range(len(a_vals)):
                    if abs(a_vals[n_idx]) < 1e-7 and abs(z_grads[n_idx]) < 1e-7:
                        dead_neurons.append(f"L{l_idx + 1}_N{n_idx + 1}")

        if dead_neurons and overall_health == "HEALTHY":
            overall_health = "DEAD_NEURONS_DETECTED"

        # SVG Graf Düğümleri (Nodes) & Bağlantıları (Synapses)
        nodes: List[Dict[str, Any]] = []
        synapses: List[Dict[str, Any]] = []

        # 1. Giriş düğümleri
        for i, val in enumerate(inputs):
            nodes.append({
                "id": f"in_{i}",
                "label": f"x_{i+1}",
                "layer_idx": 0,
                "neuron_idx": i,
                "layer_type": "input",
                "pre_activation_z": round(float(val), 4),
                "post_activation_a": round(float(val), 4),
                "gradient_z": None,
                "gradient_a": None,
                "is_dead": False,
            })

        # 2. Gizli katman düğümleri
        for l_idx in range(len(hidden_dims)):
            z_vals = z_list[l_idx].detach().cpu().numpy()[0]
            a_vals = a_list[l_idx + 1].detach().cpu().numpy()[0]
            z_g_raw = cls._to_numpy(z_list[l_idx].grad)
            z_grads = z_g_raw[0] if z_g_raw is not None else np.zeros_like(z_vals)
            a_g_raw = cls._to_numpy(a_list[l_idx + 1].grad)
            a_grads = a_g_raw[0] if a_g_raw is not None else np.zeros_like(a_vals)

            for n_idx in range(hidden_dims[l_idx]):
                node_id = f"h{l_idx+1}_{n_idx}"
                is_dead = abs(float(a_vals[n_idx])) < 1e-7 and abs(float(z_grads[n_idx])) < 1e-7
                nodes.append({
                    "id": node_id,
                    "label": f"h_{{{l_idx+1},{n_idx+1}}}",
                    "layer_idx": l_idx + 1,
                    "neuron_idx": n_idx,
                    "layer_type": "hidden",
                    "pre_activation_z": round(float(z_vals[n_idx]), 4),
                    "post_activation_a": round(float(a_vals[n_idx]), 4),
                    "gradient_z": round(float(z_grads[n_idx]), 4),
                    "gradient_a": round(float(a_grads[n_idx]), 4),
                    "is_dead": is_dead,
                })

        # 3. Çıktı katmanı düğümleri
        out_z_vals = z_list[-1].detach().cpu().numpy()[0]
        out_g_raw = cls._to_numpy(z_list[-1].grad)
        out_z_grads = out_g_raw[0] if out_g_raw is not None else np.zeros(out_dim)
        for i in range(out_dim):
            nodes.append({
                "id": f"out_{i}",
                "label": f"ŷ_{i+1}",
                "layer_idx": num_layers,
                "neuron_idx": i,
                "layer_type": "output",
                "pre_activation_z": round(float(out_z_vals[i]), 4),
                "post_activation_a": round(float(out_z_vals[i]), 4),
                "gradient_z": round(float(out_z_grads[i]), 4),
                "gradient_a": round(float(out_z_grads[i]), 4),
                "target_val": round(float(targets[i]), 4),
                "is_dead": False,
            })

        # 4. Sinapslar (Ağırlıklar & Gradiyenler)
        for l_idx in range(num_layers):
            w_mat = weights[l_idx].detach().cpu().numpy()
            w_g_mat = cls._to_numpy(weights[l_idx].grad)
            w_grad_mat = w_g_mat if w_g_mat is not None else np.zeros_like(w_mat)

            src_layer = l_idx
            tgt_layer = l_idx + 1
            src_count = layer_dims[src_layer]
            tgt_count = layer_dims[tgt_layer]

            for s_idx in range(src_count):
                src_id = f"in_{s_idx}" if src_layer == 0 else f"h{src_layer}_{s_idx}"
                for t_idx in range(tgt_count):
                    tgt_id = f"out_{t_idx}" if tgt_layer == num_layers else f"h{tgt_layer}_{t_idx}"
                    w_val = float(w_mat[s_idx, t_idx])
                    g_val = float(w_grad_mat[s_idx, t_idx])
                    w_up = -learning_rate * g_val

                    synapses.append({
                        "id": f"syn_{src_id}_{tgt_id}",
                        "source_id": src_id,
                        "target_id": tgt_id,
                        "source_layer": src_layer,
                        "target_layer": tgt_layer,
                        "weight": round(w_val, 4),
                        "gradient_w": round(g_val, 4),
                        "weight_update": round(w_up, 4),
                    })

        return {
            "loss": round(float(loss.item()), 5),
            "loss_function": loss_function,
            "activation": activation,
            "learning_rate": learning_rate,
            "architecture": layer_dims,
            "predictions": [round(float(p), 4) for p in y_pred.detach().cpu().numpy()[0]],
            "targets": targets,
            "nodes": nodes,
            "synapses": synapses,
            "chain_rule_steps": chain_rule_steps,
            "layer_health": layer_health,
            "dead_neurons": dead_neurons,
            "overall_health": overall_health,
            "activation_derivative_formula": cls.activation_derivative_formula(activation),
        }


# ============================================================================
# 2. 2D Loss Landscape & Multi-Optimizer Race Simulator
# ============================================================================

class LossLandscapes:
    """Analitik 2D test kayıp yüzeyleri ve analitik türevleri."""

    @staticmethod
    def saddle(x: float, y: float) -> float:
        """Saddle Point: f(x, y) = x² - y²"""
        return float(x**2 - y**2)

    @staticmethod
    def saddle_grad(x: float, y: float) -> Tuple[float, float]:
        return (2.0 * x, -2.0 * y)

    @staticmethod
    def rosenbrock(x: float, y: float, a: float = 1.0, b: float = 100.0) -> float:
        """Rosenbrock Banana Function: f(x, y) = (a - x)² + b(y - x²)²"""
        return float((a - x)**2 + b * (y - x**2)**2)

    @staticmethod
    def rosenbrock_grad(x: float, y: float, a: float = 1.0, b: float = 100.0) -> Tuple[float, float]:
        dx = -2.0 * (a - x) - 400.0 * x * (y - x**2)
        dy = 200.0 * (y - x**2)
        return (float(dx), float(dy))

    @staticmethod
    def beale(x: float, y: float) -> float:
        """Beale Function"""
        term1 = 1.5 - x + x * y
        term2 = 2.25 - x + x * (y**2)
        term3 = 2.625 - x + x * (y**3)
        return float(term1**2 + term2**2 + term3**2)

    @staticmethod
    def beale_grad(x: float, y: float) -> Tuple[float, float]:
        term1 = 1.5 - x + x * y
        term2 = 2.25 - x + x * (y**2)
        term3 = 2.625 - x + x * (y**3)

        dx = 2 * term1 * (-1 + y) + 2 * term2 * (-1 + y**2) + 2 * term3 * (-1 + y**3)
        dy = 2 * term1 * x + 2 * term2 * (2 * x * y) + 2 * term3 * (3 * x * (y**2))
        return (float(dx), float(dy))

    @staticmethod
    def quadratic_bowl(x: float, y: float) -> float:
        """Ill-conditioned Quadratic Bowl: f(x, y) = 0.5 * (x² + 15y²)"""
        return float(0.5 * (x**2 + 15.0 * y**2))

    @staticmethod
    def quadratic_bowl_grad(x: float, y: float) -> Tuple[float, float]:
        return (float(x), float(15.0 * y))


class OptimizerRaceSimulator:
    """
    Seçilen analitik 2D yüzey üzerinde birden çok optimizasyon algoritmasını
    (SGD, Momentum, RMSprop, Adam, AdamW) eşzamanlı yarıştırır.
    """

    LANDSCAPES = {
        "saddle": {
            "name": "Saddle Point (Eyer Noktası)",
            "fn": LossLandscapes.saddle,
            "grad": LossLandscapes.saddle_grad,
            "formula": "f(x, y) = x² - y²",
            "x_range": (-2.5, 2.5),
            "y_range": (-2.5, 2.5),
            "default_start": (-0.1, 1.8),
            "optimum": (0.0, 0.0),
            "description": "Gradiyeni sıfır olan eyer noktasından kaçış hızını ve momentum etkisini test eder.",
        },
        "rosenbrock": {
            "name": "Rosenbrock (Muz Vadisi)",
            "fn": LossLandscapes.rosenbrock,
            "grad": LossLandscapes.rosenbrock_grad,
            "formula": "f(x, y) = (1 - x)² + 100(y - x²)²",
            "x_range": (-2.0, 2.2),
            "y_range": (-1.0, 3.2),
            "default_start": (-1.5, 2.0),
            "optimum": (1.0, 1.0),
            "description": "Dar ve kavisli bir vadi boyunca gradyenin yön değiştirme ve adaptif adım ihtiyacını gösterir.",
        },
        "beale": {
            "name": "Beale Fonksiyonu",
            "fn": LossLandscapes.beale,
            "grad": LossLandscapes.beale_grad,
            "formula": "f(x, y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²",
            "x_range": (-4.5, 4.5),
            "y_range": (-4.5, 4.5),
            "default_start": (-3.0, -1.0),
            "optimum": (3.0, 0.5),
            "description": "Köşeleri çok dik, düzlükleri geniş çok modlu zorlu optimizasyon yüzeyi.",
        },
        "quadratic_bowl": {
            "name": "Kötü Şartlanmış Eliptik Çanak (Ill-Conditioned Quadratic)",
            "fn": LossLandscapes.quadratic_bowl,
            "grad": LossLandscapes.quadratic_bowl_grad,
            "formula": "f(x, y) = 0.5 * (x² + 15y²)",
            "x_range": (-3.0, 3.0),
            "y_range": (-2.0, 2.0),
            "default_start": (2.5, 1.5),
            "optimum": (0.0, 0.0),
            "description": "Farklı eksenlerdeki eğrilik dengesizliği nedeniyle SGD'nin salınım yaptığı, Adam/RMSprop'un hızla indiği yüzey.",
        },
    }

    OPTIMIZER_CONFIGS = {
        "sgd": {"name": "SGD (Standard)", "color": "#ef4444", "description": "Temel gradyen inişi; momentum veya uyarlama içermez."},
        "momentum": {"name": "SGD + Momentum", "color": "#f97316", "description": "Önceki gradyenlerin hareketini koruyarak salınımları sönümler."},
        "rmsprop": {"name": "RMSprop", "color": "#06b6d4", "description": "Son gradyenlerin karelerinin hareketli ortalamasına göre adımı ölçekler."},
        "adam": {"name": "Adam", "color": "#8b5cf6", "description": "Hem 1. (momentum) hem 2. (RMSprop) momentleri bias-düzeltmeli birleştirir."},
        "adamw": {"name": "AdamW (Decoupled Weight Decay)", "color": "#10b981", "description": "L2 regülarizasyonunu gradyen güncellemesinden ayırır; modern LLM standartıdır."},
    }

    @classmethod
    def get_landscape_contours(
        cls,
        landscape_key: str = "saddle",
        grid_size: int = 35,
    ) -> Dict[str, Any]:
        """
        Görselleştirme için 2D eşyükselti (contour) ızgarası üretir.
        """
        land = cls.LANDSCAPES.get(landscape_key, cls.LANDSCAPES["saddle"])
        x_min, x_max = land["x_range"]
        y_min, y_max = land["y_range"]
        fn = land["fn"]

        x_vals = np.linspace(x_min, x_max, grid_size).tolist()
        y_vals = np.linspace(y_min, y_max, grid_size).tolist()

        z_grid = []
        z_min = float("inf")
        z_max = float("-inf")

        for y in y_vals:
            row = []
            for x in x_vals:
                val = fn(x, y)
                if not math.isnan(val) and not math.isinf(val):
                    val = min(1000.0, max(-1000.0, val))
                else:
                    val = 0.0
                z_min = min(z_min, val)
                z_max = max(z_max, val)
                row.append(round(val, 2))
            z_grid.append(row)

        return {
            "landscape_key": landscape_key,
            "x_range": [x_min, x_max],
            "y_range": [y_min, y_max],
            "x_vals": [round(x, 3) for x in x_vals],
            "y_vals": [round(y, 3) for y in y_vals],
            "z_grid": z_grid,
            "z_min": round(z_min, 2),
            "z_max": round(z_max, 2),
            "optimum": land["optimum"],
        }

    @classmethod
    def run_race(
        cls,
        landscape_key: str = "quadratic_bowl",
        optimizers: Optional[List[str]] = None,
        start_pos: Optional[Tuple[float, float]] = None,
        learning_rate: float = 0.02,
        momentum_beta: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 0.01,
        steps: int = 60,
    ) -> Dict[str, Any]:
        """
        Belirtilen yüzeyde seçilen optimizer'ların yarış simülasyonunu koşturur.
        """
        land = cls.LANDSCAPES.get(landscape_key, cls.LANDSCAPES["quadratic_bowl"])
        fn = land["fn"]
        grad_fn = land["grad"]

        if start_pos is None:
            sx, sy = land["default_start"]
        else:
            sx, sy = start_pos

        if optimizers is None or len(optimizers) == 0:
            optimizers = ["sgd", "momentum", "rmsprop", "adam", "adamw"]

        steps = max(10, min(120, steps))
        results: Dict[str, Any] = {}
        eps = 1e-8

        for opt in optimizers:
            opt_key = opt.lower()
            if opt_key not in cls.OPTIMIZER_CONFIGS:
                continue

            x, y = float(sx), float(sy)
            traj: List[Dict[str, float]] = []

            # Optimizer durum değişkenleri
            vx, vy = 0.0, 0.0  # momentum / 1. moment
            sx_sq, sy_sq = 0.0, 0.0  # RMSprop / 2. moment

            for t in range(1, steps + 1):
                cur_loss = fn(x, y)
                gx, gy = grad_fn(x, y)

                # Gradyen kırpma (patlamayı önlemek için)
                norm = math.hypot(gx, gy)
                if norm > 50.0:
                    gx = (gx / norm) * 50.0
                    gy = (gy / norm) * 50.0

                traj.append({
                    "step": t - 1,
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "loss": round(cur_loss, 4),
                    "grad_norm": round(norm, 4),
                })

                # Optimizer Adımı
                if opt_key == "sgd":
                    x -= learning_rate * gx
                    y -= learning_rate * gy

                elif opt_key == "momentum":
                    vx = momentum_beta * vx + learning_rate * gx
                    vy = momentum_beta * vy + learning_rate * gy
                    x -= vx
                    y -= vy

                elif opt_key == "rmsprop":
                    sx_sq = 0.9 * sx_sq + 0.1 * (gx**2)
                    sy_sq = 0.9 * sy_sq + 0.1 * (gy**2)
                    step_x = (learning_rate / (math.sqrt(sx_sq) + eps)) * gx
                    step_y = (learning_rate / (math.sqrt(sy_sq) + eps)) * gy
                    x -= step_x
                    y -= step_y

                elif opt_key == "adam":
                    # Adam with L2 regularization
                    gx += weight_decay * x
                    gy += weight_decay * y

                    vx = momentum_beta * vx + (1.0 - momentum_beta) * gx
                    vy = momentum_beta * vy + (1.0 - momentum_beta) * gy
                    sx_sq = beta2 * sx_sq + (1.0 - beta2) * (gx**2)
                    sy_sq = beta2 * sy_sq + (1.0 - beta2) * (gy**2)

                    # Bias correction
                    m_x = vx / (1.0 - (momentum_beta**t))
                    m_y = vy / (1.0 - (momentum_beta**t))
                    v_x = sx_sq / (1.0 - (beta2**t))
                    v_y = sy_sq / (1.0 - (beta2**t))

                    x -= (learning_rate / (math.sqrt(v_x) + eps)) * m_x
                    y -= (learning_rate / (math.sqrt(v_y) + eps)) * m_y

                elif opt_key == "adamw":
                    # AdamW: Decoupled weight decay update
                    vx = momentum_beta * vx + (1.0 - momentum_beta) * gx
                    vy = momentum_beta * vy + (1.0 - momentum_beta) * gy
                    sx_sq = beta2 * sx_sq + (1.0 - beta2) * (gx**2)
                    sy_sq = beta2 * sy_sq + (1.0 - beta2) * (gy**2)

                    m_x = vx / (1.0 - (momentum_beta**t))
                    m_y = vy / (1.0 - (momentum_beta**t))
                    v_x = sx_sq / (1.0 - (beta2**t))
                    v_y = sy_sq / (1.0 - (beta2**t))

                    # Decoupled step: x = x*(1 - lr*decay) - lr*adam_step
                    x = x * (1.0 - learning_rate * weight_decay) - (learning_rate / (math.sqrt(v_x) + eps)) * m_x
                    y = y * (1.0 - learning_rate * weight_decay) - (learning_rate / (math.sqrt(v_y) + eps)) * m_y

                # NaN/Inf koruması
                if math.isnan(x) or math.isinf(x):
                    x = 10.0
                if math.isnan(y) or math.isinf(y):
                    y = 10.0

            final_loss = fn(x, y)
            traj.append({
                "step": steps,
                "x": round(x, 4),
                "y": round(y, 4),
                "loss": round(final_loss, 4),
                "grad_norm": round(math.hypot(*grad_fn(x, y)), 4),
            })

            # Toplam kat edilen mesafe
            total_dist = sum(
                math.hypot(traj[i]["x"] - traj[i-1]["x"], traj[i]["y"] - traj[i-1]["y"])
                for i in range(1, len(traj))
            )

            # Hedefe yakınlık (optimum mesafesi)
            ox, oy = land["optimum"]
            opt_dist = math.hypot(x - ox, y - oy)

            results[opt_key] = {
                "name": cls.OPTIMIZER_CONFIGS[opt_key]["name"],
                "color": cls.OPTIMIZER_CONFIGS[opt_key]["color"],
                "description": cls.OPTIMIZER_CONFIGS[opt_key]["description"],
                "trajectory": traj,
                "final_position": (round(x, 4), round(y, 4)),
                "final_loss": round(final_loss, 4),
                "initial_loss": traj[0]["loss"],
                "loss_reduction_pct": round(max(0.0, (traj[0]["loss"] - final_loss) / max(1e-4, abs(traj[0]["loss"]))) * 100.0, 1),
                "total_distance": round(total_dist, 3),
                "distance_to_optimum": round(opt_dist, 4),
            }

        # Kazanan optimizer (En düşük final kayıp ve en yakın optimum)
        winner_key = min(results.keys(), key=lambda k: (results[k]["final_loss"], results[k]["distance_to_optimum"]))

        return {
            "landscape": {
                "key": landscape_key,
                "name": land["name"],
                "formula": land["formula"],
                "description": land["description"],
                "optimum": land["optimum"],
                "start_position": (round(sx, 3), round(sy, 3)),
            },
            "parameters": {
                "learning_rate": learning_rate,
                "momentum": momentum_beta,
                "weight_decay": weight_decay,
                "steps": steps,
            },
            "optimizers": results,
            "winner": winner_key,
            "winner_name": results[winner_key]["name"],
            "contours": cls.get_landscape_contours(landscape_key),
        }
