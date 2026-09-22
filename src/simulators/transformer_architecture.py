"""
Positional Encoding & Transformer Architecture Simulation Engine

Provides deep-learning simulations for:
- Sinusoidal Positional Encoding (Vaswani 2017)
- RoPE (Rotary Position Embedding, Su et al. - LLaMA, Mistral, Qwen)
- ALiBi (Attention with Linear Biases, Press et al.)
- Multi-Head (MHA), Grouped-Query (GQA), and Multi-Query (MQA) Attention variants with KV-Cache memory profiling
- Transformer Block sub-layers (Pre-LN vs Post-LN, RMSNorm vs LayerNorm, SwiGLU vs standard MLP)
- Complete Model Architecture parameter & VRAM footprint calculator

Author: Kenan AY
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import torch


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


class PositionalEncodingEngine:
    """Simulates Sinusoidal PE, Rotary Position Embeddings (RoPE), and ALiBi."""

    @staticmethod
    def compute_sinusoidal(seq_len: int = 16, d_model: int = 32) -> Dict[str, Any]:
        """
        Computes Vaswani et al. (2017) Sinusoidal Positional Encoding:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = np.zeros((seq_len, d_model), dtype=float)
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Compute dot product similarity matrix between positions: PE @ PE.T
        norm_pe = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-12)
        similarity_matrix = np.dot(norm_pe, norm_pe.T)

        # 1D waves for a few selected dimension channels (fast vs slow frequencies)
        wave_channels = [0, min(2, d_model - 1), min(d_model // 2, d_model - 1), d_model - 1]
        waves = []
        for ch in sorted(list(set(wave_channels))):
            waves.append({
                "dimension_index": ch,
                "values": [round(float(v), 4) for v in pe[:, ch]],
                "type": "sin" if ch % 2 == 0 else "cos",
                "wavelength_approx": round(2 * math.pi / (div_term[ch // 2] if ch // 2 < len(div_term) else 1.0), 1),
            })

        return {
            "seq_len": seq_len,
            "d_model": d_model,
            "pe_matrix": np.round(pe, 4).tolist(),
            "similarity_matrix": np.round(similarity_matrix, 4).tolist(),
            "waves": waves,
            "description": "Sinusoidal positional encoding generates unique, deterministic harmonic frequencies. Lower dimensions change rapidly across tokens, while higher dimensions change slowly.",
        }

    @staticmethod
    def compute_rope(
        dim: int = 16,
        max_seq_len: int = 8,
        base: float = 10000.0,
        q_norm: float = 1.0,
        k_norm: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Simulates Rotary Position Embedding (RoPE) 2D vector rotations:
        For each pair of dimensions (2i, 2i+1), rotate 2D vector by angle m * theta_i.
        Demonstrates the key mathematical property:
        (R_m * q)^T (R_n * k) = q^T R_{n-m} k (depends strictly on relative distance n - m).
        """
        # Frequency scale theta_i = base^(-2i / dim)
        num_pairs = dim // 2
        thetas = [base ** (-2.0 * i / dim) for i in range(num_pairs)]

        # Let's take a sample 2D Query vector and Key vector at the first pair (theta_0)
        theta_0 = thetas[0]
        q_init = np.array([q_norm, 0.0])
        k_init = np.array([k_norm * 0.8, k_norm * 0.6])

        positions_data = []
        for m in range(max_seq_len):
            angle = m * theta_0
            # 2D Rotation matrix
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            r_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

            q_rot = np.dot(r_mat, q_init)
            k_rot = np.dot(r_mat, k_init)

            positions_data.append({
                "position": m,
                "angle_rad": round(angle, 4),
                "angle_deg": round(math.degrees(angle) % 360, 1),
                "q_rotated": [round(float(q_rot[0]), 4), round(float(q_rot[1]), 4)],
                "k_rotated": [round(float(k_rot[0]), 4), round(float(k_rot[1]), 4)],
            })

        # Calculate Relative Distance Decay Matrix: Dot product of rotated vectors
        # S[m, n] = (R_m q) . (R_n k)
        relative_dot_matrix = np.zeros((max_seq_len, max_seq_len), dtype=float)
        for m in range(max_seq_len):
            q_m = np.array(positions_data[m]["q_rotated"])
            for n in range(max_seq_len):
                k_n = np.array(positions_data[n]["k_rotated"])
                relative_dot_matrix[m, n] = np.dot(q_m, k_n)

        return {
            "dim": dim,
            "max_seq_len": max_seq_len,
            "base": base,
            "theta_values": [round(float(t), 6) for t in thetas],
            "positions_data": positions_data,
            "relative_dot_matrix": np.round(relative_dot_matrix, 4).tolist(),
            "explanation": "RoPE preserves relative distance naturally: rotating Q by m*theta and K by n*theta means their inner product depends only on (n - m). As relative distance increases, high-frequency heads decay smoothly.",
        }

    @staticmethod
    def compute_alibi(num_heads: int = 8, seq_len: int = 8) -> Dict[str, Any]:
        """
        Simulates ALiBi (Attention with Linear Biases, Press et al. 2022).
        Applies a non-learned linear distance bias directly to pre-softmax attention logits:
        Bias[h, i, j] = -m_h * (i - j) for i >= j (causal) or |i - j|.
        Slopes m_h are geometric sequence: 2^(-8/H * h).
        """
        def get_slopes(n: int) -> List[float]:
            # Returns powers of 2^(-8/n)
            start = 2.0 ** (-8.0 / n)
            return [start ** i for i in range(1, n + 1)]

        slopes = get_slopes(num_heads)
        heads_matrices = []

        for h, m_val in enumerate(slopes):
            mat = np.zeros((seq_len, seq_len), dtype=float)
            for i in range(seq_len):
                for j in range(seq_len):
                    dist = abs(i - j)
                    mat[i, j] = -m_val * dist
            heads_matrices.append({
                "head_index": h,
                "slope": round(float(m_val), 5),
                "bias_matrix": np.round(mat, 3).tolist(),
            })

        return {
            "num_heads": num_heads,
            "seq_len": seq_len,
            "slopes": [round(float(s), 5) for s in slopes],
            "heads": heads_matrices,
            "description": "ALiBi completely eliminates positional embeddings by penalizing attention scores linearly with token distance. Each head has a different slope, allowing some heads to attend locally while others attend globally.",
        }

    @staticmethod
    def compare_methods() -> List[Dict[str, Any]]:
        """Provides pedagogical comparison of modern positional encoding methods."""
        return [
            {
                "name": "RoPE (Rotary Position Embedding)",
                "authors": "Su et al. (2021)",
                "modern_usage": "LLaMA-1/2/3, Mistral, Mixtral, Qwen, DeepSeek, Gemma",
                "type": "Relative (Multiplicative Rotation in Complex Plane)",
                "parameter_overhead": "0 (No trainable weights)",
                "extrapolation": "High (Can be scaled via YaRN, RoPE Scaling, NTK-aware interpolation)",
                "mechanism": "Rotates Query and Key vector pairs in 2D coordinate spaces using position index m.",
            },
            {
                "name": "Sinusoidal Encoding",
                "authors": "Vaswani et al. (2017)",
                "modern_usage": "Original Transformer, BERT (as reference), ViT",
                "type": "Absolute (Additive Fixed Harmonics)",
                "parameter_overhead": "0 (Fixed trigonometric functions)",
                "extrapolation": "Moderate (Cannot easily generalize far beyond training length)",
                "mechanism": "Adds fixed sine and cosine waves of varying frequencies directly to token embeddings.",
            },
            {
                "name": "ALiBi (Attention with Linear Biases)",
                "authors": "Press et al. (2022)",
                "modern_usage": "BLOOM, MPT-7B/30B, Falcon",
                "type": "Relative (Static Logit Bias Penalty)",
                "parameter_overhead": "0 (Fixed geometric slopes)",
                "extrapolation": "Very High (Trained on 1K tokens, evaluates up to 8K+ without retraining)",
                "mechanism": "Subtracts static penalty proportional to token distance from attention logits before Softmax.",
            },
            {
                "name": "Learned Absolute Embeddings",
                "authors": "Gehring et al., Devlin et al. (BERT)",
                "modern_usage": "BERT, RoBERTa, GPT-2, GPT-3",
                "type": "Absolute (Learned Lookup Table)",
                "parameter_overhead": "d_model * max_seq_len (Requires memory for lookup table)",
                "extrapolation": "Zero (Strictly cannot process sequences longer than max_seq_len)",
                "mechanism": "Trains an embedding vector for each position index [0..N-1].",
            },
        ]


class AttentionVariantsEngine:
    """Analyzes Multi-Head (MHA), Grouped-Query (GQA), and Multi-Query (MQA) Attention."""

    @staticmethod
    def analyze_kv_cache(
        batch_size: int = 1,
        seq_len: int = 2048,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        num_layers: int = 32,
        dtype: str = "float16",
    ) -> Dict[str, Any]:
        """
        Profiles KV-Cache memory consumption and memory bandwidth requirements:
        KV-Cache Size = 2 (K & V) * batch_size * seq_len * num_kv_heads * head_dim * bytes_per_elem * num_layers
        """
        bytes_per_elem = 2 if dtype.lower() in ["float16", "fp16", "bfloat16", "bf16"] else 4 if dtype.lower() in ["float32", "fp32"] else 1

        # Classify variant
        if num_kv_heads == num_query_heads:
            variant_type = "MHA (Multi-Head Attention)"
            description = "Standard attention. Each Query head has its own dedicated Key and Value head."
        elif num_kv_heads == 1:
            variant_type = "MQA (Multi-Query Attention)"
            description = "Extreme compression. All Query heads share a single Key and single Value head."
        elif 1 < num_kv_heads < num_query_heads:
            group_size = num_query_heads // num_kv_heads
            variant_type = f"GQA (Grouped-Query Attention, {group_size}:1)"
            description = f"Modern LLM standard (LLaMA-3, Mistral). Every group of {group_size} Query heads shares 1 KV head."
        else:
            variant_type = "Custom Head Configuration"
            description = "Custom ratio."

        # Compute for Current Configuration
        kv_elements_per_layer = 2 * batch_size * seq_len * num_kv_heads * head_dim
        kv_bytes_per_layer = kv_elements_per_layer * bytes_per_elem
        kv_total_bytes = kv_bytes_per_layer * num_layers

        # Compute Baseline MHA for Comparison
        mha_elements_per_layer = 2 * batch_size * seq_len * num_query_heads * head_dim
        mha_total_bytes = mha_elements_per_layer * bytes_per_elem * num_layers

        # Compute MQA for Comparison
        mqa_elements_per_layer = 2 * batch_size * seq_len * 1 * head_dim
        mqa_total_bytes = mqa_elements_per_layer * bytes_per_elem * num_layers

        # Memory savings
        savings_ratio = round(mha_total_bytes / max(1, kv_total_bytes), 2)
        savings_pct = round((1.0 - (kv_total_bytes / max(1, mha_total_bytes))) * 100.0, 1)

        # Memory Bandwidth per Generation Step (Memory-bound autoregressive decoding)
        # To generate 1 token, GPU must read the entire KV-cache from HBM to SRAM!
        bandwidth_per_token_bytes = kv_total_bytes
        # If GPU has 1000 GB/s bandwidth (e.g. A100), max theoretical tokens/sec:
        theoretical_tokens_per_sec = round((1000 * 1024 ** 3) / max(1, kv_total_bytes), 1)

        # Groups visualization mapping
        queries_per_kv = num_query_heads // max(1, num_kv_heads)
        groups = []
        for kv_idx in range(min(num_kv_heads, 8)):  # Limit visual to 8 groups
            q_start = kv_idx * queries_per_kv
            q_end = q_start + queries_per_kv - 1
            groups.append({
                "kv_head_index": kv_idx,
                "shared_query_heads": list(range(q_start, min(q_end + 1, num_query_heads))),
                "query_count": queries_per_kv,
            })

        return {
            "variant_type": variant_type,
            "description": description,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "num_layers": num_layers,
            "dtype": dtype,
            "bytes_per_elem": bytes_per_elem,
            "queries_per_kv_head": queries_per_kv,
            "current_kv_cache": {
                "total_bytes": kv_total_bytes,
                "formatted": format_bytes(kv_total_bytes),
                "bytes_per_layer": kv_bytes_per_layer,
                "formatted_per_layer": format_bytes(kv_bytes_per_layer),
            },
            "comparison": {
                "mha_baseline_formatted": format_bytes(mha_total_bytes),
                "mha_baseline_bytes": mha_total_bytes,
                "mqa_formatted": format_bytes(mqa_total_bytes),
                "mqa_bytes": mqa_total_bytes,
                "savings_vs_mha_pct": savings_pct,
                "savings_multiplier": f"{savings_ratio}x",
            },
            "memory_bandwidth": {
                "read_per_step_formatted": format_bytes(bandwidth_per_token_bytes),
                "theoretical_throughput_a100": f"{theoretical_tokens_per_sec} tok/s (1 TB/s HBM)",
            },
            "head_groups": groups,
        }


class TransformerBlockEngine:
    """Simulates complete Transformer sub-layer flow, Pre-LN vs Post-LN, RMSNorm, and SwiGLU."""

    @staticmethod
    def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Root Mean Square Layer Normalization: x / sqrt(mean(x^2) + eps)."""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return x / rms

    @staticmethod
    def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Standard Layer Normalization: (x - mean) / sqrt(var + eps)."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    @staticmethod
    def simulate_forward_pass(
        norm_type: str = "rmsnorm",
        ffn_type: str = "swiglu",
        norm_placement: str = "pre_ln",
        batch_size: int = 1,
        seq_len: int = 4,
        d_model: int = 8,
        d_ff: int = 16,
    ) -> Dict[str, Any]:
        """
        Simulates one full Transformer Block forward pass with random sample embeddings.
        Tracks tensor dimensions, mean, and variance at each sub-layer.
        """
        np.random.seed(42)
        x = np.random.randn(batch_size, seq_len, d_model)

        def norm_fn(t: np.ndarray) -> np.ndarray:
            return TransformerBlockEngine.rms_norm(t) if norm_type.lower() == "rmsnorm" else TransformerBlockEngine.layer_norm(t)

        stages = []
        def log_stage(name: str, tensor: np.ndarray, desc: str):
            stages.append({
                "sublayer_name": name,
                "shape": list(tensor.shape),
                "mean": round(float(np.mean(tensor)), 4),
                "std": round(float(np.std(tensor)), 4),
                "min": round(float(np.min(tensor)), 4),
                "max": round(float(np.max(tensor)), 4),
                "description": desc,
            })

        log_stage("Block Input x", x, "Input token representations from previous layer or embedding layer.")

        # Self-Attention Sub-layer
        if norm_placement == "pre_ln":
            norm1 = norm_fn(x)
            log_stage("Pre-Attention Norm", norm1, f"Normalized inputs via {norm_type.upper()} before attention.")
            # Attention output (simulated linear projection)
            w_attn = np.random.randn(d_model, d_model) * 0.1
            attn_out = np.matmul(norm1, w_attn)
            log_stage("Multi-Head Attention", attn_out, "Query-Key-Value projections and attention weighted sum.")
            # Residual 1
            x_res1 = x + attn_out
            log_stage("Residual Addition 1 (x + Attn)", x_res1, "Skip connection prevents vanishing gradients in deep networks.")
        else:
            # Post-LN
            w_attn = np.random.randn(d_model, d_model) * 0.1
            attn_out = np.matmul(x, w_attn)
            log_stage("Multi-Head Attention", attn_out, "Attention projection.")
            x_res1 = norm_fn(x + attn_out)
            log_stage("Post-Attention Norm", x_res1, "LayerNorm applied after residual addition (requires warmup).")

        # Feed-Forward Sub-layer
        if norm_placement == "pre_ln":
            norm2 = norm_fn(x_res1)
            log_stage("Pre-FFN Norm", norm2, f"Normalized activations before Feed-Forward Network via {norm_type.upper()}.")
            curr_in = norm2
        else:
            curr_in = x_res1

        if ffn_type.lower() == "swiglu":
            # SwiGLU: (SiLU(x * W_gate) * (x * W_up)) * W_down
            w_gate = np.random.randn(d_model, d_ff) * 0.1
            w_up = np.random.randn(d_model, d_ff) * 0.1
            w_down = np.random.randn(d_ff, d_model) * 0.1

            gate = TransformerBlockEngine.silu(np.matmul(curr_in, w_gate))
            up = np.matmul(curr_in, w_up)
            ffn_out = np.matmul(gate * up, w_down)
            log_stage("SwiGLU FFN Layer", ffn_out, f"Gated Linear Unit: SiLU(x·W_gate) ⊙ (x·W_up) · W_down (Dimension {d_model} -> {d_ff} -> {d_model}).")
        else:
            # Standard MLP: GELU(x * W_1) * W_2
            w1 = np.random.randn(d_model, d_ff) * 0.1
            w2 = np.random.randn(d_ff, d_model) * 0.1
            h = np.maximum(0, np.matmul(curr_in, w1))  # ReLU / GELU approx
            ffn_out = np.matmul(h, w2)
            log_stage("Standard MLP FFN", ffn_out, f"Standard 2-layer FFN: Activation(x·W1)·W2 (Dimension {d_model} -> {d_ff} -> {d_model}).")

        # Final Block Output
        if norm_placement == "pre_ln":
            block_out = x_res1 + ffn_out
            log_stage("Block Output (Residual 2)", block_out, "Final output of Transformer block, passed to next block or LM head.")
        else:
            block_out = norm_fn(x_res1 + ffn_out)
            log_stage("Block Output (Post-Norm)", block_out, "Final normalized output.")

        return {
            "norm_type": norm_type,
            "ffn_type": ffn_type,
            "norm_placement": norm_placement,
            "d_model": d_model,
            "d_ff": d_ff,
            "stages": stages,
        }

    @staticmethod
    def calculate_architecture_params(
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        d_ff: int = 14336,
        tie_word_embeddings: bool = False,
        ffn_type: str = "swiglu",
    ) -> Dict[str, Any]:
        """
        Calculates exact parameter counts and GPU VRAM memory requirements
        for full model architectures across FP16, INT8, and INT4.
        """
        head_dim = d_model // n_heads

        # Embedding params
        emb_params = vocab_size * d_model

        # Attention params per layer
        # W_q: d_model * (n_heads * head_dim) = d_model * d_model
        # W_k: d_model * (n_kv_heads * head_dim)
        # W_v: d_model * (n_kv_heads * head_dim)
        # W_o: (n_heads * head_dim) * d_model = d_model * d_model
        w_q = d_model * d_model
        w_k = d_model * (n_kv_heads * head_dim)
        w_v = d_model * (n_kv_heads * head_dim)
        w_o = d_model * d_model
        attn_per_layer = w_q + w_k + w_v + w_o

        # FFN params per layer
        if ffn_type.lower() == "swiglu":
            # 3 matrices: W_gate, W_up, W_down
            ffn_per_layer = 3 * d_model * d_ff
        else:
            # 2 matrices: W_1, W_2
            ffn_per_layer = 2 * d_model * d_ff

        # Normalization params per layer (2 norms: pre-attn & pre-ffn)
        norm_per_layer = 2 * d_model

        # Total per layer
        layer_params = attn_per_layer + ffn_per_layer + norm_per_layer
        all_layers_params = layer_params * n_layers

        # Final norm
        final_norm_params = d_model

        # Output LM Head
        lm_head_params = 0 if tie_word_embeddings else (vocab_size * d_model)

        total_parameters = emb_params + all_layers_params + final_norm_params + lm_head_params

        # Memory footprint across precisions
        fp16_bytes = total_parameters * 2
        fp32_bytes = total_parameters * 4
        int8_bytes = total_parameters * 1
        int4_bytes = total_parameters * 0.5

        # Training memory rule of thumb (AdamW 32-bit):
        # Model Weights: 2 bytes (FP16) or 4 bytes
        # Gradients: 2 bytes
        # Optimizer States (AdamW m & v in FP32): 8 bytes
        # Total per param: ~12-16 bytes
        training_vram_bytes = total_parameters * 16

        return {
            "total_parameters": total_parameters,
            "total_millions": round(total_parameters / 1e6, 2),
            "total_billions": round(total_parameters / 1e9, 2),
            "breakdown": {
                "token_embeddings": emb_params,
                "attention_all_layers": attn_per_layer * n_layers,
                "attention_per_layer": attn_per_layer,
                "ffn_all_layers": ffn_per_layer * n_layers,
                "ffn_per_layer": ffn_per_layer,
                "norms_all_layers": (norm_per_layer * n_layers) + final_norm_params,
                "lm_head": lm_head_params,
            },
            "percentages": {
                "attention_pct": round((attn_per_layer * n_layers / total_parameters) * 100, 1),
                "ffn_pct": round((ffn_per_layer * n_layers / total_parameters) * 100, 1),
                "embeddings_pct": round(((emb_params + lm_head_params) / total_parameters) * 100, 1),
            },
            "vram_inference": {
                "fp16": format_bytes(int(fp16_bytes)),
                "fp32": format_bytes(int(fp32_bytes)),
                "int8_quantized": format_bytes(int(int8_bytes)),
                "int4_quantized": format_bytes(int(int4_bytes)),
            },
            "vram_training_adamw": format_bytes(int(training_vram_bytes)),
        }

    @staticmethod
    def get_presets() -> List[Dict[str, Any]]:
        """Returns standard open-source LLM architecture configurations."""
        return [
            {
                "name": "LLaMA-3 8B",
                "vocab_size": 128256,
                "d_model": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "n_kv_heads": 8,
                "d_ff": 14336,
                "norm_type": "rmsnorm",
                "ffn_type": "swiglu",
                "tie_word_embeddings": False,
            },
            {
                "name": "Mistral 7B",
                "vocab_size": 32000,
                "d_model": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "n_kv_heads": 8,
                "d_ff": 14336,
                "norm_type": "rmsnorm",
                "ffn_type": "swiglu",
                "tie_word_embeddings": False,
            },
            {
                "name": "TinyLlama 1.1B",
                "vocab_size": 32000,
                "d_model": 2048,
                "n_layers": 22,
                "n_heads": 32,
                "n_kv_heads": 4,
                "d_ff": 5632,
                "norm_type": "rmsnorm",
                "ffn_type": "swiglu",
                "tie_word_embeddings": False,
            },
            {
                "name": "GPT-2 Small (124M)",
                "vocab_size": 50257,
                "d_model": 768,
                "n_layers": 12,
                "n_heads": 12,
                "n_kv_heads": 12,
                "d_ff": 3072,
                "norm_type": "layernorm",
                "ffn_type": "standard_mlp",
                "tie_word_embeddings": True,
            },
        ]
