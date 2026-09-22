"""
src/simulators/systems_hardware.py

Systems for AI & Hardware Simulation Engine
Sections 28, 29, 30 & 102/103 of Master Architecture Plan

Provides:
- Roofline Model & Memory Wall Analyzer (Arithmetic Intensity, Compute vs Memory Bound)
- Quantization Error Simulator (FP32 -> FP16 -> INT8 -> INT4, MSE, MAE, SNR dB, Histograms)
- GPU VRAM & Memory Decomposition Simulator (Weights, Gradients, Optimizer States, KV-Cache, Activations, OOM Radar)
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# 1. Roofline Model Engine
# -----------------------------------------------------------------------------

class HardwarePreset(BaseModel):
    """Donanım profili ön tanımı."""
    id: str
    name: str
    architecture: str
    peak_tflops: float = Field(..., description="Zirve teorik hesaplama gücü (TFLOPs FP16/BF16 Tensor)")
    peak_bandwidth_gbps: float = Field(..., description="Zirve bellek bant genişliği (GB/s)")
    memory_type: str
    typical_vram_gb: float
    description: str


HARDWARE_PRESETS: List[HardwarePreset] = [
    HardwarePreset(
        id="h100_sxm",
        name="NVIDIA H100 SXM5",
        architecture="Hopper",
        peak_tflops=1000.0,
        peak_bandwidth_gbps=3350.0,
        memory_type="HBM3",
        typical_vram_gb=80.0,
        description="Endüstri standardı veri merkezi AI GPU'su. Devasa HBM3 bant genişliği ve dördüncü nesil Tensor Core'lar."
    ),
    HardwarePreset(
        id="a100_sxm",
        name="NVIDIA A100 SXM4",
        architecture="Ampere",
        peak_tflops=312.0,
        peak_bandwidth_gbps=2039.0,
        memory_type="HBM2e",
        typical_vram_gb=80.0,
        description="Yaygın bulut eğitimi ve çıkarımı hızlandırıcısı."
    ),
    HardwarePreset(
        id="rtx_4090",
        name="NVIDIA GeForce RTX 4090",
        architecture="Ada Lovelace",
        peak_tflops=330.0,
        peak_bandwidth_gbps=1008.0,
        memory_type="GDDR6X",
        typical_vram_gb=24.0,
        description="Tüketici sınıfı amiral gemisi. Yüksek TFLOPs ancak HBM yerine GDDR6X bellek bant genişliği sınırına sahiptir."
    ),
    HardwarePreset(
        id="apple_m3_max",
        name="Apple M3 Max (Unified)",
        architecture="Apple Silicon",
        peak_tflops=25.0,
        peak_bandwidth_gbps=400.0,
        memory_type="LPDDR5 Unified",
        typical_vram_gb=128.0,
        description="CPU ve GPU tarafından paylaşılan devasa birleşik bellek mimarisi."
    ),
    HardwarePreset(
        id="server_cpu",
        name="Server CPU (x86_64 Dual Socket)",
        architecture="Intel Xeon / AMD EPYC",
        peak_tflops=3.5,
        peak_bandwidth_gbps=120.0,
        memory_type="DDR5",
        typical_vram_gb=256.0,
        description="Geleneksel x86_64 sunucu işlemcisi. Düşük TFLOPs ve DDR5 bellek bant genişliği."
    )
]


class RooflineModelEngine:
    """
    Roofline Modeli ve Memory Wall Analizörü.
    Hesaplama ile bellek transferi arasındaki dengeyi ve darboğazı ölçer.
    """

    @staticmethod
    def get_presets() -> List[HardwarePreset]:
        return HARDWARE_PRESETS

    @staticmethod
    def analyze(
        peak_tflops: float,
        peak_bandwidth_gbps: float,
        total_flops: float,
        total_bytes: float,
        hardware_name: str = "Custom Hardware"
    ) -> Dict[str, Any]:
        """
        Verilen donanım ve iş yükü için Roofline analizi gerçekleştirir.
        """
        if peak_bandwidth_gbps <= 0:
            peak_bandwidth_gbps = 1.0
        if peak_tflops <= 0:
            peak_tflops = 1.0
        if total_bytes <= 0:
            total_bytes = 1.0
        if total_flops <= 0:
            total_flops = 1.0

        # Operational Intensity (FLOPs / Byte)
        operational_intensity = total_flops / total_bytes

        # Machine Balance / Knee Point (FLOPs / Byte)
        # 1 TFLOP = 10^12 FLOPs, 1 GB/s = 10^9 Bytes/s -> (P_peak * 10^12) / (B_peak * 10^9) = (P_peak / B_peak) * 1000
        knee_point = (peak_tflops * 1000.0) / peak_bandwidth_gbps

        # Attainable Performance in TFLOPs
        # P = min(P_peak, Intensity * Bandwidth / 1000)
        bandwidth_limit_tflops = (operational_intensity * peak_bandwidth_gbps) / 1000.0
        attainable_tflops = min(peak_tflops, bandwidth_limit_tflops)

        # Efficiency
        efficiency_pct = (attainable_tflops / peak_tflops) * 100.0

        # Bound Classification
        is_memory_bound = operational_intensity < knee_point
        bound_type = "Memory-Bound (Bellek Darboğazı - Memory Wall)" if is_memory_bound else "Compute-Bound (Hesaplama Darboğazı)"

        explanation = (
            f"İş yükünün aritmetik yoğunluğu {operational_intensity:.2f} FLOPs/Byte olarak ölçüldü. "
            f"Donanımın kırılma noktası (Machine Balance) {knee_point:.2f} FLOPs/Byte'tır. "
            f"İşlem {bound_type} bölgesindedir. "
        )
        if is_memory_bound:
            explanation += (
                "GPU hesaplama birimleri (Tensor Core'lar) boşta kalmakta ve bellekten (VRAM/HBM) "
                "veri transferinin tamamlanmasını beklemektedir. LLM otoregresif tek-token üretimi tipik olarak bu darboğaza takılır."
            )
        else:
            explanation += (
                "Bellek bant genişliği fazlasıyla yeterlidir; performans doğrudan GPU ALU / Tensor Core "
                "sayısı ve saat frekansı ile sınırlanmaktadır. Büyük batch matris çarpımları bu bölgede çalışır."
            )

        # Generate roofline curve points for plotting (log-spaced intensity from 0.01 to 10000)
        curve_points: List[Dict[str, float]] = []
        intensities = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, knee_point, 100.0, 200.0, 500.0, 1000.0, 5000.0]
        intensities = sorted(list(set(intensities)))

        for i in intensities:
            p = min(peak_tflops, (i * peak_bandwidth_gbps) / 1000.0)
            curve_points.append({"intensity": round(i, 4), "performance_tflops": round(p, 4)})

        return {
            "hardware_name": hardware_name,
            "peak_tflops": peak_tflops,
            "peak_bandwidth_gbps": peak_bandwidth_gbps,
            "operational_intensity": round(operational_intensity, 4),
            "knee_point": round(knee_point, 4),
            "attainable_tflops": round(attainable_tflops, 4),
            "efficiency_pct": round(efficiency_pct, 2),
            "is_memory_bound": is_memory_bound,
            "bound_type": bound_type,
            "explanation": explanation,
            "curve_points": curve_points,
            "operating_point": {
                "intensity": round(operational_intensity, 4),
                "performance_tflops": round(attainable_tflops, 4)
            }
        }


# -----------------------------------------------------------------------------
# 2. Quantization Simulator
# -----------------------------------------------------------------------------

class QuantizationSimulator:
    """
    Model ağırlıklarının FP32'den FP16, INT8 ve INT4'e indirgenmesini
    ve ortaya çıkan kuantizasyon gürültüsünü (MSE, MAE, SNR dB) simüle eder.
    """

    @staticmethod
    def generate_weights(
        distribution: str = "normal",
        num_elements: int = 10000,
        std: float = 0.5,
        outlier_ratio: float = 0.02
    ) -> np.ndarray:
        """Gerçekçi model ağırlık tensörü üretir."""
        np.random.seed(42)
        if distribution == "normal":
            weights = np.random.normal(0.0, std, size=num_elements).astype(np.float32)
        elif distribution == "uniform":
            weights = np.random.uniform(-std * 2, std * 2, size=num_elements).astype(np.float32)
        else:
            weights = np.random.normal(0.0, std, size=num_elements).astype(np.float32)

        # Enjekte edilmiş LLM aktivasyon aykırı değerleri (Outliers e.g. LLM.int8())
        if outlier_ratio > 0:
            num_outliers = int(num_elements * outlier_ratio)
            outlier_indices = np.random.choice(num_elements, size=num_outliers, replace=False)
            weights[outlier_indices] *= 6.0

        return weights

    @classmethod
    def simulate(
        cls,
        weights: Optional[np.ndarray] = None,
        distribution: str = "normal",
        num_elements: int = 10000,
        std: float = 0.5,
        outlier_ratio: float = 0.02
    ) -> Dict[str, Any]:
        """
        FP32, FP16, INT8 ve INT4 kuantizasyonunu gerçekleştirir ve metrikleri hesaplar.
        """
        if weights is None:
            weights = cls.generate_weights(distribution, num_elements, std, outlier_ratio)

        fp32_weights = weights.astype(np.float32)
        w_min = float(np.min(fp32_weights))
        w_max = float(np.max(fp32_weights))

        # 1. FP16 Simulation
        fp16_weights = fp32_weights.astype(np.float16).astype(np.float32)

        # 2. INT8 Affine Quantization (Asymmetric 0 to 255)
        # S = (max - min) / 255
        scale_8 = max(1e-8, (w_max - w_min) / 255.0)
        zero_point_8 = np.clip(np.round(-w_min / scale_8), 0, 255)
        q_int8 = np.clip(np.round(fp32_weights / scale_8) + zero_point_8, 0, 255)
        dequant_int8 = (q_int8 - zero_point_8) * scale_8

        # 3. INT4 Quantization (Asymmetric 0 to 15)
        scale_4 = max(1e-8, (w_max - w_min) / 15.0)
        zero_point_4 = np.clip(np.round(-w_min / scale_4), 0, 15)
        q_int4 = np.clip(np.round(fp32_weights / scale_4) + zero_point_4, 0, 15)
        dequant_int4 = (q_int4 - zero_point_4) * scale_4

        # Calculate metrics for each precision
        def get_metrics(original: np.ndarray, dequant: np.ndarray, bits_per_weight: int) -> Dict[str, Any]:
            mse = float(np.mean((original - dequant) ** 2))
            mae = float(np.mean(np.abs(original - dequant)))
            signal_power = float(np.mean(original ** 2))
            noise_power = max(1e-12, mse)
            snr_db = float(10.0 * np.log10(signal_power / noise_power))
            memory_bytes = len(original) * (bits_per_weight / 8.0)

            return {
                "bits": bits_per_weight,
                "memory_kb": round(memory_bytes / 1024.0, 2),
                "compression_ratio": round(32.0 / bits_per_weight, 2),
                "mse": round(mse, 6),
                "mae": round(mae, 6),
                "snr_db": round(snr_db, 2),
            }

        fp32_metrics = get_metrics(fp32_weights, fp32_weights, 32)
        fp16_metrics = get_metrics(fp32_weights, fp16_weights, 16)
        int8_metrics = get_metrics(fp32_weights, dequant_int8, 8)
        int4_metrics = get_metrics(fp32_weights, dequant_int4, 4)

        # Histograms for visualization (20 bins)
        bins = np.linspace(w_min, w_max, 21)
        orig_hist, _ = np.histogram(fp32_weights, bins=bins)
        int8_hist, _ = np.histogram(dequant_int8, bins=bins)
        int4_hist, _ = np.histogram(dequant_int4, bins=bins)

        bin_centers = [round(float((bins[i] + bins[i+1]) / 2.0), 3) for i in range(len(bins) - 1)]

        return {
            "num_elements": len(fp32_weights),
            "min_val": round(w_min, 4),
            "max_val": round(w_max, 4),
            "scale_int8": round(scale_8, 6),
            "zero_point_int8": int(zero_point_8),
            "scale_int4": round(scale_4, 6),
            "zero_point_int4": int(zero_point_4),
            "metrics": {
                "fp32": fp32_metrics,
                "fp16": fp16_metrics,
                "int8": int8_metrics,
                "int4": int4_metrics
            },
            "histogram": {
                "bin_centers": bin_centers,
                "fp32_counts": orig_hist.tolist(),
                "int8_counts": int8_hist.tolist(),
                "int4_counts": int4_hist.tolist()
            }
        }


# -----------------------------------------------------------------------------
# 3. GPU VRAM Memory Decomposition Simulator
# -----------------------------------------------------------------------------

class GPUMemorySimulator:
    """
    Eğitim ve Çıkarım iş yüklerinde GPU VRAM bellek ayak izini
    ayrıntılı olarak parçalara (Ağırlıklar, Gradyanlar, Optimizer, Aktivasyonlar, KV-Cache) böler.
    """

    @staticmethod
    def simulate(
        mode: str = "training",  # "training" or "inference"
        param_count_billions: float = 7.0,  # e.g. 7B
        precision: str = "fp16",  # "fp32", "fp16", "int8", "int4"
        batch_size: int = 4,
        seq_len: int = 2048,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,  # GQA
        optimizer_type: str = "adamw",  # "adamw", "adamw_8bit", "sgd"
        activation_checkpointing: bool = False,
        gpu_capacity_gb: float = 24.0  # e.g. RTX 4090 = 24GB, H100 = 80GB
    ) -> Dict[str, Any]:
        """
        Detaylı VRAM bellek dökümü ve OOM riski analizi.
        """
        # Bytes per parameter based on precision
        precision_bytes_map = {
            "fp32": 4.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "int8": 1.0,
            "int4": 0.5
        }
        param_bytes = precision_bytes_map.get(precision.lower(), 2.0)
        total_params = param_count_billions * 1e9

        # 1. Model Weights Memory (GB)
        weights_gb = (total_params * param_bytes) / (1024.0 ** 3)

        gradients_gb = 0.0
        optimizer_gb = 0.0
        activation_gb = 0.0
        kv_cache_gb = 0.0
        cuda_overhead_gb = 0.8  # ~800MB CUDA context + runtime memory

        if mode == "training":
            # Gradients (typically FP16 or FP32)
            grad_bytes = 2.0 if precision in ["fp16", "bf16", "int8", "int4"] else 4.0
            gradients_gb = (total_params * grad_bytes) / (1024.0 ** 3)

            # Optimizer States
            # AdamW: 8 bytes per param (FP32 1st moment + 2nd moment) + 4 bytes FP32 master weights if mixed precision
            if optimizer_type == "adamw":
                optimizer_bytes = 12.0 if precision in ["fp16", "bf16"] else 8.0
            elif optimizer_type == "adamw_8bit":
                optimizer_bytes = 2.0 + (4.0 if precision in ["fp16", "bf16"] else 0.0)
            elif optimizer_type == "sgd":
                optimizer_bytes = 4.0  # Momentum buffer
            else:
                optimizer_bytes = 8.0

            optimizer_gb = (total_params * optimizer_bytes) / (1024.0 ** 3)

            # Activations Memory
            # Standard Transformer block activation formula:
            # ~ (34 * B * S * H * L) bytes without recompute
            # With activation checkpointing (gradient checkpointing): ~ (2 * B * S * H * L) bytes
            act_multiplier = 2.5 if activation_checkpointing else 34.0
            act_bytes = act_multiplier * batch_size * seq_len * hidden_size * num_layers
            activation_gb = act_bytes / (1024.0 ** 3)

        else:
            # Inference Mode
            # KV-Cache Memory:
            # 2 * B * S * H_kv * d_head * L * dtype_bytes
            d_head = hidden_size // max(1, num_heads)
            cache_bytes = 2 * batch_size * seq_len * num_kv_heads * d_head * num_layers * 2.0  # FP16 KV
            kv_cache_gb = cache_bytes / (1024.0 ** 3)

            # Small transient activation buffer during inference
            activation_gb = (batch_size * seq_len * hidden_size * 4.0) / (1024.0 ** 3)

        total_vram_gb = weights_gb + gradients_gb + optimizer_gb + activation_gb + kv_cache_gb + cuda_overhead_gb
        oom_risk = total_vram_gb > gpu_capacity_gb
        headroom_gb = max(0.0, gpu_capacity_gb - total_vram_gb)

        return {
            "mode": mode,
            "param_count_billions": param_count_billions,
            "precision": precision,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "gpu_capacity_gb": gpu_capacity_gb,
            "breakdown_gb": {
                "weights": round(weights_gb, 2),
                "gradients": round(gradients_gb, 2),
                "optimizer": round(optimizer_gb, 2),
                "activations": round(activation_gb, 2),
                "kv_cache": round(kv_cache_gb, 2),
                "cuda_overhead": round(cuda_overhead_gb, 2),
                "total": round(total_vram_gb, 2)
            },
            "percentages": {
                "weights": round((weights_gb / max(1e-6, total_vram_gb)) * 100.0, 1),
                "gradients": round((gradients_gb / max(1e-6, total_vram_gb)) * 100.0, 1),
                "optimizer": round((optimizer_gb / max(1e-6, total_vram_gb)) * 100.0, 1),
                "activations": round((activation_gb / max(1e-6, total_vram_gb)) * 100.0, 1),
                "kv_cache": round((kv_cache_gb / max(1e-6, total_vram_gb)) * 100.0, 1),
                "cuda_overhead": round((cuda_overhead_gb / max(1e-6, total_vram_gb)) * 100.0, 1)
            },
            "oom_risk": oom_risk,
            "headroom_gb": round(headroom_gb, 2),
            "status": "OOM (Out Of Memory!)" if oom_risk else "FIT (Bellek Yeterli)"
        }
