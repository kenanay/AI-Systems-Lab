"""
backend/routers/systems_lab.py

Systems for AI Lab & Hardware Simulation REST API Endpoints
Sections 28, 29, 30 & 102/103 of Master Architecture Plan
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from src.simulators.systems_hardware import (
    RooflineModelEngine,
    QuantizationSimulator,
    GPUMemorySimulator,
    HardwarePreset,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/systems-lab",
    tags=["systems-lab"]
)


# Request Models

class RooflineAnalyzeRequest(BaseModel):
    """Roofline modeli analiz request."""
    peak_tflops: float = Field(..., gt=0, description="Zirve teorik TFLOPs")
    peak_bandwidth_gbps: float = Field(..., gt=0, description="Zirve bellek bant genişliği (GB/s)")
    total_flops: float = Field(..., gt=0, description="İşlem toplam FLOPs sayısı")
    total_bytes: float = Field(..., gt=0, description="İşlem toplam bellek transfer bayt miktarı")
    hardware_name: Optional[str] = Field("Custom Hardware", description="Donanım adı")


class QuantizationSimulateRequest(BaseModel):
    """Kuantizasyon simülasyonu request."""
    distribution: str = Field("normal", description="Ağırlık dağılımı (normal, uniform)")
    num_elements: int = Field(10000, ge=100, le=100000, description="Simüle edilecek ağırlık sayısı")
    std: float = Field(0.5, gt=0, description="Standart sapma")
    outlier_ratio: float = Field(0.02, ge=0.0, le=0.2, description="Aykırı değer (outlier) oranı")


class GPUMemorySimulateRequest(BaseModel):
    """GPU VRAM bellek ayak izi simülasyonu request."""
    mode: str = Field("training", description="Çalışma modu: 'training' veya 'inference'")
    param_count_billions: float = Field(7.0, gt=0, description="Model parametre sayısı (milyar cinsinden, örn 7.0)")
    precision: str = Field("fp16", description="Ağırlık hassasiyeti (fp32, fp16, bf16, int8, int4)")
    batch_size: int = Field(4, ge=1, le=1024, description="Batch boyutu")
    seq_len: int = Field(2048, ge=16, le=131072, description="Dizi uzunluğu (Sequence length)")
    hidden_size: int = Field(4096, ge=64, description="Hidden state boyutu (d_model)")
    num_layers: int = Field(32, ge=1, description="Transformer katman sayısı")
    num_heads: int = Field(32, ge=1, description="Query dikkat kafa sayısı")
    num_kv_heads: int = Field(8, ge=1, description="Key/Value dikkat kafa sayısı (GQA)")
    optimizer_type: str = Field("adamw", description="Optimizer türü (adamw, adamw_8bit, sgd)")
    activation_checkpointing: bool = Field(False, description="Gradient/Activation checkpointing aktif mi?")
    gpu_capacity_gb: float = Field(24.0, gt=0, description="Hedef GPU VRAM boyutu (GB, örn 24.0)")


# Endpoints

@router.get("/roofline/presets", response_model=List[HardwarePreset])
async def get_hardware_presets() -> List[HardwarePreset]:
    """
    Ön tanımlı popüler AI donanım profillerini (H100, RTX 4090, Apple Silicon, vb.) döner.
    """
    logger.info("Fetching hardware presets")
    return RooflineModelEngine.get_presets()


@router.post("/roofline/analyze")
async def analyze_roofline(request: RooflineAnalyzeRequest) -> Dict[str, Any]:
    """
    İş yükü aritmetik yoğunluğu ($I$) ve donanım sınırları doğrultusunda
    Roofline eğrisini, kırılma noktasını ve Memory/Compute Bound teşhisini hesaplar.
    """
    logger.info(f"Analyzing roofline for {request.hardware_name}")
    try:
        return RooflineModelEngine.analyze(
            peak_tflops=request.peak_tflops,
            peak_bandwidth_gbps=request.peak_bandwidth_gbps,
            total_flops=request.total_flops,
            total_bytes=request.total_bytes,
            hardware_name=request.hardware_name or "Custom Hardware"
        )
    except Exception as e:
        logger.error(f"Roofline analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Roofline analizi başarısız oldu: {str(e)}")


@router.post("/quantization/simulate")
async def simulate_quantization(request: QuantizationSimulateRequest) -> Dict[str, Any]:
    """
    Model ağırlıklarını FP32'den FP16, INT8 ve INT4 hassasiyetlerine kuantize eder;
    MSE, MAE, SNR (dB) gürültü metriklerini ve histogram dağılımını döner.
    """
    logger.info(f"Simulating quantization: {request.distribution} with {request.num_elements} elements")
    try:
        return QuantizationSimulator.simulate(
            distribution=request.distribution,
            num_elements=request.num_elements,
            std=request.std,
            outlier_ratio=request.outlier_ratio
        )
    except Exception as e:
        logger.error(f"Quantization simulation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Kuantizasyon simülasyonu başarısız: {str(e)}")


@router.post("/memory/simulate")
async def simulate_gpu_memory(request: GPUMemorySimulateRequest) -> Dict[str, Any]:
    """
    Eğitim veya çıkarım senaryosunda model ağırlıkları, gradyanlar,
    optimizer durumları, aktivasyonlar ve KV-Cache VRAM bellek ayak izini hesaplar.
    """
    logger.info(f"Simulating GPU memory: {request.param_count_billions}B in {request.mode} mode")
    try:
        return GPUMemorySimulator.simulate(
            mode=request.mode,
            param_count_billions=request.param_count_billions,
            precision=request.precision,
            batch_size=request.batch_size,
            seq_len=request.seq_len,
            hidden_size=request.hidden_size,
            num_layers=request.num_layers,
            num_heads=request.num_heads,
            num_kv_heads=request.num_kv_heads,
            optimizer_type=request.optimizer_type,
            activation_checkpointing=request.activation_checkpointing,
            gpu_capacity_gb=request.gpu_capacity_gb
        )
    except Exception as e:
        logger.error(f"Memory simulation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bellek simülasyonu başarısız: {str(e)}")
