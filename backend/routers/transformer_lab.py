"""
backend/routers/transformer_lab.py

Positional Encoding & Transformer Architecture Lab API Endpoints
Local AI Research Lab - Developed by Kenan AY

Provides interactive REST endpoints for:
- Sinusoidal, RoPE (Rotary Position Embedding), and ALiBi simulations
- Multi-Head (MHA), Grouped-Query (GQA), and Multi-Query (MQA) KV-Cache analysis
- Full Transformer Block forward pass simulation (Pre-LN vs Post-LN, RMSNorm, SwiGLU)
- Full model parameter & VRAM footprint calculations across FP16/INT8/INT4
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import logging

from src.simulators.transformer_architecture import (
    PositionalEncodingEngine,
    AttentionVariantsEngine,
    TransformerBlockEngine,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/transformer-lab",
    tags=["transformer-lab"]
)


# ============================================================================
# Request / Response Schemas
# ============================================================================

class SinusoidalRequest(BaseModel):
    seq_len: int = Field(16, ge=2, le=128, description="Dizi uzunluğu")
    d_model: int = Field(32, ge=4, le=512, description="Model boyutu (çift sayı olmalı)")


class RoPERequest(BaseModel):
    dim: int = Field(16, ge=2, le=128, description="Döndürülecek başlık boyutu")
    max_seq_len: int = Field(8, ge=2, le=32, description="Görselleştirilecek pozisyon sayısı")
    base: float = Field(10000.0, ge=100.0, le=1000000.0, description="Frekans tabanı theta")


class ALiBiRequest(BaseModel):
    num_heads: int = Field(8, ge=1, le=32, description="Dikkat başlığı sayısı")
    seq_len: int = Field(8, ge=2, le=32, description="Dizi uzunluğu")


class KVCacheAnalysisRequest(BaseModel):
    batch_size: int = Field(1, ge=1, le=256)
    seq_len: int = Field(2048, ge=1, le=131072)
    num_query_heads: int = Field(32, ge=1, le=128)
    num_kv_heads: int = Field(8, ge=1, le=128)
    head_dim: int = Field(128, ge=16, le=256)
    num_layers: int = Field(32, ge=1, le=128)
    dtype: str = Field("float16", description="float16, bfloat16, float32, int8")


class BlockSimulateRequest(BaseModel):
    norm_type: str = Field("rmsnorm", description="rmsnorm veya layernorm")
    ffn_type: str = Field("swiglu", description="swiglu veya standard_mlp")
    norm_placement: str = Field("pre_ln", description="pre_ln veya post_ln")
    batch_size: int = Field(1, ge=1, le=8)
    seq_len: int = Field(4, ge=1, le=32)
    d_model: int = Field(8, ge=4, le=128)
    d_ff: int = Field(16, ge=8, le=256)


class ModelParamsRequest(BaseModel):
    vocab_size: int = Field(32000, ge=100)
    d_model: int = Field(4096, ge=64)
    n_layers: int = Field(32, ge=1)
    n_heads: int = Field(32, ge=1)
    n_kv_heads: int = Field(8, ge=1)
    d_ff: int = Field(14336, ge=64)
    tie_word_embeddings: bool = Field(False)
    ffn_type: str = Field("swiglu", description="swiglu veya standard_mlp")


# ============================================================================
# Positional Encoding Endpoints
# ============================================================================

@router.post("/positional-encoding/sinusoidal")
async def get_sinusoidal_pe(request: SinusoidalRequest) -> Dict[str, Any]:
    """Sinusoidal pozisyonel kodlama matrisini, frekans dalgalarını ve benzerlik matrisini döner."""
    try:
        return PositionalEncodingEngine.compute_sinusoidal(
            seq_len=request.seq_len,
            d_model=request.d_model,
        )
    except Exception as e:
        logger.error(f"Sinusoidal PE error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/positional-encoding/rope")
async def get_rope_simulation(request: RoPERequest) -> Dict[str, Any]:
    """Rotary Position Embedding (RoPE) 2D vektör rotasyonlarını ve göreli mesafe sönümlenmesini simüle eder."""
    try:
        return PositionalEncodingEngine.compute_rope(
            dim=request.dim,
            max_seq_len=request.max_seq_len,
            base=request.base,
        )
    except Exception as e:
        logger.error(f"RoPE error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/positional-encoding/alibi")
async def get_alibi_simulation(request: ALiBiRequest) -> Dict[str, Any]:
    """ALiBi (Attention with Linear Biases) başlık eğimlerini ve mesafe ceza matrisini hesaplar."""
    try:
        return PositionalEncodingEngine.compute_alibi(
            num_heads=request.num_heads,
            seq_len=request.seq_len,
        )
    except Exception as e:
        logger.error(f"ALiBi error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positional-encoding/compare")
async def compare_positional_methods() -> List[Dict[str, Any]]:
    """Tüm modern pozisyonel kodlama yöntemlerinin pedagojik karşılaştırma tablosunu döner."""
    return PositionalEncodingEngine.compare_methods()


# ============================================================================
# Attention Variants & KV-Cache Endpoints
# ============================================================================

@router.post("/attention-variants/analyze")
async def analyze_attention_variants(request: KVCacheAnalysisRequest) -> Dict[str, Any]:
    """MHA, GQA ve MQA mimarileri için KV-Cache bellek ayak izini ve GPU bellek bant genişliğini analiz eder."""
    try:
        return AttentionVariantsEngine.analyze_kv_cache(
            batch_size=request.batch_size,
            seq_len=request.seq_len,
            num_query_heads=request.num_query_heads,
            num_kv_heads=request.num_kv_heads,
            head_dim=request.head_dim,
            num_layers=request.num_layers,
            dtype=request.dtype,
        )
    except Exception as e:
        logger.error(f"Attention variants analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Transformer Block & Parameter Calculator Endpoints
# ============================================================================

@router.post("/transformer-block/simulate")
async def simulate_transformer_block(request: BlockSimulateRequest) -> Dict[str, Any]:
    """
    Tek bir Transformer Bloğunda Pre-LN/Post-LN, RMSNorm/LayerNorm ve SwiGLU/MLP
    alt katmanlarının ileri besleme akışını simüle eder.
    """
    try:
        return TransformerBlockEngine.simulate_forward_pass(
            norm_type=request.norm_type,
            ffn_type=request.ffn_type,
            norm_placement=request.norm_placement,
            batch_size=request.batch_size,
            seq_len=request.seq_len,
            d_model=request.d_model,
            d_ff=request.d_ff,
        )
    except Exception as e:
        logger.error(f"Transformer block simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transformer-block/params")
async def calculate_model_parameters(request: ModelParamsRequest) -> Dict[str, Any]:
    """Belirtilen mimari için detaylı parametre dökümünü ve FP16/INT8/INT4 çıkarım VRAM gereksinimini hesaplar."""
    try:
        return TransformerBlockEngine.calculate_architecture_params(
            vocab_size=request.vocab_size,
            d_model=request.d_model,
            n_layers=request.n_layers,
            n_heads=request.n_heads,
            n_kv_heads=request.n_kv_heads,
            d_ff=request.d_ff,
            tie_word_embeddings=request.tie_word_embeddings,
            ffn_type=request.ffn_type,
        )
    except Exception as e:
        logger.error(f"Parameter calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transformer-block/presets")
async def get_architecture_presets() -> List[Dict[str, Any]]:
    """Popüler açık kaynak LLM mimarilerini (LLaMA-3, Mistral, TinyLlama, GPT-2) döner."""
    return TransformerBlockEngine.get_presets()
