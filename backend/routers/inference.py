"""
backend/routers/inference.py

Model Inference API Endpoints

Bu modül eğitilmiş modeller ile metin üretimi (Text Generation) ve
çıkarım (Inference) için REST ve Streaming API sağlar:
- Model yükleme/boşaltma
- Durum kontrolü (Model yüklü mü?)
- Metin tamamlama / üretim (Greedy, Top-k, Top-p, Temperature)
- Streaming metin üretimi
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import json
import time

from src.server.inference_server import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/inference",
    tags=["inference"]
)

# Global ModelManager singleton
manager = ModelManager()


class LoadModelRequest(BaseModel):
    model_name: str = Field(..., description="Yüklenecek model adı")
    version: Optional[str] = Field(None, description="Versiyon (varsayılan: en son)")
    registry_dir: str = Field("models", description="Registry dizini")


class InferenceGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Giriş metni / prompt")
    max_length: int = Field(50, ge=1, le=512, description="Üretilecek maksimum token sayısı")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Örnekleme sıcaklığı")
    top_k: Optional[int] = Field(50, ge=1, description="Top-k örnekleme")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Nucleus (top-p) örnekleme")


class InferenceGenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    tokens_generated: int
    generation_time_ms: float
    model_name: str


from typing import Optional, Dict, Any, Generator

@router.post("/load")
def load_model(request: LoadModelRequest) -> Dict[str, Any]:
    """
    Belirtilen modeli belleğe yükler.
    """
    try:
        manager.load_model(
            registry_dir=request.registry_dir,
            model_name=request.model_name,
            version=request.version
        )
        return {
            "status": "success",
            "message": f"Model {request.model_name} başarıyla yüklendi",
            "model_name": manager.model_name
        }
    except Exception as e:
        logger.error(f"Failed to load model {request.model_name}: {e}")
        raise HTTPException(status_code=400, detail=f"Model yüklenemedi: {str(e)}")


@router.get("/status")
def get_inference_status() -> Dict[str, Any]:
    """
    Çıkarım motorunun ve aktif modelin durumunu döndürür.
    """
    is_loaded = manager.model is not None
    return {
        "ready": is_loaded,
        "model_loaded": is_loaded,
        "model_name": manager.model_name,
        "device": str(manager.device)
    }


@router.post("/generate", response_model=InferenceGenerateResponse)
def generate_text(request: InferenceGenerateRequest) -> InferenceGenerateResponse:
    """
    Yüklü model ile metin üretir.
    """
    if manager.model is None:
        raise HTTPException(
            status_code=400,
            detail="Henüz hiçbir model belleğe yüklenmedi. Önce /load ile bir model yükleyin."
        )

    try:
        generated_text, tokens_generated, time_ms = manager.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )

        return InferenceGenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            tokens_generated=tokens_generated,
            generation_time_ms=time_ms,
            model_name=manager.model_name or "unknown"
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Metin üretimi hatası: {str(e)}")


@router.post("/generate/stream")
def generate_stream(request: InferenceGenerateRequest) -> StreamingResponse:
    """
    Metin üretimini SSE (Server-Sent Events) akışı olarak iletir.
    """
    if manager.model is None:
        raise HTTPException(status_code=400, detail="Model yüklü değil")

    def event_generator() -> Generator[str, None, None]:
        try:
            # Fallback stream: generate and stream tokens
            full_text, tokens_count, time_ms = manager.generate(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p
            )
            # Yield in chunks
            words = full_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'token': chunk, 'done': False})}\n\n"
                time.sleep(0.04)

            yield f"data: {json.dumps({'done': True, 'tokens_generated': tokens_count, 'time_ms': time_ms})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

