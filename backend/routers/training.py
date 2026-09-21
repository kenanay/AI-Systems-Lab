"""
backend/routers/training.py

Model Training API Endpoints

Bu modül dil modeli eğitimi için REST API sağlar:
- Eğitim işi oluşturma ve başlatma (Pretraining, SFT, LoRA)
- Canlı metrik ve ilerleme sorgulama
- Eğitim işlerini listeleme ve iptal etme
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

from backend.database import get_db
from backend.models import TrainingJob
from backend.services.training_service import TrainingService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/training",
    tags=["training"]
)


# ============================================================================
# Request / Response Schemas
# ============================================================================

class StartTrainingRequest(BaseModel):
    job_name: str = Field(..., description="Eğitim işi adı")
    model_name: str = Field(..., description="Eğitilecek model adı (örn. gpt-turkish-tiny)")
    job_type: str = Field("PRETRAIN", description="Eğitim tipi: PRETRAIN, SFT, SFT_LORA")
    dataset_id: Optional[str] = Field(None, description="Compiled Dataset ID")
    tokenizer_id: Optional[str] = Field(None, description="Tokenizer ID")
    epochs: int = Field(3, ge=1, le=50, description="Epoch sayısı")
    batch_size: int = Field(4, ge=1, le=128, description="Batch boyutu")
    learning_rate: float = Field(1e-3, ge=1e-6, le=1e-1, description="Öğrenme oranı")
    d_model: int = Field(128, description="Embedding boyutu")
    n_layers: int = Field(4, description="Transformer katman sayısı")
    n_heads: int = Field(4, description="Attention kafa sayısı")
    max_seq_len: int = Field(128, description="Maksimum sequence uzunluğu")
    lora_r: Optional[int] = Field(8, description="LoRA rank")
    lora_alpha: Optional[int] = Field(16, description="LoRA alpha")


class MetricPoint(BaseModel):
    step: int
    epoch: int
    loss: float
    perplexity: Optional[float] = None
    lr: Optional[float] = None


class TrainingJobResponse(BaseModel):
    job_id: str
    job_name: str
    job_type: str
    status: str
    model_name: str
    dataset_id: Optional[str] = None
    tokenizer_id: Optional[str] = None
    progress: float
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    metrics: List[Dict[str, Any]] = []


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/start", response_model=TrainingJobResponse)
def start_training(
    request: StartTrainingRequest,
    db: Session = Depends(get_db)
):
    """
    Yeni bir model eğitimi başlatır.
    """
    service = TrainingService(db)
    config = {
        "epochs": request.epochs,
        "batch_size": request.batch_size,
        "lr": request.learning_rate,
        "d_model": request.d_model,
        "n_layers": request.n_layers,
        "n_heads": request.n_heads,
        "max_seq_len": request.max_seq_len,
        "lora_r": request.lora_r,
        "lora_alpha": request.lora_alpha,
    }

    job = service.create_job(
        job_name=request.job_name,
        model_name=request.model_name,
        job_type=request.job_type,
        dataset_id=request.dataset_id,
        tokenizer_id=request.tokenizer_id,
        config=config
    )

    # Arka planda eğitimi başlat
    service.start_training(str(job.job_id))

    return TrainingJobResponse(**job.to_dict())


@router.get("/jobs", response_model=List[TrainingJobResponse])
def list_training_jobs(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
) -> List[TrainingJobResponse]:
    """
    Tüm eğitim işlerini listeler.
    """
    jobs = db.query(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit).all()
    return [TrainingJobResponse(**job.to_dict()) for job in jobs]


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(
    job_id: str,
    db: Session = Depends(get_db)
) -> TrainingJobResponse:
    """
    Belirli bir eğitim işinin durumunu ve canlı metriklerini getirir.
    """
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Eğitim işi bulunamadı")

    return TrainingJobResponse(**job.to_dict())


@router.post("/jobs/{job_id}/cancel")
def cancel_training_job(
    job_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Eğitim işini durdurur.
    """
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Eğitim işi bulunamadı")

    service = TrainingService(db)
    cancelled = service.cancel_job(job_id)
    if cancelled or job.status == "RUNNING":
        job.status = "CANCELLED"
        db.commit()
        return {"status": "success", "message": f"İş {job_id} iptal edildi"}
    
    return {"status": "ok", "message": f"İş durumu: {job.status}"}
