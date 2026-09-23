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
from pydantic import BaseModel, Field, model_validator
import logging

from backend.database import get_db
from backend.models import TrainingJob, UserRecord
from backend.services.training_service import TrainingService
from backend.security.dependencies import get_current_user, require_role

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/training",
    tags=["training"]
)


# ============================================================================
# Request / Response Schemas
# ============================================================================

class StartTrainingRequest(BaseModel):
    base_model: Optional[str] = None
    base_version: Optional[str] = None
    device: str = "cpu"
    seed: int = 42
    version: Optional[str] = None

    @model_validator(mode="after")
    def validate_config(self):
        if self.job_type not in {"PRETRAIN", "FULL_SFT", "LORA_SFT", "SFT", "SFT_LORA"}:
            raise ValueError("Unsupported training type")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not self.dataset_id or not self.tokenizer_id:
            raise ValueError("Dataset and tokenizer are required")
        if self.job_type != "PRETRAIN" and (not self.base_model or not self.base_version):
            raise ValueError("Fine-tuning requires base_model and base_version")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("Unsupported device")
        return self

    job_name: str = Field(..., description="Eğitim işi adı")
    model_name: str = Field(..., description="Eğitilecek model adı (örn. gpt-turkish-tiny)")
    job_type: str = Field("PRETRAIN", description="Eğitim tipi: PRETRAIN, SFT, SFT_LORA")
    dataset_id: Optional[str] = Field(None, description="Compiled Dataset ID")
    tokenizer_id: Optional[str] = Field(None, description="Tokenizer ID")
    epochs: int = Field(3, ge=1, le=50, description="Epoch sayısı")
    batch_size: int = Field(4, ge=1, le=128, description="Batch boyutu")
    learning_rate: float = Field(1e-3, ge=1e-6, le=1e-1, description="Öğrenme oranı")
    d_model: int = Field(128, ge=8, le=2048, description="Embedding boyutu")
    n_layers: int = Field(4, ge=1, le=48, description="Transformer katman sayısı")
    n_heads: int = Field(4, ge=1, le=64, description="Attention kafa sayısı")
    max_seq_len: int = Field(128, ge=4, le=8192, description="Maksimum sequence uzunluğu")
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
    best_checkpoint: Optional[str] = None
    metrics: List[Dict[str, Any]] = []
    config: Dict[str, Any] = {}


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/start", response_model=TrainingJobResponse)
def start_training(
    request: StartTrainingRequest,
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(require_role("admin", "researcher"))
):
    """
    Yeni bir model eğitimi başlatır.
    """
    service = TrainingService(db)
    config = {
        "base_model": request.base_model,
        "base_version": request.base_version,
        "device": request.device,
        "seed": request.seed,
        "version": request.version,
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

    try:
        job = service.create_job(
            job_name=request.job_name,
            model_name=request.model_name,
            job_type=request.job_type,
            dataset_id=request.dataset_id,
            tokenizer_id=request.tokenizer_id,
            config=config
        )
    
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc

    # Arka planda eğitimi başlat
    service.start_training(str(job.job_id))

    return TrainingJobResponse(**job.to_dict())


@router.get("/jobs", response_model=List[TrainingJobResponse])
def list_training_jobs(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
) -> List[TrainingJobResponse]:
    """
    Eğitim işlerini listeler. Admin tüm işleri, diğer kullanıcılar sadece kendi işlerini görür.
    """
    TrainingService(db).recover_interrupted_jobs()
    query = db.query(TrainingJob)
    if current_user.role != "admin":
        query = query.filter(TrainingJob.user_id == current_user.user_id)
    jobs = query.order_by(TrainingJob.created_at.desc()).limit(limit).all()
    return [TrainingJobResponse(**job.to_dict()) for job in jobs]


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
) -> TrainingJobResponse:
    """
    Belirli bir eğitim işinin durumunu ve canlı metriklerini getirir.
    """
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Eğitim işi bulunamadı")

    if current_user.role != "admin" and job.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Bu eğitim işine erişim yetkiniz yok")

    return TrainingJobResponse(**job.to_dict())


@router.post("/jobs/{job_id}/cancel")
def cancel_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(require_role("admin", "researcher"))
) -> Dict[str, Any]:
    """
    Eğitim işini durdurur.
    """
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Eğitim işi bulunamadı")
    
    # Kullanıcı yalnızca kendi işini veya admin ise herhangi bir işi iptal edebilir
    if current_user.role != "admin" and job.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Bu işi iptal etme yetkiniz yok")

    service = TrainingService(db)
    cancelled = service.cancel_job(job_id)
    if cancelled or job.status == "RUNNING":
        return {"status": "success", "message": f"İş {job_id} iptal edildi"}
    
    return {"status": "ok", "message": f"İş durumu: {job.status}"}


@router.post("/jobs/{job_id}/resume", response_model=TrainingJobResponse)
def resume_training(
    job_id: str, 
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(require_role("admin", "researcher"))
):
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Eğitim işi bulunamadı")
    
    # Yetki kontrolü işlem başlatılmadan ÖNCE yapılmalı (P0 Güvenlik Düzeltmesi)
    if current_user.role != "admin" and job.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Bu işi devam ettirme yetkiniz yok")

    try:
        resumed_job = TrainingService(db).resume_job(
            job_id,
            user_id=current_user.user_id,
            is_admin=(current_user.role == "admin")
        )
        return TrainingJobResponse(**resumed_job.to_dict())
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe)) from pe
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/report")
def experiment_report(
    job_id: str, 
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
):
    job = db.query(TrainingJob).filter_by(job_id=job_id).first()
    if not job:
        raise HTTPException(404, "Experiment not found")
    if current_user.role != "admin" and job.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Bu rapora erişim yetkiniz yok")
    return job.to_dict()
