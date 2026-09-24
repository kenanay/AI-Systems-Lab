"""
backend/routers/tokenizer.py

Tokenizer API Endpoints

Bu modül tokenizer operasyonları için REST API sağlar:
- Training job oluşturma ve yönetimi
- Tokenizer listing ve metadata
- Text encoding/decoding
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
import logging

from backend.database import get_db
from backend.services.tokenizer_service import TokenizerTrainingService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/tokenizer",
    tags=["tokenizer"]
)


# ============================================================================
# Request/Response Models
# ============================================================================

class TokenizerTrainingRequest(BaseModel):
    """Tokenizer training job request"""
    job_name: str = Field(..., description="Job ismi")
    dataset_ids: Optional[List[str]] = Field(None, description="Dataset ID listesi")
    file_ids: Optional[List[str]] = Field(None, description="File ID listesi")
    vocab_size: int = Field(8000, ge=100, le=100000, description="Vocabulary boyutu")
    min_frequency: int = Field(2, ge=1, description="Minimum merge frequency")
    special_tokens: Optional[List[str]] = Field(
        None,
        description="Special tokens (default: [<PAD>, <UNK>, <BOS>, <EOS>])"
    )


class TokenizerJobResponse(BaseModel):
    """Tokenizer job response"""
    job_id: str
    job_name: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: dict = {}
    error: Optional[str] = None


class TokenizerResponse(BaseModel):
    """Tokenizer metadata response"""
    tokenizer_id: str
    name: str
    tokenizer_type: str
    version: str
    vocab_size: int
    num_merges: Optional[int] = None
    special_tokens: List[str]
    num_training_documents: Optional[int] = 0
    training_duration_seconds: Optional[float] = None
    is_active: bool
    created_at: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class TokenizerListResponse(BaseModel):
    """Tokenizer list item"""
    tokenizer_id: str
    name: str
    tokenizer_type: str
    vocab_size: int
    num_training_documents: Optional[int] = 0
    is_active: bool
    created_at: str
    tags: Optional[List[str]] = None


class EncodeRequest(BaseModel):
    """Text encoding request"""
    text: str = Field(..., description="Encode edilecek text")


class EncodeResponse(BaseModel):
    """Text encoding response"""
    text: str
    token_ids: List[int]
    num_tokens: int


class DecodeRequest(BaseModel):
    """Token decoding request"""
    token_ids: List[int] = Field(..., description="Decode edilecek token ID'leri")


class DecodeResponse(BaseModel):
    """Token decoding response"""
    token_ids: List[int]
    text: str


class TokenizerUpdateRequest(BaseModel):
    """Tokenizer metadata update request"""
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/train", response_model=TokenizerJobResponse, status_code=201)
async def create_training_job(
    request: TokenizerTrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Tokenizer training job oluştur.
    
    Job arka planda çalıştırılır. Job status `/tokenizer/jobs/{job_id}` 
    endpoint'inden takip edilebilir.
    
    **Gereksinimler:**
    - En az bir dataset_id veya file_id gerekli
    - vocab_size: 100-100000 arası
    - min_frequency: >= 1
    
    **Job States:**
    - PENDING: Job oluşturuldu, henüz başlamadı
    - RUNNING: Training devam ediyor
    - COMPLETED: Başarıyla tamamlandı
    - FAILED: Hata oluştu
    - CANCELLED: İptal edildi
    """
    try:
        service = TokenizerTrainingService(db)
        
        # Job oluştur
        job = service.create_training_job(
            job_name=request.job_name,
            dataset_ids=request.dataset_ids,
            file_ids=request.file_ids,
            vocab_size=request.vocab_size,
            min_frequency=request.min_frequency,
            special_tokens=request.special_tokens
        )
        
        job_id_str = str(job.job_id)
        job_name_str = str(job.job_name)
        status_str = str(job.status)
        progress_val = float(getattr(job, "progress", 0.0) or 0.0)
        created_at_str = job.created_at.isoformat() if hasattr(job.created_at, "isoformat") else str(job.created_at)

        # Background task olarak training başlat
        background_tasks.add_task(
            service.run_training_job,
            job_id_str
        )
        
        logger.info(f"Training job created and started: {job_id_str}")
        
        return TokenizerJobResponse(
            job_id=job_id_str,
            job_name=job_name_str,
            status=status_str,
            progress=progress_val,
            created_at=created_at_str,
            metadata={}
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training job creation failed: {e}")
        raise HTTPException(status_code=500, detail="Training job oluşturulamadı")


@router.get("/jobs", response_model=List[TokenizerJobResponse])
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Training job'ları listele.
    
    **Query Parameters:**
    - status: Filter by status (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
    - limit: Maximum sonuç sayısı (default: 50)
    """
    service = TokenizerTrainingService(db)
    jobs = service.list_jobs(status=status, limit=limit)
    
    return [
        TokenizerJobResponse(
            job_id=job["job_id"],
            job_name=job["job_name"],
            status=job["status"],
            progress=job["progress"],
            created_at=job["created_at"].isoformat(),
            metadata=job["metadata"]
        )
        for job in jobs
    ]


@router.get("/jobs/{job_id}", response_model=TokenizerJobResponse)
async def get_training_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Training job durumunu sorgula.
    
    **Returns:**
    - job_id, job_name, status, progress
    - created_at, started_at, completed_at
    - metadata: Training sonuçları (tokenizer_id, vocab_size, etc.)
    - error: Hata mesajı (varsa)
    """
    service = TokenizerTrainingService(db)
    job = service.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job bulunamadı")
    
    return TokenizerJobResponse(
        job_id=job["job_id"],
        job_name=job["job_name"],
        status=job["status"],
        progress=job["progress"],
        created_at=job["created_at"].isoformat(),
        started_at=job["started_at"].isoformat() if job["started_at"] else None,
        completed_at=job["completed_at"].isoformat() if job["completed_at"] else None,
        metadata=job["metadata"],
        error=job["error"]
    )


@router.delete("/jobs/{job_id}", status_code=204)
async def cancel_training_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Running training job'ı iptal et.
    
    **Note:** Sadece PENDING veya RUNNING durumundaki job'lar iptal edilebilir.
    """
    service = TokenizerTrainingService(db)
    success = service.cancel_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job iptal edilemedi (bulunamadı veya durumu uygun değil)"
        )
    
    return None


@router.get("/list", response_model=List[TokenizerListResponse])
async def list_tokenizers(
    is_active: Optional[bool] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Trained tokenizer'ları listele.
    
    **Query Parameters:**
    - is_active: Sadece aktif tokenizer'ları getir (true/false)
    - limit: Maximum sonuç sayısı (default: 50)
    """
    service = TokenizerTrainingService(db)
    tokenizers = service.list_tokenizers(is_active=is_active, limit=limit)
    
    return [
        TokenizerListResponse(
            tokenizer_id=tok["tokenizer_id"],
            name=tok["name"],
            tokenizer_type=tok["tokenizer_type"],
            vocab_size=tok["vocab_size"],
            num_training_documents=tok.get("num_training_documents") or 0,
            is_active=tok["is_active"],
            created_at=tok["created_at"].isoformat(),
            tags=tok["tags"]
        )
        for tok in tokenizers
    ]


@router.get("/{tokenizer_id}", response_model=TokenizerResponse)
async def get_tokenizer(
    tokenizer_id: str,
    db: Session = Depends(get_db)
):
    """
    Tokenizer metadata'sını getir.
    
    **Returns:**
    - Tokenizer bilgileri
    - Training configuration
    - Data lineage (source files/datasets)
    - Usage statistics
    """
    service = TokenizerTrainingService(db)
    tokenizer = service.get_tokenizer(tokenizer_id)
    
    if not tokenizer:
        raise HTTPException(status_code=404, detail="Tokenizer bulunamadı")
    
    return TokenizerResponse(
        tokenizer_id=tokenizer["tokenizer_id"],
        name=tokenizer["name"],
        tokenizer_type=tokenizer["tokenizer_type"],
        version=tokenizer["version"],
        vocab_size=tokenizer["vocab_size"],
        num_merges=tokenizer["num_merges"],
        special_tokens=tokenizer["special_tokens"] or [],
        num_training_documents=tokenizer.get("num_training_documents") or 0,
        training_duration_seconds=tokenizer["training_duration_seconds"],
        is_active=tokenizer["is_active"],
        created_at=tokenizer["created_at"].isoformat(),
        description=tokenizer["description"],
        tags=tokenizer["tags"]
    )


@router.post("/{tokenizer_id}/encode", response_model=EncodeResponse)
async def encode_text(
    tokenizer_id: str,
    request: EncodeRequest,
    db: Session = Depends(get_db)
):
    """
    Text'i token ID'lerine encode et.
    
    **Example:**
    ```json
    {
      "text": "Merhaba dünya"
    }
    ```
    
    **Returns:**
    ```json
    {
      "text": "Merhaba dünya",
      "token_ids": [42, 34, 57],
      "num_tokens": 3
    }
    ```
    """
    try:
        service = TokenizerTrainingService(db)
        tokenizer = service.load_tokenizer(tokenizer_id)
        
        if not tokenizer:
            raise HTTPException(status_code=404, detail="Tokenizer bulunamadı")
        
        # Encode
        token_ids = tokenizer.encode(request.text)
        
        return EncodeResponse(
            text=request.text,
            token_ids=token_ids,
            num_tokens=len(token_ids)
        )
        
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Tokenizer dosyaları bulunamadı"
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise HTTPException(status_code=500, detail="Encoding başarısız")


@router.post("/{tokenizer_id}/decode", response_model=DecodeResponse)
async def decode_tokens(
    tokenizer_id: str,
    request: DecodeRequest,
    db: Session = Depends(get_db)
):
    """
    Token ID'lerini text'e decode et.
    
    **Example:**
    ```json
    {
      "token_ids": [42, 34, 57]
    }
    ```
    
    **Returns:**
    ```json
    {
      "token_ids": [42, 34, 57],
      "text": "Merhaba dünya"
    }
    ```
    """
    try:
        service = TokenizerTrainingService(db)
        tokenizer = service.load_tokenizer(tokenizer_id)
        
        if not tokenizer:
            raise HTTPException(status_code=404, detail="Tokenizer bulunamadı")
        
        # Decode
        text = tokenizer.decode(request.token_ids)
        
        return DecodeResponse(
            token_ids=request.token_ids,
            text=text
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Tokenizer dosyaları bulunamadı"
        )
    except Exception as e:
        logger.error(f"Decoding failed: {e}")
        raise HTTPException(status_code=500, detail="Decoding başarısız")


@router.patch("/{tokenizer_id}", response_model=TokenizerResponse)
async def update_tokenizer(
    tokenizer_id: str,
    request: TokenizerUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Tokenizer metadata'sını güncelle.
    
    **Güncellenebilir Alanlar:**
    - description: Açıklama
    - tags: Tag listesi
    - is_active: Aktif durumu
    """
    service = TokenizerTrainingService(db)
    
    success = service.update_tokenizer_metadata(
        tokenizer_id=tokenizer_id,
        description=request.description,
        tags=request.tags,
        is_active=request.is_active
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Tokenizer bulunamadı")
    
    # Updated tokenizer'ı getir
    tokenizer = service.get_tokenizer(tokenizer_id)
    if not tokenizer:
        raise HTTPException(status_code=404, detail="Tokenizer bulunamadı")
    
    return TokenizerResponse(
        tokenizer_id=tokenizer["tokenizer_id"],
        name=tokenizer["name"],
        tokenizer_type=tokenizer["tokenizer_type"],
        version=tokenizer["version"],
        vocab_size=tokenizer["vocab_size"],
        num_merges=tokenizer["num_merges"],
        special_tokens=tokenizer["special_tokens"] or [],
        num_training_documents=tokenizer["num_training_documents"] or 0,
        training_duration_seconds=tokenizer["training_duration_seconds"],
        is_active=tokenizer["is_active"],
        created_at=tokenizer["created_at"].isoformat(),
        description=tokenizer["description"],
        tags=tokenizer["tags"]
    )


@router.delete("/{tokenizer_id}", status_code=204)
async def delete_tokenizer(
    tokenizer_id: str,
    hard_delete: bool = False,
    db: Session = Depends(get_db)
):
    """
    Tokenizer'ı sil.
    
    **Query Parameters:**
    - hard_delete: true ise dosyalar da silinir, false ise sadece deactivate edilir (default: false)
    
    **Note:** Soft delete önerilir, hard delete geri alınamaz.
    """
    service = TokenizerTrainingService(db)
    success = service.delete_tokenizer(tokenizer_id, hard_delete=hard_delete)
    
    if not success:
        raise HTTPException(status_code=404, detail="Tokenizer bulunamadı")
    
    return None
