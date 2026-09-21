"""
backend/routers/datasets_compiler.py

Dataset Compiler API Endpoints

Bu modül dataset compilation ve version management için REST API sağlar:
- Compilation job operations
- Dataset version listing
- Dataset download
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
import logging
from pathlib import Path

from backend.database import get_db
from backend.services.dataset_service import DatasetCompilationService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/datasets",
    tags=["datasets-compiler"]
)


# ============================================================================
# Request/Response Models
# ============================================================================

class CompilationJobRequest(BaseModel):
    """Dataset compilation job request"""
    job_name: str = Field(..., description="Job ismi")
    dataset_name: str = Field(..., description="Dataset ismi")
    dataset_version: Optional[str] = Field(None, description="Dataset version (auto-increment ise None)")
    document_ids: Optional[List[str]] = Field(None, description="Document ID listesi")
    file_ids: Optional[List[str]] = Field(None, description="File ID listesi")
    tokenizer_id: str = Field(..., description="Tokenizer ID")
    compilation_params: Optional[dict] = Field(
        None,
        description="Compilation parameters",
        example={
            "min_quality_score": 0.5,
            "max_quality_score": 1.0,
            "min_length": 10,
            "max_length": 100000,
            "allow_pii": False,
            "require_training_allowed": True,
            "remove_duplicates": True
        }
    )


class CompilationJobResponse(BaseModel):
    """Compilation job response"""
    job_id: str
    job_name: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_metadata: dict = {}
    error: Optional[str] = None


class DatasetVersionListResponse(BaseModel):
    """Dataset version list item"""
    dataset_id: str
    name: str
    version: str
    num_documents: int
    total_tokens: Optional[int] = None
    file_size_bytes: Optional[int] = None
    is_active: bool
    compiled_at: Optional[str] = None
    tags: Optional[List[str]] = None


class DatasetVersionDetailResponse(BaseModel):
    """Dataset version detail"""
    dataset_id: str
    name: str
    version: str
    description: Optional[str] = None
    schema_version: str
    compiler_version: str
    storage_path: str
    metadata_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    source_document_ids: List[str]
    source_file_ids: List[str]
    tokenizer_id: str
    compilation_job_id: Optional[str] = None
    num_documents: int
    total_tokens: Optional[int] = None
    total_chars: Optional[int] = None
    compilation_params: dict
    filter_stats: dict
    is_active: bool
    is_snapshot: bool
    created_at: str
    compiled_at: Optional[str] = None
    tags: Optional[List[str]] = None
    custom_metadata: Optional[dict] = None


# ============================================================================
# Compilation Job Endpoints
# ============================================================================

@router.post("/compile", response_model=CompilationJobResponse, status_code=201)
async def create_compilation_job(
    request: CompilationJobRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Dataset compilation job oluştur.
    
    Job arka planda çalıştırılır. Job status `/datasets/jobs/{job_id}` 
    endpoint'inden takip edilebilir.
    
    **Gereksinimler:**
    - tokenizer_id: Zorunlu (trained tokenizer)
    - document_ids VEYA file_ids: En az biri gerekli
    - dataset_version: Opsiyonel (None ise auto-increment)
    
    **Compilation Parameters (default):**
    - min_quality_score: 0.5
    - max_quality_score: 1.0
    - min_length: 10 chars
    - max_length: 100000 chars
    - allow_pii: False (PII filter aktif)
    - require_training_allowed: True (training permission gerekli)
    - remove_duplicates: True (deduplication aktif)
    
    **Job States:**
    - PENDING: Job oluşturuldu
    - RUNNING: Compilation devam ediyor
    - COMPLETED: Başarıyla tamamlandı
    - FAILED: Hata oluştu
    - CANCELLED: İptal edildi
    """
    try:
        service = DatasetCompilationService(db)
        
        # Job oluştur
        job = service.create_compilation_job(
            job_name=request.job_name,
            dataset_name=request.dataset_name,
            dataset_version=request.dataset_version,
            document_ids=request.document_ids,
            file_ids=request.file_ids,
            tokenizer_id=request.tokenizer_id,
            compilation_params=request.compilation_params
        )
        
        # Background task olarak compilation başlat
        background_tasks.add_task(
            service.run_compilation_job,
            job.job_id
        )
        
        logger.info(f"Compilation job created and started: {job.job_id}")
        
        return CompilationJobResponse(
            job_id=job.job_id,
            job_name=job.job_name,
            status=job.status,
            progress=job.progress,
            created_at=job.created_at.isoformat(),
            result_metadata={}
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Compilation job creation failed: {e}")
        raise HTTPException(status_code=500, detail="Compilation job oluşturulamadı")


@router.get("/compile/jobs", response_model=List[CompilationJobResponse])
async def list_compilation_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Compilation job'ları listele.
    
    **Query Parameters:**
    - status: Filter by status (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
    - limit: Maximum sonuç sayısı (default: 50)
    """
    service = DatasetCompilationService(db)
    jobs = service.list_jobs(status=status, limit=limit)
    
    return [
        CompilationJobResponse(
            job_id=job["job_id"],
            job_name=job["job_name"],
            status=job["status"],
            progress=job["progress"],
            created_at=job["created_at"].isoformat(),
            result_metadata=job["result_metadata"]
        )
        for job in jobs
    ]


@router.get("/compile/jobs/{job_id}", response_model=CompilationJobResponse)
async def get_compilation_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Compilation job durumunu sorgula.
    
    **Returns:**
    - job_id, job_name, status, progress
    - created_at, started_at, completed_at
    - result_metadata: Compilation sonuçları (dataset_id, stats, etc.)
    - error: Hata mesajı (varsa)
    """
    service = DatasetCompilationService(db)
    job = service.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job bulunamadı")
    
    return CompilationJobResponse(
        job_id=job["job_id"],
        job_name=job["job_name"],
        status=job["status"],
        progress=job["progress"],
        created_at=job["created_at"].isoformat(),
        started_at=job["started_at"].isoformat() if job["started_at"] else None,
        completed_at=job["completed_at"].isoformat() if job["completed_at"] else None,
        result_metadata=job["result_metadata"],
        error=job["error"]
    )


@router.delete("/compile/jobs/{job_id}", status_code=204)
async def cancel_compilation_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Running compilation job'ı iptal et.
    
    **Note:** Sadece PENDING veya RUNNING durumundaki job'lar iptal edilebilir.
    """
    service = DatasetCompilationService(db)
    success = service.cancel_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job iptal edilemedi (bulunamadı veya durumu uygun değil)"
        )
    
    return None


# ============================================================================
# Dataset Version Endpoints
# ============================================================================

@router.get("/versions", response_model=List[DatasetVersionListResponse])
async def list_dataset_versions(
    name: Optional[str] = None,
    is_active: Optional[bool] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Dataset version'larını listele.
    
    **Query Parameters:**
    - name: Filter by dataset name (opsiyonel)
    - is_active: Sadece aktif dataset'leri getir (true/false)
    - limit: Maximum sonuç sayısı (default: 50)
    """
    service = DatasetCompilationService(db)
    versions = service.list_dataset_versions(name=name, is_active=is_active, limit=limit)
    
    return [
        DatasetVersionListResponse(
            dataset_id=ds["dataset_id"],
            name=ds["name"],
            version=ds["version"],
            num_documents=ds["num_documents"],
            total_tokens=ds["total_tokens"],
            file_size_bytes=ds["file_size_bytes"],
            is_active=ds["is_active"],
            compiled_at=ds["compiled_at"].isoformat() if ds["compiled_at"] else None,
            tags=ds["tags"]
        )
        for ds in versions
    ]


@router.get("/versions/{dataset_id}", response_model=DatasetVersionDetailResponse)
async def get_dataset_version(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Dataset version detayını getir.
    
    **Returns:**
    - Dataset metadata
    - Compilation parameters
    - Filter statistics
    - Data lineage (source files, documents)
    - Storage information
    """
    service = DatasetCompilationService(db)
    dataset = service.get_dataset_version(dataset_id)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset version bulunamadı")
    
    return DatasetVersionDetailResponse(
        dataset_id=dataset["dataset_id"],
        name=dataset["name"],
        version=dataset["version"],
        description=dataset["description"],
        schema_version=dataset["schema_version"],
        compiler_version=dataset["compiler_version"],
        storage_path=dataset["storage_path"],
        metadata_path=dataset["metadata_path"],
        file_size_bytes=dataset["file_size_bytes"],
        source_document_ids=dataset["source_document_ids"] or [],
        source_file_ids=dataset["source_file_ids"] or [],
        tokenizer_id=dataset["tokenizer_id"],
        compilation_job_id=dataset["compilation_job_id"],
        num_documents=dataset["num_documents"],
        total_tokens=dataset["total_tokens"],
        total_chars=dataset["total_chars"],
        compilation_params=dataset["compilation_params"] or {},
        filter_stats=dataset["filter_stats"] or {},
        is_active=dataset["is_active"],
        is_snapshot=dataset["is_snapshot"],
        created_at=dataset["created_at"].isoformat(),
        compiled_at=dataset["compiled_at"].isoformat() if dataset["compiled_at"] else None,
        tags=dataset["tags"],
        custom_metadata=dataset["custom_metadata"]
    )


@router.get("/versions/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Dataset Parquet dosyasını indir.
    
    **Returns:** Parquet file (binary download)
    
    **Response Headers:**
    - Content-Type: application/octet-stream
    - Content-Disposition: attachment; filename="{dataset_name}_v{version}.parquet"
    """
    service = DatasetCompilationService(db)
    dataset = service.get_dataset_version(dataset_id)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset version bulunamadı")
    
    parquet_path = Path(dataset["storage_path"])
    
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="Dataset dosyası bulunamadı")
    
    # Filename: dataset_name_v1.0.0.parquet
    filename = f"{dataset['name']}_v{dataset['version']}.parquet".replace(" ", "_")
    
    return FileResponse(
        path=parquet_path,
        media_type="application/octet-stream",
        filename=filename
    )


@router.get("/versions/{dataset_id}/metadata")
async def download_dataset_metadata(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Dataset metadata.json dosyasını indir.
    
    **Returns:** JSON file (binary download)
    """
    service = DatasetCompilationService(db)
    dataset = service.get_dataset_version(dataset_id)
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset version bulunamadı")
    
    metadata_path = Path(dataset["metadata_path"]) if dataset["metadata_path"] else None
    
    if not metadata_path or not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata dosyası bulunamadı")
    
    filename = f"{dataset['name']}_v{dataset['version']}_metadata.json".replace(" ", "_")
    
    return FileResponse(
        path=metadata_path,
        media_type="application/json",
        filename=filename
    )
