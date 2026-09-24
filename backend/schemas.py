"""
Pydantic Schemas

API request/response modelleri.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


class FileUploadResponse(BaseModel):
    """File upload response"""
    file_id: str
    filename: str
    size_bytes: int
    mime_type: str
    sha256: str
    is_duplicate: bool = False
    message: str
    
    class Config:
        from_attributes = True


class BatchProcessRequest(BaseModel):
    """Request body for synchronous batch file processing."""
    file_ids: List[str] = Field(..., min_length=1, max_length=100)


class FileRecordResponse(BaseModel):
    """FileRecord response"""
    file_id: str
    original_name: str
    relative_path: str
    mime_type: str
    size_bytes: int
    sha256: str
    parser_name: Optional[str] = None
    parser_version: Optional[str] = None
    schema_version: Optional[str] = None
    security_level: str
    pii_detected: bool
    license: Optional[str] = None
    copyright_status: Optional[str] = None
    training_allowed: bool
    created_at: datetime
    dataset_version: Optional[str] = None
    source: Optional[str] = None
    language: Optional[str] = None
    quality_score: Optional[float] = None
    
    class Config:
        from_attributes = True


class DocumentRecordResponse(BaseModel):
    """DocumentRecord response"""
    document_id: str
    file_id: str
    title: Optional[str] = None
    text: str
    language: Optional[str] = None
    char_count: Optional[int] = None
    word_count: Optional[int] = None
    line_count: Optional[int] = None
    quality_score: Optional[float] = None
    is_empty: bool
    is_duplicate: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentPreview(BaseModel):
    """Document preview (truncated text)"""
    document_id: str
    file_id: str
    title: Optional[str] = None
    text_preview: str  # İlk N karakter
    full_text_length: int
    language: Optional[str] = None
    char_count: Optional[int] = None
    word_count: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class ProcessingJobResponse(BaseModel):
    """Processing job response"""
    job_id: str
    job_type: str
    status: str
    progress: float
    total_items: Optional[int] = None
    processed_items: int
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class IngestionStats(BaseModel):
    """Ingestion istatistikleri"""
    total_files: int = 0
    total_documents: int = 0
    total_size_bytes: int = 0
    files_by_type: dict[str, int] = Field(default_factory=dict)
    files_by_language: dict[str, int] = Field(default_factory=dict)
    avg_quality_score: Optional[float] = None
    pii_detected_count: int = 0
    training_allowed_count: int = 0


class FileMetadataUpdate(BaseModel):
    """File metadata güncelleme request"""
    training_allowed: Optional[bool] = None
    security_level: Optional[str] = None
    license: Optional[str] = None
    copyright_status: Optional[str] = None
    source: Optional[str] = None
    dataset_version: Optional[str] = None
    
    class Config:
        # En az bir alan dolu olmalı
        json_schema_extra = {
            "example": {
                "training_allowed": True,
                "license": "MIT",
                "security_level": "PUBLIC"
            }
        }
