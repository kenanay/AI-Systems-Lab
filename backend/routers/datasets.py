"""
Dataset API Router

Dataset listeleme, istatistik ve export endpoint'leri.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, status
from sqlalchemy.orm import Session
from typing import Any, List, Optional
import logging

from backend.database import get_db
from backend.models import DocumentRecord, UserRecord, FileRecord
from backend.security.dependencies import (
    get_current_user,
    require_role,
    check_resource_access,
    filter_by_owner
)
from backend.schemas import (
    DocumentRecordResponse,
    DocumentPreview,
    IngestionStats
)
from backend.config import settings
from src.dataset import parquet_writer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@router.get("/stats", response_model=IngestionStats)
def get_dataset_statistics(
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
) -> IngestionStats:
    """
    Dataset istatistiklerini getir.
    
    Returns:
        IngestionStats: Toplam dosya, doküman, boyut, dağılımlar
    """
    # Parquet'ten istatistikleri al
    stats = parquet_writer.get_statistics()
    
    return IngestionStats(**stats)


@router.get("/documents", response_model=List[DocumentPreview])
def list_documents(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    language: Optional[str] = None,
    min_quality: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
) -> List[DocumentPreview]:
    """
    Dokümanları listele (preview mode - truncated text).
    
    Args:
        skip: Kaç kayıt atlanacak
        limit: Maksimum kaç kayıt dönülecek
        language: Dil filtresi (opsiyonel)
        min_quality: Minimum kalite skoru (opsiyonel)
        db: Database session
        
    Returns:
        DocumentPreview listesi
    """
    query = filter_by_owner(db.query(DocumentRecord), DocumentRecord, current_user, allow_unowned=True)
    
    # Filtreler
    if language:
        query = query.filter(DocumentRecord.language == language)
    
    if min_quality is not None:
        query = query.filter(DocumentRecord.quality_score >= min_quality)
    
    # Boş dokümanları hariç tut
    query = query.filter(DocumentRecord.is_empty == False)
    
    # Pagination
    documents = query.order_by(DocumentRecord.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    # Preview formatına dönüştür
    previews = []
    for doc in documents:
        d: Any = doc
        doc_text = str(d.text or "")
        preview_text = doc_text[:500]
        if len(doc_text) > 500:
            preview_text += "..."
        
        previews.append(DocumentPreview(
            document_id=str(d.document_id),
            file_id=str(d.file_id),
            title=str(d.title) if d.title is not None else None,
            text_preview=preview_text,
            full_text_length=len(doc_text),
            language=str(d.language) if d.language is not None else None,
            char_count=int(d.char_count) if d.char_count is not None else None,
            word_count=int(d.word_count) if d.word_count is not None else None,
            created_at=d.created_at
        ))
    
    return previews


@router.get("/documents/{document_id}", response_model=DocumentRecordResponse)
def get_document(
    document_id: str, 
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
) -> DocumentRecordResponse:
    """
    Belirli bir dokümanın tam içeriğini getir.
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        DocumentRecord (full text)
        
    Raises:
        HTTPException: Doküman bulunamazsa veya erişim yetkisi yoksa
    """
    document = db.query(DocumentRecord)\
        .filter(DocumentRecord.document_id == document_id)\
        .first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Doküman bulunamadı: {document_id}"
        )
    
    if not check_resource_access(document, current_user, allow_unowned=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu dokümana erişim yetkiniz bulunmuyor."
        )
    
    return document


@router.get("/documents/by-file/{file_id}", response_model=DocumentRecordResponse)
def get_document_by_file(
    file_id: str,
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
) -> DocumentRecordResponse:
    """
    File ID'ye göre dokümanı getir.
    
    Args:
        file_id: File ID
        db: Database session
        
    Returns:
        DocumentRecord (full text)
        
    Raises:
        HTTPException: Doküman bulunamazsa veya erişim yetkisi yoksa
    """
    # Dosya varlığı ve erişim kontrolü
    file_record = db.query(FileRecord).filter(FileRecord.file_id == file_id).first()
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dosya bulunamadı: {file_id}"
        )
    if not check_resource_access(file_record, current_user, allow_unowned=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu dosyaya ve bağlı dokümanlarına erişim yetkiniz bulunmuyor."
        )

    document = db.query(DocumentRecord)\
        .filter(DocumentRecord.file_id == file_id)\
        .first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bu dosya için doküman bulunamadı: {file_id}. Dosya henüz parse edilmemiş olabilir."
        )
    
    return document


@router.post("/export/parquet", status_code=status.HTTP_202_ACCEPTED)
def export_to_parquet(
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(require_role("admin", "researcher"))
) -> dict:
    """
    Database'deki tüm kayıtları Parquet formatına export et.
    
    Files → files.parquet
    Documents → documents.parquet
    
    Returns:
        Export durumu
    """
    from backend.models import FileRecord
    
    logger.info("Starting Parquet export...")
    
    # FileRecord'ları al
    file_records = db.query(FileRecord).all()
    file_dicts = [
        {
            'file_id': f.file_id,
            'original_name': f.original_name,
            'relative_path': f.relative_path,
            'mime_type': f.mime_type,
            'size_bytes': f.size_bytes,
            'sha256': f.sha256,
            'parser_name': f.parser_name,
            'parser_version': f.parser_version,
            'schema_version': f.schema_version,
            'security_level': f.security_level,
            'pii_detected': f.pii_detected,
            'license': f.license,
            'copyright_status': f.copyright_status,
            'training_allowed': f.training_allowed,
            'created_at': f.created_at,
            'modified_at': f.modified_at,
            'dataset_version': f.dataset_version,
            'source': f.source,
            'language': f.language,
            'quality_score': f.quality_score,
        }
        for f in file_records
    ]
    
    # DocumentRecord'ları al
    document_records = db.query(DocumentRecord).all()
    doc_dicts = [
        {
            'document_id': d.document_id,
            'file_id': d.file_id,
            'title': d.title,
            'text': d.text,
            'language': d.language,
            'char_count': d.char_count,
            'word_count': d.word_count,
            'line_count': d.line_count,
            'parser_name': d.parser_name,
            'parser_version': d.parser_version,
            'schema_version': d.schema_version,
            'is_empty': d.is_empty,
            'is_duplicate': d.is_duplicate,
            'quality_score': d.quality_score,
            'created_at': d.created_at,
        }
        for d in document_records
    ]
    
    # Parquet'e yaz
    files_path = parquet_writer.write_files(file_dicts) if file_dicts else None
    docs_path = parquet_writer.write_documents(doc_dicts) if doc_dicts else None
    
    logger.info(f"✅ Parquet export completed")
    
    return {
        "status": "completed",
        "files_exported": len(file_dicts),
        "documents_exported": len(doc_dicts),
        "files_path": str(files_path) if files_path else None,
        "documents_path": str(docs_path) if docs_path else None,
        "message": "Database içeriği Parquet formatına export edildi"
    }


@router.post("/export/pretraining", status_code=status.HTTP_202_ACCEPTED)
def export_for_pretraining(
    min_quality_score: float = Query(default=0.5, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(require_role("admin", "researcher"))
) -> dict:
    """
    Pretraining formatına export et (JSONL).
    
    **GÜVENLİK:** PII ve training izni otomatik kontrol edilir.
    
    Args:
        min_quality_score: Minimum kalite skoru eşiği
        db: Database session
        
    Returns:
        Export durumu
    """
    from pathlib import Path
    from src.dataset import dataset_exporter
    
    logger.info(f"Starting pretraining export (min_quality={min_quality_score})...")
    
    # Önce Parquet'e export et (güncel olması için)
    export_to_parquet(db)
    
    # Pretraining formatına export
    output_path = settings.processed_data_path / "pretraining" / "train.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        result_path = dataset_exporter.export_for_pretraining(
            output_path=output_path,
            filter_pii=True,  # ❌ Devre dışı bırakılamaz
            require_training_allowed=True,
            min_quality_score=min_quality_score
        )
        
        if result_path:
            # Dosya boyutunu al
            file_size = result_path.stat().st_size
            
            return {
                "status": "completed",
                "output_path": str(result_path),
                "file_size_bytes": file_size,
                "security_filters": {
                    "pii_filtered": True,
                    "training_allowed_required": True,
                    "min_quality_score": min_quality_score
                },
                "message": "✅ Pretraining dataset oluşturuldu (güvenlik filtreleri uygulandı)"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Güvenlik filtrelerinden geçen veri bulunamadı. "
                       "training_allowed=True, pii_detected=False ayarlı veri ekleyin."
            )
    
    except ValueError as e:
        # Güvenlik ihlali
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export başarısız: {str(e)}"
        )


@router.post("/versions/{version}", status_code=status.HTTP_201_CREATED)
def create_dataset_version(
    version: str,
    description: str = Query(default="", max_length=500),
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(require_role("admin", "researcher"))
) -> dict:
    """
    Yeni bir immutable dataset versiyonu oluştur.
    
    Semantic versioning: v1.0.0, v1.1.0, v2.0.0
    
    Args:
        version: Version string (örn: "v1.0.0")
        description: Version açıklaması
        db: Database session
        
    Returns:
        Oluşturulan version bilgisi
    """
    from src.dataset import dataset_exporter
    
    # Önce Parquet'e export et
    export_to_parquet(db)
    
    # Version oluştur
    version_dir = dataset_exporter.create_versioned_dataset(
        version=version,
        description=description
    )
    
    return {
        "status": "created",
        "version": version,
        "version_dir": str(version_dir),
        "description": description,
        "message": f"✅ Immutable dataset version {version} oluşturuldu"
    }
