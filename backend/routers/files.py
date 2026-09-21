"""
File Upload API Router

Dosya yükleme, listeleme ve yönetimi endpoint'leri.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List
import logging

from backend.database import get_db
from backend.models import FileRecord, DocumentRecord
from backend.storage import storage_manager
from backend.utils import (
    generate_file_id,
    detect_mime_type,
    get_file_extension,
    is_allowed_file,
    format_file_size
)
from backend.config import settings
from backend.schemas import FileRecordResponse, FileUploadResponse, DocumentRecordResponse, FileMetadataUpdate
from backend.services.ingestion_service import ingestion_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/files", tags=["files"])


@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> FileUploadResponse:
    """
    Dosya yükle ve sisteme kaydet.
    
    İşlem adımları:
    1. Dosya boyutu kontrolü
    2. Extension kontrolü
    3. MIME type detection
    4. SHA-256 hash hesaplama
    5. Duplicate kontrolü
    6. Fiziksel kaydetme
    7. Database kaydı
    
    Args:
        file: Upload edilen dosya
        db: Database session
        
    Returns:
        FileUploadResponse: Yükleme sonucu
        
    Raises:
        HTTPException: Dosya çok büyük, desteklenmeyen format, vb.
    """
    logger.info(f"File upload başladı: {file.filename}")
    
    # 1. Extension kontrolü (dosya okumadan önce)
    if not is_allowed_file(file.filename, settings.allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Desteklenmeyen dosya formatı. İzin verilen: {', '.join(settings.allowed_extensions)}"
        )
    
    # 2. Chunk-based streaming ile dosya boyutu kontrolü ve ilk bytes okuma
    # MIME detection için ilk chunk'ı oku
    CHUNK_SIZE = 8192  # 8KB chunks
    first_chunk = await file.read(CHUNK_SIZE)
    file_size = len(first_chunk)
    
    # Dosya boyutunu hesapla (streaming)
    chunks = [first_chunk]
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
        file_size += len(chunk)
        
        # Dosya çok büyükse erken dur
        if file_size > settings.max_upload_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Dosya çok büyük. Maksimum: {format_file_size(settings.max_upload_size)}"
            )
    
    # Tüm chunk'ları birleştir (MIME detection ve hash için gerekli)
    file_content = b"".join(chunks)
    logger.info(f"Dosya boyutu: {format_file_size(file_size)}")
    
    # 3. MIME type detection
    mime_type = detect_mime_type(file.filename, file_content)
    logger.info(f"MIME type: {mime_type}")
    
    # 4. SHA-256 hash hesaplama (streaming-friendly)
    import hashlib
    sha256_hash = hashlib.sha256()
    for chunk in chunks:
        sha256_hash.update(chunk)
    sha256 = sha256_hash.hexdigest()
    logger.info(f"SHA-256: {sha256}")
    
    # 5. Duplicate kontrolü
    existing_file = db.query(FileRecord).filter(FileRecord.sha256 == sha256).first()
    if existing_file:
        logger.warning(f"Duplicate dosya tespit edildi: {existing_file.file_id}")
        return FileUploadResponse(
            file_id=existing_file.file_id,
            filename=existing_file.original_name,
            size_bytes=existing_file.size_bytes,
            mime_type=existing_file.mime_type,
            sha256=existing_file.sha256,
            is_duplicate=True,
            message="Bu dosya daha önce yüklenmiş (duplicate)."
        )
    
    # 6. File ID üret
    file_id = generate_file_id()
    
    # 7. Fiziksel kaydetme (chunks'tan BytesIO oluştur)
    from io import BytesIO
    file_stream = BytesIO(file_content)
    relative_path, calculated_sha256 = storage_manager.save_file(
        file_stream,
        file.filename,
        file_id
    )
    
    # SHA-256 kontrolü (güvenlik)
    if sha256 != calculated_sha256:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SHA-256 uyuşmazlığı tespit edildi"
        )
    
    # 8. Database kaydı
    file_record = FileRecord(
        file_id=file_id,
        original_name=file.filename,
        relative_path=str(relative_path),
        mime_type=mime_type,
        size_bytes=file_size,
        sha256=sha256,
        parser_name=None,  # Henüz parse edilmedi
        parser_version=None,
        security_level="INTERNAL",  # Default
        pii_detected=False,  # Henüz taranmadı
        training_allowed=False,  # Default: izin yok
    )
    
    db.add(file_record)
    db.commit()
    db.refresh(file_record)
    
    logger.info(f"Dosya başarıyla kaydedildi: {file_id}")
    
    return FileUploadResponse(
        file_id=file_id,
        filename=file.filename,
        size_bytes=file_size,
        mime_type=mime_type,
        sha256=sha256,
        is_duplicate=False,
        message="Dosya başarıyla yüklendi."
    )


@router.get("/", response_model=List[FileRecordResponse])
def list_files(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> List[FileRecordResponse]:
    """
    Yüklenmiş dosyaları listele.
    
    Args:
        skip: Kaç kayıt atlanacak
        limit: Maksimum kaç kayıt dönülecek
        db: Database session
        
    Returns:
        FileRecord listesi
    """
    files = db.query(FileRecord)\
        .order_by(FileRecord.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    return files


@router.get("/{file_id}", response_model=FileRecordResponse)
def get_file(file_id: str, db: Session = Depends(get_db)) -> FileRecordResponse:
    """
    Belirli bir dosyanın bilgilerini getir.
    
    Args:
        file_id: File ID
        db: Database session
        
    Returns:
        FileRecord
        
    Raises:
        HTTPException: Dosya bulunamazsa
    """
    file_record = db.query(FileRecord).filter(FileRecord.file_id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dosya bulunamadı: {file_id}"
        )
    
    return file_record


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_file(file_id: str, db: Session = Depends(get_db)) -> None:
    """
    Dosyayı sil.
    
    Hem database kaydını hem fiziksel dosyayı siler.
    
    Args:
        file_id: File ID
        db: Database session
        
    Raises:
        HTTPException: Dosya bulunamazsa
    """
    file_record = db.query(FileRecord).filter(FileRecord.file_id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dosya bulunamadı: {file_id}"
        )
    
    # Fiziksel dosyayı sil
    storage_manager.delete_file(file_record.relative_path)
    
    # Database kaydını sil
    db.delete(file_record)
    db.commit()
    
    logger.info(f"Dosya silindi: {file_id}")


@router.patch("/{file_id}", response_model=FileRecordResponse)
def update_file_metadata(
    file_id: str,
    update_data: FileMetadataUpdate,
    db: Session = Depends(get_db)
) -> FileRecordResponse:
    """
    Dosya metadata'sını güncelle.
    
    Training izni, güvenlik seviyesi, lisans bilgisi gibi alanları güncelleyebilir.
    
    Args:
        file_id: File ID
        update_data: Güncellenecek alanlar
        db: Database session
        
    Returns:
        Güncellenmiş FileRecord
        
    Raises:
        HTTPException: Dosya bulunamazsa veya validasyon hatası
    """
    logger.info(f"Metadata update başladı: {file_id}")
    
    # Dosyayı bul
    file_record = db.query(FileRecord).filter(FileRecord.file_id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dosya bulunamadı: {file_id}"
        )
    
    # Güncelleme verilerini kontrol et (en az bir alan dolu olmalı)
    update_dict = update_data.model_dump(exclude_unset=True)
    
    if not update_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="En az bir alan güncellenmelidir"
        )
    
    # Alanları güncelle
    for field, value in update_dict.items():
        if hasattr(file_record, field):
            setattr(file_record, field, value)
            logger.info(f"  {field} = {value}")
    
    # Security level validasyonu
    if update_data.security_level and update_data.security_level not in [
        "PUBLIC", "INTERNAL", "RESTRICTED", "PERSONAL"
    ]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="security_level PUBLIC, INTERNAL, RESTRICTED veya PERSONAL olmalı"
        )
    
    db.commit()
    db.refresh(file_record)
    
    logger.info(f"Metadata güncellendi: {file_id}")
    
    return file_record


@router.post("/{file_id}/process", response_model=DocumentRecordResponse, status_code=status.HTTP_201_CREATED)
async def process_file(
    file_id: str,
    db: Session = Depends(get_db)
) -> DocumentRecordResponse:
    """
    Dosyayı parse et ve DocumentRecord oluştur.
    
    İşlem pipeline:
    1. FileRecord'u database'den al
    2. Fiziksel dosyayı oku
    3. Parser'ı seç (extension'a göre)
    4. Parse et (text extraction)
    5. Text normalization
    6. Quality scoring
    7. DocumentRecord oluştur ve kaydet
    8. FileRecord'u güncelle (parser info)
    
    Args:
        file_id: File ID
        db: Database session
        
    Returns:
        DocumentRecord
        
    Raises:
        HTTPException: Dosya bulunamazsa veya parse edilemezse
    """
    logger.info(f"Processing başladı: {file_id}")
    
    # 1. FileRecord'u al
    file_record = db.query(FileRecord).filter(FileRecord.file_id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dosya bulunamadı: {file_id}"
        )
    
    # 2. Zaten process edilmiş mi kontrol et
    existing_doc = db.query(DocumentRecord)\
        .filter(DocumentRecord.file_id == file_id)\
        .first()
    
    if existing_doc:
        logger.info(f"Dosya zaten process edilmiş: {file_id}")
        return existing_doc
    
    # 3. Dosya yolunu oluştur
    file_path = settings.raw_data_path / file_record.relative_path
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fiziksel dosya bulunamadı: {file_path}"
        )
    
    # 4. Ingestion service ile process et
    try:
        document_record = ingestion_service.process_file(
            file_record=file_record,
            file_path=file_path,
            db=db
        )
        
        logger.info(
            f"✅ Processing tamamlandı: {file_id} -> {document_record.document_id} "
            f"({document_record.word_count} words, quality={document_record.quality_score:.2f})"
        )
        
        return document_record
    
    except ValueError as e:
        # Parser bulunamadı, desteklenmeyen format
        logger.error(f"Parser error: {e}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(e)
        )
    
    except Exception as e:
        # Genel parse hatası
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dosya process edilemedi: {str(e)}"
        )


@router.post("/batch-process", status_code=status.HTTP_202_ACCEPTED)
async def batch_process_files(
    file_ids: List[str],
    db: Session = Depends(get_db)
) -> dict:
    """
    Birden fazla dosyayı batch olarak process et.
    
    Bu endpoint uzun sürebilir. İleride job queue ile asenkron yapılabilir.
    
    Args:
        file_ids: Process edilecek file ID listesi
        db: Database session
        
    Returns:
        Batch processing sonucu
    """
    logger.info(f"Batch processing başladı: {len(file_ids)} files")
    
    results = {
        "total": len(file_ids),
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "errors": []
    }
    
    for file_id in file_ids:
        try:
            # FileRecord'u al
            file_record = db.query(FileRecord).filter(FileRecord.file_id == file_id).first()
            
            if not file_record:
                results["failed"] += 1
                results["errors"].append({
                    "file_id": file_id,
                    "error": "Dosya bulunamadı"
                })
                continue
            
            # Zaten process edilmiş mi?
            existing_doc = db.query(DocumentRecord)\
                .filter(DocumentRecord.file_id == file_id)\
                .first()
            
            if existing_doc:
                results["skipped"] += 1
                continue
            
            # Process et
            file_path = settings.raw_data_path / file_record.relative_path
            
            if not file_path.exists():
                results["failed"] += 1
                results["errors"].append({
                    "file_id": file_id,
                    "error": "Fiziksel dosya bulunamadı"
                })
                continue
            
            document_record = ingestion_service.process_file(
                file_record=file_record,
                file_path=file_path,
                db=db
            )
            
            results["successful"] += 1
            logger.info(f"✅ Processed: {file_id} -> {document_record.document_id}")
        
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "file_id": file_id,
                "error": str(e)
            })
            logger.error(f"Batch processing error for {file_id}: {e}")
    
    logger.info(
        f"Batch processing tamamlandı: "
        f"{results['successful']} success, "
        f"{results['failed']} failed, "
        f"{results['skipped']} skipped"
    )
    
    return {
        "status": "completed",
        "results": results,
        "message": f"Batch processing tamamlandı. {results['successful']}/{results['total']} başarılı."
    }
