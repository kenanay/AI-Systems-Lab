"""
File Upload API Router

Dosya yükleme, listeleme ve yönetimi endpoint'leri.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Any
import logging
import threading

from backend.database import get_db
from backend.models import FileRecord, DocumentRecord, UserRecord, ContentRecord
from backend.security.dependencies import (
    get_current_user,
    require_role,
    check_resource_access,
    filter_by_owner
)
from backend.storage import storage_manager
from src.security.context import principal, Principal
from backend.utils import (
    generate_file_id,
    detect_mime_type,
    get_file_extension,
    is_allowed_file,
    format_file_size
)
from backend.config import settings
from backend.schemas import (
    BatchProcessRequest,
    FileRecordResponse,
    FileUploadResponse,
    DocumentRecordResponse,
    FileMetadataUpdate,
)
from backend.services.ingestion_service import ingestion_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/files", tags=["files"])

_content_store_lock = threading.Lock()


@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
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
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dosya adı boş olamaz"
        )
    filename: str = file.filename
    
    # 1. Extension kontrolü (dosya okumadan önce)
    if not is_allowed_file(filename, settings.allowed_extensions):
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
    mime_type = detect_mime_type(filename, file_content)
    logger.info(f"MIME type: {mime_type}")
    
    # 4. SHA-256 hash hesaplama (streaming-friendly)
    import hashlib
    sha256_hash = hashlib.sha256()
    for chunk in chunks:
        sha256_hash.update(chunk)
    sha256 = sha256_hash.hexdigest()
    logger.info(f"SHA-256: {sha256}")
    
    # 5. Duplicate kontrolü (Kullanıcı bazlı izolasyon)
    # Aynı kullanıcı daha önce aynı dosyayı yüklemişse kendi kaydını döndür
    current_user_id = str(current_user.user_id) if current_user and current_user.user_id else None
    user_query = db.query(FileRecord).filter(FileRecord.sha256 == sha256)
    if current_user_id:
        user_existing = user_query.filter(FileRecord.owner_id == current_user_id).first()
    else:
        user_existing = user_query.filter(FileRecord.owner_id.is_(None)).first()

    if user_existing:
        logger.warning(f"Duplicate dosya tespit edildi (kullanıcıya ait): {user_existing.file_id}")
        return FileUploadResponse(
            file_id=user_existing.file_id,
            filename=user_existing.original_name,
            size_bytes=user_existing.size_bytes,
            mime_type=user_existing.mime_type,
            sha256=user_existing.sha256,
            is_duplicate=True,
            message="Bu dosya daha önce sizin tarafınızdan yüklenmiş (duplicate)."
        )
    
    # 6. File ID üret
    file_id = generate_file_id()
    
    # 7. Fiziksel içerik kontrolü ve kaydetme (Content Store atomic deduplication)
    # Aynı içerik için ContentRecord ve kilit üzerinden yarış durumlarını (race condition) engelle
    with _content_store_lock:
        token = principal.set(Principal("system", "admin"))
        try:
            content_rec = db.query(ContentRecord).filter(ContentRecord.sha256 == sha256).first()
            if content_rec and content_rec.status == "ACTIVE" and storage_manager.file_exists(content_rec.relative_path):
                relative_path = content_rec.relative_path
                content_rec.ref_count += 1
                db.commit()
                logger.info(f"Mevcut ContentRecord yeniden kullanıldı: {relative_path} (ref_count={content_rec.ref_count})")
            else:
                from io import BytesIO
                file_stream = BytesIO(file_content)
                rel_path_obj, calculated_sha256 = storage_manager.save_file(
                    file_stream,
                    filename,
                    file_id
                )
                relative_path = str(rel_path_obj)
                if sha256 != calculated_sha256:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="SHA-256 uyuşmazlığı tespit edildi"
                    )
                if content_rec:
                    content_rec.relative_path = relative_path
                    content_rec.ref_count = 1
                    content_rec.status = "ACTIVE"
                    content_rec.size_bytes = file_size
                else:
                    content_rec = ContentRecord(
                        sha256=sha256,
                        relative_path=relative_path,
                        size_bytes=file_size,
                        ref_count=1,
                        status="ACTIVE"
                    )
                    db.add(content_rec)
                db.commit()
        finally:
            principal.reset(token)
    
    # 8. Kullanıcıya özel bağımsız FileRecord kaydı
    file_record = FileRecord(
        file_id=file_id,
        owner_id=current_user_id,
        original_name=filename,
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
    
    logger.info(f"Dosya kullanıcı için başarıyla kaydedildi: {file_id} (owner: {current_user_id})")
    
    return FileUploadResponse(
        file_id=file_id,
        filename=filename,
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
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
) -> Any:
    """
    Yüklenmiş dosyaları listele.
    
    Args:
        skip: Kaç kayıt atlanacak
        limit: Maksimum kaç kayıt dönülecek
        db: Database session
        
    Returns:
        FileRecord listesi
    """
    query = filter_by_owner(db.query(FileRecord), FileRecord, current_user, allow_unowned=True)
    files = query\
        .order_by(FileRecord.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    return files


@router.get("/{file_id}", response_model=FileRecordResponse)
def get_file(
    file_id: str, 
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
) -> FileRecordResponse:
    """
    Belirli bir dosyanın bilgilerini getir.
    
    Args:
        file_id: File ID
        db: Database session
        
    Returns:
        FileRecord
        
    Raises:
        HTTPException: Dosya bulunamazsa veya yetki yoksa
    """
    file_record = db.query(FileRecord).filter(FileRecord.file_id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dosya bulunamadı: {file_id}"
        )
    
    if not check_resource_access(file_record, current_user, allow_unowned=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu dosyaya erişim yetkiniz bulunmuyor."
        )
    
    return file_record


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_file(
    file_id: str, 
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(require_role("admin", "researcher"))
) -> None:
    """
    Dosyayı sil.
    
    Hem database kaydını hem fiziksel dosyayı siler.
    
    Args:
        file_id: File ID
        db: Database session
        
    Raises:
        HTTPException: Dosya bulunamazsa veya yetki yoksa
    """
    file_record = db.query(FileRecord).filter(FileRecord.file_id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dosya bulunamadı: {file_id}"
        )
    
    if not check_resource_access(file_record, current_user, allow_unowned=False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu dosyayı silme yetkiniz bulunmuyor."
        )
    
    # 1. Başka bir FileRecord kaydının aynı fiziksel dosyayı kullanıp kullanmadığını ve ContentRecord'u kontrol et
    rel_path = file_record.relative_path
    file_sha = file_record.sha256

    with _content_store_lock:
        # Kullanıcının FileRecord kaydını sil
        db.delete(file_record)
        db.commit()

        # 2. ContentRecord üzerinden atomik referans sayımı ve durum yönetimi
        token = principal.set(Principal("system", "admin"))
        should_delete_physical = False
        try:
            content_rec = db.query(ContentRecord).filter(ContentRecord.sha256 == file_sha).first()
            remaining_file_records = db.query(FileRecord).filter(FileRecord.sha256 == file_sha).count()

            if content_rec:
                content_rec.ref_count = remaining_file_records
                if remaining_file_records <= 0:
                    content_rec.status = "DELETING"
                    db.delete(content_rec)
                    db.commit()
                    should_delete_physical = True
                else:
                    db.commit()
            else:
                # ContentRecord kaydı henüz bulunmayan dosyalar için güvenli fallback
                if remaining_file_records == 0:
                    should_delete_physical = True
        finally:
            principal.reset(token)

        # 3. Yalnızca hiçbir aktif referans kalmadığında fiziksel dosyayı sil
        if should_delete_physical:
            storage_manager.delete_file(rel_path)
            logger.info(f"Fiziksel dosya ve son referans silindi: {rel_path} (ID: {file_id})")
        else:
            logger.info(
                f"Kullanıcı dosya kaydı ({file_id}) silindi. "
                f"Fiziksel dosya ({rel_path}) diğer kullanıcı referansları nedeniyle korundu."
            )


@router.patch("/{file_id}", response_model=FileRecordResponse)
def update_file_metadata(
    file_id: str,
    update_data: FileMetadataUpdate,
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(require_role("admin", "researcher"))
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
    
    if not check_resource_access(file_record, current_user, allow_unowned=False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu dosya metadatasını güncelleme yetkiniz bulunmuyor."
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
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
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
    
    if not check_resource_access(file_record, current_user, allow_unowned=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu dosyaya erişim yetkiniz bulunmuyor."
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


@router.post("/batch-process", status_code=status.HTTP_200_OK)
async def batch_process_files(
    request: BatchProcessRequest,
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
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
    file_ids = request.file_ids
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
            
            if not check_resource_access(file_record, current_user, allow_unowned=True):
                results["failed"] += 1
                results["errors"].append({
                    "file_id": file_id,
                    "error": "Bu dosyaya erişim yetkiniz bulunmuyor"
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
