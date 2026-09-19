"""
Ingestion Service

Dosya parse etme ve DocumentRecord oluşturma servisi.
"""

from pathlib import Path
from sqlalchemy.orm import Session
import logging

from backend.models import FileRecord, DocumentRecord
from backend.storage import storage_manager
from backend.utils import generate_document_id, detect_language
from src.ingestion import get_parser
from src.normalization import NormalizationPipeline

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Dosya ingestion işlemlerini yöneten servis.
    
    Sorumluluklar:
    - FileRecord'dan dosyayı okuma
    - Uygun parser seçimi
    - Parse işlemi
    - Normalization
    - DocumentRecord oluşturma
    """
    
    def __init__(self):
        self.normalization_pipeline = NormalizationPipeline()
    
    def process_file(
        self,
        file_record: FileRecord,
        file_path: Path,
        db: Session
    ) -> DocumentRecord:
        """
        Bir dosyayı işle ve DocumentRecord oluştur.
        
        İşlem adımları:
        1. FileRecord'dan dosya path'ini al
        2. Uygun parser'ı seç
        3. Dosyayı parse et
        4. Normalization uygula
        5. DocumentRecord oluştur ve kaydet
        
        Args:
            file_record: İşlenecek FileRecord
            file_path: Dosyanın fiziksel path'i
            db: Database session
            
        Returns:
            Oluşturulan DocumentRecord
            
        Raises:
            Exception: İşlem başarısız olursa
        """
        logger.info(f"Processing file: {file_record.file_id}")
        
        try:
            # 1. Dosya path kontrolü
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # 2. Parser seç (dosya extension'ından)
            file_extension = Path(file_record.original_name).suffix
            parser = get_parser(file_extension)
            
            # 3. Parse et
            logger.info(f"Parsing with {parser.PARSER_NAME}")
            parse_result = parser.parse(file_path)
            
            if not parse_result.success:
                logger.error(f"Parse failed: {parse_result.error}")
                raise Exception(f"Parse failed: {parse_result.error}")
            
            # 4. Normalization
            logger.info("Applying normalization")
            normalized_text, updated_metadata = self.normalization_pipeline.process(
                parse_result.text,
                parse_result.metadata
            )
            
            # 5. DocumentRecord oluştur
            document_id = generate_document_id()
            
            # Title belirleme (öncelik sırasına göre)
            title = (
                updated_metadata.get("pdf_title") or
                updated_metadata.get("title") or
                file_record.original_name
            )
            
            document = DocumentRecord(
                document_id=document_id,
                file_id=file_record.file_id,
                title=title,
                text=normalized_text,
                language=updated_metadata.get("language"),
                char_count=updated_metadata.get("char_count_after_normalization"),
                word_count=updated_metadata.get("word_count_after_normalization"),
                line_count=updated_metadata.get("line_count"),
                parser_name=updated_metadata.get("parser_name"),
                parser_version=updated_metadata.get("parser_version"),
                schema_version=file_record.schema_version,
                is_empty=len(normalized_text.strip()) == 0,
                is_duplicate=False,  # TODO: Duplicate detection
                quality_score=updated_metadata.get("quality_score"),
            )
            
            # Database'e kaydet
            db.add(document)
            
            # FileRecord'u güncelle (parse edildiğini belirt)
            file_record.parser_name = parser.PARSER_NAME
            file_record.parser_version = parser.PARSER_VERSION
            file_record.language = updated_metadata.get("language")
            file_record.quality_score = updated_metadata.get("quality_score")
            
            db.commit()
            db.refresh(document)
            
            logger.info(f"Document created: {document_id}")
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing file {file_record.file_id}: {e}", exc_info=True)
            db.rollback()
            raise
    
    def process_multiple_files(
        self,
        file_records: list[FileRecord],
        db: Session
    ) -> list[DocumentRecord]:
        """
        Birden fazla dosyayı işle.
        
        Args:
            file_records: İşlenecek FileRecord listesi
            db: Database session
            
        Returns:
            Oluşturulan DocumentRecord listesi
        """
        from backend.config import settings
        
        documents = []
        
        for file_record in file_records:
            try:
                # Dosya path'ini oluştur
                file_path = settings.raw_data_path / file_record.relative_path
                
                document = self.process_file(file_record, file_path, db)
                documents.append(document)
            except Exception as e:
                logger.error(f"Failed to process {file_record.file_id}: {e}")
                # Continue with other files
                continue
        
        return documents
    
    def reprocess_file(
        self,
        file_record: FileRecord,
        file_path: Path,
        db: Session
    ) -> DocumentRecord:
        """
        Dosyayı yeniden işle.
        
        Mevcut DocumentRecord'u günceller veya yeni oluşturur.
        
        Args:
            file_record: İşlenecek FileRecord
            file_path: Dosyanın fiziksel path'i
            db: Database session
            
        Returns:
            Güncellenmiş/yeni DocumentRecord
        """
        logger.info(f"Reprocessing file: {file_record.file_id}")
        
        # Mevcut dokümanları sil
        db.query(DocumentRecord)\
            .filter(DocumentRecord.file_id == file_record.file_id)\
            .delete()
        
        db.commit()
        
        # Yeniden işle
        return self.process_file(file_record, file_path, db)


# Global instance
ingestion_service = IngestionService()
