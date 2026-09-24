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
from src.pii.turkish_detector import TurkishPIIDetector

logger = logging.getLogger(__name__)


class DocumentQualityScorer:
    """
    Document quality scoring logic.
    
    Kalite skorunu hesaplar:
    - Length (çok kısa/uzun penaltı)
    - Repetition ratio (tekrar eden kelimeler)
    - Special char ratio (özel karakter oranı)
    - Language confidence (dil güveni)
    - Line/word ratio (yapısal tutarlılık)
    """
    
    def score_document(self, text: str, metadata: dict) -> float:
        """
        Document kalite skorunu hesapla (0.0 - 1.0).
        
        Args:
            text: Document text
            metadata: Parse metadata
            
        Returns:
            Quality score (0.0 - 1.0)
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        scores = []
        
        # 1. Length score
        char_count = len(text)
        word_count = metadata.get("word_count_after_normalization", len(text.split()))
        
        if char_count < 50:  # Çok kısa
            length_score = char_count / 50.0  # 0.0 - 1.0
        elif char_count > 500000:  # Çok uzun
            length_score = max(0.5, 1.0 - (char_count - 500000) / 1000000)
        else:
            length_score = 1.0
        
        scores.append(("length", length_score, 0.15))
        
        # 2. Repetition ratio
        words = text.lower().split()
        if len(words) > 0:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)  # Type-Token Ratio
            
            if repetition_ratio > 0.6:  # Yüksek çeşitlilik
                rep_score = 1.0
            elif repetition_ratio > 0.4:
                rep_score = 0.8
            elif repetition_ratio > 0.2:
                rep_score = 0.5
            else:  # Çok fazla tekrar
                rep_score = 0.2
        else:
            rep_score = 0.0
        
        scores.append(("repetition", rep_score, 0.25))
        
        # 3. Special char ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(1, char_count)
        
        if special_ratio < 0.05:  # Normal
            special_score = 1.0
        elif special_ratio < 0.15:  # Kabul edilebilir
            special_score = 0.8
        elif special_ratio < 0.30:  # Fazla
            special_score = 0.5
        else:  # Çok fazla özel karakter (muhtemelen kod/binary)
            special_score = 0.2
        
        scores.append(("special_chars", special_score, 0.20))
        
        # 4. Line/word ratio (yapısal tutarlılık)
        line_count = metadata.get("line_count", text.count("\n") + 1)
        if line_count > 0 and word_count > 0:
            words_per_line = word_count / line_count
            
            if 5 <= words_per_line <= 50:  # Normal prose
                structure_score = 1.0
            elif 2 <= words_per_line < 5 or 50 < words_per_line <= 100:
                structure_score = 0.7
            else:  # Aşırı kısa satırlar veya çok uzun satırlar
                structure_score = 0.4
        else:
            structure_score = 0.5
        
        scores.append(("structure", structure_score, 0.20))
        
        # 5. Language confidence (varsa)
        lang = metadata.get("language")
        if lang and lang in ["tr", "en", "de", "fr", "es", "it"]:  # Bilinen diller
            lang_score = 0.9
        elif lang:  # Tespit edilmiş ama bilinmeyen
            lang_score = 0.7
        else:  # Dil tespit edilememiş
            lang_score = 0.5
        
        scores.append(("language", lang_score, 0.20))
        
        # Weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        
        # Final quality score (0.0 - 1.0)
        return round(max(0.0, min(1.0, total_score)), 3)


class IngestionService:
    """
    Dosya ingestion işlemlerini yöneten servis.
    
    Sorumluluklar:
    - FileRecord'dan dosyayı okuma
    - Uygun parser seçimi
    - Parse işlemi
    - Normalization
    - Quality scoring
    - DocumentRecord oluşturma
    """
    
    def __init__(self):
        self.normalization_pipeline = NormalizationPipeline()
        self.pii_detector = TurkishPIIDetector()
        self.quality_scorer = DocumentQualityScorer()
    
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
            file_extension = Path(str(file_record.original_name)).suffix
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
            title = str(
                updated_metadata.get("pdf_title") or
                updated_metadata.get("title") or
                file_record.original_name
            )
            
            raw_lang = updated_metadata.get("language")
            doc_lang = str(raw_lang) if raw_lang is not None else None
            
            # Quality scoring
            raw_quality = updated_metadata.get("quality_score")
            if raw_quality is not None:
                doc_quality = float(raw_quality)
            else:
                # Otomatik quality scoring
                doc_quality = self.quality_scorer.score_document(
                    text=normalized_text,
                    metadata=updated_metadata
                )
                logger.debug(f"Computed quality score: {doc_quality:.3f} for {document_id}")
            
            document = DocumentRecord(
                document_id=document_id,
                file_id=str(file_record.file_id),
                owner_id=getattr(file_record, "owner_id", None),
                title=title,
                text=normalized_text,
                language=doc_lang,
                char_count=int(updated_metadata["char_count_after_normalization"]) if updated_metadata.get("char_count_after_normalization") is not None else None,
                word_count=int(updated_metadata["word_count_after_normalization"]) if updated_metadata.get("word_count_after_normalization") is not None else None,
                line_count=int(updated_metadata["line_count"]) if updated_metadata.get("line_count") is not None else None,
                parser_name=str(updated_metadata["parser_name"]) if updated_metadata.get("parser_name") is not None else None,
                parser_version=str(updated_metadata["parser_version"]) if updated_metadata.get("parser_version") is not None else None,
                schema_version=str(file_record.schema_version),
                is_empty=len(normalized_text.strip()) == 0,
                is_duplicate=False,  # TODO: Duplicate detection
                quality_score=doc_quality,
            )
            
            # Database'e kaydet
            db.add(document)
            
            # FileRecord'u güncelle (parse edildiğini belirt)
            file_record.parser_name = str(parser.PARSER_NAME)
            file_record.parser_version = str(parser.PARSER_VERSION)
            file_record.language = doc_lang
            file_record.quality_score = doc_quality
            
            # PII Taraması
            try:
                pii_matches = self.pii_detector.scan_text(normalized_text)
                file_record.pii_detected = bool(len(pii_matches) > 0)
                if file_record.pii_detected:
                    logger.info(f"PII detected in {file_record.file_id}: {len(pii_matches)} occurrences")
            except Exception as e:
                logger.warning(f"PII scan warning for {file_record.file_id}: {e}")
            
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
                file_path = Path(settings.raw_data_path) / str(file_record.relative_path)
                
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
