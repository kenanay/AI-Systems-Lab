"""
PDF Parser

PDF dosyalarından metin ve metadata çıkarma.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

try:
    from pypdf import PdfReader
except ImportError:
    # Fallback to older PyPDF2 if available
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

from src.ingestion.base import BaseParser, ParserResult, ContentExtractionError

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """
    PDF dosyaları için parser.
    
    Özellikler:
    - Metin çıkarma (tüm sayfalardan)
    - Metadata extraction (title, author, subject, creator, producer, creation date)
    - Sayfa sayısı ve sayfa bazlı istatistikler
    - Bozuk PDF error handling
    - OCR desteği yok (future feature)
    
    Not: OCR için pytesseract ve pdf2image gerekir (opsiyonel dependency)
    """
    
    PARSER_NAME = "PDFParser"
    PARSER_VERSION = "1.0.0"
    SUPPORTED_EXTENSIONS = [".pdf"]
    
    def __init__(self, extract_images: bool = False, ocr_enabled: bool = False):
        """
        Args:
            extract_images: Görselleri de çıkar (future)
            ocr_enabled: OCR kullan (future)
        """
        super().__init__()
        
        if PdfReader is None:
            raise ImportError(
                "pypdf library is required for PDF parsing. "
                "Install it with: pip install pypdf"
            )
        
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        
        if ocr_enabled:
            logger.warning("OCR feature is not implemented yet")
    
    def parse(self, file_path: Path) -> ParserResult:
        """
        PDF dosyasını parse et.
        
        İşlem adımları:
        1. PDF aç ve validate et
        2. Metadata çıkar
        3. Her sayfadan metin çıkar
        4. İstatistik hesapla
        5. Quality checks
        
        Args:
            file_path: Parse edilecek PDF dosyası
            
        Returns:
            ParserResult
        """
        # Validation
        if not self.validate(file_path):
            return ParserResult(
                text="",
                metadata={},
                success=False,
                error="File validation failed"
            )
        
        try:
            # 1. PDF aç
            reader = self._open_pdf(file_path)
            
            # 2. Metadata çıkar
            metadata = self._extract_metadata(file_path, reader)
            
            # 3. Metin çıkar
            text, page_texts = self._extract_text(reader)
            
            # 4. İstatistik hesapla
            metadata.update(self._calculate_statistics(text, page_texts))
            
            # 5. Quality checks
            metadata.update(self._quality_checks(text, reader))
            
            return ParserResult(
                text=text,
                metadata=metadata,
                success=True
            )
            
        except ContentExtractionError as e:
            self.logger.error(f"Content extraction error: {e}")
            return ParserResult(
                text="",
                metadata={"error_type": "extraction"},
                success=False,
                error=str(e)
            )
        except Exception as e:
            self.logger.error(f"PDF parse error: {e}", exc_info=True)
            return ParserResult(
                text="",
                metadata={"error_type": "unknown"},
                success=False,
                error=str(e)
            )
    
    def _open_pdf(self, file_path: Path) -> PdfReader:
        """
        PDF dosyasını aç.
        
        Args:
            file_path: PDF dosya path'i
            
        Returns:
            PdfReader instance
            
        Raises:
            ContentExtractionError: PDF açılamazsa
        """
        try:
            return PdfReader(str(file_path))
        except Exception as e:
            raise ContentExtractionError(f"Failed to open PDF: {e}")
    
    def _extract_metadata(self, file_path: Path, reader: PdfReader) -> Dict[str, Any]:
        """
        PDF metadata'sını çıkar.
        
        Args:
            file_path: Dosya path'i
            reader: PdfReader instance
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            # Parser info
            "parser_name": self.PARSER_NAME,
            "parser_version": self.PARSER_VERSION,
            
            # File info
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            
            # PDF info
            "page_count": len(reader.pages),
            "is_encrypted": reader.is_encrypted,
        }
        
        # PDF metadata (varsa)
        if reader.metadata:
            pdf_meta = reader.metadata
            
            # Standart metadata alanları
            metadata.update({
                "pdf_title": self._clean_metadata_value(pdf_meta.get("/Title")),
                "pdf_author": self._clean_metadata_value(pdf_meta.get("/Author")),
                "pdf_subject": self._clean_metadata_value(pdf_meta.get("/Subject")),
                "pdf_creator": self._clean_metadata_value(pdf_meta.get("/Creator")),
                "pdf_producer": self._clean_metadata_value(pdf_meta.get("/Producer")),
                "pdf_creation_date": self._clean_metadata_value(pdf_meta.get("/CreationDate")),
                "pdf_mod_date": self._clean_metadata_value(pdf_meta.get("/ModDate")),
            })
        
        return metadata
    
    @staticmethod
    def _clean_metadata_value(value: Any) -> Optional[str]:
        """
        Metadata değerini temizle.
        
        PDF metadata bazen garip encoding'lerle gelebilir.
        
        Args:
            value: Metadata değeri
            
        Returns:
            Temizlenmiş string veya None
        """
        if value is None:
            return None
        
        try:
            # String'e çevir
            str_value = str(value).strip()
            
            # Boş mu?
            if not str_value or str_value == "None":
                return None
            
            return str_value
        except Exception:
            return None
    
    def _extract_text(self, reader: PdfReader) -> tuple[str, list[str]]:
        """
        PDF'den metin çıkar.
        
        Args:
            reader: PdfReader instance
            
        Returns:
            Tuple[tüm metin, sayfa bazlı metinler]
            
        Raises:
            ContentExtractionError: Metin çıkarılamazsa
        """
        page_texts = []
        
        try:
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    # Sayfa metnini çıkar
                    page_text = page.extract_text()
                    
                    if page_text:
                        page_texts.append(page_text)
                    else:
                        # Boş sayfa
                        self.logger.warning(f"Page {page_num} is empty or text could not be extracted")
                        page_texts.append("")
                        
                except Exception as e:
                    self.logger.error(f"Error extracting text from page {page_num}: {e}")
                    page_texts.append("")
            
            # Tüm sayfaları birleştir
            full_text = "\n\n".join(page_texts)
            
            return full_text, page_texts
            
        except Exception as e:
            raise ContentExtractionError(f"Failed to extract text: {e}")
    
    def _calculate_statistics(
        self,
        full_text: str,
        page_texts: list[str]
    ) -> Dict[str, Any]:
        """
        Metin istatistiklerini hesapla.
        
        Args:
            full_text: Tüm metin
            page_texts: Sayfa bazlı metinler
            
        Returns:
            İstatistik dictionary
        """
        words = full_text.split()
        lines = full_text.splitlines()
        
        # Sayfa başına istatistikler
        page_stats = {
            "pages_with_text": sum(1 for page in page_texts if page.strip()),
            "pages_without_text": sum(1 for page in page_texts if not page.strip()),
            "avg_chars_per_page": sum(len(page) for page in page_texts) / len(page_texts) if page_texts else 0,
        }
        
        return {
            # Genel istatistikler
            "char_count": len(full_text),
            "word_count": len(words),
            "line_count": len(lines),
            "byte_size": len(full_text.encode('utf-8')),
            
            # Sayfa istatistikleri
            **page_stats,
        }
    
    def _quality_checks(self, text: str, reader: PdfReader) -> Dict[str, Any]:
        """
        Kalite kontrolleri.
        
        Args:
            text: Çıkarılan metin
            reader: PdfReader instance
            
        Returns:
            Kalite bilgisi
        """
        quality = {
            "is_empty": len(text.strip()) == 0,
            "is_too_short": len(text) < 50,
            "is_encrypted": reader.is_encrypted,
            "has_extractable_text": len(text.strip()) > 0,
        }
        
        # Metin çıkarma oranı (sayfa başına ortalama karakter)
        avg_chars_per_page = len(text) / len(reader.pages) if len(reader.pages) > 0 else 0
        quality["avg_chars_per_page"] = avg_chars_per_page
        
        # Çok az metin varsa muhtemelen görsel tabanlı PDF
        quality["likely_image_based"] = avg_chars_per_page < 100
        
        # Quality score hesapla (0.0-1.0)
        score = 1.0
        
        if quality["is_empty"]:
            score = 0.0
        elif quality["is_encrypted"]:
            score = 0.2
        elif quality["likely_image_based"]:
            score = 0.3  # OCR gerekebilir
        elif quality["is_too_short"]:
            score = 0.5
        
        quality["quality_score"] = score
        
        # OCR önerisi
        quality["ocr_recommended"] = quality["likely_image_based"] and not quality["is_empty"]
        
        return quality


class SecurePDFParser(PDFParser):
    """
    Güvenlik odaklı PDF parser.
    
    Şifreli PDF'leri ve potansiyel zararlı içerikleri tespit eder.
    """
    
    PARSER_NAME = "SecurePDFParser"
    
    def _quality_checks(self, text: str, reader: PdfReader) -> Dict[str, Any]:
        """
        Genişletilmiş güvenlik kontrolleri.
        
        Args:
            text: Çıkarılan metin
            reader: PdfReader instance
            
        Returns:
            Kalite ve güvenlik bilgisi
        """
        quality = super()._quality_checks(text, reader)
        
        # Güvenlik kontrolleri
        security = {
            "is_encrypted": reader.is_encrypted,
            "has_javascript": self._check_javascript(reader),
            "has_embedded_files": self._check_embedded_files(reader),
        }
        
        quality.update(security)
        
        # Güvenlik skoru düşür
        if security["has_javascript"] or security["has_embedded_files"]:
            quality["quality_score"] *= 0.8
        
        return quality
    
    def _check_javascript(self, reader: PdfReader) -> bool:
        """
        PDF'de JavaScript var mı kontrol et.
        
        Args:
            reader: PdfReader instance
            
        Returns:
            JavaScript varsa True
        """
        # Basit kontrol - katalog'da JS alanı var mı?
        try:
            if hasattr(reader, 'trailer') and reader.trailer:
                root = reader.trailer.get('/Root')
                if root and '/Names' in root:
                    names = root['/Names']
                    if '/JavaScript' in names:
                        return True
        except Exception:
            pass
        
        return False
    
    def _check_embedded_files(self, reader: PdfReader) -> bool:
        """
        PDF'de embedded file var mı kontrol et.
        
        Args:
            reader: PdfReader instance
            
        Returns:
            Embedded file varsa True
        """
        try:
            if hasattr(reader, 'trailer') and reader.trailer:
                root = reader.trailer.get('/Root')
                if root and '/Names' in root:
                    names = root['/Names']
                    if '/EmbeddedFiles' in names:
                        return True
        except Exception:
            pass
        
        return False
