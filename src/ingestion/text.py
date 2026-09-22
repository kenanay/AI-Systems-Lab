"""
TXT ve MD Parser

Plain text ve Markdown dosyaları için parser.
"""

from pathlib import Path
from typing import Dict, Any
import chardet
import logging

from src.ingestion.base import BaseParser, ParserResult, EncodingError, ParserError

logger = logging.getLogger(__name__)


class TextParser(BaseParser):
    """
    Plain text (.txt, .md) dosyaları için parser.
    
    Özellikler:
    - UTF-8 encoding kontrolü ve auto-detection
    - Boş içerik kontrolü
    - Whitespace normalization
    - Metadata extraction (char count, line count, vb.)
    """
    
    PARSER_NAME = "TextParser"
    PARSER_VERSION = "1.0.0"
    SUPPORTED_EXTENSIONS = [".txt", ".md", ".markdown", ".text"]
    
    # Encoding detection için güven eşiği
    CONFIDENCE_THRESHOLD = 0.7
    
    def __init__(self, normalize_whitespace: bool = True):
        """
        Args:
            normalize_whitespace: Whitespace'leri normalize et
        """
        super().__init__()
        self.normalize_whitespace = normalize_whitespace
    
    def parse(self, file_path: Path) -> ParserResult:
        """
        Text dosyasını parse et.
        
        İşlem adımları:
        1. Encoding detection
        2. İçerik okuma
        3. Normalization (opsiyonel)
        4. Metadata extraction
        
        Args:
            file_path: Parse edilecek dosya
            
        Returns:
            ParserResult
            
        Raises:
            ParserError: Parse başarısız olursa
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
            # 1. Encoding detection
            encoding = self._detect_encoding(file_path)
            self.logger.info(f"Detected encoding: {encoding}")
            
            # 2. İçerik okuma
            text = self._read_file(file_path, encoding)
            
            # 3. Normalization
            if self.normalize_whitespace:
                text = self._normalize_whitespace(text)
            
            # 4. Metadata extraction
            metadata = self._extract_metadata(file_path, text, encoding)
            
            # 5. Quality checks
            metadata.update(self._quality_checks(text))
            
            return ParserResult(
                text=text,
                metadata=metadata,
                success=True
            )
            
        except EncodingError as e:
            self.logger.error(f"Encoding error: {e}")
            return ParserResult(
                text="",
                metadata={"error_type": "encoding"},
                success=False,
                error=str(e)
            )
        except Exception as e:
            self.logger.error(f"Parse error: {e}", exc_info=True)
            return ParserResult(
                text="",
                metadata={"error_type": "unknown"},
                success=False,
                error=str(e)
            )
    
    def _detect_encoding(self, file_path: Path) -> str:
        """
        Dosya encoding'ini tespit et.
        
        Önce UTF-8 dener, başarısız olursa chardet ile detect eder.
        
        Args:
            file_path: Dosya path'i
            
        Returns:
            Encoding adı (örn: 'utf-8', 'latin-1')
            
        Raises:
            EncodingError: Encoding tespit edilemezse
        """
        # Önce UTF-8 dene
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read()
            return 'utf-8'
        except UnicodeDecodeError:
            pass
        
        # chardet ile detect et
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        
        self.logger.info(
            f"Encoding detection: {encoding} (confidence: {confidence:.2f})"
        )
        
        # Güven kontrolü
        if confidence < self.CONFIDENCE_THRESHOLD:
            self.logger.warning(
                f"Low confidence encoding detection: {confidence:.2f}"
            )
        
        if not encoding:
            raise EncodingError("Could not detect file encoding")
        
        return encoding
    
    def _read_file(self, file_path: Path, encoding: str) -> str:
        """
        Dosyayı oku.
        
        Args:
            file_path: Dosya path'i
            encoding: Encoding
            
        Returns:
            Dosya içeriği
            
        Raises:
            EncodingError: Okuma başarısız olursa
        """
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                return f.read()
        except Exception as e:
            raise EncodingError(f"Failed to read file with encoding {encoding}: {e}")
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Whitespace'leri normalize et.
        
        - Multiple spaces → single space
        - Multiple newlines → double newline (paragraph break)
        - Trailing/leading whitespace removal
        
        Args:
            text: Orijinal metin
            
        Returns:
            Normalize edilmiş metin
        """
        import re
        
        # Multiple spaces → single space
        text = re.sub(r' +', ' ', text)
        
        # Multiple newlines → max 2 newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def _extract_metadata(
        self,
        file_path: Path,
        text: str,
        encoding: str
    ) -> Dict[str, Any]:
        """
        Metadata çıkar.
        
        Args:
            file_path: Dosya path'i
            text: Parse edilmiş metin
            encoding: Kullanılan encoding
            
        Returns:
            Metadata dictionary
        """
        lines = text.splitlines()
        words = text.split()
        
        metadata: Dict[str, Any] = {
            # Parser info
            "parser_name": self.PARSER_NAME,
            "parser_version": self.PARSER_VERSION,
            
            # File info
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "encoding": encoding,
            
            # Content stats
            "char_count": len(text),
            "word_count": len(words),
            "line_count": len(lines),
            "byte_size": len(text.encode('utf-8')),
            
            # Content type
            "content_type": "markdown" if file_path.suffix in ['.md', '.markdown'] else "text",
        }
        
        # Markdown için ek metadata
        if metadata["content_type"] == "markdown":
            metadata["has_markdown"] = self._detect_markdown_elements(text)
        
        return metadata
    
    def _detect_markdown_elements(self, text: str) -> Dict[str, bool]:
        """
        Markdown elementlerini tespit et.
        
        Args:
            text: Metin
            
        Returns:
            Markdown element varlık bilgisi
        """
        import re
        
        return {
            "has_headers": bool(re.search(r'^#+\s', text, re.MULTILINE)),
            "has_lists": bool(re.search(r'^[\*\-\+]\s', text, re.MULTILINE)),
            "has_code_blocks": bool(re.search(r'```', text)),
            "has_links": bool(re.search(r'\[.+\]\(.+\)', text)),
            "has_images": bool(re.search(r'!\[.+\]\(.+\)', text)),
        }
    
    def _quality_checks(self, text: str) -> Dict[str, Any]:
        """
        Kalite kontrolleri.
        
        Args:
            text: Metin
            
        Returns:
            Kalite bilgisi
        """
        quality: Dict[str, Any] = {
            "is_empty": len(text.strip()) == 0,
            "is_too_short": len(text) < 10,
            "has_repeated_chars": self._has_excessive_repetition(text),
        }
        
        # Quality score hesapla (0.0-1.0)
        score = 1.0
        if quality["is_empty"]:
            score = 0.0
        elif quality["is_too_short"]:
            score = 0.3
        elif quality["has_repeated_chars"]:
            score = 0.5
        
        quality["quality_score"] = score
        
        return quality
    
    def _has_excessive_repetition(self, text: str, threshold: int = 50) -> bool:
        """
        Aşırı karakter tekrarı var mı?
        
        Örneğin: "aaaaaaa..." (spam indicator)
        
        Args:
            text: Metin
            threshold: Tekrar eşiği
            
        Returns:
            Aşırı tekrar varsa True
        """
        import re
        
        # Aynı karakterin 50+ kez tekrarı
        pattern = r'(.)\1{' + str(threshold) + ',}'
        return bool(re.search(pattern, text))


class TXTParser(TextParser):
    """TXT dosyaları için özelleştirilmiş parser"""
    
    PARSER_NAME = "TXTParser"
    SUPPORTED_EXTENSIONS = [".txt", ".text"]


class MDParser(TextParser):
    """Markdown dosyaları için özelleştirilmiş parser"""
    
    PARSER_NAME = "MDParser"
    SUPPORTED_EXTENSIONS = [".md", ".markdown"]
