"""
Base Parser Sınıfı

Tüm parser'ların türeyeceği abstract base class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ParserResult:
    """
    Parser sonucu.
    
    Her parser bu yapıda sonuç döner.
    """
    
    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None
    ):
        self.text = text
        self.metadata = metadata
        self.success = success
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e dönüştür"""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error
        }


class BaseParser(ABC):
    """
    Tüm parser'ların base class'ı.
    
    Her dosya tipi için ayrı bir parser implementasyonu olmalıdır.
    BaseParser'dan türeyerek parse() metodunu implement etmelidir.
    """
    
    # Parser adı (alt sınıflar override etmeli)
    PARSER_NAME: str = "base"
    
    # Parser versiyonu (semantic versioning)
    PARSER_VERSION: str = "1.0.0"
    
    # Desteklenen file extension'lar
    SUPPORTED_EXTENSIONS: list[str] = []
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def parse(self, file_path: Path) -> ParserResult:
        """
        Dosyayı parse et.
        
        Bu metod her parser tarafından implement edilmelidir.
        
        Args:
            file_path: Parse edilecek dosyanın path'i
            
        Returns:
            ParserResult: Parse sonucu
            
        Raises:
            ParserError: Parse işlemi başarısız olursa
        """
        pass
    
    def validate(self, file_path: Path) -> bool:
        """
        Dosya parse edilebilir mi kontrol et.
        
        Args:
            file_path: Kontrol edilecek dosya
            
        Returns:
            Geçerli ise True
        """
        # Dosya var mı?
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False
        
        # Dosya okunabilir mi?
        if not file_path.is_file():
            self.logger.error(f"Not a file: {file_path}")
            return False
        
        # Extension destekleniyor mu?
        if self.SUPPORTED_EXTENSIONS:
            ext = file_path.suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                self.logger.error(
                    f"Unsupported extension: {ext}. "
                    f"Supported: {self.SUPPORTED_EXTENSIONS}"
                )
                return False
        
        return True
    
    def get_parser_info(self) -> Dict[str, str]:
        """
        Parser bilgilerini döndür.
        
        Returns:
            Parser name ve version
        """
        return {
            "parser_name": self.PARSER_NAME,
            "parser_version": self.PARSER_VERSION
        }


class ParserError(Exception):
    """Parser işleminde hata"""
    pass


class EncodingError(ParserError):
    """Encoding hatası"""
    pass


class ContentExtractionError(ParserError):
    """İçerik çıkarma hatası"""
    pass
