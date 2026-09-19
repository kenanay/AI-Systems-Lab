"""
Text Normalization

Parse edilmiş metni normalize etme ve temizleme.
"""

import re
from typing import Optional
import unicodedata
import logging

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Metin normalization işlemleri.
    
    Özellikler:
    - Unicode normalization
    - Whitespace normalization
    - Diacritic handling
    - Control character removal
    - URL/email masking (opsiyonel)
    """
    
    def __init__(
        self,
        unicode_form: str = "NFC",
        normalize_whitespace: bool = True,
        remove_control_chars: bool = True,
        preserve_paragraphs: bool = True
    ):
        """
        Args:
            unicode_form: Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')
            normalize_whitespace: Whitespace'leri normalize et
            remove_control_chars: Kontrol karakterlerini kaldır
            preserve_paragraphs: Paragraph break'leri koru
        """
        self.unicode_form = unicode_form
        self.normalize_whitespace = normalize_whitespace
        self.remove_control_chars = remove_control_chars
        self.preserve_paragraphs = preserve_paragraphs
    
    def normalize(self, text: str) -> str:
        """
        Metni normalize et.
        
        Normalization pipeline:
        1. Unicode normalization
        2. Control character removal
        3. Whitespace normalization
        
        Args:
            text: Orijinal metin
            
        Returns:
            Normalize edilmiş metin
        """
        if not text:
            return ""
        
        # 1. Unicode normalization
        text = self._normalize_unicode(text)
        
        # 2. Control character removal
        if self.remove_control_chars:
            text = self._remove_control_chars(text)
        
        # 3. Whitespace normalization
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Unicode normalization.
        
        Args:
            text: Orijinal metin
            
        Returns:
            Normalize edilmiş metin
        """
        return unicodedata.normalize(self.unicode_form, text)
    
    def _remove_control_chars(self, text: str) -> str:
        """
        Kontrol karakterlerini kaldır.
        
        Newline ve tab karakterleri korunur.
        
        Args:
            text: Metin
            
        Returns:
            Temizlenmiş metin
        """
        # Kontrol karakterleri (\x00-\x1f ve \x7f-\x9f) ancak \n, \r, \t hariç
        return "".join(
            char for char in text
            if unicodedata.category(char)[0] != "C" or char in ['\n', '\r', '\t']
        )
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Whitespace normalization.
        
        - Multiple spaces → single space
        - Multiple newlines → max 2 newlines (paragraph break)
        - Tab → space
        - Trailing/leading whitespace removal per line
        
        Args:
            text: Metin
            
        Returns:
            Normalize edilmiş metin
        """
        # Tab → space
        text = text.replace('\t', ' ')
        
        # CRLF → LF
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        
        # Multiple spaces → single space
        text = re.sub(r' +', ' ', text)
        
        if self.preserve_paragraphs:
            # Multiple newlines → max 2 newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
        else:
            # Multiple newlines → single newline
            text = re.sub(r'\n+', '\n', text)
        
        # Her satırın başındaki ve sonundaki boşlukları kaldır
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        return text


class TextCleaner:
    """
    Metin temizleme işlemleri.
    
    Özellikler:
    - Repeated character removal
    - Empty line removal
    - Header/footer pattern removal
    - URL/email extraction veya masking
    """
    
    def __init__(
        self,
        remove_repeated_chars: bool = True,
        repeated_char_threshold: int = 50,
        remove_empty_lines: bool = False,
        remove_urls: bool = False,
        remove_emails: bool = False
    ):
        """
        Args:
            remove_repeated_chars: Aşırı tekrarlı karakterleri temizle
            repeated_char_threshold: Tekrar eşiği
            remove_empty_lines: Boş satırları kaldır
            remove_urls: URL'leri kaldır
            remove_emails: Email'leri kaldır
        """
        self.remove_repeated_chars = remove_repeated_chars
        self.repeated_char_threshold = repeated_char_threshold
        self.remove_empty_lines = remove_empty_lines
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
    
    def clean(self, text: str) -> str:
        """
        Metni temizle.
        
        Args:
            text: Orijinal metin
            
        Returns:
            Temizlenmiş metin
        """
        if not text:
            return ""
        
        # Repeated characters
        if self.remove_repeated_chars:
            text = self._remove_repeated_chars(text)
        
        # URLs
        if self.remove_urls:
            text = self._remove_urls(text)
        
        # Emails
        if self.remove_emails:
            text = self._remove_emails(text)
        
        # Empty lines
        if self.remove_empty_lines:
            text = self._remove_empty_lines(text)
        
        return text.strip()
    
    def _remove_repeated_chars(self, text: str) -> str:
        """
        Aşırı tekrarlı karakterleri kaldır.
        
        Örnek: "aaaaaaaa..." → "aaa"
        
        Args:
            text: Metin
            
        Returns:
            Temizlenmiş metin
        """
        # Aynı karakterin threshold+ tekrarı → 3 tekrar
        pattern = r'(.)\1{' + str(self.repeated_char_threshold) + ',}'
        return re.sub(pattern, r'\1\1\1', text)
    
    def _remove_urls(self, text: str) -> str:
        """
        URL'leri kaldır veya maskele.
        
        Args:
            text: Metin
            
        Returns:
            Temizlenmiş metin
        """
        # Basit URL pattern
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '[URL]', text)
    
    def _remove_emails(self, text: str) -> str:
        """
        Email adreslerini kaldır veya maskele.
        
        Args:
            text: Metin
            
        Returns:
            Temizlenmiş metin
        """
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '[EMAIL]', text)
    
    def _remove_empty_lines(self, text: str) -> str:
        """
        Boş satırları kaldır.
        
        Args:
            text: Metin
            
        Returns:
            Temizlenmiş metin
        """
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        return '\n'.join(non_empty_lines)


class LanguageDetector:
    """
    Basit dil tespiti.
    
    TODO: Daha gelişmiş bir library kullan (langdetect, fasttext)
    """
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Metinden dil tespit et.
        
        Args:
            text: Metin
            
        Returns:
            Dil kodu (ISO 639-1)
        """
        if not text:
            return "unknown"
        
        # Basit Türkçe karakter kontrolü
        turkish_chars = set('çğıöşüÇĞİÖŞÜ')
        text_chars = set(text)
        turkish_char_count = len(text_chars & turkish_chars)
        
        if turkish_char_count > 5:
            return "tr"
        
        # Basit İngilizce kontrolü (default)
        return "en"


class QualityScorer:
    """
    Metin kalite skorlama.
    
    Metin kalitesini 0.0-1.0 arasında skorlar.
    """
    
    @staticmethod
    def score(text: str, metadata: dict) -> float:
        """
        Metin kalite skoru hesapla.
        
        Kriterler:
        - Uzunluk (çok kısa → düşük skor)
        - Karakter çeşitliliği
        - Tekrar oranı
        - Boş içerik
        
        Args:
            text: Metin
            metadata: Parse metadata
            
        Returns:
            Kalite skoru (0.0-1.0)
        """
        if not text or not text.strip():
            return 0.0
        
        score = 1.0
        
        # 1. Uzunluk kontrolü
        char_count = len(text)
        if char_count < 10:
            return 0.1
        elif char_count < 50:
            score *= 0.5
        elif char_count < 100:
            score *= 0.7
        
        # 2. Karakter çeşitliliği
        unique_chars = len(set(text))
        total_chars = len(text)
        diversity_ratio = unique_chars / total_chars if total_chars > 0 else 0
        
        if diversity_ratio < 0.1:  # Çok düşük çeşitlilik
            score *= 0.3
        elif diversity_ratio < 0.2:
            score *= 0.6
        
        # 3. Kelime sayısı
        word_count = len(text.split())
        if word_count < 5:
            score *= 0.4
        elif word_count < 10:
            score *= 0.7
        
        # 4. Metadata'dan gelen ek kontroller
        if metadata.get("is_empty", False):
            return 0.0
        
        if metadata.get("has_repeated_chars", False):
            score *= 0.5
        
        if metadata.get("is_encrypted", False):
            score *= 0.3
        
        if metadata.get("likely_image_based", False):
            score *= 0.4
        
        return max(0.0, min(1.0, score))


# Pipeline class
class NormalizationPipeline:
    """
    Tam normalization pipeline.
    
    Normalizer → Cleaner → Language Detection → Quality Scoring
    """
    
    def __init__(
        self,
        normalizer: Optional[TextNormalizer] = None,
        cleaner: Optional[TextCleaner] = None
    ):
        """
        Args:
            normalizer: TextNormalizer instance
            cleaner: TextCleaner instance
        """
        self.normalizer = normalizer or TextNormalizer()
        self.cleaner = cleaner or TextCleaner()
        self.language_detector = LanguageDetector()
        self.quality_scorer = QualityScorer()
    
    def process(self, text: str, metadata: dict) -> tuple[str, dict]:
        """
        Metni pipeline'dan geçir.
        
        Args:
            text: Orijinal metin
            metadata: Parse metadata
            
        Returns:
            Tuple[normalize edilmiş metin, güncellenmiş metadata]
        """
        # 1. Normalize
        normalized_text = self.normalizer.normalize(text)
        
        # 2. Clean
        cleaned_text = self.cleaner.clean(normalized_text)
        
        # 3. Language detection
        language = self.language_detector.detect_language(cleaned_text)
        
        # 4. Quality scoring
        quality_score = self.quality_scorer.score(cleaned_text, metadata)
        
        # Metadata güncelle
        updated_metadata = {
            **metadata,
            "language": language,
            "quality_score": quality_score,
            "normalized": True,
            "char_count_after_normalization": len(cleaned_text),
            "word_count_after_normalization": len(cleaned_text.split()),
        }
        
        return cleaned_text, updated_metadata
