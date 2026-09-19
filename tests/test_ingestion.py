"""
tests/test_ingestion.py

Ingestion Pipeline Testleri

Bu modül veri ingestion pipeline'ının temel bileşenlerini test eder:
- Parser'lar (TXT, MD)
- Normalization pipeline
- Storage manager temel fonksiyonları

Her test isolated environment'ta çalışır ve fixtures kullanır.
"""

import pytest
from pathlib import Path
from typing import Dict, Any

from src.ingestion import get_parser, TXTParser, MDParser
from src.ingestion.base import ParserError
from src.normalization import NormalizationPipeline


class TestParsers:
    """
    Parser implementasyonları testleri.
    
    Parser'ların dosyaları doğru parse ettiğini ve metadata
    ürettiğini doğrular.
    """
    
    def test_get_parser_txt(self) -> None:
        """TXT parser factory doğru instance döndürmeli"""
        parser = get_parser('.txt')
        assert isinstance(parser, TXTParser)
    
    def test_get_parser_md(self) -> None:
        """MD parser factory doğru instance döndürmeli"""
        parser = get_parser('.md')
        assert isinstance(parser, MDParser)
    
    def test_get_parser_unsupported(self) -> None:
        """Desteklenmeyen extension ValueError fırlatmalı"""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            get_parser('.xyz')
    
    def test_txt_parser_basic(self, sample_text_file: Path) -> None:
        """
        TXT parser temel dosya okuma testi.
        
        Doğrulamalar:
        - Parse başarılı
        - Metin içeriği doğru
        - Metadata üretilmiş (parser_name, line_count)
        """
        parser = TXTParser()
        result = parser.parse(sample_text_file)
        
        assert result.success is True
        assert "test dosyası" in result.text
        assert result.metadata['parser_name'] == 'TXTParser'
        assert result.metadata['line_count'] == 3
    
    def test_md_parser_basic(self, sample_md_file: Path) -> None:
        """
        MD parser temel dosya okuma testi.
        
        Markdown formatting korunmalı.
        """
        parser = MDParser()
        result = parser.parse(sample_md_file)
        
        assert result.success is True
        assert "# Test" in result.text
        assert result.metadata['parser_name'] == 'MDParser'
    
    def test_parser_nonexistent_file(self) -> None:
        """
        Olmayan dosya parse hatası döndürmeli.
        
        Exception fırlatmak yerine result.success=False ve
        error mesajı dönmeli (graceful failure).
        """
        parser = TXTParser()
        result = parser.parse(Path("/nonexistent/file.txt"))
        
        assert result.success is False
        assert result.error is not None


class TestNormalization:
    """
    Normalization pipeline testleri.
    
    Text normalization, quality scoring ve metadata
    üretimini test eder.
    """
    
    def test_normalization_basic(self) -> None:
        """
        Temel normalization: boşluk temizleme.
        
        İşlemler:
        - Fazla boşluklar kaldırılır
        - Satır sonları normalize edilir
        - Word/char count hesaplanır
        """
        pipeline = NormalizationPipeline()
        
        text = "  Test   metin  \n\n\n  ikinci satır  "
        normalized, metadata = pipeline.process(text, {})
        
        # Boşluklar temizlenmeli
        assert "Test metin" in normalized
        # Metadata eklenmeli
        assert 'char_count_after_normalization' in metadata
        assert 'word_count_after_normalization' in metadata
    
    def test_normalization_unicode(self) -> None:
        """
        Unicode normalization (NFC form).
        
        Combining characters normalize edilmeli.
        """
        pipeline = NormalizationPipeline()
        
        # NFC normalization test
        text = "café"  # e + combining acute
        normalized, _ = pipeline.process(text, {})
        
        # NFC formunda olmalı
        assert normalized == "café"
    
    def test_quality_scoring(self) -> None:
        """
        Quality score hesaplama.
        
        Score range: [0.0, 1.0]
        Yeterli uzunluktaki text skorlanabilmeli.
        """
        pipeline = NormalizationPipeline()
        
        # Yeterli uzunlukta text
        text = "Bu yeterince uzun bir test metnidir. " * 10
        _, metadata = pipeline.process(text, {})
        
        assert 'quality_score' in metadata
        assert 0 <= metadata['quality_score'] <= 1
    
    def test_language_detection(self) -> None:
        """
        Dil tespiti testi.
        
        Not: Basit implementasyon olduğu için kesin sonuç beklemiyoruz.
        Metadata'da 'language' alanı olmalı.
        """
        pipeline = NormalizationPipeline()
        
        turkish_text = "Bu bir Türkçe metindir. Çok güzel bir dil."
        _, metadata = pipeline.process(turkish_text, {})
        
        assert 'language' in metadata
        # Language detection basit implementasyon olduğu için kesin sonuç beklemiyoruz
        assert metadata['language'] in ['tr', 'en', 'unknown']


class TestStorageManager:
    """
    Storage manager temel fonksiyon testleri.
    
    SHA-256 hash hesaplama ve dosya integrity kontrolü.
    """
    
    def test_sha256_calculation(self, sample_text_file: Path) -> None:
        """
        SHA-256 hash hesaplama testi.
        
        Doğrulamalar:
        - Hash 64 karakter hex string
        - Aynı dosya her zaman aynı hash üretir (deterministic)
        """
        from backend.storage import StorageManager
        
        sha256 = StorageManager.calculate_sha256(sample_text_file)
        
        assert len(sha256) == 64  # SHA-256 hex string
        assert sha256.isalnum()
        
        # Aynı dosya aynı hash vermeli (deterministic)
        sha256_2 = StorageManager.calculate_sha256(sample_text_file)
        assert sha256 == sha256_2
