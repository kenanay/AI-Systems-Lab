"""
Utility Fonksiyonları

ID üretimi, MIME type detection, vb. yardımcı fonksiyonlar.
"""

import uuid
import mimetypes
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# MIME type mapping (fallback için)
MIME_TYPE_MAPPING = {
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.pdf': 'application/pdf',
    '.csv': 'text/csv',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.json': 'application/json',
    '.jsonl': 'application/jsonl',
    '.xml': 'application/xml',
    '.html': 'text/html',
}


def generate_file_id(prefix: str = "FILE") -> str:
    """
    Unique file ID üret.
    
    Format: PREFIX-XXXXXXXX
    Örnek: FILE-12345678
    
    Args:
        prefix: ID prefix'i
        
    Returns:
        Unique identifier
    """
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def generate_document_id(prefix: str = "DOC") -> str:
    """
    Unique document ID üret.
    
    Format: PREFIX-XXXXXXXX
    Örnek: DOC-12345678
    
    Args:
        prefix: ID prefix'i
        
    Returns:
        Unique identifier
    """
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def generate_job_id(prefix: str = "JOB") -> str:
    """
    Unique job ID üret.
    
    Format: PREFIX-XXXXXXXX
    Örnek: JOB-12345678
    
    Args:
        prefix: ID prefix'i
        
    Returns:
        Unique identifier
    """
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def detect_mime_type(filename: str, content: Optional[bytes] = None) -> str:
    """
    Dosya MIME type'ını tespit et.
    
    Önce extension'a bakar, bulamazsa content'e bakar.
    
    Args:
        filename: Dosya adı
        content: Dosya içeriği (opsiyonel)
        
    Returns:
        MIME type string
    """
    # Extension'dan tahmin et
    file_ext = Path(filename).suffix.lower()
    
    # Önce manuel mapping'e bak
    if file_ext in MIME_TYPE_MAPPING:
        return MIME_TYPE_MAPPING[file_ext]
    
    # Python'ın mimetypes modülünü kullan
    mime_type, _ = mimetypes.guess_type(filename)
    
    if mime_type:
        return mime_type
    
    # Content-based detection (opsiyonel - magic library gerekir)
    # if content:
    #     import magic
    #     return magic.from_buffer(content, mime=True)
    
    # Fallback
    logger.warning(f"Could not detect MIME type for {filename}, using application/octet-stream")
    return "application/octet-stream"


def get_file_extension(filename: str) -> str:
    """
    Dosya extension'ını al.
    
    Args:
        filename: Dosya adı
        
    Returns:
        Extension (lowercase, with dot)
    """
    return Path(filename).suffix.lower()


def is_allowed_file(filename: str, allowed_extensions: list[str]) -> bool:
    """
    Dosya izin verilen extension'lardan mı kontrol et.
    
    Args:
        filename: Dosya adı
        allowed_extensions: İzin verilen extension'lar
        
    Returns:
        İzin veriliyorsa True
    """
    ext = get_file_extension(filename)
    return ext in allowed_extensions


def format_file_size(size_bytes: int) -> str:
    """
    Byte cinsinden dosya boyutunu okunabilir formata çevir.
    
    Args:
        size_bytes: Byte cinsinden boyut
        
    Returns:
        Formatted string (örn: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def count_words(text: str) -> int:
    """
    Metindeki kelime sayısını hesapla.
    
    Args:
        text: Metin
        
    Returns:
        Kelime sayısı
    """
    return len(text.split())


def count_lines(text: str) -> int:
    """
    Metindeki satır sayısını hesapla.
    
    Args:
        text: Metin
        
    Returns:
        Satır sayısı
    """
    return len(text.splitlines())


def detect_language(text: str) -> str:
    """
    Basit dil tespiti.
    
    TODO: Daha gelişmiş bir library kullanılabilir (langdetect, fasttext)
    
    Args:
        text: Metin
        
    Returns:
        Dil kodu (ISO 639-1)
    """
    # Placeholder implementation
    # Gerçek implementation için langdetect veya benzeri kullanılmalı
    
    # Basit Türkçe karakter kontrolü
    turkish_chars = ['ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü']
    turkish_char_count = sum(1 for char in text if char in turkish_chars)
    
    if turkish_char_count > 5:
        return "tr"
    
    return "en"  # Default
