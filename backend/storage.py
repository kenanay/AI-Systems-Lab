"""
Dosya Depolama Yönetimi

Dosyaların fiziksel olarak saklanması ve erişimi.
"""

import hashlib
import shutil
from pathlib import Path
from typing import BinaryIO, Tuple
import logging
from datetime import datetime

from backend.config import settings

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Dosya depolama işlemlerini yönetir.
    
    Sorumluluklar:
    - Dosyaları güvenli biçimde kaydetme
    - SHA-256 hash hesaplama
    - Duplicate detection
    - Dosya silme
    """
    
    def __init__(self):
        self.raw_path = settings.raw_data_path
        self.temp_path = settings.temp_upload_dir
        
        # Dizinleri oluştur
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        """
        Dosyanın SHA-256 hash'ini hesapla.
        
        Args:
            file_path: Dosya yolu
            
        Returns:
            Hexadecimal SHA-256 hash string
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Büyük dosyalar için chunk'lar halinde oku
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    @staticmethod
    def calculate_sha256_from_stream(file_stream: BinaryIO) -> str:
        """
        Stream'den SHA-256 hash hesapla.
        
        Args:
            file_stream: Binary file stream
            
        Returns:
            Hexadecimal SHA-256 hash string
        """
        sha256_hash = hashlib.sha256()
        
        # Stream başlangıca al
        file_stream.seek(0)
        
        for chunk in iter(lambda: file_stream.read(8192), b""):
            sha256_hash.update(chunk)
        
        # Stream'i tekrar başa al
        file_stream.seek(0)
        
        return sha256_hash.hexdigest()
    
    def save_file(
        self,
        file_stream: BinaryIO,
        original_filename: str,
        file_id: str
    ) -> Tuple[Path, str]:
        """
        Dosyayı raw/ dizinine kaydet ve SHA-256 hesapla.
        
        Dosya organizasyonu:
        raw/
          YYYY/
            MM/
              DD/
                {file_id}_{original_filename}
        
        Args:
            file_stream: Dosya stream'i
            original_filename: Orijinal dosya adı
            file_id: Unique file identifier
            
        Returns:
            Tuple[relative_path, sha256_hash]
        """
        # Tarih bazlı dizin yapısı
        now = datetime.utcnow()
        date_path = Path(f"{now.year:04d}/{now.month:02d}/{now.day:02d}")
        target_dir = self.raw_path / date_path
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Dosya adı: file_id + original_filename
        safe_filename = self._sanitize_filename(original_filename)
        filename = f"{file_id}_{safe_filename}"
        target_path = target_dir / filename
        
        # Dosyayı kaydet
        logger.info(f"Saving file to: {target_path}")
        with open(target_path, "wb") as f:
            shutil.copyfileobj(file_stream, f)
        
        # SHA-256 hesapla
        sha256 = self.calculate_sha256(target_path)
        logger.info(f"File saved. SHA-256: {sha256}")
        
        # Relative path (raw_data_path'e göre)
        relative_path = target_path.relative_to(self.raw_path)
        
        return relative_path, sha256
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Dosya adını güvenli hale getir.
        
        Args:
            filename: Orijinal dosya adı
            
        Returns:
            Sanitize edilmiş dosya adı
        """
        # Tehlikeli karakterleri temizle
        dangerous_chars = ['/', '\\', '..', '\0']
        safe_name = filename
        
        for char in dangerous_chars:
            safe_name = safe_name.replace(char, '_')
        
        # Maksimum uzunluk sınırı
        if len(safe_name) > 200:
            # Extension'ı koru
            name_parts = safe_name.rsplit('.', 1)
            if len(name_parts) == 2:
                name, ext = name_parts
                safe_name = name[:195] + '.' + ext
            else:
                safe_name = safe_name[:200]
        
        return safe_name
    
    def get_file_path(self, relative_path: str) -> Path:
        """
        Relative path'ten absolute path oluştur.
        
        Args:
            relative_path: Dataset root'a göre relative path
            
        Returns:
            Absolute file path
        """
        return settings.data_root / relative_path
    
    def delete_file(self, relative_path: str) -> bool:
        """
        Dosyayı sil.
        
        Args:
            relative_path: Silinecek dosyanın relative path'i
            
        Returns:
            Silme başarılı ise True
        """
        try:
            file_path = self.get_file_path(relative_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"File deleted: {relative_path}")
                return True
            else:
                logger.warning(f"File not found: {relative_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting file {relative_path}: {e}")
            return False
    
    def file_exists(self, relative_path: str) -> bool:
        """
        Dosya var mı kontrol et.
        
        Args:
            relative_path: Kontrol edilecek dosyanın path'i
            
        Returns:
            Dosya varsa True
        """
        return self.get_file_path(relative_path).exists()


# Global storage manager instance
storage_manager = StorageManager()
