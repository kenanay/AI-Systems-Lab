"""
Database Modelleri

FileRecord ve DocumentRecord için SQLAlchemy modelleri.
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional

from backend.database import Base


class FileRecord(Base):
    """
    Ham dosya kaydı.
    
    Her yüklenen dosya için bir FileRecord oluşturulur.
    Orijinal dosya bilgileri ve metadata burada saklanır.
    """
    __tablename__ = "files"
    
    # Primary Key
    file_id = Column(String(50), primary_key=True, index=True)
    
    # File Information
    original_name = Column(String(255), nullable=False)
    relative_path = Column(String(500), nullable=False)  # datasets/raw/ içinde path
    mime_type = Column(String(100), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    
    # Hash and Integrity
    sha256 = Column(String(64), unique=True, nullable=False, index=True)
    
    # Processing Information
    parser_name = Column(String(100))  # Hangi parser kullanıldı
    parser_version = Column(String(20))  # Parser versiyonu
    schema_version = Column(String(20), default="1.0.0")  # Data schema versiyonu
    
    # Security and Privacy
    security_level = Column(
        String(20),
        default="INTERNAL"
    )  # PUBLIC, INTERNAL, RESTRICTED, PERSONAL
    pii_detected = Column(Boolean, default=False)  # PII var mı?
    license = Column(String(100))  # Lisans bilgisi
    copyright_status = Column(String(100))  # Telif durumu
    training_allowed = Column(Boolean, default=False)  # Eğitimde kullanılabilir mi?
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    modified_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Dataset Version
    dataset_version = Column(String(20))  # Hangi dataset versiyonuna ait
    
    # Source Information
    source = Column(String(255))  # Dosya kaynağı
    language = Column(String(10))  # Ana dil (tr, en, etc.)
    
    # Quality
    quality_score = Column(Float)  # 0.0-1.0 arası kalite skoru
    
    # Relationships
    documents = relationship("DocumentRecord", back_populates="file", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<FileRecord(file_id={self.file_id}, name={self.original_name})>"


class DocumentRecord(Base):
    """
    Parse edilmiş doküman içeriği.
    
    Bir FileRecord'dan çıkarılan normalize edilmiş içerik.
    Canonical dataset formatının temel birimi.
    """
    __tablename__ = "documents"
    
    # Primary Key
    document_id = Column(String(50), primary_key=True, index=True)
    
    # Foreign Key
    file_id = Column(String(50), ForeignKey("files.file_id"), nullable=False, index=True)
    
    # Content
    title = Column(String(500))  # Doküman başlığı
    text = Column(Text, nullable=False)  # Ana metin içeriği
    
    # Metadata
    language = Column(String(10))  # Dil
    char_count = Column(Integer)  # Karakter sayısı
    word_count = Column(Integer)  # Kelime sayısı
    line_count = Column(Integer)  # Satır sayısı
    
    # Processing Info
    parser_name = Column(String(100))
    parser_version = Column(String(20))
    schema_version = Column(String(20), default="1.0.0")
    
    # Quality Indicators
    is_empty = Column(Boolean, default=False)  # Boş mu?
    is_duplicate = Column(Boolean, default=False)  # Duplicate mı?
    quality_score = Column(Float)  # Kalite skoru
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    file = relationship("FileRecord", back_populates="documents")
    
    def __repr__(self) -> str:
        return f"<DocumentRecord(document_id={self.document_id}, title={self.title})>"


class ProcessingJob(Base):
    """
    Uzun süren processing işleri için job tracking.
    
    Future: Ingestion, tokenizer training, model training için kullanılabilir.
    """
    __tablename__ = "processing_jobs"
    
    # Primary Key
    job_id = Column(String(50), primary_key=True, index=True)
    
    # Job Information
    job_type = Column(String(50), nullable=False)  # "ingestion", "tokenizer_training", etc.
    status = Column(
        String(20),
        default="PENDING"
    )  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    
    # Progress
    progress = Column(Float, default=0.0)  # 0.0-1.0
    total_items = Column(Integer)  # Toplam işlenecek item
    processed_items = Column(Integer, default=0)  # İşlenen item sayısı
    
    # Error Information
    error_message = Column(Text)
    error_traceback = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Job Metadata (JSON olarak saklanabilir)
    job_metadata = Column(Text)  # JSON string
    
    def __repr__(self) -> str:
        return f"<ProcessingJob(job_id={self.job_id}, type={self.job_type}, status={self.status})>"
