"""
Database Modelleri

FileRecord, DocumentRecord ve TokenizerJob için SQLAlchemy modelleri.
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional
import uuid

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



class TokenizerJob(Base):
    """
    Tokenizer training job tracking.
    
    BPE tokenizer training uzun sürebilir, bu yüzden job-based
    architecture kullanılır.
    
    Job States:
        PENDING: Job oluşturuldu, henüz başlamadı
        RUNNING: Training devam ediyor
        PAUSED: Geçici olarak durduruldu
        COMPLETED: Başarıyla tamamlandı
        FAILED: Hata ile sonuçlandı
        CANCELLED: İptal edildi
    """
    __tablename__ = "tokenizer_jobs"
    
    # Primary Key
    job_id = Column(String(50), primary_key=True, index=True, default=lambda: str(uuid.uuid4())[:8].upper())
    
    # Job Information
    job_name = Column(String(255), nullable=False)
    status = Column(String(20), default="PENDING", nullable=False, index=True)
    
    # Configuration (JSON)
    # {
    #   "dataset_ids": ["DS-XXX", ...],
    #   "file_ids": ["FILE-XXX", ...],
    #   "vocab_size": 8000,
    #   "min_frequency": 2,
    #   "special_tokens": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    # }
    config = Column(JSON, nullable=False)
    
    # Progress (0.0 - 1.0)
    progress = Column(Float, default=0.0)
    
    # Result Metadata (JSON) - Training sonuçları
    # {
    #   "tokenizer_id": "TOK-XXX",
    #   "vocab_size": 8000,
    #   "num_merges": 7500,
    #   "num_documents": 1000,
    #   "training_duration_seconds": 45.2,
    #   "output_path": "/path/to/tokenizer"
    # }
    result_metadata = Column(JSON)
    
    # Error Information
    error = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    def __repr__(self) -> str:
        return f"<TokenizerJob(job_id={self.job_id}, name={self.job_name}, status={self.status})>"


class TokenizerRecord(Base):
    """
    Trained tokenizer metadata.
    
    Her trained tokenizer için bir record oluşturulur.
    Tokenizer dosyaları disk'te saklanır, bu model metadata'yı tutar.
    
    Data Lineage:
        - Hangi dataset/file'lardan train edildi
        - Training job referansı
        - Training parametreleri
    """
    __tablename__ = "tokenizers"
    
    # Primary Key
    tokenizer_id = Column(String(50), primary_key=True, index=True)
    
    # Tokenizer Information
    name = Column(String(255), nullable=False)
    tokenizer_type = Column(String(50), default="BPE", nullable=False)  # BPE, WordPiece, etc.
    version = Column(String(20), default="1.0.0")
    
    # Vocabulary Stats
    vocab_size = Column(Integer, nullable=False)
    num_merges = Column(Integer)  # BPE için merge count
    
    # Special Tokens
    special_tokens = Column(JSON)  # ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    
    # Storage
    storage_path = Column(String(500), nullable=False)  # Vocab files'ın yolu
    
    # Training Configuration
    training_config = Column(JSON)  # Training parameters
    # {
    #   "vocab_size": 8000,
    #   "min_frequency": 2,
    #   "dataset_ids": [...],
    #   "file_ids": [...]
    # }
    
    # Data Lineage
    source_job_id = Column(String(50), ForeignKey("tokenizer_jobs.job_id"))  # Training job
    source_dataset_ids = Column(JSON)  # Kullanılan dataset'ler
    source_file_ids = Column(JSON)  # Kullanılan file'lar
    num_training_documents = Column(Integer)  # Training'de kullanılan doküman sayısı
    
    # Training Stats
    training_duration_seconds = Column(Float)
    training_completed_at = Column(DateTime)
    
    # Usage Stats
    usage_count = Column(Integer, default=0)  # Kaç kez kullanıldı
    last_used_at = Column(DateTime)
    
    # Status
    is_active = Column(Boolean, default=True)  # Aktif mi?
    is_deprecated = Column(Boolean, default=False)  # Deprecated mi?
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    description = Column(Text)  # Tokenizer açıklaması
    tags = Column(JSON)  # ["turkish", "general", "v1"]
    
    def __repr__(self) -> str:
        return f"<TokenizerRecord(tokenizer_id={self.tokenizer_id}, name={self.name}, vocab_size={self.vocab_size})>"


# Type alias for model imports
Document = DocumentRecord
