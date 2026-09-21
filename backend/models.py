"""
Database Modelleri

FileRecord, DocumentRecord, TokenizerJob, TokenizerRecord, 
DatasetVersion ve CompilationJob için SQLAlchemy modelleri.
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import uuid

from backend.database import Base


def utc_now() -> datetime:
    """Returns current UTC datetime."""
    return datetime.now(timezone.utc)


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
    created_at = Column(DateTime, default=utc_now, nullable=False)
    modified_at = Column(DateTime, default=utc_now, onupdate=utc_now)
    
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
    created_at = Column(DateTime, default=utc_now, nullable=False)
    
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
    created_at = Column(DateTime, default=utc_now, nullable=False)
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
    created_at = Column(DateTime, default=utc_now, nullable=False)
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
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)
    
    # Metadata
    description = Column(Text)  # Tokenizer açıklaması
    tags = Column(JSON)  # ["turkish", "general", "v1"]
    
    def __repr__(self) -> str:
        return f"<TokenizerRecord(tokenizer_id={self.tokenizer_id}, name={self.name}, vocab_size={self.vocab_size})>"


class DatasetVersion(Base):
    """
    Compiled Dataset Version tracking.
    
    Her compiled dataset için bir version record oluşturulur.
    Dataset'ler immutable'dır - yeni versiyon oluşturulur, mevcut değiştirilmez.
    
    Versioning Strategy:
        - Semantic versioning: MAJOR.MINOR.PATCH
        - MAJOR: Breaking schema changes
        - MINOR: New features, backward compatible
        - PATCH: Bug fixes, re-compilation
    
    Data Lineage:
        - source_document_ids: Hangi documents kullanıldı
        - source_file_ids: Hangi files kullanıldı
        - tokenizer_id: Hangi tokenizer ile tokenize edildi
        - compiler_version: Hangi compiler versiyonu kullanıldı
    """
    __tablename__ = "dataset_versions"
    
    # Primary Key
    dataset_id = Column(String(50), primary_key=True, index=True, default=lambda: f"DS-{str(uuid.uuid4())[:8].upper()}")
    
    # Dataset Information
    name = Column(String(255), nullable=False)
    version = Column(String(20), nullable=False)  # Semantic versioning: "1.0.0"
    description = Column(Text)
    
    # Schema & Compiler
    schema_version = Column(String(20), default="1.0.0", nullable=False)
    compiler_version = Column(String(20), nullable=False)
    
    # Storage
    storage_path = Column(String(500), nullable=False)  # Parquet file path
    metadata_path = Column(String(500))  # metadata.json path
    file_size_bytes = Column(Integer)
    
    # Data Lineage
    source_document_ids = Column(JSON)  # ["DOC-XXX", ...]
    source_file_ids = Column(JSON)  # ["FILE-XXX", ...]
    tokenizer_id = Column(String(50), ForeignKey("tokenizers.tokenizer_id"))
    
    # Compilation Job Reference
    compilation_job_id = Column(String(50))  # Job that created this dataset
    
    # Statistics
    num_documents = Column(Integer, nullable=False)
    total_tokens = Column(Integer)
    total_chars = Column(Integer)
    
    # Compilation Parameters (JSON)
    # {
    #   "min_quality_score": 0.5,
    #   "max_quality_score": 1.0,
    #   "min_length": 10,
    #   "max_length": 100000,
    #   "allow_pii": false,
    #   "require_training_allowed": true,
    #   "remove_duplicates": true
    # }
    compilation_params = Column(JSON)
    
    # Filter Statistics (JSON)
    # {
    #   "total_documents": 1000,
    #   "filtered_by_quality": 50,
    #   "filtered_by_license": 20,
    #   "filtered_by_pii": 10,
    #   "filtered_by_length": 30,
    #   "duplicates_removed": 40,
    #   "final_count": 850
    # }
    filter_stats = Column(JSON)
    
    # Status
    is_active = Column(Boolean, default=True)  # Aktif mi?
    is_snapshot = Column(Boolean, default=False)  # Snapshot mi yoksa draft mi?
    
    # Timestamps
    created_at = Column(DateTime, default=utc_now, nullable=False)
    compiled_at = Column(DateTime)  # Compilation tamamlanma zamanı
    
    # Tags & Metadata
    tags = Column(JSON)  # ["turkish", "general", "v1"]
    custom_metadata = Column(JSON)  # Kullanıcı tanımlı metadata
    
    def __repr__(self) -> str:
        return f"<DatasetVersion(dataset_id={self.dataset_id}, name={self.name}, version={self.version})>"


class CompilationJob(Base):
    """
    Dataset Compilation Job tracking.
    
    Dataset compilation uzun sürebilir, bu yüzden job-based architecture.
    
    Job States:
        PENDING: Job oluşturuldu
        RUNNING: Compilation devam ediyor
        COMPLETED: Başarıyla tamamlandı
        FAILED: Hata oluştu
        CANCELLED: İptal edildi
    """
    __tablename__ = "compilation_jobs"
    
    # Primary Key
    job_id = Column(String(50), primary_key=True, index=True, default=lambda: str(uuid.uuid4())[:8].upper())
    
    # Job Information
    job_name = Column(String(255), nullable=False)
    status = Column(String(20), default="PENDING", nullable=False, index=True)
    
    # Configuration (JSON)
    # {
    #   "document_ids": ["DOC-XXX", ...],
    #   "tokenizer_id": "TOK-XXX",
    #   "dataset_name": "Training Dataset v1",
    #   "dataset_version": "1.0.0",
    #   "compilation_params": {...}
    # }
    config = Column(JSON, nullable=False)
    
    # Progress (0.0 - 1.0)
    progress = Column(Float, default=0.0)
    
    # Result Metadata (JSON) - Compilation sonuçları
    # {
    #   "dataset_id": "DS-XXX",
    #   "output_path": "/path/to/dataset",
    #   "stats": {...}
    # }
    result_metadata = Column(JSON)
    
    # Error Information
    error = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=utc_now, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    def __repr__(self) -> str:
        return f"<CompilationJob(job_id={self.job_id}, name={self.job_name}, status={self.status})>"


class TrainingJob(Base):
    """
    Model Training Job tracking (Pre-training and SFT/LoRA).
    """
    __tablename__ = "training_jobs"
    
    job_id: Mapped[str] = mapped_column(String(50), primary_key=True, index=True, default=lambda: str(uuid.uuid4())[:8].upper())
    job_name: Mapped[str] = mapped_column(String(255), nullable=False)
    job_type: Mapped[str] = mapped_column(String(50), default="PRETRAIN", nullable=False)  # PRETRAIN, SFT, SFT_LORA
    status: Mapped[str] = mapped_column(String(20), default="PENDING", nullable=False, index=True)  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    
    # References
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    tokenizer_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Configuration
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Progress & Metrics
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, default=1)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    
    # History of metrics for charting: list of {step, epoch, loss, val_loss, perplexity, lr}
    metrics: Mapped[List[Any]] = mapped_column(JSON, default=list)
    
    # Checkpoint & Artifacts
    output_dir: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    best_checkpoint: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Error
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job record to dictionary with concrete types."""
        progress_val: Any = getattr(self, "progress", 0.0)
        epoch_val: Any = getattr(self, "current_epoch", 0)
        total_epochs_val: Any = getattr(self, "total_epochs", 1)
        step_val: Any = getattr(self, "current_step", 0)
        total_steps_val: Any = getattr(self, "total_steps", 0)
        metrics_val: Any = getattr(self, "metrics", [])

        return {
            "job_id": str(getattr(self, "job_id", "")),
            "job_name": str(getattr(self, "job_name", "")),
            "job_type": str(getattr(self, "job_type", "")),
            "status": str(getattr(self, "status", "")),
            "model_name": str(getattr(self, "model_name", "")),
            "dataset_id": str(self.dataset_id) if self.dataset_id else None,
            "tokenizer_id": str(self.tokenizer_id) if self.tokenizer_id else None,
            "progress": float(progress_val or 0.0),
            "current_epoch": int(epoch_val or 0),
            "total_epochs": int(total_epochs_val or 1),
            "current_step": int(step_val or 0),
            "total_steps": int(total_steps_val or 0),
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": str(self.error) if self.error else None,
            "metrics": list(metrics_val or [])
        }

    def __repr__(self) -> str:
        return f"<TrainingJob(job_id={self.job_id}, name={self.job_name}, type={self.job_type}, status={self.status})>"


# Type alias for model imports
Document = DocumentRecord

