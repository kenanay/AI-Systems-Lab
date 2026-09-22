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
    file_id: Mapped[str] = mapped_column(String(50), primary_key=True, index=True)
    
    # File Information
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(500), nullable=False)  # datasets/raw/ içinde path
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Hash and Integrity
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    
    # Processing Information
    parser_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # Hangi parser kullanıldı
    parser_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # Parser versiyonu
    schema_version: Mapped[str] = mapped_column(String(20), default="1.0.0")  # Data schema versiyonu
    
    # Security and Privacy
    security_level: Mapped[str] = mapped_column(
        String(20),
        default="INTERNAL"
    )  # PUBLIC, INTERNAL, RESTRICTED, PERSONAL
    pii_detected: Mapped[bool] = mapped_column(Boolean, default=False)  # PII var mı?
    license: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # Lisans bilgisi
    copyright_status: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # Telif durumu
    training_allowed: Mapped[bool] = mapped_column(Boolean, default=False)  # Eğitimde kullanılabilir mi?
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    modified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=utc_now, onupdate=utc_now, nullable=True)
    
    # Dataset Version
    dataset_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # Hangi dataset versiyonuna ait
    
    # Source Information
    source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Dosya kaynağı
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # Ana dil (tr, en, etc.)
    
    # Quality
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0.0-1.0 arası kalite skoru
    
    # Relationships
    documents: Mapped[List["DocumentRecord"]] = relationship("DocumentRecord", back_populates="file", cascade="all, delete-orphan")
    
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
    document_id: Mapped[str] = mapped_column(String(50), primary_key=True, index=True)
    
    # Foreign Key
    file_id: Mapped[str] = mapped_column(String(50), ForeignKey("files.file_id"), nullable=False, index=True)
    
    # Content
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # Doküman başlığı
    text: Mapped[str] = mapped_column(Text, nullable=False)  # Ana metin içeriği
    
    # Metadata
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # Dil
    char_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Karakter sayısı
    word_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Kelime sayısı
    line_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Satır sayısı
    
    # Processing Info
    parser_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    parser_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    
    # Quality Indicators
    is_empty: Mapped[bool] = mapped_column(Boolean, default=False)  # Boş mu?
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)  # Duplicate mı?
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Kalite skoru
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    
    # Relationships
    file: Mapped["FileRecord"] = relationship("FileRecord", back_populates="documents")
    
    def __repr__(self) -> str:
        return f"<DocumentRecord(document_id={self.document_id}, title={self.title})>"


class ProcessingJob(Base):
    """
    Uzun süren processing işleri için job tracking.
    
    Future: Ingestion, tokenizer training, model training için kullanılabilir.
    """
    __tablename__ = "processing_jobs"
    
    # Primary Key
    job_id: Mapped[str] = mapped_column(String(50), primary_key=True, index=True)
    
    # Job Information
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "ingestion", "tokenizer_training", etc.
    status: Mapped[str] = mapped_column(
        String(20),
        default="PENDING"
    )  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    
    # Progress
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0-1.0
    total_items: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Toplam işlenecek item
    processed_items: Mapped[int] = mapped_column(Integer, default=0)  # İşlenen item sayısı
    
    # Error Information
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Job Metadata (JSON olarak saklanabilir)
    job_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string
    
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
    job_id: Mapped[str] = mapped_column(String(50), primary_key=True, index=True, default=lambda: str(uuid.uuid4())[:8].upper())
    
    # Job Information
    job_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", nullable=False, index=True)
    
    # Configuration (JSON)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Progress (0.0 - 1.0)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Result Metadata (JSON) - Training sonuçları
    result_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Error Information
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
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
    tokenizer_id: Mapped[str] = mapped_column(String(50), primary_key=True, index=True)
    
    # Tokenizer Information
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    tokenizer_type: Mapped[str] = mapped_column(String(50), default="BPE", nullable=False)  # BPE, WordPiece, etc.
    version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    
    # Vocabulary Stats
    vocab_size: Mapped[int] = mapped_column(Integer, nullable=False)
    num_merges: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # BPE için merge count
    
    # Special Tokens
    special_tokens: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)  # ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    
    # Storage
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)  # Vocab files'ın yolu
    
    # Training Configuration
    training_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)  # Training parameters
    
    # Data Lineage
    source_job_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("tokenizer_jobs.job_id"), nullable=True)  # Training job
    source_dataset_ids: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)  # Kullanılan dataset'ler
    source_file_ids: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)  # Kullanılan file'lar
    num_training_documents: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Training'de kullanılan doküman sayısı
    
    # Training Stats
    training_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    training_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Usage Stats
    usage_count: Mapped[int] = mapped_column(Integer, default=0)  # Kaç kez kullanıldı
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)  # Aktif mi?
    is_deprecated: Mapped[bool] = mapped_column(Boolean, default=False)  # Deprecated mi?
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=utc_now, onupdate=utc_now, nullable=True)
    
    # Metadata
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Tokenizer açıklaması
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)  # ["turkish", "general", "v1"]
    
    @property
    def model_path(self) -> str:
        """Alias for storage_path for backward compatibility."""
        return str(getattr(self, "storage_path", "") or "")
    
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
    dataset_id: Mapped[str] = mapped_column(String(50), primary_key=True, index=True, default=lambda: f"DS-{str(uuid.uuid4())[:8].upper()}")
    
    # Dataset Information
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)  # Semantic versioning: "1.0.0"
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Schema & Compiler
    schema_version: Mapped[str] = mapped_column(String(20), default="1.0.0", nullable=False)
    compiler_version: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Storage
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)  # Parquet file path
    metadata_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # metadata.json path
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Data Lineage
    source_document_ids: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)  # ["DOC-XXX", ...]
    source_file_ids: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)  # ["FILE-XXX", ...]
    tokenizer_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("tokenizers.tokenizer_id"), nullable=True)
    
    # Compilation Job Reference
    compilation_job_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # Job that created this dataset
    
    # Statistics
    num_documents: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_chars: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Compilation Parameters (JSON)
    compilation_params: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Filter Statistics (JSON)
    filter_stats: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)  # Aktif mi?
    is_snapshot: Mapped[bool] = mapped_column(Boolean, default=False)  # Snapshot mi yoksa draft mi?
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    compiled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)  # Compilation tamamlanma zamanı
    
    # Tags & Metadata
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)  # ["turkish", "general", "v1"]
    custom_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)  # Kullanıcı tanımlı metadata
    
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
    job_id: Mapped[str] = mapped_column(String(50), primary_key=True, index=True, default=lambda: str(uuid.uuid4())[:8].upper())
    
    # Job Information
    job_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", nullable=False, index=True)
    
    # Configuration (JSON)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Progress (0.0 - 1.0)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Result Metadata (JSON) - Compilation sonuçları
    result_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Error Information
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
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
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else None,
            "metrics": list(metrics_val or [])
        }

    def __repr__(self) -> str:
        return f"<TrainingJob(job_id={self.job_id}, name={self.job_name}, type={self.job_type}, status={self.status})>"


class BenchmarkRecord(Base):
    """
    Model Benchmark & Evaluation Sonuç Kaydı.
    
    Perplexity, BLEU, ROUGE ve Accuracy değerlendirme sonuçlarını kalıcı olarak saklar.
    """
    __tablename__ = "benchmark_results"
    
    benchmark_id: Mapped[str] = mapped_column(
        String(50), primary_key=True, index=True, default=lambda: f"BENCH-{uuid.uuid4().hex[:8].upper()}"
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    benchmark_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    samples_evaluated: Mapped[int] = mapped_column(Integer, default=0)
    dataset_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark record to dictionary with concrete types."""
        ds_path = getattr(self, "dataset_path", None)
        c_at = getattr(self, "created_at", None)
        c_at_str = c_at.isoformat() if isinstance(c_at, datetime) else str(c_at or "")
        return {
            "benchmark_id": str(getattr(self, "benchmark_id", "")),
            "model_name": str(getattr(self, "model_name", "")),
            "benchmark_name": str(getattr(self, "benchmark_name", "")),
            "score": float(getattr(self, "score", 0.0)),
            "metrics": dict(getattr(self, "metrics", {}) or {}),
            "samples_evaluated": int(getattr(self, "samples_evaluated", 0)),
            "dataset_path": str(ds_path) if ds_path else None,
            "created_at": c_at_str,
        }

    def __repr__(self) -> str:
        return f"<BenchmarkRecord(id={self.benchmark_id}, model={self.model_name}, bench={self.benchmark_name}, score={self.score})>"


# Type alias for model imports
Document = DocumentRecord

