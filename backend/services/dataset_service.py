"""
backend/services/dataset_service.py

Dataset Compilation Service

Bu modül dataset compilation job'larını yönetir:
- Job-based compilation (background execution)
- Progress tracking
- Dataset version management
- Document selection & filtering
- Tokenizer integration

Job-based architecture kullanılır çünkü compilation uzun sürebilir.
"""

from typing import List, Dict, Optional, Any, Callable
from pathlib import Path
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import logging

from backend.models import (
    DocumentRecord,
    FileRecord,
    TokenizerRecord,
    DatasetVersion,
    CompilationJob
)
from backend.config import settings
from src.dataset.compiler import DatasetCompiler
from src.dataset.versioning import suggest_next_version, get_latest_version
from src.tokenizer.bpe import BPETokenizer

logger = logging.getLogger(__name__)


class DatasetCompilationService:
    """
    Dataset compilation service.
    
    Documents'ları seçip tokenize edip training-ready dataset'e compile eder.
    Job-based execution ile progress tracking sağlar.
    
    Attributes:
        db: Database session
        output_base_dir: Compiled dataset'lerin ana dizini
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.output_base_dir = Path(settings.data_root) / "compiled_datasets"
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_compilation_job(
        self,
        job_name: str,
        dataset_name: str,
        dataset_version: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        file_ids: Optional[List[str]] = None,
        tokenizer_id: Optional[str] = None,
        compilation_params: Optional[Dict[str, Any]] = None
    ) -> CompilationJob:
        """
        Dataset compilation job oluştur.
        
        Args:
            job_name: Job ismi
            dataset_name: Dataset ismi
            dataset_version: Dataset version (None ise auto-increment)
            document_ids: Kullanılacak document ID'leri (opsiyonel)
            file_ids: Kullanılacak file ID'leri (opsiyonel)
            tokenizer_id: Tokenizer ID (zorunlu)
            compilation_params: Compilation parametreleri
                - min_quality_score: float (default: 0.5)
                - max_quality_score: float (default: 1.0)
                - min_length: int (default: 10)
                - max_length: int (default: 100000)
                - allow_pii: bool (default: False)
                - require_training_allowed: bool (default: True)
                - remove_duplicates: bool (default: True)
                
        Returns:
            CompilationJob: Oluşturulan job
            
        Raises:
            ValueError: tokenizer_id veya document/file selection eksikse
        """
        if not tokenizer_id:
            raise ValueError("tokenizer_id gerekli")
        
        if not document_ids and not file_ids:
            raise ValueError("En az document_ids veya file_ids gerekli")
        
        # Tokenizer var mı kontrol et
        tokenizer = self.db.query(TokenizerRecord).filter(
            TokenizerRecord.tokenizer_id == tokenizer_id
        ).first()
        
        if not tokenizer:
            raise ValueError(f"Tokenizer bulunamadı: {tokenizer_id}")
        
        # Dataset version auto-increment
        if not dataset_version:
            # Aynı isimli dataset'lerin version'larını al
            existing_versions = self.db.query(DatasetVersion.version).filter(
                DatasetVersion.name == dataset_name
            ).all()
            
            if existing_versions:
                version_strings = [str(v[0]) for v in existing_versions if v[0] is not None]
                latest = get_latest_version(version_strings)
                if latest:
                    # PATCH bump (yeni data compilation)
                    dataset_version, _ = suggest_next_version(latest, schema_changed=False, params_changed=False)
                else:
                    dataset_version = "1.0.0"
            else:
                dataset_version = "1.0.0"
        
        # Default compilation params
        default_params = {
            "min_quality_score": 0.5,
            "max_quality_score": 1.0,
            "min_length": 10,
            "max_length": 100000,
            "allow_pii": False,
            "require_training_allowed": True,
            "remove_duplicates": True
        }
        
        if compilation_params:
            default_params.update(compilation_params)
        
        # Job configuration
        config = {
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "document_ids": document_ids or [],
            "file_ids": file_ids or [],
            "tokenizer_id": tokenizer_id,
            "compilation_params": default_params
        }
        
        # Job oluştur
        job = CompilationJob(
            job_name=job_name,
            status="PENDING",
            config=config,
            progress=0.0,
            created_at=datetime.now(timezone.utc)
        )
        
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        
        logger.info(f"Compilation job created: {job.job_id} - {job_name}")
        return job
    
    def run_compilation_job(
        self,
        job_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Compilation job'ı çalıştır.
        
        Akış:
        1. Job'ı RUNNING durumuna al
        2. Documents'ları topla
        3. Tokenizer yükle
        4. DatasetCompiler ile compile et
        5. DatasetVersion record oluştur
        6. Job'ı COMPLETED durumuna al
        
        Args:
            job_id: Job ID
            progress_callback: Progress callback function(current, total)
            
        Returns:
            Dict: Compilation sonucu
                - dataset_id: str
                - output_path: str
                - stats: Dict
                
        Raises:
            ValueError: Job bulunamazsa veya geçersiz durumdaysa
        """
        # Job'ı al
        job = self.db.query(CompilationJob).filter(
            CompilationJob.job_id == job_id
        ).first()
        
        if not job:
            raise ValueError(f"Job bulunamadı: {job_id}")
        
        current_status = str(job.status)
        if current_status not in ["PENDING", "PAUSED"]:
            raise ValueError(f"Job geçersiz durumda: {current_status}")
        
        try:
            # Job başlat
            job.status = "RUNNING"
            job.started_at = datetime.now(timezone.utc)
            job.progress = 0.0
            self.db.commit()
            
            logger.info(f"Compilation job başladı: {job_id}")
            
            job_cfg = getattr(job, "config", {})
            config: Dict[str, Any] = dict(job_cfg) if isinstance(job_cfg, dict) else {}
            
            # Step 1: Documents'ları topla
            documents = self._collect_documents(config)
            
            if not documents:
                raise ValueError("Compilation için document bulunamadı")
            
            logger.info(f"Document collection: {len(documents)} documents")
            
            job.progress = 0.1
            self.db.commit()
            
            # Step 2: Tokenizer yükle
            tokenizer_id_val = str(config.get("tokenizer_id", ""))
            tokenizer = self._load_tokenizer(tokenizer_id_val)
            
            job.progress = 0.15
            self.db.commit()
            
            # Step 3: Output directory oluştur
            dataset_name_val = str(config.get("dataset_name", "dataset"))
            dataset_version_val = str(config.get("dataset_version", "1.0.0"))
            output_dir = self.output_base_dir / f"{dataset_name_val}_v{dataset_version_val}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 4: Compiler oluştur ve compile et
            compiler = DatasetCompiler(
                output_dir=output_dir,
                tokenizer=tokenizer,
                dataset_version=dataset_version_val,
                db_session=self.db
            )
            
            # Progress callback wrapper
            def compilation_progress(current: int, total: int):
                # 0.15 - 0.9 arası progress (0.15 setup, 0.9 compilation)
                progress = 0.15 + (current / total) * 0.75
                if job is not None:
                    job.progress = progress
                    self.db.commit()
                
                if progress_callback:
                    progress_callback(current, total)
            
            # Compile
            params_raw = config.get("compilation_params", {})
            params: Dict[str, Any] = dict(params_raw) if isinstance(params_raw, dict) else {}
            result = compiler.compile_dataset(
                documents=documents,
                min_quality_score=params.get("min_quality_score", 0.5),
                max_quality_score=params.get("max_quality_score", 1.0),
                min_length=params.get("min_length", 10),
                max_length=params.get("max_length", 100000),
                allow_pii=params.get("allow_pii", False),
                mask_pii=params.get("mask_pii", False),
                require_training_allowed=params.get("require_training_allowed", True),
                remove_duplicates=params.get("remove_duplicates", True),
                use_minhash=params.get("use_minhash", True),
                minhash_threshold=params.get("minhash_threshold", 0.85),
                progress_callback=compilation_progress
            )
            
            job.progress = 0.9
            self.db.commit()
            
            # Step 5: DatasetVersion record oluştur
            dataset_version = self._create_dataset_version(
                config=config,
                result=result,
                job_id=job_id,
                documents=documents
            )
            
            # Step 6: Job tamamla
            job.status = "COMPLETED"
            job.completed_at = datetime.now(timezone.utc)
            job.progress = 1.0
            
            job.result_metadata = {
                "dataset_id": str(dataset_version.dataset_id),
                "output_path": str(result["output_path"]),
                "metadata_path": str(result["metadata_path"]),
                "stats": result["stats"]
            }
            
            self.db.commit()
            
            logger.info(f"Compilation job tamamlandı: {job_id} - {dataset_version.dataset_id}")
            
            return {
                "dataset_id": dataset_version.dataset_id,
                "output_path": result["output_path"],
                "stats": result["stats"]
            }
            
        except Exception as e:
            # Hata durumu
            job.status = "FAILED"
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc)
            self.db.commit()
            
            logger.error(f"Compilation job failed: {job_id} - {e}")
            raise
    
    def _collect_documents(self, config: Dict[str, Any]) -> List[DocumentRecord]:
        """
        Configuration'a göre documents'ları topla.
        
        Args:
            config: Job configuration
                - document_ids: List[str]
                - file_ids: List[str]
                
        Returns:
            List[DocumentRecord]: Document listesi
        """
        documents = []
        
        # Document ID'lerden topla
        document_ids = config.get("document_ids", [])
        if document_ids:
            docs = self.db.query(DocumentRecord).filter(
                DocumentRecord.document_id.in_(document_ids)
            ).all()
            documents.extend(docs)
        
        # File ID'lerden topla
        file_ids = config.get("file_ids", [])
        if file_ids:
            docs = self.db.query(DocumentRecord).filter(
                DocumentRecord.file_id.in_(file_ids)
            ).all()
            documents.extend(docs)
        
        # Duplicate document ID'leri kaldır
        seen_ids = set()
        unique_docs = []
        for doc in documents:
            if doc.document_id not in seen_ids:
                seen_ids.add(doc.document_id)
                unique_docs.append(doc)
        
        logger.info(f"Collected {len(unique_docs)} unique documents")
        return unique_docs
    
    def _load_tokenizer(self, tokenizer_id: str) -> BPETokenizer:
        """
        Tokenizer'ı disk'ten yükle.
        
        Args:
            tokenizer_id: Tokenizer ID
            
        Returns:
            BPETokenizer: Yüklenmiş tokenizer
            
        Raises:
            FileNotFoundError: Tokenizer dosyaları bulunamazsa
        """
        tokenizer_record = self.db.query(TokenizerRecord).filter(
            TokenizerRecord.tokenizer_id == tokenizer_id
        ).first()
        
        if not tokenizer_record:
            raise ValueError(f"Tokenizer record bulunamadı: {tokenizer_id}")
        
        storage_path = str(tokenizer_record.storage_path)
        tokenizer = BPETokenizer()
        tokenizer.load_vocab(Path(storage_path))
        
        logger.info(f"Tokenizer loaded: {tokenizer_id}")
        return tokenizer
    
    def _create_dataset_version(
        self,
        config: Dict[str, Any],
        result: Dict[str, Any],
        job_id: str,
        documents: List[DocumentRecord]
    ) -> DatasetVersion:
        """
        DatasetVersion record oluştur.
        
        Args:
            config: Job configuration
            result: Compilation result
            job_id: Job ID
            documents: Source documents
            
        Returns:
            DatasetVersion: Oluşturulan dataset version
        """
        # Storage path bilgileri
        output_path = Path(result["output_path"])
        metadata_path = Path(result["metadata_path"])
        
        # Source IDs
        import json
        import pyarrow.parquet as pq
        from src.tokenizer.loading import artifact_hash
        rows = pq.read_table(output_path, columns=["document_id", "file_id"]).to_pylist()
        source_doc_ids = [row["document_id"] for row in rows]
        source_file_ids = list({row["file_id"] for row in rows})
        tok = self.db.query(TokenizerRecord).filter(TokenizerRecord.tokenizer_id == config["tokenizer_id"]).one()
        fingerprints = {"dataset_sha256": artifact_hash(output_path), "tokenizer_sha256": artifact_hash(tok.storage_path), "split_strategy": "content_sha256_80_10_10"}
        
        # File size
        file_size = output_path.stat().st_size if output_path.exists() else 0
        
        # Stats
        stats = result["stats"]
        
        dataset_version = DatasetVersion(
            name=config["dataset_name"],
            version=config["dataset_version"],
            schema_version=result["schema_version"],
            compiler_version=result["compiler_version"],
            storage_path=str(output_path),
            metadata_path=str(metadata_path),
            file_size_bytes=file_size,
            source_document_ids=source_doc_ids,
            source_file_ids=source_file_ids,
            tokenizer_id=config["tokenizer_id"],
            compilation_job_id=job_id,
            num_documents=stats["final_count"],
            total_tokens=stats.get("total_tokens", 0),
            total_chars=sum(doc.char_count or 0 for doc in documents),
            compilation_params=config["compilation_params"],
            filter_stats=stats,
            is_active=True,
            custom_metadata=fingerprints,
            is_snapshot=True,
            compiled_at=datetime.now(timezone.utc)
        )
        
        self.db.add(dataset_version)
        self.db.commit()
        self.db.refresh(dataset_version)
        
        logger.info(f"DatasetVersion created: {dataset_version.dataset_id}")
        return dataset_version
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Job durumunu sorgula.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dict: Job durumu
        """
        job = self.db.query(CompilationJob).filter(
            CompilationJob.job_id == job_id
        ).first()
        
        if not job:
            return None
        
        return {
            "job_id": job.job_id,
            "job_name": job.job_name,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "result_metadata": job.result_metadata or {},
            "error": job.error
        }
    
    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Job'ları listele.
        
        Args:
            status: Filter by status (opsiyonel)
            limit: Maximum sonuç sayısı
            
        Returns:
            List[Dict]: Job listesi
        """
        query = self.db.query(CompilationJob)
        
        if status:
            query = query.filter(CompilationJob.status == status)
        
        jobs = query.order_by(CompilationJob.created_at.desc()).limit(limit).all()
        
        return [
            {
                "job_id": job.job_id,
                "job_name": job.job_name,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at,
                "result_metadata": job.result_metadata or {}
            }
            for job in jobs
        ]
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Running job'ı iptal et.
        
        Args:
            job_id: Job ID
            
        Returns:
            bool: İptal başarılı mı
        """
        job = self.db.query(CompilationJob).filter(
            CompilationJob.job_id == job_id
        ).first()
        
        if not job:
            return False
        
        current_status = str(job.status)
        if current_status in ["RUNNING", "PENDING"]:
            job.status = "CANCELLED"
            job.completed_at = datetime.now(timezone.utc)
            self.db.commit()
            
            logger.info(f"Job cancelled: {job_id}")
            return True
        
        return False
    
    def list_dataset_versions(
        self,
        name: Optional[str] = None,
        is_active: Optional[bool] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Dataset version'larını listele.
        
        Args:
            name: Filter by dataset name (opsiyonel)
            is_active: Filter by active status (opsiyonel)
            limit: Maximum sonuç sayısı
            
        Returns:
            List[Dict]: Dataset version listesi
        """
        query = self.db.query(DatasetVersion)
        
        if name:
            query = query.filter(DatasetVersion.name == name)
        
        if is_active is not None:
            query = query.filter(DatasetVersion.is_active == is_active)
        
        versions = query.order_by(DatasetVersion.created_at.desc()).limit(limit).all()
        
        return [
            {
                "dataset_id": ds.dataset_id,
                "name": ds.name,
                "version": ds.version,
                "num_documents": ds.num_documents,
                "total_tokens": ds.total_tokens,
                "file_size_bytes": ds.file_size_bytes,
                "is_active": ds.is_active,
                "compiled_at": ds.compiled_at,
                "tags": ds.tags
            }
            for ds in versions
        ]
    
    def get_dataset_version(self, dataset_id: str) -> Optional[Dict]:
        """
        Dataset version detayını getir.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dict: Dataset version detayı
        """
        dataset = self.db.query(DatasetVersion).filter(
            DatasetVersion.dataset_id == dataset_id
        ).first()
        
        if not dataset:
            return None
        
        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "version": dataset.version,
            "description": dataset.description,
            "schema_version": dataset.schema_version,
            "compiler_version": dataset.compiler_version,
            "storage_path": dataset.storage_path,
            "metadata_path": dataset.metadata_path,
            "file_size_bytes": dataset.file_size_bytes,
            "source_document_ids": dataset.source_document_ids,
            "source_file_ids": dataset.source_file_ids,
            "tokenizer_id": dataset.tokenizer_id,
            "compilation_job_id": dataset.compilation_job_id,
            "num_documents": dataset.num_documents,
            "total_tokens": dataset.total_tokens,
            "total_chars": dataset.total_chars,
            "compilation_params": dataset.compilation_params,
            "filter_stats": dataset.filter_stats,
            "is_active": dataset.is_active,
            "is_snapshot": dataset.is_snapshot,
            "created_at": dataset.created_at,
            "compiled_at": dataset.compiled_at,
            "tags": dataset.tags,
            "custom_metadata": dataset.custom_metadata
        }
