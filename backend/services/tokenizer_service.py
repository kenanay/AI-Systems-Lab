"""
backend/services/tokenizer_service.py

Tokenizer Training Service

Bu modül tokenizer training pipeline'ını yönetir:
- Dataset'ten text extraction
- BPE tokenizer training (job-based)
- Progress tracking
- Vocabulary storage

Job-based architecture kullanılır çünkü training uzun sürebilir.
"""

from typing import List, Dict, Optional, Callable
from pathlib import Path
from sqlalchemy.orm import Session
from datetime import datetime
import logging
import json

from src.tokenizer.bpe import BPETokenizer
from backend.models import FileRecord, Document, TokenizerJob, TokenizerRecord
from backend.config import settings

logger = logging.getLogger(__name__)


class TokenizerTrainingService:
    """
    Tokenizer training service.
    
    Dataset'ten text'leri toplayıp BPE tokenizer train eder.
    Progress tracking ve job management sağlar.
    
    Attributes:
        db: Database session
        output_dir: Tokenizer dosyalarının kaydedileceği dizin
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.output_dir = Path(settings.data_root) / "tokenizers"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_training_job(
        self,
        job_name: str,
        dataset_ids: Optional[List[str]] = None,
        file_ids: Optional[List[str]] = None,
        vocab_size: int = 8000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ) -> TokenizerJob:
        """
        Tokenizer training job oluştur.
        
        Args:
            job_name: Job ismi
            dataset_ids: Kullanılacak dataset ID'leri (opsiyonel)
            file_ids: Kullanılacak file ID'leri (opsiyonel)
            vocab_size: Hedef vocabulary boyutu
            min_frequency: Merge için minimum frequency
            special_tokens: Special tokens listesi
            
        Returns:
            TokenizerJob: Oluşturulan job
            
        Raises:
            ValueError: dataset_ids ve file_ids ikisi de None ise
        """
        if not dataset_ids and not file_ids:
            raise ValueError("En az bir dataset_id veya file_id gerekli")
        
        # Job configuration
        config = {
            "dataset_ids": dataset_ids or [],
            "file_ids": file_ids or [],
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "special_tokens": special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        }
        
        # Job oluştur
        job = TokenizerJob(
            job_name=job_name,
            status="PENDING",
            config=config,
            created_at=datetime.utcnow()
        )
        
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        
        logger.info(f"Tokenizer training job created: {job.job_id} - {job_name}")
        return job
    
    def run_training_job(
        self,
        job_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict:
        """
        Training job'ı çalıştır.
        
        Akış:
        1. Job'ı RUNNING durumuna al
        2. Dataset/file'lardan text'leri topla
        3. BPE tokenizer train et
        4. Vocabulary'yi kaydet
        5. Job'ı COMPLETED durumuna al
        
        Args:
            job_id: Job ID
            progress_callback: Progress callback function(current, total)
            
        Returns:
            Dict: Training sonucu
                - tokenizer_id: str
                - vocab_size: int
                - num_merges: int
                - training_duration_seconds: float
                
        Raises:
            ValueError: Job bulunamazsa veya geçersiz durumdaysa
        """
        # Job'ı al
        job = self.db.query(TokenizerJob).filter(
            TokenizerJob.job_id == job_id
        ).first()
        
        if not job:
            raise ValueError(f"Job bulunamadı: {job_id}")
        
        if job.status not in ["PENDING", "PAUSED"]:
            raise ValueError(f"Job geçersiz durumda: {job.status}")
        
        try:
            # Job başlat
            job.status = "RUNNING"
            job.started_at = datetime.utcnow()
            job.progress = 0.0
            self.db.commit()
            
            logger.info(f"Training job başladı: {job_id}")
            
            # Step 1: Text'leri topla
            texts = self._collect_texts(job.config)
            
            if not texts:
                raise ValueError("Training için text bulunamadı")
            
            logger.info(f"Text collection: {len(texts)} doküman")
            
            job.progress = 0.1
            job.result_metadata = {"num_documents": len(texts)}
            self.db.commit()
            
            # Step 2: Tokenizer oluştur ve train et
            tokenizer = BPETokenizer(
                vocab_size=job.config.get("vocab_size", 8000),
                special_tokens=job.config.get("special_tokens"),
                min_frequency=job.config.get("min_frequency", 2)
            )
            
            # Progress callback wrapper
            def training_progress(current: int, total: int):
                # 0.1 - 0.9 arası progress (0.1 text collection, 0.9 training)
                progress = 0.1 + (current / total) * 0.8
                job.progress = progress
                self.db.commit()
                
                if progress_callback:
                    progress_callback(current, total)
            
            # Train
            tokenizer.train(texts, progress_callback=training_progress)
            
            job.progress = 0.9
            self.db.commit()
            
            # Step 3: Vocabulary kaydet
            tokenizer_id = f"TOK-{job.job_id}"
            output_path = self.output_dir / tokenizer_id
            tokenizer.save_vocab(output_path)
            
            logger.info(f"Vocabulary saved: {output_path}")
            
            # Step 4: TokenizerRecord oluştur
            stats = tokenizer.get_vocab_stats()
            started = job.started_at or datetime.utcnow()
            duration = (datetime.utcnow() - started).total_seconds()
            
            tokenizer_record = TokenizerRecord(
                tokenizer_id=tokenizer_id,
                name=job.job_name,
                tokenizer_type="BPE",
                version=tokenizer.VERSION,
                vocab_size=stats["vocab_size"],
                num_merges=stats["num_merges"],
                special_tokens=job.config.get("special_tokens"),
                storage_path=str(output_path),
                training_config=job.config,
                source_job_id=job.job_id,
                source_dataset_ids=job.config.get("dataset_ids", []),
                source_file_ids=job.config.get("file_ids", []),
                num_training_documents=len(texts),
                training_duration_seconds=duration,
                training_completed_at=datetime.utcnow(),
                is_active=True
            )
            
            self.db.add(tokenizer_record)
            
            # Step 5: Job tamamla
            job.status = "COMPLETED"
            job.completed_at = datetime.utcnow()
            job.progress = 1.0
            
            res_meta = dict(job.result_metadata or {})
            res_meta.update({
                "tokenizer_id": tokenizer_id,
                "vocab_size": stats["vocab_size"],
                "num_merges": stats["num_merges"],
                "training_duration_seconds": duration,
                "output_path": str(output_path)
            })
            job.result_metadata = res_meta
            
            self.db.commit()
            
            logger.info(f"Training job tamamlandı: {job_id} - Duration: {duration:.2f}s")
            
            return {
                "tokenizer_id": tokenizer_id,
                "vocab_size": stats["vocab_size"],
                "num_merges": stats["num_merges"],
                "training_duration_seconds": duration
            }
            
        except Exception as e:
            # Hata durumu
            job.status = "FAILED"
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            self.db.commit()
            
            logger.error(f"Training job failed: {job_id} - {e}")
            raise
    
    def _collect_texts(self, config: Dict) -> List[str]:
        """
        Dataset/file'lardan text'leri topla.
        
        Args:
            config: Job configuration
                - dataset_ids: List[str]
                - file_ids: List[str]
                
        Returns:
            List[str]: Text listesi
        """
        texts = []
        
        # File ID'lerden text topla
        file_ids = config.get("file_ids", [])
        if file_ids:
            # Document'lerden text al
            documents = self.db.query(Document).filter(
                Document.file_id.in_(file_ids)
            ).all()
            
            for doc in documents:
                from src.dataset.splits import split_for_text
                if doc.text and getattr(doc, "file", None) and getattr(doc.file, "training_allowed", False) and split_for_text(doc.text) == "train":
                    texts.append(doc.text)
            
            logger.info(f"Collected {len(texts)} texts from {len(file_ids)} files")
        
        # Dataset ID'lerden text topla
        dataset_ids = config.get("dataset_ids", [])
        if dataset_ids:
            from backend.models import DatasetVersion
            import pyarrow.parquet as pq
            datasets = self.db.query(DatasetVersion).filter(
                DatasetVersion.dataset_id.in_(dataset_ids)
            ).all()
            for ds in datasets:
                ds_path = Path(str(ds.storage_path)) if ds.storage_path else None
                if ds_path and ds_path.exists():
                    try:
                        table = pq.read_table(str(ds_path))
                        if "text" in table.column_names:
                            ds_texts = [str(row["text"]) for row in table.to_pylist() if row.get("text") and row.get("split") == "train"]
                            texts.extend(ds_texts)
                            logger.info(f"Collected {len(ds_texts)} texts from dataset {ds.dataset_id}")
                    except Exception as e:
                        logger.warning(f"Could not read dataset {ds.dataset_id} parquet: {e}")
        
        return texts
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Job durumunu sorgula.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dict: Job durumu
                - job_id: str
                - job_name: str
                - status: str
                - progress: float
                - created_at: datetime
                - started_at: Optional[datetime]
                - completed_at: Optional[datetime]
                - metadata: Dict
                - error: Optional[str]
        """
        job = self.db.query(TokenizerJob).filter(
            TokenizerJob.job_id == job_id
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
            "metadata": job.result_metadata or {},
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
        query = self.db.query(TokenizerJob)
        
        if status:
            query = query.filter(TokenizerJob.status == status)
        
        jobs = query.order_by(TokenizerJob.created_at.desc()).limit(limit).all()
        
        return [
            {
                "job_id": job.job_id,
                "job_name": job.job_name,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at,
                "metadata": job.result_metadata or {}
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
        job = self.db.query(TokenizerJob).filter(
            TokenizerJob.job_id == job_id
        ).first()
        
        if not job:
            return False
        
        if job.status in ["RUNNING", "PENDING"]:
            job.status = "CANCELLED"
            job.completed_at = datetime.utcnow()
            self.db.commit()
            
            logger.info(f"Job cancelled: {job_id}")
            return True
        
        return False

    
    def get_tokenizer(self, tokenizer_id: str) -> Optional[Dict]:
        """
        Tokenizer metadata'sını getir.
        
        Args:
            tokenizer_id: Tokenizer ID
            
        Returns:
            Dict: Tokenizer metadata
        """
        tokenizer = self.db.query(TokenizerRecord).filter(
            TokenizerRecord.tokenizer_id == tokenizer_id
        ).first()
        
        if not tokenizer:
            return None
        
        return {
            "tokenizer_id": tokenizer.tokenizer_id,
            "name": tokenizer.name,
            "tokenizer_type": tokenizer.tokenizer_type,
            "version": tokenizer.version,
            "vocab_size": tokenizer.vocab_size,
            "num_merges": tokenizer.num_merges,
            "special_tokens": tokenizer.special_tokens,
            "storage_path": tokenizer.storage_path,
            "training_config": tokenizer.training_config,
            "source_job_id": tokenizer.source_job_id,
            "num_training_documents": tokenizer.num_training_documents,
            "training_duration_seconds": tokenizer.training_duration_seconds,
            "training_completed_at": tokenizer.training_completed_at,
            "usage_count": tokenizer.usage_count,
            "last_used_at": tokenizer.last_used_at,
            "is_active": tokenizer.is_active,
            "created_at": tokenizer.created_at,
            "description": tokenizer.description,
            "tags": tokenizer.tags
        }
    
    def list_tokenizers(
        self,
        is_active: Optional[bool] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Tokenizer'ları listele.
        
        Args:
            is_active: Sadece aktif tokenizer'ları getir (opsiyonel)
            limit: Maximum sonuç sayısı
            
        Returns:
            List[Dict]: Tokenizer listesi
        """
        query = self.db.query(TokenizerRecord)
        
        if is_active is not None:
            query = query.filter(TokenizerRecord.is_active == is_active)
        
        tokenizers = query.order_by(
            TokenizerRecord.created_at.desc()
        ).limit(limit).all()
        
        return [
            {
                "tokenizer_id": tok.tokenizer_id,
                "name": tok.name,
                "tokenizer_type": tok.tokenizer_type,
                "vocab_size": tok.vocab_size,
                "num_training_documents": tok.num_training_documents,
                "is_active": tok.is_active,
                "created_at": tok.created_at,
                "tags": tok.tags
            }
            for tok in tokenizers
        ]
    
    def load_tokenizer(self, tokenizer_id: str) -> Optional[BPETokenizer]:
        """
        Tokenizer'ı disk'ten yükle.
        
        Args:
            tokenizer_id: Tokenizer ID
            
        Returns:
            BPETokenizer: Loaded tokenizer instance
            
        Raises:
            FileNotFoundError: Tokenizer dosyaları bulunamazsa
        """
        # Metadata'yı al
        tokenizer_record = self.db.query(TokenizerRecord).filter(
            TokenizerRecord.tokenizer_id == tokenizer_id
        ).first()
        
        if not tokenizer_record:
            return None
        
        # Tokenizer yükle
        tokenizer = BPETokenizer()
        tokenizer.load_vocab(Path(str(tokenizer_record.storage_path)))
        
        # Usage stats güncelle
        tokenizer_record.usage_count += 1
        tokenizer_record.last_used_at = datetime.utcnow()
        self.db.commit()
        
        logger.info(f"Tokenizer loaded: {tokenizer_id}")
        return tokenizer
    
    def update_tokenizer_metadata(
        self,
        tokenizer_id: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_active: Optional[bool] = None
    ) -> bool:
        """
        Tokenizer metadata'sını güncelle.
        
        Args:
            tokenizer_id: Tokenizer ID
            description: Yeni açıklama
            tags: Yeni tags
            is_active: Aktif durumu
            
        Returns:
            bool: Güncelleme başarılı mı
        """
        tokenizer = self.db.query(TokenizerRecord).filter(
            TokenizerRecord.tokenizer_id == tokenizer_id
        ).first()
        
        if not tokenizer:
            return False
        
        if description is not None:
            tokenizer.description = description
        
        if tags is not None:
            tokenizer.tags = tags
        
        if is_active is not None:
            tokenizer.is_active = is_active
        
        tokenizer.updated_at = datetime.utcnow()
        self.db.commit()
        
        logger.info(f"Tokenizer metadata updated: {tokenizer_id}")
        return True
    
    def delete_tokenizer(self, tokenizer_id: str, hard_delete: bool = False) -> bool:
        """
        Tokenizer'ı sil.
        
        Args:
            tokenizer_id: Tokenizer ID
            hard_delete: True ise dosyalar da silinir, False ise sadece is_active=False
            
        Returns:
            bool: Silme başarılı mı
        """
        tokenizer = self.db.query(TokenizerRecord).filter(
            TokenizerRecord.tokenizer_id == tokenizer_id
        ).first()
        
        if not tokenizer:
            return False
        
        if hard_delete:
            # Disk'ten sil
            storage_path = Path(str(tokenizer.storage_path))
            if storage_path.exists():
                import shutil
                shutil.rmtree(storage_path)
                logger.info(f"Tokenizer files deleted: {storage_path}")
            
            # Database'den sil
            self.db.delete(tokenizer)
            self.db.commit()
            logger.info(f"Tokenizer hard deleted: {tokenizer_id}")
        else:
            # Soft delete - sadece deactivate et
            tokenizer.is_active = False
            tokenizer.updated_at = datetime.utcnow()
            self.db.commit()
            logger.info(f"Tokenizer deactivated: {tokenizer_id}")
        
        return True
