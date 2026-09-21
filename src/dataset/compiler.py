"""
src/dataset/compiler.py

Dataset Compiler

Bu modül documents'ları training-ready Parquet dataset'e compile eder:
- Document selection & filtering
- Tokenization (with selected tokenizer)
- Quality filtering
- Deduplication
- Parquet export with schema validation
- Data lineage tracking

Kaynaklar:
    - Hugging Face Datasets: https://huggingface.co/docs/datasets
    - PyArrow Parquet: https://arrow.apache.org/docs/python/parquet.html
"""

from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from datetime import datetime
import logging
import hashlib
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.orm import Session

from backend.models import DocumentRecord, FileRecord, TokenizerRecord
from src.tokenizer.bpe import BPETokenizer

logger = logging.getLogger(__name__)


class DatasetCompiler:
    """
    Dataset Compiler - Documents'ları training dataset'e compile eder.
    
    Compilation Pipeline:
    1. Document Selection (filters: quality, license, PII)
    2. Tokenization (with selected tokenizer)
    3. Quality Filtering (min/max length, quality score)
    4. Deduplication (exact + near-duplicate)
    5. Parquet Export (with schema validation)
    6. Metadata & Lineage (tracking source documents)
    
    Args:
        output_dir: Compiled dataset'in yazılacağı dizin
        tokenizer: Kullanılacak tokenizer instance
    
    Attributes:
        schema_version: Dataset schema versiyonu
        stats: Compilation statistics
    """
    
    VERSION = "1.0.0"
    SCHEMA_VERSION = "1.0.0"
    
    def __init__(
        self,
        output_dir: Path,
        tokenizer: BPETokenizer,
        dataset_version: Optional[str] = None,
        db_session: Optional[Session] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = tokenizer
        self.dataset_version = dataset_version or "1.0.0"
        self.db_session = db_session
        
        # Compilation statistics
        self.stats = {
            "total_documents": 0,
            "filtered_by_quality": 0,
            "filtered_by_license": 0,
            "filtered_by_pii": 0,
            "filtered_by_length": 0,
            "duplicates_removed": 0,
            "final_count": 0,
            "total_tokens": 0,
        }
    
    def compile_dataset(
        self,
        documents: List[DocumentRecord],
        min_quality_score: float = 0.5,
        max_quality_score: float = 1.0,
        min_length: int = 10,
        max_length: int = 100000,
        allow_pii: bool = False,
        require_training_allowed: bool = True,
        remove_duplicates: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Documents'ları compile edip training dataset oluştur.
        
        Args:
            documents: Document listesi
            min_quality_score: Minimum kalite skoru (0.0-1.0)
            max_quality_score: Maximum kalite skoru
            min_length: Minimum karakter sayısı
            max_length: Maximum karakter sayısı
            allow_pii: PII içeren documents'a izin ver mi
            require_training_allowed: training_allowed=True olanları filtrele
            remove_duplicates: Duplicate'leri kaldır
            progress_callback: Progress callback function(current, total)
            
        Returns:
            Dict: Compilation sonuçları
                - output_path: str
                - stats: Dict
                - metadata_path: str
                
        Raises:
            ValueError: documents boş ise veya tokenizer train edilmemişse
        """
        if not documents:
            raise ValueError("Documents listesi boş olamaz")
        
        if not self.tokenizer.is_trained:
            raise ValueError("Tokenizer train edilmemiş. Önce tokenizer train edin.")
        
        logger.info(f"Dataset compilation başladı: {len(documents)} documents")
        self.stats["total_documents"] = len(documents)
        
        # Step 1: Filter by quality score
        filtered_docs = self._filter_by_quality(
            documents,
            min_quality_score,
            max_quality_score
        )
        
        # Step 2: Filter by license & training permission
        if require_training_allowed:
            filtered_docs = self._filter_by_training_permission(filtered_docs)
        
        # Step 3: Filter by PII
        if not allow_pii:
            filtered_docs = self._filter_by_pii(filtered_docs)
        
        # Step 4: Filter by length
        filtered_docs = self._filter_by_length(filtered_docs, min_length, max_length)
        
        # Step 5: Tokenize documents
        tokenized_docs = []
        total = len(filtered_docs)
        
        for idx, doc in enumerate(filtered_docs):
            try:
                # Tokenize
                token_ids = self.tokenizer.encode(doc.text)
                
                # Create tokenized document
                tokenized_doc = {
                    "document_id": doc.document_id,
                    "file_id": doc.file_id,
                    "text": doc.text,
                    "token_ids": token_ids,
                    "num_tokens": len(token_ids),
                    "char_count": doc.char_count or len(doc.text),
                    "word_count": doc.word_count,
                    "quality_score": doc.quality_score,
                    "language": doc.language,
                    "schema_version": doc.schema_version,
                }
                
                tokenized_docs.append(tokenized_doc)
                self.stats["total_tokens"] += len(token_ids)
                
                # Progress callback
                if progress_callback and (idx % 100 == 0 or idx == total - 1):
                    progress_callback(idx + 1, total)
                
            except Exception as e:
                logger.error(f"Tokenization failed for {doc.document_id}: {e}")
                continue
        
        logger.info(f"Tokenization complete: {len(tokenized_docs)} documents")
        
        # Step 6: Deduplication
        if remove_duplicates:
            tokenized_docs = self._remove_duplicates(tokenized_docs)
        
        self.stats["final_count"] = len(tokenized_docs)
        
        # Step 7: Export to Parquet
        output_path = self._export_to_parquet(tokenized_docs)
        
        # Step 8: Write metadata
        metadata_path = self._write_metadata(documents, tokenized_docs)
        
        logger.info(f"Dataset compilation tamamlandı: {output_path}")
        
        return {
            "output_path": str(output_path),
            "stats": self.stats,
            "metadata_path": str(metadata_path),
            "schema_version": self.SCHEMA_VERSION,
            "compiler_version": self.VERSION,
        }
    
    def _filter_by_quality(
        self,
        documents: List[DocumentRecord],
        min_score: float,
        max_score: float
    ) -> List[DocumentRecord]:
        """
        Kalite skoruna göre filtrele.
        
        Args:
            documents: Document listesi
            min_score: Minimum skor
            max_score: Maximum skor
            
        Returns:
            List[DocumentRecord]: Filtrelenmiş documents
        """
        filtered = []
        
        for doc in documents:
            if doc.quality_score is None:
                continue
            
            if min_score <= doc.quality_score <= max_score:
                filtered.append(doc)
            else:
                self.stats["filtered_by_quality"] += 1
        
        logger.info(
            f"Quality filter: {len(filtered)}/{len(documents)} passed "
            f"(filtered: {self.stats['filtered_by_quality']})"
        )
        
        return filtered
    
    def _filter_by_training_permission(
        self,
        documents: List[DocumentRecord]
    ) -> List[DocumentRecord]:
        """
        Training permission'a göre filtrele.
        
        File'ın training_allowed=True olması gerekir.
        
        Args:
            documents: Document listesi
            
        Returns:
            List[DocumentRecord]: Filtrelenmiş documents
        """
        if not self.db_session:
            logger.warning("No DB session, skipping training permission filter")
            return documents
        
        filtered = []
        
        for doc in documents:
            # File record'u al
            file = self.db_session.query(FileRecord).filter(
                FileRecord.file_id == doc.file_id
            ).first()
            
            if file and file.training_allowed:
                filtered.append(doc)
            else:
                self.stats["filtered_by_license"] += 1
        
        logger.info(
            f"Training permission filter: {len(filtered)}/{len(documents)} passed "
            f"(filtered: {self.stats['filtered_by_license']})"
        )
        
        return filtered
    
    def _filter_by_pii(
        self,
        documents: List[DocumentRecord]
    ) -> List[DocumentRecord]:
        """
        PII detection'a göre filtrele.
        
        File'ın pii_detected=False olması gerekir.
        
        Args:
            documents: Document listesi
            
        Returns:
            List[DocumentRecord]: Filtrelenmiş documents
        """
        if not self.db_session:
            logger.warning("No DB session, skipping PII filter")
            return documents
        
        filtered = []
        
        for doc in documents:
            # File record'u al
            file = self.db_session.query(FileRecord).filter(
                FileRecord.file_id == doc.file_id
            ).first()
            
            if file and not file.pii_detected:
                filtered.append(doc)
            else:
                self.stats["filtered_by_pii"] += 1
        
        logger.info(
            f"PII filter: {len(filtered)}/{len(documents)} passed "
            f"(filtered: {self.stats['filtered_by_pii']})"
        )
        
        return filtered
    
    def _filter_by_length(
        self,
        documents: List[DocumentRecord],
        min_length: int,
        max_length: int
    ) -> List[DocumentRecord]:
        """
        Text uzunluğuna göre filtrele.
        
        Args:
            documents: Document listesi
            min_length: Minimum karakter
            max_length: Maximum karakter
            
        Returns:
            List[DocumentRecord]: Filtrelenmiş documents
        """
        filtered = []
        
        for doc in documents:
            char_count = doc.char_count or len(doc.text or "")
            
            if min_length <= char_count <= max_length:
                filtered.append(doc)
            else:
                self.stats["filtered_by_length"] += 1
        
        logger.info(
            f"Length filter: {len(filtered)}/{len(documents)} passed "
            f"(filtered: {self.stats['filtered_by_length']})"
        )
        
        return filtered
    
    def _remove_duplicates(
        self,
        tokenized_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Duplicate documents'ları kaldır.
        
        Exact match: SHA-256 hash of text
        
        Args:
            tokenized_docs: Tokenized document listesi
            
        Returns:
            List[Dict]: Deduplicated documents
        """
        seen_hashes: Set[str] = set()
        unique_docs = []
        
        for doc in tokenized_docs:
            # Text'in SHA-256 hash'i
            text_hash = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_docs.append(doc)
            else:
                self.stats["duplicates_removed"] += 1
        
        logger.info(
            f"Deduplication: {len(unique_docs)}/{len(tokenized_docs)} unique "
            f"(duplicates removed: {self.stats['duplicates_removed']})"
        )
        
        return unique_docs
    
    def _export_to_parquet(
        self,
        tokenized_docs: List[Dict[str, Any]]
    ) -> Path:
        """
        Tokenized documents'ları Parquet'e export et.
        
        Args:
            tokenized_docs: Tokenized document listesi
            
        Returns:
            Path: Parquet file path
        """
        if not tokenized_docs:
            raise ValueError("No documents to export")
        
        # DataFrame oluştur
        df = pd.DataFrame(tokenized_docs)
        
        # PyArrow schema tanımla (type safety)
        schema = pa.schema([
            ("document_id", pa.string()),
            ("file_id", pa.string()),
            ("text", pa.string()),
            ("token_ids", pa.list_(pa.int32())),
            ("num_tokens", pa.int32()),
            ("char_count", pa.int32()),
            ("word_count", pa.int32()),
            ("quality_score", pa.float64()),
            ("language", pa.string()),
            ("schema_version", pa.string()),
        ])
        
        # PyArrow table
        table = pa.Table.from_pandas(df, schema=schema)
        
        # Parquet'e yaz
        output_path = self.output_dir / "dataset.parquet"
        
        pq.write_table(
            table,
            output_path,
            compression="snappy",  # Hızlı compression
            use_dictionary=True,   # String compression
            write_statistics=True,  # Min/max statistics için
            row_group_size=10000,  # 10K rows per group (memory efficiency)
        )
        
        logger.info(f"Parquet export complete: {output_path} ({len(tokenized_docs)} documents)")
        
        return output_path
    
    def _write_metadata(
        self,
        source_documents: List[DocumentRecord],
        compiled_documents: List[Dict[str, Any]]
    ) -> Path:
        """
        Dataset metadata'sını yaz.
        
        Data lineage tracking için:
        - Source document IDs
        - Compilation parameters
        - Statistics
        - Schema version
        
        Args:
            source_documents: Kaynak documents
            compiled_documents: Compiled documents
            
        Returns:
            Path: Metadata JSON file path
        """
        metadata = {
            "dataset_version": self.dataset_version,
            "schema_version": self.SCHEMA_VERSION,
            "compiler_version": self.VERSION,
            "created_at": datetime.utcnow().isoformat(),
            "tokenizer": {
                "vocab_size": self.tokenizer.vocab_size,
                "version": self.tokenizer.VERSION,
            },
            "statistics": self.stats,
            "source_document_ids": [doc.document_id for doc in source_documents],
            "source_file_ids": list(set(doc.file_id for doc in source_documents)),
            "compiled_document_ids": [doc["document_id"] for doc in compiled_documents],
        }
        
        metadata_path = self.output_dir / "metadata.json"
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata written: {metadata_path}")
        
        return metadata_path
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Compiled dataset bilgilerini döndür.
        
        Returns:
            Dict: Dataset info
        """
        parquet_path = self.output_dir / "dataset.parquet"
        metadata_path = self.output_dir / "metadata.json"
        
        if not parquet_path.exists():
            return {"error": "Dataset not compiled yet"}
        
        # Parquet file size
        file_size = parquet_path.stat().st_size
        
        # Metadata oku
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        
        return {
            "parquet_path": str(parquet_path),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "metadata": metadata,
            "stats": self.stats,
        }
