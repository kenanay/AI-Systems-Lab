"""
Parquet Writer

Canonical dataset'i Parquet formatında yaz.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import logging

from backend.config import settings

logger = logging.getLogger(__name__)


class ParquetWriter:
    """
    Dataset'leri Parquet formatında yazar.
    
    Canonical dataset standardına uygun:
    - files.parquet: FileRecord'lar
    - documents.parquet: DocumentRecord'lar
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Args:
            output_dir: Parquet dosyalarının yazılacağı dizin
        """
        self.output_dir = output_dir or settings.processed_data_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_files(
        self,
        file_records: List[Dict[str, Any]],
        output_filename: str = "files.parquet"
    ) -> Path:
        """
        FileRecord'ları files.parquet'e yaz.
        
        Args:
            file_records: FileRecord dictionary'leri
            output_filename: Çıktı dosya adı
            
        Returns:
            Yazılan dosyanın path'i
        """
        if not file_records:
            logger.warning("No file records to write")
            return None
        
        logger.info(f"Writing {len(file_records)} file records to Parquet")
        
        # DataFrame'e dönüştür
        df = pd.DataFrame(file_records)
        
        # Datetime kolonları düzelt
        datetime_cols = ['created_at', 'modified_at']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Schema tanımla (type safety için)
        schema = pa.schema([
            ('file_id', pa.string()),
            ('original_name', pa.string()),
            ('relative_path', pa.string()),
            ('mime_type', pa.string()),
            ('size_bytes', pa.int64()),
            ('sha256', pa.string()),
            ('parser_name', pa.string()),
            ('parser_version', pa.string()),
            ('schema_version', pa.string()),
            ('security_level', pa.string()),
            ('pii_detected', pa.bool_()),
            ('license', pa.string()),
            ('copyright_status', pa.string()),
            ('training_allowed', pa.bool_()),
            ('created_at', pa.timestamp('us')),
            ('modified_at', pa.timestamp('us')),
            ('dataset_version', pa.string()),
            ('source', pa.string()),
            ('language', pa.string()),
            ('quality_score', pa.float64()),
        ])
        
        # Parquet'e yaz
        output_path = self.output_dir / output_filename
        table = pa.Table.from_pandas(df, schema=schema)
        
        pq.write_table(
            table,
            output_path,
            compression='snappy',  # Hızlı compression
            use_dictionary=True,   # String compression
            write_statistics=True   # Min/max statistics
        )
        
        logger.info(f"Written {output_path} ({output_path.stat().st_size} bytes)")
        
        return output_path
    
    def write_documents(
        self,
        document_records: List[Dict[str, Any]],
        output_filename: str = "documents.parquet"
    ) -> Path:
        """
        DocumentRecord'ları documents.parquet'e yaz.
        
        Args:
            document_records: DocumentRecord dictionary'leri
            output_filename: Çıktı dosya adı
            
        Returns:
            Yazılan dosyanın path'i
        """
        if not document_records:
            logger.warning("No document records to write")
            return None
        
        logger.info(f"Writing {len(document_records)} document records to Parquet")
        
        # DataFrame'e dönüştür
        df = pd.DataFrame(document_records)
        
        # Datetime kolonları düzelt
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Schema tanımla
        schema = pa.schema([
            ('document_id', pa.string()),
            ('file_id', pa.string()),
            ('title', pa.string()),
            ('text', pa.string()),
            ('language', pa.string()),
            ('char_count', pa.int32()),
            ('word_count', pa.int32()),
            ('line_count', pa.int32()),
            ('parser_name', pa.string()),
            ('parser_version', pa.string()),
            ('schema_version', pa.string()),
            ('is_empty', pa.bool_()),
            ('is_duplicate', pa.bool_()),
            ('quality_score', pa.float64()),
            ('created_at', pa.timestamp('us')),
        ])
        
        # Parquet'e yaz
        output_path = self.output_dir / output_filename
        table = pa.Table.from_pandas(df, schema=schema)
        
        pq.write_table(
            table,
            output_path,
            compression='snappy',
            use_dictionary=True,
            write_statistics=True
        )
        
        logger.info(f"Written {output_path} ({output_path.stat().st_size} bytes)")
        
        return output_path
    
    def append_files(
        self,
        file_records: List[Dict[str, Any]],
        output_filename: str = "files.parquet",
        dataset_version: Optional[str] = None
    ) -> Path:
        """
        Mevcut files.parquet'e yeni kayıtlar ekle.
        
        **UYARI:** Bu fonksiyon mevcut dosyayı değiştirir. Immutable versioning için
        yeni bir version oluşturmak daha iyidir. Bu fonksiyon yalnızca aynı dataset
        versiyonu içinde incremental update için kullanılmalıdır.
        
        Args:
            file_records: Yeni FileRecord'lar
            output_filename: Dosya adı
            dataset_version: Dataset version (varsa tüm kayıtlara eklenir)
            
        Returns:
            Güncellenmiş dosyanın path'i
        """
        logger.warning(
            "⚠️  append_files mevcut dosyayı değiştirir. "
            "Immutable versioning için create_new_version() kullanın."
        )
        
        output_path = self.output_dir / output_filename
        
        # Dataset version ekle
        if dataset_version:
            for record in file_records:
                record['dataset_version'] = dataset_version
        
        # Mevcut dosya varsa oku
        if output_path.exists():
            existing_df = pd.read_parquet(output_path)
            new_df = pd.DataFrame(file_records)
            
            # Duplicate kontrolü (SHA-256 bazlı)
            if 'sha256' in existing_df.columns and 'sha256' in new_df.columns:
                existing_hashes = set(existing_df['sha256'])
                new_df = new_df[~new_df['sha256'].isin(existing_hashes)]
                
                duplicates_found = len(file_records) - len(new_df)
                if duplicates_found > 0:
                    logger.info(f"Filtered {duplicates_found} duplicate files (by SHA-256)")
            
            # Birleştir
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Yeniden yaz
            self.write_files(combined_df.to_dict('records'), output_filename)
        else:
            # İlk yazma
            self.write_files(file_records, output_filename)
        
        return output_path
    
    def read_files(self, filename: str = "files.parquet") -> pd.DataFrame:
        """
        files.parquet'i oku.
        
        Args:
            filename: Dosya adı
            
        Returns:
            DataFrame
        """
        file_path = self.output_dir / filename
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        
        return pd.read_parquet(file_path)
    
    def read_documents(self, filename: str = "documents.parquet") -> pd.DataFrame:
        """
        documents.parquet'i oku.
        
        Args:
            filename: Dosya adı
            
        Returns:
            DataFrame
        """
        file_path = self.output_dir / filename
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        
        return pd.read_parquet(file_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Dataset istatistiklerini hesapla.
        
        Returns:
            İstatistik dictionary
        """
        files_df = self.read_files()
        docs_df = self.read_documents()
        
        if files_df.empty:
            return {
                "total_files": 0,
                "total_documents": 0,
                "total_size_bytes": 0,
            }
        
        stats = {
            "total_files": len(files_df),
            "total_documents": len(docs_df),
            "total_size_bytes": int(files_df['size_bytes'].sum()) if 'size_bytes' in files_df else 0,
        }
        
        # Dosya tipi dağılımı
        if 'mime_type' in files_df.columns:
            stats["files_by_type"] = files_df['mime_type'].value_counts().to_dict()
        
        # Dil dağılımı
        if 'language' in files_df.columns:
            stats["files_by_language"] = files_df['language'].value_counts().to_dict()
        
        # Kalite skoru ortalama
        if 'quality_score' in files_df.columns:
            stats["avg_quality_score"] = float(files_df['quality_score'].mean())
        
        # PII ve training izinli dosya sayıları
        if 'pii_detected' in files_df.columns:
            stats["pii_detected_count"] = int(files_df['pii_detected'].sum())
        
        if 'training_allowed' in files_df.columns:
            stats["training_allowed_count"] = int(files_df['training_allowed'].sum())
        
        return stats


class DatasetExporter:
    """
    Canonical dataset'ten farklı formatlara export.
    
    **GÜVENLİK:** Tüm export fonksiyonları PII ve training izni kontrolü yapar.
    
    Desteklenen formatlar:
    - Pretraining export (JSONL)
    - Metadata export (CSV)
    
    Future:
    - SFT export (messages format)
    - RAG export (chunks)
    """
    
    def __init__(self, parquet_reader: ParquetWriter):
        self.reader = parquet_reader
    
    def create_versioned_dataset(
        self,
        version: str,
        description: str = ""
    ) -> Path:
        """
        Yeni bir versiyonlu dataset oluştur (immutable).
        
        Dataset versioning standardı:
        - v1.0.0: İlk sürüm
        - v1.1.0: Yeni veri eklendi
        - v1.0.1: Metadata düzeltmesi
        - v2.0.0: Schema değişikliği
        
        Args:
            version: Semantic version (örn: "v1.0.0")
            description: Versiyon açıklaması
            
        Returns:
            Yeni version dizini
        """
        # Version dizini oluştur
        version_dir = self.reader.output_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Mevcut dataset'i kopyala
        files_df = self.reader.read_files()
        docs_df = self.reader.read_documents()
        
        # Version metadata'yı ekle
        if not files_df.empty:
            files_df['dataset_version'] = version
        if not docs_df.empty:
            docs_df['dataset_version'] = version
        
        # Yeni versiyona yaz
        if not files_df.empty:
            pq.write_table(
                pa.Table.from_pandas(files_df),
                version_dir / "files.parquet"
            )
        
        if not docs_df.empty:
            pq.write_table(
                pa.Table.from_pandas(docs_df),
                version_dir / "documents.parquet"
            )
        
        # README oluştur
        readme_path = version_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# Dataset Version {version}\n\n")
            f.write(f"**Created:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Description:** {description}\n\n")
            f.write(f"## Statistics\n\n")
            f.write(f"- Files: {len(files_df)}\n")
            f.write(f"- Documents: {len(docs_df)}\n")
        
        logger.info(f"✅ Created immutable dataset version: {version}")
        
        return version_dir
    
    def export_pretraining(
        self,
        output_path: Path,
        filter_pii: bool = True,
        require_training_allowed: bool = True,
        min_quality_score: float = 0.5
    ) -> Path:
        """
        Pretraining formatına export et.
        
        **GÜVENLİK:** PII içeren ve training izni olmayan veriler otomatik filtrelenir.
        
        Format: {"text": "..."}
        
        Args:
            output_path: Çıktı dosya path'i
            filter_pii: PII içeren dokümanları filtrele (default: True)
            require_training_allowed: training_allowed=True olanları al (default: True)
            min_quality_score: Minimum kalite skoru eşiği (default: 0.5)
            
        Returns:
            Export edilen dosyanın path'i
            
        Raises:
            ValueError: Güvenlik filtreleri devre dışı bırakılmaya çalışılırsa
        """
        # GÜVENLİK: PII filtresi zorunlu
        if not filter_pii:
            raise ValueError(
                "PII filtresini devre dışı bırakmak güvenlik ilkelerine aykırıdır. "
                "PII içeren veri training dataset'e dahil edilemez."
            )
        
        docs_df = self.reader.read_documents()
        files_df = self.reader.read_files()
        
        if docs_df.empty:
            logger.warning("No documents to export")
            return None
        
        # Documents ile files'ı birleştir (metadata için)
        merged_df = docs_df.merge(
            files_df[['file_id', 'pii_detected', 'training_allowed', 'license', 'security_level']],
            on='file_id',
            how='left'
        )
        
        initial_count = len(merged_df)
        
        # GÜVENLİK FİLTRELERİ
        # 1. PII filtresi
        if filter_pii:
            merged_df = merged_df[merged_df['pii_detected'] == False]
            logger.info(f"PII filter: {initial_count} -> {len(merged_df)} documents")
        
        # 2. Training izni kontrolü
        if require_training_allowed:
            merged_df = merged_df[merged_df['training_allowed'] == True]
            logger.info(f"Training allowed filter: {len(merged_df)} documents")
        
        # 3. Kalite filtresi
        if 'quality_score' in merged_df.columns:
            merged_df = merged_df[merged_df['quality_score'] >= min_quality_score]
            logger.info(f"Quality filter (>={min_quality_score}): {len(merged_df)} documents")
        
        # 4. Boş içerik filtresi
        merged_df = merged_df[merged_df['is_empty'] == False]
        merged_df = merged_df[merged_df['text'].notna()]
        merged_df = merged_df[merged_df['text'].str.strip() != '']
        
        if merged_df.empty:
            logger.error("No documents passed security filters!")
            logger.error(
                "Kontrol edin: training_allowed=True, pii_detected=False, "
                "quality_score>={min_quality_score} olan veri var mı?"
            )
            return None
        
        final_count = len(merged_df)
        logger.info(
            f"✅ Güvenlik filtreleri tamamlandı: {initial_count} -> {final_count} documents "
            f"({final_count/initial_count*100:.1f}% retained)"
        )
        
        # JSONL formatında yaz
        exported_count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in merged_df.iterrows():
                # Ekstra güvenlik: PII double-check
                if row.get('pii_detected', False):
                    logger.warning(f"Skipping document {row['document_id']} - PII detected")
                    continue
                
                f.write('{"text": ' + pd.Series([row['text']]).to_json(orient='values')[1:-1] + '}\n')
                exported_count += 1
        
        logger.info(f"✅ Exported {exported_count} documents to {output_path}")
        logger.info(
            f"⚠️  GÜVENLİK: PII filter={filter_pii}, "
            f"training_allowed={require_training_allowed}, "
            f"min_quality={min_quality_score}"
        )
        
        return output_path
    
    def export_metadata(self, output_path: Path) -> Path:
        """
        Metadata CSV export.
        
        Args:
            output_path: Çıktı dosya path'i
            
        Returns:
            Export edilen dosyanın path'i
        """
        files_df = self.reader.read_files()
        
        if files_df.empty:
            logger.warning("No files to export")
            return None
        
        # Text column'u çıkar (çok büyük)
        metadata_cols = [col for col in files_df.columns if col != 'text']
        metadata_df = files_df[metadata_cols]
        
        metadata_df.to_csv(output_path, index=False)
        logger.info(f"Exported metadata to {output_path}")
        
        return output_path


# Global instances
parquet_writer = ParquetWriter()
dataset_exporter = DatasetExporter(parquet_writer)
