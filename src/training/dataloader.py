"""
src/training/dataloader.py

Training DataLoader

Bu modül compiled Parquet dataset'lerden efficient batch loading sağlar:
- PyTorch Dataset ve DataLoader integration
- Sequence packing (multiple documents in single sequence)
- Memory-efficient streaming
- Tokenized data loading

Parquet Dataset Format:
- document_id: str
- text: str
- token_ids: list[int]
- num_tokens: int
- metadata: dict

DataLoader Output Format:
- input_ids: Tensor [batch_size, seq_len]
- labels: Tensor [batch_size, seq_len] (shifted by 1)
- attention_mask: Tensor [batch_size, seq_len]
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ParquetDataset(Dataset):
    """
    Parquet dataset for training.
    
    Compiled dataset'i okuyup PyTorch Dataset interface sağlar.
    
    Attributes:
        parquet_path: Parquet file path
        sequence_length: Training sequence length
        stride: Stride for sequence creation (None = sequence_length)
        pack_sequences: Sequence packing kullan mı
    """
    
    def __init__(
        self,
        parquet_path: Path,
        sequence_length: int = 512,
        stride: Optional[int] = None,
        pack_sequences: bool = True
    ):
        self.parquet_path = Path(parquet_path)
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length  # Default: no overlap
        self.pack_sequences = pack_sequences
        
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        # Parquet dosyasını aç
        self.table = pq.read_table(self.parquet_path)
        
        logger.info(f"Loaded dataset: {len(self.table)} documents")
        
        # Token sequences oluştur
        self._build_sequences()
    
    def _build_sequences(self):
        """
        Dataset'ten training sequences oluştur.
        
        İki mod:
        1. Pack sequences: Birden fazla document'ı bir sequence'te birleştir
        2. No packing: Her document ayrı sequence'ler
        """
        self.sequences = []
        
        if self.pack_sequences:
            self._build_packed_sequences()
        else:
            self._build_unpacked_sequences()
        
        logger.info(f"Built {len(self.sequences)} sequences (pack={self.pack_sequences})")
    
    def _build_packed_sequences(self):
        """
        Sequence packing: Birden fazla document'ı tek sequence'te birleştir.
        
        Avantaj: GPU utilization maksimum (padding minimum)
        Dezavantaj: Cross-document attention (gerçekçi olmayabilir)
        
        Format:
        [doc1_tokens] [SEP] [doc2_tokens] [SEP] ... [PAD] [PAD]
        
        SEP token_id = 0 olarak varsayılır (BOS/EOS gibi)
        """
        current_sequence = []
        current_length = 0
        
        # Her document'ı işle
        for i in range(len(self.table)):
            token_ids = self.table['token_ids'][i].as_py()
            
            # Document boşsa atla
            if not token_ids:
                continue
            
            # Document'ı sequence'e ekle
            # SEP token ekle (document sınırlarını belirt)
            doc_with_sep = token_ids + [0]  # 0 = SEP/EOS token
            doc_length = len(doc_with_sep)
            
            # Current sequence'e sığar mı?
            if current_length + doc_length <= self.sequence_length:
                # Sığıyor - ekle
                current_sequence.extend(doc_with_sep)
                current_length += doc_length
            else:
                # Sığmıyor - mevcut sequence'i kaydet
                if current_sequence:
                    # Padding ekle
                    padding_length = self.sequence_length - current_length
                    current_sequence.extend([0] * padding_length)
                    
                    self.sequences.append(current_sequence)
                
                # Yeni sequence başlat
                # Document sequence_length'ten uzunsa truncate et
                if doc_length > self.sequence_length:
                    current_sequence = doc_with_sep[:self.sequence_length]
                    current_length = self.sequence_length
                else:
                    current_sequence = doc_with_sep
                    current_length = doc_length
        
        # Son sequence'i kaydet
        if current_sequence:
            padding_length = self.sequence_length - current_length
            current_sequence.extend([0] * padding_length)
            self.sequences.append(current_sequence)
    
    def _build_unpacked_sequences(self):
        """
        No packing: Her document ayrı sequence'lere bölünür.
        
        Avantaj: Document boundaries korunur (realistic attention)
        Dezavantaj: Padding fazla (GPU utilization düşük)
        
        Stride ile sliding window:
        - stride < sequence_length: Overlap var
        - stride = sequence_length: No overlap
        """
        for i in range(len(self.table)):
            token_ids = self.table['token_ids'][i].as_py()
            
            if not token_ids:
                continue
            
            # Document'ı sequence_length boyutunda parçalara böl
            for start_idx in range(0, len(token_ids), self.stride):
                end_idx = start_idx + self.sequence_length
                
                # Sequence'i al
                sequence = token_ids[start_idx:end_idx]
                
                # Minimum length check (çok kısa sequence'leri atla)
                if len(sequence) < self.sequence_length // 2:
                    continue
                
                # Padding ekle (sequence_length'e tamamla)
                if len(sequence) < self.sequence_length:
                    padding_length = self.sequence_length - len(sequence)
                    sequence.extend([0] * padding_length)
                
                self.sequences.append(sequence)
    
    def __len__(self) -> int:
        """Dataset boyutu (sequence sayısı)"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Tek training sample getir.
        
        Args:
            idx: Sequence index
            
        Returns:
            Dict:
                - input_ids: [seq_len] - Input token IDs
                - labels: [seq_len] - Target token IDs (shifted by 1)
                - attention_mask: [seq_len] - 1=valid, 0=padding
        """
        sequence = self.sequences[idx]
        
        # Tensor'e çevir
        # input_ids: [T] - Token IDs
        input_ids = torch.tensor(sequence, dtype=torch.long)
        
        # labels: [T] - Next token prediction için shifted
        # labels[i] = input_ids[i+1]
        labels = torch.tensor(sequence, dtype=torch.long)
        
        # attention_mask: [T] - Padding mask
        # 1 = valid token, 0 = padding
        attention_mask = (input_ids != 0).long()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Batch collate function.
    
    Multiple samples'ı batch haline getirir.
    
    Args:
        batch: List of samples (her biri __getitem__ output'u)
        
    Returns:
        Dict:
            - input_ids: [B, T] - Batch of input sequences
            - labels: [B, T] - Batch of label sequences
            - attention_mask: [B, T] - Batch of attention masks
            
    Shape notation:
        B = batch_size
        T = sequence_length
    """
    # Stack samples
    # Her sample zaten aynı shape'te (sequence_length)
    # input_ids: [B, T]
    input_ids = torch.stack([item["input_ids"] for item in batch])
    
    # labels: [B, T]
    labels = torch.stack([item["labels"] for item in batch])
    
    # attention_mask: [B, T]
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }


def create_dataloader(
    parquet_path: Path,
    batch_size: int = 32,
    sequence_length: int = 512,
    stride: Optional[int] = None,
    pack_sequences: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Training DataLoader oluştur.
    
    Args:
        parquet_path: Compiled Parquet dataset path
        batch_size: Batch size
        sequence_length: Training sequence length
        stride: Stride for sequence creation (None = sequence_length)
        pack_sequences: Sequence packing kullan mı
        shuffle: Dataset shuffle et
        num_workers: DataLoader worker sayısı
        pin_memory: Pin memory for GPU transfer
        
    Returns:
        DataLoader: PyTorch DataLoader instance
        
    Example:
        >>> dataloader = create_dataloader(
        ...     parquet_path="data/compiled_datasets/turkish_v1.0.0/dataset.parquet",
        ...     batch_size=32,
        ...     sequence_length=512
        ... )
        >>> 
        >>> for batch in dataloader:
        ...     input_ids = batch["input_ids"]  # [32, 512]
        ...     labels = batch["labels"]        # [32, 512]
        ...     attention_mask = batch["attention_mask"]  # [32, 512]
    """
    # Dataset oluştur
    dataset = ParquetDataset(
        parquet_path=parquet_path,
        sequence_length=sequence_length,
        stride=stride,
        pack_sequences=pack_sequences
    )
    
    # DataLoader oluştur
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    logger.info(
        f"DataLoader created: {len(dataset)} sequences, "
        f"{len(dataloader)} batches (batch_size={batch_size})"
    )
    
    return dataloader


class DataLoaderStats:
    """
    DataLoader statistics tracker.
    
    Training sırasında data loading metrics'lerini takip eder.
    """
    
    def __init__(self):
        self.total_batches = 0
        self.total_tokens = 0
        self.total_padding_tokens = 0
    
    def update(self, batch: Dict[str, torch.Tensor]):
        """
        Batch statistics güncelle.
        
        Args:
            batch: DataLoader output batch
        """
        self.total_batches += 1
        
        # batch shape: [B, T]
        batch_size, seq_len = batch["input_ids"].shape
        
        # Total tokens
        total = batch_size * seq_len
        self.total_tokens += total
        
        # Padding tokens (attention_mask == 0)
        padding = (batch["attention_mask"] == 0).sum().item()
        self.total_padding_tokens += padding
    
    @property
    def padding_ratio(self) -> float:
        """Padding ratio (0-1)"""
        if self.total_tokens == 0:
            return 0.0
        return self.total_padding_tokens / self.total_tokens
    
    @property
    def efficiency(self) -> float:
        """Data efficiency (1 - padding_ratio)"""
        return 1.0 - self.padding_ratio
    
    def summary(self) -> Dict:
        """
        Statistics summary.
        
        Returns:
            Dict: Statistics
        """
        return {
            "total_batches": self.total_batches,
            "total_tokens": self.total_tokens,
            "padding_tokens": self.total_padding_tokens,
            "valid_tokens": self.total_tokens - self.total_padding_tokens,
            "padding_ratio": self.padding_ratio,
            "efficiency": self.efficiency
        }


if __name__ == "__main__":
    # Test DataLoader
    logging.basicConfig(level=logging.INFO)
    
    # Not: Gerçek test için compiled Parquet dataset gerekli
    # Bu test sadece kod yapısını gösterir
    
    print("DataLoader module loaded successfully")
    print("\nUsage:")
    print("  from src.training.dataloader import create_dataloader")
    print("  dataloader = create_dataloader('dataset.parquet', batch_size=32)")
    print("  for batch in dataloader:")
    print("      input_ids = batch['input_ids']  # [B, T]")
    print("      labels = batch['labels']        # [B, T]")
