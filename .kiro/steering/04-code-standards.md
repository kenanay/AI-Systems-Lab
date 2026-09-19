---
inclusion: always
priority: high
---

# Kod Standartları ve Yazılım Mimarisi

## Dil ve Terim Standardı

### Genel Kural
- **UI metinleri**: Türkçe
- **Dokümantasyon**: Türkçe
- **Kod**: Fonksiyon/sınıf isimleri İngilizce
- **Yorumlar**: Projenin diline uygun (Türkçe)
- **Teknik terimler**: Standartlaşmış terimler özgün biçimde

### Teknik Terim Kullanımı

Zorla Türkçeleştirilmeyecek terimler:
```
Self-Attention (öz-dikkat)
Embedding
Tokenizer
Gradient (gradyan)
Loss Function (kayıp fonksiyonu)
Backpropagation
Forward Pass
Checkpoint
Fine-Tuning
Learning Rate
Batch
Epoch
Logits
```

Kural:
- İlk kullanımda terim kısa Türkçe açıklamasıyla verilebilir
- Kod, API, class, function ve framework isimleri özgün biçimleriyle korunur
- Terimin sektörde yaygın İngilizce kullanımı varsa ana etiket İngilizce kalabilir
- Aynı kavram farklı ekranlarda farklı Türkçe karşılıklarla adlandırılmaz

## Python Kod Standartları

### Genel Yapı

```python
"""
Modül açıklaması - Ne yapıyor, neden var

Bu modül veri ingestion pipeline'ının bir parçasıdır.
PDF dosyalarından metin ve metadata çıkarır.
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PDFParser:
    """
    PDF dosyalarından metin ve metadata çıkarır.
    
    Args:
        extract_images: Görselleri de çıkar
        ocr_enabled: OCR kullan
    """
    
    def __init__(self, extract_images: bool = False, ocr_enabled: bool = False):
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        
    def parse(self, file_path: str) -> Dict:
        """
        PDF dosyasını parse eder.
        
        Args:
            file_path: PDF dosya yolu
            
        Returns:
            Dict içinde:
                - text: str - Çıkarılan metin
                - metadata: Dict - Dosya metadata
                - pages: List[Dict] - Sayfa detayları
                
        Raises:
            PDFParseError: PDF parse edilemezse
        """
        logger.info(f"PDF parsing başladı: {file_path}")
        
        # ... implementation
        
        return result
```

### Type Hints

**Her fonksiyonda** type hints kullanılmalı:

```python
# YANLIŞ
def process_data(data, config):
    return result

# DOĞRU
def process_data(
    data: List[Dict[str, Any]], 
    config: DataConfig
) -> ProcessedData:
    return result
```

### Shape Dokümantasyonu

Tensor operasyonlarında shape bilgisi **mutlaka** belirtilmeli:

```python
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Scaled dot-product attention hesaplar.
    
    Args:
        q: Query tensor, shape [batch, heads, seq_len, d_k]
        k: Key tensor, shape [batch, heads, seq_len, d_k]
        v: Value tensor, shape [batch, heads, seq_len, d_v]
        
    Returns:
        Attention output, shape [batch, heads, seq_len, d_v]
    """
    # q shape: [B, H, T, D]
    # k shape: [B, H, T, D]
    # k.T shape: [B, H, D, T]
    scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
    
    # Scaling
    scores = scores / math.sqrt(q.size(-1))
    
    # Softmax
    weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
    
    # v shape: [B, H, T, D]
    # output shape: [B, H, T, D]
    output = weights @ v
    
    return output
```

### Error Handling

```python
# Özel exception'lar tanımla
class DatasetError(Exception):
    """Dataset işlemlerinde hata"""
    pass

class TokenizerError(Exception):
    """Tokenizer işlemlerinde hata"""
    pass

# Kullanım
try:
    result = parse_pdf(file_path)
except FileNotFoundError:
    logger.error(f"Dosya bulunamadı: {file_path}")
    raise DatasetError(f"PDF dosyası mevcut değil: {file_path}")
except Exception as e:
    logger.error(f"PDF parse hatası: {e}")
    raise DatasetError(f"PDF parse edilemedi: {e}")
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Log seviyeleri
logger.debug("Detaylı debug bilgisi")
logger.info("Genel bilgi mesajı")
logger.warning("Uyarı mesajı")
logger.error("Hata mesajı")
logger.critical("Kritik hata")

# Context bilgisi ekle
logger.info(
    "Dataset oluşturuldu",
    extra={
        "dataset_id": dataset_id,
        "file_count": len(files),
        "version": version
    }
)
```

## Modül Organizasyonu

### src/ Yapısı

Her modül kendi dizininde ve net sorumluluk alanıyla:

```python
# src/ingestion/pdf.py
class PDFParser:
    """PDF parsing sorumluluğu"""
    pass

# src/tokenizer/bpe.py
class BPETokenizer:
    """BPE tokenization sorumluluğu"""
    pass

# src/model/attention.py
class MultiHeadAttention:
    """Attention mechanism sorumluluğu"""
    pass
```

### Base Classes

Ortak interface için base class kullan:

```python
# src/ingestion/base.py
from abc import ABC, abstractmethod

class BaseParser(ABC):
    """Tüm parser'ların base class'ı"""
    
    @abstractmethod
    def parse(self, file_path: str) -> Dict:
        """Dosyayı parse et"""
        pass
    
    @abstractmethod
    def validate(self, file_path: str) -> bool:
        """Dosya geçerli mi kontrol et"""
        pass
```

## Configuration Management

```python
# configs/model_config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model konfigürasyonu"""
    vocab_size: int = 8000
    context_length: int = 512
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    
    def validate(self):
        """Konfigürasyon geçerli mi kontrol et"""
        assert self.d_model % self.n_heads == 0, \
            "d_model n_heads'e bölünebilir olmalı"
        assert self.vocab_size > 0, \
            "vocab_size pozitif olmalı"
```

## Testing Standartları

```python
# tests/test_tokenizer.py
import pytest
from src.tokenizer.bpe import BPETokenizer

def test_bpe_basic():
    """BPE temel tokenization testi"""
    tokenizer = BPETokenizer(vocab_size=1000)
    
    text = "Kütahya'da eğitim"
    tokens = tokenizer.encode(text)
    
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)
    
    # Decode test
    decoded = tokenizer.decode(tokens)
    assert decoded == text

def test_bpe_empty_string():
    """Boş string testi"""
    tokenizer = BPETokenizer(vocab_size=1000)
    
    tokens = tokenizer.encode("")
    assert len(tokens) == 0

def test_bpe_shape():
    """Shape kontrolü"""
    tokenizer = BPETokenizer(vocab_size=1000)
    
    text = "test"
    tokens = tokenizer.encode(text)
    
    # tokens bir List[int] olmalı
    assert isinstance(tokens, list)
```

## Job/Worker Pattern

Uzun süren işlemler için:

```python
# src/jobs/base.py
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Job:
    job_id: str
    job_type: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = None
```

## Versioning

```python
# Her önemli bileşende version bilgisi
class PDFParser:
    VERSION = "1.0.0"
    
    def parse(self, file_path: str) -> Dict:
        return {
            "parser_name": "PDFParser",
            "parser_version": self.VERSION,
            "schema_version": "1.0.0",
            # ... data
        }
```

## Documentation

Her modül için:

```python
"""
src/model/attention.py

Multi-Head Attention implementasyonu.

Bu modül Transformer mimarisinin temel bileşeni olan
Multi-Head Attention mekanizmasını uygular.

Kaynaklar:
    - Attention Is All You Need (Vaswani et al., 2017)
    - https://arxiv.org/abs/1706.03762

Kullanım:
    >>> from src.model.attention import MultiHeadAttention
    >>> attn = MultiHeadAttention(d_model=512, n_heads=8)
    >>> output = attn(x)  # x shape: [batch, seq_len, d_model]
"""
```

## Profil ve Debugging

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    """İşlem süresini ölç"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} tamamlandı: {elapsed:.2f}s")

# Kullanım
with timer("Dataset loading"):
    dataset = load_dataset(path)
```
