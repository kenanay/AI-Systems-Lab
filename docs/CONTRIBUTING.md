# Katkıda Bulunma Rehberi

Local AI Research Lab projesine katkıda bulunmak istediğiniz için teşekkürler! 🎉

## Başlamadan Önce

1. [Proje İlkelerini](../.kiro/steering/00-project-principles.md) okuyun
2. [Kod Standartlarını](../.kiro/steering/04-code-standards.md) inceleyin
3. Mevcut issue'lara göz atın veya yeni bir issue açın

## Geliştirme Ortamını Kurma

```bash
# Repository'yi fork edin ve klonlayın
git clone https://github.com/your-username/local-ai-research-lab.git
cd local-ai-research-lab

# Python sanal ortamı oluşturun
python -m venv venv
source venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
pip install -e ".[dev]"

# Frontend bağımlılıklarını yükleyin
cd frontend
npm install
cd ..
```

## Geliştirme Süreci

### 1. Branch Oluşturma

```bash
git checkout -b feature/my-new-feature
# veya
git checkout -b fix/bug-fix-description
```

### 2. Kod Yazma

**Önemli Kurallar:**

- ✅ Type hints kullanın
- ✅ Docstring ekleyin
- ✅ Tensor operasyonlarında shape bilgisi belirtin
- ✅ Logger kullanın
- ✅ Test yazın
- ✅ PII ve güvenlik kontrollerini unutmayın

**Örnek Kod:**

```python
def process_attention(
    q: torch.Tensor,  # [batch, heads, seq_len, d_k]
    k: torch.Tensor,  # [batch, heads, seq_len, d_k]
    v: torch.Tensor,  # [batch, heads, seq_len, d_v]
) -> torch.Tensor:
    """
    Scaled dot-product attention hesaplar.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        
    Returns:
        Attention output, shape [batch, heads, seq_len, d_v]
    """
    logger.debug(f"Attention: q={q.shape}, k={k.shape}, v={v.shape}")
    
    # ... implementation
    
    return output
```

### 3. Test Yazma

```python
# tests/test_attention.py
import pytest
import torch
from src.model.attention import MultiHeadAttention

def test_attention_output_shape():
    """Attention output shape testi"""
    batch_size = 2
    seq_len = 10
    d_model = 64
    n_heads = 8
    
    x = torch.randn(batch_size, seq_len, d_model)
    attn = MultiHeadAttention(d_model, n_heads)
    
    output = attn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
```

### 4. Kod Kalitesi Kontrolü

```bash
# Linting
pylint src/

# Type checking
mypy src/

# Formatting
black src/
isort src/

# Tests
pytest tests/
```

### 5. Commit ve Push

```bash
git add .
git commit -m "feat: add multi-head attention implementation"
git push origin feature/my-new-feature
```

**Commit Mesajı Formatı:**

- `feat:` Yeni özellik
- `fix:` Bug fix
- `docs:` Dokümantasyon
- `test:` Test ekleme/düzeltme
- `refactor:` Kod yeniden yapılandırma
- `style:` Stil değişiklikleri
- `chore:` Bakım işleri

### 6. Pull Request Oluşturma

1. GitHub'da repository'nize gidin
2. "Pull Request" oluşturun
3. Şablon doldurulmalı:
   - Ne değişti?
   - Neden değişti?
   - Test edildi mi?
   - Breaking change var mı?

## Kod İnceleme Süreci

PR'ınız şunları kontrol eder:

- ✅ Kod standartlarına uygunluk
- ✅ Test coverage
- ✅ Dokümantasyon
- ✅ Performance
- ✅ Güvenlik
- ✅ Kiro hooks geçişi

## Özel Alanlar İçin Kurallar

### Veri İşleme Modülleri

- PII detection **mutlaka** yapılmalı
- Lisans ve telif hakları kontrol edilmeli
- Data lineage için metadata eklenמeli
- Immutable dataset ilkesine uyulmalı

### Model Modülleri

- Her tensor operasyonunda shape belirtilmeli
- Öğrenme modu ve profesyonel mod kod ayrımı yapılmalı
- Matematiksel formül açıklaması eklenmeli

### Öğrenme İçeriği

- [Learning Content Standard](../.kiro/steering/02-learning-content-standard.md) takip edilmeli
- Prerequisite'lar tanımlanmalı
- Sezgisel açıklama + matematik + kod üçlüsü olmalı

## Yardım Alma

- 💬 Issue'larda soru sorun
- 📧 Maintainer'lara ulaşın
- 📚 Dokümantasyonu okuyun

## Davranış Kuralları

- Saygılı olun
- Yapıcı geri bildirim verin
- Öğrenmeye açık olun
- Başkalarının katkılarını takdir edin

---

Katkılarınız için teşekkürler! 🚀
