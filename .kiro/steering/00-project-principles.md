---
inclusion: always
priority: high
---

# Local AI Research Lab - Temel Proje İlkeleri

Bu proje bir yapay zeka laboratuvarı platformu geliştiriyor. Aşağıdaki ilkeler **tüm kod ve dokümantasyon** için geçerlidir.

## 1. Açıklama-Önce (Explanation-First) İlkesi

**Her önemli işlem açıklanmalıdır.** Kod yazmadan önce kullanıcının şu soruları anlayabileceğinden emin ol:

- Ne yapıyoruz?
- Neden yapıyoruz?
- Bu adım bir önceki adımdan nasıl çıktı?
- Hangi veri giriş, çıktı tam olarak nedir?
- Matematiksel karşılığı nedir?
- Kodda hangi sınıf/fonksiyon bunu gerçekleştiriyor?
- Gerçek LLM/VLM sistemlerinde bunun karşılığı nedir?

## 2. Hiçbir Kritik Adımı Atlama

Automation kavramsal adımları görünmez yapmamalı. Kullanıcı:
```
Dataset → Sample → Tokenization → Batch → Embedding → Transformer → Logits → Target → Loss → Backward → Gradient → Optimizer Step → Updated Weights
```
zincirini istediği zaman açıp her kutuyu en düşük anlamlı seviyeye kadar inceleyebilmeli.

## 3. Modelden Bağımsız Veri Altyapısı

**YANLIŞ:**
```
Dataset → Sadece Qwen formatı
```

**DOĞRU:**
```
Canonical Dataset → Dataset Compiler → Qwen/Llama/Gemma/RAG/VLM
```

Ana veri yapısı belirli bir modele göre tasarlanmamalı.

## 4. Ham Verinin Korunması

`raw/` klasörü yalnızca orijinal kaynakları içermeli. Ham veri **hiçbir zaman** doğrudan değiştirilmemeli. Temizlenmiş veya dönüştürülmüş içerikler ayrı katmanlarda tutulmalı.

## 5. Veri ile Eğitim Verisi Ayrılmalı

Her veri model eğitimine uygun değildir:
- **Kurumsal/güncel bilgi** → RAG
- **Tablosal veri** → SQL/DuckDB/Python
- **Model davranışı ve görev örnekleri** → SFT/LoRA
- **Büyük metin koleksiyonları** → Pretraining
- **Görsel + metin ilişkileri** → VLM Training

## 6. Progressive Disclosure

Aynı anda tüm karmaşıklığı gösterme. Üç derinlik seviyesi kullan:

- **Seviye 1** — Özet
- **Seviye 2** — Öğrenme / Açıklama
- **Seviye 3** — Sistem / İleri Teknik Ayrıntı

## 7. Ön Bilgi (Prerequisite) Farkındalığı

Eksik ön bilgi varsa kullanıcıyı engelleme, uyar:
```
✓ Vektör
✓ Matris çarpımı
○ Softmax
○ Türev
○ Chain Rule
```

## 8. Local-First, Execution-Target-Independent

İlk çalışma ortamı yerel bilgisayar ama mimari yalnızca yerel çalışmaya bağlanmamalı:
```
CPU → Local GPU → Remote GPU → Multi-GPU → Cluster
```

## 9. Öğrenme ve Profesyonel Mod Bir Arada

- **Öğrenme Modu**: İşlemleri adım adım gösterir, matrisleri gösterir, her adımı açıklar
- **Profesyonel Mod**: Performans odaklıdır, optimize GPU operasyonları kullanır

## 10. Simülasyon ile Gerçek İşlem Açıkça Ayrılmalı

Her çıktı şu sınıflardan biriyle etiketlenmeli:
- `REAL EXECUTION`
- `SIMULATION`
- `ESTIMATION`
- `EDUCATIONAL APPROXIMATION`

## Kod Standartları

- **Türkçe**: UI metinleri, açıklamalar, dokümantasyon Türkçe
- **Teknik terimler**: Self-Attention, Embedding, Tokenizer gibi standart terimler özgün biçimde
- **Kod**: Fonksiyon/sınıf isimleri İngilizce, yorumlar projenin diline uygun
- **Şema**: Her veri formatı için schema_version, parser_version tutulmalı
- **Data Lineage**: Her çıktının hangi kaynaktan üretildiği izlenebilmeli
- **Immutable Datasets**: Yayınlanmış dataset sürümü değiştirilmemeli, yeni versiyon oluşturulmalı

## Job/Worker Mimarisi

Uzun süren işlemler (PDF import, tokenizer training, model training) HTTP isteğinin içinde bloklayıcı biçimde çalıştırılmamalı. Job/worker pattern kullanılmalı.

Job durumları: `PENDING`, `RUNNING`, `PAUSED`, `COMPLETED`, `FAILED`, `CANCELLED`

## Güvenlik ve Veri Kontrolü

Her veri için:
- `pii` - Kişisel veri kontrolü
- `license` - Lisans durumu
- `copyright` - Telif hakları
- `training_allowed` - Eğitimde kullanılabilir mi
- `security_level` - PUBLIC/INTERNAL/RESTRICTED/PERSONAL

PII içeren veya lisansı uygun olmayan veri otomatik olarak training dataset'e dahil edilmemeli.
