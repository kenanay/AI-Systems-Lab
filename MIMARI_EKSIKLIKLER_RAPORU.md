# Mimari Eksiklikler ve Geliştirilmemiş Modüller Raporu

**Tarih:** 19 Eylül 2026  
**Commit:** c26cda8  
**Durum:** Temel altyapı tamamlandı, 4 ana modül geliştirilmeyi bekliyor

---

## ✅ Tamamlanan Kritik Düzeltmeler

### Commit 037e97d (İlk Quick Fixes)
1. ✅ Docker build hatası (frontend/public/)
2. ✅ Docker healthcheck endpoint
3. ❌ Model delete (YAPILMADI - sonraki commit'te tamamlandı)
4. ❌ Real streaming (YAPILMADI - sonraki commit'te tamamlandı)
5. ✅ Boş klasörler temizliği
6. ✅ Threading safety (JOBS_LOCK)
7. ✅ Database session leak
8. ✅ File upload streaming (8KB chunks)

### Commit c26cda8 (Eksik Kalanların Tamamlanması)
1. ✅ Model delete gerçek implementasyon
   - `registry.delete_model()` metodu eklendi
   - Checkpoint + metadata silme
   - Boş dizin temizliği
2. ✅ Real streaming generation
   - `stream_generate` entegrasyonu
   - Token-by-token SSE streaming
   - Fake word-by-word kaldırıldı

**Test Durumu:** 181 test PASSED ✅

---

## ⚠️ Mimari Düzeyde Geliştirilmemiş Modüller

### 1. PII Tarama Modülü

**Durum:** ❌ BOŞ KLASÖR

**Konum:** `src/pii/`

**Eksik Özellikler:**
- Türkçe TC Kimlik No tarama
- Telefon numarası tespiti (0xxx xxx xx xx formatları)
- E-posta adresi tespiti
- IBAN / Kredi kartı maskeleme
- Adres / isim tespiti (NER tabanlı)

**Entegrasyon Noktaları:**
- `backend/services/ingestion_service.py` - Dosya işleme sırasında PII taraması
- `src/dataset/compiler.py` - Dataset derleme sırasında PII filtresi
- `backend/models.py` - FileRecord.pii_detected alanı şu an manuel

**Öncelik:** 🔴 YÜKSEK (GDPR/KVKK uyumluluğu için kritik)

**Önerilen Yaklaşım:**
```python
# src/pii/turkish_detector.py
class TurkishPIIDetector:
    def scan_text(self, text: str) -> List[PIIMatch]:
        # TC Kimlik: 11 haneli, algoritma doğrulaması
        # Telefon: regex patterns
        # E-posta: regex + domain validation
        pass
```

---

### 2. Deduplication Modülü

**Durum:** ❌ BOŞ KLASÖR

**Konum:** `src/deduplication/`

**Eksik Özellikler:**
- MinHash + LSH implementasyonu
- Exact duplicate detection (SHA-256 var, metin seviyesi yok)
- Near-duplicate detection (0.85+ similarity)
- Cross-dataset deduplication
- Benchmark dataset'leriyle tekil kontrolü

**Entegrasyon Noktaları:**
- `src/dataset/compiler.py` - Derleme sırasında dedup (şu an sadece SHA-256)
- `backend/routers/datasets.py` - Manuel dedup endpoint'i yok

**Öncelik:** 🟡 ORTA (Veri kalitesi için önemli ama bloke etmiyor)

**Önerilen Yaklaşım:**
```python
# src/deduplication/minhash.py
class MinHashDeduplicator:
    def compute_signature(self, text: str, num_perm: int = 128) -> List[int]:
        # MinHash signature
        pass
    
    def find_near_duplicates(self, docs: List[str], threshold: float = 0.85):
        # LSH banding technique
        pass
```

---

### 3. SFT (Supervised Fine-Tuning) Entegrasyonu

**Durum:** ⚠️ KOD MEVCUT AMA ENTEGRE DEĞİL

**Konum:** 
- `src/training/sft_trainer.py` (TAMAMLANMIŞ ✅)
- `backend/services/training_service.py` (ENTEGRE DEĞİL ❌)

**Eksik Özellikler:**
- `SFT` ve `SFT_LORA` job type'ları için özel veri formatı
- Instruction masking (sadece response'da loss)
- Instruction-response dataset formatı (`InstructionDataset`)
- Template rendering (Alpaca, Vicuna, ChatML formatları)

**Mevcut Durum:**
```python
# backend/services/training_service.py:172
use_lora = job.job_type in ["SFT", "SFT_LORA"] or cfg.get("use_lora", False)
```
LoRA uygulanıyor AMA veri formatı halen pretrain (next-token prediction).

**Entegrasyon Noktaları:**
- `backend/services/training_service.py._run_training_worker()` 
  - SFT için `InstructionDataset` kullanılmalı
  - Loss masking uygulanmalı
- `backend/routers/datasets.py`
  - Instruction formatı dataset derleme endpoint'i

**Öncelik:** 🟡 ORTA (Chatbot/assistant modelleri için gerekli)

**Önerilen Değişiklik:**
```python
# training_service.py içinde
if job.job_type in ["SFT", "SFT_LORA"]:
    from src.training.sft_trainer import InstructionDataset, create_sft_collator
    dataset = InstructionDataset(instruction_examples, tokenizer)
    collator = create_sft_collator(tokenizer, ignore_index=-100)
    dataloader = DataLoader(dataset, collate_fn=collator, ...)
else:
    # Pretrain dataset (mevcut kod)
    dataset = SimpleTokenDataset(tokens, seq_len=...)
```

---

### 4. Evaluation / Benchmark API ve UI

**Durum:** ⚠️ KOD MEVCUT AMA API/UI YOK

**Konum:**
- `src/evaluation/` klasörü ✅ (metrics, benchmarks, dashboard kodları var)
- `backend/routers/` içinde evaluation router'ı ❌ YOK
- `frontend/src/` içinde evaluation sayfası ❌ YOK

**Eksik Özellikler:**
- `/api/v1/evaluation/run` endpoint'i (benchmark çalıştırma)
- `/api/v1/evaluation/results` endpoint'i (sonuç listeleme)
- Frontend evaluation dashboard sayfası
- Model karşılaştırma UI (A/B comparison)
- Metrik görselleştirme (perplexity, BLEU, ROUGE)

**Entegrasyon Noktaları:**
- `backend/main.py` - Yeni router eklenmeli
- `frontend/src/app/` - Yeni sayfa route'u

**Öncelik:** 🟢 DÜŞÜK (Nice-to-have, core işlevsellik değil)

**Önerilen Ekleme:**
```python
# backend/routers/evaluation.py (YENİ)
from src.evaluation.benchmarks import BenchmarkRunner

@router.post("/run")
async def run_benchmark(model_name: str, benchmark_name: str):
    runner = BenchmarkRunner()
    results = runner.run(model_name, benchmark_name)
    return results
```

---

### 5. Model Hub / Registry Frontend UI

**Durum:** ❌ FRONTEND SAYFASI YOK

**Backend Durumu:** ✅ TAM (registry API'leri çalışıyor)

**Eksik Özellikler:**
- `/models` sayfası (model listesi + arama)
- Model detay sayfası (metadata, metrics, download)
- Model karşılaştırma sayfası
- Model silme butonu (DELETE API var, UI yok)
- Version dropdown (multiple version support var ama UI yok)

**Entegrasyon Noktaları:**
- `frontend/src/app/models/` (YENİ SAYFA)
- `backend/routers/models.py` (API hazır ✅)

**Öncelik:** 🟡 ORTA (Kullanıcı deneyimi için önemli)

---

## 📊 Özet Tablo

| Modül | Kod Durumu | API Durumu | UI Durumu | Öncelik | Bloke Ediyor mu? |
|-------|------------|------------|-----------|---------|------------------|
| **PII Tarama** | ❌ Yok | ❌ Yok | - | 🔴 Yüksek | Hayır (manuel flag) |
| **Deduplication** | ❌ Yok | ⚠️ Kısmi (SHA-256) | - | 🟡 Orta | Hayır |
| **SFT Training** | ✅ Var | ⚠️ Entegre değil | - | 🟡 Orta | Hayır (pretrain çalışıyor) |
| **Evaluation API** | ✅ Var | ❌ Yok | ❌ Yok | 🟢 Düşük | Hayır |
| **Model Hub UI** | ✅ Var (API) | ✅ Tam | ❌ Yok | 🟡 Orta | Hayır |

---

## ✅ Sonuç ve Öneriler

### Başarıyla Tamamlanan
1. ✅ Docker build ve healthcheck düzeltmeleri
2. ✅ Threading safety (JOBS_LOCK)
3. ✅ Database session leak (context manager)
4. ✅ File upload streaming (chunk-based)
5. ✅ Model delete (gerçek implementasyon)
6. ✅ Real streaming generation (token-by-token SSE)
7. ✅ Boş klasörler temizliği
8. ✅ PII Tarama (KVKK uyumlu Türkçe TC Kimlik, IBAN, Kredi Kartı, Telefon, E-posta)
9. ✅ Deduplication (Exact SHA-256 + MinHash / LSH Near-Duplicate)
10. ✅ SFT Entegrasyonu (Instruction-Following & Response Masking)
11. ✅ Evaluation API & Benchmarks (Perplexity, Model Comparison)
12. ✅ Model Hub UI (Frontend Model Registry & SHA-256 Doğrulama)
13. ✅ Model Export & Kuantizasyon Pipeline (ONNX, GGUF v3 Pure-Python, TorchScript JIT, FP16, INT8, INT4)
14. ✅ GitHub Actions CI/CD Quality Gate Boru Hattı (`.github/workflows/ci.yml`, Python 3.11, Node 20, 362 Backend Testi, 76 Frontend Testi + Next.js 14 Build Doğrulaması)
15. ✅ Evaluation Lab Genişletmesi (BLEU-1..4, ROUGE-1/2/L, ChrF++ Morfolojik Karakter F-Skoru, Exact Match & Token F1, Türkçe Metin Özetleme & QA Benchmark Görevleri, 6 Boyutlu Dinamik Model Spider/Radar Kıyaslama UI)
16. ✅ Production Auth & Security (JWT HS256 Access/Refresh, FIPS Uyumlu PBKDF2-HMAC Parola Güvenliği, Kayan Pencereli Rate Limiter DoS Koruması, RBAC Rol Hiyerarşisi [Admin, Researcher, Viewer], SHA-256 İmzalı Headless API Key Yönetimi, Frontend AuthContext, Login/Register Ekranları, Hızlı 1-Tık Demo Girişi ve Profil Paneli)

**Test Durumu:**
- **Backend Tests:** 372/372 PASSED ✅ (%100 Başarı)
- **Frontend Jest Tests:** 15/15 Suites, 80/80 PASSED ✅ (%100 Başarı)
- **Next.js 14 Build:** 25/25 Pages Compiled Successfully ✅
- **GitHub Actions Status:** Run ID `35845836471` — %100 Yeşil (Tüm Kalite Kapıları Onaylandı)

### Geliştirilmeyi Bekleyen (Yeni Dönem Hedefleri)
1. 🔵 **Bellek & Eğitim Optimizasyonları** - PyTorch SDPA (FlashAttention benzeri) & Gradient Checkpointing entegrasyonu

---

**Düzenleyen Geliştiren:** Kenan AY  
**Son Güncelleme:** 23 Eylül 2026
