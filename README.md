# Local AI Research Lab
## Local-First, Uçtan Uca AI Systems Research & Learning Platform

**Sürüm:** 1.2  
**Durum:** Development / MVP

---

## 📋 Proje Özeti

Local AI Research Lab, yapay zeka sistemlerinin nasıl çalıştığını **veriden başlayarak uygulayarak öğreten** ve aynı zamanda gerçek verileri işleyerek **model tasarımı, eğitimi, değerlendirmesi, inference, serving ve gerçek AI uygulamaları geliştirilebilen** bir araştırma laboratuvarı platformudur.

### Temel Özellikler

- ✅ **Açıklama-Önce Yaklaşımı**: Her işlem matematiksel, algoritmik ve sistem seviyesinde açıklanır
- 🎓 **Öğrenme ve Profesyonel Mod**: Hem öğrenmek hem üretmek için
- 📊 **Modelden Bağımsız Veri Altyapısı**: Canonical dataset → çoklu model formatları
- 🔒 **Güvenlik ve PII Kontrolü**: Veri güvenliği ilk tasarımda
- 🔄 **Data Lineage ve Versioning**: İzlenebilir ve yeniden üretilebilir sistem
- 🚀 **Local-First, Execution-Target-Independent**: CPU'dan cluster'a ölçeklenebilir

---

## 🎯 Hedef AI Yaşam Döngüsü

```
Ham Veri
   ↓
Veri Alma / Ingestion
   ↓
Veri Formatları ve Depolama
   ↓
Normalize Etme / Temizleme / Kalite
   ↓
Canonical Dataset
   ↓
Tokenizer
   ↓
Tensor / Embedding
   ↓
Neural Network Temelleri
   ↓
Attention
   ↓
Transformer ve Diğer Mimariler
   ↓
Pretraining
   ↓
Post-Training (SFT, LoRA, DPO)
   ↓
Evaluation
   ↓
Inference
   ↓
Serving
   ↓
RAG / Tools / Agents / AI Applications
   ↓
Multimodal AI
   ↓
Distributed AI
   ↓
Production / MLOps / Monitoring
```

---

## 🏗️ Proje Yapısı

```
local-ai-research-lab/
│
├── frontend/                 # React/Next.js UI
├── backend/                  # FastAPI backend
│
├── src/                      # Ana kaynak kod
│   ├── ingestion/           # Veri alma (PDF, DOCX, CSV, vb.)
│   ├── normalization/       # Veri normalizasyon
│   ├── cleaning/            # Veri temizleme
│   ├── deduplication/       # Duplicate detection
│   ├── quality/             # Kalite kontrolü
│   ├── pii/                 # PII detection
│   ├── dataset/             # Dataset yönetimi
│   ├── tokenizer/           # Tokenizer (BPE, WordPiece, vb.)
│   ├── tensor/              # Tensor operasyonları
│   ├── embeddings/          # Embedding
│   ├── attention/           # Attention mekanizması
│   ├── transformer/         # Transformer blokları
│   ├── model/               # Model mimarileri
│   ├── training/            # Training loop
│   ├── post_training/       # SFT, LoRA, DPO
│   ├── evaluation/          # Model evaluation
│   ├── inference/           # Inference engine
│   ├── serving/             # Model serving
│   ├── rag/                 # RAG sistemi
│   ├── applications/        # AI uygulamaları
│   ├── distributed/         # Distributed training
│   ├── systems/             # Hardware/sistem bilgisi
│   ├── simulators/          # Simülasyon araçları
│   ├── security/            # Güvenlik kontrolleri
│   ├── mlops/               # MLOps araçları
│   ├── finetuning/          # Fine-tuning
│   └── multimodal/          # Multimodal AI
│
├── configs/                  # Konfigürasyon dosyaları
├── datasets/                 # Dataset depolama
├── tokenizers/               # Tokenizer modelleri
├── models/                   # Eğitilmiş modeller
├── checkpoints/              # Training checkpoints
├── experiments/              # Deney kayıtları
├── evaluations/              # Evaluation sonuçları
├── notebooks/                # Jupyter notebooks
├── tests/                    # Test dosyaları
├── docs/                     # Dokümantasyon
├── scripts/                  # Utility scripts
│
└── .kiro/                    # Kiro yapılandırması
    ├── steering/            # Proje ilkeleri ve standartlar
    ├── hooks/               # Otomatik kontroller
    └── settings/            # Kiro ayarları
```

---

## 🚀 Başlangıç

### Gereksinimler

**Docker ile (Önerilen):**
- Docker 20.10+
- Docker Compose 2.0+

**Manuel kurulum için:**
- Python 3.11+
- Node.js 20+
- (Opsiyonel) CUDA-capable GPU

---

### 🐳 Docker ile Başlatma (Önerilen)

En hızlı başlangıç yolu - tek komut ile tüm sistem çalışır:

```bash
# Production mode
docker-compose up

# Development mode (hot reload)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Arka planda çalıştır
docker-compose up -d

# Logları izle
docker-compose logs -f

# Durdur
docker-compose down

# Tüm verileri temizle (dikkat: veriler silinir!)
docker-compose down -v
```

**Erişim:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Avantajlar:**
- ✅ Tek komutla başlatma
- ✅ Dependency yönetimi yok
- ✅ Tutarlı environment
- ✅ Kolay deployment

---

### 💻 Manuel Kurulum

Daha fazla kontrol ve development için:

#### 1. Repository'yi Klonla

```bash
git clone <repository-url>
cd local-ai-research-lab
```

#### 2. Backend Kurulumu

```bash
# Python sanal ortamı oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Dependencies yükle
pip install -r requirements.txt

# Backend başlat
python -m uvicorn backend.main:app --host localhost --port 8000 --reload
```

#### 3. Frontend Kurulumu (Ayrı Terminal)

```bash
# Frontend dizinine git
cd frontend

# Dependencies yükle
npm install

# Development server başlat
npm run dev
```

**Erişim:**
- Frontend: http://localhost:3000
- Backend: http://localhost:8000

---

### 🔧 Environment Variables

`.env` dosyası oluştur (opsiyonel):

```bash
# Backend
DATABASE_URL=sqlite:///data/local_ai_lab.db
LOG_LEVEL=INFO

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
API_URL=http://backend:8000  # Docker için
```

---

## 📚 Temel Kavramlar

### Açıklama-Önce İlkesi

Her önemli işlem için şu sorular cevaplanır:
- **Ne yapıyoruz?**
- **Neden yapıyoruz?**
- **Matematiksel karşılığı nedir?**
- **Kodda nasıl karşılanıyor?**
- **Gerçek AI sistemlerinde neden kullanılıyor?**

### Öğrenme Modu vs Profesyonel Mod

| Öğrenme Modu | Profesyonel Mod |
|--------------|-----------------|
| İşlemleri adım adım gösterir | Performans odaklıdır |
| Matrisleri gösterir | Optimize GPU operasyonları |
| Attention elle hesaplanabilir | PyTorch SDPA kullanır |
| Basit training loop | Gelişmiş trainer |
| Her adımı açıklar | Otomatikleştirilmiştir |

### Progressive Disclosure

Karmaşıklık üç seviyede sunulur:
- **Seviye 1** — Özet (kullanıcı arayüzü)
- **Seviye 2** — Öğrenme / Açıklama (matematik ve algoritma)
- **Seviye 3** — Sistem / İleri Teknik Ayrıntı (donanım ve optimize kod)

---

## 🎓 Öğrenme Yolu

Platform üç ana çalışma yüzeyi sunar:

### 1. Guided Journey
Basitten zora sıralı öğrenme yolu. Veri → Tokenizer → Tensor → Neural Network → Attention → Transformer → Training → Inference → RAG → Distributed → Production

### 2. Lab & Simulator
Kavramları bağımsız olarak deneyleyebileceğin alanlar:
- Attention Simulator
- Tokenizer Lab
- GPU Memory Simulator
- RAG Simulator
- Distributed Training Simulator

### 3. Workspace
Gerçek projelerin yürütüldüğü alan. Dataset oluşturma, model eğitme, evaluation, deployment.

---

## 🔧 Geliştirme İlkeleri

### 1. Ham Verinin Korunması
`raw/` klasörü değiştirilmez. Türev veriler ayrı katmanlarda.

### 2. Modelden Bağımsız Veri
Canonical Dataset → Dataset Compiler → Çoklu format

### 3. Veri ile Eğitim Verisi Ayrımı
- Kurumsal/güncel bilgi → RAG
- Tablosal veri → SQL/DuckDB
- Model davranışı → SFT/LoRA
- Büyük metin → Pretraining
- Görsel+metin → VLM

### 4. Data Lineage
Her çıktının hangi kaynaktan üretildiği izlenebilir.

### 5. Immutable Datasets
Yayınlanmış dataset sürümleri değiştirilmez, yeni versiyon oluşturulur.

### 6. PII ve Güvenlik
Kişisel veri ve lisans kontrolü varsayılan davranış.

---

## 📖 Dokümantasyon

- [Proje İlkeleri](.kiro/steering/00-project-principles.md)
- [Matematik Açıklama Standardı](.kiro/steering/01-math-explanation-standard.md)
- [Öğrenme İçeriği Standardı](.kiro/steering/02-learning-content-standard.md)
- [Veri Yönetimi](.kiro/steering/03-data-management.md)
- [Kod Standartları](.kiro/steering/04-code-standards.md)
- [Detaylı Uygulama Planı](AI_Research_Lab_Uygulama_Plani_v1.2.md)

---

## 🛠️ Teknoloji Stack'i

### Backend
- **Python 3.10+**
- **FastAPI** - API framework
- **PyTorch** - Deep learning
- **Transformers** - Hugging Face
- **PyArrow / Parquet** - Veri formatı
- **DuckDB** - Analitik sorgular
- **SQLite/PostgreSQL** - Metadata

### Frontend
- **React 18+**
- **Next.js** - Framework
- **TypeScript** - Type safety
- **TailwindCSS** - Styling
- **KaTeX/MathJax** - Matematik gösterimi

### AI/ML
- **PyTorch** - Temel framework
- **Transformers** - Model library
- **Tokenizers** - Fast tokenization
- **Datasets** - Dataset library
- **PEFT** - LoRA/QLoRA
- **FAISS/Qdrant** - Vector database

---

## 🎯 MVP Özellikleri (Faz 1)

### Veri
- ✅ TXT, MD, PDF, CSV, XLSX import
- ✅ Parquet export
- ✅ SHA-256 hash
- ✅ Train/validation/test split

### Öğrenme
- ✅ Tokenizer Lab (Character, Word, BPE)
- ✅ Tensor Lab
- ✅ Embedding Lab
- ✅ Attention Lab
- ✅ Transformer Block

### Model
- ✅ Mini-GPT (10M-30M parametre)
- ✅ Training loop
- ✅ Checkpoint sistemi
- ✅ Live loss tracking

### Inference
- ✅ Text generation
- ✅ Basic sampling (temperature, top-k, top-p)

---

## 📊 İlerleme Durumu

- [x] Proje yapısı
- [x] Kiro yapılandırması (steering + hooks)
- [ ] Backend API temel yapısı
- [ ] Frontend temel yapısı
- [ ] Veri ingestion pipeline
- [ ] Tokenizer Lab
- [ ] Attention Lab
- [ ] Mini-GPT implementasyonu
- [ ] Training loop
- [ ] Inference engine

---

## 🤝 Katkıda Bulunma

Bu proje açık bir araştırma ve öğrenme platformudur. Katkılarınızı bekliyoruz!

### Katkı Rehberi

1. `.kiro/steering/` altındaki ilkeleri okuyun
2. Issue açın veya mevcut issue'lardan birini seçin
3. Feature branch oluşturun
4. Kod standartlarına uygun geliştirme yapın
5. Test ekleyin
6. Pull request gönderin

### Kod Standartları

- Type hints kullanın
- Docstring ekleyin
- Tensor operasyonlarında shape bilgisi belirtin
- PII ve güvenlik kontrollerini unutmayın
- Data lineage için metadata ekleyin

---

## 👨‍💻 Geliştirici

**Kenan AY**  
*Düzenleyen ve Geliştiren*

Bu proje Kenan AY tarafından düzenlenmiş ve geliştirilmiştir.

---

## 📝 Lisans

[Lisans bilgisi eklenecek]

---

## 📧 İletişim

Proje Geliştiricisi: **Kenan AY**

[İletişim bilgileri eklenecek]

---

## 🙏 Teşekkürler

Bu proje şu kaynaklardan ilham almıştır:
- Attention Is All You Need (Vaswani et al., 2017)
- Hugging Face Transformers
- PyTorch Tutorials
- Fast.ai
- Andrej Karpathy's tutorials

---

## ⚡ Hızlı Başlangıç Komutları

### Docker ile (Önerilen)

```bash
# Tek komut - tüm sistem başlar
docker-compose up

# Arka planda çalıştır
docker-compose up -d

# Logları izle
docker-compose logs -f backend
docker-compose logs -f frontend

# Durdur
docker-compose down
```

### Manuel Kurulum

```bash
# Proje kurulumu
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Backend başlat (terminal 1)
python -m uvicorn backend.main:app --host localhost --port 8000 --reload

# Frontend başlat (terminal 2)
cd frontend && npm run dev

# Test çalıştır
pytest tests/ -v

# Lint kontrolü
pylint src/ backend/

# Type check
mypy src/ backend/
npm run type-check  # Frontend
```

### Kullanışlı Docker Komutları

```bash
# Image'ları yeniden build et
docker-compose build

# Sadece backend rebuild
docker-compose build backend

# Shell aç (debug için)
docker-compose exec backend bash
docker-compose exec frontend sh

# Veritabanını sıfırla
docker-compose down -v
docker-compose up -d

# Resource kullanımını izle
docker stats
```

---

**Not:** Bu proje aktif geliştirme aşamasındadır. Özellikler ve API'lar değişebilir.

---

## 🔧 Troubleshooting

### Docker İle İlgili Sorunlar

**Problem: Port zaten kullanımda**
```bash
# Çalışan process'i bul
lsof -i :8000  # Backend
lsof -i :3000  # Frontend

# Process'i durdur
kill -9 <PID>

# Veya Docker Compose ile temiz başlangıç
docker-compose down
docker-compose up
```

**Problem: Image build hatası**
```bash
# Cache'siz rebuild
docker-compose build --no-cache

# Tek servis rebuild
docker-compose build --no-cache backend
```

**Problem: Volume izin hatası**
```bash
# Volume'ları temizle
docker-compose down -v

# Dizin izinlerini kontrol et
sudo chown -R $USER:$USER datasets/ models/ uploads/
```

**Problem: Container başlamıyor**
```bash
# Logları kontrol et
docker-compose logs backend
docker-compose logs frontend

# Container'a bağlan (debug)
docker-compose exec backend bash
```

### Manuel Kurulum Sorunları

**Problem: Python package yüklenemiyor**
```bash
# pip güncelle
pip install --upgrade pip

# Sistem dependencies (Ubuntu/Debian)
sudo apt-get install python3-dev build-essential

# macOS
brew install python@3.11
```

**Problem: Node.js build hatası**
```bash
# Node modules temizle
cd frontend
rm -rf node_modules package-lock.json
npm install

# Node version kontrol et
node --version  # 20.x olmalı
```

**Problem: Database locked hatası**
```bash
# SQLite WAL mode otomatik aktif
# Eğer hala sorun varsa:
rm data/local_ai_lab.db*
# Uygulama yeniden başlatıldığında DB oluşturulur
```

**Problem: CUDA/GPU tanınmıyor**
```bash
# PyTorch CUDA version kontrol et
python -c "import torch; print(torch.cuda.is_available())"

# CPU-only kullan (requirements.txt'te torch değiştir)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Frontend Sorunları

**Problem: API bağlantı hatası**
```bash
# Backend'in çalıştığından emin ol
curl http://localhost:8000/health

# CORS hatası varsa backend/main.py'de CORS ayarları kontrol et
# Environment variable'ı kontrol et
echo $NEXT_PUBLIC_API_URL
```

**Problem: Build çok yavaş**
```bash
# ESLint cache temizle
rm -rf frontend/.next frontend/node_modules/.cache

# Type check'i atla (sadece development)
npm run build -- --no-lint
```

### Test Sorunları

**Problem: Test fail ediyor**
```bash
# Verbose mode ile detay
pytest tests/ -vv

# Specific test
pytest tests/test_model.py::test_attention -v

# Coverage report
pytest tests/ --cov=src --cov-report=html
```


---

## 📊 Sprint Status & Test Coverage

### Current Sprint: Sprint 1 - Quick Wins ✅ COMPLETED

**Completion Date:** 22 Eylül 2026  
**Platform Score:** 7.4 → 7.8 (+0.4)

#### Sprint 1 Achievements:
- ✅ BLEU/ROUGE Metrics - Fully functional
- ✅ Deduplication (MinHash/LSH) - Integrated & tested
- ✅ Quality Scoring - Automated scoring implemented
- ✅ Frontend Test Infrastructure - 47/48 tests passing
- ✅ Backend Tests - 300/300 passing
- ✅ Documentation Updates

#### Test Coverage Summary:

**Backend Tests:**
```
Total: 300 tests
Status: ALL PASSING ✅
Coverage: 82%
Execution: 14.82s
```

**Frontend Tests:**
```
Total: 48 tests
Passing: 47 (97.9%)
Coverage: 56.02%

Coverage by Lab:
├── Journey Lab: 85.00%
├── Systems Lab: 80.00%
├── Synthetic Lab: 65.44%
├── RAG Lab: 62.74%
├── Evaluation: 62.00%
├── Tensor Lab: 59.67%
├── Transformer Lab: 61.19%
├── Embedding Lab: 42.63%
└── Attention Lab: 33.05%
```

**Overall Test Pass Rate:** 347/348 (99.7%) 🎉

#### Next Sprint: Sprint 2-3 - RAG Pipeline
**Duration:** 4 weeks  
**Goal:** Complete RAG implementation  
**Estimated Impact:** 7.8 → 8.3 (+0.5 points)

For detailed sprint information, see:
- [Sprint 1 Completion Report](SPRINT_1_COMPLETION_REPORT.md)
- [Remaining Development Priorities](KALAN_GELISTIRMELER_VE_ONCELIKLER.md)

---


---

## 👨‍💻 Yazar ve Geliştirici

**Kenan AY**  
*Proje Geliştiricisi ve Düzenleyici*  
📍 Kütahya, TÜRKİYE

**İletişim ve Katkı:**
- Proje GitHub: [Repository URL]
- Detaylı bilgi için: `AUTHORS.md`

© 2026 Local AI Research Lab - Developed by Kenan AY
