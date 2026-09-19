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

- Python 3.10+
- Node.js 18+
- (Opsiyonel) CUDA-capable GPU

### Kurulum

```bash
# Repository'yi klonla
git clone <repository-url>
cd local-ai-research-lab

# Python sanal ortamı oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Python bağımlılıklarını yükle
pip install -r requirements.txt

# Frontend bağımlılıklarını yükle
cd frontend
npm install
cd ..
```

### Geliştirme

```bash
# Backend'i başlat (port 8000)
uvicorn backend.main:app --reload

# Frontend'i başlat (port 3000)
cd frontend
npm run dev
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

## 📝 Lisans

[Lisans bilgisi eklenecek]

---

## 📧 İletişim

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

```bash
# Proje kurulumu
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Backend başlat
python -m backend.main

# Frontend başlat (ayrı terminal)
cd frontend && npm run dev

# Test çalıştır
pytest tests/

# Lint kontrolü
pylint src/

# Type check
mypy src/
```

---

**Not:** Bu proje aktif geliştirme aşamasındadır. Özellikler ve API'lar değişebilir.
