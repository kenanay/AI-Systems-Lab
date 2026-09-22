# Kalan Geliştirmeler ve Öncelikler

**Tarih:** 22 Eylül 2026, 20:56  
**Uygulama Durumu:** ✅ ÇALIŞIYOR (Backend: 8000, Frontend: 3000)  
**Mevcut Skor:** 7.4/10 (74%)  
**Hedef Skor:** 9.0/10 (90%+)

---

## 📊 Mevcut Durum Özeti

### ✅ Tamamlanmış (Son Hafta)

```
✅ Model Hub Frontend UI (cf4e726)
✅ Evaluation API Router (13fc90e)
✅ SFT Training Entegrasyonu (7a3fa9b)
✅ PII Tarama Modülü (69732b3)
✅ Deduplication (MinHash/LSH) (69732b3)
✅ Real Streaming Generation
✅ 3 Interactive Lab (Tokenizer, Attention, Embedding)
✅ 190 Test (%82 coverage)
✅ 26 API Endpoint
✅ 107 Training Job Çalıştırılmış
```

### ⚠️ Eksik/Tamamlanacak

Bu rapor, **değişiklik yapmadan sadece kalan işleri** listeliyor.

---

## 🔴 KRİTİK ÖNCELİK (P0) - 2-3 Hafta

### 1. RAG Pipeline İmplementasyonu 🔴 **EN KRİTİK!**

**Neden Kritik:**
- Plan v1.2'de çok vurgulu
- Modern AI'ın temel özelliği
- Kullanıcılar tarafından en çok beklenen
- Placeholder klasör var ama kod yok

**Yapılacaklar:**

#### 1.1 Backend RAG Engine
```python
# src/rag/ klasörü oluştur
├── chunker.py              → Chunking strategies
│   - FixedTokenChunker
│   - SentenceChunker
│   - ParagraphChunker
│   - SemanticChunker
│   - RecursiveCharacterChunker
│
├── embedder.py             → Embedding generation
│   - LocalEmbedder (sentence-transformers)
│   - OpenAIEmbedder (optional)
│   - CachedEmbedder (disk cache)
│
├── vector_store.py         → Vector database interface
│   - FAISS implementation (başlangıç)
│   - Qdrant integration (optional)
│   - ChromaDB (optional)
│
├── retriever.py            → Retrieval logic
│   - SimilarityRetriever
│   - MMRRetriever (Maximum Marginal Relevance)
│   - HybridRetriever (vector + keyword)
│
└── rag_engine.py           → Main RAG orchestration
    - index_documents()
    - search(query, top_k, filters)
    - generate_with_context(query, model)
```

**Tahmini Süre:** 2 hafta  
**Dosya Sayısı:** ~8 yeni dosya  
**Satır Sayısı:** ~2,000 satır

#### 1.2 RAG API Router
```python
# backend/routers/rag.py
POST /api/v1/rag/index        → Create RAG index
GET  /api/v1/rag/indexes      → List indexes
POST /api/v1/rag/search       → Semantic search
POST /api/v1/rag/generate     → RAG generation
POST /api/v1/rag/reindex      → Rebuild index
DELETE /api/v1/rag/{id}       → Delete index
```

**Tahmini Süre:** 2-3 gün  
**Satır Sayısı:** ~400 satır

#### 1.3 RAG Lab Frontend
```typescript
// frontend/src/app/rag-lab/page.tsx
// Placeholder var, doldurulacak (~800-1000 satır)

Features:
✅ Document upload for RAG
✅ Chunking strategy selector
✅ Index creation progress
✅ Semantic search test UI
✅ Retrieved chunks visualization
✅ RAG generation with sources
✅ Comparison: With RAG vs Without RAG
```

**Tahmini Süre:** 1 hafta  
**Satır Sayısı:** ~1,000 satır

**TOPLAM RAG: 2-3 hafta, ~3,400 satır**

---

### 2. Frontend Tests 🔴 **REGRESYON RİSKİ!**

**Neden Kritik:**
- 0 frontend test var
- Lab'lar (Embedding: 947 satır) untested
- Refactoring riski çok yüksek
- Her değişiklikte manuel test gerekiyor

**Yapılacaklar:**

#### 2.1 Test Setup
```bash
# Jest + React Testing Library kurulu ama config yok
jest.config.js                → Jest configuration
.env.test                     → Test environment
setupTests.ts                 → Test utilities
```

**Tahmini Süre:** 1 gün

#### 2.2 Component Tests
```typescript
// Priority sırasına göre:

1. Embedding Lab Tests (~200 satır)
   - PCA projection render
   - Word add/remove
   - Similarity calculator
   - 2D/3D toggle
   - Rotation sliders

2. Attention Lab Tests (~150 satır)
   - Heatmap render
   - Layer/head selector
   - Hover interactions
   - Query token focus

3. Tokenizer Lab Tests (~150 satır)
   - Training form
   - Token visualization
   - Compression stats

4. Core Components (~300 satır)
   - API client mocks
   - Form validations
   - Error handling
   - Loading states

5. Integration Tests (~200 satır)
   - Page navigation
   - API integration
   - State management
```

**Tahmini Süre:** 1.5 hafta  
**Satır Sayısı:** ~1,000 satır test kodu  
**Target Coverage:** 50%+ frontend coverage

**TOPLAM FRONTEND TESTS: 1.5 hafta, ~1,000 satır**

---

### 3. BLEU/ROUGE Metrics İmplementasyonu 🟡 **PLACEHOLDER!**

**Neden Önemli:**
- Evaluation API var ama sadece perplexity implement
- BLEU/ROUGE placeholder (production'da kabul edilemez)
- Model comparison için gerekli

**Yapılacaklar:**

```python
# src/evaluation/metrics.py (mevcut, doldurulacak)

def calculate_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4
) -> Dict[str, float]:
    """
    BLEU score implementation
    - Use NLTK or sacrebleu
    - Return BLEU-1, BLEU-2, BLEU-3, BLEU-4
    """

def calculate_rouge(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    ROUGE score implementation
    - Use rouge-score library
    - Return ROUGE-1, ROUGE-2, ROUGE-L
    """

def calculate_meteor(
    predictions: List[str],
    references: List[str]
) -> float:
    """Optional: METEOR score"""

def calculate_bertscore(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """Optional: BERTScore (semantic)"""
```

**Dependencies:**
```bash
pip install nltk rouge-score sacrebleu
# optional: pip install bert-score
```

**Tahmini Süre:** 3-4 gün  
**Satır Sayısı:** ~300 satır

**TOPLAM METRICS: 3-4 gün, ~300 satır**

---

## 🟡 YÜKSEK ÖNCELİK (P1) - 1-2 Ay

### 4. Missing Labs (Educational Platform)

Plan'da 12 lab var, 3 tanesi implement edilmiş. **9 lab eksik:**

#### 4.1 Math Lab 🧮
```typescript
// frontend/src/app/math-lab/page.tsx

Modules:
├── Linear Algebra
│   - Vector visualization
│   - Matrix multiplication demo
│   - Dot product interactive
│   - Transpose, inverse
│
├── Calculus
│   - Derivative visualization
│   - Gradient descent demo
│   - Chain rule interactive
│
└── Probability
    - Distribution viewer
    - Expected value
    - Sampling demo
```

**Tahmini Süre:** 2 hafta  
**Satır Sayısı:** ~1,200 satır

#### 4.2 Tensor Lab 📊
```typescript
// frontend/src/app/tensor-lab/page.tsx

Features:
├── Shape Manipulation
│   - Reshape interactive
│   - Broadcasting demo
│   - Transpose, permute
│
├── dtype & device
│   - Memory visualization
│   - dtype comparison
│   - CPU ↔ GPU transfer
│
└── Operations
    - Element-wise ops
    - Reduction ops
    - Matrix ops with shapes
```

**Tahmini Süre:** 1.5 hafta  
**Satır Sayısı:** ~800 satır

#### 4.3 Neural Network Lab 🧠
```typescript
// frontend/src/app/nn-lab/page.tsx

Features:
├── Forward Pass Visualization
├── Backpropagation Step-by-Step
├── Loss Function Comparison
├── Optimizer Comparison
├── Activation Function Viewer
└── Weight Update Animation
```

**Tahmini Süre:** 2 hafta  
**Satır Sayısı:** ~1,000 satır

#### 4.4 Transformer Builder Lab 🏗️
```typescript
// frontend/src/app/transformer-builder/page.tsx

Features:
├── Block-by-Block Building
├── Config Playground
├── Parameter Count Calculator
├── Layer Visualization
└── Architecture Comparison
```

**Tahmini Süre:** 1.5 hafta  
**Satır Sayısı:** ~900 satır

#### 4.5 Systems Lab 💻
```typescript
// frontend/src/app/systems-lab/page.tsx

Features:
├── GPU Memory Simulator
├── CUDA Concepts
├── Batch Size Impact
├── Precision Comparison (fp32, fp16, bf16)
└── Throughput Calculator
```

**Tahmini Süre:** 1 hafta  
**Satır Sayısı:** ~600 satır

#### 4.6 Diğer Eksik Labs
```
❌ Architecture Atlas      → 1 hafta, ~800 satır
❌ Distributed Lab         → 1 hafta, ~700 satır
❌ Guided Journey          → 2 hafta, ~1,500 satır
❌ Knowledge Map           → 1 hafta, ~600 satır
```

**TOPLAM YENİ LABS: 10-12 hafta, ~8,000 satır**

---

### 5. Deduplication Entegrasyonu ⚠️

**Durum:**
- ✅ MinHash/LSH kod yazılmış (src/deduplication/minhash.py)
- ❌ Dataset compiler'a entegre edilmemiş

**Yapılacaklar:**

```python
# src/dataset/compiler.py içine:

def _remove_duplicates(
    self,
    texts: List[str],
    threshold: float = 0.85
) -> List[int]:
    """
    MinHash/LSH ile near-duplicate detection
    Returns: indices to keep
    """
    from src.deduplication.minhash import MinHashDeduplicator
    
    dedup = MinHashDeduplicator()
    keep_indices = dedup.find_duplicates(
        texts,
        threshold=threshold
    )
    return keep_indices

# compile_pretraining_dataset() içinde kullan
```

**Tahmini Süre:** 1 gün  
**Satır Sayısı:** ~50 satır entegrasyon

---

### 6. Quality Scoring ⚠️

**Durum:** Field var ama logic yok

**Yapılacaklar:**

```python
# src/quality/scorer.py (yeni dosya)

class QualityScorer:
    """Document quality scoring"""
    
    def score_document(self, text: str) -> float:
        """
        Score components:
        - Length (too short/long penalty)
        - Repetition ratio
        - Special char ratio
        - Sentence structure
        - Language confidence
        - OCR error indicators
        
        Returns: 0.0 - 1.0
        """

# backend/services/ingestion_service.py içinde kullan
```

**Tahmini Süre:** 3-4 gün  
**Satır Sayısı:** ~200 satır

---

## 🟢 ORTA ÖNCELİK (P2) - 2-3 Ay

### 7. Production Features (Opsiyonel - Local-first için)

#### 7.1 Toggle-based Authentication
```python
# backend/config.py
enable_auth: bool = Field(default=False)

# backend/auth.py (yeni)
class OptionalAuthMiddleware:
    """JWT auth, sadece enable_auth=true ise"""
```

**Tahmini Süre:** 1 hafta  
**Satır Sayısı:** ~300 satır

#### 7.2 Monitoring & Observability
```python
# Prometheus metrics
# Grafana dashboards
# Structured logging
```

**Tahmini Süre:** 1 hafta  
**Satır Sayısı:** ~200 satır

#### 7.3 Celery Migration
```python
# Replace threading with Celery + Redis
# Async job queue
# Better retry logic
```

**Tahmini Süre:** 1-2 hafta  
**Satır Sayısı:** ~500 satır değişiklik

#### 7.4 PostgreSQL Migration
```python
# SQLite → PostgreSQL
# Connection pooling
# Better concurrency
```

**Tahmini Süre:** 3-5 gün  
**Satır Sayısı:** ~100 satır değişiklik

---

### 8. Advanced Features

#### 8.1 Model Comparison UI Enhancement
```typescript
// backend/routers/evaluation.py - API var
// Frontend UI minimal, geliştirilebilir

Features:
├── Side-by-side comparison
├── Metric charts
├── Parameter comparison
└── Speed/Memory comparison
```

**Tahmini Süre:** 3-4 gün  
**Satır Sayısı:** ~400 satır

#### 8.2 Batch Inference
```python
# src/inference/batch_generator.py
class BatchInference:
    """Batch generation with dynamic batching"""
```

**Tahmini Süre:** 1 hafta  
**Satır Sayısı:** ~300 satır

#### 8.3 Language Detection
```python
# src/data/language_detector.py
# Use langdetect or fasttext
```

**Tahmini Süre:** 2 gün  
**Satır Sayısı:** ~100 satır

---

## 🔵 DÜŞÜK ÖNCELİK (P3) - 3+ Ay

### 9. Advanced AI Features

#### 9.1 Multimodal (VLM)
```python
# Image encoder integration
# VLM training pipeline
# Image-text dataset support
```

**Tahmini Süre:** 4-6 hafta  
**Satır Sayısı:** ~3,000 satır

#### 9.2 Distributed Training
```python
# DDP implementation
# FSDP support
# Multi-GPU configs
```

**Tahmini Süre:** 3-4 hafta  
**Satır Sayısı:** ~2,000 satır

#### 9.3 Model Quantization
```python
# INT8, INT4 quantization
# GPTQ, AWQ support
# Quantization Lab UI
```

**Tahmini Süre:** 2-3 hafta  
**Satır Sayısı:** ~1,500 satır

---

## 📊 Öncelik Matrisi

### Impact vs Effort

```
High Impact, Low Effort (DO FIRST):
├── BLEU/ROUGE Metrics       3-4 gün, ~300 satır
├── Dedup Integration        1 gün, ~50 satır
└── Quality Scoring          3-4 gün, ~200 satır

High Impact, High Effort (PRIORITIZE):
├── RAG Pipeline            2-3 hafta, ~3,400 satır ⭐
├── Frontend Tests          1.5 hafta, ~1,000 satır ⭐
└── Math Lab + Tensor Lab   3.5 hafta, ~2,000 satır

Medium Impact (PLAN):
├── Remaining Labs          10-12 hafta, ~8,000 satır
├── Production Features     3-4 hafta, ~1,100 satır
└── Model Comparison UI     3-4 gün, ~400 satır

Low Priority (FUTURE):
├── Multimodal             4-6 hafta
├── Distributed Training   3-4 hafta
└── Quantization           2-3 hafta
```

---

## 🎯 Önerilen Roadmap

### Sprint 1 (2 Hafta) - Quick Wins
```
Week 1:
├── BLEU/ROUGE Implementation   (3-4 gün)
├── Dedup Integration           (1 gün)
├── Quality Scoring             (3-4 gün)
└── Bug fixes                   (1-2 gün)

Week 2:
├── Frontend Test Setup         (1 gün)
├── Embedding Lab Tests         (2 gün)
├── Attention Lab Tests         (1.5 gün)
└── Documentation updates       (1.5 gün)
```

**Deliverables:**
- ✅ Complete evaluation metrics
- ✅ Better data quality
- ✅ 30%+ frontend coverage
- **Score Impact:** 7.4 → 7.8

---

### Sprint 2-3 (4 Hafta) - RAG Pipeline
```
Week 3-4:
├── RAG Backend Engine          (8 gün)
├── RAG API Router              (2 gün)
└── Tests & Documentation       (2 gün)

Week 5-6:
├── RAG Lab Frontend            (6 gün)
├── Integration Tests           (2 gün)
├── Performance Optimization    (2 gün)
└── User Documentation          (2 gün)
```

**Deliverables:**
- ✅ Complete RAG pipeline
- ✅ FAISS integration
- ✅ RAG Lab UI
- **Score Impact:** 7.8 → 8.3

---

### Sprint 4-5 (4 Hafta) - Educational Labs
```
Week 7-8:
├── Math Lab                    (10 gün)
└── Documentation               (2 gün)

Week 9-10:
├── Tensor Lab                  (8 gün)
└── Neural Network Lab          (4 gün başlangıç)
```

**Deliverables:**
- ✅ Math Lab complete
- ✅ Tensor Lab complete
- ✅ NN Lab %50
- **Score Impact:** 8.3 → 8.7

---

### Sprint 6+ (2+ Ay) - Polishing
```
Month 3:
├── Remaining Labs (NN, Transformer, Systems)
├── Production features (optional)
├── Advanced features
└── Documentation & Tutorials

Month 4+:
├── Multimodal (optional)
├── Distributed Training (optional)
└── Performance optimization
```

**Deliverables:**
- ✅ Complete learning platform
- ✅ Production-ready (optional)
- **Score Impact:** 8.7 → 9.0+

---

## 📈 Skor Projeksiyonu

```
Mevcut:                          7.4/10 (74%)

Sprint 1 tamamlandıktan sonra:   7.8/10 (78%)
├── + BLEU/ROUGE
├── + Quality scoring
├── + Frontend tests (30%)
└── + Dedup integration

Sprint 2-3 tamamlandıktan sonra: 8.3/10 (83%)
├── + RAG Pipeline (major!)
└── + Frontend tests (50%)

Sprint 4-5 tamamlandıktan sonra: 8.7/10 (87%)
├── + Math Lab
├── + Tensor Lab
└── + NN Lab

6 ay sonra (tüm plan):           9.0/10 (90%)
├── + All 12 Labs complete
├── + Production features
└── + Advanced features
```

---

## 🎓 Educational Value Projeksiyonu

```
Mevcut:     7.0/10 (3 lab excellent, 9 missing)

+2 lab:     7.5/10 (Math + Tensor)
+2 lab:     8.0/10 (NN + Transformer)
+3 lab:     8.5/10 (Systems + Atlas + Distributed)
+2 meta:    9.0/10 (Guided Journey + Knowledge Map)

Final: Complete learning platform 9.0/10
```

---

## 💰 Effort Estimation Summary

### Kısa Vade (1-2 Ay)
```
Total Effort: ~6-8 hafta
Total Code:   ~5,000 satır
Impact:       +0.9 puan (7.4 → 8.3)

Breakdown:
├── RAG Pipeline:          2-3 hafta, ~3,400 satır
├── Frontend Tests:        1.5 hafta, ~1,000 satır
├── BLEU/ROUGE:           3-4 gün, ~300 satır
├── Quality + Dedup:      4-5 gün, ~250 satır
└── Bug fixes:            1 hafta, ~50 satır
```

### Orta Vade (3-4 Ay)
```
Total Effort: ~12-14 hafta
Total Code:   ~10,000 satır
Impact:       +0.4 puan (8.3 → 8.7)

Breakdown:
├── Math Lab:             2 hafta, ~1,200 satır
├── Tensor Lab:           1.5 hafta, ~800 satır
├── NN Lab:               2 hafta, ~1,000 satır
├── Transformer Lab:      1.5 hafta, ~900 satır
├── Systems Lab:          1 hafta, ~600 satır
└── Production features:  3-4 hafta, ~1,100 satır
```

### Uzun Vade (6+ Ay)
```
Total Effort: ~20+ hafta
Total Code:   ~15,000+ satır
Impact:       +0.3 puan (8.7 → 9.0+)

Breakdown:
├── Remaining Labs:       4-5 hafta, ~2,900 satır
├── Multimodal:           4-6 hafta, ~3,000 satır
├── Distributed:          3-4 hafta, ~2,000 satır
└── Advanced features:    8+ hafta, ~7,000+ satır
```

---

## 🏁 Sonuç

### Mevcut Durum
```
✅ MVP: %100 Complete
✅ Code Quality: Excellent (9/10)
✅ 3 Labs: Exceptional (9.3/10)
✅ Active Development: 107 training jobs!
⚠️ Educational Platform: 25% (3/12 labs)
⚠️ RAG Pipeline: Missing (critical!)
⚠️ Frontend Tests: 0%
```

### En Kritik 3 İş
```
1. 🔴 RAG Pipeline          → 2-3 hafta
2. 🔴 Frontend Tests        → 1.5 hafta
3. 🟡 BLEU/ROUGE Metrics    → 3-4 gün
```

### Hedef
```
6 ay içinde → 9.0/10 (90%)
- Complete learning platform
- RAG pipeline
- 12/12 labs
- Production-ready (optional)
```

**Next Action:** Sprint 1 başlat (BLEU/ROUGE + Frontend Tests + Quick Wins)

---

**Rapor Hazırlayan:** AI Systems Analyst  
**Tarih:** 22 Eylül 2026, 20:56  
**Uygulama Durumu:** ✅ Running (Backend: 8000, Frontend: 3000)  
**Değişiklik:** YOK (Sadece değerlendirme)
