---
inclusion: auto
name: veri-yonetimi
description: Veri altyapısı, dataset yönetimi ve data lineage kuralları
---

# Veri Yönetimi Standartları

## Canonical Dataset Yapısı

Önerilen standart yapı:

```
dataset/
│
├── README.md
├── dataset.yaml
│
├── raw/                      # Orijinal dosyalar
│   ├── documents/
│   ├── images/
│   ├── tables/
│   ├── audio/
│   └── video/
│
├── normalized/               # Normalize edilmiş veriler
│   ├── documents.parquet
│   ├── tables.parquet
│   ├── images.parquet
│   └── files.parquet
│
├── extracted/               # Çıkarılmış içerikler
│   ├── text/
│   ├── tables/
│   ├── images/
│   └── metadata/
│
├── training/                # Eğitim datasetleri
│   ├── pretraining/
│   ├── sft/
│   ├── preference/
│   └── multimodal/
│
├── rag/                     # RAG verileri
│   ├── documents.parquet
│   └── chunks.parquet
│
├── evaluation/              # Değerlendirme setleri
│
├── splits/                  # Train/val/test bölünmeleri
│
└── manifest/                # Metadata ve tracking
    ├── files.parquet
    ├── relationships.parquet
    └── checksums.sha256
```

## Manifest Sistemi

Her dosyanın benzersiz kimliği olmalı:

```
DOC-00000001
IMG-00000001
TAB-00000001
AUD-00000001
VID-00000001
```

### files.parquet Şeması

```python
id: str                    # Unique identifier
relative_path: str         # Dataset içinde path
media_type: str           # document, image, table, audio, video
mime_type: str            # application/pdf, image/jpeg, etc.
sha256: str               # Dosya hash
size_bytes: int
language: str             # Dil kodu (tr, en, etc.)
source: str               # Veri kaynağı
license: str              # Lisans bilgisi
copyright_status: str     # Telif durumu
created_at: datetime
modified_at: datetime
dataset_version: str      # Hangi dataset versiyonuna ait
security_level: str       # PUBLIC, INTERNAL, RESTRICTED, PERSONAL
pii: bool                 # Kişisel veri var mı?
quality_score: float      # 0.0-1.0 arası kalite skoru
parser_name: str          # Hangi parser kullanıldı
parser_version: str       # Parser versiyonu
schema_version: str       # Şema versiyonu
```

## Data Lineage

Sistem bir çıktının hangi kaynaktan üretildiğini takip edebilmeli.

**Örnek: RAG Pipeline**
```
report.pdf
   ↓
FILE-001
   ↓
DOC-001
   ↓
PAGE-004
   ↓
CHUNK-017
   ↓
EMBEDDING-882
   ↓
RAG-INDEX-v3
```

**Örnek: Model Training**
```
DOC-001
   ↓
cleaning-v2
   ↓
dataset-v1.1.0
   ↓
tokenizer-v3
   ↓
shard-007
   ↓
RUN-0041
   ↓
MODEL-0012
```

## Immutable Dataset Versioning

Yayınlanmış dataset sürümü **sessizce değiştirilmemeli**.

```
dataset-v1.0.0  ← değiştirme
dataset-v1.1.0  ← yeni versiyon oluştur
```

SemVer yaklaşımı:
- `1.0.0` → ilk sürüm
- `1.1.0` → yeni veri eklendi
- `1.1.1` → metadata düzeltmesi
- `2.0.0` → şema değişikliği

## Veri Kalitesi Pipeline

```
Raw Data
   ↓
Encoding Check
   ↓
Normalization
   ↓
Language Detection
   ↓
Duplicate Detection
   ↓
Near-Duplicate Detection
   ↓
Bozuk İçerik Kontrolü
   ↓
PII Kontrolü
   ↓
Lisans Kontrolü
   ↓
Quality Scoring
   ↓
Clean Dataset
```

Kontrol edilmesi gerekenler:
- Boş metin
- Çok kısa belge
- Aşırı tekrar
- Bozuk encoding
- OCR hataları
- Tekrarlanan belgeler
- Spam metinler
- Gereksiz header/footer
- Aynı içeriğin farklı dosyalardaki kopyaları
- Kişisel veri
- Lisans durumu

## Dataset Split Sistemi

Split işlemi **yalnız random yapılmamalı**.

Aynı belgenin farklı sayfalarının hem train hem test'e düşmesi veri sızıntısına neden olabilir.

Dikkate alınması gerekenler:
- `document_id`
- `source_id`
- `duplicate_group`

## Dataset Compiler

Canonical dataset'ten farklı kullanım amaçlarına uygun datasetler üret:

```
Canonical Dataset
      │
      ├── Pretraining Export      # {"text": "..."}
      ├── SFT Export              # {"messages": [...]}
      ├── RAG Export              # {"chunk_id": "...", "text": "..."}
      ├── VLM Export              # {"images": [...], "messages": [...]}
      ├── Classification Export
      └── Evaluation Export
```

## Veri Güvenliği

Her dataset için gerekli alanlar:

```python
security_level: str       # PUBLIC, INTERNAL, RESTRICTED, PERSONAL
pii: bool                # Kişisel veri var mı?
license: str             # Lisans
copyright: str           # Telif hakları
permission: str          # İzin durumu
usage_scope: str         # Kullanım kapsamı
training_allowed: bool   # Eğitimde kullanılabilir mi?
commercial_use: bool     # Ticari kullanım
attribution_required: bool  # Atıf gerekli mi?
```

**UYARI:** PII içeren veya lisansı uygun olmayan veri kullanıcı onayı olmadan training export'a dahil edilmemeli.

## PII Kontrolü

Sistem aşağıdaki veri türlerini işaretleyebilmeli:
- Ad soyad
- Telefon
- E-posta
- T.C. kimlik numarası
- Adres
- Öğrenci numarası
- Kuruma özel kayıt
- Hassas içerik

## Dosya Formatları

### Ana veri formatı
**Parquet** - Metadata, doküman kayıtları, tablo verileri, RAG chunks, evaluation sonuçları

### Eğitim formatı
**JSONL** - SFT, Chat dataset, Prompt/completion, QA

### Büyük medya
**WebDataset TAR** - Milyonlarca görsel, ses, video frame koleksiyonları

### Kaynak dosyalar
Orijinal formatlarında korunmalı (PDF, DOCX, XLSX, JPG, MP3, etc.)

## Deletion Propagation

Bir kaynak dosya kaldırıldığında yalnızca raw dosyanın silinmesi yeterli değil.

Sistem ilgili türevleri izleyebilmeli:
```
Raw File
 ↓
Canonical Record
 ↓
Chunks
 ↓
Embeddings
 ↓
RAG Index
 ↓
Training Exports
```

Silme veya kullanım dışı bırakma işlemi bu bağımlılık ağını dikkate almalı.

**NOT:** RAG indeksinden silmek ile eğitilmiş modelden bilgiyi geri almak aynı işlem değildir.

## Schema Evolution

Dataset şeması zamanla değişebilir. Her kayıt için şunlar tutulmalı:

```python
schema_version: str
parser_name: str
parser_version: str
normalizer_version: str
```

Eski datasetlerin yeni şemaya nasıl migrate edileceği tanımlanmalı.

## Depolama Sorumlulukları

```
SQLite / PostgreSQL
    ↓
Metadata ve ilişkiler

Parquet
    ↓
Dataset içeriği

Filesystem / Object Storage
    ↓
Orijinal ve türetilmiş binary dosyalar
```

Dataset'in tamamı relational database içine gömülmemeli.
