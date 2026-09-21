# AI Research Lab - End-to-End Demo

Bu dizin, AI Research Lab platformunun tüm bileşenlerini entegre eden uçtan uca (end-to-end) demo uygulamalarını içerir.

## 📋 İçindekiler

- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Demo Uygulamaları](#demo-uygulamaları)
- [Pipeline Mimarisi](#pipeline-mimarisi)
- [Kullanım Örnekleri](#kullanım-örnekleri)
- [Çıktılar ve Raporlar](#çıktılar-ve-raporlar)
- [Sorun Giderme](#sorun-giderme)

---

## 🚀 Hızlı Başlangıç

### Gereksinimler

```bash
# Python 3.11+
python --version

# Gerekli kütüphaneler (proje root'ta)
pip install -r requirements.txt
```

### İlk Demo

```bash
# Proje root dizininde
cd "AI Systems Lab"

# Basit E2E demo çalıştır (3 epoch, ~3 saniye)
python3 examples/end_to_end_demo_simple.py

# Demo çıktıları
open demo_output/dashboard.html  # Dashboard'u tarayıcıda aç
```

---

## 📦 Demo Uygulamaları

### 1. End-to-End Pipeline Demo (`end_to_end_demo_simple.py`)

**Ne yapar:** Korpus hazırlıktan dashboard'a kadar tam pipeline'ı çalıştırır.

**Pipeline adımları:**
1. **Tokenizer Training** - SentencePiece BPE tokenizer eğitimi
2. **Dataset Preparation** - TextDataset ve DataLoader oluşturma
3. **Model Initialization** - GPT model başlatma (836K params)
4. **Training** - 3 epoch model eğitimi + checkpoint kaydetme
5. **Evaluation** - Validation metrics (perplexity, accuracy)
6. **Benchmarks** - Turkish generation benchmark (BLEU/ROUGE/ChrF)
7. **Model Registry** - Trained model'i registry'e kaydetme
8. **Inference** - Text generation test
9. **Dashboard** - Interactive HTML dashboard oluşturma

**Kullanım:**
```bash
python3 examples/end_to_end_demo_simple.py
```

**Çıktı:**
```
demo_output/
├── dashboard.html          # Interactive metrics dashboard
├── comparison.json         # Model comparison results
├── tokenizer/
│   ├── tokenizer.model     # SentencePiece model
│   ├── tokenizer.vocab     # Vocabulary
│   └── tokenizer.config.json
├── checkpoints/
│   ├── gpt-demo_v0.1.3_epoch3_step60.pt
│   └── metadata/
└── registry/
    ├── models/
    └── metadata/
```

**Beklenen sonuçlar:**
- Training loss: 1.037 → 0.081 (3 epoch)
- Validation perplexity: 1.244 → 1.065
- Accuracy: ~97%
- Execution time: ~3 seconds (CPU)

---

### 2. Integration Test Suite (`test_integration.py`)

**Ne yapar:** Tüm bileşenlerin entegrasyonunu test eder.

**Test kategorileri:**
1. **Tokenizer API** - Encode/decode/batch operations
2. **Dataset API** - TextDataset & DataLoader
3. **Model API** - Forward pass & parameter count
4. **Checkpoint API** - Save/load functionality
5. **Evaluation API** - Metrics computation
6. **Registry API** - Model registration
7. **Data Flow** - End-to-end pipeline flow
8. **Format Compatibility** - Cross-component format validation
9. **Error Handling** - Graceful error scenarios

**Kullanım:**
```bash
python3 examples/test_integration.py
```

**Çıktı:**
```
================================================================================
INTEGRATION TESTING - SUMMARY
================================================================================
Total Tests: 9
Passed: 9
Failed: 0
✅ ALL INTEGRATION TESTS PASSED
================================================================================
```

**Test raporu:**
```
integration_test_output/integration_test_report.json
```

---

## 🏗️ Pipeline Mimarisi

### Veri Akışı

```
┌─────────────────┐
│  Sample Corpus  │
│  (200 Turkish   │
│   sentences)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Tokenizer Training         │
│  - SentencePiece BPE        │
│  - Vocab size: 300          │
│  - Character coverage: 0.9995│
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Dataset Preparation        │
│  - TextDataset              │
│  - Train/Val split: 80/20   │
│  - Max length: 32 tokens    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Model Initialization       │
│  - GPT architecture         │
│  - 4 layers, 128 dim        │
│  - 4 heads, 836K params     │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Training Loop              │
│  - 3 epochs                 │
│  - Batch size: 8            │
│  - Gradient clipping        │
│  - CheckpointManager        │
│  - QualityTracker           │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Evaluation & Benchmarks    │
│  - Core metrics (ppl, acc)  │
│  - Generation quality       │
│  - BLEU/ROUGE/ChrF          │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Model Comparison           │
│  - Statistical tests        │
│  - Winner detection         │
│  - Effect size (Cohen's d)  │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Model Registry             │
│  - Version: 1.0.0           │
│  - Metadata tracking        │
│  - Checkpoint archival      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Inference Testing          │
│  - Text generation          │
│  - Temperature: 0.8         │
│  - Max tokens: 10           │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Dashboard Generation       │
│  - HTML report              │
│  - Interactive charts       │
│  - Comparison tables        │
└─────────────────────────────┘
```

### Komponent Bağımlılıkları

```python
# Sprint 9 Components
from src.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
from src.data.dataset import TextDataset, DataLoader
from src.model.gpt import GPTModel, GPTConfig
from src.training.checkpoint_manager import CheckpointManager
from src.registry.model_registry import ModelRegistry

# Sprint 10 Components
from src.evaluation.metrics import (
    compute_perplexity,
    compute_accuracy,
    evaluate_batch,
    aggregate_results
)
from src.evaluation.benchmarks import (
    create_turkish_generation_benchmark,
    run_generation_benchmark
)
from src.evaluation.comparison import ModelComparison
from src.evaluation.quality_tracker import QualityTracker
from src.evaluation.dashboard import DashboardGenerator
```

---

## 💡 Kullanım Örnekleri

### Örnek 1: Basit Tokenizer Eğitimi

```python
from src.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer

# Corpus hazırla
texts = ["Türkiye bir Avrupa ülkesidir."] * 100
corpus_file = "my_corpus.txt"
with open(corpus_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(texts))

# Tokenizer eğit
tokenizer = SentencePieceTokenizer()
tokenizer.train(
    corpus_path=corpus_file,
    vocab_size=500,
    model_prefix="my_tokenizer",
    model_type='bpe'
)

# Kullan
text = "Ankara başkenttir"
tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
decoded = tokenizer.decode(tokens)

print(f"Tokens: {tokens}")
print(f"Decoded: {decoded}")
```

### Örnek 2: Model Eğitimi ve Değerlendirme

```python
from src.model.gpt import GPTModel, GPTConfig
from src.training.checkpoint_manager import CheckpointManager
from src.evaluation.metrics import evaluate_batch

# Model oluştur
config = GPTConfig(
    vocab_size=500,
    max_seq_len=64,
    d_model=256,
    n_layers=6,
    n_heads=8
)
model = GPTModel(config)

# Checkpoint manager
checkpoint_mgr = CheckpointManager(
    checkpoint_dir="checkpoints",
    model_name="my-gpt"
)

# Training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    for batch in train_loader:
        # Forward pass
        output = model(batch['input_ids'])
        logits = output[0] if isinstance(output, tuple) else output
        
        # Loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch['labels'].view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            output = model(batch['input_ids'])
            logits = output[0] if isinstance(output, tuple) else output
            results = evaluate_batch(logits, batch['labels'])
    
    # Save checkpoint
    checkpoint_mgr.save_checkpoint(
        model=model,
        epoch=epoch,
        step=epoch * len(train_loader),
        metrics={'loss': loss.item()},
        optimizer=optimizer
    )
    
    model.train()
```

### Örnek 3: Model Karşılaştırma

```python
from src.evaluation.comparison import ModelComparison
from src.evaluation.benchmarks import run_generation_benchmark

# Modelleri değerlendir
comparison = ModelComparison()

for model_name, model in models.items():
    results = run_generation_benchmark(
        model, tokenizer, benchmark_dataset
    )
    comparison.add_model(model_name, results)

# Karşılaştır
report = comparison.compare()

# Sonuçlar
print(f"Overall Winner: {report.overall_winner}")
print(f"Metric Comparisons: {report.metric_comparisons}")

# Rapor kaydet
report.save("comparison_report.json")
report.save_markdown("comparison_report.md")
```

### Örnek 4: Dashboard Oluşturma

```python
from src.evaluation.dashboard import DashboardGenerator
from src.evaluation.quality_tracker import QualityTracker

# Dashboard oluştur
dashboard = DashboardGenerator(
    title="Model Training Dashboard",
    description="GPT model training results"
)

# Evaluation results ekle
dashboard.add_evaluation_results("Model-v1", eval_results_1)
dashboard.add_evaluation_results("Model-v2", eval_results_2)

# Comparison report ekle
dashboard.add_comparison_report(comparison_report)

# Quality tracker ekle
quality_tracker = QualityTracker(metrics=['loss', 'perplexity'])
# ... add evaluations during training ...
dashboard.add_quality_tracker(quality_tracker)

# Generate
dashboard.generate("training_dashboard.html")
```

---

## 📊 Çıktılar ve Raporlar

### Dashboard (`dashboard.html`)

Interactive HTML dashboard with:
- **Metrics Overview** - Perplexity, accuracy, loss
- **Training Progress** - Epoch-by-epoch metrics
- **Quality Alerts** - Regression detection
- **Model Comparison** - Side-by-side evaluation
- **Benchmark Results** - BLEU/ROUGE/ChrF scores

**Features:**
- Bootstrap 5 styling
- Chart.js visualizations
- Responsive design
- Print-friendly layout

### Comparison Report (`comparison.json`)

JSON format with:
```json
{
  "model_names": ["Model-A", "Model-B"],
  "overall_winner": "Model-A",
  "metric_comparisons": {
    "perplexity": {
      "winner": "Model-A",
      "mean_scores": {"Model-A": 1.05, "Model-B": 1.12},
      "p_value": 0.023,
      "statistically_significant": true
    }
  }
}
```

### Checkpoint Files

PyTorch checkpoint format:
```python
{
    'epoch': 3,
    'step': 60,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'metrics': {'loss': 0.08, 'perplexity': 1.06},
    'config': {...},
    'version': '0.1.3',
    'timestamp': '2026-09-20T23:30:58'
}
```

### Registry Metadata

Model registry JSON:
```json
{
  "model_name": "gpt-demo",
  "version": "1.0.0",
  "architecture": "GPT",
  "parameters": 836140,
  "metrics": {
    "perplexity": 1.065,
    "accuracy": 0.974
  },
  "tokenizer_info": {
    "vocab_size": 300,
    "model_type": "bpe"
  },
  "created_at": "2026-09-20T23:30:58",
  "file_size_mb": 9.65
}
```

---

## 🔧 Konfigürasyon

### Model Config

```python
from src.model.gpt import GPTConfig

config = GPTConfig(
    vocab_size=500,        # Vocabulary size
    max_seq_len=512,       # Maximum sequence length
    d_model=256,           # Model dimension
    n_layers=6,            # Number of transformer layers
    n_heads=8,             # Number of attention heads
    d_ff=1024,             # Feed-forward dimension
    dropout=0.1,           # Dropout rate
    activation='gelu',     # Activation function
    use_gated_ffn=False,   # Use gated feed-forward
    tie_embeddings=True    # Tie input/output embeddings
)
```

### Checkpoint Config

```python
from src.training.checkpoint_manager import CheckpointConfig

config = CheckpointConfig(
    keep_top_k=3,              # Keep top-k best checkpoints
    keep_last_k=5,             # Keep last k checkpoints
    metric_for_best='loss',    # Metric for best model selection
    mode='min',                # 'min' or 'max'
    save_optimizer=True,       # Save optimizer state
    save_scheduler=True,       # Save LR scheduler state
    auto_cleanup=True          # Auto-cleanup old checkpoints
)
```

### Quality Tracker Config

```python
from src.evaluation.quality_tracker import QualityTracker

tracker = QualityTracker(
    metrics=['loss', 'perplexity', 'accuracy'],
    regression_threshold=0.05,  # 5% degradation = regression
    moving_avg_window=3         # Moving average window
)
```

---

## 🐛 Sorun Giderme

### Hata: "Vocabulary size too high"

**Sorun:** SentencePiece training sırasında vocab_size çok büyük hatası.

**Çözüm:** Corpus'ta yeterli unique karakter/kelime olmayabilir. Vocab size'ı düşürün veya corpus'u genişletin:

```python
# Küçük corpus için
tokenizer.train(
    corpus_path=corpus_file,
    vocab_size=100,  # Düşürüldü
    model_prefix="tokenizer"
)

# Veya corpus'u genişlet
texts = texts * 10  # Daha fazla tekrar
```

### Hata: "CheckpointConfig got unexpected keyword"

**Sorun:** CheckpointManager API kullanımı yanlış.

**Çözüm:** `checkpoint_dir` config içinde değil, direkt parametre:

```python
# YANLIŞ
config = CheckpointConfig(checkpoint_dir="checkpoints")
manager = CheckpointManager(config=config)

# DOĞRU
config = CheckpointConfig(keep_top_k=3)
manager = CheckpointManager(
    checkpoint_dir="checkpoints",
    model_name="my-model",
    config=config
)
```

### Hata: "Tensor size mismatch"

**Sorun:** Model output ve label boyutları uyumsuz.

**Çözüm:** Cross-entropy için boyutları kontrol edin:

```python
# Model output
output = model(input_ids)  # [batch, seq_len, vocab_size]
logits = output[0] if isinstance(output, tuple) else output

# Loss computation
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),  # [batch*seq, vocab]
    labels.view(-1),                    # [batch*seq]
    ignore_index=-100
)
```

### Hata: "'SentencePieceTokenizer' has no attribute 'bos_token_id'"

**Sorun:** API kullanımı yanlış.

**Çözüm:** `bos_id` ve `eos_id` property'ler:

```python
# YANLIŞ
bos = tokenizer.bos_token_id
eos = tokenizer.eos_token_id

# DOĞRU
bos = tokenizer.bos_id  # property
eos = tokenizer.eos_id  # property
```

### Performance: "Training çok yavaş"

**Optimizasyonlar:**

1. **Batch size artır:**
```python
loader = DataLoader(dataset, batch_size=32)  # 8'den 32'ye
```

2. **GPU kullan:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

3. **Gradient accumulation:**
```python
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

4. **Mixed precision training:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input_ids)
    loss = compute_loss(output, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 📚 API Referansı

### Tokenizer

```python
class SentencePieceTokenizer:
    def train(corpus_path, vocab_size, model_prefix, model_type='bpe')
    def encode(text, add_bos=False, add_eos=False) -> List[int]
    def decode(tokens) -> str
    def encode_batch(texts, add_bos=False) -> List[List[int]]
    
    @property
    def vocab_size() -> int
    @property
    def bos_id() -> int
    @property
    def eos_id() -> int
```

### Dataset

```python
class TextDataset:
    def __init__(texts, tokenizer, max_length)
    def __len__() -> int
    def __getitem__(idx) -> Dict[str, Tensor]

class DataLoader:
    def __init__(dataset, batch_size, shuffle=False)
    def __iter__() -> Iterator[Dict[str, Tensor]]
```

### Model

```python
class GPTModel:
    def __init__(config: GPTConfig)
    def forward(input_ids) -> Tensor  # [batch, seq, vocab]
    def generate(input_ids, max_new_tokens, temperature) -> Tensor
    def get_num_params(non_embedding=False) -> int
```

### Evaluation

```python
def compute_perplexity(logits, targets, ignore_index=-100) -> float
def compute_accuracy(predictions, targets, ignore_index=-100) -> float
def evaluate_batch(logits, targets) -> EvaluationResults
def aggregate_results(results_list) -> EvaluationResults

class EvaluationResults:
    def get_metric(name) -> MetricResult
    def save(path)
    def to_dict() -> Dict
```

---

## 📖 Daha Fazla Bilgi

- **Proje Dökümantasyonu:** `../README.md`
- **Kod Standartları:** `.kiro/steering/04-code-standards.md`
- **Proje İlkeleri:** `.kiro/steering/00-project-principles.md`
- **Sprint 9 (Training):** `docs/sprint9_summary.md`
- **Sprint 10 (Evaluation):** `docs/sprint10_summary.md`

---

## ✅ Checklist

Demo çalıştırmadan önce:

- [ ] Python 3.11+ kurulu
- [ ] Dependencies yüklü (`pip install -r requirements.txt`)
- [ ] Proje root dizininde (`AI Systems Lab/`)
- [ ] Yeterli disk alanı (~50 MB demo çıktıları için)

Demo sonrası kontrol:

- [ ] `demo_output/dashboard.html` oluştu
- [ ] `demo_output/checkpoints/` dizini dolu
- [ ] `demo_output/registry/` model kaydı var
- [ ] Training loss azaldı (>1.0 → <0.1)
- [ ] Dashboard tarayıcıda açılıyor

Integration test sonrası:

- [ ] 9/9 test geçti
- [ ] `integration_test_output/integration_test_report.json` oluştu
- [ ] Test raporu "passed": 9, "failed": 0 gösteriyor

---

## 🤝 Katkıda Bulunma

Demo'larda sorun bulursanız veya yeni demo fikirleri varsa:

1. Issue açın (sorun tanımı + beklenen davranış)
2. Integration test ekleyin (`test_integration.py`)
3. Dökümantasyonu güncelleyin

---

## 📝 Lisans

Bu demo uygulamaları AI Research Lab platformunun bir parçasıdır.

---

**Son güncelleme:** 2026-09-20  
**Versiyon:** 1.0.0  
**Test durumu:** ✅ 9/9 integration tests passing
