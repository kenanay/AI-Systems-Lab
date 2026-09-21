# E2E Demo Architecture

Bu doküman E2E demo'nun mimari tasarımını ve komponent etkileşimlerini açıklar.

## 📐 Genel Mimari

```
┌──────────────────────────────────────────────────────────────────┐
│                     E2E Demo Application Layer                   │
│                  (end_to_end_demo_simple.py)                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Component Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │Tokenizer │  │ Dataset  │  │  Model   │  │ Training │        │
│  │  (S9)    │  │  (S9)    │  │  (S9)    │  │  (S9)    │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │Evaluation│  │Benchmarks│  │Comparison│  │ Registry │        │
│  │  (S10)   │  │  (S10)   │  │  (S10)   │  │  (S9)    │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Storage Layer                               │
│   Checkpoints  │  Registry  │  Dashboards  │  Metadata          │
└──────────────────────────────────────────────────────────────────┘
```

## 🔧 Komponent Detayları

### 1. Tokenizer Component (Sprint 9)

**Sorumluluğu:** Text → Token ID dönüşümü

```python
# Interface
class SentencePieceTokenizer:
    def train(corpus_path, vocab_size, model_prefix) -> None
    def encode(text, add_bos, add_eos) -> List[int]
    def decode(tokens) -> str
    
# Implementation Details
- Algorithm: BPE (Byte Pair Encoding)
- Library: sentencepiece
- Character coverage: 0.9995 (Turkish support)
- Special tokens: <unk>=0, <s>=1, </s>=2, <pad>=3
```

**Veri Akışı:**
```
Sample Corpus (200 sentences)
    ↓
SentencePieceTrainer.train()
    ↓
tokenizer.model + tokenizer.vocab
    ↓
encode(): "Türkiye" → [1, 145, 67, 2]
    ↓
decode(): [1, 145, 67, 2] → "Türkiye"
```

**Dependencies:**
- `sentencepiece` library
- UTF-8 encoded corpus file

### 2. Dataset Component (Sprint 9)

**Sorumluluğu:** Token sequences → PyTorch batches

```python
# Interface
class TextDataset:
    def __init__(texts, tokenizer, max_length)
    def __len__() -> int
    def __getitem__(idx) -> Dict[str, Tensor]
    
class DataLoader:
    def __init__(dataset, batch_size, shuffle)
    def __iter__() -> Iterator[Dict[str, Tensor]]
```

**Veri Transformasyonu:**
```
Raw text: "Türkiye bir ülkedir"
    ↓ tokenizer.encode()
Token IDs: [1, 145, 67, 89, 2]
    ↓ padding/truncation (max_length=32)
Padded: [1, 145, 67, 89, 2, 3, 3, ..., 3]
    ↓ create labels (shifted by 1)
input_ids:  [1, 145, 67, 89,   2, 3, 3, ..., 3]
labels:     [145, 67, 89, 2,   3, 3, 3, ..., -100]
    ↓ batch collation
Batch: {
    'input_ids': Tensor[batch=8, seq=32],
    'labels': Tensor[batch=8, seq=32]
}
```

**Design Decisions:**
- Causal LM masking: labels shifted by 1
- Padding token ignored in loss: `ignore_index=-100`
- Max length: 32 tokens (demo) / configurable (production)

### 3. Model Component (Sprint 9)

**Sorumluluğu:** Token IDs → Logits (next token prediction)

```python
# Interface
class GPTModel:
    def __init__(config: GPTConfig)
    def forward(input_ids: Tensor) -> Tensor
    def generate(input_ids, max_new_tokens, temperature) -> Tensor
```

**Architecture:**
```
Input: [batch, seq_len]
    ↓
Token Embedding: [batch, seq_len, d_model]
    ↓
Positional Embedding: [batch, seq_len, d_model]
    ↓
Transformer Blocks × n_layers:
    ↓ LayerNorm
    ↓ Multi-Head Attention (causal)
    ↓ Residual Connection
    ↓ LayerNorm
    ↓ Feed-Forward Network (GELU)
    ↓ Residual Connection
    ↓
Final LayerNorm
    ↓
Output Projection: [batch, seq_len, vocab_size]
```

**Demo Config:**
```python
GPTConfig(
    vocab_size=300,
    max_seq_len=32,
    d_model=128,        # Embedding dimension
    n_layers=4,         # Transformer blocks
    n_heads=4,          # Attention heads
    d_ff=512,           # FFN inner dimension
    dropout=0.1,
    tie_embeddings=True # Input/output weight sharing
)
# Total parameters: 836,140
```

**Memory Layout:**
```
Component               Parameters      %
─────────────────────────────────────────
Token Embedding         38,400          4.6%
Position Embedding      4,096           0.5%
Transformer Layers      755,200         90.3%
  - Attention           502,784         60.1%
  - Feed-Forward        252,416         30.2%
Output Projection       38,400          4.6% (tied)
─────────────────────────────────────────
Total                   836,140         100%
```

### 4. Training Component (Sprint 9)

**Sorumluluğu:** Model optimization + checkpoint management

```python
# Main training loop
for epoch in range(num_epochs):
    # Training
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch['input_ids'])
        loss = compute_loss(output, batch['labels'])
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            results = evaluate_batch(model(batch['input_ids']), batch['labels'])
    
    # Quality tracking
    quality_tracker.add_evaluation(epoch, results)
    
    # Checkpoint
    checkpoint_manager.save_checkpoint(
        model, epoch, step, metrics, optimizer
    )
```

**Checkpoint Format:**
```python
{
    'epoch': int,
    'step': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': Dict,
    'metrics': Dict[str, float],
    'config': Dict[str, Any],
    'version': str,
    'timestamp': str,
    'manager_version': str
}
```

**Checkpoint Strategy:**
- Keep top-k: 2 (best by loss)
- Keep last-k: 5 (most recent)
- Auto cleanup: Yes
- Metadata tracking: JSON sidecar files

### 5. Evaluation Component (Sprint 10)

**Sorumluluğu:** Model performance measurement

```python
# Core Metrics
def compute_perplexity(logits, targets) -> float:
    """exp(cross_entropy_loss)"""
    
def compute_accuracy(predictions, targets) -> float:
    """token-level accuracy"""
    
def compute_top_k_accuracy(logits, targets, k) -> float:
    """target in top-k predictions"""

# Batch Evaluation
def evaluate_batch(logits, targets) -> EvaluationResults:
    """All metrics in one call"""
```

**Metrics Details:**

| Metric | Formula | Range | Better |
|--------|---------|-------|--------|
| Perplexity | exp(CE_loss) | [1, ∞) | Lower |
| Accuracy | correct / total | [0, 1] | Higher |
| Top-5 Accuracy | target_in_top5 / total | [0, 1] | Higher |
| Entropy | -Σ p(x)log p(x) | [0, log(V)] | Lower |
| Confidence | mean(max_prob) | [0, 1] | Higher |

**EvaluationResults Structure:**
```python
class EvaluationResults:
    metrics: Dict[str, MetricResult]
    num_samples: int
    num_tokens: int
    
class MetricResult:
    name: str
    value: float
    unit: str
    higher_is_better: bool
```

### 6. Benchmarks Component (Sprint 10)

**Sorumluluğu:** Generation quality measurement

```python
# Generation Metrics
def compute_bleu(hypothesis, references) -> Dict[str, float]:
    """BLEU-1/2/3/4 scores"""
    
def compute_rouge(hypothesis, references) -> Dict[str, float]:
    """ROUGE-1/2/L F1 scores"""
    
def compute_chrf(hypothesis, references) -> float:
    """Character-level F-score"""
```

**Turkish Generation Benchmark:**
```python
BenchmarkDataset(
    examples=[
        BenchmarkExample(
            prompt="Türkiye",
            references=[
                "Türkiye Avrupa ve Asya kıtalarında yer alır.",
                "Türkiye Cumhuriyeti Anadolu yarımadasında bulunur."
            ],
            domain="geography"
        ),
        # ... 4 more examples
    ]
)
```

**Metric Formulas:**

**BLEU:**
```
BLEU-N = BP × exp(Σ log(precision_n))
BP = brevity_penalty = min(1, exp(1 - ref_len/hyp_len))
precision_n = matched_n-grams / total_n-grams
```

**ROUGE:**
```
ROUGE-L = LCS_F1
LCS_F1 = 2 × (LCS_precision × LCS_recall) / (LCS_precision + LCS_recall)
```

**ChrF:**
```
ChrF = (1 + β²) × (chrP × chrR) / (β² × chrP + chrR)
chrP = character_n-gram_precision
chrR = character_n-gram_recall
```

### 7. Comparison Component (Sprint 10)

**Sorumluluğu:** Statistical model comparison

```python
class ModelComparison:
    def add_model(name: str, results: EvaluationResults)
    def compare() -> ComparisonReport
```

**Statistical Tests:**

1. **Paired t-test:**
```python
t_statistic, p_value = scipy.stats.ttest_rel(scores_a, scores_b)
H0: μ_A = μ_B (no difference)
H1: μ_A ≠ μ_B (significant difference)
α = 0.05
```

2. **Bootstrap Confidence Interval:**
```python
# Resample with replacement
bootstrap_samples = [resample(scores) for _ in range(1000)]
ci_lower, ci_upper = percentile(bootstrap_samples, [2.5, 97.5])
```

3. **Cohen's d (Effect Size):**
```python
d = (μ_A - μ_B) / pooled_std
Interpretation:
  |d| < 0.2: Small effect
  0.2 ≤ |d| < 0.5: Medium effect
  |d| ≥ 0.5: Large effect
```

**ComparisonReport Structure:**
```python
{
    "model_names": ["Model-A", "Model-B"],
    "overall_winner": "Model-A",
    "metric_comparisons": {
        "perplexity": {
            "winner": "Model-A",
            "mean_scores": {"Model-A": 1.05, "Model-B": 1.12},
            "p_value": 0.023,
            "statistically_significant": true,
            "effect_size": 0.42,
            "confidence_interval": [0.01, 0.13]
        }
    }
}
```

### 8. Quality Tracker Component (Sprint 10)

**Sorumluluğu:** Regression detection

```python
class QualityTracker:
    def add_evaluation(epoch: int, results: EvaluationResults)
    def detect_regressions() -> List[QualityAlert]
```

**Alert Levels:**
```python
class AlertLevel(Enum):
    INFO = "info"           # Improvement detected
    WARNING = "warning"     # <5% degradation
    ERROR = "error"         # 5-10% degradation
    CRITICAL = "critical"   # >10% degradation
```

**Regression Detection:**
```python
# Compare current vs best
current_value = metric_history[-1].value
best_value = min(metric_history, key=lambda x: x.value).value

degradation_pct = (current_value - best_value) / best_value * 100

if degradation_pct < 5:
    alert_level = WARNING
elif degradation_pct < 10:
    alert_level = ERROR
else:
    alert_level = CRITICAL
```

**Trend Analysis:**
```python
# Moving average (window=3)
recent_values = [h.value for h in metric_history[-3:]]
ma = sum(recent_values) / len(recent_values)

# Slope (linear regression, last 5 points)
slope = linregress(epochs, values).slope
# slope > 0: degrading, slope < 0: improving
```

### 9. Registry Component (Sprint 9)

**Sorumluluğu:** Model versioning & metadata

```python
class ModelRegistry:
    def register_model(
        model_name, version, checkpoint_path,
        tokenizer_path, metrics, training_config
    ) -> ModelMetadata
    
    def list_models(
        model_name=None, tag=None, environment=None
    ) -> List[ModelMetadata]
    
    def load_model(model_name, version) -> Tuple[Model, Tokenizer]
```

**Directory Structure:**
```
registry/
├── index.json                    # Central index
├── models/
│   └── {model_name}/
│       └── {version}/
│           ├── model.pt          # PyTorch checkpoint
│           ├── tokenizer.model   # SentencePiece model
│           ├── tokenizer.vocab
│           └── tokenizer.config.json
└── metadata/
    └── {model_name}_{version}.json
```

**Metadata Schema:**
```python
{
    "model_name": str,
    "version": str,              # Semantic versioning
    "description": str,
    "architecture": str,         # "GPT", "BERT", etc.
    "parameters": int,
    "metrics": Dict[str, float],
    "training_config": Dict[str, Any],
    "tokenizer_info": {
        "vocab_size": int,
        "model_type": str
    },
    "tags": List[str],
    "environment": str,          # "development", "staging", "production"
    "created_at": str,           # ISO timestamp
    "created_by": str,
    "model_hash": str,           # MD5 checksum
    "file_size_mb": float
}
```

### 10. Dashboard Component (Sprint 10)

**Sorumluluğu:** HTML visualization generation

```python
class DashboardGenerator:
    def add_evaluation_results(name, results)
    def add_comparison_report(report)
    def add_quality_tracker(tracker)
    def generate(output_path)
```

**Dashboard Structure:**
```html
<!DOCTYPE html>
<html>
<head>
    <link href="bootstrap.min.css" rel="stylesheet">
    <script src="chart.js"></script>
</head>
<body>
    <!-- Header -->
    <div class="dashboard-header">
        <h1>Model Training Dashboard</h1>
    </div>
    
    <!-- Metrics Overview -->
    <div class="metrics-overview">
        <div class="metric-card">Perplexity: 1.065</div>
        <div class="metric-card">Accuracy: 97.4%</div>
    </div>
    
    <!-- Training Progress Chart -->
    <canvas id="trainingChart"></canvas>
    
    <!-- Model Comparison Table -->
    <table class="comparison-table">...</table>
    
    <!-- Quality Alerts -->
    <div class="alerts">...</div>
</body>
</html>
```

## 🔄 Veri Akışı (Detaylı)

### Training Phase

```
1. Sample Corpus
   ↓ (text lines)
2. SentencePieceTokenizer.train()
   ↓ (tokenizer.model)
3. tokenizer.encode_batch(texts)
   ↓ (List[List[int]])
4. TextDataset(tokens, max_length=32)
   ↓ (Dict[str, Tensor])
5. DataLoader(dataset, batch_size=8)
   ↓ (batched Dict[str, Tensor])
6. model(input_ids)
   ↓ (logits: [batch, seq, vocab])
7. cross_entropy(logits, labels)
   ↓ (loss: scalar)
8. loss.backward()
   ↓ (gradients)
9. optimizer.step()
   ↓ (updated weights)
10. CheckpointManager.save_checkpoint()
    ↓ (checkpoint.pt + metadata.json)
```

### Evaluation Phase

```
1. Model in eval mode
   ↓
2. Validation batches
   ↓ (Dict[str, Tensor])
3. model(input_ids)
   ↓ (logits: [batch, seq, vocab])
4. evaluate_batch(logits, labels)
   ↓ (EvaluationResults)
5. aggregate_results([results1, results2, ...])
   ↓ (Aggregated EvaluationResults)
6. QualityTracker.add_evaluation(epoch, results)
   ↓ (Regression alerts)
7. run_generation_benchmark(model, tokenizer, benchmark)
   ↓ (Benchmark results: BLEU/ROUGE/ChrF)
8. ModelComparison.compare()
   ↓ (ComparisonReport with statistics)
9. DashboardGenerator.generate()
   ↓ (HTML dashboard)
```

### Registry Phase

```
1. Latest checkpoint
   ↓ (checkpoint.pt)
2. ModelRegistry.register_model()
   ↓
3. Copy checkpoint → registry/models/{name}/{version}/
   ↓
4. Copy tokenizer → registry/models/{name}/{version}/
   ↓
5. Create metadata → registry/metadata/{name}_{version}.json
   ↓
6. Update index → registry/index.json
   ↓
7. Calculate hash (MD5)
   ↓
8. Return ModelMetadata
```

## 🔌 Integration Points

### Tokenizer ↔ Dataset

**Contract:**
```python
# Tokenizer provides
tokenizer.encode(text) -> List[int]
tokenizer.bos_id -> int
tokenizer.eos_id -> int
tokenizer.pad_id -> int

# Dataset requires
tokens: List[int]
special_token_ids: int (for padding/masking)
```

### Dataset ↔ Model

**Contract:**
```python
# Dataset provides
{
    'input_ids': Tensor[batch, seq_len],  # dtype=torch.long
    'labels': Tensor[batch, seq_len]       # dtype=torch.long
}

# Model requires
input_ids: Tensor[batch, seq_len]  # dtype=torch.long
# Model returns
logits: Tensor[batch, seq_len, vocab_size]  # dtype=torch.float
```

### Model ↔ Evaluation

**Contract:**
```python
# Model provides
logits: Tensor[batch, seq_len, vocab_size]

# Evaluation requires
logits: Tensor[batch, seq_len, vocab_size]
targets: Tensor[batch, seq_len]

# Evaluation returns
EvaluationResults(
    metrics={'perplexity': 1.06, 'accuracy': 0.974, ...}
)
```

### Evaluation ↔ Dashboard

**Contract:**
```python
# Evaluation provides
EvaluationResults with:
- get_metric(name) -> MetricResult
- to_dict() -> Dict

# Dashboard requires
Any object with:
- get_metric(name) or dict-like interface
- Metric names: 'perplexity', 'accuracy', 'loss'

# Dashboard generates
HTML file with Bootstrap + Chart.js
```

## 🛡️ Error Handling Strategy

### 1. Input Validation

```python
# Tokenizer
assert corpus_path.exists(), f"Corpus not found: {corpus_path}"
assert vocab_size > 0, "vocab_size must be positive"

# Dataset
assert len(texts) > 0, "Empty dataset"
assert max_length > 0, "max_length must be positive"

# Model
assert config.d_model % config.n_heads == 0, \
    "d_model must be divisible by n_heads"
```

### 2. Graceful Degradation

```python
# Checkpoint save failures
try:
    checkpoint_manager.save_checkpoint(...)
except Exception as e:
    logger.warning(f"Checkpoint save failed: {e}")
    # Continue training

# Dashboard generation failures
try:
    dashboard.generate(output_path)
except Exception as e:
    logger.error(f"Dashboard generation failed: {e}")
    # Save raw metrics as JSON fallback
```

### 3. Recovery Mechanisms

```python
# Resume from checkpoint
if checkpoint_exists:
    checkpoint = load_checkpoint(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0
```

## 📊 Performance Considerations

### Memory Usage

```python
# Model: ~3.3 MB (836K params × 4 bytes)
# Checkpoint: ~9.6 MB (model + optimizer state)
# Dataset: ~0.1 MB (200 samples × 32 tokens × 4 bytes)
# Total: ~13 MB (demo)
```

### Computation

```python
# Forward pass: O(batch × seq_len × d_model²)
# Training step: ~0.1s (CPU, batch=8, seq=32)
# Epoch (20 batches): ~2s
# Full training (3 epochs): ~6s
```

### Optimization Opportunities

1. **Batch size:** 8 → 32 (4x speedup)
2. **GPU:** CPU → CUDA (10-100x speedup)
3. **Mixed precision:** FP32 → FP16 (2x speedup + memory)
4. **Gradient accumulation:** Memory efficiency

## 🧪 Testing Strategy

### Unit Tests (Individual Components)

```python
# test_tokenizer.py
def test_encode_decode():
    tokenizer = SentencePieceTokenizer()
    text = "test"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text

# test_model.py
def test_forward_shape():
    model = GPTModel(config)
    input_ids = torch.randint(0, vocab_size, (batch, seq))
    logits = model(input_ids)
    assert logits.shape == (batch, seq, vocab_size)
```

### Integration Tests (Cross-Component)

```python
# test_integration.py
def test_tokenizer_to_dataset():
    tokenizer = train_tokenizer(corpus)
    dataset = TextDataset(texts, tokenizer)
    item = dataset[0]
    assert 'input_ids' in item
    assert 'labels' in item

def test_end_to_end_flow():
    # Full pipeline test
    tokenizer → dataset → model → eval → dashboard
    assert all_components_work()
```

### Regression Tests

```python
# test_regression.py
def test_api_compatibility():
    # Ensure API signatures haven't changed
    tokenizer.encode(text, add_bos=True)  # must work
    model(input_ids)  # must return logits
    checkpoint_manager.save_checkpoint(...)  # must accept params
```

## 📖 Design Patterns

### 1. Builder Pattern (Config)

```python
config = GPTConfig(
    vocab_size=300,
    max_seq_len=32,
    ...
)
model = GPTModel(config)
```

### 2. Strategy Pattern (Metrics)

```python
metrics = {
    'perplexity': compute_perplexity,
    'accuracy': compute_accuracy,
    'bleu': compute_bleu
}
for name, compute_fn in metrics.items():
    value = compute_fn(predictions, targets)
```

### 3. Observer Pattern (Quality Tracker)

```python
# Observer (QualityTracker) monitors Subject (Training Loop)
for epoch in range(num_epochs):
    results = train_and_evaluate()
    quality_tracker.add_evaluation(epoch, results)  # Notify
    alerts = quality_tracker.detect_regressions()   # React
```

### 4. Facade Pattern (Dashboard)

```python
# Simplified interface to complex subsystems
dashboard = DashboardGenerator()
dashboard.add_evaluation_results(...)  # Hides HTML generation
dashboard.add_comparison_report(...)    # Hides chart creation
dashboard.generate(path)                # Single call generates everything
```

## 🔮 Extension Points

### Adding New Metrics

```python
# 1. Implement metric function
def compute_f1(predictions, targets) -> float:
    # ... implementation
    return f1_score

# 2. Add to evaluate_batch
def evaluate_batch(logits, targets):
    ...
    f1 = compute_f1(predictions, targets)
    results.add_metric("f1", f1, "", higher_is_better=True)
    return results
```

### Adding New Benchmarks

```python
# 1. Create benchmark dataset
math_benchmark = BenchmarkDataset(
    name="turkish-math",
    examples=[
        BenchmarkExample(
            prompt="2+2=",
            references=["4", "dört"],
            domain="arithmetic"
        )
    ]
)

# 2. Run benchmark
results = run_generation_benchmark(model, tokenizer, math_benchmark)
```

### Supporting New Model Architectures

```python
# 1. Implement model interface
class BERTModel:
    def __init__(self, config): ...
    def forward(self, input_ids): ...  # Must return logits

# 2. Use with existing pipeline
model = BERTModel(config)
# Rest of the pipeline works unchanged
```

---

**Last Updated:** 2026-09-20  
**Version:** 1.0.0  
**Components:** Sprint 9 (Training) + Sprint 10 (Evaluation)
