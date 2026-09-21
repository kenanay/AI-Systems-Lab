# Training Configurations

Bu dizin model training için configuration template'lerini içerir.

## Dizin Yapısı

```
configs/
├── models/          # Model architecture configs
│   ├── tiny_model.yaml
│   ├── small_model.yaml
│   └── medium_model.yaml
└── training/        # Custom training configs (kullanıcı tarafından)
```

## Predefined Configs

### Tiny Model (~1.8M parameters)
Test ve debug için minimal model:
- **d_model**: 128
- **n_layers**: 4
- **n_heads**: 4
- **context_length**: 256
- **Kullanım**: Hızlı test, algorithm validation

```bash
python src/training/config.py
```

### Small Model (~7M parameters)
Educational ve prototype için:
- **d_model**: 256
- **n_layers**: 6
- **n_heads**: 8
- **context_length**: 512
- **Kullanım**: Öğrenme, küçük dataset'ler, rapid prototyping

### Medium Model (~42M parameters)
Production-ready small model:
- **d_model**: 512
- **n_layers**: 12
- **n_heads**: 8
- **context_length**: 1024
- **Kullanım**: Real-world applications, Turkish language modeling

## Config Format

Her YAML config şu yapıyı takip eder:

```yaml
version: "1.0.0"
name: "model_name"

model:
  architecture: "transformer_decoder"
  vocab_size: 8000
  context_length: 512
  d_model: 256
  n_layers: 6
  n_heads: 8
  d_ff: 1024
  dropout: 0.1
  activation: "gelu"
  use_bias: true
  tie_embeddings: true

training:
  batch_size: 32
  learning_rate: 0.0003
  max_steps: 100000
  warmup_steps: 1000
  gradient_clip: 1.0
  weight_decay: 0.01
  eval_interval: 1000
  save_interval: 5000
  scheduler: "cosine"
  min_lr: 1e-5

data:
  dataset_path: "/path/to/dataset.parquet"
  tokenizer_id: "tokenizer_id"
  sequence_length: 512
  stride: null
  shuffle: true
  num_workers: 4
  pin_memory: true

metadata:
  description: "Model açıklaması"
  tags: ["turkish", "education"]
```

## Kullanım

### Python'dan Config Yükleme

```python
from pathlib import Path
from src.training.config import TrainingRunConfig

# Template config yükle (data validation skip)
config = TrainingRunConfig.from_yaml(
    Path("configs/models/small_model.yaml"),
    skip_data=True
)

# Dataset bilgilerini ekle
config.data.dataset_path = "/path/to/dataset.parquet"
config.data.tokenizer_id = "tokenizer_abc123"

# Validate
config.validate()  # Full validation with data

# Training'de kullan
print(f"Model: {config.model.architecture}")
print(f"Parameters: ~{config.model.num_parameters / 1e6:.1f}M")
```

### Custom Config Oluşturma

```python
from src.training.config import TrainingRunConfig, ModelArchitectureConfig

# Custom config
config = TrainingRunConfig(
    name="my_custom_model",
    version="1.0.0",
    model=ModelArchitectureConfig(
        vocab_size=10000,
        context_length=1024,
        d_model=384,
        n_layers=8,
        n_heads=12
    )
)

# Save
config.to_yaml(Path("configs/training/my_model.yaml"))
```

## Validation Rules

Config validation otomatik olarak şunları kontrol eder:

### Model Architecture
- `d_model` `n_heads`'e bölünebilir olmalı
- Tüm boyutlar pozitif olmalı
- `dropout` [0, 1) aralığında olmalı
- `activation` geçerli olmalı (gelu, relu, swish)

### Training
- Learning rate pozitif olmalı
- `warmup_steps` <= `max_steps`
- `min_lr` < `learning_rate`
- `scheduler` geçerli olmalı (cosine, linear, constant)

### Data
- `dataset_path` ve `tokenizer_id` gerekli
- `sequence_length` <= `model.context_length`

## Parameter Count Estimation

Config sistem approximate parameter count hesaplar:

```python
config = TrainingRunConfig.from_yaml("configs/models/small_model.yaml")
print(f"Parameters: {config.model.num_parameters:,}")
# Output: Parameters: 6,946,816
```

Formula:
```
Embeddings = vocab_size * d_model + context_length * d_model
Layer = 4 * d_model^2 + 2 * d_model * d_ff + 4 * d_model
Total = Embeddings + n_layers * Layer + (Output layer)
```

## Training Hyperparameters

### Learning Rate Schedule

**Cosine** (default):
```
lr(step) = min_lr + 0.5 * (lr - min_lr) * (1 + cos(π * step / max_steps))
```

**Linear**:
```
lr(step) = lr - (lr - min_lr) * (step / max_steps)
```

**Constant**:
```
lr(step) = lr (warmup sonrası)
```

### Warmup
İlk `warmup_steps` adımda linear warmup:
```
lr(step) = lr * (step / warmup_steps)
```

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

## Best Practices

1. **Tiny ile test et**: Yeni kod değişikliklerini tiny model ile test et
2. **Small ile prototype**: Küçük dataset'lerde small model kullan
3. **Medium için scale**: Production için medium'dan başla
4. **Hyperparameter tuning**: Learning rate en kritik parametre
5. **Eval interval**: Her 1000 step'te evaluation yap
6. **Checkpoint**: Her 5000 step'te checkpoint kaydet

## Version Control

Config dosyaları git'te tutulmalı:
```bash
git add configs/training/my_model.yaml
git commit -m "Add custom model config"
```

Training run'ları config version'ını log'lamalı:
```python
config.metadata["git_hash"] = get_git_hash()
config.metadata["timestamp"] = datetime.now().isoformat()
```
