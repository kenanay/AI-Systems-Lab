# Ön Bilgi Haritası (Prerequisites Map)

Bu doküman AI Research Lab platformunda öğrenme yolculuğu için gereken ön bilgileri ve bunlar arasındaki bağımlılık ilişkilerini gösterir.

## Ön Bilgi Bağımlılık Grafiği

### Temel Matematik

```
Scalar (Sayı)
  ↓
Vector (Vektör)
  ↓
Matrix (Matris)
  ↓
Tensor
```

### Linear Algebra (Doğrusal Cebir)

```
Vector
  ↓
Dot Product (İç Çarpım)
  ├→ Vector Similarity
  └→ Projection
  
Matrix
  ↓
Matrix Multiplication
  ├→ Linear Transformation
  └→ Shape Rules
  
Transpose
  ↓
QK^T in Attention
```

### Calculus (Analiz)

```
Function
  ↓
Derivative (Türev)
  ↓
Partial Derivative
  ↓
Chain Rule
  ↓
Gradient (Gradyan)
  ↓
Backpropagation
  ↓
Optimizer
```

### Probability & Statistics

```
Probability Distribution
  ↓
Softmax
  ├→ Attention Weights
  └→ Classification
  
Expectation
  ↓
Loss Functions
  
Sampling
  ↓
Token Generation
```

## AI/ML Öğrenme Yolu

### Veri İşleme

```
Raw Data
  ↓
[Ön bilgi: File formats, text processing]
  ↓
Data Cleaning
  ↓
[Ön bilgi: Regular expressions, string manipulation]
  ↓
Normalization
  ↓
Dataset
```

### Tokenization

```
Text
  ↓
[Ön bilgi: String, Unicode]
  ↓
Character Tokenization
  ↓
[Ön bilgi: Algorithms, data structures]
  ↓
Word Tokenization
  ↓
[Ön bilgi: Greedy algorithms]
  ↓
BPE / Subword Tokenization
```

### Model Temelleri

```
Tensor
  ↓
[Ön bilgi: Matrix, broadcasting]
  ↓
Linear Layer
  ↓
[Ön bilgi: Matrix multiplication, bias]
  ↓
Activation Function
  ↓
[Ön bilgi: Function, non-linearity]
  ↓
Neural Network
```

### Embedding

```
Token ID
  ↓
[Ön bilgi: Vector, lookup table]
  ↓
Token Embedding
  ↓
[Ön bilgi: Dot product, cosine similarity]
  ↓
Vector Space
```

### Attention

```
Query / Key / Value
  ↓
[Ön bilgi: Matrix multiplication, linear layer]
  ↓
Dot Product Scores
  ↓
[Ön bilgi: Scaling, numerical stability]
  ↓
Softmax
  ↓
[Ön bilgi: Probability distribution]
  ↓
Self-Attention
  ↓
[Ön bilgi: Parallel processing]
  ↓
Multi-Head Attention
```

### Transformer

```
Multi-Head Attention
  +
Feed-Forward Network
  +
[Ön bilgi: Residual connections]
  +
Layer Normalization
  ↓
Transformer Block
  ↓
[Ön bilgi: Stacking layers]
  ↓
Transformer Model
```

### Training

```
Forward Pass
  ↓
[Ön bilgi: Tensor operations]
  ↓
Loss Function
  ↓
[Ön bilgi: Cross entropy, optimization]
  ↓
Backward Pass
  ↓
[Ön bilgi: Chain rule, gradient]
  ↓
Optimizer Step
  ↓
[Ön bilgi: Learning rate, momentum]
  ↓
Model Update
```

### Inference

```
Trained Model
  ↓
[Ön bilgi: Autoregressive generation]
  ↓
Prefill
  ↓
[Ön bilgi: KV cache optimization]
  ↓
Decode Loop
  ↓
[Ön bilgi: Sampling strategies]
  ↓
Token Generation
```

## Kritik Ön Bilgi Noktaları

### Attention'a Geçmeden Önce

**Zorunlu:**
- ✅ Matrix Multiplication
- ✅ Dot Product
- ✅ Softmax
- ✅ Transpose

**Önerilen:**
- △ Probability Distribution
- △ Vector Similarity
- △ Broadcasting

### Transformer'a Geçmeden Önce

**Zorunlu:**
- ✅ Self-Attention
- ✅ Linear Layer
- ✅ Activation Function
- ✅ Layer Normalization

**Önerilen:**
- △ Residual Connection mantığı
- △ Positional Encoding fikri

### Training'e Geçmeden Önce

**Zorunlu:**
- ✅ Forward Pass
- ✅ Loss Function
- ✅ Gradient kavramı
- ✅ Backpropagation fikri

**Önerilen:**
- △ Optimizer çeşitleri
- △ Learning rate scheduling
- △ Regularization

### Distributed Training'e Geçmeden Önce

**Zorunlu:**
- ✅ Single GPU Training
- ✅ Batch processing
- ✅ Memory management
- ✅ Data parallelism fikri

**Önerilen:**
- △ GPU architecture
- △ Communication patterns
- △ Sharding concepts

## Ön Bilgi Durumu İşaretleri

Platformda kullanılan işaretler:

- ✅ **Tamamlandı**: Bu kavram öğrenildi
- ○ **Görülmedi**: Henüz bu kavrama geçilmedi
- ◐ **Başlandı**: Kavram tanıtıldı ama tamamlanmadı
- △ **Önerilen**: Zorunlu değil ama faydalı
- ⚠️ **Eksik**: Bu ileri kavram için gerekli ama henüz görülmedi

## Matematik Ön Bilgi Haritası

### Aritmetik → Linear Algebra

```
Sayılar (Scalar)
  ↓
Toplama, Çarpma
  ↓
Vector
  ↓
Vector İşlemleri
  ├→ Addition
  ├→ Scalar Multiplication
  └→ Dot Product
  ↓
Matrix
  ↓
Matrix İşlemleri
  ├→ Addition
  ├→ Scalar Multiplication
  ├→ Matrix Multiplication
  └→ Transpose
```

### Calculus Yolu

```
Function kavramı
  ↓
Limit fikri
  ↓
Derivative (tek değişkenli)
  ↓
Partial Derivative (çok değişkenli)
  ↓
Gradient (vektör türev)
  ↓
Chain Rule (zincir kuralı)
  ↓
Backpropagation
```

## İhtiyaç Anında Öğrenme (Just-in-Time Learning)

Platform, bir kavrama geçildiğinde eksik ön bilgileri gösterir:

```
Attention'a geçmek istiyorsun
  ↓
Kontrol: Matrix Multiplication? ○
  ↓
Uyarı: "Bu konuyu tam anlamak için Matrix Multiplication gerekli"
  ↓
Seçenekler:
  [Önce Matrix Multiplication'ı Öğren]
  [Hızlı Açıklamayı Aç]
  [Yine de Devam Et]
```

## Minimum Viable Knowledge (MVK)

Her ana kavram için minimum bilgi seti:

### Tokenizer İçin MVK
- String manipulation
- Basic algorithms
- Dictionary/Map veri yapısı

### Attention İçin MVK
- Matrix multiplication
- Dot product
- Softmax
- Transpose

### Training İçin MVK
- Forward pass fikri
- Loss function kavramı
- Gradient fikri (detaylı calculus gerekmez)
- Optimization kavramı

## Ön Bilgi Sınıflandırması

### Tier 0 - Mutlak Temel
İlerlemek için zorunlu, platform içinde öğretilebilir.
- Scalar, Vector, Matrix
- Function kavramı
- Basic programming

### Tier 1 - Temel AI
AI'a başlamak için gerekli.
- Matrix multiplication
- Derivative fikri
- Tensor shape
- Gradient kavramı

### Tier 2 - Orta Seviye
Transformer ve training için gerekli.
- Attention mechanism
- Backpropagation
- Optimizer types
- Loss functions

### Tier 3 - İleri
Production ve optimization için.
- Distributed training
- Mixed precision
- Memory optimization
- System-level considerations

---

**Not:** Bu harita dinamiktir. Kullanıcının öğrenme yoluna göre şekillenecektir.
