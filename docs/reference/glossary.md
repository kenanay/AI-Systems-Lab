# Glossary - Terimler Sözlüğü

Bu sözlük, AI Research Lab platformunda kullanılan teknik terimleri açıklar.

Güncel modül kapsamı için [Dokümantasyon Ana Sayfası](../README.md) ve [Veri Sözleşmeleri](data-contracts.md) esas alınır.

## A

### Activation Function (Aktivasyon Fonksiyonu)
Neural network'te doğrusal olmayan dönüşüm sağlayan fonksiyon. ReLU, GELU, Sigmoid gibi.

**Neden önemli:** Doğrusal katmanların üst üste gelmesi yine doğrusal bir fonksiyon verir. Activation function karmaşık ilişkileri öğrenmeyi sağlar.

**İlgili kavramlar:** ReLU, GELU, Sigmoid, Tanh

---

### Attention (Dikkat Mekanizması)
Token'ların birbirine ne kadar "dikkat ettiğini" hesaplayan mekanizma. Transformer'ın temel bileşeni.

**Formül:**
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
```

**Neden önemli:** Model, cümledeki kelimeler arasındaki ilişkileri dinamik olarak öğrenir.

**İlgili kavramlar:** Self-Attention, Multi-Head Attention, Query, Key, Value

---

## B

### Backpropagation
Model ağırlıklarını güncellemek için gradient'lerin geriye doğru hesaplanması süreci.

**Temel fikir:** Chain rule kullanarak her katmanın loss'a katkısını hesapla.

**Neden önemli:** Deep learning'in temelini oluşturur. Backpropagation olmadan model öğrenemez.

**Ön bilgiler:** Derivative, Chain Rule, Gradient

**İlgili kavramlar:** Gradient, Chain Rule, Optimizer

---

### Batch
Modele aynı anda verilen örnek grubu.

**Örnek:** Batch size = 16 → 16 cümle aynı anda işlenir

**Neden önemli:** 
- GPU'yu verimli kullanır
- Training'i hızlandırır
- Gradient'leri daha stabil yapar

**Trade-off:** Büyük batch → daha fazla memory, daha az güncelleme

**İlgili kavramlar:** Batch Size, Mini-Batch, Gradient Accumulation

---

### BPE (Byte-Pair Encoding)
Sık görülen karakter çiftlerini birleştirerek vocabulary oluşturan tokenization yöntemi.

**Örnek:**
```
"öğretmen" → ["öğ", "ret", "men"]
```

**Avantajları:**
- Subword level tokenization
- Unknown token sorununu azaltır
- Türkçe gibi agglutinative diller için uygun

**İlgili kavramlar:** Tokenization, Vocabulary, Subword

---

## C

### Checkpoint
Training sırasında model ağırlıklarının, optimizer state'inin ve diğer bilgilerin kaydedilmesi.

**İçerik:**
- Model weights
- Optimizer state
- Scheduler state
- Training step
- Config

**Neden önemli:** Training kesintiye uğrarsa kaldığı yerden devam edebilir.

**İlgili kavramlar:** Model Weights, State Dict, Resume Training

---

### Context Length (Bağlam Uzunluğu)
Modelin bir seferde işleyebileceği maksimum token sayısı.

**Örnek:** GPT-3.5 → 4096 tokens, GPT-4 → 8192/32768 tokens

**Trade-off:** Uzun context → daha fazla bilgi ama daha fazla hesaplama ve memory

**İlgili kavramlar:** Sequence Length, Positional Encoding, KV Cache

---

## D

### Dataset
Model eğitimi veya evaluation için hazırlanmış veri koleksiyonu.

**Tipler:**
- **Training set:** Model öğrenmesi için
- **Validation set:** Hiperparametre tuning için
- **Test set:** Final evaluation için

**Canonical Dataset:** Modelden bağımsız, standart format veri havuzu

**İlgili kavramlar:** Data Split, Data Augmentation, Data Lineage

---

### d_model
Transformer'da ana representation boyutu. Embedding ve hidden state'lerin boyutu.

**Örnek:** `d_model=512` → Her token 512 boyutlu vektörle temsil edilir

**Trade-off:** Büyük d_model → daha zengin representation ama daha fazla parametre

**İlgili kavramlar:** Embedding Dimension, Hidden Size

---

## E

### Embedding
Token'ların dense vector'lere dönüştürülmesi.

**Örnek:**
```
"öğretmen" (token_id: 1427) → [0.12, -0.45, 0.89, ...]
```

**Neden önemli:** Token'lar arasındaki semantic ilişkileri yakalayan sürekli uzay.

**İlgili kavramlar:** Token Embedding, Positional Embedding, Vector Space

---

### Epoch
Training verisisininin tamamının bir kez işlenmesi.

**Örnek:** 10 epoch → dataset 10 kez görüldü

**İlgili kavramlar:** Step, Iteration, Training Loop

---

## F

### Fine-Tuning
Pre-trained modelin belirli bir görev için uyarlanması.

**Tipler:**
- **Full Fine-Tuning:** Tüm parametreler güncellenir
- **LoRA:** Sadece ek parametreler eklenir ve güncellenir
- **Prompt Tuning:** Sadece prompt optimize edilir

**İlgili kavramlar:** Transfer Learning, SFT, LoRA, Adapter

---

### Forward Pass
Girdi verisinin model katmanlarından geçerek çıktı üretmesi.

**Süreç:**
```
Input → Embedding → Transformer Blocks → Output → Loss
```

**İlgili kavramlar:** Backward Pass, Inference

---

## G

### Gradient (Gradyan)
Loss fonksiyonunun parametrelere göre türevi. Ağırlıkların hangi yönde güncelleneceğini gösterir.

**Formül:**
```
∂L/∂w
```

**Neden önemli:** Gradient descent ile model öğrenir.

**İlgili kavramlar:** Backpropagation, Gradient Descent, Learning Rate

---

### GELU (Gaussian Error Linear Unit)
Modern Transformer'larda yaygın kullanılan activation function.

**Neden GELU:** ReLU'dan daha smooth, probabilistic yorumu var

**İlgili kavramlar:** ReLU, Activation Function

---

## I

### Inference
Eğitilmiş modelin yeni veri üzerinde tahmin yapması.

**Training vs Inference:**
- Training: Backward pass var, ağırlıklar güncellenir
- Inference: Sadece forward pass, ağırlıklar sabit

**İlgili kavramlar:** Generation, Sampling, KV Cache

---

## K

### KV Cache
Inference sırasında Key ve Value değerlerinin cache'lenmesi. Autoregressive generation'ı hızlandırır.

**Neden önemli:** Her token için önceki token'ların K, V'lerini yeniden hesaplamaya gerek kalmaz.

**Trade-off:** Hız artışı ↔ Memory kullanımı

**İlgili kavramlar:** Autoregressive Generation, Inference Optimization

---

## L

### Learning Rate
Gradient descent'te adım büyüklüğü.

**Örnek:** `lr=0.0001` → Küçük adımlar, yavaş ama stabil öğrenme

**Trade-off:** 
- Yüksek LR → Hızlı ama instabil
- Düşük LR → Yavaş ama stabil

**İlgili kavramlar:** Optimizer, Scheduler, Warmup

---

### Logits
Modelin son linear layer'ının çıktısı. Softmax öncesi skorlar.

**Örnek:**
```
logits = [2.3, -1.5, 4.8, 0.7, ...]
```

Softmax sonrası → olasılıklar: [0.05, 0.001, 0.62, 0.01, ...]

**İlgili kavramlar:** Softmax, Probability, Next Token Prediction

---

### LoRA (Low-Rank Adaptation)
Fine-tuning için parameter-efficient yöntem. Sadece ek düşük boyutlu matrisler eklenir.

**Avantajı:** Çok daha az parametre güncellenir → daha az memory ve hesaplama

**İlgili kavramlar:** PEFT, QLoRA, Adapter, Fine-Tuning

---

### Loss Function (Kayıp Fonksiyonu)
Modelin tahminleriyle gerçek değerler arasındaki farkı ölçen fonksiyon.

**Yaygın loss'lar:**
- **Cross Entropy:** Classification için
- **MSE:** Regression için

**Hedef:** Loss'u minimize et

**İlgili kavramlar:** Cross Entropy, Optimization, Gradient Descent

---

## M

### Multi-Head Attention
Attention mekanizmasının paralel olarak birden fazla "head" ile uygulanması.

**Neden önemli:** Farklı head'ler farklı ilişki tiplerini öğrenebilir.

**Örnek:** 8 head → 8 farklı attention pattern

**İlgili kavramlar:** Attention, Query, Key, Value

---

## O

### Optimizer
Gradient'leri kullanarak model parametrelerini güncelleyen algoritma.

**Yaygın optimizer'lar:**
- **SGD:** Basit, momentum eklenebilir
- **Adam:** Adaptive learning rate
- **AdamW:** Adam + weight decay

**İlgili kavramlar:** Learning Rate, Gradient Descent, Momentum

---

## P

### Perplexity
Language model performansını ölçen metrik. Modelin ne kadar "şaşırdığını" gösterir.

**Formül:**
```
Perplexity = exp(loss)
```

**Yorumlama:** Düşük perplexity → Model metni daha iyi tahmin ediyor

**İlgili kavramlar:** Loss, Evaluation, Language Modeling

---

### Positional Encoding
Token'ların cümledeki pozisyonunu modele bildirmek için kullanılan encoding.

**Neden gerekli:** Attention inherently position-agnostic → pozisyon bilgisi eklenmeli

**Tipler:**
- **Sinusoidal:** Sabit, matematiksel pattern
- **Learned:** Eğitimle öğrenilen

**İlgili kavramlar:** Positional Embedding, RoPE, Sequence Order

---

### Pretraining
Modelin büyük bir dataset üzerinde genel dil yeteneği öğrenmesi.

**Görev:** Genellikle next token prediction

**Sonuç:** Base model (henüz assistant değil)

**İlgili kavramlar:** Base Model, Foundation Model, Next Token Prediction

---

## Q

### Query, Key, Value (Q, K, V)
Attention mekanizmasında kullanılan üç projection.

**Sezgisel açıklama:**
- **Query:** "Ne arıyorum?"
- **Key:** "Ben neyim?"
- **Value:** "Bilgim ne?"

**İlgili kavramlar:** Attention, Multi-Head Attention

---

## R

### RAG (Retrieval-Augmented Generation)
Model output'unu external knowledge ile zenginleştirme.

**Süreç:**
```
Query → Retrieval → Context + Query → LLM → Answer
```

**Neden RAG:** Güncel bilgi, domain knowledge, model halüsinasyonunu azaltma

**İlgili kavramlar:** Vector Database, Embedding, Chunking

---

### ReLU (Rectified Linear Unit)
Basit ama etkili activation function.

**Formül:**
```
ReLU(x) = max(0, x)
```

**İlgili kavramlar:** Activation Function, GELU, Leaky ReLU

---

## S

### Sampling
Modelin probability distribution'ından token seçme yöntemi.

**Stratejiler:**
- **Greedy:** En yüksek olasılıklı token
- **Temperature:** Dağılımı düzleştirme/keskinleştirme
- **Top-k:** En yüksek k token arasından seç
- **Top-p (nucleus):** Kümülatif olasılık p'ye ulaşana kadar

**İlgili kavramlar:** Generation, Temperature, Top-k, Top-p

---

### Softmax
Logit'leri probability distribution'a dönüştüren fonksiyon.

**Formül:**
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

**Özellik:** Çıktı toplamı 1

**İlgili kavramlar:** Logits, Probability, Temperature

---

### SFT (Supervised Fine-Tuning)
Instruction-following için supervised örneklerle fine-tuning.

**Dataset format:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Amaç:** Base model → Assistant model

**İlgili kavramlar:** Instruction Tuning, Fine-Tuning, Chat Model

---

## T

### Temperature
Sampling sırasında probability distribution'ı kontrol eden parametre.

**Etki:**
- **T < 1:** Daha deterministik, güvenli
- **T = 1:** Orijinal dağılım
- **T > 1:** Daha random, yaratıcı

**İlgili kavramlar:** Sampling, Creativity, Determinism

---

### Tensor
Çok boyutlu array. Scalar, vector, matrix'in genelleştirilmişi.

**Boyutlar:**
- 0D: Scalar
- 1D: Vector
- 2D: Matrix
- 3D+: Tensor

**Shape örneği:** `[batch, seq_len, d_model]`

**İlgili kavramlar:** Shape, Matrix, Broadcasting

---

### Token
Metnin atomik birimleri. Kelime, subword veya karakter olabilir.

**Örnek:**
```
"öğretmenlerimizin" → ["öğret", "men", "lerimiz", "in"]
```

**İlgili kavramlar:** Tokenization, Vocabulary, BPE

---

### Tokenizer
Metni token'lara ayıran ve token ID'ye dönüştüren sistem.

**Süreç:**
```
Text → Normalize → Tokenize → ID'lere çevir
```

**İlgili kavramlar:** BPE, Vocabulary, Token ID

---

### Transformer
Attention mekanizmasına dayanan neural network mimarisi.

**Temel bileşenler:**
- Multi-Head Attention
- Feed-Forward Network
- Layer Normalization
- Residual Connections

**Kaynak:** "Attention Is All You Need" (Vaswani et al., 2017)

**İlgili kavramlar:** Attention, GPT, BERT, Encoder, Decoder

---

## V

### Vocabulary (Sözlük)
Tokenizer'ın bildiği tüm token'ların seti.

**Örnek:** `vocab_size=8000` → 8000 unique token

**Trade-off:** 
- Büyük vocab → Daha az token per text ama daha fazla parametre
- Küçük vocab → Daha fazla token per text ama daha az parametre

**İlgili kavramlar:** Token, Tokenizer, Embedding Table

---

## W

### Weight (Ağırlık)
Model parametreleri. Training sırasında öğrenilir.

**Örnek:** Linear layer'daki W matrisi

**Neden önemli:** Model'in tüm bilgisi ağırlıklarda saklıdır

**İlgili kavramlar:** Parameter, Training, Gradient

---

## İlgili Kaynaklar

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

**Not:** Bu sözlük sürekli güncellenmektedir. Yeni terimler eklendikçe genişleyecektir.
