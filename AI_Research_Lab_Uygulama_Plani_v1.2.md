# Local AI Research Lab
## Local-First, Uçtan Uca AI Systems Research & Learning Platform

**Belge Türü:** Uygulama Planı / Teknik Yol Haritası  
**Sürüm:** 1.2  
**Revizyon:** v1.1’deki pedagojik açıklama katmanı, ön bilgi haritası, matematik/yazılım açıklama standardı, job/worker mimarisi, data lineage ve immutable dataset ilkeleri korunarak; **local-first ancak local-only olmayan execution target mimarisi, Architecture Atlas, Post-Training, Inference & Serving, Systems for AI, Data & Model Format Explorer, Simulation Lab, Distributed AI, MLOps/Production, AI Application Layer ve AI Security** kapsamları eklendi. Ana kullanıcı deneyimi **Guided Journey / Lab & Simulator / Workspace** modeliyle sadeleştirildi.  
**Amaç:** Gerçek verinin sisteme girişinden başlayarak veri formatları, dataset üretimi, tokenizer, tensor işlemleri, neural network temelleri, embedding, attention, Transformer, farklı model mimarileri, pretraining, post-training, evaluation, inference, serving, RAG, tool/agent uygulamaları, multimodal sistemler, distributed training ve production/MLOps süreçlerine kadar modern bir AI sisteminin yaşam döngüsünü **uygulayarak, simüle ederek, kodlayarak ve gerçek projeler üzerinde çalıştırarak öğrenmeyi** sağlayan bütünleşik bir araştırma ve öğrenme platformu geliştirmektir. İlk çalışma ortamı yerel bilgisayar olacaktır; ancak tasarım **CPU, local GPU, remote GPU, multi-GPU ve cluster** ortamlarına taşınabilir olacaktır. Uygulamadaki her işlem yalnızca çalıştırılmayacak; **ne yapıldığı, neden yapıldığı, hangi ön bilginin gerektiği, matematiksel olarak ne anlama geldiği, kodda nasıl karşılandığı, donanım ve sistem seviyesinde ne olduğu, hangi alternatiflerin bulunduğu ve gerçek AI sistemlerinde neden kullanıldığı** Türkçe ve adım adım açıklanacaktır.

---

# 1. Proje Özeti

Bu projenin amacı yalnızca bir yapay zekâ arayüzü veya dataset yöneticisi geliştirmek değildir. Hedef, kullanıcıya yapay zekâ sistemlerinin nasıl çalıştığını **veriden başlayarak uygulayarak öğreten**, aynı zamanda gerçek verileri işleyerek **model tasarımı, eğitimi, değerlendirmesi, inference, serving ve gerçek AI uygulamaları geliştirilebilen** bir araştırma laboratuvarı oluşturmaktır.

Platform **local-first** yaklaşımıyla başlayacaktır. Bunun anlamı; ilk deneylerin ve öğrenme adımlarının mümkün olduğunca yerel CPU/GPU üzerinde gerçekleştirilmesi, ancak kavramların ve yazılım mimarisinin yalnızca yerel çalışmaya kilitlenmemesidir.

Uygulama şu temel AI yaşam döngüsünü baştan sona kapsayacaktır:

```text
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
Post-Training
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

Sistem iki ana amacı aynı anda yerine getirecektir:

1. **Öğrenme amacı**
   - Transformer mimarisini teorik değil uygulamalı öğrenmek.
   - Tokenizer, embedding, attention, backpropagation gibi yapıların iç işleyişini görmek.
   - Küçük modelleri sıfırdan yazarak eğitmek.
   - Hazır kütüphanelerin arka planda ne yaptığını anlamak.

2. **Üretim ve araştırma amacı**
   - Metin, PDF, Word, Excel, CSV, görsel, ses ve video gibi verileri düzenli bir dataset haline getirmek.
   - RAG, fine-tuning, SFT, LoRA ve multimodal eğitim için veri üretmek.
   - Yerel LLM/VLM modellerini çalıştırmak ve karşılaştırmak.
   - Aynı deneyleri gerektiğinde remote GPU, multi-GPU ve cluster ortamlarında çalıştırabilmek.
   - Model eğitiminden inference ve serving aşamasına kadar AI sistem mühendisliğini öğrenmek.
   - Gerçek AI uygulamalarında RAG, structured output, tool calling, workflow ve agent yapılarını deneyebilmek.
   - Gelecekte daha büyük modeller veya GPU kümeleri ile ölçeklenebilecek bir altyapı oluşturmak.

---

# 2. Temel Proje İlkeleri

Uygulamanın geliştirilmesinde aşağıdaki ilkeler esas alınacaktır.

## 2.1. Modelden bağımsız veri altyapısı

Ana veri yapısı belirli bir modele göre tasarlanmayacaktır.

Yanlış yaklaşım:

```text
Dataset
   ↓
Sadece Qwen formatı
```

Doğru yaklaşım:

```text
Canonical Dataset
      ↓
Dataset Compiler
      ↓
 ┌────┼─────┬──────┬──────┐
 ↓    ↓     ↓      ↓      ↓
Qwen Llama Gemma  RAG    VLM
```

Bu sayede model değişse bile ana veri havuzu yeniden oluşturulmayacaktır.

---

## 2.2. Ham verinin korunması

Ham veri hiçbir zaman doğrudan değiştirilmemelidir.

```text
raw/
```

klasörü yalnızca orijinal kaynakları içerecektir.

Örneğin:

- PDF, PDF olarak
- DOCX, DOCX olarak
- XLSX, XLSX olarak
- JPG, JPG olarak
- WAV, WAV olarak

saklanacaktır.

Temizlenmiş veya dönüştürülmüş içerikler ayrı katmanlarda tutulacaktır.

---

## 2.3. Veri ile eğitim verisi birbirinden ayrılmalıdır

Her veri model eğitimine uygun değildir.

Aşağıdaki ayrım sistemin temel mimari kararlarından biridir:

```text
Kurumsal / güncel bilgi
        ↓
       RAG

Tablosal veri
        ↓
SQL / DuckDB / Python

Model davranışı ve görev örnekleri
        ↓
SFT / LoRA

Büyük metin koleksiyonları
        ↓
Pretraining

Görsel + metin ilişkileri
        ↓
VLM Training
```

---

## 2.4. Öğrenme Modu ve Profesyonel Mod birlikte bulunmalıdır

Uygulama iki çalışma moduna sahip olacaktır.

| Öğrenme Modu | Profesyonel Mod |
|---|---|
| İşlemleri adım adım gösterir | Performans odaklıdır |
| Matrisleri gösterir | Optimize GPU operasyonları kullanır |
| Attention elle hesaplanabilir | PyTorch SDPA kullanır |
| Basit training loop | Gelişmiş trainer |
| Küçük modeller | Büyük modeller |
| Eğitim amaçlı | Gerçek üretim amaçlı |
| Her adımı açıklar | Otomatikleştirilmiştir |

Bu yapı, kullanıcıya hem konuyu öğretir hem de öğrendikten sonra aynı uygulamanın profesyonel kullanımına devam etmesini sağlar.

---

## 2.5. Açıklama-önce (Explanation-First) ilkesi

Uygulamada hiçbir önemli işlem yalnızca bir düğme, grafik veya kod çıktısı olarak gösterilmemelidir. Kullanıcı her aşamada şu soruların cevabını görebilmelidir:

1. **Ne yapıyoruz?**
2. **Neden yapıyoruz?**
3. **Bu adım bir önceki adımdan nasıl çıktı?**
4. **Bu işlem yapılmazsa ne olur?**
5. **Hangi veri giriş olarak kullanılıyor?**
6. **Çıktı tam olarak nedir?**
7. **Matematiksel karşılığı nedir?**
8. **Kodda hangi sınıf/fonksiyon bunu gerçekleştiriyor?**
9. **Gerçek LLM/VLM sistemlerinde bunun karşılığı nedir?**
10. **Bir sonraki adım neden buna ihtiyaç duyuyor?**

Bu açıklamalar isteğe bağlı bir yardım metni değil, uygulamanın temel ürün davranışı olacaktır.

---

## 2.6. Dil ve teknik terim standardı

Uygulamanın ana dili **Türkçe** olacaktır. Ancak yapay zekâ ve yazılım ekosisteminde standartlaşmış teknik terimler zorla Türkçeleştirilmeyecektir.

Örnek kullanım:

```text
Self-Attention (öz-dikkat)
Embedding
Tokenizer
Gradient (gradyan)
Loss Function (kayıp fonksiyonu)
Backpropagation
Forward Pass
Checkpoint
Fine-Tuning
Learning Rate
Batch
Epoch
Logits
```

Kural:

- İlk kullanımda terim kısa Türkçe açıklamasıyla verilebilir.
- Kod, API, class, function ve framework isimleri özgün biçimleriyle korunur.
- Terimin sektörde yaygın İngilizce kullanımı varsa ana etiket İngilizce kalabilir.
- Kullanıcı terimin üzerine geldiğinde veya tıkladığında Türkçe açıklama açılır.
- Uygulama içinde ayrı bir **Glossary / Terimler Sözlüğü** tutulur.
- Aynı kavram farklı ekranlarda farklı Türkçe karşılıklarla adlandırılmaz.

Amaç, Türkçe öğrenme deneyimi ile uluslararası teknik dokümantasyon arasında kopukluk oluşturmamaktır.

---

## 2.7. Ön bilgi (Prerequisite) farkındalığı

Öğrenme rotası basitten zora ilerler; ancak Transformer gibi konularda bazı ileri kavramlar daha erken aşamalarda gerekli olabilir.

Bu durumda uygulama kullanıcıyı engellemek yerine uyarır:

```text
Bu adımı tam anlamak için önerilen ön bilgiler:

✓ Vektör
✓ Matris çarpımı
○ Softmax
○ Türev
○ Chain Rule
```

Eksik bir ön bilgi varsa kullanıcı:

```text
[Devam Et]
[Önce Bu Konuyu Öğren]
[Hızlı Açıklamayı Aç]
```

seçeneklerinden birini kullanabilir.

Böylece öğrenme yolu katı bir doğrusal kurs haline gelmez; **ihtiyaç anında öğrenme (just-in-time learning)** desteklenir.

---

## 2.8. Ön bilgi bağımlılık grafiği

Kavramlar bir knowledge graph / prerequisite graph ile ilişkilendirilecektir.

Örnek:

```text
Scalar
  ↓
Vector
  ↓
Matrix
  ↓
Matrix Multiplication
  ↓
Linear Layer
  ↓
Embedding
  ↓
Q / K / V
  ↓
Self-Attention
  ↓
Multi-Head Attention
  ↓
Transformer Block
```

Backpropagation için:

```text
Function
  ↓
Derivative
  ↓
Partial Derivative
  ↓
Chain Rule
  ↓
Gradient
  ↓
Backpropagation
  ↓
Optimizer
```

Uygulama kullanıcının hangi konuları gördüğünü, hangilerini atladığını ve hangi ileri konuda hangi eksik ön bilginin bulunduğunu gösterebilir.

---

## 2.9. Matematik açıklama standardı

Matematiksel formüller yalnızca gösterilmeyecek; parçalanarak açıklanacaktır.

Her formül için mümkün olduğunda şu yapı kullanılacaktır:

1. **Formül**
2. **Sembollerin anlamı**
3. **Tensor / matris shape bilgisi**
4. **Neden bu formül kullanılıyor?**
5. **Adım adım sayısal örnek**
6. **Geometrik veya sezgisel açıklama**
7. **PyTorch karşılığı**
8. **Gerçek modeldeki kullanımı**
9. **Sık yapılan hata**
10. **İlgili ön bilgiler**

Örnek:

```math
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
```

Açıklama tablosu:

| Sembol | Anlam |
|---|---|
| `Q` | Query matrisi |
| `K` | Key matrisi |
| `V` | Value matrisi |
| `K^T` | Key matrisinin transpose işlemi |
| `d_k` | Key vektör boyutu |
| `softmax` | Skorları normalize eden fonksiyon |

Ayrıca shape bilgisi açıkça gösterilmelidir:

```text
Q : [batch, heads, seq_len, d_k]
K : [batch, heads, seq_len, d_k]
V : [batch, heads, seq_len, d_v]
```

---

## 2.10. Matematik temel alanları

Uygulama gerektiği noktada aşağıdaki matematik alanlarını ayrı mikro öğrenme içerikleri olarak sunmalıdır:

### Aritmetik ve fonksiyonlar

- toplam
- çarpım
- üs
- logaritma
- fonksiyon kavramı

### Linear Algebra

- scalar
- vector
- matrix
- tensor
- dot product
- matrix multiplication
- transpose
- norm
- basis
- projection

### Calculus

- limit fikri
- derivative
- partial derivative
- gradient
- chain rule

### Probability / Statistics

- probability distribution
- expectation
- variance
- sampling
- likelihood

### Optimization

- objective function
- gradient descent
- learning rate
- momentum
- Adam / AdamW

### Information Theory

İleri aşamalarda:

- entropy
- cross-entropy
- KL divergence

Bu matematik modülleri yapay zekâdan kopuk soyut dersler olarak değil, ihtiyaç duyuldukları model adımına bağlı olarak açılmalıdır.

---

## 2.11. LaTeX ve matematiksel gösterim

Formüller uygulamada standart matematik notasyonu ile gösterilecektir.

Frontend tarafında:

```text
KaTeX
veya
MathJax
```

kullanılabilir.

Her formül için istenirse:

```text
[Görsel Formül]
[LaTeX Kaynağı]
[Sayısal Örnek]
[Kod Karşılığı]
```

sekmeleri gösterilebilir.

Bu sayede kullanıcı yalnız formülü anlamakla kalmaz, teknik dokümanlarda kullanılan LaTeX gösterimine de alışır.

---

## 2.12. Yazılım altyapısının açıklanması

Uygulama yalnız model matematiğini değil, bu matematiğin yazılım sistemi içinde nasıl gerçekleştirildiğini de açıklamalıdır.

Örneğin bir `Attention` ekranında:

```text
Matematik:
Q = XWq

Kod:
q = self.q_proj(x)

Framework:
PyTorch

Dosya:
src/model/attention.py

Çalıştığı cihaz:
CUDA GPU

dtype:
bfloat16

Input Shape:
[B, T, C]

Output Shape:
[B, T, C]
```

gösterilebilir.

Her önemli kod parçasında mümkün olduğunda:

- input
- output
- shape
- dtype
- device
- class/function
- çağıran modül
- bağımlılıklar
- memory etkisi
- computational complexity
- optimize sürüm
- eğitim amaçlı sade sürüm

bilgileri erişilebilir olmalıdır.

---

## 2.13. “Hiçbir kritik adımı atlama” ilkesi

Uygulamanın eğitim modunda otomasyon, kavramsal adımları görünmez hale getirmemelidir.

Örneğin kullanıcı:

```text
Train Model
```

düğmesine bastığında yalnızca progress bar görmek yerine şu zinciri açabilmelidir:

```text
Dataset
 ↓
Sample
 ↓
Tokenization
 ↓
Batch
 ↓
Embedding
 ↓
Transformer
 ↓
Logits
 ↓
Target
 ↓
Loss
 ↓
Backward
 ↓
Gradient
 ↓
Optimizer Step
 ↓
Updated Weights
```

Kullanıcı dilerse üst düzey görünümde kalabilir; ancak her kutuyu açarak işlemi en düşük anlamlı seviyeye kadar inceleyebilmelidir.

---

## 2.14. Motivasyon, ipucu ve tarihçe kartları

Öğrenmenin sürekliliğini desteklemek için uygun noktalarda kısa içerikler gösterilebilir:

### İpucu

> Shape uyuşmazlıkları Transformer kodlarken en sık karşılaşacağın hata sınıflarından biridir. Matris çarpımından önce son iki boyutu kontrol et.

### Neden önemli?

> Tokenizer kalitesi yalnız veri hazırlama konusu değildir; context window'un ne kadar verimli kullanıldığını da etkiler.

### Sık yapılan hata

> Train ve test verisine aynı dokümanın parçalarının dağılması evaluation sonucunu yapay biçimde yükseltebilir.

### Tarihçe / Anekdot

Attention, Transformer, BPE, CUDA veya optimizasyon yöntemleriyle ilgili kısa tarihçe notları gösterilebilir. Bu içerikler mümkün olduğunca doğrulanabilir teknik kaynaklara dayanmalı; eğlence amacıyla uydurma anekdot kullanılmamalıdır.

### Bugün ne öğrendin?

Bir modülün sonunda:

```text
✓ Token ile Token ID arasındaki fark
✓ Vocabulary'nin rolü
✓ BPE merge işlemi
✓ Tokenizer'ın modelden neden ayrı olduğu
```

gibi kısa bir ilerleme özeti gösterilebilir.

---

## 2.15. Kontrol soruları ve aktif öğrenme

Kullanıcı yalnızca içeriği okumamalıdır. Uygulama küçük kontrol noktaları sunabilir.

Örnek:

> `Q` shape'i `[2, 8, 16, 64]`, `K` shape'i `[2, 8, 16, 64]` ise `QKᵀ` sonucunun shape'i nedir?

Kullanıcı yanıt verdikten sonra:

- doğru cevap
- neden
- matris çarpımı açıklaması
- ilgili ön bilgi bağlantısı

gösterilir.

Bu mekanizma zorunlu sınav olarak değil, öğrenmeyi pekiştiren isteğe bağlı bir araç olarak tasarlanmalıdır.

---

## 2.16. Local-first, execution-target-independent ilkesi

Platformun ilk çalışma ortamı yerel bilgisayardır; ancak mimari yalnızca yerel çalışmaya bağlanmamalıdır.

Aynı deney mantıksal olarak farklı execution target'larda çalıştırılabilmelidir:

```text
CPU
 ↓
Local GPU
 ↓
Remote GPU
 ↓
Multi-GPU
 ↓
Cluster
```

Uygulama öğrenme sırasında şu farkları görünür hale getirmelidir:

- aynı kodun farklı donanımda ne kadar değiştiği,
- device seçiminin tensor hareketlerine etkisi,
- veri transferi,
- VRAM / RAM farkı,
- tek GPU ile multi-GPU arasındaki senkronizasyon,
- local filesystem ile object storage farkı,
- tek makine ile cluster orchestration farkı.

Amaç, yerelde öğrenilen kavramların büyük AI altyapılarında nasıl ölçeklendiğini göstermektir.

---

## 2.17. Progressive Disclosure — karmaşıklığı aşamalı açma

Platform çok geniş bir kapsam taşımasına rağmen kullanıcı arayüzü aynı anda tüm ayrıntıları göstermemelidir.

Ana prensip:

> **Özellik sayısını değil, aynı anda görünür olan karmaşıklığı azalt.**

Bir işlem üç derinlik seviyesinde sunulabilir:

```text
Seviye 1 — Özet
Seviye 2 — Öğrenme / Açıklama
Seviye 3 — Sistem / İleri Teknik Ayrıntı
```

Örneğin model eğitimi ekranında yeni başlayan kullanıcı yalnız:

```text
Dataset
Model
Loss
Progress
```

görürken; ileri görünümde:

```text
dataloader workers
device transfers
activation memory
gradient norm
optimizer states
CUDA utilization
communication overhead
```

gibi ayrıntılar açılabilir.

---

## 2.18. Ana kullanıcı deneyimi: Guided Journey / Lab & Simulator / Workspace

Uygulamanın geniş kapsamı üç ana çalışma yüzeyinde toplanmalıdır:

### Guided Journey

Basitten zora sıralı öğrenme yolu.

```text
Veri
 ↓
Tokenizer
 ↓
Tensor
 ↓
Neural Network
 ↓
Attention
 ↓
Transformer
 ↓
Training
 ↓
...
```

### Lab & Simulator

Bir kavramı gerçek bir büyük eğitim çalıştırmadan bağımsız olarak deneyleme alanı.

Örnek:

- Attention Simulator
- Tokenizer Lab
- GPU Memory Simulator
- Quantization Simulator
- RAG Simulator
- Distributed Training Simulator

### Workspace

Gerçek projelerin yürütüldüğü alan.

```text
Dataset
Model
Training Run
Checkpoint
Evaluation
Deployment
```

Bu üçlü yapı, eğitim içeriği ile gerçek mühendislik çalışmalarının birbirini boğmadan aynı platformda bulunmasını sağlar.

---

## 2.19. Simülasyon ile gerçek işlem açıkça ayrılmalıdır

Uygulamada bazı ileri düzey işlemler gerçek donanım veya çok büyük hesaplama gerektirebilir. Bu durumlarda öğretici simülasyon kullanılabilir.

Her çıktı açıkça şu sınıflardan biriyle etiketlenmelidir:

```text
REAL EXECUTION
SIMULATION
ESTIMATION
EDUCATIONAL APPROXIMATION
```

Örneğin 70B modelin eğitim maliyeti yerel bilgisayarda gerçek olarak ölçülemez; ancak parametre, precision, optimizer state ve activation varsayımları üzerinden öğretici bir memory simulation yapılabilir.

Simülasyon hiçbir zaman gerçek benchmark sonucu gibi sunulmamalıdır.

---

## 2.20. AI Journey Map

Kullanıcının tüm öğrenme yolculuğunu tek bakışta görebileceği bir harita bulunmalıdır:

```text
FOUNDATIONS
    ↓
DATA
    ↓
FORMATS
    ↓
TOKENIZATION
    ↓
TENSORS
    ↓
NEURAL NETWORKS
    ↓
ATTENTION
    ↓
TRANSFORMERS
    ↓
MODEL ARCHITECTURES
    ↓
PRETRAINING
    ↓
POST-TRAINING
    ↓
EVALUATION
    ↓
INFERENCE
    ↓
SERVING
    ↓
RAG / TOOLS / AGENTS
    ↓
MULTIMODAL
    ↓
DISTRIBUTED AI
    ↓
PRODUCTION / MLOPS
```

Durum göstergeleri:

```text
○ NOT_SEEN
◐ IN_PROGRESS
● COMPLETED
△ NEEDS_REVIEW
```

---

# 3. Hedef Kullanıcı Profili

Uygulama aşağıdaki kullanıcı grupları için tasarlanabilir:

- Yapay zekâ öğrenmek isteyen yazılımcılar
- Transformer mimarilerini uygulamalı öğrenmek isteyen araştırmacılar
- Yerel LLM/VLM kullanmak isteyen geliştiriciler
- Kendi veri setini oluşturmak isteyen kurumlar
- Eğitim alanında yapay zekâ geliştirmek isteyen araştırmacılar
- Üniversite öğrencileri
- Yapay zekâ laboratuvarları
- Veri bilimi ekipleri
- Kurum içi bilgi sistemleri geliştiren ekipler

İlk sürüm öncelikle **tek kullanıcı / yerel araştırma laboratuvarı** olarak geliştirilecektir.

---

# 4. Uygulamanın Ana Modülleri

Uygulama aşağıdaki ana modüllerden oluşacaktır.

```text
Local AI Research Lab
│
├── Guided Journey
├── Lab & Simulator
├── Workspace
│
├── Project Manager
├── Data Lab
├── Dataset Manager
├── Dataset Compiler
├── Data & Model Format Explorer
│
├── Learning Guidance Engine
├── Prerequisite / Knowledge Map
├── Glossary
├── Math Lab
├── Systems for AI Lab
│
├── Tokenizer Lab
├── Tensor Lab
├── Neural Network Lab
├── Embedding Lab
├── Attention Lab
├── Transformer Lab
├── Architecture Atlas
├── Model Builder
│
├── Training Lab
├── Post-Training Lab
├── Evaluation Lab
├── Inference & Serving Lab
├── Distributed AI Lab
│
├── RAG Lab
├── AI Application Lab
├── Multimodal Lab
├── AI Security Lab
│
├── Experiment Manager
├── Model Registry
├── Deployment / MLOps
└── System Monitor
```

---

# 5. Proje Yönetim Modülü

Her çalışma bir proje olarak oluşturulacaktır.

Örnek:

```text
projects/
├── kutahya-mte-model/
├── turkish-mini-gpt/
├── document-rag-test/
└── vision-language-test/
```

Her projenin aşağıdaki bileşenleri olacaktır:

```text
Project
│
├── Datasets
├── Tokenizers
├── Models
├── Experiments
├── Training Runs
├── Checkpoints
├── Evaluations
├── RAG Indexes
└── Reports
```

## Proje bilgileri

- Proje adı
- Açıklama
- Oluşturma tarihi
- Veri seti sürümü
- Tokenizer sürümü
- Model mimarisi
- Eğitim konfigürasyonu
- Donanım
- Kullanılan GPU
- Son deney
- Son checkpoint
- Son evaluation sonuçları

---

# 6. Data Lab

Data Lab, uygulamanın ilk ve en önemli modüllerinden biridir.

Amaç:

> Farklı kaynaklardan gelen verileri modele hazır hale getirmek.

Desteklenecek veri türleri:

## Metin

- TXT
- MD
- HTML
- JSON
- JSONL

## Belgeler

- PDF
- DOCX
- PPTX

## Tablo

- XLSX
- XLS
- CSV
- TSV
- Parquet

## Görsel

- JPG
- JPEG
- PNG
- WebP
- TIFF

## Ses

- WAV
- FLAC
- MP3

## Video

- MP4
- MOV
- WebM

---

# 7. Veri Alma — Ingestion Pipeline

Her veri tipi için ayrı parser bulunacaktır.

```text
src/
└── ingestion/
    ├── text.py
    ├── pdf.py
    ├── docx.py
    ├── xlsx.py
    ├── csv.py
    ├── image.py
    ├── audio.py
    └── video.py
```

Pipeline:

```text
Dosya
 ↓
Dosya Türü Algılama
 ↓
Metadata Çıkarma
 ↓
İçerik Çıkarma
 ↓
İlişkileri Çıkarma
 ↓
Canonical Dataset
```

Örneğin bir PDF'den:

- Başlık
- Sayfa sayısı
- Metin
- Bölüm başlıkları
- Tablolar
- Görseller
- Sayfa numaraları
- Dosya hash değeri
- Kaynak bilgisi

çıkarılacaktır.

---

# 8. Canonical Dataset Standardı

Uygulamanın en önemli bileşenlerinden biri modelden bağımsız veri formatıdır.

Önerilen yapı:

```text
dataset/
│
├── README.md
├── dataset.yaml
│
├── raw/
│   ├── documents/
│   ├── images/
│   ├── tables/
│   ├── audio/
│   └── video/
│
├── normalized/
│   ├── documents.parquet
│   ├── tables.parquet
│   ├── images.parquet
│   └── files.parquet
│
├── extracted/
│   ├── text/
│   ├── tables/
│   ├── images/
│   └── metadata/
│
├── training/
│   ├── pretraining/
│   ├── sft/
│   ├── preference/
│   └── multimodal/
│
├── rag/
│   ├── documents.parquet
│   └── chunks.parquet
│
├── evaluation/
│
├── splits/
│
└── manifest/
    ├── files.parquet
    ├── relationships.parquet
    └── checksums.sha256
```

---

# 9. Temel Dosya Formatları

## Ana veri formatı

**Parquet**

Kullanılacağı alanlar:

- Metadata
- Doküman kayıtları
- Tablo verileri
- RAG chunk verileri
- Evaluation sonuçları
- Dataset indeksleri

## Eğitim formatı

**JSONL**

Özellikle:

- SFT
- Chat dataset
- Prompt/completion
- QA

## Büyük medya

**WebDataset TAR**

Özellikle:

- milyonlarca görsel
- ses
- video frame koleksiyonları

## Kaynak dosyalar

Orijinal formatlarında korunacaktır.

---

# 10. Manifest Sistemi

Her dosyanın benzersiz kimliği olmalıdır.

Örnek:

```text
DOC-00000001
IMG-00000001
TAB-00000001
AUD-00000001
VID-00000001
```

`files.parquet` örnek alanları:

```text
id
relative_path
media_type
mime_type
sha256
size_bytes
language
source
license
copyright_status
created_at
modified_at
dataset_version
security_level
pii
quality_score
```

Bu sistem aşağıdaki faydaları sağlar:

- Duplicate tespiti
- Dosya bütünlük kontrolü
- Veri kaynağını izleme
- Sürüm kontrolü
- Lisans kontrolü
- PII kontrolü
- Eğitim verisi seçimi

---

# 11. Veri Kalitesi Pipeline'ı

Bir dataset yalnızca çok fazla veri içerdiği için kaliteli değildir.

Pipeline:

```text
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

## Kalite kontrolleri

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

---

# 12. Dataset Split Sistemi

Dataset otomatik veya manuel bölünebilmelidir.

```text
Train      %80
Validation %10
Test       %10
```

Ancak split işlemi yalnız random yapılmamalıdır.

Örneğin aynı belgenin farklı sayfalarının hem train hem test'e düşmesi veri sızıntısına neden olabilir.

Bu nedenle:

```text
document_id
source_id
duplicate_group
```

alanları dikkate alınmalıdır.

---

# 13. Dataset Compiler

Canonical dataset'ten farklı kullanım amaçlarına uygun datasetler üretilecektir.

```text
Canonical Dataset
      │
      ├── Pretraining Export
      ├── SFT Export
      ├── RAG Export
      ├── VLM Export
      ├── Classification Export
      └── Evaluation Export
```

## Pretraining örneği

```json
{"text": "Kütahya ilinde mesleki eğitim..."}
```

## SFT örneği

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Bu performans göstergesini değerlendir."
    },
    {
      "role": "assistant",
      "content": "Performans göstergesinin..."
    }
  ]
}
```

## RAG örneği

```json
{
  "chunk_id": "CHUNK-00001",
  "document_id": "DOC-00001",
  "page": 12,
  "section": "Demografik Veriler",
  "text": "...",
  "source": "...",
  "language": "tr"
}
```

---

# 14. Tokenizer Lab

Tokenizer Lab, Transformer öğrenme sürecinin ilk eğitim modülüdür.

Desteklenecek tokenizer türleri:

- Character tokenizer
- Word tokenizer
- BPE
- Byte-Level BPE
- WordPiece
- Unigram

Kullanıcı şu işlemleri yapabilmelidir:

```text
Metin
 ↓
Normalization
 ↓
Pre-tokenization
 ↓
Vocabulary Training
 ↓
Tokenization
 ↓
Token IDs
```

Örneğin:

```text
Kütahya'da eğitim çalışmaları yürütülmektedir.
```

çıktısı:

```text
["Kütahya", "'", "da", "eğitim", "çalışmaları", ...]
```

ve:

```text
[1427, 18, 54, 913, 3412, ...]
```

olarak gösterilebilir.

## Tokenizer ekranında gösterilecek bilgiler

- Vocabulary size
- Unknown token oranı
- Ortalama token / kelime oranı
- Türkçe kelimelerin parçalanma oranı
- En sık tokenlar
- En uzun tokenlar
- Compression ratio
- Token dağılımı

---

# 15. Tensor Lab

Transformer öğrenmeden önce tensor kavramlarının anlaşılması gerekir.

Modül konuları:

- Scalar
- Vector
- Matrix
- Tensor
- Shape
- Reshape
- Broadcasting
- Matrix multiplication
- Batch
- GPU tensor
- dtype

Örnek:

```text
Input:

[4, 8, 16]

Batch size = 4
Sequence length = 8
Embedding dimension = 16
```

Kullanıcı gerçek tensor değerlerini görebilmelidir.

---

# 16. Neural Network Lab

Transformer'a geçmeden önce temel sinir ağı kavramları uygulanacaktır.

Konular:

- Linear Layer
- Weight
- Bias
- Activation
- ReLU
- GELU
- Softmax
- Loss
- Cross Entropy
- Gradient
- Backpropagation
- Optimizer
- Learning Rate

Örnek:

```text
Input
 ↓
Linear
 ↓
GELU
 ↓
Linear
 ↓
Output
```

Kullanıcı weight değerlerinin eğitim sırasında nasıl değiştiğini görebilmelidir.

---

# 17. Embedding Lab

Amaç:

> Token ID'lerinin vektör uzayına nasıl dönüştüğünü göstermek.

```text
Token
 ↓
Token ID
 ↓
Embedding Table
 ↓
Vector
```

Örnek:

```text
öğretmen
 ↓
9134
 ↓
[0.12, -0.32, 0.88, ...]
```

Görselleştirmeler:

- 2D embedding projection
- Cosine similarity
- En yakın tokenlar
- Token benzerlik matrisi

---

# 18. Positional Encoding Lab

Transformer'ın kelime sırasını nasıl öğrendiği ayrı bir modülde gösterilecektir.

Desteklenebilecek yöntemler:

- Sinusoidal positional encoding
- Learned positional embeddings
- RoPE

Örnek:

```text
Token Embedding
      +
Position Information
      ↓
Transformer Input
```

---

# 19. Attention Lab

Bu modül uygulamanın en önemli eğitim bileşenlerinden biridir.

Temel denklem:

```text
Q = XWq
K = XWk
V = XWv
```

Ardından:

```text
Attention(Q,K,V) =
softmax(QKᵀ / √d_k)V
```

Uygulama aşağıdaki aşamaları ayrı ayrı gösterecektir:

1. Input matrix
2. Wq
3. Wk
4. Wv
5. Q
6. K
7. V
8. QKᵀ
9. Scaling
10. Mask
11. Softmax
12. Attention weights
13. Output

## Görsel Attention Heatmap

Örneğin:

```text
             Kütahya  eğitim  okul  öğrenci

Kütahya       0.41     0.12   0.31   0.16
eğitim        0.08     0.29   0.31   0.32
okul          0.11     0.32   0.36   0.21
öğrenci       0.04     0.41   0.24   0.31
```

---

# 20. Multi-Head Attention Lab

Kullanıcı:

- head sayısını değiştirebilmeli
- her head'i ayrı inceleyebilmeli
- attention matrislerini karşılaştırabilmeli
- head outputlarının concat işlemini görebilmeli

Pipeline:

```text
Input
 │
 ├── Head 1
 ├── Head 2
 ├── Head 3
 └── Head N
 │
 ↓
Concat
 ↓
Linear Projection
```

---

# 21. Transformer Block Lab

Transformer bloğu görsel olarak oluşturulacaktır.

```text
Input
 ↓
LayerNorm
 ↓
Multi-Head Attention
 ↓
Residual Connection
 ↓
LayerNorm
 ↓
Feed Forward Network
 ↓
Residual Connection
 ↓
Output
```

Kullanıcı parametreleri değiştirebilmelidir:

- d_model
- num_heads
- d_ff
- dropout
- layer count
- activation
- normalization type

---

# 22. Model Builder

Kullanıcı GUI üzerinden model oluşturabilmelidir.

Örnek:

```text
Vocabulary Size: 8000
Context Length: 512
Embedding Size: 256
Layers: 6
Heads: 8
FFN Size: 1024
Dropout: 0.1
```

Uygulama tahmini parametre sayısını hesaplamalıdır.

Örnek:

```text
Model Parameters:
28,413,952
```

---

# 23. Mini GPT Modülü

İlk gerçek sıfırdan model bu modülde oluşturulacaktır.

Önerilen ilk model:

```text
10M – 30M parametre
```

Daha sonra:

```text
30M
↓
50M
↓
100M
↓
300M
```

şeklinde ölçeklenebilir.

İlk modelde amaç performans değil, sürecin öğrenilmesidir.

---

# 24. Eğitim Pipeline'ı

```text
Dataset
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Batch
 ↓
Forward Pass
 ↓
Logits
 ↓
Loss
 ↓
Backward
 ↓
Gradient
 ↓
Optimizer
 ↓
Weight Update
 ↓
Checkpoint
```

---

# 25. Training Lab

Eğitim ekranında aşağıdaki bilgiler canlı gösterilmelidir.

```text
Training Run: RUN-0042

Step            427 / 12000
Epoch           1
Train Loss      3.84
Validation Loss 3.91
Learning Rate   0.00027
Tokens/sec      41,228
GPU             92%
VRAM            7.8 / 12 GB
```

Grafikler:

- Training loss
- Validation loss
- Learning rate
- Gradient norm
- Tokens/sec
- GPU usage
- VRAM usage
- CPU usage
- RAM usage

---

# 26. Backpropagation Görselleştirmesi

Öğrenme Modu'nda model ağırlıklarının nasıl değiştiği gösterilecektir.

Örnek:

```text
Weight:
0.23541

Gradient:
-0.00831

Learning Rate:
0.001
```

Hesap:

```text
new_weight =
weight - learning_rate × gradient
```

Sonuç:

```text
0.23541831
```

Bu bölüm sayesinde backpropagation teorik olmaktan çıkacaktır.

---

# 27. Optimizer Lab

Karşılaştırılacak optimizerlar:

- SGD
- Momentum
- Adam
- AdamW

Gösterilecek özellikler:

- convergence
- loss değişimi
- weight update
- learning rate
- gradient behavior

---

# 28. Learning Rate Lab

Desteklenecek scheduler türleri:

- Constant
- Linear
- Cosine
- Warmup + Cosine
- Step decay

Grafiklerle karşılaştırılacaktır.

---

# 29. Checkpoint Sistemi

Her eğitim çalışması checkpoint üretebilmelidir.

```text
checkpoints/
├── step-001000/
├── step-002000/
├── step-003000/
└── best/
```

Checkpoint içinde:

- Model weights
- Optimizer state
- Scheduler state
- Training step
- Dataset version
- Tokenizer version
- Config
- Random seed

saklanmalıdır.

---

# 30. Resume Training

Eğitim yarıda kesildiğinde:

```text
Resume from checkpoint
```

seçeneği bulunmalıdır.

Bu özellik büyük eğitimlerde kritik olacaktır.

---

# 31. Evaluation Lab

Model yalnız loss ile değerlendirilmemelidir.

Evaluation dataset kategorileri:

```text
evaluation/
├── language_modeling.jsonl
├── qa.jsonl
├── document_analysis.jsonl
├── table_reasoning.jsonl
├── image_understanding.jsonl
└── domain_tasks.jsonl
```

Ölçümler:

- Validation loss
- Perplexity
- Exact match
- Accuracy
- F1
- Semantic similarity
- Human evaluation
- Domain specific score

---

# 32. Model Karşılaştırma

Aynı dataset üzerinde farklı modeller karşılaştırılacaktır.

Örnek:

| Model | Parametre | Val Loss | Hız | VRAM |
|---|---:|---:|---:|---:|
| MiniGPT-20M | 20M | 3.21 | yüksek | düşük |
| MiniGPT-50M | 50M | 2.83 | orta | orta |
| MiniGPT-100M | 100M | 2.49 | düşük | yüksek |

Amaç:

> Model büyüdükçe gerçekten ne kazanıldığını ölçmek.

---

# 33. Inference & Serving Lab

Eğitilen veya dışarıdan yüklenen modelin yalnızca çıktı üretmesi değil, **modelin inference sırasında içeride ne yaptığı** da öğrenilecektir.

Temel inference zinciri:

```text
Prompt
 ↓
Tokenizer
 ↓
Input Tokens
 ↓
Prefill
 ↓
KV Cache
 ↓
Decode Loop
 ↓
Logits
 ↓
Sampling
 ↓
Next Token
 ↓
Token Streaming
```

## 33.1. Generation parametreleri

- Temperature
- Top-k
- Top-p
- Max tokens
- Repetition penalty
- Seed

Örnek:

```text
Prompt:
Kütahya ilinde mesleki eğitim

Model:
...
```

## 33.2. Inference kavramları

- Prefill
- Decode
- KV Cache
- Autoregressive generation
- Sampling
- Greedy decoding
- Beam search
- Token streaming
- Batch inference
- Continuous batching

## 33.3. Serving metrikleri

Uygulama mümkün olduğunda aşağıdaki metrikleri açıklamalı ve ölçmelidir:

```text
TTFT    = Time To First Token
ITL     = Inter-Token Latency
TPS     = Tokens Per Second
Latency
Throughput
Concurrent Requests
VRAM Usage
```

## 33.4. Inference engine karşılaştırması

İleri aşamada aynı model farklı runtime / inference engine'lerde denenebilir:

- PyTorch / Transformers
- llama.cpp
- vLLM
- TensorRT-LLM
- MLX
- diğer desteklenen runtime'lar

Amaç yalnızca “hangisi hızlı?” sorusunu değil, **neden hız farkı oluştuğunu** da anlamaktır.

## 33.5. Serving katmanı

Model gerçek bir servise dönüştürüldüğünde:

```text
Model
 ↓
Inference Engine
 ↓
API Server
 ↓
Streaming
 ↓
Client Application
```

zinciri gösterilmelidir.

İleri konular:

- REST
- WebSocket
- streaming responses
- authentication
- rate limiting
- batching
- load balancing
- request queue
- observability

---

# 34. RAG Lab

RAG sistemi ana uygulamanın önemli bir parçasıdır.

Pipeline:

```text
Document
 ↓
Chunking
 ↓
Embedding
 ↓
Vector Database
 ↓
Query
 ↓
Retrieval
 ↓
Context
 ↓
LLM
 ↓
Answer
```

Desteklenebilecek vector database sistemleri:

- FAISS
- Qdrant
- Chroma
- Milvus
- pgvector

İlk sürüm için:

```text
FAISS veya Qdrant
```

yeterlidir.

---

# 35. Chunking Lab

Kullanıcı farklı chunking yöntemlerini karşılaştırabilmelidir.

- Fixed token chunk
- Sentence based
- Paragraph based
- Heading based
- Semantic chunking

Parametreler:

```text
chunk_size
chunk_overlap
```

Evaluation ile en iyi yöntem karşılaştırılabilir.

---

# 36. Table / SQL Agent

Tablosal veriler RAG'e zorla dönüştürülmemelidir.

XLSX / CSV / Parquet için:

```text
Table
 ↓
DuckDB
 ↓
SQL
 ↓
Result
 ↓
LLM
```

Bu sayede örneğin:

> 2026 yılında öğrenci sayısı en fazla olan okul hangisidir?

gibi sorular doğrudan tablo üzerinden hesaplanabilir.

---

# 37. Fine-Tuning Lab

Fine-Tuning Lab, daha geniş **Post-Training** sürecinin bir alt bileşenidir.

Desteklenecek yöntemler:

- Full Fine-Tuning
- SFT
- LoRA
- QLoRA

Akış:

```text
Base Model
 ↓
Training Dataset
 ↓
LoRA Configuration
 ↓
Training
 ↓
Adapter
 ↓
Evaluation
```

---

# 38. Fine-Tuning Dataset Builder

Canonical dataset'ten:

```text
messages.jsonl
```

üretilecektir.

Örnek:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

Dataset belirli modele özel tokenlar içermemelidir.

Modelin chat template'i eğitim sırasında uygulanmalıdır.

---

# 39. Multimodal Lab

Multimodal özellikler ilk sürümde değil, sonraki aşamalarda geliştirilecektir.

Hedef:

```text
Text
Image
Audio
Video
Table
   ↓
Encoder
   ↓
Common Representation
   ↓
LLM
```

---

# 40. Vision Language Model Lab

Pipeline:

```text
Image
 ↓
Vision Encoder
 ↓
Projection Layer
 ↓
LLM Token Space
 ↓
Transformer
 ↓
Text Output
```

Dataset örneği:

```json
{
  "images": ["images/IMG-0001.jpg"],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": "Bu görseli açıkla."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "..."}
      ]
    }
  ]
}
```

---

# 41. Görsel Veri Yönetimi

Her görsel için metadata tutulmalıdır.

```text
id
path
mime_type
width
height
sha256
caption
ocr_text
source
license
document_id
```

Aynı görsel:

- VLM
- RAG
- Image retrieval
- Caption training
- Classification

için kullanılabilir.

---

# 42. Audio Lab

İleri sürüm özellikleri:

```text
Audio
 ↓
Feature Extraction
 ↓
Audio Encoder
 ↓
Projection
 ↓
LLM
```

Kullanım alanları:

- Speech-to-text
- Audio classification
- Multimodal reasoning

---

# 43. Deney Yönetimi

Her eğitim veya test bir deney olarak kaydedilecektir.

```text
Experiment ID
Dataset Version
Tokenizer Version
Model Config
Training Config
Hardware
Metrics
Checkpoint
Notes
```

Örnek:

```text
EXP-2026-0042
```

Bu sayede aynı deney tekrar üretilebilir.

---

# 44. Reproducibility

Her deney için aşağıdaki bilgiler kaydedilmelidir:

- Random seed
- Python version
- PyTorch version
- CUDA version
- GPU modeli
- Dataset hash
- Config hash
- Git commit
- Training parameters

---

# 45. Model Registry

Eğitilen modeller kayıt altına alınacaktır.

```text
models/
├── mini-gpt-20m/
├── mini-gpt-50m/
├── domain-model-v1/
└── fine-tuned-model-v2/
```

Bilgiler:

- Model name
- Architecture
- Parameters
- Tokenizer
- Dataset
- Training date
- Evaluation
- Checkpoint
- License
- Notes

---

# 46. Dataset Registry

Benzer biçimde datasetler versiyonlanmalıdır.

```text
dataset-v1.0.0
dataset-v1.1.0
dataset-v2.0.0
```

SemVer yaklaşımı:

```text
1.0.0 → ilk sürüm
1.1.0 → yeni veri
1.1.1 → metadata düzeltmesi
2.0.0 → şema değişikliği
```

---

# 47. Veri Güvenliği

Dataset'e aşağıdaki alanlar eklenmelidir:

```text
PUBLIC
INTERNAL
RESTRICTED
PERSONAL
```

Ek metadata:

```text
pii
license
copyright
permission
usage_scope
```

Özellikle kişisel veya kurum içi veri:

> Pretraining dataset'e otomatik olarak dahil edilmemelidir.

---

# 48. PII Kontrolü

Sistem aşağıdaki veri türlerini işaretleyebilmelidir:

- Ad soyad
- Telefon
- E-posta
- T.C. kimlik numarası
- Adres
- Öğrenci numarası
- Kuruma özel kayıt
- Hassas içerik

Kullanıcı onayı olmadan training export'a dahil edilmemelidir.

---

# 49. Lisans Takibi

Her veri kaynağında:

```text
license
copyright_status
training_allowed
commercial_use
attribution_required
```

alanları bulunmalıdır.

Bu sayede eğitim dataset'i otomatik filtrelenebilir.

---

# 50. Uygulamanın Teknik Mimarisi

Önerilen mimari:

```text
Frontend
   │
   │ REST / WebSocket
   ↓
Backend API
   │
   ├── Dataset Engine
   ├── Learning Guidance Engine
   ├── Training Engine
   ├── Model Engine
   ├── RAG Engine
   └── Experiment Engine
   │
   ├──────────────→ Metadata Database
   │
   ↓
Job / Worker Layer
   │
   ├── ingestion jobs
   ├── extraction jobs
   ├── dataset jobs
   ├── tokenizer jobs
   └── training jobs
   │
   ↓
Compute Layer
   │
   ├── CPU
   ├── CUDA GPU
   └── Multi-GPU
```

## 50.1. Job / Worker mimarisi

Kısa API çağrıları ile uzun süren işlemler birbirinden ayrılmalıdır.

Örneğin:

```text
GET /projects
```

kısa süren bir API işlemidir.

Buna karşılık:

```text
1000 PDF import
Tokenizer training
Model training
Embedding generation
```

uzun süreli işlerdir ve HTTP isteğinin içinde bloklayıcı biçimde çalıştırılmamalıdır.

İlk sürüm:

```text
FastAPI
+
SQLite Job Table
+
Python Worker
```

ile başlayabilir.

İleri sürüm:

```text
Redis
+
Celery / Dramatiq / RQ
```

veya benzeri bir queue sistemi kullanabilir.

Job durumları:

```text
PENDING
RUNNING
PAUSED
COMPLETED
FAILED
CANCELLED
```

olmalıdır.

## 50.2. Depolama sorumluluklarının ayrılması

```text
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

Dataset'in tamamı relational database içine gömülmemelidir.

## 50.3. Data Lineage

Sistem bir çıktının hangi kaynaktan üretildiğini takip edebilmelidir.

Örnek:

```text
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

Model eğitimi:

```text
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

Böylece:

> Bu model hangi veri, tokenizer, kod ve ayarlarla üretildi?

sorusu cevaplanabilir.

## 50.4. Immutable Dataset Versioning

Yayınlanmış dataset sürümü sessizce değiştirilmemelidir.

```text
dataset-v1.0.0
```

değiştirilmek yerine:

```text
dataset-v1.1.0
```

oluşturulmalıdır.

Her training run belirli bir immutable dataset sürümüne bağlanır.

## 50.5. Schema Evolution

Canonical Dataset şeması zamanla değişebilir.

Bu nedenle her kayıt ve dataset için:

```text
schema_version
parser_name
parser_version
normalizer_version
```

alanları tutulmalıdır.

Eski datasetlerin yeni şemaya nasıl migrate edileceği tanımlanmalıdır.

## 50.6. Deletion Propagation

Bir kaynak dosya kaldırıldığında yalnızca raw dosyanın silinmesi yeterli değildir.

Sistem ilgili türevleri izleyebilmelidir:

```text
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

Silme veya kullanım dışı bırakma işlemi bu bağımlılık ağını dikkate almalıdır.

Model ağırlıklarına daha önce dahil olmuş bir verinin durumu ayrıca kayıt altına alınmalıdır; RAG indeksinden silmek ile eğitilmiş modelden bilgiyi geri almak aynı işlem değildir.

---

# 51. Frontend

Öneri:

```text
React
+
Next.js
+
TypeScript
```

Ana ekranlar:

- Dashboard
- Guided Journey
- Lab & Simulator
- Workspace
- Projects
- Dataset Explorer
- Format Explorer
- Learning Path / Knowledge Map
- Glossary
- Math Lab
- Systems for AI
- Tokenizer Lab
- Tensor Lab
- Attention Lab
- Transformer Builder
- Architecture Atlas
- Training Dashboard
- Post-Training
- Evaluation
- Inference & Serving
- RAG / Applications
- Distributed AI
- MLOps / Deployment
- Security
- Models
- Experiments
- Settings

---

# 52. Backend

Öneri:

```text
Python
+
FastAPI
```

Neden Python:

- PyTorch ekosistemi
- Hugging Face
- PyArrow
- DuckDB
- CUDA
- veri bilimi araçları

Neden FastAPI:

- hızlı API geliştirme
- WebSocket desteği
- Python AI servisleriyle doğal entegrasyon

---

# 53. AI / ML Katmanı

Temel:

```text
PyTorch
```

Ek araçlar:

```text
Transformers
Datasets
Tokenizers
Accelerate
PEFT
TRL
PyArrow
DuckDB
FAISS / Qdrant
```

İleri aşamada:

```text
FSDP
DeepSpeed
Megatron Core
```

---

# 54. Execution Targets — Local, Remote ve Cluster

Platform **local-first** olacaktır; ancak çalışma motoru execution target kavramıyla tasarlanacaktır.

Destek yolu:

```text
Local CPU
   ↓
Local GPU
   ↓
Remote GPU
   ↓
Multi-GPU Node
   ↓
Cluster
```

İlk sürümde:

```text
PyTorch
+
CPU / Local CUDA GPU
```

yeterlidir.

İleri sürümlerde:

- SSH ile remote machine
- remote GPU server
- Slurm cluster
- Kubernetes tabanlı workload
- cloud VM / GPU instance
- multi-node training

gibi çalışma hedefleri eklenebilir.

Runtime / inference engine örnekleri:

- Transformers
- llama.cpp
- vLLM
- MLX
- TensorRT-LLM

Uygulama aynı deneyin farklı execution target'larda hangi katmanlarının değiştiğini gösterebilmelidir.

---

# 55. Data & Model Format Explorer

Dataset, medya, model ağırlığı ve deployment formatları birbirinden ayrılmalıdır.

## 55.1. Metin / yapılandırılmış veri

```text
TXT
CSV
TSV
JSON
JSONL
XML
YAML
Parquet
Arrow
binary shards
```

## 55.2. Doküman

```text
PDF
DOCX
PPTX
HTML
```

## 55.3. Görsel

```text
JPEG
PNG
WebP
TIFF
```

## 55.4. Ses / video

```text
WAV
FLAC
MP3
MP4
WebM
```

## 55.5. Büyük dataset taşıma / shard formatları

```text
Parquet shards
Arrow
WebDataset TAR
tokenized binary shards
```

## 55.6. Model / weight formatları

```text
PyTorch checkpoint
safetensors
GGUF
ONNX
runtime-specific artifacts
```

`GGUF` dataset formatı değildir; modelin belirli runtime'larda taşınması / çalıştırılması için kullanılan bir model formatıdır.

Format Explorer ekranında her format için:

- nedir,
- text mi binary mi,
- nasıl saklanır,
- compression desteği,
- streaming yapılabilir mi,
- random access desteği,
- avantajları,
- dezavantajları,
- AI lifecycle içinde nerede kullanıldığı,
- Python ile nasıl okunup yazıldığı,
- hangi formatlara dönüştürülebildiği

gösterilmelidir.

---

# 56. Eğitim / AI Journey Aşamaları

Projenin ana öğrenme yolculuğu basitten zora şu sırayla ilerlemelidir.

## Aşama 0 — Temel Matematik ve Yazılım Ön Bilgileri

İhtiyaç anında açılacak mikro dersler:

```text
Python
Functions
Data Structures
Linear Algebra
Calculus
Probability
Optimization
Computer Systems
```

## Aşama 1 — Veri

```text
TXT
PDF
DOCX
XLSX
CSV
JSON
Image
Audio
Video
```

## Aşama 2 — Veri Formatları ve Dataset

```text
Normalization
Cleaning
Deduplication
Metadata
Data Lineage
Parquet / Arrow / JSONL
Train / Validation / Test
```

## Aşama 3 — Tokenizer

```text
Character
Word
BPE
Byte-Level BPE
WordPiece
Unigram
```

## Aşama 4 — Tensor ve Compute

```text
Scalar
Vector
Matrix
Tensor
Batch
dtype
CPU
GPU
```

## Aşama 5 — Neural Network

```text
Linear
Activation
Loss
Gradient
Optimizer
```

## Aşama 6 — Embedding

```text
Token → Vector
Similarity
Representation
```

## Aşama 7 — Attention

```text
Q
K
V
Softmax
Mask
Scaled Dot-Product Attention
```

## Aşama 8 — Transformer

```text
MHA
FFN
Residual
Normalization
Positional Information
```

## Aşama 9 — Mini GPT / Language Model

```text
Causal Language Modeling
Next Token Prediction
Generation
```

## Aşama 10 — Pretraining

```text
Dataset
Tokenizer
Dataloader
Forward
Loss
Backward
Optimizer
Checkpoint
```

## Aşama 11 — Architecture Atlas

Ana yol Transformer'dır; diğer mimariler isteğe bağlı dallar olarak incelenir:

```text
MLP
CNN
RNN / LSTM / GRU
Transformer
ViT
Autoencoder / VAE
GAN
Diffusion
GNN
MoE
State Space Models
```

## Aşama 12 — Post-Training

```text
Continued Pretraining
Instruction Tuning / SFT
LoRA / QLoRA
Preference Data
DPO / Preference Optimization
Distillation
Pruning
Quantization
```

## Aşama 13 — Evaluation

```text
Loss
Perplexity
Task Metrics
Human Evaluation
Benchmarking
Safety Evaluation
```

## Aşama 14 — Inference

```text
Prefill
KV Cache
Decode
Sampling
Streaming
```

## Aşama 15 — Serving

```text
API
Batching
Continuous Batching
Latency
Throughput
Scaling
```

## Aşama 16 — RAG / Tools / Agents

```text
Retrieval
Structured Output
Tool Calling
Workflow
Agent
```

## Aşama 17 — Multimodal

```text
Text
Image
Audio
Video
```

## Aşama 18 — Systems for AI

```text
RAM
VRAM
CPU
GPU
PCIe
Disk I/O
CUDA
Kernel
Memory Bandwidth
```

## Aşama 19 — Distributed AI

```text
Single GPU
DDP
FSDP
Tensor Parallel
Pipeline Parallel
Context Parallel
Expert Parallel
Multi-Node
```

## Aşama 20 — Production / MLOps

```text
Dataset Version
Training Run
Model Version
Deployment
Monitoring
Feedback
Retraining
Rollback
```

## Aşama 21 — AI Security

```text
PII
Secrets
Prompt Injection
RAG Injection
Data Poisoning
Training Data Leakage
Unsafe Tool Calls
```

---
# 57. İlk MVP

İlk sürüm özellikle küçük tutulmalıdır.

## MVP kapsamı

### Veri

- TXT
- MD
- PDF
- CSV
- XLSX

### Dataset

- Parquet
- JSONL
- train/validation/test split
- metadata
- SHA-256

### Öğrenme

- Tokenizer
- Tensor
- Embedding
- Attention
- Transformer Block

### Model

- 10M–30M parametre Mini-GPT

### Eğitim

- CPU
- tek GPU
- checkpoint
- live loss

### Inference

- Text generation

MVP'nin temel başarı kriteri:

```text
TXT / PDF
   ↓
Dataset
   ↓
Tokenizer
   ↓
Mini GPT
   ↓
Training
   ↓
Checkpoint
   ↓
Text Generation
```

zincirinin eksiksiz çalışmasıdır.

---

# 58. MVP Sonrası Sürüm 2

Eklenebilir:

- DOCX
- gelişmiş PDF parsing
- duplicate detection
- quality scoring
- RAG
- FAISS
- evaluation framework
- model comparison
- LoRA
- Hugging Face model import

---

# 59. Sürüm 3

Eklenebilir:

- görsel dataset
- VLM
- multimodal SFT
- image-text embedding
- Qdrant
- advanced experiment tracking
- dataset versioning

---

# 60. Sürüm 4

Eklenebilir:

- audio
- video
- distributed training
- multi-GPU
- FSDP
- DeepSpeed
- remote GPU node

---

# 61. Sürüm 5

Hedef:

```text
AI Systems Research & Learning Platform
```

Özellikler:

- büyük dataset
- remote / cluster execution
- kendi tokenizer
- 1B+ model deneyleri
- Architecture Atlas
- pretraining + post-training
- multimodal
- model registry
- dataset registry
- RAG / tools / agents
- inference server
- serving metrics
- MLOps / deployment
- monitoring
- AI security
- simulation labs

Bu aşamada platform local-first özelliğini korur; ancak artık yalnızca yerel model geliştirme aracı değildir.

---

# 62. Repository Yapısı

Önerilen repository:

```text
local-ai-research-lab/
│
├── frontend/
│
├── backend/
│
├── src/
│   ├── ingestion/
│   ├── normalization/
│   ├── cleaning/
│   ├── deduplication/
│   ├── quality/
│   ├── pii/
│   ├── dataset/
│   ├── tokenizer/
│   ├── tensor/
│   ├── embeddings/
│   ├── attention/
│   ├── transformer/
│   ├── model/
│   ├── training/
│   ├── post_training/
│   ├── evaluation/
│   ├── inference/
│   ├── serving/
│   ├── rag/
│   ├── applications/
│   ├── distributed/
│   ├── systems/
│   ├── simulators/
│   ├── security/
│   ├── mlops/
│   ├── finetuning/
│   └── multimodal/
│
├── configs/
│
├── datasets/
│
├── tokenizers/
│
├── models/
│
├── checkpoints/
│
├── experiments/
│
├── evaluations/
│
├── notebooks/
│
├── tests/
│
├── docs/
│
└── scripts/
```

---

# 63. Backend Modül Yapısı

```text
src/
├── data/
│   ├── registry.py
│   ├── manifest.py
│   └── schemas.py
│
├── ingestion/
│   ├── base.py
│   ├── text.py
│   ├── pdf.py
│   ├── docx.py
│   ├── spreadsheet.py
│   └── image.py
│
├── tokenizer/
│   ├── character.py
│   ├── word.py
│   ├── bpe.py
│   └── trainer.py
│
├── model/
│   ├── embedding.py
│   ├── attention.py
│   ├── feedforward.py
│   ├── transformer.py
│   └── gpt.py
│
├── training/
│   ├── dataloader.py
│   ├── trainer.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── checkpoint.py
│
├── inference/
│   ├── generation.py
│   ├── kv_cache.py
│   └── sampling.py
│
├── serving/
│   ├── server.py
│   ├── batching.py
│   └── metrics.py
│
├── systems/
│   ├── hardware.py
│   ├── memory.py
│   └── runtime.py
│
├── simulators/
│   ├── memory.py
│   ├── attention.py
│   ├── quantization.py
│   └── distributed.py
│
└── security/
    ├── data_checks.py
    ├── prompt_checks.py
    └── secrets.py
```

---

# 64. Konfigürasyon Sistemi

Model ve eğitim ayarları kod içine gömülmemelidir.

Örnek:

```yaml
model:
  vocab_size: 8000
  context_length: 512
  d_model: 256
  layers: 6
  heads: 8
  d_ff: 1024
  dropout: 0.1

training:
  batch_size: 16
  learning_rate: 0.0003
  steps: 10000
  optimizer: adamw
  scheduler: cosine
  warmup_steps: 500
```

Bu yapı deneylerin tekrar edilebilir olmasını sağlar.

---

# 65. Kullanıcı Arayüzü Ana Sayfası

Dashboard örneği:

```text
LOCAL AI RESEARCH LAB

Projects           6
Datasets           12
Models             8
Experiments        47
Training Runs      23

GPU
RTX xxxx

VRAM
8.2 / 12 GB

Last Training
MiniGPT-30M

Loss
2.91
```

---

# 66. Dataset Explorer

Kullanıcı:

- dosya listesi
- metadata
- text preview
- table preview
- image preview
- duplicate status
- quality score
- security
- license

görebilmelidir.

Filtreleme:

```text
language = tr
quality_score > 0.9
pii = false
training_allowed = true
```

---

# 67. Attention Explorer

Kullanıcı bir cümle girer:

```text
Öğrenciler okulda yapay zekâ öğreniyor.
```

Uygulama:

- tokenlar
- Q/K/V
- attention score
- softmax
- her head
- heatmap

gösterir.

Bu ekran uygulamanın en öğretici alanlarından biri olacaktır.

Attention Explorer ekranında ayrıca şu sabit eğitim bileşenleri bulunmalıdır:

```text
[Ne Yapıyoruz?]
[Neden?]
[Ön Bilgi]
[Formül]
[Sembol Tablosu]
[Shape Adımları]
[Kod Karşılığı]
[Gerçek Örnek]
[Sık Hata]
[İpucu]
[Kontrol Sorusu]
[Sonraki Adım]
```

Aynı tasarım dili Tokenizer, Tensor, Embedding, Neural Network ve Transformer ekranlarında da kullanılmalıdır.

---

# 68. Training Dashboard

Canlı veri:

```text
RUN ID
MODEL
DATASET
TOKENIZER

STEP
LOSS
VAL LOSS
LR

GPU
VRAM
CPU
RAM
TOKENS/SEC
```

WebSocket üzerinden canlı güncelleme yapılabilir.

---

# 69. Compute ve Donanım Stratejisi

Öğrenme için başlangıç:

```text
CPU
veya
1 Local GPU
```

İlk model hedefi:

```text
10M–100M
```

Daha sonra:

```text
300M
1B
```

deneylerine geçilebilir.

Execution target ölçeklenmesi:

```text
Local CPU
   ↓
Local GPU
   ↓
Remote GPU
   ↓
Multi-GPU
   ↓
Multi-Node / Cluster
```

Platform, bir işlemi çalıştırmadan önce mümkün olduğunda:

- tahmini RAM,
- tahmini VRAM,
- precision etkisi,
- batch size etkisi,
- sequence length etkisi,
- model parameter memory,
- optimizer state memory

gibi bilgileri açıklamalıdır.

---

# 70. Sıfırdan Model Eğitimi Stratejisi

İlk hedef hiçbir zaman doğrudan 7B olmamalıdır.

Önerilen sıra:

```text
10M
 ↓
30M
 ↓
50M
 ↓
100M
 ↓
300M
 ↓
1B
```

Her ölçek artışında:

- dataset
- tokenizer
- memory
- loss
- throughput
- evaluation

yeniden analiz edilmelidir.

---

# 71. Pretraining Dataset

Sıfırdan model eğitimi için veri:

```json
{"text": "..."}
```

şeklinde normalize edilmelidir.

Daha büyük sistemlerde:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Binary Shards
```

yaklaşımına geçilebilir.

---

# 72. Türkçe Odaklı Tokenizer Araştırması

Uygulama Türkçe için ayrı analiz yapabilmelidir.

Örneğin:

```text
öğretmenlerimizin
```

kelimesinin farklı tokenizerlarda kaç token olduğunu karşılaştırabilir.

Metric:

```text
tokens_per_word
tokens_per_character
unknown_rate
compression_ratio
```

Bu çalışma ileride Türkçe ağırlıklı model geliştirilmesinde değerli olacaktır.

---

# 73. Eğitim Alanına Özel Model Geliştirme

İleride dataset aşağıdaki alanlardan oluşturulabilir:

- eğitim dokümanları
- açık mevzuat
- ders materyalleri
- açık akademik içerikler
- kamuya açık eğitim raporları
- izinli kurumsal belgeler
- proje dokümanları
- açık istatistikler

Amaç:

```text
General LLM
    +
Education Domain Data
    ↓
Education Domain Model
```

---

# 74. RAG ve Fine-Tuning Karar Sistemi

Uygulama kullanıcıya öneri verebilir.

Örneğin:

### Veri sık değişiyorsa

```text
RAG kullan
```

### Model belirli bir üslup öğrenmeli ise

```text
SFT / LoRA kullan
```

### Model temel dil yeteneğini sıfırdan öğrenmeli ise

```text
Pretraining kullan
```

### Tablo analizi ise

```text
SQL / DuckDB kullan
```

Bu karar modülü eğitim amacıyla açıklama da sunabilir.

---

# 75. Eğitim Modu İçerik Yapısı

Eğitim Modu'nun amacı yalnızca açıklama göstermek değil, kullanıcının **işlemi kavramsal, matematiksel, yazılımsal ve pratik olarak birbirine bağlayabilmesini** sağlamaktır.

Her önemli modül ve işlem mümkün olduğunca aşağıdaki ortak şablonu kullanmalıdır.

## 75.1. Standart öğrenme kartı

### 1. Bu adımda ne yapıyoruz?

İşlemin kısa, teknik olmayan açıklaması.

### 2. Neden yapıyoruz?

Bu işlemin model zincirindeki rolü.

### 3. Önce ne bilmelisin?

Ön bilgi listesi:

```text
✓ Gerekli ve tamamlanmış
○ Gerekli ancak henüz görülmemiş
△ Faydalı ancak zorunlu değil
```

### 4. Girdi nedir?

- veri tipi
- örnek değer
- shape
- dtype
- device

### 5. Çıktı nedir?

Aynı ayrıntılar çıktı için de gösterilir.

### 6. Matematiksel açıklama

Formül, semboller, shape ve sayısal örnek.

### 7. Sezgisel açıklama

Formülün günlük dilde ne yaptığı.

### 8. Kod karşılığı

Önce sade/öğretici uygulama, ardından optimize/profesyonel uygulama.

### 9. Yazılım mimarisindeki yeri

Örneğin:

```text
src/model/attention.py
        ↓
TransformerBlock.forward()
        ↓
GPTModel.forward()
```

### 10. Gerçek veri üzerinde gösterim

Kullanıcının kendi dataset'inden küçük bir örnekle çalışma.

### 11. Alternatifler

Örneğin:

```text
BPE
WordPiece
Unigram
```

gibi alternatif yöntemlerin neden var olduğu açıklanır.

### 12. Ne yanlış gidebilir?

- yaygın hata
- yanlış shape
- veri sızıntısı
- numerical instability
- GPU OOM
- yanlış parametre seçimi

### 13. İpucu

Kısa pratik öneri.

### 14. Tarihçe / Anekdot

Yalnız uygun ve doğrulanabilir olduğunda.

### 15. Kontrol sorusu

Kavramın anlaşılıp anlaşılmadığını test eden kısa soru.

### 16. Bu adım bir sonrakine nasıl bağlanıyor?

Kullanıcı neden bir sonraki konuya geçtiğini bilmelidir.

---

## 75.2. Ön bilgi uyarı kartı

Kullanıcı ileri bir adıma erken geçtiğinde:

```text
Bu konu için önerilen ön bilgiler eksik.

Gerekli:
○ Matrix Multiplication
○ Softmax

Faydalı:
△ Probability Distribution

[Konuyu Aç]
[Hızlı Açıklama]
[Yine de Devam Et]
```

Uygulama kullanıcıyı gereksiz biçimde kilitlememeli; fakat eksik altyapıyı görünür kılmalıdır.

---

## 75.3. Bağlam içi mikro ders

Örneğin Attention formülünde `sqrt(d_k)` görüldüğünde kullanıcı bunu anlamıyorsa ana dersten çıkmadan küçük bir panel açılabilir:

```text
Neden √d_k ile bölüyoruz?
```

Panel:

- scaling kavramını
- dot product büyüklüğünü
- softmax üzerindeki etkisini
- küçük sayısal örneği

açıklar.

Panel kapatıldığında kullanıcı tam olarak kaldığı adıma geri döner.

---

## 75.4. Matematik gösterim biçimi

Örneğin Cross Entropy:

```math
L = - \sum_i y_i \log(p_i)
```

tek başına bırakılmaz.

Uygulama aynı formülün altında:

```text
y_i = gerçek hedef
p_i = modelin tahmin olasılığı
log = doğal logaritma
L   = loss
```

açıklamasını verir.

Ardından gerçek sayılarla hesap yapılır.

Son olarak kod:

```python
loss = torch.nn.functional.cross_entropy(logits, targets)
```

gösterilir.

---

## 75.5. Calculus öğrenme noktaları

Backpropagation'a geçildiğinde aşağıdaki kavramların durumu kontrol edilir:

```text
Function
Derivative
Partial Derivative
Chain Rule
Gradient
```

Örneğin `Chain Rule` eksikse:

> Backpropagation'ın temelinde Chain Rule bulunur. Bu kavramı bilmeden kodu çalıştırabilirsin, ancak gradient'in katmanlar boyunca nasıl taşındığını tam olarak anlamak zorlaşır.

uyarısı gösterilebilir.

Ardından kullanıcı doğrudan ilgili mikro derse gidebilir.

---

## 75.6. Linear Algebra öğrenme noktaları

Attention'a geçmeden önce:

```text
Vector
Matrix
Transpose
Dot Product
Matrix Multiplication
Shape
```

konuları kontrol edilir.

Bu yapı özellikle `QKᵀ` hesabını mekanik ezber olmaktan çıkarır.

---

## 75.7. Kod yürütme görünümü

Kullanıcı bir fonksiyonu çalıştırdığında isterse aşama aşama izleyebilmelidir.

Örneğin:

```python
scores = q @ k.transpose(-2, -1)
```

için:

```text
q.shape               = [1, 4, 8, 32]
k.shape               = [1, 4, 8, 32]
k.transpose().shape   = [1, 4, 32, 8]
scores.shape          = [1, 4, 8, 8]
```

gösterilir.

Bu yaklaşım matematik, tensor shape ve kod arasındaki ilişkiyi doğrudan görünür hale getirir.

---

## 75.8. Soyutlama seviyeleri

Kullanıcı aynı işlemi üç seviyede inceleyebilir:

### Seviye A — Sezgisel

> Query, bir tokenın ne aradığını temsil eder.

### Seviye B — Matematiksel

```math
Q = XW_Q
```

### Seviye C — Kod / Sistem

```python
q = self.q_proj(x)
```

İleri kullanıcılar doğrudan C seviyesine geçebilir; yeni başlayanlar A → B → C sırasını takip edebilir.

---

## 75.9. Öğrenme ilerleme kaydı

Uygulama mümkün olduğunda şu durumları kaydedebilir:

```text
NOT_SEEN
INTRODUCED
PRACTICED
UNDERSTOOD
NEEDS_REVIEW
```

Bu sistem sertifika veya puan odaklı olmak zorunda değildir. Amaç, kullanıcının kendi kavramsal boşluklarını görmesidir.

---

## 75.10. Motivasyon ve keşif öğeleri

Uygulamada ölçülü biçimde:

- “Neden önemli?” kartları
- pratik ipuçları
- gerçek hata örnekleri
- kısa tarihçe
- “Bunu gerçek modeller nerede kullanıyor?” açıklamaları
- modül sonunda öğrenilenler özeti
- isteğe bağlı mini deneyler

bulunabilir.

Örneğin:

> **Mini Deney:** `d_k` büyüdükçe scaling uygulamadan softmax dağılımının nasıl değiştiğini gözlemle.

Bu tür küçük deneyler, teoriyi pasif okumaktan aktif keşfe dönüştürür.

---

## 75.11. Pedagojik içerik veri modeli

Açıklamaların frontend içine sabit metin olarak dağılması yerine yapılandırılmış olarak tutulması önerilir.

Örnek:

```yaml
concept_id: self_attention
title: Self-Attention
language: tr

standard_term: Self-Attention

prerequisites:
  required:
    - matrix_multiplication
    - dot_product
    - softmax
  recommended:
    - probability_distribution

sections:
  what: ...
  why: ...
  intuition: ...
  math: ...
  code: ...
  pitfalls: ...
  next_step: multi_head_attention

tips:
  - ...

quiz:
  - ...
```

Böylece eğitim içeriği:

- versiyonlanabilir
- test edilebilir
- yeniden kullanılabilir
- ileride farklı dillere çevrilebilir
- aynı kavram farklı ekranlarda tutarlı biçimde gösterilebilir

hale gelir.

---

## 75.12. Öğrenme içeriği ile çalışma motorunun ayrılması

Pedagojik içerik ile gerçek hesaplama motoru birbirine bağlı ama ayrı katmanlar olmalıdır.

```text
Learning Content
       │
       ├── açıklama
       ├── prerequisite
       ├── formül
       ├── ipucu
       └── quiz
       │
       ↓
Interactive Lab
       │
       ↓
Real Compute Engine
       │
       ├── NumPy / PyTorch
       └── CPU / GPU
```

Bu ayrım sayesinde eğitim metni değiştirildiğinde model motorunun kodu etkilenmez; hesaplama motoru optimize edildiğinde de öğrenme içeriği kaybolmaz.

---

# 76. Kod Karşılaştırma Modu

Örneğin Attention:

## Eğitim kodu

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
output = weights @ v
```

## Profesyonel kod

```python
torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

Amaç:

> Hazır fonksiyonun arka planda ne yaptığını göstermek.

---

# 77. Notebook Entegrasyonu

Uygulama yanında Jupyter notebook örnekleri bulunabilir.

```text
notebooks/
├── 01_dataset.ipynb
├── 02_tokenizer.ipynb
├── 03_tensor.ipynb
├── 04_embedding.ipynb
├── 05_attention.ipynb
├── 06_transformer.ipynb
└── 07_train_gpt.ipynb
```

GUI ile notebook arasında aynı backend modülleri kullanılmalıdır.

---

# 78. Test Stratejisi

Testler:

## Unit Test

- tokenizer
- attention
- dataset parser
- split
- hash

## Integration Test

```text
PDF → Dataset → Tokenizer → Model
```

## Training Test

Küçük dataset ile birkaç step eğitim.

## Regression Test

Yeni değişiklik model sonuçlarını bozuyor mu?

---

# 79. Logging

Her işlem loglanmalıdır.

```text
logs/
├── application.log
├── dataset.log
├── training.log
└── errors.log
```

---

# 80. Hata Yönetimi

Özellikle:

- GPU OOM
- bozuk PDF
- bozuk XLSX
- encoding hatası
- CUDA hatası
- checkpoint bozulması
- disk dolması

uygulamada anlaşılır hata mesajlarıyla gösterilmelidir.

---

# 81. Gelecekte Plugin Sistemi

Yeni parser veya model desteği plugin olarak eklenebilir.

```text
plugins/
├── parsers/
├── models/
├── trainers/
└── exporters/
```

Bu sayede sistem modüler kalır.

---

# 82. CLI Desteği

GUI yanında komut satırı da olmalıdır.

Örnek:

```bash
ailab dataset import ./data
ailab tokenizer train dataset-v1
ailab model create config.yaml
ailab train run experiment.yaml
ailab evaluate model-001
```

Bu özellik otomasyon ve ileri kullanıcılar için önemlidir.

---

# 83. REST API

İleride diğer uygulamalar sistemi kullanabilmelidir.

Örnek:

```text
POST /datasets
POST /tokenizers
POST /models
POST /training-runs
GET  /experiments
POST /inference
```

---

# 84. WebSocket

Canlı training verisi:

```text
loss
GPU
VRAM
step
tokens/sec
```

WebSocket üzerinden frontend'e gönderilebilir.

---

# 85. Veri Depolama

İlk sürüm:

```text
Local filesystem
+
SQLite
+
Parquet
```

Daha sonra:

```text
PostgreSQL
Object Storage
```

eklenebilir.

---

# 86. Metadata Database

SQLite/PostgreSQL içinde:

```text
projects
datasets
files
models
experiments
training_runs
checkpoints
evaluations
```

tabloları tutulabilir.

Gerçek dataset içeriği ise Parquet ve dosya sistemi üzerinde bulunabilir.

---

# 87. Büyük Dataset Stratejisi

Dataset büyüdüğünde:

```text
Parquet shards
WebDataset shards
```

kullanılacaktır.

Örneğin:

```text
train-00001.parquet
train-00002.parquet
train-00003.parquet
```

---

# 88. Bellek Dostu Veri Okuma

Dataset RAM'e tamamen yüklenmemelidir.

Desteklenecek yöntemler:

- streaming
- memory mapping
- batch loading
- lazy loading

---

# 89. GPU Bellek Yönetimi

Training Lab ileride aşağıdaki seçenekleri sunabilir:

- gradient accumulation
- gradient checkpointing
- mixed precision
- bf16
- fp16
- Flash Attention
- quantization

---

# 90. Distributed AI ve Büyük Model Altyapısı

İleri aşamada dağıtık hesaplama yalnız bir araç olarak değil, ayrı bir öğrenme yolu olarak ele alınmalıdır.

Önerilen sıra:

```text
Single GPU
 ↓
Data Parallel
 ↓
Distributed Data Parallel (DDP)
 ↓
FSDP / Parameter Sharding
 ↓
Tensor Parallel
 ↓
Pipeline Parallel
 ↓
Context Parallel
 ↓
Expert Parallel
 ↓
Multi-Node Training
```

Framework / altyapı örnekleri:

- PyTorch Distributed
- FSDP
- DeepSpeed
- Megatron Core

Her yöntemde şu sorular cevaplanmalıdır:

- hangi problemi çözüyor?
- model mi, veri mi, optimizer state mi parçalanıyor?
- GPU'lar arasında ne haberleşiyor?
- communication overhead nedir?
- all-reduce nedir?
- hangi durumda tek GPU daha mantıklıdır?
- hangi durumda model tek GPU'ya sığmaz?

---

# 91. Öğrenme Başarı Kriterleri

Kullanıcı aşağıdaki işlemleri kendi başına yapabiliyorsa ilk eğitim hedefi tamamlanmış kabul edilebilir:

- Dataset oluşturmak
- Tokenizer eğitmek
- Token ID mantığını açıklamak
- Embedding hesaplamak
- Q/K/V üretmek
- Attention hesaplamak
- Multi-head attention kurmak
- Transformer block yazmak
- Mini GPT oluşturmak
- Loss hesaplamak
- Backpropagation açıklamak
- Model eğitmek
- Checkpoint yüklemek
- Inference yapmak

---

# 92. Teknik Başarı Kriterleri

MVP aşağıdaki işlemleri gerçekleştirebilmelidir:

```text
PDF/TXT
 ↓
Extract
 ↓
Normalize
 ↓
Parquet
 ↓
BPE Tokenizer
 ↓
Token Dataset
 ↓
Mini GPT
 ↓
Training
 ↓
Checkpoint
 ↓
Inference
```

---

# 93. Uzun Vadeli Vizyon

Projenin nihai hedefi:

> Verinin oluşmasından modelin üretim ortamında çalışmasına kadar modern AI sistemlerinin tüm yaşam döngüsünü; matematik, algoritma, yazılım, veri formatı, donanım, dağıtık sistem ve uygulama katmanlarıyla birlikte uygulayarak öğrenmeyi sağlayan açık ve modüler bir **AI Systems Research & Learning Platform** oluşturmaktır.

Platformun başlangıç noktası yerel bilgisayardır; nihai mimari execution target'tan bağımsızdır.

```text
                      AI SYSTEMS RESEARCH & LEARNING LAB
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
   GUIDED JOURNEY               LAB & SIMULATOR                 WORKSPACE
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
      DATA                        MODELING                      SYSTEMS
        │                             │                             │
 Formats / Dataset          Architecture / Training      CPU / GPU / Distributed
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                AI LIFECYCLE
                                      │
                   Pretraining → Post-Training → Evaluation
                                      │
                              Inference → Serving
                                      │
                          Applications → Production
```

Ana tasarım ilkesi:

> **Hiçbir kritik AI kavramı kara kutu olarak kalmamalıdır.**

Kullanıcı gerektiğinde:

```text
iş akışı
 ↓
algoritma
 ↓
matematik
 ↓
tensor
 ↓
kod
 ↓
runtime
 ↓
donanım
```

katmanları arasında aşağı ve yukarı hareket edebilmelidir.

---

# 94. Önerilen Geliştirme Sırası

## Faz 0 — Proje Temeli

- Repository
- Python environment
- FastAPI
- React / Next.js / TypeScript
- SQLite
- config sistemi
- logging
- temel job/worker altyapısı
- metadata / storage sorumluluklarının ayrılması
- pedagojik UI bileşen standardı
- Türkçe teknik terim sözlüğü altyapısı
- prerequisite / knowledge graph veri modeli
- LaTeX formül render altyapısı
- öğrenme içeriği için YAML/JSON şeması

## Faz 1 — Data Lab

- TXT
- MD
- PDF
- CSV
- XLSX
- manifest
- SHA-256
- Parquet
- parser_name / parser_version / schema_version
- temel data lineage
- immutable dataset version
- her ingestion adımı için “ne / neden / girdi / çıktı” açıklaması

## Faz 2 — Dataset Engine

- cleaning
- deduplication
- split
- registry
- dataset version

## Faz 3 — Tokenizer Lab

- character
- word
- BPE
- visualization
- tokenizer prerequisite kartları
- vocabulary / merge adımlarının açıklaması
- algoritmanın adım adım çalıştırılması
- Türkçe örnekler
- ipucu / sık hata / mini deney

## Faz 4 — Tensor ve Neural Network Lab

- tensors
- matrix multiplication
- linear
- loss
- gradient
- Linear Algebra mikro dersleri
- Calculus mikro dersleri
- shape debugger
- formül → sayısal örnek → PyTorch eşlemesi

## Faz 5 — Attention Lab

- Q/K/V
- attention
- mask
- heatmap
- sembol ve shape açıklamaları
- matrix multiplication prerequisite kontrolü
- softmax prerequisite kontrolü
- formülün adım adım yürütülmesi
- sade kod / optimize PyTorch karşılaştırması

## Faz 6 — Transformer Lab

- MHA
- FFN
- residual
- normalization

## Faz 7 — Mini GPT

- causal model
- next-token prediction

## Faz 8 — Training Lab

- trainer
- checkpoint
- dashboard
- GPU

## Faz 9 — Evaluation

- evaluation dataset
- model comparison

## Faz 10 — RAG

- chunking
- embedding
- vector DB

## Faz 11 — Fine-Tuning

- Hugging Face model import
- LoRA
- QLoRA
- SFT

## Faz 12 — Multimodal

- image
- vision encoder
- VLM dataset

## Faz 13 — Scale

- 300M
- 1B
- multi-GPU
- FSDP / DeepSpeed

## Faz 14 — Architecture Atlas

- MLP
- CNN
- RNN/LSTM
- ViT
- Autoencoder/VAE
- GAN
- Diffusion
- GNN
- MoE
- State Space Models

## Faz 15 — Post-Training

- continued pretraining
- SFT
- LoRA/QLoRA
- preference data
- DPO / preference optimization
- distillation
- pruning
- quantization

## Faz 16 — Inference & Serving

- prefill
- KV cache
- decode
- sampling
- streaming
- batching
- serving metrics
- inference server

## Faz 17 — Systems & Simulation

- CPU/GPU memory path
- VRAM simulator
- attention complexity simulator
- quantization simulator
- training scale simulator
- RAG simulator

## Faz 18 — Distributed AI

- DDP
- FSDP
- tensor parallel
- pipeline parallel
- context parallel
- expert parallel
- multi-node

## Faz 19 — AI Applications

- structured output
- tool calling
- workflows
- agents
- application integration

## Faz 20 — Production / MLOps

- deployment
- model registry
- monitoring
- drift
- rollback
- retraining
- A/B / canary concepts

## Faz 21 — AI Security

- prompt injection
- RAG injection
- data poisoning
- training data leakage
- secrets
- unsafe tool calls

---

# 95. İlk Geliştirme Sprintleri

## Sprint 0 — Öğrenme UX ve Mimari Temel

Hedef:

```text
Her işlem açıklanabilir ve izlenebilir olsun
```

Görevler:

- ortak `LearningCard` UI bileşeni
- `PrerequisiteWarning` bileşeni
- `FormulaExplorer` bileşeni
- `ShapeViewer` bileşeni
- `TipCard` / `PitfallCard` bileşenleri
- Glossary altyapısı
- learning content YAML/JSON şeması
- KaTeX/MathJax entegrasyonu
- job/worker temel modeli
- data lineage kimlik modeli

## Sprint 1

Hedef:

```text
Dosya → Dataset
```

Görevler:

- proje oluşturma
- TXT import
- PDF import
- file manifest
- SHA-256
- metadata
- Parquet export

## Sprint 2

Hedef:

```text
Dataset → Tokenizer
```

Görevler:

- character tokenizer
- word tokenizer
- BPE
- vocabulary ekranı
- tokenization preview

## Sprint 3

Hedef:

```text
Token → Tensor → Embedding
```

## Sprint 4

Hedef:

```text
Attention
```

## Sprint 5

Hedef:

```text
Transformer Block
```

## Sprint 6

Hedef:

```text
Mini GPT
```

## Sprint 7

Hedef:

```text
Training Dashboard
```

## Sprint 8

Hedef:

```text
Inference
```

Bu noktada ilk gerçek MVP tamamlanmış olur.

MVP sonrasında sprintler tek tek ekran eklemek yerine aşağıdaki **epic** gruplarında yürütülebilir:

```text
EPIC-A  Post-Training
EPIC-B  Inference & Serving
EPIC-C  Systems & Simulation
EPIC-D  Architecture Atlas
EPIC-E  Distributed AI
EPIC-F  AI Applications
EPIC-G  Production / MLOps
EPIC-H  AI Security
```

Bu yaklaşım ana kullanıcı deneyimini sade tutarken kapsamın kontrollü büyümesini sağlar.

---

# 96. İlk Demo Senaryosu

Önerilen demo:

### Veri

```text
Türkçe metin koleksiyonu
```

### Dataset

```text
10–100 MB temiz text
```

### Tokenizer

```text
BPE
Vocabulary: 8000
```

### Model

```text
MiniGPT
20M parametre
```

### Eğitim

```text
Next-token prediction
```

### Sonuç

Kullanıcı bir prompt yazar ve kendi eğittiği modelden çıktı alır.

Bu demo projenin hem eğitim hem teknik açıdan ilk büyük kilometre taşı olacaktır.

---

# 97. İlk Gerçek Veri Senaryosu

Uygun ve izinli belgeler kullanılarak:

```text
PDF
DOCX
XLSX
TXT
```

dataset oluşturulur.

Daha sonra:

```text
Canonical Dataset
```

üzerinden iki farklı sistem geliştirilir.

## Sistem A

```text
RAG
```

## Sistem B

```text
Mini Language Model
```

Sonuçlar karşılaştırılır.

Böylece:

> Bilgiyi modele öğretmek ile bilgiyi RAG üzerinden vermek arasındaki fark

uygulamalı olarak öğrenilir.

---

# 98. Projenin En Önemli Kazanımları

Bu proje tamamlandığında kullanıcı:

1. Yapay zekâ datasetlerinin nasıl oluşturulduğunu öğrenir.
2. Ham veri ile model-ready veri arasındaki farkı öğrenir.
3. Tokenizer mantığını öğrenir.
4. Embedding kavramını uygulamalı öğrenir.
5. Attention'ın matematiğini ve kodunu öğrenir.
6. Transformer block oluşturabilir.
7. Küçük GPT modeli yazabilir.
8. Modeli sıfırdan eğitebilir.
9. GPU eğitim sürecini öğrenir.
10. RAG kurabilir.
11. Fine-tuning yapabilir.
12. LoRA/QLoRA kullanabilir.
13. Multimodal modele geçiş altyapısını öğrenir.
14. Dataset ve model versiyonlamasını öğrenir.
15. Yapay zekâ araştırmalarını sistematik biçimde yönetebilir.
16. Aynı AI iş yükünün CPU, local GPU, remote GPU ve cluster ortamlarında nasıl değiştiğini anlayabilir.
17. Pretraining ile post-training arasındaki farkı ve modelin base modelden assistant modele dönüşümünü anlayabilir.
18. Inference sırasında prefill, KV cache, decode ve sampling süreçlerini açıklayabilir.
19. Bir modelin API/serving katmanına nasıl dönüştüğünü ve latency/throughput gibi metrikleri anlayabilir.
20. RAM, VRAM, CUDA, disk I/O ve GPU utilization gibi sistem kavramlarını AI iş yükleriyle ilişkilendirebilir.
21. Model ve dataset formatlarının neden farklı olduğunu ve hangi aşamada hangi formatın kullanıldığını anlayabilir.
22. Distributed training stratejilerinin hangi problemi çözdüğünü açıklayabilir.
23. RAG, structured output, tool calling, workflow ve agent yapılarının modelden farkını anlayabilir.
24. Deployment, monitoring, feedback ve retraining içeren production AI yaşam döngüsünü takip edebilir.
25. AI sistemlerinde veri, model, RAG ve tool katmanlarına ilişkin temel güvenlik risklerini tanıyabilir.

---

# 99. Sonuç

Bu projenin ilk hedefi büyük bir yapay zekâ modeli üretmek olmamalıdır.

İlk hedef:

```text
VERİ
 ↓
FORMAT / DATASET
 ↓
TOKENIZER
 ↓
TENSOR
 ↓
NEURAL NETWORK
 ↓
ATTENTION
 ↓
TRANSFORMER
 ↓
EĞİTİM
 ↓
MODEL
```

zincirinin her aşamasını anlamak, kodlamak ve gerçek veri üzerinde çalıştırmaktır.

Ancak nihai öğrenme yolu burada bitmez:

```text
MODEL
 ↓
POST-TRAINING
 ↓
EVALUATION
 ↓
INFERENCE
 ↓
SERVING
 ↓
AI APPLICATIONS
 ↓
DISTRIBUTED SYSTEMS
 ↓
PRODUCTION / MLOPS
```

Projenin temel felsefesi:

> **Modeli yalnızca kullanma; veriden başlayarak nasıl oluştuğunu gör, neden öyle yapıldığını anla, matematiğini izle, kodunu uygula, sistemde ve donanımda ne olduğunu gör, ölç, simüle et, karşılaştır, ölçekle ve gerçek uygulamada çalıştır.**

Uygulama hiçbir kritik kavramı yalnızca “çalışan kod” seviyesinde bırakmamalıdır. Kullanıcı gerektiğinde en üst seviye iş akışından tensor shape'lerine, formüldeki tek bir sembolden gradient hesabına, Python sınıfından CUDA/GPU işlemine, tek makineden dağıtık cluster'a, model ağırlığından inference server'a kadar inebilmelidir.

Öğrenme rotası basitten zora ilerlemeli; ileri bir konuda gereken ön bilgi eksikse bu eksik tam ihtiyaç duyulduğu yerde görünür hale getirilmeli ve öğrenilebilmelidir.

Ana ürün yüzeyi üç bölümle sade tutulmalıdır:

```text
GUIDED JOURNEY
      +
LAB & SIMULATOR
      +
WORKSPACE
```

Böylece uygulama çok geniş bir AI kapsamını desteklerken kullanıcı aynı anda gereksiz karmaşıklıkla karşılaşmaz.

Bu nedenle platformun nihai tanımı:

```text
AI Dataset & Format Engine
        +
AI Learning Guidance System
        +
Model & Architecture Lab
        +
Training / Post-Training Engine
        +
Inference & Serving Lab
        +
Systems & Distributed AI Lab
        +
Simulation Engine
        +
AI Application Platform
        +
MLOps / Security Layer
```

Bu yapı zamanla tam kapsamlı bir **AI Systems Research & Learning Lab** platformuna dönüşecektir.

---

# 100. Architecture Atlas

Architecture Atlas'ın amacı Transformer dışındaki önemli AI mimarilerini ana öğrenme yolunu bozmadan erişilebilir hale getirmektir.

Ana yol:

```text
Neural Network
 ↓
Attention
 ↓
Transformer
 ↓
GPT
```

olarak kalacaktır.

İsteğe bağlı mimari dallar:

## 100.1. MLP

Öğretilecek kavramlar:

- dense / linear layer
- activation
- hidden layer
- universal approximation fikri

## 100.2. CNN

- convolution
- kernel
- stride
- padding
- pooling
- feature map

## 100.3. RNN / LSTM / GRU

- recurrent state
- sequence processing
- vanishing gradient
- gating

## 100.4. Vision Transformer

- patch embedding
- image tokens
- positional representation
- vision attention

## 100.5. Autoencoder / VAE

- encoder
- latent space
- decoder
- reconstruction
- probabilistic latent representation

## 100.6. GAN

- generator
- discriminator
- adversarial training

## 100.7. Diffusion Models

- noise process
- denoising
- timestep
- sampling

## 100.8. GNN

- graph
- node
- edge
- message passing

## 100.9. Mixture of Experts

- experts
- router
- sparse activation
- expert parallelism

## 100.10. State Space Models

Transformer dışındaki sequence modeling yaklaşımlarını kavramsal olarak karşılaştırmak için kullanılacaktır.

Her mimari için ortak şablon:

```text
Problem
 ↓
Core Idea
 ↓
Math
 ↓
Minimal Implementation
 ↓
Real Framework Implementation
 ↓
Use Cases
 ↓
Strengths / Limitations
 ↓
Comparison
```

---

# 101. Post-Training Lab

Bir base modelin assistant / domain model haline gelmesi adım adım incelenmelidir.

```text
Base Model
 ↓
Continued Pretraining
 ↓
Instruction Tuning / SFT
 ↓
Preference Data
 ↓
Preference Optimization
 ↓
Evaluation
```

Konular:

- Continued Pretraining
- Domain-Adaptive Pretraining
- Instruction Tuning
- SFT
- LoRA
- QLoRA
- Preference datasets
- DPO ve benzeri preference optimization yaklaşımları
- Distillation
- Pruning
- Quantization

Her yöntem için:

- neyi değiştirdiği,
- model ağırlıklarına etkisi,
- gereken dataset tipi,
- hesaplama ihtiyacı,
- kullanım amacı,
- base model / adapter / merged model farkı

açıklanmalıdır.

---

# 102. Systems for AI Lab

Bu modül AI kodunun bilgisayarda gerçekten nasıl çalıştığını öğretir.

Temel veri yolu:

```text
Disk
 ↓
RAM
 ↓
CPU preprocessing
 ↓
PCIe / Unified Memory Path
 ↓
GPU VRAM
 ↓
CUDA Kernel
 ↓
Compute Units / Tensor Cores
```

Konular:

- process
- thread
- CPU core
- RAM
- cache
- disk I/O
- memory mapping
- GPU
- VRAM
- PCIe
- CUDA
- kernel
- memory bandwidth
- compute bound vs memory bound
- dtype
- FP32 / FP16 / BF16
- DataLoader workers
- pinned memory
- asynchronous transfer

Her kavram AI örneğine bağlanmalıdır.

---

# 103. Simulation Lab

Simulation Lab ileri kavramları gerçek büyük kaynaklar olmadan deneysel olarak anlamaya yardımcı olacaktır.

## 103.1. GPU / VRAM Memory Simulator

Girdiler:

```text
Parameters
Precision
Optimizer
Batch Size
Sequence Length
Layers
Hidden Size
```

Çıktılar:

- weight memory
- gradient memory
- optimizer state memory
- estimated activation memory
- toplam tahmini VRAM

Sonuç açıkça `ESTIMATION` olarak etiketlenmelidir.

## 103.2. Attention Complexity Simulator

Sequence length değiştirildiğinde:

```text
128
512
2048
8192
```

attention matrix boyutu, tahmini memory ve klasik attention karmaşıklığı gösterilir.

## 103.3. Quantization Simulator

```text
FP32
BF16
FP16
INT8
INT4
```

formatlarının teorik memory etkileri karşılaştırılır.

## 103.4. RAG Simulator

Parametreler:

- chunk size
- overlap
- top_k
- embedding model

değiştirildiğinde retrieval sonuçları karşılaştırılır.

## 103.5. Distributed Training Simulator

Bir modelin:

- data parallel,
- sharded,
- tensor parallel,
- pipeline parallel

şeklinde nasıl bölünebileceği görsel olarak gösterilir.

---

# 104. Distributed AI Lab

Distributed AI ayrı bir öğrenme rotası olmalıdır.

Karar sorusu:

```text
Model ve eğitim tek GPU'da çalışıyor mu?
      │
      ├── Evet → Tek GPU
      │
      └── Hayır / Yetersiz hız
              ↓
       Dağıtık strateji seç
```

Öğrenme sırası:

1. Data Parallel
2. Distributed Data Parallel
3. Parameter / Optimizer Sharding
4. FSDP
5. Tensor Parallel
6. Pipeline Parallel
7. Context Parallel
8. Expert Parallel
9. Multi-node orchestration

Görselleştirilecek kavramlar:

- rank
- world size
- process group
- synchronization
- all-reduce
- broadcast
- communication overhead
- network bandwidth

---

# 105. AI Application Lab

Model ile AI uygulaması aynı şey değildir.

Bu modülde:

```text
Model
 ↓
Prompt / Structured Input
 ↓
Structured Output
 ↓
Tool Calling
 ↓
External System
 ↓
Workflow
```

zinciri incelenmelidir.

Konular:

- structured output
- JSON schema
- function/tool calling
- RAG
- SQL tools
- file tools
- workflow orchestration
- agent kavramı
- memory kavramı
- deterministic workflow ile agent farkı

Ana Transformer yolundan sonra isteğe bağlı olarak açılmalıdır.

---

# 106. Production / MLOps Lab

AI sisteminin yaşam döngüsü:

```text
Dataset Version
 ↓
Training Run
 ↓
Model Version
 ↓
Evaluation
 ↓
Deployment
 ↓
Monitoring
 ↓
Feedback
 ↓
New Dataset
 ↓
Retraining
```

Konular:

- experiment tracking
- model registry
- dataset registry
- deployment version
- rollback
- monitoring
- latency monitoring
- quality monitoring
- drift
- feedback loop
- retraining
- A/B testing kavramı
- canary deployment kavramı

---

# 107. AI Security Lab

Mevcut PII ve lisans yönetiminin üzerine sistem güvenliği eklenmelidir.

Alanlar:

## Veri güvenliği

- PII
- secrets
- access control
- data poisoning
- untrusted datasets

## Model güvenliği

- training data leakage
- model extraction kavramı
- unsafe model artifacts

## RAG güvenliği

- malicious documents
- RAG injection
- source trust

## Application güvenliği

- prompt injection
- unsafe tool calls
- excessive permissions
- secret leakage

Bu modül saldırı üretmekten çok **tehdit modelini, riskleri ve güvenli tasarım ilkelerini** öğretmeye odaklanmalıdır.

---

# 108. Execution Target Abstraction

Platformun çalışma motorunda ortak bir execution tanımı bulunmalıdır.

Örnek:

```yaml
execution_target:
  type: local_cuda
  device: cuda:0
```

İleri örnek:

```yaml
execution_target:
  type: remote
  host: gpu-node-01
```

Daha ileri:

```yaml
execution_target:
  type: cluster
  scheduler: slurm
  nodes: 4
  gpus_per_node: 8
```

İlk sürüm yalnız local değerleri uygular; ancak veri modeli gelecekteki target'ları destekleyebilecek şekilde tasarlanır.

---

# 109. Tek Bir İşlemi Uçtan Uca İnceleme Standardı

Platformun en ayırt edici özelliklerinden biri, aynı işlemi farklı soyutlama seviyelerinde gösterebilmesidir.

Örneğin:

```text
"Model bir sonraki tokenı üretiyor"
```

### Seviye 1 — Kullanıcı

```text
Prompt → Cevap
```

### Seviye 2 — AI Pipeline

```text
Tokenize → Forward → Logits → Sample
```

### Seviye 3 — Model

```text
Embedding → Transformer Blocks → LM Head
```

### Seviye 4 — Matematik

```text
Matrix Multiplication
Softmax
Normalization
```

### Seviye 5 — Yazılım

```text
PyTorch ops
Modules
Functions
```

### Seviye 6 — Runtime

```text
Tensor allocation
Kernel launch
Memory transfer
```

### Seviye 7 — Donanım

```text
CPU / GPU
RAM / VRAM
```

Bu standardın amacı herhangi bir önemli AI işleminin kara kutu olarak kalmamasıdır.

---

# 110. Kapsam Yönetimi ve Mimari Dondurma

v1.2 sonrasında ana AI lifecycle mimarisi büyük ölçüde dondurulmalıdır.

Yeni bir teknoloji görüldüğünde varsayılan yaklaşım:

```text
Yeni ana menü öğesi ekle
```

olmamalıdır.

Önce şu sorular sorulmalıdır:

1. Mevcut bir Lab altında açıklanabilir mi?
2. Architecture Atlas'a bir alt dal olarak eklenebilir mi?
3. Simulator'a bir deney olarak eklenebilir mi?
4. Guided Journey'de isteğe bağlı ileri konu olabilir mi?
5. Gerçekten AI lifecycle'a yeni bir ana aşama mı ekliyor?

Yalnız son sorunun cevabı açıkça “evet” ise yeni ana modül düşünülmelidir.

Bu yaklaşım uygulamanın uzun vadede anlaşılır kalmasını sağlar.

---

# 111. v1.2 Mimari Tamamlanma Kriteri

Bu planın ana mimarisi aşağıdaki soruların tamamını bir yere bağlayabiliyorsa kapsam açısından yeterli kabul edilir:

```text
Veri nereden geliyor?
Hangi formatta?
Nasıl temizleniyor?
Nasıl dataset oluyor?
Nasıl tokenize ediliyor?
Tensor nasıl oluşuyor?
Model nasıl hesap yapıyor?
Matematiği ne?
Mimari neden böyle?
Model nasıl eğitiliyor?
Nasıl post-train ediliyor?
Nasıl değerlendiriliyor?
Inference sırasında ne oluyor?
Nasıl servis ediliyor?
Donanımda ne oluyor?
Nasıl ölçekleniyor?
RAG / tool / agent nasıl ekleniyor?
Multimodal veri nasıl işleniyor?
Production'da nasıl izleniyor?
Güvenlik riskleri neler?
```

Bu soruların her biri:

```text
Açıklama
+
Matematik
+
Kod
+
Gerçek işlem veya simülasyon
+
İlgili veri / format
+
Sistem seviyesi karşılığı
```

ile incelenebiliyorsa platform hedeflenen öğrenme amacına ulaşır.
