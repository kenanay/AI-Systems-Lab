---
inclusion: auto
name: ogrenme-icerik
description: Eğitim içeriği ve öğrenme modülü standartları
---

# Öğrenme İçeriği Standardı

Bu dosya öğretici içerik oluştururken uyulması gereken standartları tanımlar.

## Standart Öğrenme Kartı Yapısı

Her önemli modül ve işlem için aşağıdaki bölümleri içeren bir öğrenme kartı oluşturulmalı:

### 1. Bu adımda ne yapıyoruz?
İşlemin kısa, teknik olmayan açıklaması.

### 2. Neden yapıyoruz?
Bu işlemin model zincirindeki rolü.

### 3. Önce ne bilmelisin?
Ön bilgi listesi:
```
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
```
src/model/attention.py
        ↓
TransformerBlock.forward()
        ↓
GPTModel.forward()
```

### 10. Gerçek veri üzerinde gösterim
Kullanıcının kendi dataset'inden küçük bir örnekle çalışma.

### 11. Alternatifler
Örneğin: BPE, WordPiece, Unigram gibi alternatif yöntemlerin neden var olduğu.

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
Kullanıcı neden bir sonraki konuya geçtiğini bilmeli.

## Ön Bilgi Uyarı Kartı

Kullanıcı ileri bir adıma erken geçtiğinde:

```
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

## Motivasyon ve Keşif Öğeleri

Uygulamada ölçülü biçimde şunlar bulunabilir:

### İpucu
> Shape uyuşmazlıkları Transformer kodlarken en sık karşılaşacağın hata sınıflarından biridir. Matris çarpımından önce son iki boyutu kontrol et.

### Neden önemli?
> Tokenizer kalitesi yalnız veri hazırlama konusu değildir; context window'un ne kadar verimli kullanıldığını da etkiler.

### Sık yapılan hata
> Train ve test verisine aynı dokümanın parçalarının dağılması evaluation sonucunu yapay biçimde yükseltebilir.

### Mini Deney
> **Mini Deney:** `d_k` büyüdükçe scaling uygulamadan softmax dağılımının nasıl değiştiğini gözlemle.

### Bugün ne öğrendin?
Bir modülün sonunda:
```
✓ Token ile Token ID arasındaki fark
✓ Vocabulary'nin rolü
✓ BPE merge işlemi
✓ Tokenizer'ın modelden neden ayrı olduğu
```

## Kontrol Soruları ve Aktif Öğrenme

Örnek:
> `Q` shape'i `[2, 8, 16, 64]`, `K` shape'i `[2, 8, 16, 64]` ise `QKᵀ` sonucunun shape'i nedir?

Kullanıcı yanıt verdikten sonra:
- doğru cevap
- neden
- matris çarpımı açıklaması
- ilgili ön bilgi bağlantısı

gösterilir. Bu mekanizma **zorunlu sınav olarak değil**, öğrenmeyi pekiştiren isteğe bağlı bir araç olarak tasarlanmalı.

## Pedagojik İçerik Veri Modeli

Açıklamaların frontend içine sabit metin olarak dağılması yerine yapılandırılmış olarak tutulması önerilir.

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
  what: "..."
  why: "..."
  intuition: "..."
  math: "..."
  code: "..."
  pitfalls: "..."
  next_step: multi_head_attention

tips:
  - "..."

quiz:
  - question: "..."
    answer: "..."
    explanation: "..."
```

## Öğrenme İçeriği ile Hesaplama Motorunun Ayrılması

Pedagojik içerik ile gerçek hesaplama motoru birbirine bağlı ama **ayrı katmanlar** olmalı:

```
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

## Öğrenme İlerleme Kaydı

Sistem mümkün olduğunca şu durumları kaydedebilir:
```
NOT_SEEN
INTRODUCED
PRACTICED
UNDERSTOOD
NEEDS_REVIEW
```

Bu sistem sertifika veya puan odaklı olmak zorunda değildir. Amaç, kullanıcının kendi kavramsal boşluklarını görmesidir.

## Kod Karşılaştırma Modu

Örneğin Attention:

**Eğitim kodu:**
```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
output = weights @ v
```

**Profesyonel kod:**
```python
torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

Amaç: Hazır fonksiyonun arka planda ne yaptığını göstermek.

## UI Bileşen Standartları

Her eğitim ekranında şu sabit bileşenler bulunmalı:

- `[Ne Yapıyoruz?]`
- `[Neden?]`
- `[Ön Bilgi]`
- `[Formül]`
- `[Sembol Tablosu]`
- `[Shape Adımları]`
- `[Kod Karşılığı]`
- `[Gerçek Örnek]`
- `[Sık Hata]`
- `[İpucu]`
- `[Kontrol Sorusu]`
- `[Sonraki Adım]`
