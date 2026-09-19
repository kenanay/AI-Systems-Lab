---
inclusion: auto
name: matematik-aciklama
description: Matematiksel formüllerin açıklanma standardı ve matematik gösterim kuralları
---

# Matematik Açıklama Standardı

Bu projede matematiksel formüller **yalnızca gösterilmez, parçalanarak açıklanır.**

## Her Formül İçin Standart Yapı

Her matematiksel formül için mümkün olduğunca şu yapı kullanılmalı:

1. **Formül**
2. **Sembollerin anlamı**
3. **Tensor/matris shape bilgisi**
4. **Neden bu formül kullanılıyor?**
5. **Adım adım sayısal örnek**
6. **Geometrik veya sezgisel açıklama**
7. **PyTorch karşılığı**
8. **Gerçek modeldeki kullanımı**
9. **Sık yapılan hata**
10. **İlgili ön bilgiler**

## Örnek: Attention Formülü

### Formül
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
```

### Semboller
| Sembol | Anlam |
|--------|-------|
| Q | Query matrisi |
| K | Key matrisi |
| V | Value matrisi |
| K^T | Key matrisinin transpose işlemi |
| d_k | Key vektör boyutu |
| softmax | Skorları normalize eden fonksiyon |

### Shape Bilgisi
```
Q : [batch, heads, seq_len, d_k]
K : [batch, heads, seq_len, d_k]
V : [batch, heads, seq_len, d_v]
```

### PyTorch Karşılığı
```python
# Öğrenme modu - adım adım
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
output = weights @ v

# Profesyonel mod - optimize
output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

## LaTeX Gösterimi

Formüller için KaTeX veya MathJax kullanılmalı. Her formül için aşağıdaki sekmeler sunulabilir:

- `[Görsel Formül]` - Render edilmiş formül
- `[LaTeX Kaynağı]` - LaTeX kodu
- `[Sayısal Örnek]` - Gerçek sayılarla hesap
- `[Kod Karşılığı]` - PyTorch implementasyonu

## Matematik Temel Alanları

Kod yazarken bu matematik alanlarının gerekli olabileceğini unutma:

### Aritmetik ve Fonksiyonlar
- toplam, çarpım, üs, logaritma, fonksiyon kavramı

### Linear Algebra
- scalar, vector, matrix, tensor
- dot product, matrix multiplication, transpose, norm
- basis, projection

### Calculus
- limit fikri, derivative, partial derivative
- gradient, chain rule

### Probability/Statistics
- probability distribution, expectation, variance
- sampling, likelihood

### Optimization
- objective function, gradient descent, learning rate
- momentum, Adam/AdamW

### Information Theory (İleri)
- entropy, cross-entropy, KL divergence

## Shape Debugger

Her tensor operasyonu için shape bilgisi açıkça gösterilmeli:

```python
# YANLIŞ - shape bilgisi yok
output = x @ weight

# DOĞRU - shape bilgisi açık
# x shape: [batch, seq_len, d_model]
# weight shape: [d_model, d_model]
# output shape: [batch, seq_len, d_model]
output = x @ weight
```

## Kod Yürütme Görünümü

Kullanıcı bir fonksiyonu çalıştırdığında aşama aşama izleyebilmeli:

```python
scores = q @ k.transpose(-2, -1)
```

İçin:
```
q.shape               = [1, 4, 8, 32]
k.shape               = [1, 4, 8, 32]
k.transpose().shape   = [1, 4, 32, 8]
scores.shape          = [1, 4, 8, 8]
```

## Soyutlama Seviyeleri

Aynı işlemi üç seviyede açıklayabilmelisin:

**Seviye A — Sezgisel**
> Query, bir tokenın ne aradığını temsil eder.

**Seviye B — Matematiksel**
```
Q = XW_Q
```

**Seviye C — Kod/Sistem**
```python
q = self.q_proj(x)
```

## Calculus Öğrenme Noktaları

Backpropagation'a geçildiğinde şu kavramların durumu kontrol edilmeli:
- Function
- Derivative
- Partial Derivative
- Chain Rule
- Gradient

## Linear Algebra Öğrenme Noktaları

Attention'a geçmeden önce:
- Vector
- Matrix
- Transpose
- Dot Product
- Matrix Multiplication
- Shape

kontrol edilmeli.

## Bağlam İçi Mikro Ders

Kullanıcı bir formülde anlamadığı bir şey görürse (örn. `sqrt(d_k)`), ana dersten çıkmadan küçük bir panel açılabilir:

**Neden √d_k ile bölüyoruz?**
- scaling kavramı
- dot product büyüklüğü
- softmax üzerindeki etkisi
- küçük sayısal örnek

Panel kapatıldığında kullanıcı kaldığı adıma geri döner.
