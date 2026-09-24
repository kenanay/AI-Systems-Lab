# Öğrenme Yolu ve Laboratuvarlar

Laboratuvarlar düşük maliyetli, gözlemlenebilir deneylerdir. Her laboratuvarın amacı kavramı açıklamak; production model davranışı hakkında kanıt üretmek değildir.

## Öğrenme Yolu (`/journey`)

**Tanım:** Curriculum, prerequisite graph, glossary, soru kontrolü ve ilerleme ekranı.

**Amaç:** Kullanıcının rastgele sayfa gezmek yerine kavram bağımlılıklarını izleyerek ilerlemesi.

**Kullanım:** Önce stage seçin, ön koşulları tamamlayın, soruyu yanıtlayın, sonra ilgili laboratuvara geçin.

API: `/api/v1/journey/curriculum`, `/graph`, `/glossary`, `/check-question` ve `/progress`.

## Laboratuvar matrisi

| Rota | Tanım ve amaç | Örnek kullanım senaryosu | İpucu |
|---|---|---|---|
| `/math-lab` | Matris, vektör, türev, gradient descent ve chain rule | Attention veya backprop öncesi temel işlemleri deneyin | Shape ve sayısal kararlılığı not edin |
| `/tensor-lab` | Shape, reshape, transpose, broadcast, matmul, activation ve backprop | Bir batch'in katmanlardan geçişini izleyin | Her işlemde `[batch, seq, dim]` boyutlarını yazın |
| `/nn-lab` | MLP, aktivasyonlar, loss/gradient ve optimizer yarışları | SGD/Adam davranışını aynı landscape'te karşılaştırın | Hızlı yakınsamayı genelleme olarak yorumlamayın |
| `/embedding-lab` | Vektör projeksiyonu, benzerlik ve analogy | Token vektörlerinin uzaydaki ilişkisini inceleyin | 2D projeksiyon orijinal uzayın tamamı değildir |
| `/attention-lab` | Scaled dot-product, head, mask ve heatmap | Token ilişkilerinin Q/K/V ile değişimini görün | Heatmap'i tek başına açıklama saymayın |
| `/transformer-lab` | Positional encoding, attention variant, block ve parametre hesabı | Model boyutu değişince maliyeti gözlemleyin | Parametre ve aktivasyon belleğini ayırın |
| `/systems-lab` | Roofline, quantization, memory ve throughput simülasyonu | GPU/context length kararı öncesi hesap yapın | Sonuçlar varsayım tabanlıdır |
| `/synthetic-lab` | Template tabanlı sentetik instruction/response üretimi | Kontrollü SFT örnekleri üretip filtreleyin | İnsan kalite kontrolü gerekir |
| `/rag-lab` | Chunk, embedding collection, search ve query | Kaynak dokümandan bağlamlı cevap prototipi kurun | Retrieval kalitesini generation'dan ayrı ölçün |

## Öğrenme biçimi

1. Kavramı sezgisel olarak ifade edin.
2. Küçük girdiyi değiştirin.
3. Formül ve shape değişimini takip edin.
4. Sistem maliyetini veya hata modunu not edin.
5. Aynı fikri gerçek veri/model yaşam döngüsündeki karşılığıyla bağlayın.

Bu sırayı atlayıp yalnızca grafiğe bakmak laboratuvarları görsel demo düzeyine indirir.
