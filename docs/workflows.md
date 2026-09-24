# Uçtan Uca Kullanım Senaryoları

## Senaryo 1: Attention'ı kavramdan ölçüme taşımak

**Amaç:** Attention skorunun matematiksel formülden görsel ve sistemsel
davranışa nasıl dönüştüğünü anlamak.

1. `/math-lab` içinde matris çarpımı, transpose ve softmax'ı deneyin.
2. `/tensor-lab` içinde `[batch, sequence, hidden]` şekillerinin nasıl
   değiştiğini gözlemleyin.
3. `/attention-lab` içinde token, head ve causal mask seçeneklerini değiştirin.
4. `d_k` büyüdüğünde neden `√d_k` ile ölçekleme yapıldığını kontrol edin.
5. `/transformer-lab` içinde attention'ın residual ve feed-forward akışını izleyin.

**İpucu:** Heatmap'i yalnızca “model bunu önemsedi” şeklinde yorumlamayın.
Skor; seçilen giriş, mask, head ve simülasyon parametrelerine bağlıdır.

## Senaryo 2: Küçük Türkçe dataset'ten inference'a

1. Lisanslı birkaç `.txt`/`.md` kaynak hazırlayın.
2. `/upload` ile yükleyin; source ve license metadata'sını doldurun.
3. `/dataset-explorer` içinde dosya ve doküman istatistiklerini kontrol edin.
4. Gerekirse batch process çalıştırın.
5. `/dataset-compiler` içinde veri split'i ve kaynak dokümanları seçin.
6. Dataset version metadata'sını kaydedin.
7. `/tokenizer` ile küçük vocabulary denemesi yapın; encode/decode çıktısını kontrol edin.
8. `/training` içinde kısa bir test job'ı başlatın.
9. `/models` model artifact'ını hash ile doğrulayın.
10. `/playground` üzerinde sampling ayarlarını karşılaştırın.
11. `/evaluation` içinde aynı prompt setiyle ölçüm alın.

**Başarı ölçütü:** Çıktının hangi dataset version, tokenizer ve model
checkpoint'inden üretildiği geriye doğru bulunabiliyor olmalıdır.

## Senaryo 3: RAG prototipi

1. Dosyaları upload/process edin.
2. `/rag-lab` içinde chunk boyutu ve overlap'i küçük bir örnekle inceleyin.
3. Dokümanları embedding collection'a indexleyin.
4. Search ile en yakın parçaları tek başına kontrol edin.
5. Query akışında retrieval sonuçlarının prompt bağlamına nasıl girdiğini inceleyin.
6. Kaynak göstermeyen veya alakasız retrieval sonuçlarını başarılı yanıt kabul etmeyin.

**İpucu:** RAG kalitesi yalnızca LLM'e bağlı değildir. Chunk sınırları,
embedding modeli, collection içeriği ve retrieval k değeri sonucu doğrudan etkiler.

## Senaryo 4: Donanım ve model boyutu kararı

1. `/systems-lab` içinde roofline, quantization ve memory simülasyonlarını açın.
2. `/transformer-lab` içinde model boyutu ve katman parametrelerini değiştirin.
3. Ağırlık, gradient, optimizer, activation ve KV-cache maliyetlerini ayrı değerlendirin.
4. Quantization kararını bellek kazancı, kalite ve throughput varsayımlarıyla birlikte değerlendirin.

## Senaryo 5: Sentetik veri ile kontrollü deney

1. `/synthetic-lab` içinde görev template'lerinden birini seçin.
2. Tohum, örnek sayısı ve difficulty ayarlarını sabitleyin.
3. Üretilen örnekleri score/filter adımlarından geçirin.
4. İnsan kontrolünden sonra dataset'e ingest edin.
5. Sentetik veriyi gerçek kaynaklardan ayrı metadata ile izleyin.

Sentetik veri üretimi veri kalitesi ve telif/lisans risklerini ortadan kaldırmaz;
yalnızca kaynak üretim yöntemini değiştirir.
