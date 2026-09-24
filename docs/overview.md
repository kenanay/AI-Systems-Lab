# Genel Bakış

## AI Systems Lab nedir?

AI Systems Lab, yapay zekâ sistemlerini yalnızca kullanan değil, nasıl
çalıştığını deneyerek öğreten local-first bir araştırma ve öğrenme
platformudur. Aynı uygulamada iki kullanım biçimi birleştirilir:

- **Öğrenme:** Matematik, tensör, embedding, attention, transformer ve sistem
  davranışını görselleştiren laboratuvarlar.
- **Üretim/araştırma:** Dosya ve dataset yönetimi, tokenizer, training,
  checkpoint, model doğrulama, inference, RAG ve evaluation.

Uygulama beta durumundadır. Her ekran aynı olgunluk seviyesinde değildir:
gerçek veri/model işlemleri backend API'leri üzerinden yürürken laboratuvarların
bir bölümü kontrollü simülasyon ve açıklama üretir. Arayüzdeki durum rozetleri
ve bu dokümandaki açıklamalar bu farkı görünür tutmalıdır.

## Felsefe

### 1. Açıklama önce

Kullanıcıya yalnızca “çalıştı” sonucu verilmez. Mümkün olan her yerde işlem;
amaç, matematiksel temel, algoritma, sistem maliyeti ve çıktı ile birlikte
sunulur. Örneğin attention ekranı yalnızca bir heatmap değil, `QKᵀ / √dₖ`,
softmax ve causal mask etkisini inceleme alanıdır.

### 2. Veri modelden önce gelir

Model formatları ve eğitim stratejileri değişebilir; ham kaynak ve canonical
dataset kaynağın kalıcı izidir. Dosya, işlenmiş doküman, dataset sürümü,
tokenizer ve model arasında lineage kurulması bu nedenle temel tasarım
kararıdır.

### 3. Ham veri korunur, çıktılar türetilir

Ingestion ve normalizasyon ham dosyayı yok etmez. Temizlenmiş doküman,
Parquet dataset veya eğitim çıktısı yeni bir türevdir. Geriye dönük
izlenebilirlik; kalite, güvenlik, lisans ve yeniden üretilebilirlik için
zorunludur.

### 4. Öğrenme ve profesyonel kullanım aynı zeminde buluşur

Bir kavram laboratuvarda küçük bir örnekle incelenebilir; daha sonra aynı
kavram gerçek veri hattında kullanılabilir. Bu iki mod birbirinden kopuk demo
uygulamaları değil, aynı kavramsal hattın farklı ölçekleridir.

### 5. Yerel ve yürütme-hedefinden bağımsız tasarım

Varsayılan geliştirme akışı tek makinede çalışır. CPU ile başlanabilir, uygun
donanımda GPU kullanılabilir, servisler Docker veya farklı çalışma ortamlarına
taşınabilir. Local-first yaklaşım güvenlik ve veri sahipliğini önceliklendirir;
üretim dağıtımı için ayrıca secret, TLS, veritabanı ve gözlemlenebilirlik
sertleştirmesi gerekir.

## Hedef yaşam döngüsü

```text
Kaynak dosya
  → upload ve metadata
  → ingestion / parsing / PII ve kalite kontrolleri
  → DocumentRecord
  → canonical dataset ve sürümleme
  → tokenizer
  → training (pretraining, SFT, LoRA)
  → checkpoint ve model registry
  → inference / playground
  → evaluation ve karşılaştırma
  → RAG veya uygulama entegrasyonu
```

Laboratuvarlar bu hattın kavramsal katmanlarını paralel olarak öğretir:
matematik → tensor → neural network → embedding → attention → transformer →
sistem maliyeti.

## Kimler için?

- AI/ML öğrenen ve bir kavramı görerek anlamak isteyenler
- Küçük ve orta ölçekli local model deneyleri yapan araştırmacılar
- Veri hattı ve model yaşam döngüsünü tek yerde denemek isteyen geliştiriciler
- Eğitim, benchmark ve RAG prototiplerini izlenebilir biçimde yürütmek isteyen ekipler

## Ne değildir?

- Hazır bir genel amaçlı model sağlayıcısı değildir.
- Her laboratuvar çıktısı üretim kalitesinde bir benchmark sonucu değildir.
- Production secret yönetimi, yüksek erişilebilirlik ve çok worker'lı SQLite
  dağıtımı varsayılan olarak çözülmüş değildir.

Bu sınırlar dokümantasyonda özellikle belirtilir; simülasyon sonucu gerçek
model ölçümü gibi sunulmamalıdır.
