# Model Yaşam Döngüsü

## Tokenizer Lab (`/tokenizer`)

**Tanım:** Dataset'ten tokenizer job başlatma, tokenizer listesi ve encode/decode inceleme ekranı.

**Amaç:** Metin ile modelin gördüğü token id dizisi arasındaki sözleşmeyi kurmak.

**Kullanım:** Dataset ve vocabulary parametrelerini seçin, training job'ını
başlatın, sonra örnek metni encode/decode ederek Türkçe karakterler, özel token'lar ve unknown davranışını kontrol edin.

API: `/api/v1/tokenizer/train`, `/jobs`, `/list`, `/{tokenizer_id}/encode` ve `/{tokenizer_id}/decode`.

**İpucu:** Tokenizer değişirse aynı model config ile kıyas yapmak yanıltıcı olur.
Tokenizer id'sini dataset ve model metadata'sı ile birlikte saklayın.

## Training Studio (`/training`)

**Tanım:** Pretraining, SFT ve LoRA eğitim job'larını başlatma ve izleme alanı.

**Amaç:** Eğitim kararlarını veri, tokenizer, model config ve checkpoint ile birlikte kaydetmek.

**Kullanım:** Eğitim türünü seçin; dataset version, tokenizer, model/config,
epoch, batch size, learning rate, context length ve checkpoint ayarlarını gözden geçirin.
Job durumunu, log ve report akışını izleyin.

API: `POST /api/v1/training/start`, `GET /jobs`, `GET /jobs/{id}`, cancel, resume ve report endpoint'leri.

### Pretraining, SFT ve LoRA ayrımı

- **Pretraining:** Ham/continual text ile next-token öğrenimi.
- **SFT:** `instruction` → `response` örnekleriyle görev davranışı öğretimi.
- **LoRA:** Tüm ağırlıkları değiştirmek yerine düşük-rank adaptör parametreleri ile uyarlama.

Bu seçenekler yalnızca UI etiketi değildir; veri formatı, optimizer/checkpoint durumu ve beklenen çıktı farklıdır.

## Checkpoint

Checkpoint; model weights, optimizer/scheduler durumu, eğitim adımı ve config gibi devam etmek için gereken bilgileri taşır. Yükleme noktalarında güvenli `weights_only` yaklaşımı kullanılır; ancak eski/özel checkpoint formatlarının uyumluluğu otomatik varsayılmamalıdır.

**İpucu:** Checkpoint'i dosya adıyla değil hash, model config, tokenizer id ve dataset version ile tanımlayın.

## Model Hub (`/models`)

**Tanım:** Kayıtlı model artifact'ları, sürümleri, hash doğrulaması ve export işlemleri.

**Amaç:** Model dosyasının beklenen artifact olduğunu doğrulamak ve ONNX/GGUF/TorchScript gibi hedeflere kontrollü export yapabilmek.

**Kullanım:** Modeli seçin, hash verify çalıştırın, compatibility kontrolünü okuyun, export gerekiyorsa hedef format ve version'ı kaydedin.

API: `/api/v1/models`, `/{model_name}/verify`, `/export`, `/exports` ve `/download`.

## Playground (`/playground`)

**Tanım:** Yüklenmiş modelle interaktif generation ve decoding inceleme ekranı.

**Amaç:** Aynı prompt üzerinde sampling kararlarının ve token üretim akışının etkisini gözlemlemek.

**Kullanım:** Modeli yükleyin, prompt girin, temperature/top-k/top-p/max tokens ayarlarını değiştirin; gerekiyorsa stream, beam search, next-token olasılıkları ve attention ayrıntılarını inceleyin.

API: `/api/v1/inference/load`, `/status`, `/generate`, `/generate/stream`, `/beam-search`, `/next-token-probs` ve `/attention`.

**İpucu:** Seed, decoding ayarları ve model version sabitlenmeden karşılaştırma yapılmamalıdır.

## Evaluation (`/evaluation`)

**Tanım:** Metin inceleme, benchmark çalıştırma, sonuç listeleme ve model karşılaştırma ekranı.

**Amaç:** “Çalışıyor” ile “iyi ölçülmüş” arasındaki farkı korumak.

API: `/api/v1/evaluation/inspect-text`, `/run`, `/results`, `/compare`, `/metrics/{model_name}` ve radar karşılaştırması.

**İpucu:** Benchmark örnekleri, prompt template'i, model version ve metriklerin gerçek mi simüle mi olduğu raporda belirtilmelidir.
