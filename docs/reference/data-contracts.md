# Veri Sözleşmeleri

Bu sözleşmeler, modüllerin birbirine hangi anlamlı çıktıyı verdiğini açıklar. Alanların kesin tipi ve zorunluluğu için Pydantic şemaları ile çalışan OpenAPI şeması esas alınır.

## FileRecord

Upload edilmiş fiziksel kaynağın metadata kaydıdır. Tipik bağlam: file id ve owner, original filename/size/content type, hash ve storage path, source/license/copyright, security level, `training_allowed`, processing status ve timestamps.

FileRecord, DocumentRecord'ın kendisi değildir; kaynak ile işlenmiş metin arasındaki lineage başlangıcıdır.

## DocumentRecord

Parser/ingestion sonucunda modelden bağımsız normalize edilmiş dokümandır. Dosya id'si, metin, kalite/PII bilgisi ve işlem metadata'sı ile ilişkilidir.

## Canonical Dataset

Training/evaluation formatından bağımsız, version'lı dataset artifact'ıdır. Kaynak dokümanlar, split bilgisi, schema, compiler parametreleri ve version metadata'sı saklanmalıdır.

## SFT örneği

Gerçek eğitim akışının beklediği temel sözleşme:

```json
{
  "instruction": "Kullanıcı görevi veya soru",
  "response": "Modelin öğrenmesi beklenen yanıt"
}
```

Ek metadata alanları kullanılabilir; ancak `instruction` ve `response` alanlarının adını değiştirip eğitim katmanına sessizce farklı schema vermeyin.

## Job sözleşmesi

Tokenizer, compiler ve training job'larında ortak kavramlar: `job_id`, `status` (queued/running/completed/failed/cancelled), `created_at`, `started_at`, `completed_at`, hata mesajı/report bağlantısı, giriş artifact/version ve kullanılan config.

Job sonuçları idempotent okunabilir olmalı; frontend yalnızca başlatma anındaki yanıta güvenmemelidir.

## Model artifact

Model adı ve version'a ek olarak model hash, architecture/config, tokenizer id/version, dataset version, checkpoint/parent model, training config/seed ve export formatı ilişkilendirilmelidir.

## RAG contract

RAG akışında bir chunk; source document id, collection, text/span bilgisi ve embedding/index metadata'sını korur. Search sonucu generation'a aktarılırken source reference kaybolmamalıdır.
