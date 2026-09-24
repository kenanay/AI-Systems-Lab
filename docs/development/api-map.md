# Backend API Haritası

Backend taban adresi varsayılan olarak `http://localhost:8000`, API prefix'i `/api/v1`'dir. En güncel request/response şeması için çalışan sunucunun `/docs` veya `/openapi.json` çıktısı esas alınmalıdır.

| Alan | Prefix | Temel amaç |
|---|---|---|
| Auth | `/api/v1/auth` | register, login, refresh, logout, kullanıcı ve API key yönetimi |
| Files | `/api/v1/files` | upload, listeleme, metadata, process, batch process, silme |
| Datasets | `/api/v1/datasets` | stats, documents, export, dataset version işlemleri |
| Compiler | `/api/v1/datasets/compile` ve `/versions` | canonical dataset job ve artifact yönetimi |
| Tokenizer | `/api/v1/tokenizer` | tokenizer train, job, list, encode/decode |
| Training | `/api/v1/training` | pretraining/SFT/LoRA job, cancel, resume, report |
| Models | `/api/v1/models` | model registry, verify, compatibility, export, download |
| Inference | `/api/v1/inference` | load, status, generate, stream, beam, token probs, attention |
| Evaluation | `/api/v1/evaluation` | inspect, benchmark, sonuç, compare, radar |
| RAG | `/api/v1/rag` | chunk, index, collections, search, query |
| Embeddings | `/api/v1/embeddings` | project, similarity, analogy |
| Journey | `/api/v1/journey` | curriculum, graph, glossary, progress, question check |
| Math Lab | `/api/v1/math-lab` | matrix/vector/calculus işlemleri |
| Tensor Lab | `/api/v1/tensor-lab` | shape, reshape, transpose, broadcast, matmul, activation, backprop |
| NN Lab | `/api/v1/nn-lab` | MLP, optimizer race, landscapes, activations |
| Architecture Atlas | `/api/v1/architecture-atlas` | catalog, MoE route, simulate, scaling |
| Transformer Lab | `/api/v1/transformer-lab` | positional encoding, attention variants, block simülasyonu |
| Systems Lab | `/api/v1/systems-lab` | roofline, quantization, memory |
| Synthetic Lab | `/api/v1/synthetic-lab` | templates, generate, filter, score, export, ingest |

## İstemci davranışı

`frontend/src/lib/api.ts` Axios instance'ı cookie taşır, JSON için timeout kullanır, 401 durumunda refresh akışı dener ve network/backend hatalarını ayırır.

Yeni endpoint eklerken API client'a typed fonksiyon ekleyin; request/response tiplerini `frontend/src/types` ile eşleştirin.

## Endpoint kullanma sırası

Uzun süren işlerde HTTP istemcisi şu sözleşmeye uymalıdır:

```text
POST action → job_id
GET job/status → queued/running/completed/failed
GET report/artifact → sonuç
```

Kullanıcı sayfayı yenilediğinde yalnızca local state'e güvenilmemeli, job id ile backend'den durum tekrar okunmalıdır.

## API değişikliği kontrol listesi

- Pydantic request/response şeması güncellendi mi?
- Auth dependency ve scope doğru mu?
- Ownership/tenant filtresi korunuyor mu?
- Frontend typed client güncellendi mi?
- OpenAPI'da endpoint görünüyor mu?
- Başarılı, yetkisiz, bulunamayan ve geçersiz input testleri var mı?
- Dokümandaki API tablosu veya veri sözleşmesi değişti mi?
