# Mimari

## Katmanlar

```text
┌─────────────────────────────────────────────────────────────┐
│ Next.js + React + TypeScript frontend                       │
│ Sayfalar · Navbar · auth context · API client · deney state │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP/JSON, cookie auth
┌──────────────────────────────▼──────────────────────────────┐
│ FastAPI backend                                             │
│ auth · files · datasets · training · inference · labs       │
└───────────────┬───────────────────┬────────────────────────┘
                │                   │
     ┌──────────▼─────────┐  ┌─────▼────────────────────────┐
     │ Metadata database   │  │ Domain services and src/     │
     │ SQLite/PostgreSQL   │  │ ingestion · model · training │
     └──────────┬─────────┘  │ evaluation · rag · systems   │
                │            └──────────────┬───────────────┘
     ┌──────────▼──────────────────────────▼───────────────┐
     │ Filesystem artifacts: datasets, uploads, models,     │
     │ checkpoints, experiments, logs                        │
     └───────────────────────────────────────────────────────┘
```

## Frontend

`frontend/src/app/` Next.js App Router sayfalarını içerir. Ortak davranışlar
şu bileşenlerde toplanır:

- `components/Navbar.tsx`: Ana navigasyon; Veri Hattı, Model & Eğitim ve
  Laboratuvarlar gruplarını yönetir.
- `lib/api.ts`: Axios tabanlı API istemcisi, cookie gönderimi ve 401 sonrası
  refresh akışı.
- `lib/auth-context.tsx`: Giriş yapılmış kullanıcı ve oturum durumunu sağlar.
- `contexts/ExperimentContext.tsx`: Deney bağlamını frontend içinde taşır.
- `components/ModeBadge.tsx`: gerçek, simülasyon veya demo çıktılarının
  kullanıcıya ayrıştırılmasına yardımcı olur.

| Grup | Rotalar |
|---|---|
| Veri hattı | `/upload`, `/dataset-explorer`, `/tokenizer`, `/dataset-compiler` |
| Model yaşam döngüsü | `/training`, `/models`, `/playground`, `/evaluation` |
| Öğrenme yolu | `/journey` |
| Laboratuvarlar | `/math-lab`, `/tensor-lab`, `/nn-lab`, `/embedding-lab`, `/attention-lab`, `/transformer-lab`, `/systems-lab`, `/synthetic-lab`, `/rag-lab` |
| Hesap | `/login`, `/register`, `/profile` |

## Backend

`backend/main.py` FastAPI uygulamasını, CORS'u, lifespan başlangıç/temizlik
akışını ve router kayıtlarını yönetir. Router'lar ince HTTP katmanı; iş
kuralları mümkün olduğunca `backend/services/` ve `src/` altında tutulur.

Başlangıçta gerekli dizinler ve veritabanı hazırlanır, ertelenmiş content-store
silme kayıtları temizlenir ve periyodik cleanup görevi başlatılır.

## Veri ve dosya bütünlüğü

Dosya upload akışında fiziksel dosya ile `FileRecord`/`ContentRecord` metadata
ilişkisi korunur. Aynı süreç içindeki eşzamanlı işlemler kilit ve retry ile
kontrol edilir; cleanup fiziksel silme başarısızlıklarını sonraki çalışmaya
devreder. Birden fazla backend worker ile SQLite kullanımı production için
uygun değildir; `backend_workers > 1` olduğunda PostgreSQL gibi cross-process
kilitleme sağlayan bir veritabanı gerekir.

## API ve asenkron işler

Frontend `NEXT_PUBLIC_API_URL` üzerinden backend'e bağlanır. Next.js geliştirme
sunucusu ayrıca `API_URL` ile `/api/*` rewrite kullanır. Ayrıntılı endpoint
grupları [API Haritası](development/api-map.md) içindedir.

Tokenizer, dataset compilation ve training işlemleri job kayıtları ile takip
edilir:

```text
POST /start veya /compile → job_id + queued/running
GET /jobs/{job_id}         → completed | failed | cancelled
GET /report veya artifact  → sonuç
```

## Çalıştırma topolojileri

1. **Local launcher:** `scripts/start_app.sh` veya Windows'ta
   `scripts/start_app.cmd`; geliştirme ve öğrenme için.
2. **Docker Compose:** `docker-compose.yml` ile backend ve frontend konteynerleri.
3. **Üretim dağıtımı:** TLS, secret yönetimi, PostgreSQL, yedekleme,
   gözlemlenebilirlik ve uygun worker/queue mimarisi ayrıca yapılandırılmalıdır.
