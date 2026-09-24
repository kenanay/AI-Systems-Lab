# Kurulum ve Konfigürasyon

## Gereksinimler

- Python 3.10 veya üzeri
- Node.js ve npm
- Yerel geliştirme için `venv` ve `frontend/node_modules`
- İsteğe bağlı GPU/CUDA ve uyumlu PyTorch kurulumu
- Docker kullanımı için Docker Engine/Compose

## Başlatma seçenekleri

```bash
./scripts/start_app.sh start
./scripts/start_app.sh start --backend-port 8100 --frontend-port 3100
./scripts/start_app.sh start --detached
```

Windows:

```cmd
scripts\start_app.cmd start
scripts\start_app.cmd start -BackendPort 8100 -FrontendPort 3100
```

Launcher mevcut portu öldürmez. Doluysa sonraki boş portu seçer. `--strict-ports`
veya Windows'ta `-StrictPorts` port değişikliğini engeller.

## Docker Compose

Production sunucuya geçiş, [Güvenli Deployment Rehberi](deployment.md) ile
birlikte uygulanmalıdır. Bu dosyadaki Compose örneği local/staging başlangıcıdır.

```bash
docker compose up --build
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Compose portları host üzerinde `8000` ve `3000` olarak yayınlar. Bu portlar
doluysa compose otomatik olarak bir sonraki porta geçmez; port mapping'i kendiniz değiştirin.

## Environment değişkenleri

| Değişken | Varsayılan | Açıklama |
|---|---:|---|
| `BACKEND_HOST` | `localhost` | Bind adresi |
| `BACKEND_PORT` | `8000` | API portu |
| `BACKEND_RELOAD` | `true` | Geliştirme reload davranışı |
| `BACKEND_WORKERS` | `1` | SQLite ile 1 worker sınırı |
| `DATABASE_URL` | `sqlite:///./local_ai_lab.db` | Metadata veritabanı |
| `SECRET_KEY` | geliştirme placeholder | Production'da değiştirilmelidir |
| `JWT_SECRET_KEY` | yok | Production'da en az 32 karakterlik secret |
| `MAX_UPLOAD_SIZE` | 100 MB | Upload üst sınırı |
| `DATA_ROOT` | `./datasets` | Dataset kökü |
| `LOG_FILE` | `./logs/app.log` | Backend logu |
| `ALLOWED_ORIGINS` | localhost 3000/8000 | JSON liste olarak parse edilir |
| `RAG_EMBEDDER_TYPE` | `local` | RAG embedding backend'i |

Frontend için `NEXT_PUBLIC_API_URL` browser'ın backend'e erişeceği URL,
`API_URL` ise Next.js server-side rewrite hedefidir. Launcher bu iki değeri seçilen backend portuna göre ayarlar.

## Dizinler ve kalıcılık

- `datasets/`: ham/normalize/canonical dataset çıktıları
- `uploads/` veya `temp/`: geçici upload alanları
- `models/`: model artifact'ları
- `checkpoints/`: training checkpoint'leri
- `experiments/`: deney çıktıları
- `logs/`: uygulama logları
- `data/`: database veya runtime data

## Sağlık kontrolleri

```bash
curl http://localhost:8000/health
curl http://localhost:3000/
```

API şeması: `http://localhost:8000/docs` ve `http://localhost:8000/openapi.json`.
