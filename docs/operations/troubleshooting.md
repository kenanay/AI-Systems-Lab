# Sorun Giderme

## Port dolu

Local launcher sonraki boş portu seçer. Portları görmek için macOS/Linux:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:3000 -sTCP:LISTEN
```

Windows:

```powershell
Get-NetTCPConnection -State Listen -LocalPort 8000,3000
```

Docker Compose kullanıyorsanız port mapping'i ayrıca değiştirmeniz gerekir.

## Frontend açılıyor ama API çalışmıyor

1. `curl http://localhost:<backend-port>/health` deneyin.
2. `NEXT_PUBLIC_API_URL` ve `API_URL` değerlerinin aynı backend portunu gösterdiğini kontrol edin.
3. Backend `ALLOWED_ORIGINS` listesinin frontend origin'ini içerdiğini kontrol edin.
4. Frontend'i backend hazır olduktan sonra yeniden başlatın. Launcher port değiştirirse
   ekranda yazdırılan frontend adresini kullanın; frontend ve backend adresleri aynı
   seçilmiş backend portunu göstermelidir.

## `next dev` chunk veya `.next` hatası

`next dev` ve `next build` aynı anda çalıştırılırsa `.next` içinde geçici chunk tutarsızlığı oluşabilir. Dev server'ı durdurup yeniden başlatın. Gerekirse yalnızca frontend build cache'ini temizleyin:

```bash
rm -rf frontend/.next
./scripts/start_app.sh start
```

## Backend ayar parse hatası

`ALLOWED_ORIGINS` gibi liste alanları JSON formatı bekler:

```dotenv
ALLOWED_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]
```

Virgülle ayrılmış ama JSON olmayan metin Pydantic settings parse hatası üretir.

## Login/401/refresh sorunları

Yerel launcher boş bir veritabanında demo kullanıcılarını otomatik oluşturur:
`admin/admin` ve `researcher/researcher123`. `SEED_DEMO_USERS=false` ise önce
register ile kullanıcı oluşturun veya yalnızca geliştirme veritabanında bu ayarı
geçici olarak açın. Browser cookie'lerinin silinmediğini, launcher çıktısındaki
frontend adresini kullandığınızı, backend/frontend portlarının aynı backend'i
gösterdiğini ve `/api/v1/auth/refresh` isteğinin cookie ile gönderildiğini kontrol edin.

## Training job başlamıyor

Dataset version'ın mevcut ve training'e izinli olduğunu kontrol edin. SFT için `instruction` ve `response` alanlarının bulunduğunu, tokenizer/model config'in uyumlu olduğunu ve output/checkpoint dizinlerinin yazılabilir olduğunu doğrulayın.

## RAG sonuçları boş veya alakasız

Chunk ve index adımlarını ayrı ayrı kontrol edin. Collection'ın document içerdiğini, embedder'ın yüklenebildiğini, sorgu dilinin modelle uyumlu olduğunu ve top-k/threshold değerlerini doğrulayın.

## Test ve type-check

```bash
pytest -q
cd frontend
npm run type-check
npm run build
```

Type-check'i build ile aynı anda çalıştırmayın; `.next/types` üretimi ile yarış oluşabilir.
