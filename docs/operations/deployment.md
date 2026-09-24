# Güvenli Sunucu Kurulumu ve Deployment Rehberi

Bu doküman AI Systems Lab'ın bir Linux sunucuya güvenli biçimde alınmasını
anlatır. Örnek topoloji Docker Compose + PostgreSQL + Caddy/Nginx reverse proxy
üzerinedir. Alan adları ve IP'ler örnektir; gerçek secret değerleri hiçbir
zaman repository'ye yazılmaz.

## 1. Production topolojisi

```text
Internet
   │ HTTPS :443
   ▼
Reverse proxy (Caddy/Nginx)
   ├── /       → frontend:3000
   ├── /api/   → backend:8000
   └── /health → backend:8000
                       │
                       ├── PostgreSQL metadata
                       └── Kalıcı diskler:
                           datasets, uploads, models, checkpoints, logs
```

Kullanıcıya yalnızca `80/443` yayınlanmalıdır. Backend ve frontend portları
Docker ağı içinde veya yalnızca `127.0.0.1` üzerinde erişilebilir olmalıdır.

## 2. Sunucu ön koşulları

Önerilen başlangıç:

- Ubuntu 22.04/24.04 LTS veya eşdeğeri Linux
- En az 4 vCPU, 8 GB RAM, SSD
- Model training için ayrıca uygun GPU ve NVIDIA Container Toolkit
- Docker Engine ve Docker Compose plugin
- Bir domain: örneğin `ai.example.com`
- Sunucu sağlayıcısında firewall/security group

Firewall:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

SSH portunu değiştirdiyseniz `OpenSSH` yerine gerçek portu açın. `3000` ve
`8000` portlarını public firewall kuralı olarak açmayın.

## 3. Repository'yi sunucuya alma

En güvenli yöntem kodu secret'sız GitHub repository'den clone etmektir:

```bash
sudo mkdir -p /opt/ai-systems-lab
sudo chown "$USER":"$USER" /opt/ai-systems-lab
git clone https://github.com/kenanay/AI-Systems-Lab.git /opt/ai-systems-lab
cd /opt/ai-systems-lab
git checkout main
```

Production için hareketli `main` yerine doğrulanmış release tag'i veya commit
SHA tercih edin:

```bash
git fetch --tags origin
git checkout <dogrulanmis-tag-veya-commit>
```

Yerel dosyaları topluca `scp` ile kopyalamak yerine clone/pull kullanın.
Böylece `.env`, local database, `node_modules`, `.next`, model
checkpoint'leri ve geliştirici logları yanlışlıkla sunucuya taşınmaz.

## 4. Gizli bilgilerin korunması

### Git'e girmemesi gerekenler

Şunlar repository'ye, commit'e, GitHub Actions loguna veya Docker image'ına
girmemelidir:

- `.env`, `.env.production`, `.env.local` ve benzeri dosyalar
- `SECRET_KEY`, `JWT_SECRET_KEY`, database password ve API key'ler
- SSH private key, TLS private key, `.pem`, `.key`, certificate chain
- PostgreSQL dump, kullanıcı upload'ları ve kişisel veri içeren dataset'ler
- Model checkpoint'leri, registry credential'ları ve cloud access token'ları
- Production logları, cookie/token çıktıları ve browser export'ları

Repository `.gitignore` bu dosyaların önemli bölümünü korur; ancak
`.gitignore` yalnızca henüz izlenmeyen dosyaları engeller.

Kontrol:

```bash
git status --short
git ls-files | rg '(^|/)(\\.env|.*\\.pem|.*\\.key|.*\\.secret|.*\\.token)$'
```

Bir secret daha önce commit edildiyse:

1. Secret'ı hemen revoke/rotate edin.
2. Git geçmişinden kaldırma sürecini ayrıca uygulayın.
3. Yeni değeri yalnızca secret store veya sunucudaki korumalı dosyaya yazın.

### Sunucuda secret dosyası

Secret'ları repository dışında tutun:

```bash
sudo install -d -o root -g docker -m 0750 /etc/ai-systems-lab
sudo install -m 0640 -o root -g docker /dev/null \
  /etc/ai-systems-lab/production.env
sudoedit /etc/ai-systems-lab/production.env
```

Örnek içerik; gerçek değerleri kendiniz üretin:

```dotenv
ENVIRONMENT=production
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
BACKEND_RELOAD=false
BACKEND_WORKERS=1

SECRET_KEY=<openssl-rand-hex-32>
JWT_SECRET_KEY=<openssl-rand-hex-32>
COOKIE_SECURE=true
SEED_DEMO_USERS=false
DATABASE_URL=postgresql+psycopg://ai_lab:<password>@postgres:5432/ai_lab

ALLOWED_ORIGINS=["https://ai.example.com"]
NEXT_PUBLIC_API_URL=https://ai.example.com
```

Secret üretmek için:

```bash
openssl rand -hex 32
openssl rand -hex 32
```

`ALLOWED_ORIGINS` virgülle ayrılmış düz metin değil, JSON liste olmalıdır.

## 5. Production konfigürasyonu

Mevcut `docker-compose.yml` local/staging için başlangıç sağlar; doğrudan
production olarak kullanılmamalıdır. Ayrı bir `docker-compose.prod.yml` veya
production override içinde:

1. Backend `env_file` ile korumalı production env dosyasını okumalıdır.
2. SQLite yerine PostgreSQL kullanılmalıdır.
3. `BACKEND_RELOAD=false` olmalıdır.
4. `SEED_DEMO_USERS=false` kalmalıdır.
5. `COOKIE_SECURE=true` ve HTTPS kullanılmalıdır.
6. Frontend build sırasında gerçek public API/origin değeri verilmelidir.
7. Host portları yalnızca reverse proxy'ye bağlanmalıdır.
8. Healthcheck ve kalıcı volume'lar korunmalıdır.

PostgreSQL URL'si kullanılacaksa Python bağımlılıklarına uygun
`psycopg[binary]` sürücüsü eklenmeli ve image yeniden build edilmelidir.
Yalnızca `DATABASE_URL` değiştirmek yeterli değildir.

## 6. Veritabanı ve kalıcı dosyalar

Yedekleme kapsamına en az şunları alın:

- PostgreSQL veritabanı
- `data/`, `datasets/`, `uploads/`
- `models/`, `checkpoints/`
- gerekli metadata ve audit log'ları

Örnek PostgreSQL yedekleme:

```bash
docker compose exec -T postgres pg_dump -U ai_lab ai_lab \
  | gzip > /secure-backup/ai_lab_$(date +%F).sql.gz
```

Backup dizini uygulama repository'sinin altında tutulmamalıdır. Periyodik
backup yanında düzenli restore testi yapılmalıdır.

## 7. Domain ve HTTPS

Reverse proxy yönlendirmesi:

- `https://ai.example.com/` → frontend
- `https://ai.example.com/api/` → backend
- `https://ai.example.com/health` → backend health endpoint'i

TLS sertifikası Caddy/Let's Encrypt veya kurumun sertifika altyapısı ile
otomatik yenilenmelidir. Frontend'in public API adresi `localhost` değil,
kullanıcının eriştiği HTTPS origin'i olmalıdır.

## 8. İlk admin ve kullanıcı kurulumu

Normal `/api/v1/auth/register` akışı `admin` rolü oluşturulmasına izin
vermez; normal kayıtlar `researcher` veya `viewer` olur. Production'da
demo seed'i açılmamalıdır.

Bu nedenle go-live öncesinde güvenli bir admin bootstrap işlemi
tanımlanmalıdır:

- yalnızca bakım sırasında çalıştırılan tek seferlik bootstrap scripti,
- erişimi kısıtlı migration/SQL işlemi veya
- güvenilir bir identity provider entegrasyonu.

İlk admin yöntemi netleşmeden production deploy tamamlanmış kabul edilmemelidir.

## 9. Build, deploy ve doğrulama

```bash
cd /opt/ai-systems-lab
git fetch origin
git checkout <release-commit>

docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs --tail=200 backend frontend
```

Smoke test:

```bash
curl -fsS https://ai.example.com/health
curl -fsSI https://ai.example.com/
curl -fsSI https://ai.example.com/login
```

Ardından register/login cookie, test dosyası upload/process, iki kullanıcıyla
ownership izolasyonu, küçük dataset compiler, model/inference ve evaluation
akışlarını doğrulayın. Loglarda secret, token veya kişisel veri sızmadığını
kontrol edin.

## 10. Güncelleme ve rollback

Güncellemeden önce veritabanı ve artifact backup alın; çalışan image/commit ve
migration durumunu kaydedin. Önce staging veya küçük bir smoke test yapın.

```bash
git checkout <onceki-dogrulanmis-commit>
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml logs --tail=200
```

Veritabanı migration'ı geri alınamıyorsa uygulama rollback'i tek başına yeterli
değildir; backup restore planı gerekir.

## 11. Go-live kontrol listesi

- [ ] Domain DNS doğru sunucuya gidiyor.
- [ ] Yalnızca 22/80/443 public.
- [ ] HTTPS aktif ve otomatik yenileniyor.
- [ ] Production secret'ları Git dışında ve 0600/0640 izinleriyle korunuyor.
- [ ] `ENVIRONMENT=production` ve `COOKIE_SECURE=true`.
- [ ] Demo seed kapalı.
- [ ] PostgreSQL ve PostgreSQL driver hazır.
- [ ] Frontend public API adresi HTTPS origin'i gösteriyor.
- [ ] Admin bootstrap tamamlandı.
- [ ] Kalıcı volume ve backup/restore testi tamamlandı.
- [ ] Healthcheck, log rotation ve monitoring aktif.
- [ ] Upload, auth, ownership, training ve inference smoke testleri başarılı.
