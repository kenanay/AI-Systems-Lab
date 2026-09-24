# Güvenlik ve Veri Yönetişimi

## Kimlik doğrulama ve yetkilendirme

Auth endpoint'leri `/api/v1/auth` altındadır. Oturum cookie'leri frontend ve
backend arasında `withCredentials` ile taşınır; client 401 gördüğünde refresh akışını dener. Korumalı router'lar `require_access` bağımlılığı ile erişim kontrol eder.

Kullanıcıya ait dosya, dataset ve deney sorgularında sahiplik izolasyonu korunmalıdır. Admin-only kullanıcı ve API key yönetimi yalnızca yetkili hesapta görünmelidir.

## Secret ve varsayılan hesaplar

- `.env`, `.env.local`, key/cert ve secret dosyaları commit edilmez.
- Production'da `SECRET_KEY`/`JWT_SECRET_KEY` placeholder bırakılamaz.
- `COOKIE_SECURE=true` ve HTTPS production için gereklidir.
- Demo kullanıcı seed'i production'da kapalı olmalıdır.
- README, log veya ekran görüntüsünde gerçek token/API key yayınlamayın.

Geliştirme varsayılanları production credential'ı değildir. Gerçek kurulumda admin/researcher password değerlerini environment secret ile verin.

## Dosya ve PII güvenliği

Upload öncesi kaynağın lisansını ve training iznini kontrol edin. PII detection aktifken uyarıları yok sayıp veriyi export etmek veri yönetişimi ihlaline yol açabilir. PII kontrolleri erişim kontrolünün yerine geçmez.

Dosya yönetiminde kabul edilen uzantı ve boyut sınırları config ile kontrol edilir; dosya ile metadata transaction sınırları korunur; fiziksel silme sonrası tekrar deneme için cleanup kaydı tutulabilir.

## Checkpoint ve artifact güvenliği

Güvenilmeyen checkpoint'leri genel deserialize akışına sokmayın. Model hash'i,
kaynak dataset version'ı ve tokenizer id'siyle artifact provenance oluşturun.
Export edilen modelin hedef formatını ve compatibility kontrolünü kaydedin.

## Production sınırları

Production öncesi en az şunlar gerekir:

1. PostgreSQL gibi cross-process transaction/lock destekli veritabanı
2. Secret manager ve key rotation
3. HTTPS/TLS ve güvenli cookie
4. Reverse proxy, rate limit ve request size politikası
5. Merkezi log, audit ve yedekleme
6. Worker/job queue tasarımı ve kaynak limitleri
7. Model/dataset erişim politikaları ve silme yaşam döngüsü
