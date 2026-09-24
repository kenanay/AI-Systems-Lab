# Veri Hattı

## Amaç

Veri hattı, kullanıcı kaynaklarını modelden bağımsız ve izlenebilir bir
canonical dataset'e dönüştürür:

```text
Upload → FileRecord → Process/parse → DocumentRecord
       → quality/PII/metadata → compile → DatasetVersion
```

## Upload (`/upload`)

**Tanım:** TXT, Markdown ve parser'ı mevcut PDF kaynaklarını sisteme alır.

**Amaç:** Fiziksel dosyayı metadata ve kullanıcı sahipliği ile birlikte korumak.

**Kullanım:** Dosyayı seçin, upload sonucundaki file id'yi kaydedin, source,
license, copyright, security level ve `training_allowed` alanlarını doğrulayın.

**Teknik ayrıntı:** API karşılığı `POST /api/v1/files/upload`'dır. Listeleme,
metadata güncelleme, process ve silme işlemleri `files` router'ında bulunur.
Content/file kayıtları transaction ve cleanup mekanizması ile birlikte yönetilir;
fiziksel dosya silme başarısızsa cleanup tekrar dener.

**İpuçları:**

- Büyük dosyada önce küçük bir kesit ile parser davranışını doğrulayın.
- Kaynağı ve lisansı bilinmeyen dosyayı training'e izinli işaretlemeyin.
- Aynı dosyayı yeniden yüklemek yerine mevcut file metadata'sını kontrol edin.

## Dataset Explorer (`/dataset-explorer`)

**Tanım:** Dosya, doküman, boyut, kalite ve işlem durumlarını keşfetme ekranı.

**Amaç:** Hangi kaynağın canonical dataset'e gireceğini seçmeden önce veri kalitesini görünür kılmak.

**Kullanım senaryosu:** Stats ekranını açın, doküman listesini filtreleyin,
tekil kaynağın içeriğini ve metadata'sını inceleyin, sonra compiler'a geçin.

API: `GET /api/v1/datasets/stats`, `/documents`, `/documents/{id}` ve dosya tabanlı process endpoint'leri.

## Process ve ingestion

Process adımı parser seçer, metni normalize eder ve `DocumentRecord` üretir.
Tekli işlem `POST /api/v1/files/{file_id}/process`, toplu işlem
`POST /api/v1/files/batch-process` ile yapılır. Upload fiziksel kaynağı kabul
eder; process modelin tüketebileceği doküman türevini üretir.

## Dataset Compiler (`/dataset-compiler`)

**Tanım:** Seçili dokümanları version'lı canonical dataset artifact'ına çeviren job tabanlı derleyici.

**Amaç:** Training ve evaluation'ın doğrudan ham dosya davranışına bağlı olmasını engellemek.

**Kullanım:** Dataset adı/açıklaması girin, kaynak dokümanları seçin, split ve
compile parametrelerini belirleyin, job'ı başlatın. Job listesi ve version metadata'sı üzerinden sonucu doğrulayın.

API grubu:

- `POST /api/v1/datasets/compile`
- `GET /api/v1/datasets/compile/jobs`
- `GET /api/v1/datasets/versions`
- `GET /api/v1/datasets/versions/{dataset_id}/metadata`
- `GET /api/v1/datasets/versions/{dataset_id}/download`

**İpuçları:**

- Kullanılmış bir version'ı değiştirmeyin; yeni version üretin.
- SFT için gerçek eğitim akışının beklediği `instruction` ve `response` alanlarını koruyun.
- Split oranlarının toplamını ve boş split oluşmadığını kontrol edin.

## Veri yönetişimi

Her türevin yanında en az şu bağlam bulunmalıdır: kaynak file/document id,
dataset version, üretim zamanı, kullanıcı, lisans/güvenlik metadata'sı ve kullanılan parametreler.
Bu bilgiler yoksa çıktıyı yeniden üretilebilir kabul etmeyin.
