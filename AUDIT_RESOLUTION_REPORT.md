# AI Systems Lab – Kapsamlı Güvenlik, Veri Bütünlüğü ve Eşzamanlılık Çözüm Raporu

> **Durum notu:** Bu rapor `a42ac2b`/`7ce3abb` dönemindeki denetim sonuçlarını
> içerir. Sonraki düzeltmelerde ContentRecord/FileRecord transaction sınırı,
> upload-delete interleaving testi, SFT runtime şema uyumu ve repository
> genelindeki checkpoint yükleme güvenliği güncellenmiştir. Güncel doğrulama,
> bu değişiklikleri içeren son commit ve CI çalışmasıyla birlikte okunmalıdır.

**Tarih:** 24 Eylül 2026  
**Depo:** `kenanay/AI-Systems-Lab`  
**İncelenen Commit Serisi:** `eaac050` -> `1ac8483` -> `a42ac2b` -> Güncel Güvenlik Düzeltmeleri  

---

## 1. Yönetici Özeti ve Çözülen Maddeler

AI Systems Lab üzerinde gerçekleştirilen bağımsız güvenlik, veri bütünlüğü ve mimari denetiminde tespit edilen bulgular (P1 seviyesindeki eşzamanlılık, checkpoint doğrulama, veri kümesi şeması ve test kapsamı gereksinimleri dahil) düzeltme kapsamına alınmış ve ilgili kaynak kod/test değişiklikleri uygulanmıştır:

| # | Seviye | Konu | İlgili Dosya(lar) | Durum |
|---|--------|------|-------------------|-------|
| 1 | **P1 - Veri Bütünlüğü** | Ortak Fiziksel Dosya Yönetiminde Atomik İşlemler & `ContentRecord` | [`backend/models.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/backend/models.py)<br>[`backend/routers/files.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/backend/routers/files.py) | **Uygulandı; güncel CI doğrulaması gerekli** |
| 2 | **P1 - Eşzamanlılık** | Çok Kullanıcılı Eşzamanlı Yükleme ve Silme Testleri | [`tests/test_auth_and_security.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/tests/test_auth_and_security.py) | **Tamamlandı** |
| 3 | **P1 - Model Güvenliği** | Checkpoint Güvenlik Testlerini Genişletme (Kayıt, Yükleme ve Inference) | [`src/registry/model_registry.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/src/registry/model_registry.py)<br>[`tests/test_auth_and_security.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/tests/test_auth_and_security.py) | **Uygulandı; güncel CI doğrulaması gerekli** |
| 4 | **P1 - Eğitim Güvenilirliği**| SFT Veri Kümesi Şema Doğrulaması (`instruction` + `response` Zorunluluğu) | [`backend/services/training_service.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/backend/services/training_service.py)<br>[`tests/test_auth_and_security.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/tests/test_auth_and_security.py) | **Tamamlandı** |
| 5 | **Dokümantasyon** | Güncel Teknik Rapor Senkronizasyonu & Eski Rapor Arşivleme | [`AUDIT_RESOLUTION_REPORT.md`](file:///Users/asel/Documents/AI%20Systems%20Lab/AUDIT_RESOLUTION_REPORT.md)<br>[`UYGULAMA_DEGERLENDIRME_RAPORU_2026-09-24.md`](file:///Users/asel/Documents/AI%20Systems%20Lab/UYGULAMA_DEGERLENDIRME_RAPORU_2026-09-24.md) | **Tamamlandı** |

---

## 2. Detaylı Teknik Çözümler ve Uygulama

### 2.1. Ortak Fiziksel Dosya Yönetiminde Atomik İşlemler & `ContentRecord` (P1)
* **Tespit Edilen Risk:**
  - Farklı kullanıcıların aynı SHA-256 içeriğini paylaştığı senaryoda, sıralı silmede referans sayımı veri kaybını önlüyordu; ancak kullanıcı kaydının silinmesi, kalan referansların tespiti ve fiziksel dosyanın kaldırılması tek bir atomik işlem olmadığından eşzamanlı silme veya eşzamanlı yükleme-silme yarış durumlarında (race conditions) ya artık dosya kalma ya da silinmekte olan yolun yeniden tahsis edilip veri kaybına uğraması riski mevcuttu.
* **Uygulanan Mimari Çözüm:**
  1. [`backend/models.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/backend/models.py) içerisinde bağımsız `ContentRecord` tablosu oluşturuldu (`sha256`, `relative_path`, `size_bytes`, `ref_count`, `status`).
  2. [`backend/routers/files.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/backend/routers/files.py) içerisine süreç içi thread lock ve veritabanı satır kilidi (`with_for_update`) eklendi.
  3. **Yükleme:** Duplicate kontrolü, ContentRecord güncellemesi ve FileRecord oluşturulması aynı transaction içinde yapılır. Unique yarışlarında rollback sonrası retry uygulanır.
  4. **Silme:** FileRecord silme, kalan referans sayımı ve ContentRecord kararı tek transaction içindedir. Fiziksel silme yalnızca commit sonrasında ve son referans yoksa yapılır.

### 2.2. Eşzamanlı Yükleme ve Silme Testleri (P1)
* **Kapsam:**
  - [`tests/test_auth_and_security.py::test_concurrent_file_uploads_and_deletions`](file:///Users/asel/Documents/AI%20Systems%20Lab/tests/test_auth_and_security.py):
    1. 5 farklı bağımsız kullanıcı dinamik olarak kaydedilir (`POST /api/v1/auth/register`) ve kimlik doğrulamaları yapılır.
    2. `ThreadPoolExecutor(max_workers=5)` kullanılarak 5 kullanıcı aynı anda aynı dosya içeriğini yükler (`POST /api/v1/files/upload`). Her kullanıcıya izole `file_id` atanırken tek bir fiziksel dosya deduplication ile diskte oluşturulur.
    3. 4 kullanıcı aynı anda (`ThreadPoolExecutor(max_workers=4)`) kendi dosya kayıtlarını siler (`DELETE /api/v1/files/{file_id}`).
    4. Eşzamanlı silme sonrasında 5. kullanıcının kaydının eksiksiz erişilebilir kaldığı (`HTTP 200`) ve fiziksel dosyanın diskte korunduğu doğrulanır.
    5. Son kullanıcı da sildiğinde fiziksel dosyanın diskten güvenle temizlendiği teyit edilir.
    6. Üç tekrar boyunca upload ve son referans silme işlemlerinin interleaving senaryosu da doğrulanır.

### 2.3. Checkpoint Güvenlik Testlerini Genişletme (P1)
* **Tespit Edilen Risk:**
  - Önceki testte yalnızca exploit payload'unun çalışmadığı kontrol ediliyordu. `register_model`'in hata üretmesi zorunlu tutulmuyor ve inference yükleme hattının reddi ayrı ayrı doğrulanmıyordu.
* **Uygulanan Çözüm:**
  - [`src/registry/model_registry.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/src/registry/model_registry.py) içindeki `register_model` metoduna checkpoint doğrulama adımı eklendi. `torch.load(..., weights_only=True)` ile dosya pickle kontrollerinden geçirilir; zararlı veya geçersiz veri tespit edildiğinde `ValueError("Zararlı veya geçersiz model checkpoint'i reddedildi: weights_only=True doğrulaması başarısız.")` fırlatılır.
  - [`tests/test_auth_and_security.py::test_malicious_checkpoint_weights_only_blocking`](file:///Users/asel/Documents/AI%20Systems%20Lab/tests/test_auth_and_security.py) genişletildi:
    1. **Kayıt Aşaması:** `register_model` çağrısının `ValueError("weights_only=True")` ile açıkça reddedildiği doğrulandı.
    2. **Yükleme Aşaması:** `torch.load(..., weights_only=True)` ile doğrudan deserialization istisnası ürettiği doğrulandı.
    3. **Inference Aşaması:** Gerçek `InferencePipeline.from_pretrained` girişinin zararlı checkpoint'i `weights_only=True` ile reddettiği doğrulandı; endpoint için de hata yolu korunur.
    4. **Sistem Bütünlüğü:** İşletim sistemi seviyesinde tetiklenmek istenen komut dosyasının (`flag_file`) hiçbir aşamada oluşturulmadığı doğrulandı.

### 2.4. SFT Veri Kümesi Şema Doğrulaması (P1)
* **Tespit Edilen Risk:**
  - SFT kontrolünde `instruction` bulunup `response` (veya hedef tokenlar) eksik olduğunda eğitim başlatılabilme riski mevcuttu.
* **Uygulanan Çözüm:**
  - [`backend/services/training_service.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/backend/services/training_service.py) içerisinde SFT preflight denetimi katı kurallara bağlandı:
    - Eğer veri kümesinde `instruction` sütunu varsa, `response` sütunu ZORUNLUDUR. Yoksa `ValueError("SFT veri kümesinde 'instruction' sütunu mevcut ancak zorunlu 'response' sütunu eksik!")` fırlatılır.
    - SFT için yalnızca (`instruction` + `response`) veya response alanlarını taşıyan JSON `text` kayıtları kabul edilir; `token_ids` yalnızca PRETRAIN akışında geçerlidir.
  - [`tests/test_auth_and_security.py::test_sft_preflight_missing_response_column`](file:///Users/asel/Documents/AI%20Systems%20Lab/tests/test_auth_and_security.py) testi ile eksik `response` içeren şemanın, alakasız sütunlar içeren geçersiz şemanın ve geçerli şemanın davranışları ayrı ayrı doğrulandı.

### 2.5. Dokümantasyon ve Tarihsel Rapor Senkronizasyonu
* [`UYGULAMA_DEGERLENDIRME_RAPORU_2026-09-24.md`](file:///Users/asel/Documents/AI%20Systems%20Lab/UYGULAMA_DEGERLENDIRME_RAPORU_2026-09-24.md) dosyasının en üstüne arşiv uyarısı (`[!WARNING] TARİHSEL ARŞİV BİLDİRİMİ`) eklenerek, bu dosyanın `425edf2` sürümüne ait eski bir durum özeti olduğu ve güncel durum için `AUDIT_RESOLUTION_REPORT.md` dosyasının esas alınması gerektiği açıkça belirtildi.
* Depo kökü ve dokümanlar güncel commit ve test metrikleriyle senkronize edildi.

---

## 3. Doğrulama ve Test Sonuçları

Tüm test paketleri yerel ortamda ve CI üzerinde tam izolasyon altında çalıştırılmıştır:

* **Backend Test Süiti (`pytest tests/ -q`):**
  - **430 Test Başarılı** (0 Hata, 0 Atlama, %100 Başarı Oranı) — tarihsel CI sonucu.
  - Bu rapordan sonraki commitlerde yeni interleaving ve checkpoint giriş noktası testleri eklenmiştir; güncel toplam test sayısı son CI çalışmasından alınmalıdır.
* **Frontend Test Süiti (`npm test -- --watchAll=false`):**
  - **16 Test Paketi / 83 Test Başarılı** (0 Hata, %100 Başarı Oranı)
  - Süre: ~2.8 saniye
* **Eşzamanlılık ve Güvenlik:**
  - 18 Güvenlik & Yetkilendirme Testi (`tests/test_auth_and_security.py`) tam başarıyla tamamlandı.
