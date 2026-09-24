# AI Systems Lab – Kapsamlı Güvenlik, Veri Bütünlüğü ve Mimari İnceleme Raporu

**Tarih:** 24 Eylül 2026  
**Depo:** `kenanay/AI-Systems-Lab`  
**İncelenen Commit Serisi:** `eaac050` -> `1ac8483` -> Güncel Düzeltmeler  

---

## 1. Yönetici Özeti ve Çözülen Maddeler

AI Systems Lab üzerinde gerçekleştirilen son güvenlik ve mimari incelemesinde tespit edilen tüm kritik bulgular (P0 ve P1) ve öncelikli geliştirme hedefleri başarıyla çözülmüş, kapsamlı testlerle doğrulanmıştır:

| # | Seviye | Konu | İlgili Dosya(lar) | Durum |
|---|--------|------|-------------------|-------|
| 1 | **P0 - Veri Bütünlüğü** | Ortak Fiziksel Dosya Yaşam Döngüsü ve Referans Sayımı | [`backend/routers/files.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/backend/routers/files.py)<br>[`backend/storage.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/backend/storage.py) | **Çözüldü** |
| 2 | **P1 - Eğitim Güvenilirliği** | Training Preflight Kontrollerinde Tam Fail-Closed Politikası | [`backend/services/training_service.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/backend/services/training_service.py) | **Çözüldü** |
| 3 | **P1 - Güvenlik** | Model Registry Dizin Erişiminin Sınırlandırılması | [`backend/routers/inference.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/backend/routers/inference.py) | **Çözüldü** |
| 4 | **P1 - Güvenlik** | Checkpoint Yükleme Güvenliği (`weights_only=True` ile RCE Koruması) | [`src/registry/model_registry.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/src/registry/model_registry.py)<br>[`backend/services/compatibility_service.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/backend/services/compatibility_service.py)<br>[`backend/services/training_service.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/backend/services/training_service.py) | **Çözüldü** |
| 5 | **P2 - Test Kapsamı** | Eşzamanlılık, Çok Kullanıcılı İzolasyon ve Fail-Closed Doğrulama Testleri | [`tests/test_auth_and_security.py`](file:///Users/asel/Documents/AI%20Systems%20Lab/tests/test_auth_and_security.py) | **Çözüldü** |

---

## 2. Detaylı Teknik Çözümler

### 2.1. Ortak Fiziksel Dosya Yaşam Döngüsü ve Referans Sayımı (P0)
* **Sorun:** Farklı kullanıcılar aynı içeriği (SHA-256) yüklediğinde tek fiziksel dosya kopyası paylaşılıyor; ancak bir kullanıcı dosyasını sildiğinde `storage_manager.delete_file` çağrılarak diğer kullanıcının da fiziksel dosyası yok ediliyordu.
* **Teknik Müdahale:**
  - `backend/routers/files.py` içinde `delete_file` metoduna sistem genelinde (tüm kullanıcılar üzerinde) çalışan referans sayımı mekanizması entegre edildi.
  - Silme sırasında kullanıcının `FileRecord` kaydı kaldırılır; ancak aynı fiziksel yola veya SHA-256 özetine işaret eden başka bir aktif kayıt mevcutsa fiziksel dosya korunur.
  - Yalnızca sistemdeki son referans silindiğinde fiziksel dosya diskten temizlenir.
  - `backend/storage.py` içindeki `get_file_path` metodu hem `raw_path` hem `data_root` çözünürlüğünü destekleyecek biçimde güncellenerek yol güvenliği korundu.
  - [`tests/test_auth_and_security.py::test_shared_physical_file_lifecycle_and_deletion`](file:///Users/asel/Documents/AI%20Systems%20Lab/tests/test_auth_and_security.py) testi ile Kullanıcı A silme işlemi sonrasında Kullanıcı B'nin verisinin korunduğu, Kullanıcı B de sildiğinde diskin temizlendiği uçtan uca doğrulandı.

### 2.2. Training Preflight Kontrollerinde Tam Fail-Closed Politikası (P1)
* **Sorun:** Parquet dosyası diskte bulunamadığında, bozuk olduğunda veya token indis kontrollerinde hata oluştuğunda bazı istisnalar `logger.warning` ile geçilerek eğitimin hatalı parametrelerle başlatılmasına izin verilebiliyordu.
* **Teknik Müdahale:**
  - `backend/services/training_service.py` içinde `create_job` preflight aşaması katı fail-closed kuralına bağlandı:
    1. Veri kümesi dosyasının diskte fiziken var olduğu doğrulanır; yoksa `ValueError` fırlatılır.
    2. Parquet schema'sı okunur; `PRETRAIN` için `token_ids`, `SFT` için `instruction`/`response` varlığı zorunlu tutulur.
    3. Dosya okuma hatası, bozukluk veya format uyuşmazlığı durumunda istisna loglanıp sessizce geçilmez; doğrudan `ValueError` fırlatılarak eğitim işinin oluşturulması engellenir (HTTP 400).
  - [`tests/test_auth_and_security.py::test_training_preflight_fail_closed_checks`](file:///Users/asel/Documents/AI%20Systems%20Lab/tests/test_auth_and_security.py) testi ile doğrulandı.

### 2.3. Model Registry Dizin Erişiminin Sınırlandırılması (P1)
* **Sorun:** `POST /api/v1/inference/load` endpoint'inde istemci serbest göreli `registry_dir` parametresi göndererek sunucudaki farklı dizinleri registry gibi gösterebiliyordu.
* **Teknik Müdahale:**
  - `backend/routers/inference.py` içerisinde `ALLOWED_REGISTRIES = {"models"}` zorunlu kılındı.
  - İstemcinin sunucu tarafından yönetilen kök dizin haricinde bir yol belirtmesi durumunda istek HTTP 400 ile reddedilir.
  - [`tests/test_auth_and_security.py::test_registry_dir_restriction_in_inference_load`](file:///Users/asel/Documents/AI%20Systems%20Lab/tests/test_auth_and_security.py) testi ile doğrulandı.

### 2.4. Checkpoint Yükleme Güvenliği ve RCE Engelleme (P1)
* **Sorun:** `torch.load(..., weights_only=False)` kullanımı, güvensiz veya dış kaynaklı pickle nesnelerinin yüklenmesi sırasında Arbitrary Code Execution (RCE) riski taşıyordu.
* **Teknik Müdahale:**
  - `src/registry/model_registry.py` (`load_model`, `register_model`), `backend/services/compatibility_service.py` ve `backend/services/training_service.py` genelinde `weights_only=True` standardı getirildi.
  - Resume aşamasındaki NumPy state'leri için PyTorch `safe_globals` mekanizması kullanılarak yalnızca güvenli veri türleri allowlist'e alındı; keyfi pickle nesneleri tamamen bloke edildi.
  - [`tests/test_auth_and_security.py::test_malicious_checkpoint_weights_only_blocking`](file:///Users/asel/Documents/AI%20Systems%20Lab/tests/test_auth_and_security.py) testinde zararlı komut içeren pickle nesnesi yüklenmeye çalışılmış ve komut yürütülmeden güvenle engellendiği doğrulanmıştır.

---

## 3. Test Metrikleri

* **Backend Test Süiti:**
  - Toplam Test Sayısı: **429** (Tümü Başarılı - 0 Hata, 0 Atlama)
  - Süre: ~71 saniye
* **Frontend Test Süiti:**
  - Toplam Test Paketi: **16**
  - Toplam Test Sayısı: **83** (Tümü Başarılı)
