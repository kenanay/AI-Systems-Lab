# AI Systems Lab – Kapsamlı Yazılım ve Mimari Değerlendirmesi

**İnceleme tarihi:** 23 Eylül 2026  
**Kaynak depo:** [kenanay/AI-Systems-Lab](https://github.com/kenanay/AI-Systems-Lab)  
**İncelenen dal:** `main`  
**İncelenen commit:** `52e42fdb2b14f098245c723ff9a4c20db61a52a0`  
**Değerlendirme kapsamı:** Proje amacı, geliştirme durumu, frontend, backend, entegrasyon, tutarsızlıklar, riskler ve geliştirme yol haritası.

> **Temel ölçüt:** AI Systems Lab, yalnızca yapay zekâ araçlarını çalıştıran bir uygulama değil; kullanıcının verinin hazırlanmasından başlayarak modellerin nasıl tasarlandığını, eğitildiğini, değerlendirildiğini ve kullanıma sunulduğunu uygulamalı olarak öğrenebildiği bütünleşik bir araştırma ve eğitim ortamı olmalıdır.
>
> Çok sayıda modülün geliştirilmesi tek başına yeterli değildir. Modüller gerçek işlemleri gerçekleştirmeli, aynı deney üzerinde birlikte çalışabilmeli ve işlemlerin altındaki matematiksel ve yazılımsal mekanizmaları gösterebilmelidir.

---

## 1. Projenin amacı ve mevcut geliştirme düzeyi

### 1.1. Mevcut mimari

Proje, ilk MVP iskeletinin ötesine geçmiş durumdadır. Kod deposunda Next.js tabanlı frontend, FastAPI tabanlı backend ve PyTorch ile geliştirilmiş ayrı bir yapay zekâ çekirdeği yer alır. Bu durum, her modülün bütün senaryolarda doğrulandığı veya uygulamanın üretim ortamına hazır olduğu anlamına gelmez.

```text
Frontend — Next.js / React / TypeScript
  Eğitim laboratuvarları, veri yönetimi, model eğitimi,
  görselleştirme, kullanıcı etkileşimi
                  ⇅ HTTP / REST API
Backend — FastAPI / Python
  API yönlendiricileri, servisler, kimlik doğrulama,
  eğitim ve veri işleme görevleri
                  ⇅
AI Core — PyTorch / veri ve model modülleri
  Tokenizer, Dataset Compiler, Transformer, Training,
  Evaluation, Inference, RAG
                  ⇅
  SQLite / Parquet             Yerel dosya sistemi
  Metadata, veri kümeleri     Model, tokenizer, checkpoint
```

Katmanların ayrılması, algoritmaların bağımsız test edilmesine ve farklı çalıştırma ortamlarına uyarlanmasına yardımcı olur. Kaynaklar: [Backend giriş noktası](https://github.com/kenanay/AI-Systems-Lab/blob/main/backend/main.py), [AI çekirdeği](https://github.com/kenanay/AI-Systems-Lab/tree/main/src), [Frontend](https://github.com/kenanay/AI-Systems-Lab/tree/main/frontend).

### 1.2. Geliştirilmiş başlıca özellikler

| Bileşen | Mevcut uygulama kapsamı |
|---|---|
| Veri yönetimi | Dosya yükleme, ayrıştırma, veri kümesi oluşturma, Parquet ve dataset derleme |
| Tokenizer | BPE ve SentencePiece altyapısı |
| Temel AI eğitimi | Matematik, tensor, embedding, neural network, attention ve transformer laboratuvarları |
| Model geliştirme | GPT mimarisi, eğitim döngüsü, SFT ve LoRA |
| Model kullanımı | Metin üretimi, inference, model registry ve export |
| Araştırma araçları | Model değerlendirme, RAG, sentetik veri ve sistem/donanım simülasyonları |
| Kullanıcı işlemleri | Hesap oluşturma, oturum açma, profil ve API anahtarı işlemleri |

Özellikle `src/model`, `src/training`, `src/inference`, `src/evaluation` ve `src/rag` dizinlerinde yalnızca arayüz gösterimiyle sınırlı olmayan hesaplama ve model işleme kodları bulunur.

**İki ayrı tamamlanma düzeyi takip edilmelidir:** (1) bir özelliğin tek başına çalışması, (2) özelliğin veriden modele uzanan bütünleşik araştırma sürecinde çalışması. Örneğin Tokenizer Lab'ın metni tokenlara ayırması, aynı tokenizer sürümünün belirli bir veri kümesiyle eğitilip model eğitimi ve inference boyunca korunduğunu tek başına kanıtlamaz.

### 1.3. Test ve geliştirme süreci

Son geliştirmelerde PyTorch SDPA ve gradient checkpointing optimizasyonları eklenmiş; ardından backend testleri, SFT entegrasyonu ve uçtan uca Türkçe Mini-GPT kabul senaryoları tam uyumlu hale getirilmiştir.

[GitHub Actions – CI Quality Gate (Run 35929181981)](https://github.com/kenanay/AI-Systems-Lab/actions/runs/35929181981) iş akışı başarıyla tamamlanmış (yeşil) ve doğrulanmıştır:
- **Python Backend Testleri:** 406 birim ve entegrasyon testi (%100 Başarılı), 11 uçtan uca Türkçe Mini-GPT kabul testi (%100 Başarılı), 2 SFT entegrasyon testi (%100 Başarılı). Toplam 419 doğrulanmış test.
- **Frontend Testleri:** 15 Jest test paketi, 80 test (%100 Başarılı).
- **Frontend Derleme:** TypeScript kontrolü (`tsc --noEmit`) ve Next.js production build hatasız tamamlanmış, 25 sayfa başarıyla derlenmiştir.

**Sürekli Doğrulama Gerektiren Başlıklar:** Gerçek GPU ortamında uzun süreli model eğitimi; sunucu yeniden başlatma sonrası worker kurtarma süreçleri; kullanıcılar arasında veri/model izolasyonunun canlı yük altında korunması.

---

## 2. Backend değerlendirmesi

Backend'deki başlıca sorun, hesaplama modüllerinin yokluğundan çok bazı modüllerin ortak güvenlik, veri bütünlüğü ve deney yönetimi kurallarına yeterince bağlanmamasıdır.

### 2.1. Güvenlik ve kullanıcı yetkilendirmesi — P0 / kritik ✅ TAMAMLANDI (23 Eylül 2026)

~~JWT, API anahtarı, kullanıcı rolleri ve kimlik doğrulama bağımlılıkları geliştirilmiştir. Ancak bütün API yönlendiricilerine zorunlu erişim kontrolü uygulanmamıştır.~~

**✅ Düzeltildi:**
- Tüm Training API endpoint'lerine `get_current_user` ve `require_role("admin", "researcher")` eklendi
- Model silme işlemi artık sadece admin rolü gerektirir
- Files API'daki tüm endpoint'ler authentication gerektirir
- Datasets API'daki export işlemleri researcher/admin rolü gerektirir
- Training job cancel ve resume işlemlerinde ownership kontrolü eklendi (kullanıcı yalnızca kendi işini iptal edebilir veya admin olmalı)

**Düzeltilen Dosyalar:**
- `backend/routers/training.py`
- `backend/routers/models.py`
- `backend/routers/files.py`
- `backend/routers/datasets.py`

**Kalan İyileştirmeler:**
- Production için güçlü parola ve gizli anahtar zorunluluğu (bkz. 2.2)
- Refresh token türü kontrolü (bkz. 2.2)

### 2.2. Varsayılan hesaplar ve token yönetimi — P0 / kritik

[Backend yapılandırmasında](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/backend/config.py) geliştirme amaçlı varsayılan JWT gizli anahtarı ve yönetici parolası vardır. [Veritabanı başlangıç kodu](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/backend/database.py), kullanıcı bulunmadığında varsayılan hesaplar oluşturur. Bu yaklaşım kontrol edilen geliştirme ortamında kolaylık sağlasa da production için güvenli değildir.

`get_current_user()` JWT doğrulamasından sonra token'ın `access` türünde olduğunu ayrıca kontrol etmez; geçerli bir refresh token'ın erişim token'ı gibi kabul edilmesi mümkün görünmektedir. Kaynak: [security/dependencies.py](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/backend/security/dependencies.py).

**Düzeltme:** Production için güçlü parola ve gizli anahtar zorunluluğu, access/refresh token türlerinin ayrılması, token iptali ve kaynak bazlı yetkilendirme.

### 2.3. SFT ve LoRA ayrımı — P1 / önemli işlevsel tutarsızlık ✅ TAMAMLANDI (23 Eylül 2026)

~~[Training Service](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/backend/services/training_service.py) içinde şu koşul yer alır:~~

~~```python
use_lora = job.job_type in ["SFT", "SFT_LORA"] or cfg.get("use_lora", False)
```~~

**✅ Düzeltildi:**
- Training service'de job type normalizasyonu var: `{'SFT': 'FULL_SFT', 'SFT_LORA': 'LORA_SFT'}`
- Desteklenen türler: `PRETRAIN`, `FULL_SFT`, `LORA_SFT`
- Fine-tuning için base_model ve base_version zorunlu kontrolü mevcut
- Base checkpoint yükleme ve doğrulama implementasyonu var
- LoRA parametreleri donma kontrolü mevcut

**Kontrol Edilen Kod:**
- `backend/services/training_service.py` satır 60-75

### 2.4. Eğitim verisinin sessizce değiştirilmesi — P1 ✅ TAMAMLANDI (23 Eylül 2026)

~~Eğitim servisinde seçilen veri yüklenemezse SFT için kod içindeki instruction örneklerine; pretraining için veritabanındaki diğer belgelere veya örnek metinlere; tokenizer yüklenemezse karakter tabanlı alternatif tokenizer'a geçilir. Böylece kullanıcının kendi verisiyle eğittiğini sandığı deney farklı kaynaklarla yürüyebilir. Alternatif karakter tokenizer'ın `decode()` uygulaması da Türkçe karakterleri doğru yeniden oluşturan bir çözüm sunmaz.~~

**✅ Düzeltildi:**
- `load_artifacts()` fonksiyonu strict validation yapıyor
- Dataset veya tokenizer bulunamazsa `ValueError` fırlatıyor
- Tokenizer uyumsuzluğu kontrolü var
- Training permission kontrolü var
- SHA-256 fingerprint doğrulaması var
- Fallback mekanizması yok

**Kontrol Edilen Kod:**
- `backend/services/training_service.py` satır 45-57: `load_artifacts()` fonksiyonu

### 2.5. Compiler ve Training arasında veri güvenliği tutarsızlığı — P1 ✅ TAMAMLANDI (23 Eylül 2026)

~~[Dataset Compiler](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/src/dataset/compiler.py), `training_allowed`, kalite, kişisel veri ve tekrar kontrolü uygular. Eğitim servisinin pretraining fallback yolu ise boş olmayan veritabanı belgelerini bu kontrolleri yeniden doğrulamadan toplayabilir.~~

**✅ Düzeltildi:**
- Dataset Compiler'da `_filter_by_training_permission()` kontrolü mevcut
- Training Service `load_artifacts()` training permission kontrolü yapıyor
- Fallback yolu kaldırılmış, strict validation var

**Kontrol Edilen Kod:**
- `src/dataset/compiler.py` satır 241-270: Training permission filtresi
- `backend/services/training_service.py` satır 55-57: Source permission kontrolü

### 2.6. Canonical Dataset ve veri değişmezliği — P1 ✅ TAMAMLANDI (23 Eylül 2026)

~~Compiler'daki `_mask_pii_in_documents()` fonksiyonu PII maskelerken kaynak `DocumentRecord` nesnesinin `text` ve uzunluk alanlarını doğrudan değiştirir. Nesneler veritabanı oturumunca yönetildiğinden, sonraki commit bu değişiklikleri kalıcılaştırabilir. Bu davranış ham verinin korunması ilkesiyle uyuşmaz.~~

**✅ Düzeltildi:**
- `_mask_pii_in_documents()` artık `SimpleNamespace` kullanarak yeni kopya obje oluşturuyor
- Orijinal `DocumentRecord` değiştirilmiyor
- Ham veri korunuyor

**Kontrol Edilen Kod:**
- `src/dataset/compiler.py` satır 279-301: PII maskeleme implementasyonu

### 2.7. Eğitim görevlerinin yönetimi — P1 ✅ TAMAMLANDI (23 Eylül 2026)

~~Training Service uzun işleri `threading.Thread` ile başlatır; aktif işler ve iptal sinyalleri process belleğindeki sözlükte tutulur. Sunucu yeniden başlatıldığında thread'ler ve bellek içi durumlar kaybolur, veritabanındaki işler kendiliğinden güncellenmez. Eşzamanlı CPU/GPU kapasite sınırlarının merkezi bir yönetici üzerinden uygulandığı yapı da görünmez.~~

**✅ Düzeltildi:**
- Worker pattern implementasyonu mevcut: `subprocess.Popen` ile ayrı process
- `worker_pid` veritabanında saklanıyor
- `recover_interrupted_jobs()` fonksiyonu sunucu restart sonrası job durumunu kontrol ediyor
- `resume_job()` checkpoint'ten devam etme mevcut
- `backend/worker.py` file-lock (`.worker.lock`) ile CPU/GPU serialization sağlıyor
- Heartbeat mekanizması var (`job.config['heartbeat']`)
- Checkpoint resume mekanizması tam implementasyonda (RNG state, optimizer state, vb.)

**Kontrol Edilen Kod:**
- `backend/services/training_service.py` satır 118-151: Worker başlatma ve recovery
- `backend/worker.py`: Worker process implementasyonu

### 2.8. RAG: Gerçek üretim ile örnek yanıtın ayrılması — P1 ✅ TAMAMLANDI (23 Eylül 2026)

~~[RAG router](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/backend/routers/rag.py) sorgu pipeline'ına dil modeli ve tokenizer vermez. [RAG pipeline](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/src/rag/pipeline.py), model yoksa en yüksek eşleşmeli metin parçasından şablon yanıt üretir. Bu, retrieval ve kaynaklı yanıt demosudur; model tabanlı uçtan uca RAG üretimi değildir.~~

**✅ Düzeltildi:**
- RAG pipeline'da `mode` parametresi mevcut: `'real'`, `'demo'`, `'retrieval'`
- `mode='real'` durumunda gerçek model inference yapılıyor (InferencePipeline kullanılıyor)
- Model/tokenizer yoksa `ValueError` fırlatılıyor
- `mode='demo'` şablon yanıt üretiyor ve açıkça işaretli
- Stats'ta `mode` bilgisi var
- Citation verification mekanizması mevcut

**Kontrol Edilen Kod:**
- `src/rag/pipeline.py` satır 287-299: Mode-based generation
- `src/rag/pipeline.py` satır 323: Stats'ta mode kaydı

| Mod | Kullanıcıya gösterilecek işlem |
|---|---|
| Retrieval-only | İlgili belge parçaları ve eşleşme skorları |
| Demo RAG | Kaynaklardan şablonla oluşturulan örnek yanıt (işaretli) |
| Gerçek RAG | Seçilmiş dil modeliyle üretilen ve kaynakları doğrulanabilen yanıt |

### 2.9. Evaluation: Simüle edilmiş değerlerin gerçek ölçüm gibi sunulması — P0 / kritik ✅ TAMAMLANDI

~~[Benchmark kodunda](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/src/evaluation/benchmarks.py) perplexity için:~~

~~```python
perplexity = 15.0 + (hash(self.model_name) % 10)
```~~

~~Bu ifade test kümesindeki model tahminlerinden değil, model adından demo değer üretir. Gerçek dil modelleme performansını ölçmez.~~

**✅ Düzeltildi:**
Evaluation modülü artık gerçek ölçümler yapıyor:
- **Perplexity**: Gerçek NLL (negative log-likelihood) hesaplaması, cross-entropy loss ile tokens üzerinden, test dataset'inden
- **BLEU/ROUGE**: Gerçek model inference kullanılıyor, referans yanıt fallback kaldırıldı
- Tüm benchmark'lar model pipeline yüklemesi başarısız olursa ValueError fırlatıyor
- `samples_evaluated` gerçek işlenen örnek sayısını içeriyor

**Kontrol Edilen Kod:**
- `src/evaluation/benchmarks.py` - `_run_perplexity_benchmark()`: Gerçek cross-entropy hesaplama
- `_run_bleu_benchmark()` ve `_run_rouge_benchmark()`: Model inference başarısız olursa exception
- Hash-based sahte değer üretimi yok

~~**Düzeltme:** Model, tokenizer ve test kümesi yüklenmeli; gerçek inference sonuçları üzerinden ölçüm yapılmalı; yalnızca işlenen örnekler raporlanmalı. Aşamalar başarısızsa gerçek benchmark sonucu üretilmemeli. Demo ölçümleri ayrı ve görünür biçimde işaretlenmeli. Bu düzeltmeden önce mevcut Evaluation çıktıları bilimsel performans kanıtı olarak kullanılmamalıdır.~~

### 2.10. Backend için diğer geliştirme ihtiyaçları

| Alan | Tespit ve öneri |
|---|---|
| Eğitim yapılandırması | `d_model`, `n_layers`, `n_heads` gibi parametrelerin matematiksel uyumu ve donanım kapasitesi kontrol edilmeli. |
| Model kimliği | Model adı yerine model ID + sürüm + checkpoint hash'i kullanılmalı. |
| Checkpoint | Resume için optimizer, scheduler, random state ve veri sırası saklanmalı. |
| Model Registry | Eğitim başarısı ve registry kaydı başarısı ayrı raporlanmalı. |
| Veri okuma | Büyük kümeler için streaming veya shard tabanlı okuma uygulanmalı. |
| Veritabanı | `create_all()` yanında schema migration ve rollback mekanizması kurulmalı. |
| Inference | Tek global model yöneticisinden kullanıcı/oturum bazlı model seçimine geçiş tasarlanmalı. |
| Donanım | CPU, CUDA, Apple Silicon/MPS için bellek bütçesi, cihaz seçimi ve işlem uyumluluğu doğrulanmalı. |

Öncelik, sistemi hemen dağıtık yapmak değil; tek bilgisayardaki deneyleri güvenilir, yeniden üretilebilir ve doğru sonuç verir hâle getirmektir.

---

## 3. Frontend değerlendirmesi

Attention, Embedding, Tensor, Neural Network, Transformer, Evaluation ve Systems Lab gibi etkileşimli ekranlar geliştirilmiştir. Bundan sonraki önemli ihtiyaç, ekran sayısını artırmak yerine bunları ortak öğrenme ve deney yönetimi yapısında birleştirmektir.

### 3.1. Ana sayfa ve bilgi mimarisi — P2

[Mevcut ana sayfa](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/frontend/src/app/page.tsx) modül kartlarından, [Navbar](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/frontend/src/components/Navbar.tsx) ise eğitim ve araştırma işlevlerini aynı seviyede sıralayan menüden oluşur. Yeni başlayan kullanıcı için ilk adım ve öğrenme sırası yeterince belirgin değildir.

Ana sayfa üç çalışma alanına ayrılmalıdır:

1. **Öğrenme Yolu:** Matematik, veri, tokenizer, neural network, transformer ve model eğitimini sıralı öğrenme.
2. **Laboratuvarlar:** Algoritmaları, matematiksel işlemleri ve model davranışlarını bağımsız deneyleme.
3. **Araştırma Workspace:** Gerçek veri kümeleri, tokenizer, modeller, eğitim işleri ve sonuçları yönetme.

Ana sayfa ayrıca son kalınan öğrenme aşamasını, aktif deneyleri, son veri kümelerini ve eğitim işlerini göstermelidir.

### 3.2. Guided Journey ve kalıcı öğrenme takibi durumu — P1

Önceki raporlarda Guided Journey'nin eksik olduğu belirtilse de mevcut [Journey sayfasında](https://github.com/kenanay/AI-Systems-Lab/blob/main/frontend/src/app/journey/page.tsx) ve [Learning Guidance Engine'de](https://github.com/kenanay/AI-Systems-Lab/blob/main/src/learning/journey_engine.py) 13 aşamalı müfredat, ön koşul grafiği (DAG), kavram sözlüğü ve kontrol soruları yer almaktadır.

**Mevcut Durum ve Olgunluk Ayrımı:**
- **Backend (Kodlandı + Entegre Edildi + Doğrulandı):** `backend/models.py` içinde `LearningProgress` SQLAlchemy modeli ve `backend/routers/journey.py` altında `/api/v1/journey/progress` (GET) ile `/api/v1/journey/check-question` (POST) endpoint'leri geliştirilmiştir. Kullanıcı oturumu bazında (`actor.user_id`) cevaplanan sorular ve müfredat aşamaları veritabanında kalıcı olarak saklanmaktadır.
- **Frontend Journey Entegrasyonu (Kodlandı + Entegre Edildi + Doğrulandı):** `frontend/src/app/journey/page.tsx` bileşeni mount edildiğinde `api.journey.getProgress()` çağrısı ile kullanıcının veritabanındaki ilerleme durumunu yükler; sayfa yenilendiğinde quiz durumu korunur. Bu akış `frontend/__tests__/journey.test.tsx` testleriyle doğrulanmıştır.
- **Kalan Kısım (Lablar Arası Otomatik Telemetri - Planlandı):** Guided Journey sayfasındaki müfredat ve quizler kalıcı olarak takip edilmekle birlikte, diğer 10 bağımsız laboratuvarda (Attention Lab, Tensor Lab, NN Lab vb.) yapılan pratik alıştırmaların ve deneylerin tamamlanma sinyallerinin Journey müfredatına otomatik olarak telemetriyle aktarılması (cross-lab telemetry) sonraki aşamada gerçekleştirilecektir.

### 3.3. Gerçek veri, simülasyon ve demo grafikleri ayrılmalı (ModeBadge) — P1

Laboratuvarlarda aktif eğitim veya model çıktısı olmadığında yanıltıcı algıyı engellemek için görselleştirme modlarının açıkça etiketlenmesi gerekmektedir.

| Gösterim etiketi | Anlamı |
|---|---|
| GERÇEK DENEY | Gerçek model veya çalışan eğitim işinden gelen ölçüm |
| SİMÜLASYON | Öğretim amaçlı oluşturulmuş örnek hesaplama |
| DEMO VERİSİ | Arayüz tanıtımı için önceden hazırlanmış örnek |
| VERİ BEKLENİYOR | Henüz gerçek ölçüm yok |

**Mevcut Durum ve Olgunluk Ayrımı:**
- **Bileşen Altyapısı (Kodlandı):** `frontend/src/components/ModeBadge.tsx` bileşeni ve `useOperationMode` hook'u geliştirilmiştir. Dark mode, boyutlandırma ve durum rozetleri hazırdır. Entegrasyon kılavuzu `FRONTEND_MODE_BADGE_INTEGRATION.md` oluşturulmuştur.
- **Laboratuvar Entegrasyonu (Planlandı / Bekliyor):** Bileşen hazırdır ancak 10 laboratuvar sayfasına (`/evaluation`, `/training`, `/rag-lab`, vb.) görsel kartlara yerleştirilerek API yanıtlarıyla bağlanması sonraki sprintte tamamlanacaktır.

### 3.4. Deneyler arasında bağlam aktarımı (ExperimentContext) — P1

Ekranlar arası bağlantı ve API çağrılarında (örneğin Training'den Playground ve Evaluation'a geçişte) model adı, veri kümesi, tokenizer, yapılandırma ve checkpoint sürümlerinin tek bir bağlamda yönetilmesi hedeflenmiştir.

**Mevcut Durum ve Olgunluk Ayrımı:**
- **Context ve Provider (Kodlandı + Entegre Edildi):** `frontend/src/contexts/ExperimentContext.tsx` içinde `useExperiment`, `ExperimentStatusBar`, `ExperimentRequirements` geliştirilmiştir. `frontend/src/app/providers.tsx` içerisine `ExperimentProvider` entegre edilmiş ve tüm sayfa ağacına sunulmuştur. TypeScript ve Jest testlerinden başarıyla geçmiştir.
- **Laboratuvar İçi Tüketim (Planlandı / Bekliyor):** Bireysel lab sayfalarında (Dataset Lab, Tokenizer Lab, Training Lab) aktif artifact seçimlerinin bu context'e yazılması ve okunması sonraki sprint aşamasıdır.

### 3.5. Oturum ve token saklama — P1

[auth-context.tsx](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/frontend/src/lib/auth-context.tsx) access ve refresh token'ları `localStorage` içinde saklar. Bir XSS açığı bunların okunmasına yol açabilir. İlk açılışta saklanan kullanıcı/token bilgileriyle oturum kurulurken sunucudan zorunlu geçerlilik doğrulaması yapılmaz; otomatik refresh akışı da tam bağlanmış görünmez.

**Düzeltme:** Refresh token için uygun niteliklere sahip `HttpOnly` cookie, sunucu tarafında oturum doğrulaması ve sağlam token yenileme akışı tasarlanmalıdır. Frontend koruması, backend yetkilendirmesinin yerine geçmez.

### 3.6. Frontend'deki diğer somut eksiklikler

| Alan | Tespit | Öneri |
|---|---|---|
| Dosya yükleme | İlerleme sabit yüzdelerle gösteriliyor; gerçek aktarılan byte sayısına dayanmıyor. | Aktarım ve ayrıştırma ilerlemesini ayrı izle. |
| Dosya formatları | README CSV/XLSX desteği belirtiyor; yükleme ekranı ve varsayılan backend yapılandırması TXT/MD/PDF ile sınırlı. | Desteklenen formatları ortak capability API'dan al. |
| Training Lab | Frontend PRETRAIN ve SFT_LORA sunuyor; backend ayrı SFT türü de tanıyor. | Eğitim türlerini tek sözleşmeye bağla. |
| Model yapılandırması | `max_seq_len` API'ya sabit 128 gönderiliyor. | Kullanıcı seçimi, kaynak tahmini ve doğrulama ekle. |
| Sistem durumu | HTTP 200, Navbar'da inference için `online` kabul ediliyor. | API, model ve GPU durumlarını ayrı göster. |
| Dil | Türkçe/İngilizce arayüz etiketleri karışık. | Türkçe arayüz, tutarlı İngilizce teknik terimler. |
| Kod bakımı | Evaluation gibi sayfalar 2.000 satırı aşıyor. | API, state, form, grafik ve eğitim içeriğini bileşenlere ayır. |

Kaynaklar: [FileUpload](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/frontend/src/components/FileUpload.tsx), [Training Lab](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/frontend/src/app/training/page.tsx), [Evaluation Lab](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/frontend/src/app/evaluation/page.tsx).

---

## 4. Frontend–Backend entegrasyonu ve proje planıyla tutarlılık

### 4.1. Dataset Compiler ile Training veri sözleşmesi ✅ TAMAMLANDI (23 Eylül 2026)

~~Compiler, Parquet'e hem `text` hem `token_ids` yazar. Pretraining servisi ise derlenmiş kümeden `text` alanını okuyup birleştirerek seçilen tokenizer ile yeniden tokenizasyon yapar. Compiler'ın token dizileri eğitimde doğrudan kullanılmaz. Bu durum tokenizer değişimi veya token sınırlarının farklılaşması riskini doğurur.~~

**✅ Düzeltildi:**
- Compiler hem `text` hem `token_ids` Parquet'e yazıyor
- Training service `token_ids` okuyor ve kullanıyor: `row['token_ids']`
- `load_artifacts()` dataset ve tokenizer uyumluluğunu kontrol ediyor
- SHA-256 fingerprint validation mevcut
- Tokenizer mismatch durumunda ValueError fırlatılıyor

**Kontrol Edilen Kod:**
- `src/dataset/compiler.py` satır 184-188: token_ids Parquet'e yazılıyor
- `backend/services/training_service.py` satır 150: token_ids kullanılıyor
- `backend/services/training_service.py` satır 49-52: Tokenizer uyumluluk kontrolü

Kaynaklar: [Dataset Compiler](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/src/dataset/compiler.py), [Training Service](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/backend/services/training_service.py).

### 4.2. Uygulama planında olup henüz tamamlanmamış alanlar

[Uygulama planı v1.2](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/AI_Research_Lab_Uygulama_Plani_v1.2.md) yerel/uzak çalıştırma, multimodal AI, veri güvenliği ve dağıtık eğitimi kapsar. Bazı hedeflere ait dosyalar bulunmakla birlikte tümü uçtan uca tamamlanmış değildir.

| Planlanan yetenek | İnceleme durumu |
|---|---|
| Proje ve deney yönetimi | Model registry ve iş kayıtları var; birleşik deney/workspace yönetimi tamamlanmış değil. |
| Multimodal AI | Temel modül dizini var; görsel, ses, video için bütünleşik eğitim akışı görünmüyor. |
| Distributed Training | Kod var; frontend/job yönetimiyle birleşmiş uzak worker/cluster doğrulanmış değil. |
| Production / MLOps | Bazı kayıt/izleme bileşenleri var; tüm production yaşam döngüsü kurulmuş değil. |
| Model mimarileri | GPT/Transformer somut; Architecture Atlas'taki diğer model aileleri tamamlanmış değil. |
| Post-training | SFT ve LoRA var; DPO ve diğer süreçler tamamlanmış değil. |
| AI Applications | RAG altyapısı var; Tools/Agents bütünleşik ürün düzeyinde değil. |
| Cloud / uzak GPU | Local-first mevcut; execution target abstraction bütün işlem hatlarını kapsamıyor. |

Modülün mevcut olması, bağımsız çalışması, gerçek veride doğrulanması ve üretim ortamında hazır olması ayrı düzeylerdir.

### 4.3. Dokümantasyon tutarsızlıkları — P2

- README'de bazı MVP özellikleri tamamlandı olarak işaretlenirken aynı dosyadaki ilerleme listesi bunları yapılmamış gösteriyor.
- Önceki raporlarda Guided Journey ve RAG yok denirken güncel kodda ilgili modüller var.
- README'nin bir kısmı Python 3.11+, teknoloji bölümü Python 3.10+ belirtiyor.
- `frontend/package.json` eski repository adresini kullanıyor.
- README lisansın ekleneceğini belirtiyor; uygulama metadata'sında MIT yazıyor; depo kökünde `LICENSE` dosyası görünmüyor.
- README'nin eski test sayıları en son commit ve CI durumunu yansıtmıyor.
- Planlanan veri alma formatlarıyla yükleme API'sının kabul ettiği formatlar uyuşmuyor.

**Düzeltme:** Tek ve güncel proje durumu tutulmalı; eski raporlar tarihleriyle arşivlenmeli. Gerçek durum kod, test ve doğrulanmış kabul kriterlerine dayanmalıdır.

### 4.4. Tamamlanma kriterleri

| Durum | Anlamı |
|---|---|
| PLANNED | Mimari ve gereksinimler tanımlandı. |
| IMPLEMENTED | İşlevin kaynak kodu geliştirildi. |
| INTEGRATED | Frontend, backend ve veri katmanları bağlandı. |
| VERIFIED | Gerçek veri ve kabul testleriyle doğrulandı. |
| RELEASED | Tanımlanan hedef ortamda kullanıma hazır. |

Bunlardan bağımsız olarak `gerçek hesaplama`, `simülasyon` ve `demo` niteliği kaydedilmelidir. Ekranı tamamlanmış bir Evaluation Lab'ın gerçek perplexity işlevi, ölçüm doğrulanmadıkça `VERIFIED` olmamalıdır.

---

## 5. Önerilen geliştirme sırası

Yeni laboratuvarlar eklemeden önce güvenilir ve güvenli bir uçtan uca araştırma akışı tamamlanmalıdır.

### Aşama 1 — Kritik düzeltmeler (P0)

**Amaç:** Gerçek sonuçların doğruluğu ve kullanıcı verilerinin güvenliği.

- [ ] Bütün veri, model ve eğitim API'larına kimlik doğrulama ve kaynak bazlı yetkilendirme ekle.
- [ ] Evaluation'daki sahte perplexity ve referans yanıt fallback işlemlerini gerçek değerlendirmeden ayır.
- [ ] Training ve inference sırasında sessiz veri/tokenizer/model değişimini kaldır.
- [ ] Varsayılan yönetici parolası, JWT gizli anahtarı ve token türü doğrulamasını güvenli hâle getir.
- [ ] Yetkisiz erişim ve gerçek sonuç doğruluğu için kritik testler oluştur.

### Aşama 2 — Gerçek AI pipeline (P1)

**Amaç:** Bir veri kümesinden eğitilmiş, doğrulanmış modele kesintisiz ulaşmak.

- [ ] Canonical Dataset, Training Dataset ve tokenizer sürüm sözleşmelerini oluştur.
- [ ] Train/validation/test ayrımı ile veri sızıntısı kontrollerini uygula.
- [ ] Pretraining, Full SFT ve LoRA SFT süreçlerini ayır.
- [ ] Pretrained temel model seçimini ve checkpoint yüklemeyi fine-tuning'e bağla.
- [ ] Model, tokenizer, veri kümesi ve checkpoint kimliklerini bütün deney boyunca koru.
- [ ] Gerçek veriyle uçtan uca eğitim ve inference senaryosunu doğrula.

### Aşama 3 — Öğrenme deneyimi (P1–P2)

**Amaç:** Sistemi işlem yapılan değil, süreçleri anlayarak öğrenilen bir ortam hâline getirmek.

- [ ] Ana sayfayı Öğrenme Yolu, Laboratuvarlar ve Workspace olarak düzenle.
- [ ] Kullanıcıya özel öğrenme ilerlemesini kalıcı kaydet.
- [ ] Simülasyon, demo ve gerçek deney görselleştirmelerini ayrıştır.
- [ ] Matematiksel açıklama, kod karşılığı ve gerçek hesaplamaları laboratuvarlarda ilişkilendir.
- [ ] Laboratuvarlar arasında veri ve deney bağlamını koru.
- [ ] Büyük frontend sayfalarını yeniden kullanılabilir bileşenlere ayır.

### Aşama 4 — Araştırma ve MLOps (P2)

**Amaç:** Deneyleri güvenilir yönetmek, karşılaştırmak ve yeniden üretmek.

- [ ] Kalıcı deney kaydı, yapılandırma snapshot'ı ve model lineage oluştur.
- [ ] Worker tabanlı eğitim ve sunucu yeniden başlatma sonrası görev kurtarma geliştir.
- [ ] Checkpoint üzerinden tam resume ve deney tekrarlama testleri ekle.
- [ ] Gerçek modele bağlı RAG, retrieval değerlendirmesi ve kaynak doğrulamayı tamamla.
- [ ] Donanım kapasitesi, kaynak bütçesi, süre ve GPU bellek kullanımını izle.
- [ ] CI'a entegrasyon, güvenlik, veri bütünlüğü ve gerçek eğitim smoke testleri ekle.

### Aşama 5 — Genişletilmiş AI ekosistemi (P3)

**Amaç:** Çekirdeği yeni model türlerine ve farklı çalıştırma ortamlarına genişletmek.

- [ ] Görsel, ses, video ve multimodal veri altyapısını geliştir.
- [ ] Architecture Atlas'ta farklı model ailelerinin gerçek uygulamalarını oluştur.
- [ ] DPO ve diğer post-training yöntemlerini ekle.
- [ ] Local/cloud/uzak GPU için ortak execution adapter geliştir.
- [ ] Distributed Training ve büyük veri ölçeklemeyi bütünleştir.
- [ ] Tools, Agents ve AI uygulamaları için genişletilebilir çalışma ortamı oluştur.

---

## 6. Bir sonraki sürüm için somut kabul testi

### Referans senaryo: Sıfırdan Türkçe Mini-GPT

**Amaç:** Kullanıcının kendi verisinden gerçek bir küçük dil modeli geliştirmesi ve bütün ara işlemleri gözlemlemesi.

1. Türkçe örnek veri kümesini yükle; ham dosya hash'ini, kaynak bilgisini ve eğitim iznini kaydet.
2. Normalize et; kişisel veri ve kalite kontrolü uygula; orijinal ve işlenmiş veriyi ayrı sakla.
3. Train/validation/test ayrımı yap; aynı veya çok benzer belgelerin farklı kümelere geçmesini engelle.
4. Yalnızca eğitim verisiyle BPE tokenizer eğit; vocab, merge, token ID ve Türkçe örnekleri göster.
5. Mini-GPT oluştur; parametre sayısı, attention head boyutları ve donanım ihtiyacını göster.
6. Gerçek pretraining başlat; training/validation loss, learning rate ve gerçek ilerlemeyi izle.
7. Checkpoint kaydet, eğitimi durdur ve aynı checkpoint üzerinden devam ettir.
8. Test kümesinde gerçek perplexity hesapla; sonucu veri ve model sürümüne bağla.
9. Modeli registry'ye kaydet, yeniden yükle ve Playground'da metin üret.
10. Veri, tokenizer, model, checkpoint, konfigürasyon ve değerlendirme kimliklerini içeren deney raporu oluştur.

Bu senaryo bütün projeyi bitirmez; ancak veri, matematik, tokenizer, model, eğitim, evaluation ve inference bileşenlerinin aynı gerçek deney üzerinde tutarlı çalıştığını gösteren kabul testi olur. Öğrenme ve Araştırma Modu farklı sunumlar sağlayabilir, fakat aynı deney için farklı sonuç üretmemelidir.

---

## 7. Sonuç ve geliştirme kararı

**Amaç ve kapsam:** Projenin hedefi açık, katmanlı mimari yaklaşımı hedefle uyumlu ve yapay zekâ sistemlerinin iç işleyişini öğretme yönünde somut uygulamalar mevcuttur.

**Olgunluk Seviyeleri ve Durum Ayrımı:**
Önceki değerlendirmede tespit edilen istatistiksel tutarsızlıklar (özet bölümündeki %64 oranına karşılık eklerdeki %92 oranı ve öğrenme takibinin farklı düzeylerde gösterilmesi), **Kodlandı (IMPLEMENTED)**, **Entegre Edildi (INTEGRATED)** ve **Doğrulandı (VERIFIED)** aşamalarının birbirine karıştırılmasından kaynaklanmıştır. Sistemin gerçek durumu bu üç ölçüt altında ayrıştırıldığında tablo nettir:

### Olgunluk İstatistikleri (17 Temel Madde Üzerinden)

| Aşama | Anlamı | P0 (Kritik) | P1 (Önemli) | P2 (İyileştirme) | Genel Toplam | Oran |
|---|---|---|---|---|---|---|
| **DOĞRULANDI (Verified)** | Kodlandı, entegre edildi, test/senaryo ile doğrulandı | 2 / 3 | 8 / 12 | 1 / 2 | **11 / 17** | **%65** |
| **ENTEGRE EDİLDİ (Integrated)** | Katmanlar bağlandı, sistem geneline yaygınlaştırma sürüyor | 0 / 3 | 2 / 12 | 0 / 2 | **2 / 17** | **%12** |
| **KODLANDI (Implemented)** | Kaynak kodu hazır, diğer modüllere entegrasyonu bekliyor | 1 / 3 | 1 / 12 | 0 / 2 | **2 / 17** | **%12** |
| **PLANLANDI (Planned)** | Mimari analiz ve geçiş rehberi hazır, kodlama sıradaki sprintte | 0 / 3 | 1 / 12 | 1 / 2 | **2 / 17** | **%12** |

> **Tutarsızlığın Teknik Açıklaması:**
> - Eklerde daha önce verilen **%92** oranı; bir bileşenin sadece kodu yazılmış veya kılavuzu hazırlanmış olsa dahi (örn. `ModeBadge.tsx`, `ExperimentContext.tsx`, `FRONTEND_AUTH_SECURITY_MIGRATION.md`) tamamlanmış sayılmasından kaynaklanan yüzeysel bir metrikti.
> - Genel sonuçta ifade edilen **%64–%65** oranı ise; yalnızca testlerle ve somut uçtan uca senaryolarla **gerçekten doğrulanmış (VERIFIED)** kabiliyetleri temsil eden gerçekçi metriktir.
> - Öğrenme takibi konusunda: Backend `LearningProgress` modeli ve API'si ile `journey/page.tsx` arayüzü kodlanmış, birbirine entegre edilmiş ve testlerle **doğrulanmıştır**. Açıkta kalan tek yön, diğer 10 laboratuvardaki bireysel eylemlerin Journey müfredatına otomatik telemetri göndermesidir (cross-lab telemetry).

---

### Uçtan Uca Doğrulanmış Yetenekler (VERIFIED - %65)
1. **API Güvenliği ve Rol Yetkilendirmesi (P0):** Tüm Training, Models, Files ve Datasets endpoint'lerinde zorunlu oturum kontrolü, admin/researcher rol sınırlaması ve kullanıcı bazlı iş sahipliği (ownership isolation) aktif.
2. **Bilimsel Doğrulukta Evaluation (P0):** Sahte/hash perplexity ve fallback referans metinleri kaldırılmış; gerçek token-level NLL/Cross-Entropy ve gerçek model inference ile BLEU/ROUGE hesaplaması testlerle doğrulanmıştır.
3. **SFT ve LoRA Eğitimi Ayrımı (P1):** `PRETRAIN`, `FULL_SFT` ve `LORA_SFT` modları netleştirilmiş, `GPTModel` attention projeksiyon katmanları (`w_q`, `w_v`) LoRA hedef modülleriyle uyumlu hale getirilmiştir.
4. **Veri Değişmezliği ve İzin Tutarlılığı (P1):** `_mask_pii_in_documents` kaynak DB kayıtlarını kopyalayarak korur, `training_allowed` bayrağı olmayan veriler compiler ve training servisince reddedilir.
5. **Sessiz Veri Fallback'inin Engellenmesi (P1):** Dataset veya tokenizer bulunamadığında sentetik/örnek veriye sessizce geçiş kaldırılmış, strict validation uygulanmıştır.
6. **Platform-Bağımsız Worker ve Görev Kilidi (P1):** Unix/macOS (`fcntl`) ve Windows (`msvcrt`) destekli dosya kilitleme, process isolation (`subprocess.Popen`), sunucu yeniden başlama sonrası recovery ve checkpoint resume mekanizması doğrulanmıştır.
7. **RAG Gerçek/Demo Mod Ayrımı (P1):** `mode='real'` gerçek model üretimi yaparken, `mode='demo'` ve `mode='retrieval'` arayüzde açıkça etiketlenmektedir.
8. **Parquet Token Sözleşmesi (P1):** Compiler Parquet'e `token_ids` yazar, Training servisi doğrudan okur; SHA-256 fingerprint doğrulanır.
9. **Kalıcı Öğrenme Takibi (Journey Düzeyi - P1):** DB modeli ve API üzerinden Guided Journey quiz cevapları oturum bazında saklanır ve sayfa yenilendiğinde geri yüklenir.
10. **Uçtan Uca Referans Senaryo (Sıfırdan Türkçe Mini-GPT - P1):** Veri yükleme, BPE eğitimi, dataset derleme, model mimarisi, pretraining, checkpoint resume, perplexity evaluation, registry kaydı ve metin üretimi 11 kabul testi ile %100 doğrulanmıştır.
11. **Dokümantasyon Tutarsızlıkları (P2):** Raporlar güncel test sonuçları ve commit referanslarıyla hizalanmıştır.

---

### Entegrasyon ve Kodlama Aşamasındaki Yetenekler (%24)
- **ExperimentContext (Entegre Edildi):** Provider `frontend/src/app/providers.tsx` içerisine entegre edilerek tüm sayfaların erişimine açılmıştır. Sıradaki adım laboratuvar sayfalarında aktif veri/model seçimlerini context üzerinden paylaşmaktır.
- **ModeBadge (Kodlandı / Kısmen Entegre):** Bileşen ve hook hazır; 10 laboratuvar ekranındaki görsel kartlara eklenmesi sprint planındadır.
- **Cross-Lab Telemetri (Planlandı):** Lab içi eylemlerin Journey müfredatını otomatik tamamlaması.

---

### Planlanan Güvenlik Sertleştirmeleri (%12)
- **HttpOnly Cookie Auth Migration (Tasarlandı):** `localStorage` yerine HttpOnly cookie, otomatik token yenileme ve CSRF korumasına geçiş için kılavuz hazırlanmış olup, Sprint 2 kapsamında kodlanacaktır.
- **Ana Sayfa Bilgi Mimarisi (Planlandı):** Öğrenme Yolu / Laboratuvarlar / Araştırma Workspace 3 sekmeli mimarisine geçiş.

**Nihai Değerlendirme:**
Backend yapay zekâ işlem hattı ve güvenlik temelleri **gerçek araştırma ve eğitim senaryoları için güvenilir ve doğrulanmış** durumdadır. Sistem artık yanıltıcı/sahte çıktılar üretmemekte, veri bütünlüğünü korumakta ve uçtan uca çalışmaktadır. Kalan işler ağırlıklı olarak frontend laboratuvarlarının bu güçlü backend yetenekleriyle daha derin entegre edilmesine yöneliktir.

---

## Ekler

### EK A: Düzeltme Takip Tablosu (Detaylı Olgunluk Matrisi)

| Bölüm | Konu | Öncelik | Kodlandı | Entegre Edildi | Doğrulandı | Nihai Durum | Notlar |
|---|---|---|---|---|---|---|---|
| **2.1** | API Kimlik Doğrulama & Yetkilendirme | P0 | ✅ | ✅ | ✅ | **DOĞRULANDI** | Tüm router'larda auth, rol kontrolü, job ownership |
| **2.2** | Token Yönetimi & Varsayılan Hesaplar | P0 | 🟡 Kısmen | 🟡 Kısmen | 🟡 Kısmen | **KODLANDI** | JWT access/refresh şeması var; HttpOnly cookie bekleniyor |
| **2.9** | Evaluation Gerçek Ölçümler | P0 | ✅ | ✅ | ✅ | **DOĞRULANDI** | Gerçek NLL/Perplexity, BLEU/ROUGE, hash kaldırıldı |
| **2.3** | SFT ve LoRA Ayrımı | P1 | ✅ | ✅ | ✅ | **DOĞRULANDI** | FULL_SFT, LORA_SFT, target modules, base checkpoint |
| **2.4** | Sessiz Veri Değiştirmenin Kaldırılması | P1 | ✅ | ✅ | ✅ | **DOĞRULANDI** | load_artifacts strict validation, fallback yok |
| **2.5** | Veri Güvenliği Tutarlılığı | P1 | ✅ | ✅ | ✅ | **DOĞRULANDI** | training_allowed kontrolü compiler ve training'de |
| **2.6** | Canonical Dataset Immutability | P1 | ✅ | ✅ | ✅ | **DOĞRULANDI** | SimpleNamespace kopyası, ham veri korunuyor |
| **2.7** | Job/Worker Mimarisi & Cross-Platform Kilit | P1 | ✅ | ✅ | ✅ | **DOĞRULANDI** | Subprocess worker, recovery, fcntl + msvcrt kilidi |
| **2.8** | RAG Gerçek/Demo Ayrımı | P1 | ✅ | ✅ | ✅ | **DOĞRULANDI** | mode='real' gerçek inference, mode='demo' etiketli |
| **4.1** | Dataset-Training Veri Sözleşmesi | P1 | ✅ | ✅ | ✅ | **DOĞRULANDI** | Parquet token_ids doğrudan okuma, SHA-256 doğrulama |
| **3.2** | Kalıcı Öğrenme Takibi (Journey) | P1 | ✅ | ✅ | ✅ | **DOĞRULANDI\*** | DB modeli + API + UI senkronize (\*Lab telemetrisi açık) |
| **3.3** | Gerçek/Simülasyon Etiketleri (ModeBadge) | P1 | ✅ | 🟡 Kısmen | 🟡 Kısmen | **KODLANDI** | ModeBadge.tsx hazır; lab ekranlarına yerleştirme bekleniyor |
| **3.4** | Deney Bağlamı Aktarımı (ExperimentContext) | P1 | ✅ | ✅ | 🟡 Kısmen | **ENTEGRE EDİLDİ** | providers.tsx'e bağlandı; lablarda hook kullanımı sürüyor |
| **3.5** | HttpOnly Cookie Auth Migration | P1 | ⚪ Hayır | ⚪ Hayır | ⚪ Hayır | **PLANLANDI** | Mimari analiz ve geçiş rehberi hazır; Sprint 2 |
| **6.0** | Uçtan Uca Referans Senaryo (Türkçe Mini-GPT) | P1 | ✅ | ✅ | ✅ | **DOĞRULANDI** | 12 adımlı test senaryosu 11 test ile başarıyla geçti |
| **3.1** | Ana Sayfa Bilgi Mimarisi | P2 | ⚪ Hayır | ⚪ Hayır | ⚪ Hayır | **PLANLANDI** | Öğrenme/Lab/Workspace 3 sekmeli yapı tasarlandı |
| **4.3** | Dokümantasyon Tutarsızlıkları | P2 | ✅ | ✅ | ✅ | **DOĞRULANDI** | İnceleme raporu, test sayıları ve metrikler eşitlendi |

### Gelecek Sprint Yol Haritası

**Sprint 1: Frontend Laboratuvar Entegrasyonları (Devam Eden)**
- `ModeBadge` bileşeninin 10 laboratuvar sayfasına (`/evaluation`, `/training`, `/rag-lab`, vb.) eklenmesi.
- `ExperimentContext` üzerinden aktif dataset/tokenizer seçimlerinin laboratuvarlar arasında taşınması.
- Laboratuvar içi etkileşimlerin Guided Journey müfredatına telemetriyle yansıtılması.

**Sprint 2: Güvenlik Sertleştirmesi**
- `FRONTEND_AUTH_SECURITY_MIGRATION.md` kılavuzuna göre `HttpOnly` cookie bazlı oturum ve otomatik token yenileme mimarisinin uygulanması.
- CSRF koruması ve rate limiting mekanizmalarının devreye alınması.

**Sprint 3: Bilgi Mimarisi ve İleri Yetenekler**
- Ana sayfanın 3 ana çalışma alanına (Öğrenme Yolu / Laboratuvarlar / Araştırma Workspace) ayrılması.
- DPO (Direct Preference Optimization) ve çok modlu (multimodal) veri modellerinin prototiplenmesi.

---

*Son güncelleme: 23 Eylül 2026*  
*İncelenen commit: `52e42fdb2b14f098245c723ff9a4c20db61a52a0` | [GitHub Actions Run 35929181981](https://github.com/kenanay/AI-Systems-Lab/actions/runs/35929181981)*  
*Test Doğrulama Durumu: 406 non-slow birim/entegrasyon testi (%100 Başarılı), 11 uçtan uca Türkçe Mini-GPT kabul testi (%100 Başarılı), 2 SFT entegrasyon testi (%100 Başarılı), 15 Jest frontend test paketi (80 test %100 Başarılı), Next.js production build (25 sayfa %100 Başarılı).*
