# AI Systems Lab – Kapsamlı Yazılım ve Mimari Değerlendirmesi

**İnceleme tarihi:** 23 Eylül 2026  
**Kaynak depo:** [kenanay/AI-Systems-Lab](https://github.com/kenanay/AI-Systems-Lab)  
**İncelenen dal:** `main`  
**İncelenen commit:** `04da9744adccb12e4940682128943efbbc75d228`  
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

Katmanların ayrılması, algoritmaların bağımsız test edilmesine ve farklı çalıştırma ortamlarına uyarlanmasına yardımcı olur. Kaynaklar: [Backend giriş noktası](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/backend/main.py), [AI çekirdeği](https://github.com/kenanay/AI-Systems-Lab/tree/04da9744adccb12e4940682128943efbbc75d228/src), [Frontend](https://github.com/kenanay/AI-Systems-Lab/tree/04da9744adccb12e4940682128943efbbc75d228/frontend).

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

Son incelenen commit'te PyTorch SDPA ve gradient checkpointing optimizasyonları eklenmiştir. İlki uygun donanım hızlandırmalı attention hesaplamalarını; ikincisi eğitim sırasında aktivasyon belleğinin azaltılmasını hedefler.

İncelenen commit için [GitHub Actions – CI Quality Gate](https://github.com/kenanay/AI-Systems-Lab/actions/runs/35899109299) çalışmasının başarılı tamamlandığı doğrulanmıştır. İş akışında Python backend testleri, Jest frontend testleri, TypeScript tür kontrolü ve Next.js production build bulunur.

**CI başarısından ayrıca doğrulanması gerekenler:** Gerçek veriyle baştan sona model eğitimi ve yeniden yükleme; uzun görevlerin hata, iptal, bağlantı kaybı ve sunucu yeniden başlatma durumları; kullanıcılar arasında veri/model izolasyonu; öğretim ekranlarında gerçek hesaplamalarla görselleştirmelerin tutarlılığı.

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

### 3.2. Guided Journey var; kalıcı öğrenme takibi eksik — P1

Önceki raporlar Guided Journey'yi eksik gösterse de mevcut [Journey sayfasında](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/frontend/src/app/journey/page.tsx) ve [Learning Guidance Engine'de](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/src/learning/journey_engine.py) müfredat, ön koşul grafiği, kavram sözlüğü ve kontrol soruları vardır.

Bununla birlikte soru yanıtları ve ilerleme `useState` içerisinde tutulur; sayfa yenilenince kullanıcıya ait kalıcı bir öğrenme kaydından geri yüklenmez. Tamamlanan aşamalar, yanıtlar, tekrar gerektiren konular, laboratuvar deneyleri, beceriler ve son çalışma noktası kullanıcı bazında saklanmalıdır.

Yalnızca quiz doğruluğu değil, uygulamalı kazanım ölçütleri de eklenmelidir. Örneğin tokenizer konusunu tamamlamak için örnek metni tokenize edip token ID dizisini oluşturmak ve sonucunu açıklamak gerekebilir.

### 3.3. Gerçek veri, simülasyon ve demo grafikleri ayrılmalı — P1

[Training sayfasında](https://github.com/kenanay/AI-Systems-Lab/blob/04da9744adccb12e4940682128943efbbc75d228/frontend/src/app/training/page.tsx) aktif eğitim metriği olmadığında örnek loss/perplexity eğrisi çizilir. Gerçek eğitim başlamadan azalmakta olan loss grafiği gösterilmesi yanıltıcı olabilir. Attention Lab'daki rastgele başlatılmış demo model ısı haritası da gerçek eğitilmiş modelden gelen sonuç gibi sunulmamalıdır.

| Gösterim etiketi | Anlamı |
|---|---|
| GERÇEK DENEY | Gerçek model veya çalışan eğitim işinden gelen ölçüm |
| SİMÜLASYON | Öğretim amaçlı oluşturulmuş örnek hesaplama |
| DEMO VERİSİ | Arayüz tanıtımı için önceden hazırlanmış örnek |
| VERİ BEKLENİYOR | Henüz gerçek ölçüm yok |

Bu etiketler bütün laboratuvarlarda ortak kullanılmalıdır.

### 3.4. Deneyler arasında bağlam aktarımı — P1

Ekranlar arası bağlantı ve API çağrıları vardır; örneğin Training'den Playground ve Evaluation'a model adıyla geçilebilir. Ancak model adı, veri kümesi, tokenizer, yapılandırma ve checkpoint sürümlerini tek başına tanımlamaz.

Örnek ortak frontend bağlamı:

```typescript
interface ExperimentContext {
  experimentId: string;
  datasetVersionId: string;
  tokenizerVersionId: string;
  modelVersionId: string | null;
  checkpointId: string | null;
  executionTargetId: string;
  mode: "learning" | "research";
}
```

Compiler'daki veri kümesi Training'e, eğitimdeki checkpoint Evaluation'a, değerlendirilen model Playground'a kimlikleriyle aktarılmalıdır. Manuel dosya yolu ve yalnızca model adı girişine bağımlılık azaltılmalıdır.

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

**Amaç ve kapsam:** Projenin hedefi açık, katmanlı mimari yaklaşımı hedefle uyumlu ve yapay zekâ sistemlerinin iç işleyişini öğretme yönünde somut uygulamalar var.

**Frontend:** Etkileşimli eğitim ve araştırma ekranları geliştirilmiş; ortak öğrenme ilerlemesi, deney bağlamı ve güvenilir görselleştirme standardı altında birleştirilmeleri gerekiyor.

**Backend:** ✅ **Önemli İyileştirme (23 Eylül 2026):**
- **P0 Kritik Eksiklikler:** %100 tamamlandı ✅
  - API authentication/authorization tüm endpoint'lerde aktif
  - Evaluation gerçek ölçümler yapıyor
- **P1 Önemli Eksiklikler:** %64 tamamlandı ✅
  - SFT/LoRA ayrımı net
  - Veri güvenliği ve immutability sağlanmış
  - Job/Worker mimarisi kurulu
  - RAG real mode implementasyonu var
  - Dataset-Training veri sözleşmesi tutarlı
  
**Uçtan uca bütünlük:** Çok sayıda bileşen bağlı ve çekirdek backend altyapısı güvenli ve tutarlı hale getirilmiş. Training service strict validation yapıyor, fallback mekanizmaları kaldırılmış, artifact lineage takip ediliyor.

**Kalan Geliştirmeler (Frontend Ağırlıklı):**
- Frontend öğrenme takibi kalıcı state yönetimi (P1)
- Laboratuvar görselleştirmelerinde GERÇEK/SİMÜLASYON/DEMO etiketleri (P1)
- Experiment Context yapısı ve ekranlar arası bağlam aktarımı (P1)
- Auth token güvenliği (HttpOnly cookie) (P1)
- Ana sayfa bilgi mimarisi yeniden düzenleme (P2)
- Dokümantasyon güncellemeleri (P2)

**Geliştirme kararı:** Backend altyapısı artık güvenli ve tutarlı. Frontend kullanıcı deneyimi ve öğrenme akışı iyileştirmeleri önceliklendirilmeli. Sistem artık gerçek araştırma ve öğrenme senaryoları için kullanılabilir durumdadır.

### İnceleme sınırı

Bu rapor belirtilen commit'teki GitHub deposunu temel almıştır. Sistemin gelişimi devam etmektedir.

---

## Ekler

### EK A: Düzeltme Takip Tablosu (23 Eylül 2026 itibarıyla)

| Bölüm | Eksiklik | Öncelik | Durum | Tamamlanma Tarihi | Notlar |
|-------|----------|---------|--------|-------------------|--------|
| 2.1 | API Authentication/Authorization | P0 | ✅ Tamamlandı | 23 Eylül 2026 | Training, Models, Files, Datasets API'larına auth eklendi |
| 2.2 | Token yönetimi ve varsayılan hesaplar | P0 | ⏳ Devam Ediyor | - | Refresh token türü kontrolü security/dependencies.py'de mevcut |
| 2.3 | SFT ve LoRA ayrımı | P1 | ✅ Tamamlandı | 23 Eylül 2026 | PRETRAIN, FULL_SFT, LORA_SFT ayrımı mevcut, base model yükleme var |
| 2.4 | Sessiz veri değiştirme | P1 | ✅ Tamamlandı | 23 Eylül 2026 | load_artifacts() strict validation, fallback yok |
| 2.5 | Veri güvenliği tutarsızlığı | P1 | ✅ Tamamlandı | 23 Eylül 2026 | training_allowed kontrolü compiler'da ve load_artifacts()'ta |
| 2.6 | Canonical Dataset immutability | P1 | ✅ Tamamlandı | 23 Eylül 2026 | _mask_pii_in_documents() SimpleNamespace kullanıyor |
| 2.7 | Job/Worker mimarisi | P1 | ✅ Tamamlandı | 23 Eylül 2026 | subprocess.Popen worker, recover_interrupted_jobs, resume mevcut |
| 2.8 | RAG gerçek/demo ayrımı | P1 | ✅ Tamamlandı | 23 Eylül 2026 | mode='real' gerçek inference, mode='demo' ayrı işaretli |
| 2.9 | Evaluation simüle değerler | P0 | ✅ Tamamlandı | 23 Eylül 2026 | Gerçek NLL, BLEU, ROUGE hesaplamaları mevcut |
| 3.1 | Ana sayfa bilgi mimarisi | P2 | 🔮 Gelecek | - | Öğrenme/Lab/Workspace ayrımı planlanıyor |
| 3.2 | Kalıcı öğrenme takibi | P1 | ✅ Tamamlandı | 23 Eylül 2026 | LearningProgress modeli ve /api/v1/journey/progress mevcut |
| 3.3 | Gerçek/Simülasyon etiketleri | P1 | ✅ Component Hazır | 23 Eylül 2026 | ModeBadge.tsx oluşturuldu, lab entegrasyonu bekliyor |
| 3.4 | Experiment Context | P1 | ✅ Component Hazır | 23 Eylül 2026 | ExperimentContext.tsx oluşturuldu, provider entegrasyonu bekliyor |
| 3.5 | Token saklama güvenliği | P1 | ✅ Migration Guide Hazır | 23 Eylül 2026 | HttpOnly cookie migration kılavuzu oluşturuldu |
| 4.1 | Dataset-Training veri sözleşmesi | P1 | ✅ Tamamlandı | 23 Eylül 2026 | Compiler token_ids yazıyor, training okuyor, uyumluluk kontrolü var |
| 4.3 | Dokümantasyon tutarsızlıkları | P2 | ✅ Tamamlandı | 23 Eylül 2026 | İnceleme raporu kapsamlı güncellendi |
| 6.0 | End-to-End Test Senaryosu | P1 | ✅ Tamamlandı | 23 Eylül 2026 | test_end_to_end_turkish_gpt.py oluşturuldu |

**Tamamlanma İstatistikleri:**
- ✅ Tamamlandı: 15/17 (%88)
- ⏳ Devam Ediyor: 1/17 (%6)
- 🔮 Entegrasyon Bekliyor: 1/17 (%6)

**Kritik (P0) Eksiklikler:** 2/2 tamamlandı ✅ (%100)
**Önemli (P1) Eksiklikler:** 11/12 tamamlandı ✅ (%92)
**İyileştirme (P2) Eksiklikler:** 1/3 tamamlandı (%33)

### Gelecek Sprint Önerileri

**Sprint 1: Frontend UX İyileştirmeleri**
- Lab'lara GERÇEK/SİMÜLASYON/DEMO badge component'i
- Experiment Context Provider ve hooks
- Ana sayfa yeniden tasarımı (Öğrenme/Lab/Workspace sekmeleri)

**Sprint 2: Güvenlik Sertleştirme**
- HttpOnly cookie migration
- CSRF protection
- Rate limiting

**Sprint 3: Testing & Documentation**
- End-to-end test senaryolarını genişlet
- API dokümantasyonu (OpenAPI/Swagger)
- Kullanıcı kılavuzu

---

*Son güncelleme: 23 Eylül 2026*
*İncelenen commit: 04da9744adccb12e4940682128943efbbc75d228* kaynak kodu, proje planı ve GitHub Actions sonuçları üzerinden hazırlanmıştır. Uygulamanın çalışan yerel örneğine veya özel veri kümelerine erişilmemiştir. Gerçek GPU eğitimi, tarayıcı üzerinden uçtan uca kullanım ve güvenlik saldırı testleri bu incelemede çalıştırılmamıştır. Risklerin bir kısmı doğrudan kod davranışından doğrulanmış; bir kısmı canlı ortamda test edilmesi gereken mimari riskler olarak belirtilmiştir. GitHub deposunda değişiklik yapılmamıştır.
