# Kullanıcı Kılavuzu

## 1. Kurulum

### macOS/Linux

```bash
./scripts/setup_dev.sh
./scripts/start_app.sh start
```

### Windows

PowerShell veya Komut İstemi:

```powershell
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
cd frontend
npm install
cd ..
scripts\start_app.cmd start
```

## 2. Uygulamayı başlatma

Varsayılan adresler:

- Frontend: `http://127.0.0.1:3000`
- Backend: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

```bash
# macOS/Linux
./scripts/start_app.sh status
./scripts/start_app.sh stop
./scripts/start_app.sh restart
./scripts/start_app.sh start --backend-port 8100 --frontend-port 3100
./scripts/start_app.sh start --strict-ports
```

```cmd
:: Windows
scripts\start_app.cmd start
scripts\start_app.cmd status
scripts\start_app.cmd stop
scripts\start_app.cmd start -BackendPort 8100 -FrontendPort 3100
scripts\start_app.cmd start -StrictPorts
```

Port doluysa launcher mevcut süreci durdurmaz; bir sonraki boş portu seçer ve
frontend'in API adresini seçilen backend portuna bağlar.

## 3. İlk kullanım: önerilen sıra

### A. Kavramsal başlangıç

1. `/journey` üzerinden öğrenme aşamasını seçin.
2. Math, Tensor, NN, Embedding, Attention ve Transformer laboratuvarları ile
   gerekli kavramları küçük örneklerde inceleyin.
3. Her laboratuvarda girdiyi değiştirip şekil, değer, maliyet ve açıklama
   çıktısını karşılaştırın.

### B. Gerçek veri hattı

1. `/upload` ile `.txt`, `.md` veya desteklenen `.pdf` dosyasını yükleyin.
2. Metadata'da kaynak, lisans, güvenlik seviyesi ve training iznini doğrulayın.
3. Dosyayı process edin; oluşan `DocumentRecord` kaydını
   `/dataset-explorer` içinde inceleyin.
4. PII ve kalite uyarılarını çözmeden dataset'i training'e taşımayın.
5. `/dataset-compiler` ile seçili dokümanlardan version'lı canonical Parquet
   dataset üretin.

### C. Tokenizer ve model

1. `/tokenizer` ile dataset'e uygun tokenizer eğitin veya mevcut tokenizer'ı seçin.
2. `/training` içinde pretraining, SFT veya LoRA amacını açıkça seçin.
3. Dataset version, tokenizer, context length, batch size, learning rate ve
   checkpoint ayarlarını kaydedin.
4. Job durumunu takip edin; hata durumunda config ve veri sözleşmesini inceleyin.
5. `/models` içinde model hash ve sürüm bilgisini doğrulayın.
6. `/playground` ile inference, sampling ve token/attention ayrıntılarını inceleyin.
7. `/evaluation` ile benchmark çalıştırıp modeli karşılaştırın.

## 4. Navigasyonun anlamı

Navbar üç ana bağlam sunar:

- **Veri Hattı:** Kaynak dosyadan canonical dataset'e kadar.
- **Model & Eğitim:** Tokenizer, training, model artifact, inference ve ölçüm.
- **Laboratuvarlar:** Kavramları düşük maliyetli deneylerle öğrenme.

Bir sayfada kaybolursanız “hangi girdiyi alıyor, hangi kaydı üretiyor, sonraki
aşama ne?” sorularıyla ilerleyin. Platformun temel zihinsel modeli sayfa listesi
değil, veri ve deney yaşam döngüsüdür.

## 5. Hesap ve profil

- `/register`: Yeni kullanıcı hesabı oluşturma.
- `/login`: Cookie tabanlı oturum açma.
- `/profile`: Kullanıcı bilgileri, API key ve yetkili hesaplarda kullanıcı yönetimi.

**Kullanım senaryosu:** Önce kişisel hesapla giriş yapın, API key gerekiyorsa
profile ekranından oluşturun, anahtarı yalnızca güvenli local secret alanında
tutun. API key'i commit, log veya prompt içine koymayın.

**İpucu:** Oturum yenileme başarısızsa önce cookie, backend adresi ve browser
origin'ini kontrol edin; doğrudan yeni key üretmek sorunun kaynağını çözmeyebilir.

## 6. İpucu ve iyi uygulamalar

- Küçük ve bilinen bir örnekle başlayın; büyük dataset'i ilk denemede kullanmayın.
- Her deneyde dataset version, tokenizer, model config ve seed'i not edin.
- Training'e izin verilmeyen veya PII içeren dosyaları açıkça işaretleyin.
- Model çıktısını tek bir örnekle değerlendirmeyin; benchmark ve hata örnekleri
  birlikte incelenmelidir.
- Gerçek ölçüm, simülasyon ve demo çıktılarının etiketini kontrol edin.
- SQLite local geliştirme içindir; çok worker'lı production için uygun veri
  tabanı ve lock stratejisi kurun.
