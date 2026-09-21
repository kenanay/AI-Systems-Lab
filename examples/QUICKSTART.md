# 🚀 Quick Start Guide - E2E Demo

5 dakikada AI Research Lab pipeline'ını çalıştırın!

## 1️⃣ Kurulum (1 dakika)

```bash
# Proje dizinine git
cd "AI Systems Lab"

# Dependencies kontrol et
python3 -c "import torch, sentencepiece, scipy; print('✅ Ready!')"
```

Hata alırsanız:
```bash
pip install torch sentencepiece scipy
```

## 2️⃣ İlk Demo (3 dakika)

```bash
# E2E demo çalıştır
python3 examples/end_to_end_demo_simple.py
```

**Beklenen çıktı:**
```
================================================================================
AI RESEARCH LAB - END-TO-END DEMO (SIMPLIFIED)
================================================================================
[1/9] Tokenizer Training...
✓ Tokenizer: 300 vocab
[2/9] Dataset Preparation...
✓ Dataset: 160 train, 40 val
[3/9] Model Initialization...
✓ Model: 836,140 params
[4/9] Training (3 epochs)...
Epoch 1/3
  Train loss: 1.0372
  Val perplexity: 1.2443
...
✓ Training complete
[5/9] Evaluation & Benchmarks...
✓ Benchmark: BLEU-4=0.00
[6/9] Model Comparison...
✓ Winner: Final Model
[7/9] Model Registry...
✓ Registered: gpt-demo v1.0.0
[8/9] Inference Testing...
  'Türkiye' → 'Türkiye Cumhuriyeti Anadolu yarımadasında yer alır.'
✓ Inference complete
[9/9] Dashboard Generation...
✓ Dashboard: demo_output/dashboard.html
================================================================================
DEMO COMPLETED SUCCESSFULLY!
================================================================================
Time: 2.9s
```

## 3️⃣ Sonuçları İncele (1 dakika)

```bash
# Dashboard'u tarayıcıda aç
open demo_output/dashboard.html

# Veya manuel olarak:
# 1. Finder'da demo_output/ klasörünü aç
# 2. dashboard.html dosyasına çift tıkla
```

**Dashboard içeriği:**
- 📊 Training metrics (loss, perplexity, accuracy)
- 📈 Quality tracking (regression detection)
- 🏆 Model comparison
- 📝 Benchmark results (BLEU/ROUGE/ChrF)

## ✅ Başarı Kontrol

Demo başarılı olduysa:
- ✅ Training loss: ~1.0 → ~0.08
- ✅ Perplexity: ~1.24 → ~1.06
- ✅ Accuracy: ~97%
- ✅ `demo_output/dashboard.html` var
- ✅ `demo_output/checkpoints/` 3 checkpoint var
- ✅ `demo_output/registry/models/gpt-demo/1.0.0/` model kayıtlı

## 🧪 Integration Test (İsteğe bağlı)

```bash
# Tüm bileşenleri test et
python3 examples/test_integration.py
```

**Beklenen:**
```
Total Tests: 9
Passed: 9
Failed: 0
✅ ALL INTEGRATION TESTS PASSED
```

## 📂 Çıktı Yapısı

```
demo_output/
├── dashboard.html              # 📊 Ana dashboard
├── comparison.json             # 🔬 Model karşılaştırma
├── tokenizer/
│   ├── tokenizer.model         # 🔤 SentencePiece model
│   ├── tokenizer.vocab         # 📖 Vocabulary
│   └── tokenizer.config.json   # ⚙️  Config
├── checkpoints/
│   ├── gpt-demo_v0.1.1_epoch1_step20.pt   # 💾 Epoch 1
│   ├── gpt-demo_v0.1.2_epoch2_step40.pt   # 💾 Epoch 2
│   ├── gpt-demo_v0.1.3_epoch3_step60.pt   # 💾 Epoch 3 (son)
│   └── metadata/
│       └── *.json              # 📋 Checkpoint metadata
└── registry/
    ├── index.json              # 📇 Model registry index
    ├── models/gpt-demo/1.0.0/
    │   ├── model.pt            # 🎯 Registered model
    │   ├── tokenizer.model     # 🔤 Tokenizer kopyası
    │   └── tokenizer.vocab
    └── metadata/
        └── gpt-demo_1.0.0.json # 📄 Model metadata
```

## 🎯 Sonraki Adımlar

1. **README okuyun:** `examples/README.md` - Detaylı kullanım kılavuzu
2. **Kodu inceleyin:** `examples/end_to_end_demo_simple.py` - 267 satır açıklamalı kod
3. **Kendi modelinizi eğitin:** Config'i değiştirin ve tekrar çalıştırın

## 🔧 Özelleştirme

### Hızlı config değişiklikleri:

```python
# end_to_end_demo_simple.py içinde

# Model boyutunu değiştir
model_config = GPTConfig(
    vocab_size=300,
    max_seq_len=64,      # 32'den 64'e (daha uzun cümleler)
    d_model=256,         # 128'den 256'ya (daha büyük model)
    n_layers=6,          # 4'ten 6'ya (daha derin)
    n_heads=8            # 4'ten 8'e (daha fazla attention head)
)

# Training epoch sayısını artır
num_epochs = 10  # 3'ten 10'a

# Batch size'ı değiştir
train_loader = DataLoader(train_dataset, batch_size=16)  # 8'den 16'ya
```

## ❓ Sorun mu var?

### "Vocabulary size too high" hatası
```bash
# Vocab size'ı küçült
vocab_size=100  # 300 yerine
```

### "CUDA out of memory" hatası
```bash
# CPU'da çalışıyor, sorun değil
# GPU varsa batch size'ı küçült
batch_size=4  # 8 yerine
```

### Demo çalışmıyor
```bash
# Clean start
rm -rf demo_output
python3 examples/end_to_end_demo_simple.py
```

## 📚 Daha Fazla

- **Detaylı Docs:** `examples/README.md`
- **Sorun Giderme:** `examples/README.md#sorun-giderme`
- **API Referans:** `examples/README.md#api-referansı`

---

**Toplam süre:** ~5 dakika  
**Başarı oranı:** 100% (9/9 test geçiyor)  
**Destek:** İssue açın veya `examples/README.md` kontrol edin
