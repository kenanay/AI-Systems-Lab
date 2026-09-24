# AI Systems Lab Dokümantasyonu

Bu dizin, AI Systems Lab'ın güncel kullanıcı kılavuzunu, mimarisini, teknik
sözleşmelerini ve geliştirme rehberini içerir. Uygulama; veriyi içeri almaktan
model değerlendirmeye ve öğrenme laboratuvarlarına kadar local-first bir AI
sistemleri araştırma ortamı olarak tasarlanmıştır.

## Nereden başlamalı?

- Yeni kullanıcı: [Kullanıcı Kılavuzu](user-guide.md)
- Projeyi anlamak isteyen: [Genel Bakış](overview.md) ve [Mimari](architecture.md)
- Uçtan uca deneme yapmak isteyen: [Senaryolar](workflows.md)
- Özellikleri ayrıntılı incelemek isteyen: [Özellik Rehberleri](features/)
- API ile entegrasyon yapmak isteyen: [API Haritası](development/api-map.md) ve
  çalışan [OpenAPI arayüzü](http://localhost:8000/docs)
- Katkı vermek isteyen: [Katkı Rehberi](CONTRIBUTING.md)

## Doküman haritası

### Ürün ve mimari

- [Genel Bakış, felsefe ve hedefler](overview.md)
- [Sistem mimarisi ve veri akışları](architecture.md)
- [Uçtan uca kullanım senaryoları](workflows.md)

### Özellikler

- [Veri hattı: upload, ingestion, canonical dataset ve compiler](features/data-pipeline.md)
- [Tokenizer, training, checkpoint, model ve inference](features/model-lifecycle.md)
- [Öğrenme yolu ve laboratuvarlar](features/learning-labs.md)
- [RAG, evaluation ve sonuçların yorumlanması](features/rag-and-evaluation.md)

### İşletim ve güvenlik

- [Kurulum, çalıştırma ve konfigürasyon](operations/configuration.md)
- [Güvenli sunucu kurulumu ve deployment](operations/deployment.md)
- [Güvenlik ve veri yönetişimi](operations/security.md)
- [Sorun giderme](operations/troubleshooting.md)

### Geliştirme ve referans

- [Backend API haritası](development/api-map.md)
- [Veri sözleşmeleri](reference/data-contracts.md)
- [Ön bilgi haritası](reference/prerequisites.md)
- [Terimler sözlüğü](reference/glossary.md)
- [Tarihsel raporların kullanım rehberi](reference/historical-reports.md)
- [Katkı rehberi](CONTRIBUTING.md)

## Kaynak ilkeleri

Uygulamanın davranışını anlatırken öncelik sırası şöyledir:

1. Çalışan kod ve API sözleşmeleri (`frontend/`, `backend/`, `src/`)
2. Bu dizindeki güncel kullanım ve mimari dokümanları
3. Proje ilkeleri (`.kiro/steering/`)
4. Tarihsel inceleme ve sprint raporları

Kök dizindeki tarihli raporlar karar geçmişini ve önceki eksiklikleri saklar;
güncel ürün davranışının tek başına referansı değildir. Bir rapor ile güncel
kod arasında fark varsa [Genel Bakış](overview.md) ve ilgili özellik rehberi
esas alınmalıdır.

## Sürüm notu

Bu dokümantasyon, mevcut beta uygulamasındaki frontend rotaları, FastAPI
router'ları ve cross-platform launcher'lar temel alınarak hazırlanmıştır. Yeni
bir özellik eklendiğinde aynı değişiklik içinde ilgili özellik sayfası, API
haritası ve gerekiyorsa veri sözleşmesi güncellenmelidir.
