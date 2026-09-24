# RAG ve Evaluation

## RAG Lab (`/rag-lab`)

Retrieval-Augmented Generation, yanıt üretmeden önce dış bilgi kaynağından ilgili parçaları bulup modele bağlam olarak verme desenidir.

### Bileşenler

1. **Chunk:** Dokümanın arama birimlerine bölünmesi.
2. **Embedding/index:** Chunk'ların vektör temsili ve collection'a yazılması.
3. **Search:** Sorguya en yakın parçaların alınması.
4. **Query:** Retrieved context ile generation isteğinin birleştirilmesi.

API akışı:

```text
POST /api/v1/rag/chunk
POST /api/v1/rag/index
GET  /api/v1/rag/collections
POST /api/v1/rag/search
POST /api/v1/rag/query
```

### Kullanım ve ipuçları

- Chunk boyutunu konu bütünlüğünü bozmayacak kadar küçük, retrieval gürültüsünü artırmayacak kadar sınırlı seçin.
- Index'e giren document id ve collection adını kaydedin.
- Search sonucunu generation'dan önce tek başına inceleyin.
- Kaynak dönmeyen bir yanıtı “RAG başarılı” kabul etmeyin.
- Embedder tipi, model adı, top-k ve score threshold'ı deney metadata'sına yazın.

## Evaluation Lab (`/evaluation`)

Evaluation; bir modelin belirli bir görevde, belirli bir örnek seti ve ölçüm kuralı altında nasıl davrandığını ölçer. Genel “model iyi” iddiası değildir.

### İş akışı

1. Benchmark veya örnek prompt setini seçin.
2. Model version ve generation parametrelerini sabitleyin.
3. Inspect-text ile tekil örneklerin metriklerini anlayın.
4. Benchmark run başlatın.
5. Sonuçları model/version bazında saklayın.
6. Compare ve radar görünümleriyle güçlü/zayıf alanları karşılaştırın.
7. Hata örneklerini ham çıktıyla birlikte raporlayın.

### Metrik yorumlama

- Aynı metrik farklı veri dağılımlarında farklı anlam taşır.
- Ortalama skor hata dağılımını gizleyebilir.
- Küçük benchmark istatistiksel olarak kırılgan olabilir.
- Simüle edilen veya demo amaçlı sonuçlar gerçek model ölçümü gibi raporlanmamalıdır.

## Embedding Lab ile farkı

Embedding Lab bir vektör uzayını öğretir ve görselleştirir. RAG embedding/index akışı retrieval pipeline'ının bir parçasıdır. Benzerlik grafiği görmek tek başına üretim RAG kalitesi kanıtı değildir.
