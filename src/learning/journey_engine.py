"""
src/learning/journey_engine.py

Pedagogical Learning Guidance Engine & Knowledge Graph Subsystem
Local-First AI Research Lab - Section 56 Architecture
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class MilestoneQuestion(BaseModel):
    """Aşama sonu interaktif kontrol sorusu."""
    id: str
    stage_id: str
    question: str
    options: List[str]
    correct_index: int
    explanation: str
    math_intuition: Optional[str] = None


class JourneyStage(BaseModel):
    """Rehberli öğrenme yolculuğu aşaması."""
    id: str
    order: int
    title: str
    category: str
    summary: str
    concepts: List[str]
    objectives: List[str]
    prerequisites: List[str]
    lab_url: Optional[str] = None
    lab_title: Optional[str] = None
    questions: List[MilestoneQuestion]


class KnowledgeNode(BaseModel):
    """Knowledge DAG düğümü."""
    id: str
    label: str
    stage_id: str
    category: str
    level: int
    lab_url: Optional[str] = None


class KnowledgeEdge(BaseModel):
    """Knowledge DAG kenarı (ön koşul / bağımlılık ilişkisi)."""
    id: str
    source: str
    target: str
    label: Optional[str] = None


class GlossaryTerm(BaseModel):
    """AI & LLM Terimler Sözlüğü girdisi."""
    term: str
    category: str
    short_def: str
    detailed_explanation: str
    formula: Optional[str] = None
    related_lab_url: Optional[str] = None
    related_stage_id: Optional[str] = None


class JourneyEngine:
    """
    Öğrenme rehberliği, kavram haritası ve pedagojik değerlendirme motoru.
    """

    def __init__(self):
        self._stages = self._build_stages()
        self._nodes, self._edges = self._build_knowledge_graph()
        self._glossary = self._build_glossary()

    def get_curriculum(self) -> List[Dict[str, Any]]:
        """Tüm sıralı aşamaları ve mini quizleri döner."""
        return [stage.model_dump() for stage in self._stages]

    def get_stage(self, stage_id: str) -> Optional[Dict[str, Any]]:
        """Belirli bir aşamayı döner."""
        for s in self._stages:
            if s.id == stage_id:
                return s.model_dump()
        return None

    def get_knowledge_graph(self) -> Dict[str, Any]:
        """DAG düğüm ve kenarlarını döner."""
        return {
            "nodes": [n.model_dump() for n in self._nodes],
            "edges": [e.model_dump() for e in self._edges],
        }

    def get_glossary(self, search: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Terimler sözlüğünü arama ve kategoriye göre filtreler."""
        terms = self._glossary
        if category and category.lower() != "all":
            terms = [t for t in terms if t.category.lower() == category.lower()]
        if search:
            q = search.lower().strip()
            terms = [
                t for t in terms
                if q in t.term.lower() or q in t.short_def.lower() or q in t.detailed_explanation.lower()
            ]
        return [t.model_dump() for t in terms]

    def check_question(self, question_id: str, selected_option: int) -> Dict[str, Any]:
        """Soru cevabını kontrol eder ve açıklama döner."""
        for stage in self._stages:
            for q in stage.questions:
                if q.id == question_id:
                    is_correct = (selected_option == q.correct_index)
                    return {
                        "question_id": q.id,
                        "is_correct": is_correct,
                        "selected_option": selected_option,
                        "correct_index": q.correct_index,
                        "explanation": q.explanation,
                        "math_intuition": q.math_intuition,
                    }
        return {
            "question_id": question_id,
            "is_correct": False,
            "selected_option": selected_option,
            "correct_index": -1,
            "explanation": "Soru bulunamadı.",
            "math_intuition": None,
        }

    # -------------------------------------------------------------------------
    # Internal Builders
    # -------------------------------------------------------------------------

    def _build_stages(self) -> List[JourneyStage]:
        return [
            JourneyStage(
                id="stage_0",
                order=0,
                title="Aşama 0 — Temel Matematik & Hesaplama İlkeleri",
                category="Matematik & Altyapı",
                summary="Yapay zeka modellerinin kalbinde yer alan lineer cebir, çok değişkenli kalkülüs, olasılık ve CPU/GPU hesaplama mantığını kavrayın.",
                concepts=["Matris Çarpımı", "Gradient Descent", "Zincir Kuralı", "Softmax", "Cross-Entropy", "GPU Paralelliği"],
                objectives=[
                    "Matris çarpımının boyut kurallarını anlamak (M, K) x (K, N) = (M, N)",
                    "Zincir kuralı ile türevlerin geriye doğru nasıl aktığını görmek",
                    "Softmax ile logit değerlerinin olasılık dağılımına dönüşümünü incelemek"
                ],
                prerequisites=[],
                lab_url="/tensor-lab",
                lab_title="Tensor Lab (Aktivasyon & Autograd)",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_0_1",
                        stage_id="stage_0",
                        question="A boyutu (32, 128) ve B boyutu (128, 64) olan iki tensör çarpıldığında sonuç tensörünün boyutu ne olur?",
                        options=["(32, 128)", "(32, 64)", "(128, 128)", "(64, 32)"],
                        correct_index=1,
                        explanation="Matris çarpımında iç boyutlar (128) birbirini eşler, sonuç dış boyutlar olan (32, 64) tensörünü üretir.",
                        math_intuition="(M, K) \\times (K, N) \\to (M, N)"
                    ),
                    MilestoneQuestion(
                        id="q_stage_0_2",
                        stage_id="stage_0",
                        question="Softmax fonksiyonu logit vektörüne uygulandığında hangi iki temel özelliği garanti eder?",
                        options=[
                            "Değerler sıfırdan küçük olabilir ve toplamları 0'dır",
                            "Tüm elemanlar (0, 1) aralığında pozitif olur ve toplamları tam olarak 1'dir",
                            "En büyük değer her zaman 1, diğerleri 0 olur",
                            "Vektörün boyutunu yarıya indirir"
                        ],
                        correct_index=1,
                        explanation="Softmax üssel fonksiyon (exp) kullanarak negatif logitleri pozitife çevirir ve toplama bölerek geçerli bir olasılık dağılımı (toplam = 1) oluşturur.",
                        math_intuition="P(y_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}"
                    )
                ]
            ),
            JourneyStage(
                id="stage_1",
                order=1,
                title="Aşama 1 — Ham Veri & Ayrıştırma (Data Ingestion)",
                category="Veri Mühendisliği",
                summary="Gerçek dünya veri kaynaklarından (PDF, TXT, DOCX, CSV, XLSX) ham metinleri çıkarın, metadata üretin ve SHA-256 ile bütünlüğü doğrulayın.",
                concepts=["Dosya Formatları", "Text Extraction", "Encoding (UTF-8)", "Metadata Extraction", "SHA-256 Checksum"],
                objectives=[
                    "Farklı dosya biçimlerinden saf metin akışını ayıklamak",
                    "Karakter kodlama (UTF-8/ASCII) bozulmalarını engellemek",
                    "Her veri kaynağını benzersiz SHA-256 parmak iziyle etiketlemek"
                ],
                prerequisites=["stage_0"],
                lab_url="/upload",
                lab_title="Upload & Data Explorer",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_1_1",
                        stage_id="stage_1",
                        question="Bir veri alma (ingestion) hattında SHA-256 hash'i hesaplamanın temel amacı nedir?",
                        options=[
                            "Metnin içeriğini sıkıştırmak",
                            "Dosya içeriğinde tek bir bayt dahi değiştiğinde bunu tespit edip tekrarlanabilirliği ve bütünlüğü güvenceye almak",
                            "Metni otomatik olarak Türkçe'ye çevirmek",
                            "Veriyi vektör embedding uzayına aktarmak"
                        ],
                        correct_index=1,
                        explanation="Kriptografik özet fonksiyonu SHA-256, veri dosyalarının bütünlüğünü (integrity) ve eğitim setlerinin deterministik tekrarlanabilirliğini garanti eder.",
                        math_intuition="H(x) \\neq H(x') \\text{ even if 1 bit differs}"
                    ),
                    MilestoneQuestion(
                        id="q_stage_1_2",
                        stage_id="stage_1",
                        question="Metin çıkarımında UTF-8 yerine yanlış bir encoding (ör. ISO-8859-9) kullanılırsa ne olur?",
                        options=[
                            "Dosya boyutu iki katına çıkar",
                            "Türkçe özel karakterler (ğ, ş, ı, ö, ç) bozuk karakterlere (mojibake) dönüşür ve tokenizer kelime köklerini parçalayamaz",
                            "Model eğitimi daha hızlı tamamlanır",
                            "Hiçbir etki olmaz"
                        ],
                        correct_index=1,
                        explanation="Mojibake bozulmaları tokenizer'ın aşırı parçalanmasına (over-tokenization) ve modelin o kelimeleri öğrenememesine yol açar.",
                    )
                ]
            ),
            JourneyStage(
                id="stage_2",
                order=2,
                title="Aşama 2 — Canonical Dataset Standardı & Veri Kalitesi",
                category="Veri Mühendisliği",
                summary="Büyük dil modellerinin kalitesi verinin kalitesiyle belirlenir. MinHash ile veri tekilleştirme (dedup), PII maskeleme ve Train/Val/Test bölme uygulayın.",
                concepts=["Parquet Formatı", "MinHash LSH", "PII Anonimleştirme", "Veri Kaçağı (Data Contamination)", "Train/Validation Split"],
                objectives=[
                    "Benzer ve mükerrer metinleri MinHash Jaccard benzerliğiyle tespit etmek",
                    "TCKN, e-posta, telefon gibi PII verilerini maskelemek",
                    "Eğitim ve doğrulama kümeleri arasında veri sızıntısını önlemek"
                ],
                prerequisites=["stage_1"],
                lab_url="/dataset-compiler",
                lab_title="Dataset Compiler & Quality",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_2_1",
                        stage_id="stage_2",
                        question="Eğitim veri setinde mükerrer (duplicate) verilerin bulunması model üzerinde hangi olumsuz etkiyi yaratır?",
                        options=[
                            "Modelin kelime dağarcığını küçültür",
                            "Model tekrarlanan cümleleri ezberler (overfitting), üretimde takılıp kalma (repetition loop) eğilimi gösterir ve genelleme yeteneği düşer",
                            "Modelin bağlam penceresini daraltır",
                            "Loss değerinin sonsuza gitmesine neden olur"
                        ],
                        correct_index=1,
                        explanation="Deduplication yapılmayan veri setleri ezberlemeyi artırır ve test verisindeki perplexity skorlarını sahte şekilde şişirir.",
                    ),
                    MilestoneQuestion(
                        id="q_stage_2_2",
                        stage_id="stage_2",
                        question="MinHash algoritması hangi metrik üzerine yaklaşık bir hesaplama yapar?",
                        options=[
                            "Öklid Mesafesi",
                            "Jaccard Benzerliği (Küme Kesişimi / Birleşimi)",
                            "Manhattan Uzaklığı",
                            "Cosine Similarity"
                        ],
                        correct_index=1,
                        explanation="MinHash n-gram kümeleri arasındaki Jaccard benzerliğini $J(A, B) = |A \\cap B| / |A \\cup B|$ hash permutasyonları üzerinden $O(1)$ sürede yaklaştırır.",
                        math_intuition="P(h(A) = h(B)) = J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}"
                    )
                ]
            ),
            JourneyStage(
                id="stage_3",
                order=3,
                title="Aşama 3 — Tokenizasyon & BPE (Subword Modeling)",
                category="NLP & Tokenizer",
                summary="Metinleri sayılara dönüştürme sanatı. Karakter ve kelime seviyesi tokenizasyonun sınırlarını aşan Byte-Pair Encoding (BPE) algoritmasını adım adım çalıştırın.",
                concepts=["BPE (Byte-Pair Encoding)", "Vocabulary Size", "Merge Rules", "Special Tokens (<bos>, <eos>, <pad>)", "Compression Ratio"],
                objectives=[
                    "BPE frekans sayım ve en sık geçen çiftleri birleştirme mantığını anlamak",
                    "Türkçe sondan eklemeli yapıda subword parçalanmalarını incelemek",
                    "Karakter/Token sıkıştırma oranını (compression ratio) maksimize etmek"
                ],
                prerequisites=["stage_2"],
                lab_url="/tokenizer",
                lab_title="Tokenizer Lab",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_3_1",
                        stage_id="stage_3",
                        question="BPE (Byte-Pair Encoding) algoritması her eğitim adımında hangi işlemi yapar?",
                        options=[
                            "En uzun kelimeyi sözlükten siler",
                            "Metinde en yüksek frekansla yan yana gelen karakter/token ikilisini bularak yeni bir birleşik token olarak sözlüğe ekler",
                            "Kelimeleri rastgele vektörlerle çarpar",
                            "Sadece sesli harfleri sözlüğe kaydeder"
                        ],
                        correct_index=1,
                        explanation="BPE sık geçen ikilileri (`'y', 'a' -> 'ya'`) birleştirerek OOV (Out-Of-Vocabulary) sorununu ortadan kaldıran dinamik bir alt-kelime sözlüğü kurar.",
                    ),
                    MilestoneQuestion(
                        id="q_stage_3_2",
                        stage_id="stage_3",
                        question="Sözlük boyutu (Vocab Size) çok küçük seçilirse (ör. 256) model üzerinde ne tür bir etki gözlenir?",
                        options=[
                            "Eğitim hiç başlayamaz",
                            "Metinler çok fazla küçük parçaya (karaktere) bölünür; aynı metni ifade etmek için çok uzun token dizileri gerekir ve bağlam penceresi çabuk dolar",
                            "Model kelimeleri daha iyi anlar",
                            "VRAM kullanımı sıfıra iner"
                        ],
                        correct_index=1,
                        explanation="Küçük sözlük token dizisini aşırı uzatırken (sequence length burden), devasa sözlükler ise embedding katmanında dev parametre maliyeti doğurur.",
                    )
                ]
            ),
            JourneyStage(
                id="stage_4",
                order=4,
                title="Aşama 4 — Tensörler, MatMul & Autograd Hesaplama Grafı",
                category="Derin Öğrenme Altyapısı",
                summary="Transformer'ın temel hesaplama blokları: N-boyutlu tensörler, yayınlama (broadcasting), matris çarpımı, GELU aktivasyonu ve türevlerin geriye aktarımı (Backprop).",
                concepts=["Tensör Boyutları (B, T, C)", "Broadcasting", "GELU & Swish", "Computational Graph", "Autograd & Backpropagation"],
                objectives=[
                    "Tensör boyut manipülasyonlarını (reshape, transpose, permute) kavramak",
                    "Boyut eşleşmeyen tensörlerin NumPy/PyTorch broadcasting kurallarını görmek",
                    "İleri geçişte hesaplanan ara aktivasyonların geri geçişte nasıl kullanıldığını simüle etmek"
                ],
                prerequisites=["stage_0", "stage_3"],
                lab_url="/tensor-lab",
                lab_title="Tensor Lab (Simülasyon & Hesaplama)",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_4_1",
                        stage_id="stage_4",
                        question="(B, 1, D) boyutundaki bir tensör ile (B, T, D) boyutundaki bir tensör toplanabilir mi?",
                        options=[
                            "Hayır, boyut sayıları aynı olsa da orta boyut farklı olduğu için hata verir",
                            "Evet, NumPy/PyTorch broadcasting kuralına göre 1 olan boyut T kadar çoğaltılarak toplanır",
                            "Sadece tensörler sıfır matrisi ise toplanabilir",
                            "Evet, ancak sonuç (B, 1, D) olur"
                        ],
                        correct_index=1,
                        explanation="Broadcasting kuralında boyutlardan biri 1 ise diğer boyuta otomatik olarak esnetilerek işlem gerçekleştirilir.",
                        math_intuition="(B, 1, D) \\xrightarrow{\\text{broadcast}} (B, T, D)"
                    ),
                    MilestoneQuestion(
                        id="q_stage_4_2",
                        stage_id="stage_4",
                        question="Modern LLM'lerde (GPT-2, LLaMA) geleneksel ReLU yerine neden GELU veya SwiGLU aktivasyon fonksiyonları tercih edilir?",
                        options=[
                            "ReLU negatif değerlerde türevi tamamen sıfır yaptığı için nöron ölümüne yol açabilirken, GELU pürüzsüz (smooth) ve sıfır etrafında küçük negatif gradyanlara izin verir",
                            "GELU hesaplama açısından ReLU'dan 10 kat daha hızlıdır",
                            "GELU parametre sayısını azaltır",
                            "ReLU sadece 2 boyutlu tensörlerde çalışır"
                        ],
                        correct_index=0,
                        explanation="GELU (Gaussian Error Linear Unit) pürüzsüz geçişi ve gradyan akışını koruyarak derin mimarilerde çok daha kararlı yakınsama sağlar.",
                    )
                ]
            ),
            JourneyStage(
                id="stage_5",
                order=5,
                title="Aşama 5 — Gömme Uzayı (Embeddings & Cosine Similarity)",
                category="Temsil Öğrenimi",
                summary="Ayrık token ID'lerini sürekli d-boyutlu anlamsal vektör uzayına taşıyın. Vektör aritmetiği ve Cosine Benzerliği ile semantik yakınlığı ölçün.",
                concepts=["Word Embedding", "Cosine Similarity", "Vector Arithmetic", "High-Dimensional Space", "PCA & t-SNE Projeksiyonu"],
                objectives=[
                    "Token ID'lerinin (Vocab, d_model) ağırlık matrisinden vektör seçtiğini (lookup) görmek",
                    "İki vektör arasındaki açı kosinüsünü $A \\cdot B / (||A|| ||B||)$ hesaplamak",
                    "Çok boyutlu anlamsal uzayı 2D/3D düzleme indirgeyerek görselleştirmek"
                ],
                prerequisites=["stage_3", "stage_4"],
                lab_url="/embedding-lab",
                lab_title="Embedding Lab (2D/3D & Vektör Aritmetiği)",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_5_1",
                        stage_id="stage_5",
                        question="Cosine similarity değeri 1.0 olan iki embedding vektörü ne anlama gelir?",
                        options=[
                            "İki vektör birbirine diktir (ortogonaldir)",
                            "İki vektör uzayda birebir aynı doğrultuyu göstermektedir (aralarındaki açı 0 derecedir)",
                            "İki vektörün büyüklükleri birbirine eşittir",
                            "İki vektör birbirinin zıttıdır"
                        ],
                        correct_index=1,
                        explanation="Cosine similarity vektörlerin boyuna değil aralarındaki açının kosinüsüne bakar. Açı 0 ise $\\cos(0) = 1.0$ olur ve aynı anlamsal doğrultuyu gösterirler.",
                        math_intuition="\\cos(\\theta) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\| \\|\\mathbf{v}\\|}"
                    ),
                    MilestoneQuestion(
                        id="q_stage_5_2",
                        stage_id="stage_5",
                        question="'Kral' - 'Erkek' + 'Kadın' vektör işlemi yapıldığında model embedding uzayında hangi kelimeye en yakın noktaya varır?",
                        options=["Taç", "Kraliçe", "Prens", "Kale"],
                        correct_index=1,
                        explanation="Vektör uzayları doğrusal anlamsal ilişkileri (gender, tense, başkent vb.) temsil eder. Kraldan erillik çıkarılıp dişillik eklendiğinde Kraliçe elde edilir.",
                    )
                ]
            ),
            JourneyStage(
                id="stage_6",
                order=6,
                title="Aşama 6 — Dikkat Mekanizması (Self-Attention & MHA)",
                category="Transformer Çekirdeği",
                summary="Transformer'ın devrim yaratan kalbi. Sorgu (Query), Anahtar (Key) ve Değer (Value) projeksiyonları ile her kelimenin diğer kelimelerle bağlamsal ilişkisini hesaplayın.",
                concepts=["Query, Key, Value (Q, K, V)", "Scaled Dot-Product", "Causal Masking (Look-ahead mask)", "Multi-Head Attention (MHA)", "Attention Rollout"],
                objectives=[
                    "Ölçekleme faktörü $1/\\sqrt{d_k}$'nın gradyan patlamasını nasıl önlediğini anlamak",
                    "Causal maskenin gelecekteki kelimeleri $-\\infty$ ile maskeleyerek hile yapmayı nasıl engellediğini görmek",
                    "Çoklu kafaların farklı anlamsal ve sözdizimsel bağlamları eşzamanlı nasıl öğrendiğini incelemek"
                ],
                prerequisites=["stage_4", "stage_5"],
                lab_url="/attention-lab",
                lab_title="Attention Lab (Isı Haritası & Maskeleme)",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_6_1",
                        stage_id="stage_6",
                        question="Scaled Dot-Product Attention formülünde $Q K^T$ çarpımı neden $\\sqrt{d_k}$ değerine bölünür?",
                        options=[
                            "İşlemi hızlandırmak için",
                            "Yüksek boyutlarda iç çarpım değerleri çok büyüyerek Softmax fonksiyonunu aşırı doygunluğa (saturation) itip gradyanları sıfırlamasın diye",
                            "Parametre sayısını azaltmak için",
                            "Dizilim uzunluğunu normalize etmek için"
                        ],
                        correct_index=1,
                        explanation="Vektör boyutu $d_k$ büyüdükçe rastgele vektörlerin iç çarpım varyansı $d_k$ olur. $\\sqrt{d_k}$'ya bölmek varyansı 1'e çekerek Softmax gradyanlarının yok olmasını önler.",
                        math_intuition="\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V"
                    ),
                    MilestoneQuestion(
                        id="q_stage_6_2",
                        stage_id="stage_6",
                        question="Otoregresif (üretici) dil modellerinde Causal Attention Mask neden üst üçgen matris olarak uygulanır?",
                        options=[
                            "Modelin geçmiş kelimeleri unutmasını sağlamak için",
                            "Model sonraki tokeni tahmin ederken henüz üretilmemiş gelecekteki tokenleri göremesin diye dikkat skorlarına $-\\infty$ eklemek için",
                            "Bellek kullanımını yarıya düşürmek için",
                            "Kelime sırasını tersine çevirmek için"
                        ],
                        correct_index=1,
                        explanation="$\\text{softmax}(-\\infty) = 0$ olduğu için model gelecekteki tokenlere sıfır dikkat ağırlığı verir ve yalnızca geçmişe bakarak üretim yapar.",
                    )
                ]
            ),
            JourneyStage(
                id="stage_7",
                order=7,
                title="Aşama 7 — Modern Transformer Mimarisi (RoPE, GQA, SwiGLU)",
                category="Transformer Çekirdeği",
                summary="LLaMA-3, Mistral ve modern açık kaynak modellerin kullandığı son teknoloji mimari bileşenleri: RoPE konum kodlaması, GQA bellek tasarrufu, RMSNorm ve SwiGLU FFN.",
                concepts=["RoPE (Rotary Position Embedding)", "ALiBi", "GQA (Grouped-Query Attention)", "RMSNorm vs LayerNorm", "Pre-LN vs Post-LN", "SwiGLU"],
                objectives=[
                    "RoPE'nin 2D rotasyon matrisleriyle bağıl mesafe korunumunu anlamak",
                    "GQA'nın KV-Cache bellek ayak izini $4\\times$ nasıl düşürdüğünü hesaplamak",
                    "RMSNorm'un ortalama çıkarma adımını atlayarak hız kazandırdığını görmek"
                ],
                prerequisites=["stage_6"],
                lab_url="/transformer-lab",
                lab_title="Transformer Lab (RoPE, GQA, VRAM)",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_7_1",
                        stage_id="stage_7",
                        question="Grouped-Query Attention (GQA) mimarisinin Standart MHA'ya kıyasla sağladığı en büyük operasyonel avantaj nedir?",
                        options=[
                            "Eğitim veri setini küçültmesi",
                            "Birden fazla Query kafasının aynı Key ve Value kafasını paylaşması sayesinde KV-Cache boyutunu büyük oranda düşürmesi ve çıkarım hızını katlaması",
                            "Tüm model ağırlıklarını 8-bite düşürmesi",
                            "Pozisyonel embedding ihtiyacını ortadan kaldırması"
                        ],
                        correct_index=1,
                        explanation="Örneğin LLaMA-3 8B'de 32 Query kafasına karşılık 8 KV kafası vardır; bu durum KV-Cache belleğinde %75 (4 kat) tasarruf sağlar.",
                    ),
                    MilestoneQuestion(
                        id="q_stage_7_2",
                        stage_id="stage_7",
                        question="RoPE (Rotary Position Embedding) neden mutlak pozisyon vektörlerini toplamak yerine rotasyon matrisleri kullanır?",
                        options=[
                            "Modelin kelime dağarcığını artırmak için",
                            "İki token arasındaki dikkat skorunun yalnızca onların mutlak pozisyonlarına değil, aralarındaki göreli mesafeye $(m - n)$ bağlı olmasını matematiksel olarak güvenceye almak için",
                            "Softmax katmanını kaldırmak için",
                            "Matris çarpımını lineer karmaşıklığa düşürmek için"
                        ],
                        correct_index=1,
                        explanation="RoPE $R_m q$ ve $R_n k$ rotasyonlarını çarparak $(R_m q)^T (R_n k) = q^T R_{n-m} k$ bağıntısını sağlar ve göreli mesafeyi korur.",
                        math_intuition="\\langle R_m q, R_n k \\rangle = q^T R_{n-m} k"
                    )
                ]
            ),
            JourneyStage(
                id="stage_8",
                order=8,
                title="Aşama 8 — Dil Modeli Ön Eğitimi (Pretraining)",
                category="Model Eğitimi",
                summary="Milyarlarca token üzerinde ham metin ile Causal LM eğitimi. Sonraki token tahmini, Cross-Entropy Loss, Gradient Accumulation ve Cosine Annealing LR.",
                concepts=["Causal Language Modeling", "Next-Token Prediction", "Cross-Entropy Loss", "Perplexity", "Gradient Accumulation", "Cosine Learning Rate"],
                objectives=[
                    "Girdinin bir token sağa kaydırılarak hedef (target) dizisinin oluşturulmasını görmek",
                    "Batch boyutunu sanal olarak büyüten Gradient Accumulation mekanizmasını anlamak",
                    "Eğitim kaybı (Loss) ile Perplexity arasındaki üssel ilişkiyi kavramak"
                ],
                prerequisites=["stage_6", "stage_7"],
                lab_url="/training",
                lab_title="Training Lab (Pretrain & Monitor)",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_8_1",
                        stage_id="stage_8",
                        question="Bir dil modelinin validation cross-entropy loss değeri 2.30 olarak ölçüldüyse, bu modelin Perplexity (PPL) skoru yaklaşık kaçtır?",
                        options=["~2.30", "~5.29", "~9.97 (e^2.30)", "~100.0"],
                        correct_index=2,
                        explanation="Perplexity doğrudan Cross-Entropy kaybının doğal üssüdür: $\\text{PPL} = e^{\\text{Loss}} = e^{2.30} \\approx 9.97$. Model her adımda ortalama 10 kelime arasında kararsız kalmaktadır.",
                        math_intuition="\\text{PPL} = \\exp(\\text{Loss}) = e^{2.30} \\approx 9.97"
                    ),
                    MilestoneQuestion(
                        id="q_stage_8_2",
                        stage_id="stage_8",
                        question="GPU VRAM'iniz küçük olduğunda büyük bir efektif batch boyutu (ör. 64) elde etmek için hangi teknik kullanılır?",
                        options=[
                            "Learning rate'i sıfıra indirmek",
                            "Gradient Accumulation: Küçük batch'lerle gradyanları optimizer adımı atmadan biriktirip, birikim tamamlandığında tek bir ağırlık güncellemesi yapmak",
                            "Modelin katmanlarını silmek",
                            "Metinleri kırpmak"
                        ],
                        correct_index=1,
                        explanation="Örneğin batch_size=8 ile 8 adım gradyan toplanıp `optimizer.step()` çağrıldığında 64'lük efektif batch elde edilir.",
                    )
                ]
            ),
            JourneyStage(
                id="stage_9",
                order=9,
                title="Aşama 9 — Post-Training & İnce Ayar (SFT & LoRA)",
                category="Hizalama & Adaptasyon",
                summary="Ham metin tamamlayıcı modeli yardımcı bir asistana dönüştürün. Talimat veri setleri (Instruction Tuning), LoRA düşük sıralı matris adaptasyonu.",
                concepts=["Supervised Fine-Tuning (SFT)", "LoRA (Low-Rank Adaptation)", "Rank (r) & Alpha (\\alpha)", "Prompt Templates", "Adapter Merging"],
                objectives=[
                    "Kullanıcı-Asistan diyalog formatlarını tokenleştirmeyi öğrenmek",
                    "Milyarlarca parametreyi dondurup sadece $W + B \\cdot A$ düşük sıralı matrislerini eğitmeyi kavramak",
                    "Eğitilen LoRA adaptörlerini ana model ağırlıklarıyla kayıpsız birleştirmek"
                ],
                prerequisites=["stage_8"],
                lab_url="/training",
                lab_title="Training Lab (SFT & LoRA)",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_9_1",
                        stage_id="stage_9",
                        question="LoRA (Low-Rank Adaptation) tekniğinin tüm modeli fine-tune etmeye göre en büyük üstünlüğü nedir?",
                        options=[
                            "Modeli tamamen sıfırdan eğitmesidir",
                            "Ana model ağırlıklarını dondurarak sadece iki küçük rank matrisi ($d \\times r$ ve $r \\times k$) eğitmesi, böylece eğitilebilir parametre sayısını %99 azaltması ve VRAM tasarrufu sağlaması",
                            "Modelin hızını iki katına çıkarması",
                            "Tokenizer gereksinimini kaldırması"
                        ],
                        correct_index=1,
                        explanation="LoRA $\\Delta W = B \\cdot A$ matris ayrıştırmasıyla 4096 boyutlu bir katmanda rank $r=8$ seçildiğinde parametreleri 100 kattan fazla azaltır.",
                        math_intuition="W_{\\text{new}} = W_0 + \\frac{\\alpha}{r} (B \\times A)"
                    ),
                    MilestoneQuestion(
                        id="q_stage_9_2",
                        stage_id="stage_9",
                        question="SFT (Supervised Fine-Tuning) sırasında loss hesaplaması genellikle hangi tokenler üzerinde yapılır?",
                        options=[
                            "Yalnızca prompt/kullanıcı girdisi üzerinde",
                            "Yalnızca asistanın yanıt tokenleri üzerinde (Prompt tokenleri loss maskesiyle yoksayılır)",
                            "Rastgele seçilen %10 token üzerinde",
                            "Tüm metindeki özel tokenler üzerinde"
                        ],
                        correct_index=1,
                        explanation="Modelin kullanıcı talimatını ezberlemesi değil, verilen talimata uygun yanıt üretmesi hedeflendiği için kayıp yalnızca asistan çıktı tokenlerinde hesaplanır.",
                    )
                ]
            ),
            JourneyStage(
                id="stage_10",
                order=10,
                title="Aşama 10 — Model Değerlendirme & Benchmark Lab",
                category="Değerlendirme & Metrikler",
                summary="Modelinizin gerçek başarımını ölçün. BLEU, ROUGE n-gram örtüşmeleri, Brevity Penalty cezası ve iki modeli kafa kafaya yarıştıran Karşılaştırma Arenası.",
                concepts=["BLEU Score (Precision)", "Brevity Penalty (BP)", "ROUGE Score (Recall & LCS)", "Perplexity Benchmark", "Model Arena Comparison"],
                objectives=[
                    "Aday ve referans metin arasındaki unigram, bigram ve trigram örtüşmelerini görselleştirmek",
                    "Kısa çıktılarda Brevity Penalty'nin skoru nasıl düşürdüğünü gözlemlemek",
                    "Modelleri SQLite üzerinde geçmişe dayalı kıyaslayıp kazananı belirlemek"
                ],
                prerequisites=["stage_8", "stage_9"],
                lab_url="/evaluation",
                lab_title="Evaluation Lab (BLEU, ROUGE, Arena)",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_10_1",
                        stage_id="stage_10",
                        question="BLEU metriğinde Brevity Penalty (BP) cezasının görevi nedir?",
                        options=[
                            "Çok uzun cümleleri cezalandırmak",
                            "Modelin sadece tek bir doğru kelime üreterek %100 precision almasını engellemek için, referanstan kısa çıktılara ceza katsayısı ($BP < 1.0$) uygulamak",
                            "Yazım hatalarını düzeltmek",
                            "ROUGE skorunu artırmak"
                        ],
                        correct_index=1,
                        explanation="Eğer model 'the' tek kelimesini üretseydi unigram hassasiyeti 1.0 olurdu. Brevity Penalty $e^{1 - r/c}$ çarpanı ile metin uzunluğunu zorunlu kılar.",
                        math_intuition="\\text{BP} = \\min\\left(1, e^{1 - r/c}\\right)"
                    ),
                    MilestoneQuestion(
                        id="q_stage_10_2",
                        stage_id="stage_10",
                        question="Metin özetleme (summarization) görevlerinde neden BLEU yerine çoğunlukla ROUGE metriği tercih edilir?",
                        options=[
                            "Çünkü BLEU sadece Almanca için çalışır",
                            "Çünkü özetlemede önemli olan altının (referansın) ne kadarının kapsandığıdır (Recall odaklılık) ve ROUGE en uzun ortak alt diziyi (LCS) ölçer",
                            "Çünkü ROUGE daha hızlı hesaplanır",
                            "Çünkü BLEU'da skorlar 0-10 arasındadır"
                        ],
                        correct_index=1,
                        explanation="BLEU hassasiyete (precision) bakarken ROUGE geri çağırmaya (recall) ve LCS örtüşmesine odaklanır.",
                    )
                ]
            ),
            JourneyStage(
                id="stage_11",
                order=11,
                title="Aşama 11 — RAG (Retrieval-Augmented Generation)",
                category="Uygulama Mimarisi",
                summary="Modelin halüsinasyon görmesini engelleyin. Özel dokümanları chunk'lara bölün, vektör veritabanına indeksleyin ve sorgu anında bağlam olarak enjekte edin.",
                concepts=["Chunking Stratejileri", "Vector Database", "Top-K Benzerlik Araması", "Context Injection", "Halüsinasyon Azaltma"],
                objectives=[
                    "Sabit boyutlu ve semantik parçalama (chunking) yöntemlerini denemek",
                    "Cosine similarity ile en alakalı doküman parçalarını getirmek",
                    "Gelen parçaları prompt şablonuna bağlam (context) olarak ekleyip cevap ürettirmek"
                ],
                prerequisites=["stage_5", "stage_9"],
                lab_url="/rag-lab",
                lab_title="RAG Lab (Chunking & Vektör Arama)",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_11_1",
                        stage_id="stage_11",
                        question="RAG mimarisinde metin parçalama (chunking) yaparken 'chunk overlap' (parça örtüşmesi) bırakmanın temel amacı nedir?",
                        options=[
                            "Dosya boyutunu artırmak",
                            "Bir cümlenin veya fikrin tam sınır noktasında ikiye bölünmesi durumunda cümlenin anlamsal bağlamının kaybolmasını engellemek",
                            "Vektör aramasını iki kat hızlandırmak",
                            "Modelin daha fazla token üretmesini sağlamak"
                        ],
                        correct_index=1,
                        explanation="Overlap (ör. 50 token örtüşme) parçalanma sınırlarında kalan bilgilerin her iki parçada da korunmasını ve vektör aramasında bulunabilmesini sağlar.",
                    ),
                    MilestoneQuestion(
                        id="q_stage_11_2",
                        stage_id="stage_11",
                        question="RAG sistemi kullanan bir modelin halüsinasyon oranı neden çıplak modele göre çok daha düşüktür?",
                        options=[
                            "Modelin ağırlıkları anında güncellendiği için",
                            "Model cevabı kendi parametrik ezberinden değil, doğrudan arama motorundan gelen kanıt doküman parçalarına dayanarak ürettiği için",
                            "Model sıcaklığı (temperature) sıfır yapıldığı için",
                            "Tüm dokümanlar modelin içine kalıcı gömüldüğü için"
                        ],
                        correct_index=1,
                        explanation="RAG parametrik olmayan harici bellek (non-parametric memory) sağlayarak modelin doğrulanabilir olgusal kaynaklara referans vermesini sağlar.",
                    )
                ]
            ),
            JourneyStage(
                id="stage_12",
                order=12,
                title="Aşama 12 — Çıkarım & Playground (Inference & Serving)",
                category="Dağıtım & Çıkarım",
                summary="Eğitilen modelleri gerçek zamanlı çalıştırma. Sıcaklık (Temperature), Top-K, Top-P (Nucleus) örnekleme dinamikleri ve Server-Sent Events ile streaming.",
                concepts=["Temperature (Sıcaklık)", "Top-K Sampling", "Top-P (Nucleus) Sampling", "KV-Cache Re-use", "Streaming (SSE)"],
                objectives=[
                    "Sıcaklık katsayısının olasılık dağılımını nasıl sivrilttiğini veya düzleştirdiğini görmek",
                    "Top-P ile kümülatif olasılık eşiği filtrelemesini denemek",
                    "Otoregresif üretimde KV-cache sayesinde her adımda sadece tek bir yeni token hesaplamanın hızını gözlemlemek"
                ],
                prerequisites=["stage_4", "stage_7", "stage_8"],
                lab_url="/playground",
                lab_title="Playground (İnteraktif Çıkarım)",
                questions=[
                    MilestoneQuestion(
                        id="q_stage_12_1",
                        stage_id="stage_12",
                        question="Inference esnasında Temperature değeri 0.1'e düşürüldüğünde modelin çıktı davranışı nasıl değişir?",
                        options=[
                            "Tamamen rastgele ve tutarsız çıktılar üretir",
                            "Olasılık dağılımı en olası kelime üzerinde aşırı sivrilir; model son derece deterministik, odaklanmış ve olgusal yanıtlar üretir",
                            "Çıktı üretimi durur",
                            "Model kelimeleri ters sırada yazar"
                        ],
                        correct_index=1,
                        explanation="Düşük sıcaklık logitleri büyüterek en yüksek skora sahip tokenin olasılığını 1'e yaklaştırır (greedy-like behavior).",
                        math_intuition="P(x_i) = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}"
                    ),
                    MilestoneQuestion(
                        id="q_stage_12_2",
                        stage_id="stage_12",
                        question="Metin üretiminde KV-Cache kullanılmazsa, $N$ tokenlik bir yanıt üretirken toplam hesaplama karmaşıklığı ne olur?",
                        options=[
                            "$O(N)$",
                            "$O(N^2)$ (Her yeni token için tüm geçmiş baştan hesaplanır)",
                            "$O(\\log N)$",
                            "$O(1)$"
                        ],
                        correct_index=1,
                        explanation="KV-cache olmadan $n$. token üretilirken $1$ ile $n-1$ arasındaki tüm geçmiş yeniden hesaplanır ($1 + 2 + ... + N = O(N^2)$). KV-cache bunu $O(N)$'e düşürür.",
                    )
                ]
            ),
        ]

    def _build_knowledge_graph(self) -> tuple[List[KnowledgeNode], List[KnowledgeEdge]]:
        nodes = [
            KnowledgeNode(id="math", label="Temel Matematik & Lineer Cebir", stage_id="stage_0", category="Matematik", level=0, lab_url="/tensor-lab"),
            KnowledgeNode(id="raw_data", label="Ham Veri & Metin Ayrıştırma", stage_id="stage_1", category="Veri", level=1, lab_url="/upload"),
            KnowledgeNode(id="canonical_data", label="Dataset Standardı & MinHash", stage_id="stage_2", category="Veri", level=2, lab_url="/dataset-compiler"),
            KnowledgeNode(id="tokenizer", label="BPE Tokenizasyon & Vocab", stage_id="stage_3", category="NLP", level=3, lab_url="/tokenizer"),
            KnowledgeNode(id="tensors", label="Tensörler & Autograd", stage_id="stage_4", category="Altyapı", level=4, lab_url="/tensor-lab"),
            KnowledgeNode(id="embeddings", label="Gömme Uzayı (Embeddings)", stage_id="stage_5", category="Temsil", level=5, lab_url="/embedding-lab"),
            KnowledgeNode(id="attention", label="Self-Attention & MHA", stage_id="stage_6", category="Mimari", level=6, lab_url="/attention-lab"),
            KnowledgeNode(id="transformer", label="Modern Transformer (RoPE & GQA)", stage_id="stage_7", category="Mimari", level=7, lab_url="/transformer-lab"),
            KnowledgeNode(id="pretraining", label="Pretraining & Causal LM", stage_id="stage_8", category="Eğitim", level=8, lab_url="/training"),
            KnowledgeNode(id="sft_lora", label="SFT & LoRA İnce Ayar", stage_id="stage_9", category="Hizalama", level=9, lab_url="/training"),
            KnowledgeNode(id="evaluation", label="Model Değerlendirme & Benchmark", stage_id="stage_10", category="Değerlendirme", level=10, lab_url="/evaluation"),
            KnowledgeNode(id="rag", label="RAG & Bilgi Tabanı", stage_id="stage_11", category="Uygulama", level=10, lab_url="/rag-lab"),
            KnowledgeNode(id="inference", label="Çıkarım & Playground", stage_id="stage_12", category="Çıkarım", level=11, lab_url="/playground"),
        ]

        edges = [
            KnowledgeEdge(id="e1", source="math", target="raw_data", label="Veri Temsili"),
            KnowledgeEdge(id="e2", source="raw_data", target="canonical_data", label="Temizleme & Dedup"),
            KnowledgeEdge(id="e3", source="canonical_data", target="tokenizer", label="Eğitim Külliyatı"),
            KnowledgeEdge(id="e4", source="math", target="tensors", label="Matematiksel Alt Yapı"),
            KnowledgeEdge(id="e5", source="tokenizer", target="embeddings", label="Token ID Lookup"),
            KnowledgeEdge(id="e6", source="tensors", target="embeddings", label="Vektör Boyutları"),
            KnowledgeEdge(id="e7", source="embeddings", target="attention", label="Q, K, V Projeksiyonları"),
            KnowledgeEdge(id="e8", source="tensors", target="attention", label="MatMul & Softmax"),
            KnowledgeEdge(id="e9", source="attention", target="transformer", label="Transformer Blokları"),
            KnowledgeEdge(id="e10", source="transformer", target="pretraining", label="Model Mimarisi"),
            KnowledgeEdge(id="e11", source="canonical_data", target="pretraining", label="Tokenized Dataset"),
            KnowledgeEdge(id="e12", source="pretraining", target="sft_lora", label="Temel Model Checkpoint"),
            KnowledgeEdge(id="e13", source="pretraining", target="evaluation", label="Model Başarımı"),
            KnowledgeEdge(id="e14", source="sft_lora", target="evaluation", label="Fine-Tuned Başarım"),
            KnowledgeEdge(id="e15", source="embeddings", target="rag", label="Vektör Benzerliği"),
            KnowledgeEdge(id="e16", source="sft_lora", target="rag", label="Bağlam Yanıtlayıcı"),
            KnowledgeEdge(id="e17", source="sft_lora", target="inference", label="Çıkarım & Üretim"),
            KnowledgeEdge(id="e18", source="transformer", target="inference", label="KV-Cache Hızlandırma"),
        ]

        return nodes, edges

    def _build_glossary(self) -> List[GlossaryTerm]:
        return [
            GlossaryTerm(
                term="BPE (Byte-Pair Encoding)",
                category="NLP & Tokenizer",
                short_def="En sık geçen karakter veya bayt çiftlerini yinelemeli olarak birleştiren alt-kelime tokenizasyon algoritması.",
                detailed_explanation="Kelime seviyesi tokenizasyonun bilinmeyen kelime (OOV) ve karakter seviyesinin aşırı uzun dizi problemlerini çözerek en uygun sözlüğü inşa eder.",
                formula="\\text{freq}(pair) = \\max_{(c_1, c_2)} \\text{count}(c_1 c_2)",
                related_lab_url="/tokenizer",
                related_stage_id="stage_3"
            ),
            GlossaryTerm(
                term="Self-Attention",
                category="Mimari",
                short_def="Bir dizideki her tokenin diğer tüm tokenlerle anlamsal ilişkisini hesaplayan dikkat mekanizması.",
                detailed_explanation="Query, Key ve Value vektörlerinin normalize edilmiş nokta çarpımı ile hangi kelimenin nereye odaklanması gerektiğini ağırlıklandırır.",
                formula="\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V",
                related_lab_url="/attention-lab",
                related_stage_id="stage_6"
            ),
            GlossaryTerm(
                term="RoPE (Rotary Position Embedding)",
                category="Mimari",
                short_def="Token vektörlerini uzayda pozisyonlarına bağlı açılarla döndürerek göreli mesafeyi koruyan modern konum kodlama yöntemi.",
                detailed_explanation="LLaMA-3, Mistral ve modern LLM'lerin standardıdır. Mutlak toplama yerine ortogonal rotasyon matrisi kullanarak uzun dizilerde mükemmel genelleme sağlar.",
                formula="R_{\\Theta, m}^d = \\text{diag}\\left(R_{\\theta_1, m}, \\dots, R_{\\theta_{d/2}, m}\\right)",
                related_lab_url="/transformer-lab",
                related_stage_id="stage_7"
            ),
            GlossaryTerm(
                term="GQA (Grouped-Query Attention)",
                category="Mimari & Donanım",
                short_def="Birden fazla Query kafasının ortak tek bir Key ve Value kafasını paylaştığı optimize dikkat varyantı.",
                detailed_explanation="MHA'nın yüksek temsil gücü ile MQA'nın düşük bellek maliyeti arasında ideal bir köprüdür; KV-Cache bellek boyutunu 4 ila 8 kat azaltır.",
                formula="H_{KV} = H_Q / G \\quad (G: \\text{Grup Boyutu})",
                related_lab_url="/transformer-lab",
                related_stage_id="stage_7"
            ),
            GlossaryTerm(
                term="Perplexity (PPL)",
                category="Değerlendirme",
                short_def="Bir dil modelinin test verisindeki bir sonraki kelimeyi tahmin ederken yaşadığı ortalama belirsizlik veya şaşkınlık skoru.",
                detailed_explanation="Düşük olması daha iyidir. Cross-Entropy kaybının doğal üssü ($e^{\\text{Loss}}$) olarak hesaplanır. Modelin seçim yaparken kaç kelime arasında tereddüt ettiğini gösterir.",
                formula="\\text{PPL} = \\exp\\left(-\\frac{1}{N} \\sum_{i=1}^N \\ln P(w_i | w_{<i})\\right)",
                related_lab_url="/evaluation",
                related_stage_id="stage_10"
            ),
            GlossaryTerm(
                term="BLEU Score",
                category="Değerlendirme",
                short_def="Modelin ürettiği metin ile altın referans arasındaki n-gram örtüşme hassasiyetini ölçen değerlendirme metriği.",
                detailed_explanation="Özellikle çeviri ve üretim kalitesinde kullanılır. Kısa metinlerin sahte yüksek skor almasını engellemek için Brevity Penalty (BP) ile çarpılır.",
                formula="\\text{BLEU} = \\text{BP} \\times \\exp\\left(\\sum_{n=1}^N w_n \\ln p_n\\right)",
                related_lab_url="/evaluation",
                related_stage_id="stage_10"
            ),
            GlossaryTerm(
                term="Brevity Penalty (BP)",
                category="Değerlendirme",
                short_def="Aday metin referanstan daha kısa olduğunda BLEU skorunu düşüren uzunluk ceza katsayısı.",
                detailed_explanation="Aday uzunluğu referanstan büyük veya eşitse 1.0; kısaysa $e^{1 - r/c}$ değerini alır.",
                formula="\\text{BP} = \\begin{cases} 1 & \\text{if } c > r \\\\ e^{1 - r/c} & \\text{if } c \\le r \\end{cases}",
                related_lab_url="/evaluation",
                related_stage_id="stage_10"
            ),
            GlossaryTerm(
                term="ROUGE Score",
                category="Değerlendirme",
                short_def="Özetleme ve metin üretiminde referans metnin ne kadarının kapsandığını (Recall ve LCS) ölçen metrik.",
                detailed_explanation="ROUGE-1 unigram, ROUGE-2 bigram ve ROUGE-L en uzun ortak alt dizi (Longest Common Subsequence) F1 skorlarını hesaplar.",
                formula="\\text{ROUGE-L}_{F1} = \\frac{(1 + \\beta^2) R_{LCS} P_{LCS}}{R_{LCS} + \\beta^2 P_{LCS}}",
                related_lab_url="/evaluation",
                related_stage_id="stage_10"
            ),
            GlossaryTerm(
                term="LoRA (Low-Rank Adaptation)",
                category="Hizalama & Fine-Tuning",
                short_def="Büyük model ağırlıklarını dondurup, ağırlık değişimini iki düşük sıralı küçük matrisin çarpımıyla modelleyen parametre verimli eğitim tekniği.",
                detailed_explanation="Eğitilebilir parametre sayısını %99 azaltarak tüketici sınıfı GPU'larda dev modellerin özelleştirilmesini mümkün kılar.",
                formula="W = W_0 + \\Delta W = W_0 + \\frac{\\alpha}{r} (B \\times A)",
                related_lab_url="/training",
                related_stage_id="stage_9"
            ),
            GlossaryTerm(
                term="KV-Cache",
                category="Çıkarım & Donanım",
                short_def="Otoregresif üretimde daha önce üretilmiş tokenlerin Key ve Value tensörlerini bellekte saklayarak tekrar hesaplanmasını önleyen mekanizma.",
                detailed_explanation="Her yeni token üretildiğinde tüm diziyi baştan geçirmek yerine sadece yeni tokenin KV'si eklenir ve dikkat skoru $O(N)$ sürede hesaplanır.",
                formula="\\text{Memory}_{KV} = 2 \\times B \\times T \\times H_{KV} \\times d_{head} \\times \\text{bytes}",
                related_lab_url="/playground",
                related_stage_id="stage_12"
            ),
            GlossaryTerm(
                term="MinHash LSH",
                category="Veri Kalitesi",
                short_def="Büyük metin koleksiyonlarında benzer dokümanları devasa karşılaştırma maliyeti olmadan bulan olasılıksal tekilleştirme algoritması.",
                detailed_explanation="N-gram kümelerinin Jaccard benzerliğini koruyan hash imzaları oluşturur ve mükerrer verileri temizler.",
                formula="P(h(A) = h(B)) = J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}",
                related_lab_url="/dataset-compiler",
                related_stage_id="stage_2"
            ),
            GlossaryTerm(
                term="Cosine Similarity",
                category="Temsil & Embedding",
                short_def="İki vektör arasındaki açının kosinüsünü bularak anlamsal yönelim benzerliğini ölçen metrik.",
                detailed_explanation="Vektörlerin uzunluklarından bağımsız olarak uzaydaki doğrultularının ne kadar örtüştüğünü (-1 ile +1 aralığında) ifade eder.",
                formula="\\text{sim}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\| \\|\\mathbf{v}\\|}",
                related_lab_url="/embedding-lab",
                related_stage_id="stage_5"
            ),
            GlossaryTerm(
                term="Temperature (Örnekleme Sıcaklığı)",
                category="Çıkarım",
                short_def="Logit değerlerini bölerek Softmax olasılık dağılımının sivriliğini veya düzlüğünü kontrol eden hiperparametre.",
                detailed_explanation="Düşük sıcaklık (T < 0.7) modeli daha muhafazakar ve tutarlı yaparken, yüksek sıcaklık (T > 0.9) yaratıcılığı ve çeşitliliği artırır.",
                formula="P(x_i) = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}",
                related_lab_url="/playground",
                related_stage_id="stage_12"
            ),
            GlossaryTerm(
                term="SwiGLU",
                category="Mimari",
                short_def="Gated Linear Unit (GLU) ailesinden, SiLU/Swish aktivasyonuyla çalışan ve modern LLM'lerde MLP katmanının yerini alan ileri beslemeli blok.",
                detailed_explanation="LLaMA ve PaLM gibi modern mimarilerde standart MLP'den belirgin şekilde daha iyi yakınsama ve temsil kapasitesi sağlar.",
                formula="\\text{SwiGLU}(x) = (\\text{SiLU}(x W_{gate}) \\odot x W_{up}) W_{down}",
                related_lab_url="/transformer-lab",
                related_stage_id="stage_7"
            ),
            GlossaryTerm(
                term="RAG (Retrieval-Augmented Generation)",
                category="Uygulama",
                short_def="Modelin bilgi sınırlarını aşmak ve halüsinasyonu önlemek için harici vektör veritabanından ilgili bağlamı alıp prompt'a ekleyen mimari.",
                detailed_explanation="Chunking, Embedding ve Vektör Veritabanı araması bileşenlerini birleştirerek güncel ve özel veriler üzerinde doğru cevap üretimini sağlar.",
                formula="P(Y | X) = \\sum_{D} P(Y | X, D) P(D | X)",
                related_lab_url="/rag-lab",
                related_stage_id="stage_11"
            )
        ]
