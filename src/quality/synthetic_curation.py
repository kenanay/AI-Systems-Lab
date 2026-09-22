"""
src/quality/synthetic_curation.py

Synthetic Data Generation, Quality Scoring & Multi-Stage Filtering Engine.
Covers:
- Multi-paradigm synthetic data generator (Self-Instruct, Chain-of-Thought, Code Synthesis, Textbook QA)
- Quality scoring: Perplexity estimation, N-gram repetition loops, Lexical diversity (TTR), Turkish PII scanning
- Multi-stage filter pipeline funnel with rejection diagnostics & deduplication
"""

import math
import re
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter

from src.pii.turkish_detector import TurkishPIIDetector, PIIType


@dataclass
class SyntheticSample:
    """A synthetic instruction/data sample."""
    id: str
    instruction: str
    input_context: str
    response: str
    paradigm: str  # 'self_instruct' | 'chain_of_thought' | 'code_synthesis' | 'textbook_qa'
    domain: str    # 'computer_science' | 'mathematics' | 'natural_sciences' | 'turkish_knowledge'
    complexity: str # 'basic' | 'intermediate' | 'advanced'
    metrics: Dict[str, Any] = field(default_factory=dict)
    rejection_stage: Optional[str] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FilterThresholds:
    """User-configurable thresholds for the curation pipeline."""
    min_perplexity: float = 5.0
    max_perplexity: float = 120.0
    min_words: int = 15
    max_words: int = 1500
    max_repetition_ratio: float = 0.18
    min_quality_score: float = 65.0
    pii_action: str = "reject"  # 'reject' | 'mask' | 'allow'
    max_jaccard_similarity: float = 0.82


@dataclass
class FilterStageMetric:
    """Funnel statistics for each filtering stage."""
    stage_name: str
    input_count: int
    passed_count: int
    rejected_count: int
    retention_rate: float  # percentage (0-100)


# ============================================================================
# 1. Quality Scoring Engine
# ============================================================================

class QualityScoringEngine:
    """
    Evaluates individual text samples across multiple linguistic and structural dimensions.
    """

    def __init__(self):
        self.pii_detector = TurkishPIIDetector()

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation tokenizer."""
        clean = re.sub(r'[^\w\s]', ' ', text.lower(), flags=re.UNICODE)
        tokens = [t.strip() for t in clean.split() if t.strip()]
        return tokens

    def compute_ngram_repetition_ratio(self, text: str, n: int = 2) -> float:
        """
        Computes the ratio of repeated n-grams in text:
        ratio = (total_ngrams - unique_ngrams) / total_ngrams
        High ratio indicates degenerate repetitive loops.
        """
        tokens = self.tokenize(text)
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0

        total_ngrams = len(ngrams)
        unique_ngrams = len(set(ngrams))
        repeated = total_ngrams - unique_ngrams
        return max(0.0, float(repeated) / float(total_ngrams))

    def compute_lexical_diversity(self, text: str) -> float:
        """Type-Token Ratio (TTR): unique_words / total_words."""
        tokens = self.tokenize(text)
        if not tokens:
            return 0.0
        return float(len(set(tokens))) / float(len(tokens))

    def compute_perplexity_estimate(self, text: str) -> float:
        """
        Approximates text perplexity using character and subword n-gram entropy.
        Calibrated to standard LLM ranges:
        - Repetitive text ('a a a a a'): PPL < 5
        - Fluent Turkish / English text: PPL 15 - 45
        - Dense / Technical text: PPL 45 - 85
        - Incoherent / Broken / Gibberish: PPL > 130
        """
        clean_text = text.strip()
        if len(clean_text) < 5:
            return 999.0

        tokens = self.tokenize(clean_text)
        if not tokens:
            return 999.0

        # Check for degenerate token loop
        repetition_2g = self.compute_ngram_repetition_ratio(clean_text, n=2)
        repetition_1g = self.compute_ngram_repetition_ratio(clean_text, n=1)
        if repetition_2g > 0.65 or repetition_1g > 0.75:
            # Model collapse loop
            return max(1.5, 4.0 * (1.0 - repetition_2g))

        # Bigram character entropy calculation
        chars = list(clean_text.lower())
        char_bigrams = [f"{chars[i]}{chars[i+1]}" for i in range(len(chars) - 1)]
        if not char_bigrams:
            return 50.0

        counts = Counter(char_bigrams)
        total = len(char_bigrams)
        entropy = -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())

        # Length and vocabulary richness factors
        ttr = self.compute_lexical_diversity(clean_text)
        avg_word_len = sum(len(t) for t in tokens) / max(1, len(tokens))

        # Calibration formula (calibrated so typical fluent text is 15-55 PPL)
        base_ppl = 2.0 ** (entropy * 0.75)
        # Penalize excessive average word length (random gibberish like 'asdfasdfasdf')
        if avg_word_len > 13.0 or avg_word_len < 2.2:
            base_ppl *= 2.5

        # Penalize unnaturally low diversity
        if ttr < 0.25:
            base_ppl = max(3.0, base_ppl * 0.4)

        return round(float(max(1.2, min(500.0, base_ppl))), 2)

    def scan_pii(self, text: str) -> List[Dict[str, Any]]:
        """Scans text for Turkish PII entities."""
        matches = self.pii_detector.scan_text(text)
        return [m.to_dict() for m in matches]

    def mask_pii(self, text: str) -> str:
        """Masks detected PII in text."""
        matches = self.pii_detector.scan_text(text)
        if not matches:
            return text
        return self.pii_detector.mask_text(text, matches)

    def score_sample(
        self,
        instruction: str,
        response: str,
        input_context: str = ""
    ) -> Dict[str, Any]:
        """
        Computes a comprehensive multi-criteria quality score (0-100) and diagnostics.
        """
        full_text = f"{instruction}\n{input_context}\n{response}".strip()
        tokens = self.tokenize(response)
        word_count = len(tokens)
        char_count = len(response)

        # 1. Repetition Metrics
        rep_ratio_2g = self.compute_ngram_repetition_ratio(response, n=2)
        rep_ratio_4g = self.compute_ngram_repetition_ratio(response, n=4)
        ttr = self.compute_lexical_diversity(response)

        # 2. Perplexity Heuristic
        ppl = self.compute_perplexity_estimate(response)

        # 3. PII Detection
        pii_matches = self.scan_pii(full_text)
        pii_count = len(pii_matches)

        # 4. Component Scores (0-100 scale)
        # Length adequacy score
        if word_count < 10:
            length_score = max(10.0, word_count * 5.0)
        elif word_count <= 400:
            length_score = 95.0
        elif word_count <= 800:
            length_score = 90.0
        else:
            length_score = max(50.0, 100.0 - (word_count - 800) * 0.05)

        # Fluency / Perplexity score (ideal is 15 - 55)
        if 15.0 <= ppl <= 60.0:
            fluency_score = 98.0
        elif 8.0 <= ppl < 15.0 or 60.0 < ppl <= 95.0:
            fluency_score = 80.0
        elif ppl < 8.0:
            # Too repetitive
            fluency_score = max(10.0, ppl * 5.0)
        elif 95.0 < ppl <= 140.0:
            fluency_score = 55.0
        else:
            # Likely incoherent or noisy
            fluency_score = max(10.0, 100.0 - (ppl - 140.0))

        # Repetition score
        if rep_ratio_2g < 0.10:
            rep_score = 100.0
        elif rep_ratio_2g < 0.20:
            rep_score = 80.0
        elif rep_ratio_2g < 0.35:
            rep_score = 50.0
        else:
            rep_score = max(5.0, 100.0 - (rep_ratio_2g * 200.0))

        # Safety / PII score
        pii_score = 100.0 if pii_count == 0 else max(0.0, 100.0 - pii_count * 40.0)

        # Structural Formatting Score
        format_score = 70.0
        if "\n" in response or "1." in response or "-" in response or "```" in response:
            format_score += 20.0
        if len(instruction.strip()) > 15:
            format_score += 10.0
        format_score = min(100.0, format_score)

        # Weighted Total Quality Score (0-100)
        composite_score = (
            fluency_score * 0.30 +
            rep_score * 0.25 +
            length_score * 0.20 +
            format_score * 0.15 +
            pii_score * 0.10
        )
        composite_score = round(max(0.0, min(100.0, composite_score)), 1)

        # Verdict
        reasons = []
        if pii_count > 0:
            reasons.append(f"PII Tespiti: {pii_count} adet hassas veri bulundu.")
        if rep_ratio_2g > 0.22:
            reasons.append(f"Yüksek Tekrarlama: 2-gram tekrarlama oranı (%{rep_ratio_2g * 100:.1f}) eşiğin üzerinde.")
        if ppl < 6.0:
            reasons.append(f"Model Çökmesi (Düşük Perplexity): {ppl:.1f} (Aşırı tekrar).")
        elif ppl > 125.0:
            reasons.append(f"Yüksek Perplexity (Tutarsızlık Riski): {ppl:.1f}.")
        if word_count < 12:
            reasons.append(f"Yetersiz Uzunluk: {word_count} kelime çok kısa.")

        if composite_score >= 70.0 and pii_count == 0 and rep_ratio_2g <= 0.20 and 6.0 <= ppl <= 110.0:
            verdict = "ACCEPT"
        elif composite_score >= 50.0:
            verdict = "NEEDS_REVISION"
        else:
            verdict = "REJECT"

        return {
            "composite_score": composite_score,
            "verdict": verdict,
            "verdict_reasons": reasons,
            "perplexity": ppl,
            "repetition_ratio_2g": round(rep_ratio_2g, 4),
            "repetition_ratio_4g": round(rep_ratio_4g, 4),
            "lexical_diversity_ttr": round(ttr, 4),
            "word_count": word_count,
            "char_count": char_count,
            "pii_count": pii_count,
            "pii_matches": pii_matches,
            "subscores": {
                "fluency": round(fluency_score, 1),
                "repetition": round(rep_score, 1),
                "length": round(length_score, 1),
                "formatting": round(format_score, 1),
                "safety": round(pii_score, 1),
            }
        }


# ============================================================================
# 2. Synthetic Data Generator
# ============================================================================

class SyntheticDataGenerator:
    """
    Generates synthetic instruction-response pairs across multiple paradigms and domains.
    """

    TEMPLATES: Dict[str, Dict[str, List[Dict[str, str]]]] = {
        "self_instruct": {
            "computer_science": [
                {
                    "instruction": "İkili arama (Binary Search) algoritmasının mantığını ve zaman karmaşıklığını açıklayın.",
                    "input_context": "Sıralı bir dizi üzerinde arama senaryosu.",
                    "response": "İkili arama (Binary Search), sıralı bir dizi üzerinde hedef elemanı bulmak için 'böl ve fethet' (divide and conquer) yaklaşımını kullanır.\n\nÇalışma Mantığı:\n1. Dizinin ortasındaki eleman kontrol edilir.\n2. Aranan değer orta elemana eşitse işlem biter.\n3. Aranan değer orta elemandan küçükse sol yarıya, büyükse sağ yarıya odaklanılır.\n4. Bu işlem aralık tükenene kadar tekrarlanır.\n\nKarmaşıklık Analizi:\n- Zaman Karmaşıklığı: O(log n) - Her adımda arama uzayı yarıya iner.\n- Alan Karmaşıklığı: İteratif yöntemde O(1), özyinelemeli yöntemde O(log n) çağrı yığını.",
                },
                {
                    "instruction": "Bir REST API tasarımında idempotent metodların ne anlama geldiğini ve PUT ile POST arasındaki farkı belirtin.",
                    "input_context": "HTTP standartları ve web mimarisi.",
                    "response": "Idempotent bir HTTP metodu, aynı parametrelerle bir veya birden fazla kez çalıştırıldığında sunucu durumunda aynı etkiyi bırakan operasyondur.\n\nPUT vs POST Farkı:\n- PUT (Idempotent): Belirli bir URI'deki kaynağı tamamen günceller veya yoksa oluşturur. Art arda 5 kez çağrılsa da sonuç değişmez.\n- POST (Non-idempotent): Genellikle yeni bir alt kaynak oluşturur. Art arda 5 kez çağrılırsa 5 ayrı kayıt oluşur.",
                },
            ],
            "mathematics": [
                {
                    "instruction": "Özdeğer (Eigenvalue) ve Özvektör (Eigenvector) kavramlarını doğrusal cebir bağlamında tanımlayın.",
                    "input_context": "Kare matris dönüşümleri A * v = lambda * v.",
                    "response": "Doğrusal cebirde, bir kare matris (A) bir vektör (v) ile çarpıldığında, eğer vektörün yönü değişmiyor ve sadece skaler bir çarpan (λ) ile ölçekleniyorsa:\n\nA * v = λ * v\n\n- v: Sıfırdan farklı bir Özvektördür (Eigenvector).\n- λ: İlgili Özdeğerdir (Eigenvalue).\n\nÖnem ve Kullanım Alanları:\n- PCA (Temel Bileşen Analizi) ile veri boyut indirgeme.\n- Sayfa sıralama algoritmaları (Google PageRank).\n- Kuantum mekaniği ve titreşim modları analizi.",
                },
            ],
            "turkish_knowledge": [
                {
                    "instruction": "Türk dilinde ünlü düşmesi kuralını örneklerle açıklayın.",
                    "input_context": "İki heceli sözcüklere ünlüyle başlayan ek getirilmesi.",
                    "response": "Türkçede son hecesinde dar ünlü (ı, i, u, ü) bulunan bazı iki heceli sözcüklere, ünlü ile başlayan bir ek getirildiğinde ikinci hecedeki dar ünlü düşer. Bu ses olayına ünlü düşmesi denir.\n\nÖrnekler:\n- burun + u -> burnu\n- akıl + ı -> aklı\n- şehir + e -> şehre\n- göğüs + ü -> göğsü",
                },
            ]
        },
        "chain_of_thought": {
            "mathematics": [
                {
                    "instruction": "Bir torbada 4 kırmızı, 6 mavi ve 5 yeşil top vardır. İadesiz olarak çekilen iki topun ikisinin de mavi olma olasılığı nedir?",
                    "input_context": "Kombinatorik ve koşullu olasılık problemi.",
                    "response": "Adım adım çözelim:\n\nAdım 1: Toplam top sayısını belirleyelim:\nToplam = 4 (kırmızı) + 6 (mavi) + 5 (yeşil) = 15 top.\n\nAdım 2: İlk çekilen topun mavi olma olasılığı:\nP(1. Mavi) = 6 / 15 = 2 / 5.\n\nAdım 3: İadesiz çekim yapıldığı için kalan topları hesaplayalım:\nKalan toplam top sayısı = 15 - 1 = 14 top.\nKalan mavi top sayısı = 6 - 1 = 5 mavi top.\n\nAdım 4: İkinci topun mavi olma olasılığı:\nP(2. Mavi | 1. Mavi) = 5 / 14.\n\nAdım 5: Bileşik olasılığı çarpalım:\nP = (6 / 15) * (5 / 14) = (2 / 5) * (5 / 14) = 2 / 14 = 1 / 7.\n\nSonuç: Her iki topun da mavi olma olasılığı 1/7'dir (~%14.28).",
                },
            ],
            "computer_science": [
                {
                    "instruction": "Bir grafikte döngü (cycle) olup olmadığını tespit etmek için DFS algoritmasını nasıl adım adım uygularsınız?",
                    "input_context": "Yönlü bir graf (Directed Graph) analizi.",
                    "response": "Yönlü bir grafikte döngü tespiti için 3-renkli (White/Gray/Black) DFS algoritması kullanılır:\n\nAdım 1: Durum Tanımları:\n- BEYAZ: Ziyaret edilmemiş düğüm.\n- GRİ: Keşif aşamasında olan düğüm (aktif çağrı yığınında).\n- SİYAH: Ziyareti ve tüm komşuları tamamlanmış düğüm.\n\nAdım 2: Ziyaret Adımları:\n1. Başlangıçta tüm düğümler BEYAZ olarak işaretlenir.\n2. Herhangi bir BEYAZ düğümden DFS başlatılır ve rengi GRİ yapılır.\n3. Komşu düğümlere bakılır:\n   - Komşu BEYAZ ise, özyinelemeli olarak ziyaret edilir.\n   - Komşu GRİ ise, aktif çağrı yığınındaki bir ataya geri dönülmüştür (Back-edge), DÖNGÜ BULUNDU!\n   - Komşu SİYAH ise, önceden işlenmiştir, atlanır.\n4. Bir düğümün tüm komşuları incelendiğinde rengi SİYAH yapılır.\n\nSonuç: Eğer ziyaret esnasında GRİ bir komşuya rastlanırsa grafik döngü içerir.",
                },
            ],
        },
        "code_synthesis": {
            "computer_science": [
                {
                    "instruction": "Python dilinde iki sıralı listeyi birleştiren (merge sorted lists) verimli bir fonksiyon yazın.",
                    "input_context": "O(m+n) zaman karmaşıklığı hedefi.",
                    "response": "```python\ndef merge_sorted_lists(list1: list[int], list2: list[int]) -> list[int]:\n    \"\"\"\n    İki sıralı tamsayı listesini tek bir sıralı listede birleştirir.\n    Zaman Karmaşıklığı: O(m + n)\n    Alan Karmaşıklığı: O(m + n)\n    \"\"\"\n    merged = []\n    i, j = 0, 0\n    \n    while i < len(list1) and j < len(list2):\n        if list1[i] <= list2[j]:\n            merged.append(list1[i])\n            i += 1\n        else:\n            merged.append(list2[j])\n            j += 1\n            \n    # Kalan elemanları ekle\n    merged.extend(list1[i:])\n    merged.extend(list2[j:])\n    \n    return merged\n\n# Test örneği\nprint(merge_sorted_lists([1, 3, 5], [2, 4, 6]))  # [1, 2, 3, 4, 5, 6]\n```",
                },
            ],
        },
        "textbook_qa": {
            "natural_sciences": [
                {
                    "instruction": "Hücresel solunumda ATP sentaz enziminin kemiosmotik mekanizmasını açıklayın.",
                    "input_context": "Mitokondri iç zarı ve proton gradyanı.",
                    "response": "ATP sentaz, hücresel solunumun oksidatif fosforilasyon evresinde görev yapan döner moleküler bir motordur.\n\nKemiosmoz Mekanizması:\n1. Elektron Taşıma Sistemi (ETS), elektronları aktarırken protonları (H+) matriksten zarlar arası boşluğa pompalar.\n2. Bu durum zarlar arası boşlukta yüksek bir proton ve pozitif yük derişimi oluşturur (Elektrokimyasal proton gradyanı / Proton İtici Güç).\n3. Protonlar, seçici geçirgen iç zarı doğrudan geçemezler; tek kaçış yolu ATP sentazın F0 kanal ünitesidir.\n4. Proton akışı F0 rotorunu döndürür. Bu dönme hareketi F1 katalitik ünitesinin şeklini değiştirerek ADP + Pi birleşimini sağlar ve ATP sentezlenir.",
                },
            ],
        }
    }

    # Low-quality templates deliberately included to test filtering pipelines
    DEGENERATE_TEMPLATES = [
        {
            "instruction": "Yapay zeka modelleri hakkında bilgi verin.",
            "input_context": "",
            "response": "model model model model model model model model model model model model model model model model model model model model model model model model model model",
            "flaw": "repetition_loop"
        },
        {
            "instruction": "Gelişmiş kuantum fiziği formülleri.",
            "input_context": "",
            "response": "xkcd qwerty zzz 1928475 xkjhfsd @#$ %^&* random gibberish phrase lorem ipsum qwertyuiop asdfghjkl zxcvbnm 987654321",
            "flaw": "high_perplexity"
        },
        {
            "instruction": "Müşteri iletişim bilgisi oluştur.",
            "input_context": "Kullanıcı formu.",
            "response": "Müşteri temsilcimiz Ahmet Yılmaz ile iletişime geçebilirsiniz. TC Kimlik: 10000000146, Telefon: 0555 123 45 67, E-posta: ahmet.yilmaz@sirket.com.",
            "flaw": "pii_leakage"
        },
        {
            "instruction": "Derin öğrenme nedir?",
            "input_context": "",
            "response": "Yapay zekadır.",
            "flaw": "too_short"
        }
    ]

    def __init__(self):
        self.scorer = QualityScoringEngine()

    def generate_batch(
        self,
        paradigm: str = "self_instruct",
        domain: str = "computer_science",
        complexity: str = "intermediate",
        count: int = 5,
        include_edge_cases: bool = True
    ) -> List[SyntheticSample]:
        """
        Generates synthetic samples with quality metrics pre-computed.
        """
        samples: List[SyntheticSample] = []
        domain_templates = self.TEMPLATES.get(paradigm, {}).get(domain, [])

        # Fallback to any domain in paradigm if requested domain is empty
        if not domain_templates and paradigm in self.TEMPLATES:
            for d, t_list in self.TEMPLATES[paradigm].items():
                if t_list:
                    domain_templates = t_list
                    break

        # Fallback to self_instruct computer_science
        if not domain_templates:
            domain_templates = self.TEMPLATES["self_instruct"]["computer_science"]

        # 1. Generate normal samples
        for i in range(count):
            base = domain_templates[i % len(domain_templates)]
            instr_seed = base["instruction"]
            hash_id = hashlib.md5(f"{i}-{instr_seed}".encode()).hexdigest()[:6]
            sample_id = f"SYNTH-{paradigm[:4].upper()}-{hash_id}"
            
            # Add subtle variations if count exceeds template count
            instruction = base["instruction"]
            response = base["response"]
            if i >= len(domain_templates):
                variation_num = (i // len(domain_templates)) + 1
                instruction = f"{instruction} (Varyasyon {variation_num})"

            metrics = self.scorer.score_sample(instruction, response, base.get("input_context", ""))

            samples.append(
                SyntheticSample(
                    id=sample_id,
                    instruction=instruction,
                    input_context=base.get("input_context", ""),
                    response=response,
                    paradigm=paradigm,
                    domain=domain,
                    complexity=complexity,
                    metrics=metrics
                )
            )

        # 2. Add realistic degenerate samples if requested (for filtering demonstrations)
        if include_edge_cases:
            for idx, degen in enumerate(self.DEGENERATE_TEMPLATES):
                sample_id = f"EDGE-{degen['flaw'][:4].upper()}-{idx}"
                metrics = self.scorer.score_sample(degen["instruction"], degen["response"], degen.get("input_context", ""))
                samples.append(
                    SyntheticSample(
                        id=sample_id,
                        instruction=degen["instruction"],
                        input_context=degen.get("input_context", ""),
                        response=degen["response"],
                        paradigm=paradigm,
                        domain=domain,
                        complexity="basic",
                        metrics=metrics
                    )
                )

        return samples


# ============================================================================
# 3. Multi-Stage Filter Pipeline Funnel
# ============================================================================

class QualityFilterPipeline:
    """
    Executes a multi-stage filtering funnel on synthetic datasets.
    """

    def __init__(self):
        self.scorer = QualityScoringEngine()

    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        """Computes word-level Jaccard similarity for deduplication."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return float(intersection) / float(union)

    def run_pipeline(
        self,
        samples: List[SyntheticSample],
        thresholds: FilterThresholds
    ) -> Dict[str, Any]:
        """
        Applies sequential quality filters and records funnel statistics:
        Stage 1: Length Bounds
        Stage 2: Repetition Loops
        Stage 3: PII Gate
        Stage 4: Perplexity Boundary
        Stage 5: Quality Score Rubric
        Stage 6: Near-Duplicate Deduplication
        """
        stages: List[FilterStageMetric] = []
        rejected_samples: List[SyntheticSample] = []
        current_pool = list(samples)
        total_input = len(current_pool)

        # Helper to record stage
        def record_stage(name: str, in_cnt: int, passed_cnt: int):
            rejected = in_cnt - passed_cnt
            retention = (passed_cnt / in_cnt * 100.0) if in_cnt > 0 else 0.0
            stages.append(
                FilterStageMetric(
                    stage_name=name,
                    input_count=in_cnt,
                    passed_count=passed_cnt,
                    rejected_count=rejected,
                    retention_rate=round(retention, 1)
                )
            )

        # Ensure all samples have metrics computed
        for s in current_pool:
            if not s.metrics:
                s.metrics = self.scorer.score_sample(s.instruction, s.response, s.input_context)

        # STAGE 1: Length Bounds
        passed_stage1 = []
        for s in current_pool:
            w_cnt = s.metrics.get("word_count", 0)
            if w_cnt < thresholds.min_words:
                s.rejection_stage = "Length Filter"
                s.rejection_reason = f"Kelime sayısı yetersiz ({w_cnt} < {thresholds.min_words})."
                rejected_samples.append(s)
            elif w_cnt > thresholds.max_words:
                s.rejection_stage = "Length Filter"
                s.rejection_reason = f"Kelime sayısı çok fazla ({w_cnt} > {thresholds.max_words})."
                rejected_samples.append(s)
            else:
                passed_stage1.append(s)
        record_stage("1. Uzunluk Filtresi", len(current_pool), len(passed_stage1))
        current_pool = passed_stage1

        # STAGE 2: Repetition Loops
        passed_stage2 = []
        for s in current_pool:
            rep_2g = s.metrics.get("repetition_ratio_2g", 0.0)
            if rep_2g > thresholds.max_repetition_ratio:
                s.rejection_stage = "Repetition Loop Filter"
                s.rejection_reason = f"Aşırı n-gram tekrarlama döngüsü (%{rep_2g * 100:.1f} > %{thresholds.max_repetition_ratio * 100:.1f})."
                rejected_samples.append(s)
            else:
                passed_stage2.append(s)
        record_stage("2. Tekrarlama & Döngü Filtresi", len(current_pool), len(passed_stage2))
        current_pool = passed_stage2

        # STAGE 3: PII Security Gate
        passed_stage3 = []
        for s in current_pool:
            pii_cnt = s.metrics.get("pii_count", 0)
            if pii_cnt > 0:
                if thresholds.pii_action == "reject":
                    s.rejection_stage = "PII Security Gate"
                    s.rejection_reason = f"Metinde {pii_cnt} adet hassas kişisel veri (PII) tespit edildi."
                    rejected_samples.append(s)
                elif thresholds.pii_action == "mask":
                    # Mask PII in response and allow
                    s.response = self.scorer.mask_pii(s.response)
                    s.instruction = self.scorer.mask_pii(s.instruction)
                    s.metrics["pii_masked"] = True
                    passed_stage3.append(s)
                else:
                    passed_stage3.append(s)
            else:
                passed_stage3.append(s)
        record_stage("3. PII & KVKK Güvenlik Kapısı", len(current_pool), len(passed_stage3))
        current_pool = passed_stage3

        # STAGE 4: Perplexity Boundary
        passed_stage4 = []
        for s in current_pool:
            ppl = s.metrics.get("perplexity", 50.0)
            if ppl < thresholds.min_perplexity:
                s.rejection_stage = "Perplexity Boundary"
                s.rejection_reason = f"Düşük Perplexity ({ppl:.1f} < {thresholds.min_perplexity}): Düşük entropili model çökmesi."
                rejected_samples.append(s)
            elif ppl > thresholds.max_perplexity:
                s.rejection_stage = "Perplexity Boundary"
                s.rejection_reason = f"Yüksek Perplexity ({ppl:.1f} > {thresholds.max_perplexity}): Dilsel tutarsızlık ve halüsinasyon riski."
                rejected_samples.append(s)
            else:
                passed_stage4.append(s)
        record_stage("4. Perplexity Akıcılık Sınırı", len(current_pool), len(passed_stage4))
        current_pool = passed_stage4

        # STAGE 5: Quality Score Rubric
        passed_stage5 = []
        for s in current_pool:
            q_score = s.metrics.get("composite_score", 0.0)
            if q_score < thresholds.min_quality_score:
                s.rejection_stage = "Quality Score Rubric"
                s.rejection_reason = f"Genel kalite puanı yetersiz ({q_score:.1f} < {thresholds.min_quality_score:.1f})."
                rejected_samples.append(s)
            else:
                passed_stage5.append(s)
        record_stage("5. Kalite Rubrik Eşiği", len(current_pool), len(passed_stage5))
        current_pool = passed_stage5

        # STAGE 6: Near-Duplicate Deduplication
        passed_stage6 = []
        seen_responses: List[str] = []
        for s in current_pool:
            is_dup = False
            for seen in seen_responses:
                sim = self._jaccard_similarity(s.response, seen)
                if sim >= thresholds.max_jaccard_similarity:
                    is_dup = True
                    s.rejection_stage = "Near-Duplicate Filter"
                    s.rejection_reason = f"Başka bir sentetik örnek ile yüksek benzerlik (%{sim * 100:.1f} Jaccard)."
                    rejected_samples.append(s)
                    break
            if not is_dup:
                seen_responses.append(s.response)
                passed_stage6.append(s)
        record_stage("6. Benzerlik & Tekilleştirme", len(current_pool), len(passed_stage6))
        current_pool = passed_stage6

        final_yield_pct = (len(current_pool) / total_input * 100.0) if total_input > 0 else 0.0

        # Quality improvements
        avg_initial_score = (
            sum(s.metrics.get("composite_score", 0.0) for s in samples) / total_input
            if total_input > 0 else 0.0
        )
        avg_final_score = (
            sum(s.metrics.get("composite_score", 0.0) for s in current_pool) / len(current_pool)
            if current_pool else 0.0
        )

        return {
            "total_input_count": total_input,
            "passed_count": len(current_pool),
            "rejected_count": len(rejected_samples),
            "final_yield_pct": round(final_yield_pct, 1),
            "avg_initial_quality": round(avg_initial_score, 1),
            "avg_final_quality": round(avg_final_score, 1),
            "funnel_stages": [asdict(st) for st in stages],
            "passed_samples": [s.to_dict() for s in current_pool],
            "rejected_samples": [s.to_dict() for s in rejected_samples],
        }
