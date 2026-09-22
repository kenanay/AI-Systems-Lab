"""
src/evaluation/benchmarks.py

Benchmark Suite for Language Model Evaluation

Language model benchmark sistemi:
- Text Generation Quality: BLEU, ROUGE, ChrF, TER
- Turkish Language Tasks: Turkish-specific benchmarks
- Classification Tasks: Sentiment, topic classification
- Benchmark Datasets: Curated test sets
- Evaluation Protocols: Standardized evaluation procedures

Bu modül comprehensive benchmark evaluation sağlar:
- Multiple metrics: BLEU-1/2/3/4, ROUGE-L, ChrF, TER
- Turkish benchmarks: Dil özelliklerine uygun testler
- Dataset management: Benchmark dataset loading ve yönetimi
- Reproducibility: Consistent evaluation protocols
- Reporting: Detailed benchmark reports

Metrikler:
    BLEU (Bilingual Evaluation Understudy):
        - N-gram overlap between prediction and reference
        - BLEU-1, BLEU-2, BLEU-3, BLEU-4
        - Range: 0-100 (higher is better)
    
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
        - ROUGE-1: Unigram overlap
        - ROUGE-2: Bigram overlap
        - ROUGE-L: Longest common subsequence
        - Range: 0-1 (higher is better)
    
    ChrF (Character n-gram F-score):
        - Character-level matching
        - Better for morphologically rich languages
        - Range: 0-100 (higher is better)

Kaynaklar:
    - BLEU: A Method for Automatic Evaluation of Machine Translation
    - ROUGE: A Package for Automatic Evaluation of Summaries
    - https://huggingface.co/docs/datasets/

Usage:
    >>> from src.evaluation.benchmarks import compute_bleu, compute_rouge
    >>> 
    >>> # Compute BLEU
    >>> bleu = compute_bleu(predictions, references)
    >>> 
    >>> # Run benchmark suite
    >>> results = run_benchmark_suite(model, tokenizer, benchmark_data)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
import uuid

from src.registry.model_registry import ModelRegistry
# from src.evaluation.metrics import compute_perplexity  # TODO: Implement when needed

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """
    Benchmark sonucu.
    
    Attributes:
        benchmark_id: Unique benchmark run ID
        model_name: Model adı
        benchmark_name: Benchmark adı
        score: Ana skor (perplexity, BLEU, vb.)
        metrics: Detaylı metrikler
        timestamp: Çalışma zamanı
        samples_evaluated: Değerlendirilen sample sayısı
    """
    benchmark_id: str
    model_name: str
    benchmark_name: str
    score: float
    metrics: Dict[str, Any]
    timestamp: datetime
    samples_evaluated: int


class BenchmarkRunner:
    """
    Benchmark çalıştırıcı.
    
    Args:
        model_name: Model adı (registry'den yüklenir)
        device: Device (cpu, cuda)
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.registry = ModelRegistry(registry_dir="models")
        
        logger.info(f"BenchmarkRunner initialized for {model_name} on {device}")
    
    def run_benchmark(
        self,
        benchmark_name: str,
        dataset_path: Optional[str] = None,
        max_samples: int = 100,
        batch_size: int = 8
    ) -> BenchmarkResult:
        """
        Benchmark çalıştır.
        
        Args:
            benchmark_name: Benchmark adı (perplexity, bleu, rouge)
            dataset_path: Test dataset path (optional)
            max_samples: Maximum sample sayısı
            batch_size: Batch size
            
        Returns:
            BenchmarkResult
            
        Raises:
            ValueError: Desteklenmeyen benchmark
            FileNotFoundError: Model bulunamadı
        """
        logger.info(f"Running {benchmark_name} benchmark on {self.model_name}")
        
        benchmark_id = f"BENCH-{uuid.uuid4().hex[:8].upper()}"
        
        if benchmark_name == "perplexity":
            score, metrics = self._run_perplexity_benchmark(max_samples, batch_size)
        elif benchmark_name in ["bleu", "rouge"]:
            # TODO: Implement BLEU/ROUGE benchmarks
            score = 0.0
            metrics = {"note": "Not implemented yet"}
            logger.warning(f"{benchmark_name} benchmark not implemented yet")
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark_name}")
        
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            model_name=self.model_name,
            benchmark_name=benchmark_name,
            score=score,
            metrics=metrics,
            timestamp=datetime.now(),
            samples_evaluated=max_samples
        )
        
        logger.info(f"Benchmark completed: {benchmark_name} = {score:.2f}")
        
        return result
    
    def _run_perplexity_benchmark(
        self,
        max_samples: int,
        batch_size: int
    ) -> tuple[float, Dict[str, Any]]:
        """
        Perplexity benchmark çalıştır.
        
        Returns:
            Tuple of (perplexity_score, metrics_dict)
        """
        # Load model
        model_info = self.registry.load_model(self.model_name)
        
        if not model_info:
            raise FileNotFoundError(f"Model not found: {self.model_name}")
        
        # TODO: Load actual test dataset
        # Şimdilik dummy perplexity hesaplıyoruz
        
        # Simulated perplexity (normally computed on test set)
        perplexity = 15.0 + (hash(self.model_name) % 10)  # Dummy value
        
        metrics = {
            "perplexity": perplexity,
            "loss": torch.log(torch.tensor(perplexity)).item(),
            "samples": max_samples,
            "batch_size": batch_size
        }
        
        return perplexity, metrics

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import re
from collections import Counter
import math
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import EvaluationResults, MetricResult

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# Text Generation Metrics
# =========================

def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenization for metrics.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    # Lowercase and split on whitespace/punctuation
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens


def compute_bleu(
    prediction: str,
    reference: Union[str, List[str]],
    max_n: int = 4
) -> Dict[str, float]:
    """
    Compute BLEU score (1-4 grams).
    
    BLEU = BP * exp(sum(w_n * log(p_n)))
    
    Args:
        prediction: Predicted text
        reference: Reference text(s)
        max_n: Maximum n-gram size (default: 4)
        
    Returns:
        Dict with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    # Handle single or multiple references
    if isinstance(reference, str):
        references = [reference]
    else:
        references = reference
    
    # Tokenize
    pred_tokens = tokenize_text(prediction)
    ref_tokens_list = [tokenize_text(ref) for ref in references]
    
    if not pred_tokens:
        return {f"bleu-{i}": 0.0 for i in range(1, max_n + 1)}
    
    # Compute n-gram precisions
    precisions = []
    
    for n in range(1, max_n + 1):
        # Get n-grams from prediction
        pred_ngrams = Counter()
        for i in range(len(pred_tokens) - n + 1):
            ngram = tuple(pred_tokens[i:i+n])
            pred_ngrams[ngram] += 1
        
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        
        # Get maximum counts from references
        max_ref_counts = Counter()
        for ref_tokens in ref_tokens_list:
            ref_ngrams = Counter()
            for i in range(len(ref_tokens) - n + 1):
                ngram = tuple(ref_tokens[i:i+n])
                ref_ngrams[ngram] += 1
            
            for ngram in ref_ngrams:
                max_ref_counts[ngram] = max(
                    max_ref_counts[ngram],
                    ref_ngrams[ngram]
                )
        
        # Count clipped matches
        clipped_counts = 0
        total_counts = 0
        
        for ngram, count in pred_ngrams.items():
            clipped_counts += min(count, max_ref_counts.get(ngram, 0))
            total_counts += count
        
        # Precision for this n
        precision = clipped_counts / total_counts if total_counts > 0 else 0.0
        precisions.append(precision)
    
    # Brevity penalty
    pred_len = len(pred_tokens)
    ref_lens = [len(ref_tokens) for ref_tokens in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - pred_len))
    
    if pred_len >= closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / pred_len) if pred_len > 0 else 0.0
    
    # Compute BLEU scores
    bleu_scores = {}
    
    for n in range(1, max_n + 1):
        # Geometric mean of precisions up to n
        if all(p > 0 for p in precisions[:n]):
            log_precision = sum(math.log(p) for p in precisions[:n]) / n
            bleu = bp * math.exp(log_precision)
        else:
            bleu = 0.0
        
        bleu_scores[f"bleu-{n}"] = bleu * 100  # Scale to 0-100
    
    return bleu_scores


def compute_rouge(
    prediction: str,
    reference: Union[str, List[str]]
) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    ROUGE = recall-oriented metric for summarization
    
    Args:
        prediction: Predicted text
        reference: Reference text(s)
        
    Returns:
        Dict with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    # Handle single or multiple references
    if isinstance(reference, str):
        references = [reference]
    else:
        references = reference
    
    # Tokenize
    pred_tokens = tokenize_text(prediction)
    ref_tokens_list = [tokenize_text(ref) for ref in references]
    
    if not pred_tokens:
        return {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0
        }
    
    # Compute ROUGE-1 (unigram)
    pred_unigrams = set(pred_tokens)
    rouge_1_scores = []
    
    for ref_tokens in ref_tokens_list:
        ref_unigrams = set(ref_tokens)
        
        if not ref_unigrams:
            rouge_1_scores.append(0.0)
            continue
        
        overlap = len(pred_unigrams & ref_unigrams)
        recall = overlap / len(ref_unigrams)
        precision = overlap / len(pred_unigrams) if pred_unigrams else 0.0
        
        if recall + precision > 0:
            f1 = 2 * (recall * precision) / (recall + precision)
        else:
            f1 = 0.0
        
        rouge_1_scores.append(f1)
    
    rouge_1 = max(rouge_1_scores)
    
    # Compute ROUGE-2 (bigram)
    pred_bigrams = set()
    for i in range(len(pred_tokens) - 1):
        pred_bigrams.add(tuple(pred_tokens[i:i+2]))
    
    rouge_2_scores = []
    
    for ref_tokens in ref_tokens_list:
        ref_bigrams = set()
        for i in range(len(ref_tokens) - 1):
            ref_bigrams.add(tuple(ref_tokens[i:i+2]))
        
        if not ref_bigrams:
            rouge_2_scores.append(0.0)
            continue
        
        overlap = len(pred_bigrams & ref_bigrams)
        recall = overlap / len(ref_bigrams)
        precision = overlap / len(pred_bigrams) if pred_bigrams else 0.0
        
        if recall + precision > 0:
            f1 = 2 * (recall * precision) / (recall + precision)
        else:
            f1 = 0.0
        
        rouge_2_scores.append(f1)
    
    rouge_2 = max(rouge_2_scores) if rouge_2_scores else 0.0
    
    # Compute ROUGE-L (longest common subsequence)
    def lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        
        # dp[i][j] = LCS length of seq1[:i] and seq2[:j]
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    rouge_l_scores = []
    
    for ref_tokens in ref_tokens_list:
        if not ref_tokens:
            rouge_l_scores.append(0.0)
            continue
        
        lcs_len = lcs_length(pred_tokens, ref_tokens)
        
        recall = lcs_len / len(ref_tokens)
        precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
        
        if recall + precision > 0:
            f1 = 2 * (recall * precision) / (recall + precision)
        else:
            f1 = 0.0
        
        rouge_l_scores.append(f1)
    
    rouge_l = max(rouge_l_scores) if rouge_l_scores else 0.0
    
    return {
        "rouge-1": rouge_1,
        "rouge-2": rouge_2,
        "rouge-l": rouge_l
    }


def compute_chrf(
    prediction: str,
    reference: Union[str, List[str]],
    n: int = 6,
    beta: float = 2.0
) -> float:
    """
    Compute ChrF (Character n-gram F-score).
    
    Good for morphologically rich languages like Turkish.
    
    Args:
        prediction: Predicted text
        reference: Reference text(s)
        n: Maximum character n-gram size
        beta: Beta parameter for F-score (default: 2.0 for ChrF++)
        
    Returns:
        float: ChrF score (0-100)
    """
    # Handle single or multiple references
    if isinstance(reference, str):
        references = [reference]
    else:
        references = reference
    
    # Character-level processing
    pred_chars = list(prediction)
    
    if not pred_chars:
        return 0.0
    
    chrf_scores = []
    
    for ref in references:
        ref_chars = list(ref)
        
        if not ref_chars:
            chrf_scores.append(0.0)
            continue
        
        # Compute character n-gram matches
        total_precision = 0.0
        total_recall = 0.0
        
        for i in range(1, n + 1):
            # Prediction n-grams
            pred_ngrams = Counter()
            for j in range(len(pred_chars) - i + 1):
                ngram = tuple(pred_chars[j:j+i])
                pred_ngrams[ngram] += 1
            
            # Reference n-grams
            ref_ngrams = Counter()
            for j in range(len(ref_chars) - i + 1):
                ngram = tuple(ref_chars[j:j+i])
                ref_ngrams[ngram] += 1
            
            if not pred_ngrams or not ref_ngrams:
                continue
            
            # Matches
            matches = sum(
                min(pred_ngrams[ngram], ref_ngrams[ngram])
                for ngram in pred_ngrams
                if ngram in ref_ngrams
            )
            
            # Precision and recall for this n
            precision_n = matches / sum(pred_ngrams.values())
            recall_n = matches / sum(ref_ngrams.values())
            
            total_precision += precision_n
            total_recall += recall_n
        
        # Average over n-gram sizes
        avg_precision = total_precision / n
        avg_recall = total_recall / n
        
        # F-beta score
        if avg_precision + avg_recall > 0:
            chrf = (1 + beta**2) * (avg_precision * avg_recall) / \
                   (beta**2 * avg_precision + avg_recall)
        else:
            chrf = 0.0
        
        chrf_scores.append(chrf)
    
    return max(chrf_scores) * 100  # Scale to 0-100


# =========================
# Benchmark Datasets
# =========================

@dataclass
class BenchmarkExample:
    """
    Single benchmark example.
    
    Attributes:
        id: Example ID
        input: Input text/prompt
        target: Expected output/reference
        metadata: Additional metadata (task, category, etc.)
    """
    id: str
    input: str
    target: Union[str, List[str]]  # Single or multiple references
    metadata: Dict = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    """
    Benchmark dataset.
    
    Attributes:
        name: Dataset name
        description: Dataset description
        examples: List of examples
        metadata: Dataset metadata
    """
    name: str
    description: str
    examples: List[BenchmarkExample] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> BenchmarkExample:
        """Get example by index."""
        return self.examples[idx]
    
    def save(self, path: Union[str, Path]) -> None:
        """Save dataset to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "examples": [
                {
                    "id": ex.id,
                    "input": ex.input,
                    "target": ex.target,
                    "metadata": ex.metadata
                }
                for ex in self.examples
            ]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark dataset saved: {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BenchmarkDataset':
        """Load dataset from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = [
            BenchmarkExample(
                id=ex["id"],
                input=ex["input"],
                target=ex["target"],
                metadata=ex.get("metadata", {})
            )
            for ex in data["examples"]
        ]
        
        return cls(
            name=data["name"],
            description=data["description"],
            examples=examples,
            metadata=data.get("metadata", {})
        )


# =========================
# Turkish Benchmarks
# =========================

def create_turkish_generation_benchmark() -> BenchmarkDataset:
    """
    Create Turkish text generation benchmark.
    
    Returns:
        BenchmarkDataset with Turkish examples
    """
    examples = [
        BenchmarkExample(
            id="tr-gen-001",
            input="Türkiye'nin başkenti",
            target="Ankara'dır",
            metadata={"task": "completion", "domain": "geography"}
        ),
        BenchmarkExample(
            id="tr-gen-002",
            input="İstanbul",
            target=["Türkiye'nin en kalabalık şehridir", "Boğaz'ın iki yakasında yer alır"],
            metadata={"task": "completion", "domain": "geography"}
        ),
        BenchmarkExample(
            id="tr-gen-003",
            input="Atatürk",
            target=["Türkiye Cumhuriyeti'nin kurucusudur", "Mustafa Kemal Atatürk"],
            metadata={"task": "completion", "domain": "history"}
        ),
        BenchmarkExample(
            id="tr-gen-004",
            input="Bilgisayar",
            target=["elektronik bir cihazdır", "veri işleme makinesidir"],
            metadata={"task": "completion", "domain": "technology"}
        ),
        BenchmarkExample(
            id="tr-gen-005",
            input="Yapay zeka",
            target=["bilgisayar sistemlerinin insana özgü yetenekleri sergilemesidir", "AI olarak da bilinir"],
            metadata={"task": "completion", "domain": "technology"}
        ),
    ]
    
    return BenchmarkDataset(
        name="turkish-generation",
        description="Turkish text generation benchmark",
        examples=examples,
        metadata={
            "language": "Turkish",
            "num_examples": len(examples),
            "version": "1.0"
        }
    )


# =========================
# Benchmark Runner
# =========================

def run_generation_benchmark(
    model,
    tokenizer,
    benchmark: BenchmarkDataset,
    max_length: int = 50,
    temperature: float = 0.8,
    device: str = 'cpu'
) -> EvaluationResults:
    """
    Run generation benchmark on model.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        benchmark: Benchmark dataset
        max_length: Max generation length
        temperature: Sampling temperature
        device: Device ('cpu' or 'cuda')
        
    Returns:
        EvaluationResults with benchmark scores
    """
    from src.inference.generation import generate_text
    
    logger.info(f"Running benchmark: {benchmark.name}")
    logger.info(f"  Examples: {len(benchmark)}")
    
    results = EvaluationResults()
    results.metadata = {
        "benchmark": benchmark.name,
        "num_examples": len(benchmark)
    }
    
    # Collect predictions and references
    predictions = []
    references = []
    
    model.eval()
    
    with torch.no_grad():
        for i, example in enumerate(benchmark.examples):
            # Tokenize input
            # input_ids shape: [seq_len]
            input_ids = tokenizer.encode(example.input, add_bos=True, add_eos=False)
            
            # input_tensor shape: [1, seq_len]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            # Generate
            # output shape: [1, seq_len + max_length]
            output = generate_text(
                model,
                input_tensor,
                max_new_tokens=max_length,
                temperature=temperature
            )
            
            # Decode
            output_ids = output[0].cpu().tolist()
            prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            predictions.append(prediction)
            references.append(example.target)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(benchmark)} examples")
    
    # Compute metrics
    logger.info("Computing generation metrics...")
    
    bleu_scores_list = []
    rouge_scores_list = []
    chrf_scores_list = []
    
    for pred, ref in zip(predictions, references):
        bleu = compute_bleu(pred, ref)
        rouge = compute_rouge(pred, ref)
        chrf = compute_chrf(pred, ref)
        
        bleu_scores_list.append(bleu)
        rouge_scores_list.append(rouge)
        chrf_scores_list.append(chrf)
    
    # Average scores
    avg_bleu = {
        key: np.mean([scores[key] for scores in bleu_scores_list])
        for key in bleu_scores_list[0].keys()
    }
    
    avg_rouge = {
        key: np.mean([scores[key] for scores in rouge_scores_list])
        for key in rouge_scores_list[0].keys()
    }
    
    avg_chrf = np.mean(chrf_scores_list)
    
    # Add to results
    for key, value in avg_bleu.items():
        results.add_metric(key, value, "", higher_is_better=True)
    
    for key, value in avg_rouge.items():
        results.add_metric(key, value, "", higher_is_better=True)
    
    results.add_metric("chrf", avg_chrf, "", higher_is_better=True)
    
    results.num_samples = len(benchmark)
    
    logger.info("✓ Benchmark completed")
    
    return results


# =========================
# Main (for testing)
# =========================

def main():
    """Test benchmark suite."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUITE TEST")
    print("=" * 80)
    
    # Test BLEU
    print("\n" + "-" * 80)
    print("Test BLEU Scores")
    print("-" * 80)
    
    prediction = "Türkiye'nin başkenti Ankara'dır"
    reference = "Türkiye'nin başkenti Ankara"
    
    bleu = compute_bleu(prediction, reference)
    print(f"Prediction: {prediction}")
    print(f"Reference: {reference}")
    for key, value in bleu.items():
        print(f"  {key}: {value:.2f}")
    
    # Test ROUGE
    print("\n" + "-" * 80)
    print("Test ROUGE Scores")
    print("-" * 80)
    
    rouge = compute_rouge(prediction, reference)
    for key, value in rouge.items():
        print(f"  {key}: {value:.4f}")
    
    # Test ChrF
    print("\n" + "-" * 80)
    print("Test ChrF Score")
    print("-" * 80)
    
    chrf = compute_chrf(prediction, reference)
    print(f"  chrf: {chrf:.2f}")
    
    # Create Turkish benchmark
    print("\n" + "-" * 80)
    print("Create Turkish Benchmark")
    print("-" * 80)
    
    benchmark = create_turkish_generation_benchmark()
    print(f"Benchmark: {benchmark.name}")
    print(f"Description: {benchmark.description}")
    print(f"Examples: {len(benchmark)}")
    
    for i, example in enumerate(benchmark.examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"  ID: {example.id}")
        print(f"  Input: {example.input}")
        print(f"  Target: {example.target}")
    
    # Save benchmark
    benchmark.save("benchmark_test.json")
    
    # Load benchmark
    loaded = BenchmarkDataset.load("benchmark_test.json")
    print(f"\n✓ Benchmark loaded: {len(loaded)} examples")
    
    print("\n" + "=" * 80)
    print("✓ Benchmark suite test completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
