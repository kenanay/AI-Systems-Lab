"""
src/evaluation/metrics.py

Core Evaluation Metrics for Language Models

Language model evaluation için temel metrikler:
- Perplexity: Model'in prediction kalitesi
- Accuracy: Token-level doğruluk
- Loss: Cross-entropy loss hesaplama
- Top-K Accuracy: Top-k predictions içinde doğru token
- Confidence: Prediction confidence scores
- Entropy: Prediction entropy (model certainty)

Bu modül production-ready metric computation sağlar:
- Batch processing: Verimli batch-wise hesaplama
- Numerical stability: Log-space hesaplamalar
- Multiple metrics: Comprehensive evaluation
- Aggregation: Batch results aggregation
- Statistical analysis: Mean, std, confidence intervals

Formüller:
    Perplexity = exp(cross_entropy_loss)
    Accuracy = correct_predictions / total_predictions
    Top-K Accuracy = predictions_in_top_k / total_predictions
    Entropy = -sum(p * log(p))

Kaynaklar:
    - Language Modeling Evaluation Metrics
    - Perplexity and Cross-Entropy
    - https://huggingface.co/docs/transformers/perplexity

Usage:
    >>> from src.evaluation.metrics import compute_perplexity, compute_accuracy
    >>> 
    >>> # Compute perplexity
    >>> perplexity = compute_perplexity(logits, targets)
    >>> 
    >>> # Compute accuracy
    >>> accuracy = compute_accuracy(predictions, targets)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Sequence
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import re
import math
from collections import Counter

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# Metric Results
# =========================

@dataclass
class MetricResult:
    """
    Single metric result.
    
    Attributes:
        name: Metric name
        value: Metric value
        unit: Unit (e.g., '%', 'bits')
        higher_is_better: True if higher values are better
        metadata: Additional metadata
    """
    name: str
    value: float
    unit: str = ""
    higher_is_better: bool = True
    metadata: Dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation."""
        direction = "↑" if self.higher_is_better else "↓"
        return f"{self.name}: {self.value:.4f}{self.unit} {direction}"


@dataclass
class EvaluationResults:
    """
    Complete evaluation results.
    
    Attributes:
        metrics: List of metric results
        num_samples: Number of samples evaluated
        num_tokens: Number of tokens evaluated
        metadata: Additional metadata
    """
    metrics: List[MetricResult] = field(default_factory=list)
    num_samples: int = 0
    num_tokens: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def add_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        higher_is_better: bool = True,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add a metric result."""
        self.metrics.append(MetricResult(
            name=name,
            value=value,
            unit=unit,
            higher_is_better=higher_is_better,
            metadata=metadata or {}
        ))
    
    def get_metric(self, name: str) -> Optional[MetricResult]:
        """Get metric by name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "metrics": {
                m.name: {
                    "value": m.value,
                    "unit": m.unit,
                    "higher_is_better": m.higher_is_better,
                    "metadata": m.metadata
                }
                for m in self.metrics
            },
            "num_samples": self.num_samples,
            "num_tokens": self.num_tokens,
            "metadata": self.metadata
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {path}")
    
    def __str__(self) -> str:
        """String representation."""
        lines = ["Evaluation Results:"]
        lines.append(f"  Samples: {self.num_samples:,}")
        lines.append(f"  Tokens: {self.num_tokens:,}")
        lines.append("  Metrics:")
        for metric in self.metrics:
            lines.append(f"    {metric}")
        return "\n".join(lines)


# =========================
# Core Metrics
# =========================

def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute perplexity from logits and targets.
    
    Perplexity = exp(cross_entropy_loss)
    
    Lower perplexity = better model (daha iyi tahmin)
    
    Args:
        logits: Model predictions, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        ignore_index: Index to ignore (e.g., padding)
        
    Returns:
        float: Perplexity value
        
    Example:
        >>> logits = torch.randn(2, 10, 1000)  # [B, T, V]
        >>> targets = torch.randint(0, 1000, (2, 10))  # [B, T]
        >>> perplexity = compute_perplexity(logits, targets)
    """
    # logits shape: [B, T, V]
    # targets shape: [B, T]
    
    # Flatten for loss computation
    # logits_flat shape: [B*T, V]
    logits_flat = logits.reshape(-1, logits.size(-1))
    
    # targets_flat shape: [B*T]
    targets_flat = targets.reshape(-1)
    
    # Compute cross-entropy loss
    # loss is scalar
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()
    
    return perplexity


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute token-level accuracy.
    
    Accuracy = correct_predictions / total_predictions
    
    Args:
        predictions: Predicted token IDs, shape [batch, seq_len]
        targets: Target token IDs, shape [batch, seq_len]
        ignore_index: Index to ignore (e.g., padding)
        
    Returns:
        float: Accuracy (0.0 to 1.0)
    """
    # predictions shape: [B, T]
    # targets shape: [B, T]
    
    # Create mask for valid positions
    # mask shape: [B, T]
    mask = (targets != ignore_index)
    
    # Check correct predictions
    # correct shape: [B, T]
    correct = (predictions == targets) & mask
    
    # Calculate accuracy
    accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
    
    return accuracy


def compute_top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
    ignore_index: int = -100
) -> float:
    """
    Compute top-k accuracy.
    
    Top-K Accuracy = predictions_in_top_k / total_predictions
    
    Args:
        logits: Model predictions, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        k: Top-k value
        ignore_index: Index to ignore
        
    Returns:
        float: Top-k accuracy (0.0 to 1.0)
    """
    # logits shape: [B, T, V]
    # targets shape: [B, T]
    
    # Get top-k predictions
    # top_k_preds shape: [B, T, k]
    _, top_k_preds = torch.topk(logits, k, dim=-1)
    
    # Create mask for valid positions
    # mask shape: [B, T]
    mask = (targets != ignore_index)
    
    # Expand targets for comparison
    # targets_expanded shape: [B, T, 1]
    targets_expanded = targets.unsqueeze(-1)
    
    # Check if target is in top-k
    # in_top_k shape: [B, T, k]
    in_top_k = (top_k_preds == targets_expanded)
    
    # Any match in top-k?
    # matches shape: [B, T]
    matches = in_top_k.any(dim=-1) & mask
    
    # Calculate accuracy
    accuracy = matches.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
    
    return accuracy


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Model predictions, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        ignore_index: Index to ignore
        
    Returns:
        float: Loss value
    """
    # logits shape: [B, T, V]
    # targets shape: [B, T]
    
    # Flatten
    logits_flat = logits.reshape(-1, logits.size(-1))  # [B*T, V]
    targets_flat = targets.reshape(-1)  # [B*T]
    
    # Compute loss
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    return loss.item()


def compute_confidence_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Compute confidence-related metrics.
    
    Returns:
        Dict with:
            - mean_confidence: Average confidence for correct predictions
            - mean_max_prob: Average max probability
            - calibration_error: Simple calibration error
    
    Args:
        logits: Model predictions, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        ignore_index: Index to ignore
    """
    # logits shape: [B, T, V]
    # targets shape: [B, T]
    
    # Get probabilities
    # probs shape: [B, T, V]
    probs = F.softmax(logits, dim=-1)
    
    # Get max probabilities
    # max_probs shape: [B, T]
    max_probs, predictions = torch.max(probs, dim=-1)
    
    # Create mask
    mask = (targets != ignore_index)
    
    # Get target probabilities (confidence for correct class)
    # target_probs shape: [B, T]
    target_probs = torch.gather(probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    
    # Calculate metrics on valid positions
    valid_max_probs = max_probs[mask]
    valid_target_probs = target_probs[mask]
    valid_correct = (predictions == targets)[mask]
    
    if mask.sum().item() == 0:
        return {
            "mean_confidence": 0.0,
            "mean_max_prob": 0.0,
            "calibration_error": 0.0
        }
    
    # Mean confidence for correct predictions
    mean_confidence = valid_target_probs.mean().item()
    
    # Mean max probability
    mean_max_prob = valid_max_probs.mean().item()
    
    # Simple calibration error (difference between confidence and accuracy)
    accuracy = valid_correct.float().mean().item()
    calibration_error = abs(mean_max_prob - accuracy)
    
    return {
        "mean_confidence": mean_confidence,
        "mean_max_prob": mean_max_prob,
        "calibration_error": calibration_error
    }


def compute_entropy(
    logits: torch.Tensor,
    ignore_index: int = -100,
    targets: Optional[torch.Tensor] = None
) -> float:
    """
    Compute average prediction entropy.
    
    Entropy = -sum(p * log(p))
    
    Lower entropy = more confident predictions
    
    Args:
        logits: Model predictions, shape [batch, seq_len, vocab_size]
        ignore_index: Index to ignore
        targets: Optional targets for masking, shape [batch, seq_len]
        
    Returns:
        float: Average entropy
    """
    # logits shape: [B, T, V]
    
    # Get probabilities
    # probs shape: [B, T, V]
    probs = F.softmax(logits, dim=-1)
    
    # Compute entropy: -sum(p * log(p))
    # entropy shape: [B, T]
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    
    if targets is not None:
        # Mask invalid positions
        mask = (targets != ignore_index)
        entropy = entropy[mask]
    
    # Average entropy
    avg_entropy = entropy.mean().item()
    
    return avg_entropy


# =========================
# Batch Evaluation
# =========================

def evaluate_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    compute_all: bool = True
) -> EvaluationResults:
    """
    Evaluate a single batch with all metrics.
    
    Args:
        logits: Model predictions, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        ignore_index: Index to ignore
        compute_all: Compute all metrics (slower but comprehensive)
        
    Returns:
        EvaluationResults with all metrics
    """
    results = EvaluationResults()
    
    # Count samples and tokens
    results.num_samples = targets.size(0)
    
    mask = (targets != ignore_index)
    results.num_tokens = int(mask.sum().item())
    
    # Get predictions
    # predictions shape: [B, T]
    predictions = torch.argmax(logits, dim=-1)
    
    # Core metrics
    loss = compute_loss(logits, targets, ignore_index)
    perplexity = compute_perplexity(logits, targets, ignore_index)
    accuracy = compute_accuracy(predictions, targets, ignore_index)
    
    results.add_metric("loss", loss, "", higher_is_better=False)
    results.add_metric("perplexity", perplexity, "", higher_is_better=False)
    results.add_metric("accuracy", accuracy, "%", higher_is_better=True)
    
    if compute_all:
        # Top-k accuracy
        top5_acc = compute_top_k_accuracy(logits, targets, k=5, ignore_index=ignore_index)
        top10_acc = compute_top_k_accuracy(logits, targets, k=10, ignore_index=ignore_index)
        
        results.add_metric("top5_accuracy", top5_acc, "%", higher_is_better=True)
        results.add_metric("top10_accuracy", top10_acc, "%", higher_is_better=True)
        
        # Confidence scores
        confidence_scores = compute_confidence_scores(logits, targets, ignore_index)
        results.add_metric("mean_confidence", confidence_scores["mean_confidence"], "", higher_is_better=True)
        results.add_metric("mean_max_prob", confidence_scores["mean_max_prob"], "", higher_is_better=True)
        results.add_metric("calibration_error", confidence_scores["calibration_error"], "", higher_is_better=False)
        
        # Entropy
        entropy = compute_entropy(logits, ignore_index, targets)
        results.add_metric("entropy", entropy, "bits", higher_is_better=False)
    
    return results


def aggregate_results(results_list: List[EvaluationResults]) -> EvaluationResults:
    """
    Aggregate multiple evaluation results.
    
    Args:
        results_list: List of EvaluationResults
        
    Returns:
        Aggregated EvaluationResults (weighted by num_tokens)
    """
    if not results_list:
        return EvaluationResults()
    
    # Collect all metric names
    metric_names = set()
    for results in results_list:
        for metric in results.metrics:
            metric_names.add(metric.name)
    
    # Aggregate each metric (weighted by num_tokens)
    aggregated = EvaluationResults()
    
    total_samples = sum(r.num_samples for r in results_list)
    total_tokens = sum(r.num_tokens for r in results_list)
    
    aggregated.num_samples = total_samples
    aggregated.num_tokens = total_tokens
    
    for metric_name in metric_names:
        # Collect values and weights
        values = []
        weights = []
        
        for results in results_list:
            metric = results.get_metric(metric_name)
            if metric is not None:
                values.append(metric.value)
                weights.append(results.num_tokens)
        
        if values:
            # Weighted average
            weighted_value = np.average(values, weights=weights)
            
            # Get metadata from first occurrence
            unit = ""
            higher_is_better = True
            for r in results_list:
                m = r.get_metric(metric_name)
                if m is not None:
                    unit = m.unit
                    higher_is_better = m.higher_is_better
                    break
            
            aggregated.add_metric(
                metric_name,
                weighted_value,
                unit,
                higher_is_better,
                {"std": float(np.std(values))}
            )
    
    return aggregated


# =========================
# Text Generation Metrics (BLEU & ROUGE)
# =========================

def tokenize_text(text: str) -> List[str]:
    """
    Metin tokenizasyonu (küçük harfe çevirip alfanümerik token'lara böler).
    
    Args:
        text: Girdi metni
        
    Returns:
        Token listesi
    """
    if not text:
        return []
    text = text.lower()
    return re.findall(r'\w+', text)


def compute_bleu(
    prediction: str,
    reference: Union[str, Sequence[str]],
    max_n: int = 4
) -> Dict[str, float]:
    """
    BLEU (Bilingual Evaluation Understudy) skorlarını hesaplar (BLEU-1 .. BLEU-4 ve kümülatif BLEU).
    
    Formül:
        BLEU = BP * exp(sum(w_n * log(p_n)))
        BP = min(1.0, exp(1 - r / c))
        
    Args:
        prediction: Modelin ürettiği metin
        reference: Referans metin veya referans metinler listesi
        max_n: Maksimum n-gram mertebesi (varsayılan: 4)
        
    Returns:
        Dict: {"bleu-1": float, ..., "bleu-4": float, "bleu": float} (0.0 - 100.0 aralığında)
    """
    if isinstance(reference, str):
        references = [reference]
    else:
        references = reference
        
    pred_tokens = tokenize_text(prediction)
    ref_tokens_list = [tokenize_text(ref) for ref in references if ref]
    
    if not pred_tokens or not ref_tokens_list:
        res = {f"bleu-{i}": 0.0 for i in range(1, max_n + 1)}
        res["bleu"] = 0.0
        return res
        
    precisions = []
    
    for n in range(1, max_n + 1):
        pred_ngrams: Counter[Tuple[str, ...]] = Counter()
        for i in range(len(pred_tokens) - n + 1):
            ngram = tuple(pred_tokens[i:i+n])
            pred_ngrams[ngram] += 1
            
        if not pred_ngrams:
            precisions.append(0.0)
            continue
            
        max_ref_counts: Counter[Tuple[str, ...]] = Counter()
        for ref_tokens in ref_tokens_list:
            ref_ngrams: Counter[Tuple[str, ...]] = Counter()
            for i in range(len(ref_tokens) - n + 1):
                ngram = tuple(ref_tokens[i:i+n])
                ref_ngrams[ngram] += 1
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
                
        clipped_counts = 0
        total_counts = 0
        for ngram, count in pred_ngrams.items():
            clipped_counts += min(count, max_ref_counts.get(ngram, 0))
            total_counts += count
            
        precision = clipped_counts / total_counts if total_counts > 0 else 0.0
        precisions.append(precision)
        
    # Brevity Penalty
    pred_len = len(pred_tokens)
    ref_lens = [len(ref_tokens) for ref_tokens in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - pred_len))
    
    if pred_len >= closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - closest_ref_len / pred_len) if pred_len > 0 else 0.0
        
    bleu_scores: Dict[str, float] = {}
    for n in range(1, max_n + 1):
        if all(p > 0 for p in precisions[:n]):
            log_prec = sum(math.log(p) for p in precisions[:n]) / n
            score = bp * math.exp(log_prec)
        else:
            score = 0.0
        bleu_scores[f"bleu-{n}"] = round(score * 100.0, 2)
        
    bleu_scores["bleu"] = bleu_scores.get(f"bleu-{min(4, max_n)}", 0.0)
    return bleu_scores


def compute_rouge(
    prediction: str,
    reference: Union[str, Sequence[str]]
) -> Dict[str, float]:
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) skorlarını hesaplar:
    - ROUGE-1 (Unigram F1)
    - ROUGE-2 (Bigram F1)
    - ROUGE-L (Longest Common Subsequence F1)
    
    Args:
        prediction: Modelin ürettiği metin
        reference: Referans metin(ler)
        
    Returns:
        Dict: {"rouge-1": float, "rouge-2": float, "rouge-l": float} (0.0 - 100.0 aralığında)
    """
    if isinstance(reference, str):
        references = [reference]
    else:
        references = reference
        
    pred_tokens = tokenize_text(prediction)
    ref_tokens_list = [tokenize_text(ref) for ref in references if ref]
    
    if not pred_tokens or not ref_tokens_list:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        
    # ROUGE-1
    pred_unigrams = Counter(pred_tokens)
    rouge_1_scores = []
    for ref_tokens in ref_tokens_list:
        ref_unigrams = Counter(ref_tokens)
        if not ref_unigrams:
            continue
        overlap = sum((pred_unigrams & ref_unigrams).values())
        recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        precision = overlap / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0
        rouge_1_scores.append(f1)
        
    rouge_1 = max(rouge_1_scores) if rouge_1_scores else 0.0
    
    # ROUGE-2
    pred_bigrams: Counter[Tuple[str, str]] = Counter()
    for i in range(len(pred_tokens) - 1):
        pred_bigrams[(pred_tokens[i], pred_tokens[i+1])] += 1
        
    rouge_2_scores = []
    for ref_tokens in ref_tokens_list:
        ref_bigrams: Counter[Tuple[str, str]] = Counter()
        for i in range(len(ref_tokens) - 1):
            ref_bigrams[(ref_tokens[i], ref_tokens[i+1])] += 1
        if not ref_bigrams:
            continue
        overlap = sum((pred_bigrams & ref_bigrams).values())
        total_ref_bigrams = max(len(ref_tokens) - 1, 0)
        total_pred_bigrams = max(len(pred_tokens) - 1, 0)
        recall = overlap / total_ref_bigrams if total_ref_bigrams > 0 else 0.0
        precision = overlap / total_pred_bigrams if total_pred_bigrams > 0 else 0.0
        f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0
        rouge_2_scores.append(f1)
        
    rouge_2 = max(rouge_2_scores) if rouge_2_scores else 0.0
    
    # ROUGE-L (LCS)
    def lcs_len(s1: List[str], s2: List[str]) -> int:
        m, n = len(s1), len(s2)
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i-1] == s2[j-1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j-1])
                prev = temp
        return dp[n]
        
    rouge_l_scores = []
    for ref_tokens in ref_tokens_list:
        if not ref_tokens:
            continue
        lcs = lcs_len(pred_tokens, ref_tokens)
        recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        precision = lcs / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0
        rouge_l_scores.append(f1)
        
    rouge_l = max(rouge_l_scores) if rouge_l_scores else 0.0
    
    return {
        "rouge-1": round(rouge_1 * 100.0, 2),
        "rouge-2": round(rouge_2 * 100.0, 2),
        "rouge-l": round(rouge_l * 100.0, 2)
    }


def normalize_text_for_eval(text: str) -> str:
    """
    Değerlendirme için metin normalizasyonu:
    - Küçük harfe dönüştürme (lowercase)
    - Noktalama işaretlerini ayıklama
    - Fazla boşlukları temizleme
    - Türkçe karakterleri (ç, ğ, ı, ö, ş, ü) koruma
    """
    text = str(text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def compute_exact_match(
    prediction: str,
    reference: Union[str, Sequence[str]]
) -> float:
    """
    Normalize edilmiş metinler arasında Exact Match (EM) tam eşleşme skoru.
    Eğer tahmin, referanslardan herhangi biriyle birebir eşleşirse 100.0, aksi halde 0.0 döner.
    """
    refs = [reference] if isinstance(reference, str) else list(reference)
    norm_pred = normalize_text_for_eval(prediction)
    for r in refs:
        if norm_pred == normalize_text_for_eval(r):
            return 100.0
    return 0.0


def compute_token_f1(
    prediction: str,
    reference: Union[str, Sequence[str]]
) -> Dict[str, float]:
    """
    SQuAD / QA tarzı token düzeyinde Precision, Recall ve F1 skoru.
    Birden fazla referans varsa en yüksek F1 skorunu veren referans baz alınır.
    """
    refs = [reference] if isinstance(reference, str) else list(reference)
    norm_pred = normalize_text_for_eval(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0

    pred_counts = Counter(pred_tokens)

    for r in refs:
        ref_tokens = normalize_text_for_eval(r).split()
        if not ref_tokens:
            continue
        ref_counts = Counter(ref_tokens)
        common = sum((pred_counts & ref_counts).values())
        if common == 0:
            continue
        p = common / len(pred_tokens)
        rec = common / len(ref_tokens)
        f1 = (2 * p * rec) / (p + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_p = p
            best_r = rec

    return {
        "precision": round(best_p * 100.0, 2),
        "recall": round(best_r * 100.0, 2),
        "f1": round(best_f1 * 100.0, 2)
    }


def compute_chrf(
    prediction: str,
    reference: Union[str, Sequence[str]],
    n: int = 6,
    beta: float = 2.0
) -> float:
    """
    ChrF (Character n-gram F-score):
    Türkçe gibi eklemeli ve morfolojik olarak zengin diller için karakter n-gram eşleşme F-skoru (Popović 2015).
    n=6 ve beta=2.0 (recall ağırlıklı) standart değerleri kullanılır.
    """
    refs = [reference] if isinstance(reference, str) else list(reference)
    pred_clean = "".join(str(prediction or "").lower().split())
    if not pred_clean:
        return 0.0

    best_score = 0.0
    for r in refs:
        ref_clean = "".join(str(r or "").lower().split())
        if not ref_clean:
            continue

        precisions = []
        recalls = []
        for order in range(1, n + 1):
            pred_ngrams = Counter(pred_clean[i:i+order] for i in range(len(pred_clean) - order + 1))
            ref_ngrams = Counter(ref_clean[i:i+order] for i in range(len(ref_clean) - order + 1))
            total_pred = sum(pred_ngrams.values())
            total_ref = sum(ref_ngrams.values())
            overlap = sum((pred_ngrams & ref_ngrams).values())

            p = overlap / total_pred if total_pred > 0 else 0.0
            rec = overlap / total_ref if total_ref > 0 else 0.0
            precisions.append(p)
            recalls.append(rec)

        avg_p = sum(precisions) / n if n > 0 else 0.0
        avg_r = sum(recalls) / n if n > 0 else 0.0

        if avg_p + avg_r == 0:
            f = 0.0
        else:
            beta_sq = beta ** 2
            f = (1 + beta_sq) * (avg_p * avg_r) / (beta_sq * avg_p + avg_r)
        best_score = max(best_score, f)

    return round(best_score * 100.0, 2)


def evaluate_generation(
    predictions: Sequence[str],
    references: Union[Sequence[str], Sequence[Sequence[str]]]
) -> Dict[str, float]:
    """
    Birden fazla tahmin ve referans için toplu BLEU, ROUGE, ChrF ve F1 değerlendirmesi yapar.
    
    Args:
        predictions: Model tahminleri
        references: Doğru referans metinler
        
    Returns:
        Dict: Ortalama metrik sonuçları
    """
    if not predictions or not references or len(predictions) != len(references):
        return {
            "bleu-1": 0.0, "bleu-2": 0.0, "bleu-3": 0.0, "bleu-4": 0.0,
            "bleu": 0.0, "rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0,
            "chrf": 0.0, "exact_match": 0.0, "token_f1": 0.0
        }
        
    all_bleu = [compute_bleu(p, r) for p, r in zip(predictions, references)]
    all_rouge = [compute_rouge(p, r) for p, r in zip(predictions, references)]
    all_chrf = [compute_chrf(p, r) for p, r in zip(predictions, references)]
    all_em = [compute_exact_match(p, r) for p, r in zip(predictions, references)]
    all_f1 = [compute_token_f1(p, r)["f1"] for p, r in zip(predictions, references)]
    
    n = len(predictions)
    summary: Dict[str, float] = {}
    for key in ["bleu-1", "bleu-2", "bleu-3", "bleu-4", "bleu"]:
        summary[key] = round(sum(b[key] for b in all_bleu) / n, 2)
    for key in ["rouge-1", "rouge-2", "rouge-l"]:
        summary[key] = round(sum(r[key] for r in all_rouge) / n, 2)
    summary["chrf"] = round(sum(all_chrf) / n, 2)
    summary["exact_match"] = round(sum(all_em) / n, 2)
    summary["token_f1"] = round(sum(all_f1) / n, 2)
        
    return summary


# =========================
# Main (for testing)
# =========================

def main():
    """Test metrics computation."""
    print("\n" + "=" * 80)
    print("METRICS COMPUTATION TEST")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 4
    seq_len = 10
    vocab_size = 100
    
    print(f"\nTest Data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocab size: {vocab_size}")
    
    # Random logits and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("\n" + "-" * 80)
    print("Computing Metrics...")
    print("-" * 80)
    
    # Evaluate batch
    results = evaluate_batch(logits, targets, compute_all=True)
    
    print("\n" + str(results))
    
    # Save results
    results.save("evaluation_results_test.json")
    
    print("\n" + "=" * 80)
    print("✓ Metrics computation test completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
