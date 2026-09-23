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
from src.evaluation.metrics import (
    evaluate_generation,
    compute_bleu,
    compute_rouge,
    compute_chrf,
    compute_exact_match,
    compute_token_f1
)

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
            benchmark_name: Benchmark adı (perplexity, bleu, rouge, turkish_summarization, turkish_qa vb.)
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
        
        if max_samples < 1 or batch_size < 1:
            raise ValueError("Sample and batch counts must be positive")
        self.dataset_path = dataset_path
        benchmark_id = f"BENCH-{uuid.uuid4().hex[:8].upper()}"
        
        if benchmark_name == "perplexity":
            score, metrics = self._run_perplexity_benchmark(max_samples, batch_size)
        elif benchmark_name == "bleu":
            score, metrics = self._run_bleu_benchmark(max_samples, batch_size)
        elif benchmark_name == "rouge":
            score, metrics = self._run_rouge_benchmark(max_samples, batch_size)
        elif benchmark_name in ("gsm8k_cot", "gsm8k", "math_reasoning"):
            score, metrics = self._run_gsm8k_cot_benchmark(max_samples, batch_size)
        elif benchmark_name in ("turkish_knowledge", "turkish_facts"):
            score, metrics = self._run_turkish_knowledge_benchmark(max_samples, batch_size)
        elif benchmark_name in ("turkish_summarization", "summarization", "turkish_summary"):
            score, metrics = self._run_turkish_summarization_benchmark(max_samples, batch_size)
        elif benchmark_name in ("turkish_qa", "qa", "reading_comprehension", "turkish_reading"):
            score, metrics = self._run_turkish_qa_benchmark(max_samples, batch_size)
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark_name}")
        
        metrics["mode"] = "real"
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            model_name=self.model_name,
            benchmark_name=benchmark_name,
            score=score,
            metrics=metrics,
            timestamp=datetime.now(),
            samples_evaluated=int(metrics["samples"])
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
        import math
        import hashlib
        import json
        import torch.nn.functional as F
        if not getattr(self, "dataset_path", None):
            raise ValueError("Perplexity requires an explicit held-out dataset")
        path = Path(self.dataset_path)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            rows = pq.read_table(path).to_pylist()
            texts = [str(row["text"]) for row in rows if row.get("split", "test") == "test"]
        elif path.suffix == ".jsonl":
            texts = [json.loads(line)["text"] for line in path.read_text().splitlines() if line.strip()]
        else:
            texts = [line for line in path.read_text().splitlines() if line.strip()]
        pipeline = self._load_pipeline()
        nll, tokens, samples = 0.0, 0, 0
        context = pipeline.model.config.max_seq_len
        pipeline.model.eval()
        with torch.inference_mode():
            for text in texts[:max_samples]:
                ids = pipeline.tokenizer.encode(text)
                if len(ids) < 2:
                    continue
                samples += 1
                for offset in range(0, len(ids) - 1, context):
                    chunk = ids[offset:offset + context + 1]
                    x = torch.tensor([chunk[:-1]], device=pipeline.device)
                    y = torch.tensor(chunk[1:], device=pipeline.device)
                    logits, _ = pipeline.model(x)
                    nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y, reduction="sum").item()
                    tokens += y.numel()
        if not tokens:
            raise ValueError("Test dataset contains no evaluable tokens")
        loss = nll / tokens
        score = math.exp(loss)
        return score, {"perplexity": score, "loss": loss, "samples": samples,
                       "tokens_evaluated": tokens, "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def _load_pipeline(self):
        from src.inference.pipeline import InferencePipeline
        info = self.registry.load_model(self.model_name, verify_integrity=True)
        if not info.get("tokenizer_path"):
            raise ValueError("Model has no tokenizer; evaluation aborted")
        return InferencePipeline.from_pretrained(info["checkpoint_path"], info["tokenizer_path"], device=self.device)

    def _run_bleu_benchmark(
        self,
        max_samples: int,
        batch_size: int
    ) -> tuple[float, Dict[str, Any]]:
        """BLEU benchmark çalıştır."""
        model_info = self.registry.load_model(self.model_name)
        if not model_info:
            raise FileNotFoundError(f"Model not found: {self.model_name}")
            
        eval_pairs = [
            ("Türkiye'nin başkenti neresidir?", "Türkiye'nin başkenti Ankara'dır."),
            ("Yapay zeka nedir?", "Yapay zeka, makinelerin problem çözme ve öğrenme yetenekleridir."),
            ("Python hangi tür bir programlama dilidir?", "Python yüksek seviyeli, yorumlanan ve nesne yönelimli bir dildir."),
            ("Dünya'nın uydusu nedir?", "Dünya'nın doğal uydusu Ay'dır."),
        ]
        
        pipeline = self._load_pipeline()

        predictions = []
        references = []
        for prompt, ref in eval_pairs[:max_samples]:
            references.append(ref)
            if pipeline is not None:
                try:
                    pred = pipeline.generate(prompt=prompt, max_new_tokens=32, temperature=0.7)
                    predictions.append(str(pred).strip() if pred else "")
                except Exception:
                    raise ValueError("Model inference failed; benchmark aborted")
            else:
                raise ValueError("Model inference failed; benchmark aborted")
                
        bleu_metrics = evaluate_generation(predictions, references)
        score = bleu_metrics.get("bleu", 0.0)
        metrics = {
            "bleu": score,
            "bleu-1": bleu_metrics.get("bleu-1", 0.0),
            "bleu-2": bleu_metrics.get("bleu-2", 0.0),
            "bleu-3": bleu_metrics.get("bleu-3", 0.0),
            "bleu-4": bleu_metrics.get("bleu-4", 0.0),
            "samples": len(predictions),
            "batch_size": batch_size
        }
        return score, metrics

    def _run_rouge_benchmark(
        self,
        max_samples: int,
        batch_size: int
    ) -> tuple[float, Dict[str, Any]]:
        """ROUGE benchmark çalıştır."""
        model_info = self.registry.load_model(self.model_name)
        if not model_info:
            raise FileNotFoundError(f"Model not found: {self.model_name}")
            
        eval_pairs = [
            ("Türkiye'nin başkenti neresidir?", "Türkiye'nin başkenti Ankara'dır."),
            ("Yapay zeka nedir?", "Yapay zeka, makinelerin problem çözme ve öğrenme yetenekleridir."),
            ("Python hangi tür bir programlama dilidir?", "Python yüksek seviyeli, yorumlanan ve nesne yönelimli bir dildir."),
            ("Dünya'nın uydusu nedir?", "Dünya'nın doğal uydusu Ay'dır."),
        ]
        
        pipeline = self._load_pipeline()

        predictions = []
        references = []
        for prompt, ref in eval_pairs[:max_samples]:
            references.append(ref)
            if pipeline is not None:
                try:
                    pred = pipeline.generate(prompt=prompt, max_new_tokens=32, temperature=0.7)
                    predictions.append(str(pred).strip() if pred else "")
                except Exception:
                    raise ValueError("Model inference failed; benchmark aborted")
            else:
                raise ValueError("Model inference failed; benchmark aborted")
                
        rouge_metrics = evaluate_generation(predictions, references)
        score = rouge_metrics.get("rouge-l", 0.0)
        metrics = {
            "rouge-l": score,
            "rouge-1": rouge_metrics.get("rouge-1", 0.0),
            "rouge-2": rouge_metrics.get("rouge-2", 0.0),
            "samples": len(predictions),
            "batch_size": batch_size
        }
        return score, metrics

    def _run_gsm8k_cot_benchmark(
        self,
        max_samples: int,
        batch_size: int
    ) -> tuple[float, Dict[str, Any]]:
        """Çok Adımlı Matematiksel Akıl Yürütme (GSM8K CoT) benchmark çalıştır."""
        import re
        model_info = self.registry.load_model(self.model_name)
        if not model_info:
            raise FileNotFoundError(f"Model not found: {self.model_name}")

        benchmark = create_gsm8k_cot_benchmark()
        examples = benchmark.examples[:max_samples]

        pipeline = self._load_pipeline()

        def extract_answer(text: str) -> Optional[float]:
            # 1. Look for #### <num>
            cot_match = re.search(r"####\s*([0-9\.,\-]+)", text)
            if cot_match:
                raw = cot_match.group(1).replace(",", "").strip()
                try:
                    return float(raw)
                except ValueError:
                    pass
            # 2. Look for answer phrases
            ans_match = re.search(r"(?:cevap|sonuç|netice|yanıt|eşittir)\s*[:=]?\s*([0-9\.,\-]+)", text, re.IGNORECASE)
            if ans_match:
                raw = ans_match.group(1).replace(",", "").strip()
                try:
                    return float(raw)
                except ValueError:
                    pass
            # 3. Last number in text
            nums = re.findall(r"-?\d+(?:\.\d+)?", text)
            if nums:
                try:
                    return float(nums[-1])
                except ValueError:
                    pass
            return None

        correct_count = 0
        details = []

        for ex in examples:
            expected_num = float(ex.metadata.get("numeric_answer", 0))
            target_str = ex.target if isinstance(ex.target, str) else (ex.target[0] if ex.target else "")
            if pipeline is not None:
                try:
                    prompt = f"Soru: {ex.input}\nAdım adım çözüm:\n"
                    pred = pipeline.generate(prompt=prompt, max_new_tokens=96, temperature=0.3)
                    pred_str = str(pred).strip() if pred else ""
                except Exception:
                    raise ValueError("Model inference failed; benchmark aborted")
            else:
                raise ValueError("Model inference failed; benchmark aborted")

            pred_num = extract_answer(pred_str)
            is_correct = False
            if pred_num is not None:
                is_correct = abs(pred_num - expected_num) < 1e-4

            if is_correct:
                correct_count += 1

            steps_count = len([line for line in pred_str.split("\n") if line.strip()])

            details.append({
                "id": ex.id,
                "input": ex.input,
                "target_cot": target_str,
                "target_answer": expected_num,
                "model_output": pred_str,
                "predicted_answer": pred_num,
                "is_correct": is_correct,
                "steps_count": steps_count,
                "domain": ex.metadata.get("domain", "math"),
                "difficulty": ex.metadata.get("difficulty", "medium"),
            })

        total = len(examples)
        accuracy = (correct_count / total * 100.0) if total > 0 else 0.0
        avg_steps = sum(d["steps_count"] for d in details) / max(1, total)

        metrics = {
            "accuracy": round(accuracy, 2),
            "correct_count": correct_count,
            "total_questions": total,
            "avg_reasoning_steps": round(avg_steps, 2),
            "samples": total,
            "batch_size": batch_size,
            "details": details,
        }
        return round(accuracy, 2), metrics

    def _run_turkish_knowledge_benchmark(
        self,
        max_samples: int,
        batch_size: int
    ) -> tuple[float, Dict[str, Any]]:
        """Türkçe Olgusal Bilgi & Doğruluk benchmark çalıştır."""
        model_info = self.registry.load_model(self.model_name)
        if not model_info:
            raise FileNotFoundError(f"Model not found: {self.model_name}")

        benchmark = create_turkish_knowledge_benchmark()
        examples = benchmark.examples[:max_samples]

        pipeline = self._load_pipeline()

        correct_count = 0
        details = []
        predictions = []
        references = []

        for ex in examples:
            ref_str = ex.target if isinstance(ex.target, str) else ex.target[0]
            references.append(ref_str)
            if pipeline is not None:
                try:
                    pred = pipeline.generate(prompt=ex.input, max_new_tokens=48, temperature=0.5)
                    pred_str = str(pred).strip() if pred else ""
                except Exception:
                    raise ValueError("Model inference failed; benchmark aborted")
            else:
                raise ValueError("Model inference failed; benchmark aborted")

            predictions.append(pred_str)
            keywords = [k.lower() for k in ex.metadata.get("keywords", [])]
            pred_lower = pred_str.lower()
            hit_keywords = [k for k in keywords if k in pred_lower]
            is_correct = len(hit_keywords) > 0

            if is_correct:
                correct_count += 1

            details.append({
                "id": ex.id,
                "input": ex.input,
                "target": ref_str,
                "model_output": pred_str,
                "matched_keywords": hit_keywords,
                "is_correct": is_correct,
                "domain": ex.metadata.get("domain", "general")
            })

        total = len(examples)
        accuracy = (correct_count / total * 100.0) if total > 0 else 0.0
        gen_metrics = evaluate_generation(predictions, references)
        bleu = gen_metrics.get("bleu", 0.0)
        rouge_l = gen_metrics.get("rouge-l", 0.0)

        # Composite factual score: 70% keyword accuracy + 30% ROUGE-L (scaled 0-100)
        composite_score = round(0.7 * accuracy + 0.3 * (rouge_l * 100.0), 2)

        metrics = {
            "accuracy": round(accuracy, 2),
            "composite_score": composite_score,
            "correct_count": correct_count,
            "total_questions": total,
            "bleu": round(bleu, 2),
            "rouge_l": round(rouge_l, 4),
            "samples": total,
            "batch_size": batch_size,
            "details": details,
        }
        return composite_score, metrics

    def _run_turkish_summarization_benchmark(
        self,
        max_samples: int,
        batch_size: int
    ) -> tuple[float, Dict[str, Any]]:
        """Türkçe Metin Özetleme (Summarization) benchmark çalıştır."""
        model_info = self.registry.load_model(self.model_name)
        if not model_info:
            raise FileNotFoundError(f"Model not found: {self.model_name}")

        benchmark = create_turkish_summarization_benchmark()
        examples = benchmark.examples[:max_samples]

        pipeline = self._load_pipeline()

        details = []
        predictions = []
        references = []
        compression_ratios = []

        for ex in examples:
            ref_str = ex.target if isinstance(ex.target, str) else ex.target[0]
            references.append(ref_str)
            context = ex.metadata.get("context", ex.input)

            if pipeline is not None:
                try:
                    pred = pipeline.generate(prompt=ex.input, max_new_tokens=48, temperature=0.6)
                    pred_str = str(pred).strip() if pred else ""
                except Exception:
                    raise ValueError("Model inference failed; benchmark aborted")
            else:
                raise ValueError("Model inference failed; benchmark aborted")

            predictions.append(pred_str)

            rg = compute_rouge(pred_str, ref_str)
            chrf_val = compute_chrf(pred_str, ref_str)
            comp_ratio = round(len(pred_str) / max(1, len(context)), 3)
            compression_ratios.append(comp_ratio)

            details.append({
                "id": ex.id,
                "input": ex.input,
                "context": context,
                "model_output": pred_str,
                "target_summary": ref_str,
                "rouge_1": rg.get("rouge-1", 0.0),
                "rouge_2": rg.get("rouge-2", 0.0),
                "rouge_l": rg.get("rouge-l", 0.0),
                "chrf": chrf_val,
                "compression_ratio": comp_ratio,
                "domain": ex.metadata.get("domain", "genel"),
                "difficulty": ex.metadata.get("difficulty", "orta"),
            })

        gen_metrics = evaluate_generation(predictions, references)
        rouge_l = gen_metrics.get("rouge-l", 0.0)
        avg_comp = round(sum(compression_ratios) / max(1, len(compression_ratios)), 3)

        metrics = {
            "rouge_l": rouge_l,
            "rouge_1": gen_metrics.get("rouge-1", 0.0),
            "rouge_2": gen_metrics.get("rouge-2", 0.0),
            "chrf": gen_metrics.get("chrf", 0.0),
            "bleu": gen_metrics.get("bleu", 0.0),
            "avg_compression_ratio": avg_comp,
            "samples": len(examples),
            "batch_size": batch_size,
            "details": details,
        }
        return rouge_l, metrics

    def _run_turkish_qa_benchmark(
        self,
        max_samples: int,
        batch_size: int
    ) -> tuple[float, Dict[str, Any]]:
        """Türkçe Okuduğunu Anlama & Soru-Cevap (QA) benchmark çalıştır."""
        model_info = self.registry.load_model(self.model_name)
        if not model_info:
            raise FileNotFoundError(f"Model not found: {self.model_name}")

        benchmark = create_turkish_qa_benchmark()
        examples = benchmark.examples[:max_samples]

        pipeline = self._load_pipeline()

        correct_count = 0
        details = []
        predictions = []
        references = []
        f1_scores = []
        em_scores = []

        for ex in examples:
            ref_str = ex.target if isinstance(ex.target, str) else ex.target[0]
            references.append(ref_str)
            context = ex.metadata.get("context", "")
            question = ex.metadata.get("question", ex.input)

            if pipeline is not None:
                try:
                    pred = pipeline.generate(prompt=ex.input, max_new_tokens=32, temperature=0.3)
                    pred_str = str(pred).strip() if pred else ""
                except Exception:
                    raise ValueError("Model inference failed; benchmark aborted")
            else:
                raise ValueError("Model inference failed; benchmark aborted")

            predictions.append(pred_str)

            em_val = compute_exact_match(pred_str, ex.target)
            f1_dict = compute_token_f1(pred_str, ex.target)
            f1_val = f1_dict.get("f1", 0.0)
            chrf_val = compute_chrf(pred_str, ex.target)

            em_scores.append(em_val)
            f1_scores.append(f1_val)

            # Doğru kabul kriteri: Exact Match veya F1 >= 60.0
            is_correct = (em_val >= 99.0 or f1_val >= 60.0)
            if is_correct:
                correct_count += 1

            details.append({
                "id": ex.id,
                "input": ex.input,
                "question": question,
                "context": context,
                "model_output": pred_str,
                "target_answer": ref_str,
                "exact_match": em_val,
                "token_f1": f1_val,
                "precision": f1_dict.get("precision", 0.0),
                "recall": f1_dict.get("recall", 0.0),
                "chrf": chrf_val,
                "is_correct": is_correct,
                "domain": ex.metadata.get("domain", "genel"),
                "difficulty": ex.metadata.get("difficulty", "orta"),
            })

        total = len(examples)
        avg_f1 = round(sum(f1_scores) / max(1, total), 2)
        avg_em = round(sum(em_scores) / max(1, total), 2)
        accuracy = round(correct_count / max(1, total) * 100.0, 2)

        metrics = {
            "f1": avg_f1,
            "exact_match": avg_em,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_questions": total,
            "samples": total,
            "batch_size": batch_size,
            "details": details,
        }
        return avg_f1, metrics

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


def create_gsm8k_cot_benchmark() -> BenchmarkDataset:
    """
    Çok Adımlı Matematiksel Akıl Yürütme (GSM8K Chain-of-Thought) benchmark veri seti.
    
    Returns:
        BenchmarkDataset with CoT math reasoning examples
    """
    examples = [
        BenchmarkExample(
            id="gsm8k-001",
            input="Bir manavda tanesi 12 TL olan elmalardan 5 kilo, tanesi 18 TL olan portakallardan 3 kilo alan bir müşteri satıcıya 150 TL vermiştir. Müşteri kaç TL para üstü almalıdır?",
            target="Elmaların toplam maliyeti: 5 * 12 = 60 TL.\nPortakalların toplam maliyeti: 3 * 18 = 54 TL.\nToplam alışveriş tutarı: 60 + 54 = 114 TL.\nPara üstü: 150 - 114 = 36 TL.\n#### 36",
            metadata={"domain": "aritmetik", "difficulty": "easy", "numeric_answer": 36, "steps": 4}
        ),
        BenchmarkExample(
            id="gsm8k-002",
            input="Ahmet günde 45 sayfa kitap okuyor. 360 sayfalık bir kitabı bitirmek için kaç gün okuması gerekir?",
            target="Kitabın toplam sayfa sayısı 360'tır.\nAhmet her gün 45 sayfa okumaktadır.\nGereken toplam gün sayısı: 360 / 45 = 8 gündür.\n#### 8",
            metadata={"domain": "bölme", "difficulty": "easy", "numeric_answer": 8, "steps": 3}
        ),
        BenchmarkExample(
            id="gsm8k-003",
            input="Bir sınıfta 24 öğrenci vardır. Öğrencilerin %25'i matematik kulübüne, kalanın 1/3'ü ise bilim kulübüne üyedir. Bilim kulübüne üye kaç öğrenci vardır?",
            target="Matematik kulübündeki öğrenci sayısı: 24 * 0.25 = 6 öğrenci.\nGeriye kalan öğrenci sayısı: 24 - 6 = 18 öğrenci.\nBilim kulübündeki öğrenci sayısı: 18 * (1/3) = 6 öğrenci.\n#### 6",
            metadata={"domain": "yüzde_kesir", "difficulty": "medium", "numeric_answer": 6, "steps": 3}
        ),
        BenchmarkExample(
            id="gsm8k-004",
            input="Bir fabrikada 3 işçi 6 günde 72 parça üretiyor. Aynı tempoda çalışan 5 işçi 4 günde kaç parça üretir?",
            target="3 işçi 6 günde toplam 18 işçi-gününde 72 parça üretir.\n1 işçi 1 günde: 72 / 18 = 4 parça üretir.\n5 işçi 4 günde toplam 20 işçi-gün çalışır.\nÜretilen toplam parça: 20 * 4 = 80 parçadır.\n#### 80",
            metadata={"domain": "oran_oranti", "difficulty": "medium", "numeric_answer": 80, "steps": 4}
        ),
        BenchmarkExample(
            id="gsm8k-005",
            input="Bir araç 240 kilometrelik yolu saatte 60 km hızla gidip, saatte 80 km hızla geri dönmüştür. Bu gidiş-dönüş yolculuğu toplam kaç saat sürmüştür?",
            target="Gidiş süresi: 240 / 60 = 4 saat.\nDönüş süresi: 240 / 80 = 3 saat.\nToplam yolculuk süresi: 4 + 3 = 7 saattir.\n#### 7",
            metadata={"domain": "hareket_problemi", "difficulty": "medium", "numeric_answer": 7, "steps": 3}
        ),
        BenchmarkExample(
            id="gsm8k-006",
            input="Bir su deposundaki 120 litre suyun önce 1/4'ü, sonra kalan suyun yarısı kullanılmıştır. Depoda son durumda kaç litre su kalmıştır?",
            target="İlk kullanılan su miktarı: 120 * (1/4) = 30 litredir.\nKalan su miktarı: 120 - 30 = 90 litredir.\nİkinci kullanılan su miktarı: 90 / 2 = 45 litredir.\nDepoda son kalan su: 90 - 45 = 45 litredir.\n#### 45",
            metadata={"domain": "kesir", "difficulty": "medium", "numeric_answer": 45, "steps": 4}
        ),
        BenchmarkExample(
            id="gsm8k-007",
            input="Bir sinemada bilet fiyatı tam 80 TL, öğrenci 50 TL'dir. 4 tam ve 6 öğrenci bileti alan bir arkadaş grubu toplam kaç TL öder?",
            target="Tam biletlerin tutarı: 4 * 80 = 320 TL.\nÖğrenci biletlerinin tutarı: 6 * 50 = 300 TL.\nToplam ödenecek tutar: 320 + 300 = 620 TL'dir.\n#### 620",
            metadata={"domain": "aritmetik", "difficulty": "easy", "numeric_answer": 620, "steps": 3}
        ),
        BenchmarkExample(
            id="gsm8k-008",
            input="Kenar uzunluğu 15 metre olan kare şeklindeki bir bahçenin etrafına 3 sıra tel çekilecektir. Toplam kaç metre tele ihtiyaç vardır?",
            target="Karenin çevresi: 4 * 15 = 60 metredir.\n3 sıra tel çekileceği için toplam tel miktarı: 3 * 60 = 180 metredir.\n#### 180",
            metadata={"domain": "geometri", "difficulty": "easy", "numeric_answer": 180, "steps": 2}
        ),
    ]
    return BenchmarkDataset(
        name="gsm8k_cot",
        description="Çok Adımlı Matematiksel Akıl Yürütme (Chain-of-Thought) Benchmark",
        examples=examples,
        metadata={"category": "reasoning", "num_examples": len(examples), "format": "CoT with #### <number>"}
    )


def create_turkish_knowledge_benchmark() -> BenchmarkDataset:
    """
    Türkçe Olgusal Bilgi & Doğruluk Testi (Factual Knowledge Benchmark).
    
    Returns:
        BenchmarkDataset with Turkish factual Q&A examples
    """
    examples = [
        BenchmarkExample(
            id="tr-know-001",
            input="Türkiye Cumhuriyeti'nin başkenti hangi şehirdir?",
            target="Türkiye Cumhuriyeti'nin başkenti Ankara'dır.",
            metadata={"domain": "coğrafya", "keywords": ["ankara"], "category": "knowledge"}
        ),
        BenchmarkExample(
            id="tr-know-002",
            input="Çanakkale Deniz Zaferi hangi yıl kazanılmıştır?",
            target="Çanakkale Deniz Zaferi 18 Mart 1915 tarihinde kazanılmıştır.",
            metadata={"domain": "tarih", "keywords": ["1915"], "category": "knowledge"}
        ),
        BenchmarkExample(
            id="tr-know-003",
            input="Türkiye'nin en yüksek dağı hangisidir ve rakımı kaç metredir?",
            target="Türkiye'nin en yüksek dağı 5137 metre rakımıyla Ağrı Dağı'dır.",
            metadata={"domain": "coğrafya", "keywords": ["ağrı", "5137"], "category": "knowledge"}
        ),
        BenchmarkExample(
            id="tr-know-004",
            input="İstiklal Marşı'mızın şairi kimdir?",
            target="İstiklal Marşı'mızın şairi Mehmet Akif Ersoy'dur.",
            metadata={"domain": "edebiyat", "keywords": ["mehmet akif ersoy", "mehmet akif"], "category": "knowledge"}
        ),
        BenchmarkExample(
            id="tr-know-005",
            input="Mimar Sinan'ın 'ustalık eserim' olarak adlandırdığı ve Edirne'de bulunan tarihi cami hangisidir?",
            target="Mimar Sinan'ın ustalık eseri Edirne'de bulunan Selimiye Camii'dir.",
            metadata={"domain": "mimari", "keywords": ["selimiye"], "category": "knowledge"}
        ),
        BenchmarkExample(
            id="tr-know-006",
            input="Türkiye'nin yüzölçümü bakımından en büyük gölü hangisidir?",
            target="Türkiye'nin en büyük gölü Doğu Anadolu Bölgesi'ndeki Van Gölü'dür.",
            metadata={"domain": "coğrafya", "keywords": ["van gölü", "van"], "category": "knowledge"}
        ),
        BenchmarkExample(
            id="tr-know-007",
            input="Hücre çekirdeğinde genetik bilgiyi taşıyan çift sarmal yapılı nükleik asit nedir?",
            target="Genetik bilgiyi taşıyan molekül Deoksiribonükleik asit (DNA)'tir.",
            metadata={"domain": "biyoloji", "keywords": ["dna", "deoksiribonükleik"], "category": "knowledge"}
        ),
        BenchmarkExample(
            id="tr-know-008",
            input="Türk edebiyatının ilk yazılı metinleri sayılan Orhun Yazıtları (Göktürk Kitabeleri) hangi yüzyılda dikilmiştir?",
            target="Orhun Yazıtları 8. yüzyılda (MS 732-735 yılları civarında) dikilmiştir.",
            metadata={"domain": "tarih_dil", "keywords": ["8. yüzyıl", "sekizinci yüzyıl", "8"], "category": "knowledge"}
        ),
    ]
    return BenchmarkDataset(
        name="turkish_knowledge",
        description="Türkçe Olgusal Bilgi ve Doğruluk Değerlendirme Benchmark",
        examples=examples,
        metadata={"category": "knowledge", "num_examples": len(examples)}
    )


def create_turkish_summarization_benchmark() -> BenchmarkDataset:
    """
    Türkçe Metin Özetleme (Summarization) Benchmark veri seti oluşturur.
    ROUGE-1, ROUGE-2, ROUGE-L ve ChrF metrikleri ile değerlendirilir.
    """
    examples = [
        BenchmarkExample(
            id="tr-sum-001",
            input="Metin:\nJames Webb Uzay Teleskobu, evrenin büyük patlamadan sonraki ilk yüz milyon yılına ait daha önce hiç gözlemlenmemiş en erken galaksileri keşfetti. Bu keşif, ilk yıldız kümelerinin astronomların tahmin ettiğinden çok daha hızlı ve parlak şekilde oluştuklarını kanıtladı.\n\nYukarıdaki metni Türkçe olarak tek ve öz bir cümle ile özetleyiniz:\nÖzet:",
            target="James Webb Teleskobu, erken evren galaksilerinin beklenenden çok daha hızlı ve parlak oluştuğunu ortaya koydu.",
            metadata={
                "context": "James Webb Uzay Teleskobu, evrenin büyük patlamadan sonraki ilk yüz milyon yılına ait daha önce hiç gözlemlenmemiş en erken galaksileri keşfetti. Bu keşif, ilk yıldız kümelerinin astronomların tahmin ettiğinden çok daha hızlı ve parlak şekilde oluştuklarını kanıtladı.",
                "domain": "astronomi",
                "category": "summarization",
                "difficulty": "orta",
                "keywords": ["james webb", "galaksi", "erken evren"]
            }
        ),
        BenchmarkExample(
            id="tr-sum-002",
            input="Metin:\nYapay zeka destekli tıbbi görüntüleme sistemleri, akciğer röntgenleri ve manyetik rezonans (MR) taramalarında insan gözünden kaçabilecek erken evre lezyonları saniyeler içerisinde tespit edebilmektedir. Bu teknoloji, uzman hekimlerin tanı koyma doğruluğunu artırırken tedaviye başlama sürecini belirgin şekilde kısaltmaktadır.\n\nYukarıdaki metni Türkçe olarak tek ve öz bir cümle ile özetleyiniz:\nÖzet:",
            target="Yapay zekalı tıbbi görüntüleme, erken lezyon tespitini hızlandırarak doktorların teşhis doğruluğunu yükseltir.",
            metadata={
                "context": "Yapay zeka destekli tıbbi görüntüleme sistemleri, akciğer röntgenleri ve manyetik rezonans (MR) taramalarında insan gözünden kaçabilecek erken evre lezyonları saniyeler içerisinde tespit edebilmektedir. Bu teknoloji, uzman hekimlerin tanı koyma doğruluğunu artırırken tedaviye başlama sürecini belirgin şekilde kısaltmaktadır.",
                "domain": "sağlık",
                "category": "summarization",
                "difficulty": "kolay",
                "keywords": ["yapay zeka", "tıbbi görüntüleme", "erken teşhis"]
            }
        ),
        BenchmarkExample(
            id="tr-sum-003",
            input="Metin:\nYenilenebilir enerji teknolojilerinde son dönemde geliştirilen perovskit tabanlı yeni nesil güneş pilleri, geleneksel silikon panellerin %22 olan teorik enerji dönüşüm verimliliğini %30 seviyelerine taşımıştır. Üstelik bu yeni malzemelerin üretim maliyeti daha düşüktür.\n\nYukarıdaki metni Türkçe olarak tek ve öz bir cümle ile özetleyiniz:\nÖzet:",
            target="Perovskit güneş pilleri, geleneksel panellere kıyasla daha düşük maliyetle yüzde 30 daha yüksek verimlilik sunar.",
            metadata={
                "context": "Yenilenebilir enerji teknolojilerinde son dönemde geliştirilen perovskit tabanlı yeni nesil güneş pilleri, geleneksel silikon panellerin %22 olan teorik enerji dönüşüm verimliliğini %30 seviyelerine taşımıştır. Üstelik bu yeni malzemelerin üretim maliyeti daha düşüktür.",
                "domain": "enerji",
                "category": "summarization",
                "difficulty": "orta",
                "keywords": ["perovskit", "güneş pilleri", "verimlilik"]
            }
        ),
        BenchmarkExample(
            id="tr-sum-004",
            input="Metin:\nŞanlıurfa yakınlarındaki Göbeklitepe ve Karahantepe kazıları, insanlığın yerleşik hayata geçmeden ve tarıma başlamadan önce de anıtsal tapınaklar ve karmaşık sosyal yapılar kurabildiğini göstererek arkeoloji tarihindeki geleneksel teorileri tamamen değiştirmiştir.\n\nYukarıdaki metni Türkçe olarak tek ve öz bir cümle ile özetleyiniz:\nÖzet:",
            target="Göbeklitepe bulguları, insanların tarımdan önce anıtsal tapınaklar ve sosyal yapılar inşa ettiğini kanıtlamıştır.",
            metadata={
                "context": "Şanlıurfa yakınlarındaki Göbeklitepe ve Karahantepe kazıları, insanlığın yerleşik hayata geçmeden ve tarıma başlamadan önce de anıtsal tapınaklar ve karmaşık sosyal yapılar kurabildiğini göstererek arkeoloji tarihindeki geleneksel teorileri tamamen değiştirmiştir.",
                "domain": "arkeoloji",
                "category": "summarization",
                "difficulty": "zor",
                "keywords": ["göbeklitepe", "tarım", "anıtsal tapınak"]
            }
        ),
        BenchmarkExample(
            id="tr-sum-005",
            input="Metin:\nOtonom sürüş sistemleri; lidar, radar ve yüksek çözünürlüklü kameralardan toplanan çoklu duyusal verileri derin sinir ağlarıyla gerçek zamanlı birleştirerek yayaları, şeritleri ve trafik işaretlerini milisaniyeler düzeyinde kusursuz algılar.\n\nYukarıdaki metni Türkçe olarak tek ve öz bir cümle ile özetleyiniz:\nÖzet:",
            target="Otonom araçlar, sensör ve kamera verilerini sinir ağlarıyla birleştirerek çevrelerini anlık ve hatasız algılar.",
            metadata={
                "context": "Otonom sürüş sistemleri; lidar, radar ve yüksek çözünürlüklü kameralardan toplanan çoklu duyusal verileri derin sinir ağlarıyla gerçek zamanlı birleştirerek yayaları, şeritleri ve trafik işaretlerini milisaniyeler düzeyinde kusursuz algılar.",
                "domain": "otomasyon",
                "category": "summarization",
                "difficulty": "orta",
                "keywords": ["otonom sürüş", "lidar", "sensör füzyonu"]
            }
        ),
        BenchmarkExample(
            id="tr-sum-006",
            input="Metin:\nKuantum bilgisayarlar, klasik bitler yerine süperpozisyon ve dolanıklık ilkelerine dayanan kübitleri kullanarak günümüzdeki en güçlü süper bilgisayarların binlerce yılda tamamlayabileceği karmaşık kimyasal simülasyonları ve optimizasyon hesaplarını dakikalar içinde çözebilir.\n\nYukarıdaki metni Türkçe olarak tek ve öz bir cümle ile özetleyiniz:\nÖzet:",
            target="Kuantum bilgisayarlar kübitler yardımıyla, süper bilgisayarların yıllar alacak hesaplamalarını dakikalar içinde çözer.",
            metadata={
                "context": "Kuantum bilgisayarlar, klasik bitler yerine süperpozisyon ve dolanıklık ilkelerine dayanan kübitleri kullanarak günümüzdeki en güçlü süper bilgisayarların binlerce yılda tamamlayabileceği karmaşık kimyasal simülasyonları ve optimizasyon hesaplarını dakikalar içinde çözebilir.",
                "domain": "kuantum",
                "category": "summarization",
                "difficulty": "zor",
                "keywords": ["kuantum", "kübit", "hesaplama"]
            }
        )
    ]
    return BenchmarkDataset(
        name="turkish_summarization",
        description="Türkçe Metin Özetleme ve Bilgi Sıkıştırma Benchmark",
        examples=examples,
        metadata={"category": "summarization", "num_examples": len(examples)}
    )


def create_turkish_qa_benchmark() -> BenchmarkDataset:
    """
    Türkçe Okuduğunu Anlama ve Soru-Cevap (Reading Comprehension / QA) Benchmark veri seti oluşturur.
    Exact Match (EM) ve Token F1 metrikleri ile değerlendirilir.
    """
    examples = [
        BenchmarkExample(
            id="tr-qa-001",
            input="Bağlam: Osmanlı padişahı II. Mehmed, 53 gün süren kuşatmanın ardından 29 Mayıs 1453 Salı günü İstanbul'u fethederek Doğu Roma (Bizans) İmparatorluğu'na son vermiştir.\n\nSoru: İstanbul hangi tarihte fethedilmiştir?\n\nCevap:",
            target=["29 Mayıs 1453", "1453", "29 Mayıs 1453 Salı"],
            metadata={
                "context": "Osmanlı padişahı II. Mehmed, 53 gün süren kuşatmanın ardından 29 Mayıs 1453 Salı günü İstanbul'u fethederek Doğu Roma (Bizans) İmparatorluğu'na son vermiştir.",
                "question": "İstanbul hangi tarihte fethedilmiştir?",
                "domain": "tarih",
                "category": "qa",
                "difficulty": "kolay",
                "keywords": ["29 mayıs 1453", "1453"]
            }
        ),
        BenchmarkExample(
            id="tr-qa-002",
            input="Bağlam: Van Gölü, Doğu Anadolu Bölgesi'nde yer alan kapalı havzalı bir göldür. Suları bol miktarda soda ve tuz içerdiğinden gölde yalnızca sodalı suya uyum sağlamış olan inci kefali balığı yaşayabilmektedir.\n\nSoru: Van Gölü'nün suyu hangi kimyasal özelliğe sahiptir?\n\nCevap:",
            target=["Sodalı ve tuzludur", "sodalı", "tuzlu ve sodalı"],
            metadata={
                "context": "Van Gölü, Doğu Anadolu Bölgesi'nde yer alan kapalı havzalı bir göldür. Suları bol miktarda soda ve tuz içerdiğinden gölde yalnızca sodalı suya uyum sağlamış olan inci kefali balığı yaşayabilmektedir.",
                "question": "Van Gölü'nün suyu hangi kimyasal özelliğe sahiptir?",
                "domain": "coğrafya",
                "category": "qa",
                "difficulty": "kolay",
                "keywords": ["soda", "tuz", "sodalı"]
            }
        ),
        BenchmarkExample(
            id="tr-qa-003",
            input="Bağlam: Modern fizikte ışık hızı temel evrensel bir sabittir ve vakum (boşluk) ortamında saniyede yaklaşık 299.792 kilometre (yaklaşık 300.000 km/s) hızla hareket eder.\n\nSoru: Işığın boşluktaki yayılma hızı saniyede yaklaşık kaç kilometredir?\n\nCevap:",
            target=["300.000 km/s", "299.792 km", "yaklaşık 300.000 kilometre"],
            metadata={
                "context": "Modern fizikte ışık hızı temel evrensel bir sabittir ve vakum (boşluk) ortamında saniyede yaklaşık 299.792 kilometre (yaklaşık 300.000 km/s) hızla hareket eder.",
                "question": "Işığın boşluktaki yayılma hızı saniyede yaklaşık kaç kilometredir?",
                "domain": "fizik",
                "category": "qa",
                "difficulty": "kolay",
                "keywords": ["300.000", "299.792"]
            }
        ),
        BenchmarkExample(
            id="tr-qa-004",
            input="Bağlam: DNA molekülü dört temel azotlu organik bazdan meydana gelir: Adenin (A), Timin (T), Guanin (G) ve Sitozin (C). Watson-Crick eşleşmesine göre adenin daima timin ile ikili hidrojen bağı yapar.\n\nSoru: DNA molekülünde adeninin karşısına hangi baz gelir?\n\nCevap:",
            target=["Timin", "Timin bazı", "T"],
            metadata={
                "context": "DNA molekülü dört temel azotlu organik bazdan meydana gelir: Adenin (A), Timin (T), Guanin (G) ve Sitozin (C). Watson-Crick eşleşmesine göre adenin daima timin ile ikili hidrojen bağı yapar.",
                "question": "DNA molekülünde adeninin karşısına hangi baz gelir?",
                "domain": "biyoloji",
                "category": "qa",
                "difficulty": "kolay",
                "keywords": ["timin"]
            }
        ),
        BenchmarkExample(
            id="tr-qa-005",
            input="Bağlam: İngiliz matematikçi Alan Turing tarafından 1950 yılında önerilen Turing Testi, bir makinenin sergilediği zekanın bir insan tarafından ayırt edilemeyecek seviyede olup olmadığını ölçmeyi amaçlar.\n\nSoru: Turing Testi hangi amaç doğrultusunda geliştirilmiştir?\n\nCevap:",
            target=["Makinelerin insan benzeri zeka ve düşünme yeteneğini ölçmek", "yapay zeka başarısını sınamak"],
            metadata={
                "context": "İngiliz matematikçi Alan Turing tarafından 1950 yılında önerilen Turing Testi, bir makinenin sergilediği zekanın bir insan tarafından ayırt edilemeyecek seviyede olup olmadığını ölçmeyi amaçlar.",
                "question": "Turing Testi hangi amaç doğrultusunda geliştirilmiştir?",
                "domain": "bilgisayar",
                "category": "qa",
                "difficulty": "orta",
                "keywords": ["insan", "zeka", "makine", "ölçmek"]
            }
        ),
        BenchmarkExample(
            id="tr-qa-006",
            input="Bağlam: Gazi Mustafa Kemal Atatürk'ün kaleme aldığı ve 1927 yılında CHP kurultayında bizzat okuduğu Nutuk adlı abidevi eser, 1919'da Samsun'a çıkıştan başlayarak 1927 yılına kadarki Milli Mücadele ve Cumhuriyetin kuruluş sürecini belgelerle anlatır.\n\nSoru: Nutuk hangi zaman aralığındaki olayları kapsamaktadır?\n\nCevap:",
            target=["1919 ile 1927 yılları arası", "1919-1927"],
            metadata={
                "context": "Gazi Mustafa Kemal Atatürk'ün kaleme aldığı ve 1927 yılında CHP kurultayında bizzat okuduğu Nutuk adlı abidevi eser, 1919'da Samsun'a çıkıştan başlayarak 1927 yılına kadarki Milli Mücadele ve Cumhuriyetin kuruluş sürecini belgelerle anlatır.",
                "question": "Nutuk hangi zaman aralığındaki olayları kapsamaktadır?",
                "domain": "tarih",
                "category": "qa",
                "difficulty": "orta",
                "keywords": ["1919", "1927"]
            }
        ),
        BenchmarkExample(
            id="tr-qa-007",
            input="Bağlam: Bitkiler ve algler, güneş ışığı enerjisini kullanarak karbondioksit ve sudan glikoz (besin) sentezlerken fotosentez reaksiyonunun bir yan ürünü olarak atmosfere serbest oksijen gazı salarlar.\n\nSoru: Fotosentez sonucunda atmosfere hangi gaz salınır?\n\nCevap:",
            target=["Oksijen", "Oksijen gazı", "O2"],
            metadata={
                "context": "Bitkiler ve algler, güneş ışığı enerjisini kullanarak karbondioksit ve sudan glikoz (besin) sentezlerken fotosentez reaksiyonunun bir yan ürünü olarak atmosfere serbest oksijen gazı salarlar.",
                "question": "Fotosentez sonucunda atmosfere hangi gaz salınır?",
                "domain": "biyoloji_kimya",
                "category": "qa",
                "difficulty": "kolay",
                "keywords": ["oksijen"]
            }
        ),
        BenchmarkExample(
            id="tr-qa-008",
            input="Bağlam: Mars gezegenine 'Kızıl Gezegen' lakabının verilmesinin başlıca sebebi, gezegenin kayalık kabuğunda ve yüzey tozunda bolca bulunan demiroksit (pas) minerallerinin gökyüzünden ve uzaydan bakıldığında belirgin bir kızıl renk yansıtmasıdır.\n\nSoru: Mars'ın kırmızı görünmesinin sebebi nedir?\n\nCevap:",
            target=["Yüzeyindeki yoğun demiroksit (pas) bileşikleri", "demiroksit mineralleri", "demir oksit"],
            metadata={
                "context": "Mars gezegenine 'Kızıl Gezegen' lakabının verilmesinin başlıca sebebi, gezegenin kayalık kabuğunda ve yüzey tozunda bolca bulunan demiroksit (pas) minerallerinin gökyüzünden ve uzaydan bakıldığında belirgin bir kızıl renk yansıtmasıdır.",
                "question": "Mars'ın kırmızı görünmesinin sebebi nedir?",
                "domain": "astronomi",
                "category": "qa",
                "difficulty": "orta",
                "keywords": ["demiroksit", "pas", "demir oksit"]
            }
        )
    ]
    return BenchmarkDataset(
        name="turkish_qa",
        description="Türkçe Okuduğunu Anlama ve Soru-Cevap (Reading Comprehension & QA) Benchmark",
        examples=examples,
        metadata={"category": "qa", "num_examples": len(examples)}
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
        results.add_metric(key, float(value), "", higher_is_better=True)
    
    for key, value in avg_rouge.items():
        results.add_metric(key, float(value), "", higher_is_better=True)
    
    results.add_metric("chrf", float(avg_chrf), "", higher_is_better=True)
    
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
