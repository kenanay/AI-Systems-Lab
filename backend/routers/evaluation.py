"""
backend/routers/evaluation.py

Model Evaluation ve Benchmark API Endpoints

Bu modül trained modellerin evaluation ve benchmark işlemleri için
REST API sağlar:
- Benchmark çalıştırma (perplexity, BLEU, ROUGE, vb.)
- Evaluation sonuçlarını listeleme
- Model karşılaştırma
- Metrics görselleştirme

Version: 1.0.0
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime
from pathlib import Path
import re
import math

from backend.database import get_db
from backend.models import BenchmarkRecord
from src.evaluation.benchmarks import BenchmarkRunner, BenchmarkResult
from src.evaluation.metrics import (
    compute_perplexity,
    compute_bleu,
    compute_rouge,
    compute_chrf,
    compute_exact_match,
    compute_token_f1,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/evaluation",
    tags=["evaluation"]
)


# Request/Response Models

class TextMetricInspectRequest(BaseModel):
    """Metin düzeyi BLEU ve ROUGE değerlendirme request."""
    candidate: str = Field(..., description="Model çıktısı / Aday metin")
    reference: str = Field(..., description="Referans hedef metin")
    max_n: int = Field(4, ge=1, le=4, description="Maksimum n-gram boyutu (1-4)")


class TextMetricInspectResponse(BaseModel):
    """Metin düzeyi değerlendirme ve n-gram eşleşme response."""
    candidate_tokens: List[str]
    reference_tokens: List[str]
    matched_unigrams: List[str]
    matched_bigrams: List[str]
    matched_trigrams: List[str]
    bleu: Dict[str, float]
    brevity_penalty: float
    candidate_len: int
    reference_len: int
    rouge: Dict[str, float]
    chrf: float = 0.0
    exact_match: float = 0.0
    token_f1: Dict[str, float] = Field(default_factory=dict)


class BenchmarkRunRequest(BaseModel):
    """Benchmark çalıştırma request."""
    model_name: str = Field(..., description="Model adı (registry'den)")
    benchmark_name: str = Field(..., description="Benchmark adı (perplexity, bleu, rouge)")
    dataset_path: Optional[str] = Field(None, description="Test dataset path (optional)")
    max_samples: int = Field(100, ge=1, le=10000, description="Maximum sample sayısı")
    batch_size: int = Field(8, ge=1, le=128, description="Batch size")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "turkish-gpt",
                "benchmark_name": "perplexity",
                "max_samples": 100,
                "batch_size": 8
            }
        }


class BenchmarkResultResponse(BaseModel):
    """Benchmark sonucu response."""
    benchmark_id: str
    model_name: str
    benchmark_name: str
    score: float
    metrics: Dict[str, Any]
    timestamp: datetime
    samples_evaluated: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "benchmark_id": "BENCH-ABC123",
                "model_name": "turkish-gpt",
                "benchmark_name": "perplexity",
                "score": 15.3,
                "metrics": {
                    "perplexity": 15.3,
                    "loss": 2.73,
                    "tokens_evaluated": 50000
                },
                "timestamp": "2026-09-19T12:00:00",
                "samples_evaluated": 100
            }
        }


class ModelComparisonRequest(BaseModel):
    """Model karşılaştırma request."""
    dataset_path: Optional[str] = None
    model_names: List[str] = Field(..., min_length=1, description="Karşılaştırılacak model isimleri")
    benchmark_name: str = Field(..., description="Benchmark adı")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_names": ["turkish-gpt-v1", "turkish-gpt-v2"],
                "benchmark_name": "perplexity"
            }
        }


class ModelComparisonResponse(BaseModel):
    """Model karşılaştırma response."""
    benchmark_name: str
    comparisons: List[Dict[str, Any]]
    winner: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "benchmark_name": "perplexity",
                "comparisons": [
                    {"model_name": "turkish-gpt-v1", "score": 18.5},
                    {"model_name": "turkish-gpt-v2", "score": 15.3}
                ],
                "winner": "turkish-gpt-v2"
            }
        }


class BenchmarkSampleQuestion(BaseModel):
    """Benchmark soru havuzu örneği."""
    id: str
    input: str
    target: str
    domain: str
    category: str
    difficulty: Optional[str] = None
    numeric_answer: Optional[float] = None
    steps: Optional[int] = None
    keywords: Optional[List[str]] = None
    context: Optional[str] = None
    question: Optional[str] = None


class RadarDimensionScore(BaseModel):
    """Radar grafiği tekil boyut skoru."""
    dimension_key: str
    dimension_name: str
    score: float  # Normalized 0-100
    raw_metric: str
    raw_score: float


class RadarModelScore(BaseModel):
    """Bir modelin tüm radar boyutlarındaki skorları."""
    model_name: str
    overall_average: float
    dimensions: List[RadarDimensionScore]


class RadarComparisonRequest(BaseModel):
    dataset_path: Optional[str] = None
    """Çoklu model radar kıyaslama request."""
    model_names: List[str] = Field(..., min_length=1, max_length=6, description="Karşılaştırılacak modeller (1-6 model)")


class RadarComparisonResponse(BaseModel):
    """Çoklu model radar kıyaslama response."""
    models: List[RadarModelScore]
    dimensions: List[str]
    winner_by_dimension: Dict[str, str]
    overall_winner: str


# Endpoints

@router.post("/inspect-text", response_model=TextMetricInspectResponse)
async def inspect_text_metrics(request: TextMetricInspectRequest) -> TextMetricInspectResponse:
    """
    Model tahmini ve referans metin arasındaki BLEU-1..4, ROUGE-1/2/L,
    Brevity Penalty ve örtüşen n-gramları hesaplar.
    """
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    cand_tokens = tokenize(request.candidate)
    ref_tokens = tokenize(request.reference)

    cand_len = len(cand_tokens)
    ref_len = len(ref_tokens)

    if cand_len == 0:
        bp = 0.0
    elif cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_len / max(1, cand_len))

    matched_unigrams = sorted(list(set(cand_tokens) & set(ref_tokens)))

    cand_bigrams = {" ".join(cand_tokens[i:i+2]) for i in range(len(cand_tokens) - 1)}
    ref_bigrams = {" ".join(ref_tokens[i:i+2]) for i in range(len(ref_tokens) - 1)}
    matched_bigrams = sorted(list(cand_bigrams & ref_bigrams))

    cand_trigrams = {" ".join(cand_tokens[i:i+3]) for i in range(len(cand_tokens) - 2)}
    ref_trigrams = {" ".join(ref_tokens[i:i+3]) for i in range(len(ref_tokens) - 2)}
    matched_trigrams = sorted(list(cand_trigrams & ref_trigrams))

    bleu_res = compute_bleu(request.candidate, request.reference, max_n=request.max_n)
    rouge_res = compute_rouge(request.candidate, request.reference)
    chrf_res = compute_chrf(request.candidate, request.reference)
    em_res = compute_exact_match(request.candidate, request.reference)
    f1_res = compute_token_f1(request.candidate, request.reference)

    return TextMetricInspectResponse(
        candidate_tokens=cand_tokens,
        reference_tokens=ref_tokens,
        matched_unigrams=matched_unigrams,
        matched_bigrams=matched_bigrams,
        matched_trigrams=matched_trigrams,
        bleu=bleu_res,
        brevity_penalty=round(bp, 4),
        candidate_len=cand_len,
        reference_len=ref_len,
        rouge=rouge_res,
        chrf=chrf_res,
        exact_match=em_res,
        token_f1=f1_res,
    )


@router.post("/run", response_model=BenchmarkResultResponse)
async def run_benchmark(
    request: BenchmarkRunRequest,
    db: Session = Depends(get_db)
) -> BenchmarkResultResponse:
    """
    Benchmark çalıştır ve sonucu veritabanına kaydet.
    """
    logger.info(f"Running benchmark: {request.benchmark_name} on {request.model_name}")
    
    try:
        # Initialize benchmark runner
        runner = BenchmarkRunner(
            model_name=request.model_name,
            device="cpu"  # TODO: CUDA support
        )
        
        # Run benchmark
        result = runner.run_benchmark(
            benchmark_name=request.benchmark_name,
            dataset_path=request.dataset_path,
            max_samples=request.max_samples,
            batch_size=request.batch_size
        )
        
        logger.info(
            f"Benchmark completed: {request.benchmark_name} on {request.model_name} "
            f"= {result.score:.2f}"
        )
        
        # Persist result to database
        try:
            record = BenchmarkRecord(
                benchmark_id=result.benchmark_id,
                model_name=result.model_name,
                benchmark_name=result.benchmark_name,
                score=float(result.score),
                metrics=dict(result.metrics or {}),
                samples_evaluated=int(result.samples_evaluated or 0),
                dataset_path=request.dataset_path,
                created_at=result.timestamp or datetime.utcnow(),
            )
            db.add(record)
            db.commit()
            db.refresh(record)
        except Exception as db_err:
            logger.warning(f"Could not persist benchmark to DB: {db_err}")
            db.rollback()

        # Convert to response
        response = BenchmarkResultResponse(
            benchmark_id=result.benchmark_id,
            model_name=result.model_name,
            benchmark_name=result.benchmark_name,
            score=result.score,
            metrics=result.metrics,
            timestamp=result.timestamp,
            samples_evaluated=result.samples_evaluated
        )
        
        return response
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid benchmark: {str(e)}")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.get("/results", response_model=List[BenchmarkResultResponse])
async def list_benchmark_results(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    benchmark_name: Optional[str] = Query(None, description="Filter by benchmark name"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    db: Session = Depends(get_db)
) -> List[BenchmarkResultResponse]:
    """
    Veritabanından geçmiş benchmark sonuçlarını listele.
    """
    logger.info(f"Listing benchmark results: model={model_name}, benchmark={benchmark_name}")
    
    query = db.query(BenchmarkRecord)
    if model_name:
        query = query.filter(BenchmarkRecord.model_name == model_name)
    if benchmark_name:
        query = query.filter(BenchmarkRecord.benchmark_name == benchmark_name)
        
    records = query.order_by(BenchmarkRecord.created_at.desc()).limit(limit).all()
    
    results = [
        BenchmarkResultResponse(
            benchmark_id=str(getattr(r, "benchmark_id", "")),
            model_name=str(getattr(r, "model_name", "")),
            benchmark_name=str(getattr(r, "benchmark_name", "")),
            score=float(getattr(r, "score", 0.0)),
            metrics=dict(getattr(r, "metrics", {}) or {}),
            timestamp=getattr(r, "created_at", None) or datetime.utcnow(),
            samples_evaluated=int(getattr(r, "samples_evaluated", 0)),
        )
        for r in records
    ]
    
    logger.info(f"Found {len(results)} benchmark results")
    return results


@router.post("/compare", response_model=ModelComparisonResponse)
async def compare_models(
    request: ModelComparisonRequest,
    db: Session = Depends(get_db)
) -> ModelComparisonResponse:
    """
    Modelleri karşılaştır.
    
    Args:
        request: Karşılaştırma parametreleri
        db: Database session
        
    Returns:
        Karşılaştırma sonuçları
        
    Raises:
        HTTPException: Model bulunamazsa veya benchmark başarısız olursa
    """
    logger.info(f"Comparing models: {request.model_names} on {request.benchmark_name}")
    
    try:
        comparisons = []
        
        for model_name in request.model_names:
            # Run benchmark for each model
            runner = BenchmarkRunner(
                model_name=model_name,
                device="cpu"
            )
            
            result = runner.run_benchmark(
                benchmark_name=request.benchmark_name,
                dataset_path=request.dataset_path,
                max_samples=50,  # Smaller sample for comparison
                batch_size=8
            )
            
            comparisons.append({
                "model_name": model_name,
                "score": result.score,
                "metrics": result.metrics
            })
        
        # Determine winner (lower is better for perplexity, loss)
        if request.benchmark_name in ["perplexity", "loss"]:
            winner = min(comparisons, key=lambda x: x["score"])["model_name"]
        else:
            # Higher is better for BLEU, ROUGE, accuracy
            winner = max(comparisons, key=lambda x: x["score"])["model_name"]
        
        logger.info(f"Comparison completed: winner={winner}")
        
        response = ModelComparisonResponse(
            benchmark_name=request.benchmark_name,
            comparisons=comparisons,
            winner=winner
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Model comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/metrics/{model_name}")
async def get_model_metrics(
    model_name: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Model'in tüm metriklerini getir.
    
    Args:
        model_name: Model adı
        db: Database session
        
    Returns:
        Model metrikleri
        
    Raises:
        HTTPException: Model bulunamazsa
    """
    logger.info(f"Fetching metrics for model: {model_name}")
    
    try:
        from src.registry.model_registry import ModelRegistry
        
        registry = ModelRegistry(registry_dir="models")
        
        # Get model info
        model_versions = registry.get_model_versions(model_name)
        
        if not model_versions:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        
        # Get latest version metrics
        latest = model_versions[0]
        
        response = {
            "model_name": model_name,
            "version": latest.get("version"),
            "metrics": latest.get("metrics", {}),
            "training_config": latest.get("training_config", {}),
            "created_at": latest.get("created_at")
        }
        
        logger.info(f"Metrics fetched for {model_name}")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")


@router.get("/benchmarks")
async def list_available_benchmarks() -> Dict[str, Any]:
    """
    Mevcut benchmark'leri listele.
    
    Returns:
        Benchmark listesi ve açıklamaları
    """
    benchmarks = {
        "perplexity": {
            "name": "Perplexity",
            "description": "Model'in test verisindeki belirsizlik skoru (düşük = iyi)",
            "metric": "perplexity",
            "lower_is_better": True
        },
        "bleu": {
            "name": "BLEU Score",
            "description": "Text generation kalitesi (machine translation ve akıcılık)",
            "metric": "bleu",
            "lower_is_better": False
        },
        "rouge": {
            "name": "ROUGE Score",
            "description": "Text summarization kalitesi",
            "metric": "rouge",
            "lower_is_better": False
        },
        "gsm8k_cot": {
            "name": "Çok Adımlı Matematiksel Akıl Yürütme (GSM8K CoT)",
            "description": "Chain-of-Thought formatında çok adımlı matematik ve mantık problemleri çözümü",
            "metric": "accuracy",
            "category": "reasoning",
            "lower_is_better": False
        },
        "turkish_knowledge": {
            "name": "Türkçe Bilgi & Doğruluk Testi",
            "description": "Tarih, coğrafya, bilim ve kültür alanında Türkçe olgusal soru-cevap testi",
            "metric": "composite_score",
            "category": "knowledge",
            "lower_is_better": False
        },
        "turkish_summarization": {
            "name": "Türkçe Metin Özetleme Benchmark",
            "description": "Haber, bilim, ekonomi ve teknoloji metinlerinin özünü yakalama ve özetleme kalitesi (ROUGE-L, ChrF)",
            "metric": "rouge-l",
            "category": "summarization",
            "lower_is_better": False
        },
        "turkish_qa": {
            "name": "Türkçe Okuduğunu Anlama & Soru-Cevap (QA)",
            "description": "Bağlam metnine dayalı Türkçe anlama, çıkarım ve soru yanıtlama kabiliyeti (Exact Match, Token F1)",
            "metric": "token_f1",
            "category": "qa",
            "lower_is_better": False
        },
        "accuracy": {
            "name": "Accuracy",
            "description": "Classification doğruluğu",
            "metric": "accuracy",
            "lower_is_better": False
        }
    }
    
    return {
        "benchmarks": benchmarks,
        "count": len(benchmarks)
    }


@router.get("/results/{benchmark_id}", response_model=BenchmarkResultResponse)
async def get_benchmark_result(
    benchmark_id: str,
    db: Session = Depends(get_db)
) -> BenchmarkResultResponse:
    """
    Belirli bir benchmark sonucunu getir.
    """
    logger.info(f"Fetching benchmark result: {benchmark_id}")
    record = db.query(BenchmarkRecord).filter(BenchmarkRecord.benchmark_id == benchmark_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Benchmark result not found: {benchmark_id}")

    return BenchmarkResultResponse(
        benchmark_id=str(getattr(record, "benchmark_id", "")),
        model_name=str(getattr(record, "model_name", "")),
        benchmark_name=str(getattr(record, "benchmark_name", "")),
        score=float(getattr(record, "score", 0.0)),
        metrics=dict(getattr(record, "metrics", {}) or {}),
        timestamp=getattr(record, "created_at", None) or datetime.utcnow(),
        samples_evaluated=int(getattr(record, "samples_evaluated", 0)),
    )


@router.delete("/results/{benchmark_id}")
async def delete_benchmark_result(
    benchmark_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Benchmark sonucunu sil.
    """
    logger.info(f"Deleting benchmark result: {benchmark_id}")
    record = db.query(BenchmarkRecord).filter(BenchmarkRecord.benchmark_id == benchmark_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Benchmark result not found: {benchmark_id}")

    db.delete(record)
    db.commit()
    return {"message": f"Benchmark result {benchmark_id} deleted successfully"}


@router.get("/benchmarks/sample-questions", response_model=List[BenchmarkSampleQuestion])
async def get_benchmark_sample_questions(
    benchmark_name: str = Query("gsm8k_cot", description="Benchmark adı: gsm8k_cot, turkish_knowledge, turkish_summarization, turkish_qa vb.")
) -> List[BenchmarkSampleQuestion]:
    """
    Belirtilen benchmark'ın örnek sorularını ve hedef çözüm/cevaplarını listeler.
    """
    from src.evaluation.benchmarks import (
        create_gsm8k_cot_benchmark,
        create_turkish_knowledge_benchmark,
        create_turkish_summarization_benchmark,
        create_turkish_qa_benchmark,
    )
    
    if benchmark_name in ("gsm8k_cot", "gsm8k", "math_reasoning"):
        ds = create_gsm8k_cot_benchmark()
    elif benchmark_name in ("turkish_knowledge", "turkish_facts"):
        ds = create_turkish_knowledge_benchmark()
    elif benchmark_name in ("turkish_summarization", "summarization", "ozetleme"):
        ds = create_turkish_summarization_benchmark()
    elif benchmark_name in ("turkish_qa", "qa", "reading_comprehension"):
        ds = create_turkish_qa_benchmark()
    else:
        ds = create_gsm8k_cot_benchmark()

    questions = []
    for ex in ds.examples:
        target_str = ex.target if isinstance(ex.target, str) else ex.target[0]
        questions.append(BenchmarkSampleQuestion(
            id=ex.id,
            input=ex.input,
            target=target_str,
            domain=ex.metadata.get("domain", "genel"),
            category=ex.metadata.get("category", "genel"),
            difficulty=ex.metadata.get("difficulty"),
            numeric_answer=float(ex.metadata["numeric_answer"]) if "numeric_answer" in ex.metadata else None,
            steps=ex.metadata.get("steps"),
            keywords=ex.metadata.get("keywords"),
            context=ex.metadata.get("context"),
            question=ex.metadata.get("question"),
        ))
    return questions


@router.post("/radar-comparison", response_model=RadarComparisonResponse)
async def compare_models_radar(
    request: RadarComparisonRequest,
    db: Session = Depends(get_db)
) -> RadarComparisonResponse:
    """
    Seçilen modeller için 5 temel boyutta (Akıl Yürütme, Türkçe Bilgi, Akıcılık, Özetleme, Tutarlılık)
    radar kıyaslama metriklerini hesaplar.
    """
    dimension_names = [
        "Akıl Yürütme (GSM8K CoT)",
        "Türkçe Olgusal Bilgi",
        "Okuduğunu Anlama & QA",
        "Metin Özetleme",
        "Metin Akıcılığı (BLEU & ChrF)",
        "Model Tutarlılığı (PPL)"
    ]

    models_scores: List[RadarModelScore] = []
    dimension_max: Dict[str, float] = {d: -1.0 for d in dimension_names}
    winner_by_dim: Dict[str, str] = {d: "" for d in dimension_names}

    for model_name in request.model_names:
        runner = BenchmarkRunner(model_name=model_name, device="cpu")

        # 1. Reasoning (GSM8K CoT)
        try:
            r_res = runner.run_benchmark("gsm8k_cot", max_samples=8)
            reasoning_score = max(0.0, min(100.0, float(r_res.score)))
            reasoning_raw = reasoning_score
        except Exception as exc:
            raise HTTPException(422, f"Real benchmark unavailable: {exc}") from exc

        # 2. Knowledge (Turkish)
        try:
            k_res = runner.run_benchmark("turkish_knowledge", max_samples=8)
            knowledge_score = max(0.0, min(100.0, float(k_res.score)))
            knowledge_raw = knowledge_score
        except Exception as exc:
            raise HTTPException(422, f"Real benchmark unavailable: {exc}") from exc

        # 3. QA & Reading Comprehension (Turkish QA)
        try:
            qa_res = runner.run_benchmark("turkish_qa", max_samples=8)
            qa_raw = float(qa_res.score)
            qa_score = max(0.0, min(100.0, qa_raw))
        except Exception as exc:
            raise HTTPException(422, f"Real benchmark unavailable: {exc}") from exc

        # 4. Summarization (Turkish Summarization)
        try:
            sm_res = runner.run_benchmark("turkish_summarization", max_samples=8)
            sm_raw = float(sm_res.score)
            sm_score = max(0.0, min(100.0, sm_raw * 100.0 if sm_raw <= 1.0 else sm_raw))
        except Exception as exc:
            raise HTTPException(422, f"Real benchmark unavailable: {exc}") from exc

        # 5. Fluency (BLEU & ChrF)
        try:
            b_res = runner.run_benchmark("bleu", max_samples=8)
            fluency_score = max(0.0, min(100.0, float(b_res.score)))
            fluency_raw = fluency_score
        except Exception as exc:
            raise HTTPException(422, f"Real benchmark unavailable: {exc}") from exc

        # 6. Stability (Perplexity inverse)
        try:
            p_res = runner.run_benchmark("perplexity", dataset_path=request.dataset_path, max_samples=8)
            ppl = float(p_res.score)
            stability_raw = ppl
            stability_score = max(0.0, min(100.0, 100.0 - (ppl - 5.0) * 2.0))
        except Exception as exc:
            raise HTTPException(422, f"Real benchmark unavailable: {exc}") from exc

        dim_scores = [
            RadarDimensionScore(
                dimension_key="reasoning",
                dimension_name=dimension_names[0],
                score=round(reasoning_score, 1),
                raw_metric="accuracy",
                raw_score=round(reasoning_raw, 1)
            ),
            RadarDimensionScore(
                dimension_key="knowledge",
                dimension_name=dimension_names[1],
                score=round(knowledge_score, 1),
                raw_metric="composite",
                raw_score=round(knowledge_raw, 1)
            ),
            RadarDimensionScore(
                dimension_key="qa",
                dimension_name=dimension_names[2],
                score=round(qa_score, 1),
                raw_metric="token_f1",
                raw_score=round(qa_raw, 1)
            ),
            RadarDimensionScore(
                dimension_key="summarization",
                dimension_name=dimension_names[3],
                score=round(sm_score, 1),
                raw_metric="rouge-l",
                raw_score=round(sm_raw, 3)
            ),
            RadarDimensionScore(
                dimension_key="fluency",
                dimension_name=dimension_names[4],
                score=round(fluency_score, 1),
                raw_metric="bleu",
                raw_score=round(fluency_raw, 1)
            ),
            RadarDimensionScore(
                dimension_key="stability",
                dimension_name=dimension_names[5],
                score=round(stability_score, 1),
                raw_metric="perplexity",
                raw_score=round(stability_raw, 2)
            ),
        ]

        overall_avg = round(sum(d.score for d in dim_scores) / len(dim_scores), 1)

        for ds in dim_scores:
            if ds.score > dimension_max[ds.dimension_name]:
                dimension_max[ds.dimension_name] = ds.score
                winner_by_dim[ds.dimension_name] = model_name

        models_scores.append(RadarModelScore(
            model_name=model_name,
            overall_average=overall_avg,
            dimensions=dim_scores
        ))

    overall_winner = max(models_scores, key=lambda m: m.overall_average).model_name if models_scores else ""

    return RadarComparisonResponse(
        models=models_scores,
        dimensions=dimension_names,
        winner_by_dimension=winner_by_dim,
        overall_winner=overall_winner
    )
