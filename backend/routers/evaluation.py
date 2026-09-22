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

from backend.database import get_db
from src.evaluation.benchmarks import BenchmarkRunner, BenchmarkResult
# from src.evaluation.metrics import compute_perplexity  # TODO: Implement when needed

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/evaluation",
    tags=["evaluation"]
)


# Request/Response Models

class BenchmarkRunRequest(BaseModel):
    """Benchmark çalıştırma request."""
    model_name: str = Field(..., description="Model adı (registry'den)")
    benchmark_name: str = Field(..., description="Benchmark adı (perplexity, bleu, rouge)")
    dataset_path: Optional[str] = Field(None, description="Test dataset path (optional)")
    max_samples: int = Field(100, description="Maximum sample sayısı")
    batch_size: int = Field(8, description="Batch size")
    
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
    model_names: List[str] = Field(..., description="Karşılaştırılacak model isimleri")
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


# Endpoints

@router.post("/run", response_model=BenchmarkResultResponse)
async def run_benchmark(
    request: BenchmarkRunRequest,
    db: Session = Depends(get_db)
) -> BenchmarkResultResponse:
    """
    Benchmark çalıştır.
    
    Args:
        request: Benchmark parametreleri
        db: Database session
        
    Returns:
        Benchmark sonuçları
        
    Raises:
        HTTPException: Model bulunamazsa veya benchmark başarısız olursa
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
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    db: Session = Depends(get_db)
) -> List[BenchmarkResultResponse]:
    """
    Benchmark sonuçlarını listele.
    
    Args:
        model_name: Model name filter (optional)
        benchmark_name: Benchmark name filter (optional)
        limit: Maximum sonuç sayısı
        db: Database session
        
    Returns:
        Benchmark sonuçları listesi
    """
    logger.info(f"Listing benchmark results: model={model_name}, benchmark={benchmark_name}")
    
    # TODO: Database'den benchmark results çekme
    # Şimdilik boş liste dönüyoruz (future implementation)
    
    results = []
    
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
            "description": "Text generation kalitesi (machine translation için)",
            "metric": "bleu",
            "lower_is_better": False
        },
        "rouge": {
            "name": "ROUGE Score",
            "description": "Text summarization kalitesi",
            "metric": "rouge",
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


@router.delete("/results/{benchmark_id}")
async def delete_benchmark_result(
    benchmark_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Benchmark sonucunu sil.
    
    Args:
        benchmark_id: Benchmark ID
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        HTTPException: Benchmark bulunamazsa
    """
    logger.info(f"Deleting benchmark result: {benchmark_id}")
    
    # TODO: Database'den benchmark result silme
    # Şimdilik not implemented
    
    raise HTTPException(status_code=501, detail="Not implemented yet")
