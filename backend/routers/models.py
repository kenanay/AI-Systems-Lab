"""
backend/routers/models.py

Model Registry API Endpoints

Bu modül eğitilmiş modellerin yönetimi için REST API sağlar:
- Model listeleme ve arama
- Model detayları, konfigürasyonu ve metrikleri
- Model silme
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging
from pathlib import Path

from src.registry.model_registry import ModelRegistry, ModelMetadata

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/models",
    tags=["models"]
)


class ModelItemResponse(BaseModel):
    model_name: str
    version: str
    description: str
    architecture: str
    parameters: int
    metrics: Dict[str, Any]
    training_config: Dict[str, Any]
    tags: List[str]
    created_at: str


@router.get("", response_model=List[Dict[str, Any]])
def list_models(
    environment: Optional[str] = Query(None, description="Filtre: development, staging, production")
) -> List[Dict[str, Any]]:
    """
    Kayıtlı tüm modelleri listeler.
    """
    registry = ModelRegistry(registry_dir="models")
    try:
        models = registry.list_models(environment=environment)
        result = []
        for m in models:
            result.append({
                "model_name": m.model_name,
                "version": m.version,
                "description": m.description,
                "architecture": m.architecture,
                "parameters": m.parameters,
                "metrics": m.metrics,
                "training_config": m.training_config,
                "tags": m.tags,
                "created_at": m.created_at,
                "environment": m.environment,
                "model_hash": m.model_hash,
                "file_size_mb": m.file_size_mb
            })
        return result
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return []


@router.get("/{model_name}", response_model=Dict[str, Any])
def get_model(
    model_name: str,
    version: Optional[str] = Query(None, description="Belirli versiyon (varsayılan: en son)")
) -> Dict[str, Any]:
    """
    Belirli bir modelin detaylarını getirir.
    """
    registry = ModelRegistry(registry_dir="models")
    try:
        model_info = registry.load_model(model_name=model_name, version=version)
        meta = model_info["metadata"]
        return {
            "model_name": meta.model_name,
            "version": meta.version,
            "description": meta.description,
            "architecture": meta.architecture,
            "parameters": meta.parameters,
            "metrics": meta.metrics,
            "training_config": meta.training_config,
            "tokenizer_info": meta.tokenizer_info,
            "tags": meta.tags,
            "created_at": meta.created_at,
            "model_hash": meta.model_hash,
            "checkpoint_path": str(model_info.get("checkpoint_path", ""))
        }
    except Exception as e:
        logger.error(f"Error getting model {model_name}: {e}")
        raise HTTPException(status_code=404, detail=f"Model bulunamadı: {e}")


@router.post("/{model_name}/verify", response_model=Dict[str, Any])
def verify_model(
    model_name: str,
    version: Optional[str] = Query(None, description="Doğrulanacak versiyon (varsayılan: en son)")
) -> Dict[str, Any]:
    """
    Model checkpoint dosyasının SHA-256 bütünlüğünü ve imzasını doğrular.
    Diskteki dosya hash'i ile kayıtlı model_hash'i karşılaştırır.
    """
    registry = ModelRegistry(registry_dir="models")
    try:
        result = registry.verify_model(model_name=model_name, version=version)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        logger.error(f"Error verifying model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model_name}")
def delete_model(
    model_name: str,
    version: Optional[str] = Query(None, description="Silinecek versiyon")
) -> Dict[str, Any]:
    """
    Model kaydını registry'den siler.
    """
    registry = ModelRegistry(registry_dir="models")
    try:
        # Check if exists
        models = registry.get_model_versions(model_name)
        if not models:
            raise HTTPException(status_code=404, detail="Model bulunamadı")
        
        # Delete model from registry
        registry.delete_model(model_name, version=version)
        target = f"{model_name} v{version}" if version else model_name
        return {"status": "success", "message": f"{target} başarıyla kaldırıldı"}
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
