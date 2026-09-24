"""
backend/routers/models.py

Model Registry API Endpoints

Bu modül eğitilmiş modellerin yönetimi için REST API sağlar:
- Model listeleme ve arama
- Model detayları, konfigürasyonu ve metrikleri
- Model silme
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from sqlalchemy.orm import Session
import logging
from pathlib import Path

from backend.database import get_db
from backend.models import UserRecord
from backend.security.dependencies import get_current_user, require_role
from src.registry.model_registry import ModelRegistry, ModelMetadata
from src.export.model_exporter import ModelExporter

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


class ExportModelRequest(BaseModel):
    version: Optional[str] = None
    export_format: str = "onnx"  # onnx, torchscript, gguf
    quantization: str = "none"   # none, fp16, int8, int4


class ArtifactCompatibilityCheckRequest(BaseModel):
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_vocab_size: Optional[int] = None
    model_tokenizer_id: Optional[str] = None
    tokenizer_id: Optional[str] = None
    tokenizer_vocab_size: Optional[int] = None
    dataset_version: Optional[str] = None
    tokenizer_dataset_version: Optional[str] = None


class ArtifactCompatibilityCheckResponse(BaseModel):
    compatible: bool
    status: str  # "compatible" | "warning" | "incompatible"
    warnings: List[str]
    notes: List[str]
    checks: Dict[str, Any]


@router.post("/validate-compatibility", response_model=ArtifactCompatibilityCheckResponse)
def validate_artifact_compatibility(
    req: ArtifactCompatibilityCheckRequest,
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user)
) -> ArtifactCompatibilityCheckResponse:
    """
    Model, Tokenizer ve Dataset artefaktlarının karşılıklı uyumluluğunu doğrular.
    - Model vocab_size vs Tokenizer vocab_size (indeks taşması kontrolü)
    - Tokenizer kimliği / mimari eşleşmesi
    - Dataset versiyon takibi
    """
    warnings: List[str] = []
    notes: List[str] = []
    checks: Dict[str, Any] = {}

    m_vocab = req.model_vocab_size
    m_tok_id = req.model_tokenizer_id

    # Eğer model_name verildiyse registry'den metadata oku
    if req.model_name:
        try:
            registry = ModelRegistry(registry_dir="models")
            loaded = registry.load_model(req.model_name, version=req.model_version)
            meta = loaded.get("metadata")
            if meta:
                tr_cfg = meta.training_config or {}
                if m_vocab is None:
                    m_vocab = tr_cfg.get("vocab_size")
                if m_tok_id is None:
                    m_tok_id = tr_cfg.get("tokenizer_id") or tr_cfg.get("tokenizer_name")
        except Exception:
            pass

    t_vocab = req.tokenizer_vocab_size
    t_ds_ver = req.tokenizer_dataset_version

    # Eğer tokenizer_id verildiyse DB'den TokenizerRecord kontrol et
    if req.tokenizer_id:
        from backend.models import TokenizerRecord
        tok_rec = db.query(TokenizerRecord).filter(TokenizerRecord.tokenizer_id == req.tokenizer_id).first()
        if tok_rec:
            if t_vocab is None:
                t_vocab = tok_rec.vocab_size
            if t_ds_ver is None:
                t_ds_ver = tok_rec.dataset_version

    # 1. Sözlük boyutu kontrolü
    checks["model_vocab_size"] = m_vocab
    checks["tokenizer_vocab_size"] = t_vocab
    if m_vocab is not None and t_vocab is not None:
        if m_vocab < t_vocab:
            warnings.append(
                f"Kritik İndeks Taşması Riski: Model sözlük boyutu ({m_vocab}), Tokenizer sözlük boyutundan ({t_vocab}) küçük! Model çıktılarında sınır aşımı (IndexError) meydana gelecektir."
            )
        elif m_vocab > t_vocab:
            notes.append(
                f"Bilgi: Model sözlük kapasitesi ({m_vocab}), Tokenizer sözlüğünden ({t_vocab}) büyük. Ekstra embedding kapasitesi mevcut."
            )

    # 2. Tokenizer eşleşmesi
    checks["model_tokenizer_id"] = m_tok_id
    checks["tokenizer_id"] = req.tokenizer_id
    if m_tok_id and req.tokenizer_id and str(m_tok_id) != str(req.tokenizer_id):
        warnings.append(
            f"Tokenizer Uyuşmazlığı: Model '{m_tok_id}' tokenizer'ı ile eğitilmiş; seçili tokenizer '{req.tokenizer_id}'. Token-ID haritaları farklı olabilir."
        )

    # 3. Dataset versiyonu
    checks["dataset_version"] = req.dataset_version
    checks["tokenizer_dataset_version"] = t_ds_ver
    if req.dataset_version and t_ds_ver and req.dataset_version != t_ds_ver:
        notes.append(
            f"Bilgi: Tokenizer derleme dataset versiyonu ({t_ds_ver}) ile mevcut dataset versiyonu ({req.dataset_version}) farklılık gösteriyor."
        )

    is_incompatible = any("Kritik" in w for w in warnings)
    status_str = "incompatible" if is_incompatible else ("warning" if warnings else "compatible")

    return ArtifactCompatibilityCheckResponse(
        compatible=not is_incompatible,
        status=status_str,
        warnings=warnings,
        notes=notes,
        checks=checks,
    )


@router.get("", response_model=List[Dict[str, Any]])
def list_models(
    environment: Optional[str] = Query(None, description="Filtre: development, staging, production"),
    current_user: UserRecord = Depends(get_current_user)
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
    version: Optional[str] = Query(None, description="Belirli versiyon (varsayılan: en son)"),
    current_user: UserRecord = Depends(get_current_user)
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
    version: Optional[str] = Query(None, description="Silinecek versiyon"),
    current_user: UserRecord = Depends(require_role("admin"))
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


@router.post("/{model_name}/export", response_model=Dict[str, Any])
def export_model(
    model_name: str,
    payload: ExportModelRequest
) -> Dict[str, Any]:
    """
    Modeli ONNX, TorchScript veya GGUF formatına dönüştürür ve opsiyonel FP16/INT8/INT4 kuantizasyonu uygular.
    """
    exporter = ModelExporter(registry_dir="models")
    try:
        result = exporter.export_pipeline(
            model_name=model_name,
            version=payload.version,
            export_format=payload.export_format,
            quantization=payload.quantization
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        logger.error(f"Error exporting model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Dışa aktarma hatası: {str(e)}")


@router.get("/{model_name}/exports", response_model=List[Dict[str, Any]])
def list_model_exports(
    model_name: str,
    version: Optional[str] = Query(None, description="Filtrelenecek model versiyonu")
) -> List[Dict[str, Any]]:
    """
    Model için daha önce dışa aktarılmış dosyaların ve kuantizasyonların manifest listesini döner.
    """
    exporter = ModelExporter(registry_dir="models")
    try:
        return exporter.list_exports(model_name=model_name, version=version)
    except Exception as e:
        logger.error(f"Error listing exports for {model_name}: {e}")
        return []


@router.get("/{model_name}/download/{file_name}")
def download_exported_model(
    model_name: str,
    file_name: str,
    version: Optional[str] = Query(None, description="Opsiyonel model versiyonu")
):
    """
    Dışa aktarılan ONNX, TorchScript veya GGUF dosyasını indirir.
    Path traversal saldırılarına karşı güvenli dosya kontrolü içerir.
    """
    safe_filename = Path(file_name).name
    registry = ModelRegistry()
    base_dir = Path(registry.models_dir) / model_name

    if not base_dir.exists():
        raise HTTPException(status_code=404, detail="Model bulunamadı")

    target_file = None
    if version:
        candidate = base_dir / version / "exports" / safe_filename
        if candidate.is_file():
            target_file = candidate
    else:
        # Search in any version exports dir
        for candidate in base_dir.glob(f"*/exports/{safe_filename}"):
            if candidate.is_file():
                target_file = candidate
                break

    if not target_file or not target_file.is_file():
        raise HTTPException(status_code=404, detail=f"Dışa aktarılan dosya bulunamadı: {safe_filename}")

    return FileResponse(
        path=str(target_file),
        filename=safe_filename,
        media_type="application/octet-stream"
    )

