"""
backend/services/compatibility_service.py

Model, Tokenizer ve Dataset Artefakt Uyumluluk Doğrulama Servisi.
İstemciden gelen doğrulanmamış parametrelere güvenmeden, doğrudan
sunucu tarafındaki ModelRegistry ve veritabanı kayıtları üzerinden
uyumluluğu otoriter biçimde denetler.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
from sqlalchemy.orm import Session
from pathlib import Path

from src.registry.model_registry import ModelRegistry
from backend.models import TokenizerRecord, DatasetVersion

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityResult:
    compatible: bool
    status: str  # "compatible" | "warning" | "incompatible" | "verification_failed"
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compatible": self.compatible,
            "status": self.status,
            "warnings": self.warnings + self.errors,
            "notes": self.notes,
            "checks": self.checks,
        }


class CompatibilityService:
    """Sunucu tarafı doğrulanmış artefakt uyumluluk kontrolcüsü."""

    def __init__(self, db: Session, registry_dir: str = "models"):
        self.db = db
        self.registry = ModelRegistry(registry_dir=registry_dir)

    def verify(
        self,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
    ) -> CompatibilityResult:
        errors: List[str] = []
        warnings: List[str] = []
        notes: List[str] = []
        checks: Dict[str, Any] = {
            "model_verified": False,
            "tokenizer_verified": False,
            "dataset_verified": False,
        }

        # Hiçbir parametre verilmediyse doğrulanamaz
        if not model_name and not tokenizer_id and not dataset_id:
            return CompatibilityResult(
                compatible=False,
                status="verification_failed",
                errors=["Doğrulama için en az bir Model, Tokenizer veya Dataset belirtilmelidir."],
                checks=checks
            )

        m_vocab: Optional[int] = None
        m_tok_id: Optional[str] = None
        m_meta = None

        # 1. Model Doğrulaması (Sunucu Registry'sinden)
        if model_name:
            try:
                loaded = self.registry.load_model(model_name, version=model_version, verify_integrity=True)
                m_meta = loaded.get("metadata")
                if not m_meta:
                    errors.append(f"Model metadata kaydı eksik veya bozuk: {model_name}")
                else:
                    checks["model_verified"] = True
                    checks["model_name"] = model_name
                    checks["model_version"] = m_meta.get("version")
                    tr_cfg = m_meta.get("training_config") or {}
                    m_vocab = tr_cfg.get("vocab_size")
                    m_tok_id = tr_cfg.get("tokenizer_id") or tr_cfg.get("tokenizer_name")
                    checks["server_model_vocab_size"] = m_vocab
                    checks["server_model_tokenizer_id"] = m_tok_id
            except (ValueError, FileNotFoundError) as e:
                errors.append(f"Model sunucu registry'sinde doğrulanamadı: {e}")
            except Exception as e:
                logger.error(f"Error loading model from registry for verification: {e}")
                errors.append(f"Model registry okuma hatası: {str(e)}")

        # 2. Tokenizer Doğrulaması (Veritabanından ve Diskten)
        t_vocab: Optional[int] = None
        t_ds_ver: Optional[str] = None

        if tokenizer_id:
            tok_rec = self.db.query(TokenizerRecord).filter(
                TokenizerRecord.tokenizer_id == tokenizer_id
            ).first()

            if not tok_rec:
                errors.append(f"Tokenizer veritabanı kayıtlarında bulunamadı: {tokenizer_id}")
            else:
                checks["tokenizer_verified"] = True
                checks["tokenizer_id"] = tokenizer_id
                t_vocab = tok_rec.vocab_size
                t_ds_ver = getattr(tok_rec, "dataset_version", None) or (
                    tok_rec.source_dataset_ids[0] if getattr(tok_rec, "source_dataset_ids", None) else None
                )
                checks["server_tokenizer_vocab_size"] = t_vocab
                checks["server_tokenizer_dataset_version"] = t_ds_ver

        # 3. Dataset Doğrulaması
        if dataset_id:
            ds_rec = self.db.query(DatasetVersion).filter(
                DatasetVersion.version_id == dataset_id
            ).first()
            if ds_rec:
                checks["dataset_verified"] = True
                checks["dataset_id"] = dataset_id
                checks["server_dataset_version"] = ds_rec.version_name
            else:
                notes.append(f"Dataset kaydı veritabanında bulunamadı veya yerel dosya: {dataset_id}")

        # Eğer model veya tokenizer verilip sunucuda doğrulanamadıysa, sessiz geçme -> verification_failed!
        if (model_name and not checks["model_verified"]) or (tokenizer_id and not checks["tokenizer_verified"]):
            return CompatibilityResult(
                compatible=False,
                status="verification_failed",
                errors=errors,
                warnings=warnings,
                notes=notes,
                checks=checks
            )

        # 4. Model & Tokenizer Karşılıklı Uyumluluk Kontrolleri
        if checks["model_verified"] and checks["tokenizer_verified"]:
            # A. Tokenizer Kimliği Eşleşmesi
            if m_tok_id and str(m_tok_id) != str(tokenizer_id):
                errors.append(
                    f"Kritik Tokenizer Uyuşmazlığı: Model '{m_tok_id}' tokenizer'ı ile eğitilmiş; seçili tokenizer '{tokenizer_id}'. Farklı sözlük eşlemeleri modelin bozuk veya anlamsız çıktı üretmesine yol açacaktır."
                )

            # B. Sözlük Boyutu Sınır Aşımı (Index Overflow)
            if m_vocab is not None and t_vocab is not None:
                if m_vocab < t_vocab:
                    errors.append(
                        f"Kritik İndeks Taşması: Model sözlük boyutu ({m_vocab}), Tokenizer sözlük boyutundan ({t_vocab}) küçük! Model inference/training sırasında sınır dışı token ID'leri IndexError üretecektir."
                    )
                elif m_vocab > t_vocab:
                    notes.append(
                        f"Bilgi: Model sözlük boyutu ({m_vocab}), Tokenizer sözlük boyutundan ({t_vocab}) büyük. Fazladan embedding rezervi mevcuttur."
                    )

            # C. Dataset Sürüm Bilgilendirmesi
            active_ds_ver = dataset_version or checks.get("server_dataset_version")
            if active_ds_ver and t_ds_ver and active_ds_ver != t_ds_ver:
                notes.append(
                    f"Bilgi: Tokenizer'ın eğitildiği dataset versiyonu ({t_ds_ver}) ile mevcut dataset versiyonu ({active_ds_ver}) farklı."
                )

        is_compatible = len(errors) == 0
        if not is_compatible:
            status = "incompatible"
        elif len(warnings) > 0:
            status = "warning"
        else:
            status = "compatible"

        return CompatibilityResult(
            compatible=is_compatible,
            status=status,
            errors=errors,
            warnings=warnings,
            notes=notes,
            checks=checks
        )
