"""
backend/routers/synthetic_lab.py

Synthetic Data Generation, Quality Scoring & Filtering Lab REST API Endpoints.
Covers:
- Template definitions & generation paradigms
- Batch synthetic generation (Self-Instruct, Chain-of-Thought, Code, Textbook)
- Multi-stage filtering pipeline & funnel analytics
- Single sample scoring & diagnostic evaluation
- Curated synthetic dataset export
"""

import json
import logging
from io import BytesIO
from uuid import uuid4
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import FileRecord, DocumentRecord
from backend.storage import storage_manager
from src.quality.synthetic_curation import (
    SyntheticDataGenerator,
    QualityScoringEngine,
    QualityFilterPipeline,
    SyntheticSample,
    FilterThresholds,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/synthetic-lab",
    tags=["synthetic-lab"]
)

generator = SyntheticDataGenerator()
scorer = QualityScoringEngine()
pipeline = QualityFilterPipeline()


# ============================================================================
# Request / Response Models
# ============================================================================

class FilterThresholdsModel(BaseModel):
    """Filtreleme eşik parametreleri."""
    min_perplexity: float = Field(5.0, ge=0.1, le=100.0, description="Minimum kabul edilebilir perplexity")
    max_perplexity: float = Field(120.0, ge=10.0, le=500.0, description="Maksimum kabul edilebilir perplexity")
    min_words: int = Field(15, ge=1, le=500, description="Minimum kelime sayısı")
    max_words: int = Field(1500, ge=50, le=10000, description="Maksimum kelime sayısı")
    max_repetition_ratio: float = Field(0.18, ge=0.01, le=0.8, description="Maksimum n-gram tekrarlama oranı")
    min_quality_score: float = Field(65.0, ge=0.0, le=100.0, description="Minimum genel kalite puanı")
    pii_action: str = Field("reject", description="PII aksiyonu: 'reject', 'mask', 'allow'")
    max_jaccard_similarity: float = Field(0.82, ge=0.1, le=1.0, description="Maksimum benzerlik eşiği (dedup)")


class SyntheticGenerateRequest(BaseModel):
    """Sentetik veri üretimi isteği."""
    paradigm: str = Field("self_instruct", description="Üretim paradigması (self_instruct, chain_of_thought, code_synthesis, textbook_qa)")
    domain: str = Field("computer_science", description="Alan / Konu (computer_science, mathematics, natural_sciences, turkish_knowledge)")
    complexity: str = Field("intermediate", description="Zorluk seviyesi (basic, intermediate, advanced)")
    count: int = Field(5, ge=1, le=50, description="Üretilecek örnek sayısı")
    include_edge_cases: bool = Field(True, description="Filtreleme testi için kasıtlı kusurlu uç örnekler eklensin mi?")


class SyntheticFilterRequest(BaseModel):
    """Filtreleme pipeline çalıştırma isteği."""
    samples: List[Dict[str, Any]] = Field(..., description="Filtrelenecek sentetik örnek listesi")
    thresholds: Optional[FilterThresholdsModel] = Field(None, description="Özel eşik değerleri (boş ise varsayılan)")


class ScoreSampleRequest(BaseModel):
    """Tekil metin puanlama isteği."""
    instruction: str = Field(..., description="Kullanıcı talimatı / soru")
    response: str = Field(..., description="Model yanıtı")
    input_context: Optional[str] = Field("", description="Varsa girdi bağlamı")


class ExportSyntheticRequest(BaseModel):
    """Seçili/filtrelenmiş verileri export isteği."""
    samples: List[Dict[str, Any]] = Field(..., description="Export edilecek örnekler")
    dataset_name: str = Field("synthetic_curated_v1", description="Oluşturulacak dataset adı")
    export_format: str = Field("jsonl", description="Format: 'jsonl' veya 'parquet'")


class IngestToDatasetRequest(BaseModel):
    """Filtrelenmiş sentetik örnekleri doğrudan FileRecord ve DocumentRecord olarak platforma aktarma isteği."""
    samples: List[Dict[str, Any]] = Field(..., description="Aktarılacak sentetik örnekler")
    dataset_name: str = Field("synthetic_curated_v1", description="Dataset / dosya ismi")
    domain: Optional[str] = Field("general", description="Alan / konu")
    paradigm: Optional[str] = Field("self_instruct", description="Üretim paradigması")
    training_allowed: bool = Field(True, description="Eğitim için izinli olarak işaretlensin mi?")


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/templates")
async def get_templates_metadata() -> Dict[str, Any]:
    """
    Kullanılabilir üretim paradigmalarını, alanları ve hazır filtreleme profillerini döner.
    """
    return {
        "paradigms": [
            {
                "id": "self_instruct",
                "name": "Self-Instruct (Alpaca Tarzı)",
                "description": "Görev tohumlarından çok yönlü talimat-yanıt çiftleri türetme.",
                "icon": "⚡",
            },
            {
                "id": "chain_of_thought",
                "name": "Düşünce Zinciri (Chain-of-Thought)",
                "description": "Adım adım mantıksal akıl yürütme ve problem çözme çiftleri.",
                "icon": "🧠",
            },
            {
                "id": "code_synthesis",
                "name": "Kod & Algoritma Sentezi",
                "description": "Programlama problemleri, kod blokları ve karmaşıklık analizleri.",
                "icon": "💻",
            },
            {
                "id": "textbook_qa",
                "name": "Ders Kitabı Q&A (Cosmopedia Tarzı)",
                "description": "Didaktik, pedagojik açıklamalar ve derin kavramsal okuma metinleri.",
                "icon": "📚",
            },
        ],
        "domains": [
            {"id": "computer_science", "name": "Bilgisayar Bilimleri & Algoritmalar"},
            {"id": "mathematics", "name": "Matematik & Olasılık"},
            {"id": "natural_sciences", "name": "Doğa Bilimleri & Biyoloji"},
            {"id": "turkish_knowledge", "name": "Türk Dili & Genel Kültür"},
        ],
        "complexities": ["basic", "intermediate", "advanced"],
        "preset_profiles": {
            "strict": {
                "name": "Katı Filtre (Yüksek Kalite SFT)",
                "min_perplexity": 10.0,
                "max_perplexity": 85.0,
                "min_words": 25,
                "max_words": 1000,
                "max_repetition_ratio": 0.12,
                "min_quality_score": 75.0,
                "pii_action": "reject",
                "max_jaccard_similarity": 0.75,
            },
            "balanced": {
                "name": "Dengeli Filtre (Varsayılan Ön Eğitim / Post-Training)",
                "min_perplexity": 5.0,
                "max_perplexity": 120.0,
                "min_words": 15,
                "max_words": 1500,
                "max_repetition_ratio": 0.18,
                "min_quality_score": 65.0,
                "pii_action": "mask",
                "max_jaccard_similarity": 0.82,
            },
            "lenient": {
                "name": "Esnek Filtre (Geniş Kapsamlı Veri Çeşitliliği)",
                "min_perplexity": 3.0,
                "max_perplexity": 160.0,
                "min_words": 8,
                "max_words": 3000,
                "max_repetition_ratio": 0.28,
                "min_quality_score": 50.0,
                "pii_action": "mask",
                "max_jaccard_similarity": 0.90,
            }
        }
    }


@router.post("/generate")
async def generate_synthetic_samples(request: SyntheticGenerateRequest) -> Dict[str, Any]:
    """
    Belirtilen paradigma, alan ve parametrelerde sentetik veri kümesi üretir.
    """
    try:
        samples = generator.generate_batch(
            paradigm=request.paradigm,
            domain=request.domain,
            complexity=request.complexity,
            count=request.count,
            include_edge_cases=request.include_edge_cases
        )
        return {
            "total_generated": len(samples),
            "paradigm": request.paradigm,
            "domain": request.domain,
            "complexity": request.complexity,
            "samples": [s.to_dict() for s in samples]
        }
    except Exception as e:
        logger.error(f"Sentetik veri üretimi başarısız: {e}")
        raise HTTPException(status_code=500, detail=f"Sentetik veri üretimi başarısız: {str(e)}")


@router.post("/filter")
async def filter_synthetic_pipeline(request: SyntheticFilterRequest) -> Dict[str, Any]:
    """
    Girdi olarak verilen sentetik örneklere çok aşamalı kalite filtresini uygular ve huni analizini döner.
    """
    try:
        # Convert dictionary samples to SyntheticSample objects
        sample_objs: List[SyntheticSample] = []
        for s_dict in request.samples:
            sample_objs.append(
                SyntheticSample(
                    id=s_dict.get("id", "SYNTH-UNKNOWN"),
                    instruction=s_dict.get("instruction", ""),
                    input_context=s_dict.get("input_context", ""),
                    response=s_dict.get("response", ""),
                    paradigm=s_dict.get("paradigm", "self_instruct"),
                    domain=s_dict.get("domain", "computer_science"),
                    complexity=s_dict.get("complexity", "intermediate"),
                    metrics=s_dict.get("metrics", {})
                )
            )

        # Build filter thresholds
        if request.thresholds:
            t_obj = FilterThresholds(
                min_perplexity=request.thresholds.min_perplexity,
                max_perplexity=request.thresholds.max_perplexity,
                min_words=request.thresholds.min_words,
                max_words=request.thresholds.max_words,
                max_repetition_ratio=request.thresholds.max_repetition_ratio,
                min_quality_score=request.thresholds.min_quality_score,
                pii_action=request.thresholds.pii_action,
                max_jaccard_similarity=request.thresholds.max_jaccard_similarity,
            )
        else:
            t_obj = FilterThresholds()

        result = pipeline.run_pipeline(sample_objs, t_obj)
        return result
    except Exception as e:
        logger.error(f"Filtreleme pipeline hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Filtreleme pipeline hatası: {str(e)}")


@router.post("/score-sample")
async def score_single_sample(request: ScoreSampleRequest) -> Dict[str, Any]:
    """
    Tekil bir metin örneğinin perplexity, tekrarlama ve kalite metriklerini detaylı teşhis eder.
    """
    try:
        metrics = scorer.score_sample(
            instruction=request.instruction,
            response=request.response,
            input_context=request.input_context or ""
        )
        return metrics
    except Exception as e:
        logger.error(f"Örnek puanlama hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Örnek puanlama hatası: {str(e)}")


@router.post("/export")
async def export_synthetic_dataset(request: ExportSyntheticRequest) -> Dict[str, Any]:
    """
    Ayıklanan sentetik örnekleri JSONL olarak dışa aktarır veya platformun dataset sistemine bağlar.
    """
    try:
        sample_count = len(request.samples)
        if sample_count == 0:
            raise HTTPException(status_code=400, detail="Export edilecek örnek bulunamadı.")

        jsonl_lines = [json.dumps(s, ensure_ascii=False) for s in request.samples]
        raw_content = "\n".join(jsonl_lines)

        return {
            "success": True,
            "dataset_name": request.dataset_name,
            "export_format": request.export_format,
            "total_samples": sample_count,
            "approx_tokens": sum(len(s.get("response", "").split()) for s in request.samples),
            "preview_jsonl": jsonl_lines[:3],
            "raw_payload_size_bytes": len(raw_content.encode("utf-8")),
            "message": f"{sample_count} adet filtrelenmiş sentetik örnek başarıyla hazırlandı."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Export hatası: {str(e)}")


@router.post("/ingest-to-dataset")
async def ingest_synthetic_to_dataset(
    request: IngestToDatasetRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Küratörlü sentetik veri çiftlerini doğrudan canonical veritabanına (FileRecord ve DocumentRecord)
    yazar, böylece Dataset Compiler ve Eğitim modelleri için anında seçilebilir hale gelir.
    """
    try:
        sample_count = len(request.samples)
        if sample_count == 0:
            raise HTTPException(status_code=400, detail="Aktarılacak örnek bulunamadı.")

        # 1. JSONL formatına dönüştür ve stream hazırla
        file_id = f"file_synth_{uuid4().hex[:8]}"
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in request.dataset_name)
        if not safe_name:
            safe_name = "synthetic_curated"
        filename = f"{safe_name}.jsonl"

        # Her satırı JSONL formatında hazırla
        jsonl_lines = [json.dumps(s, ensure_ascii=False) for s in request.samples]
        file_content = "\n".join(jsonl_lines).encode("utf-8")
        file_size = len(file_content)

        # 2. Fiziksel dosya depolama
        file_stream = BytesIO(file_content)
        relative_path, calculated_sha256 = storage_manager.save_file(
            file_stream,
            filename,
            file_id
        )

        # 3. Duplicate kontrolü: Eğer aynı sha256 ile dosya varsa, mevcut kaydı dönelim
        existing_file = db.query(FileRecord).filter(FileRecord.sha256 == calculated_sha256).first()
        if existing_file:
            doc_count = db.query(DocumentRecord).filter(DocumentRecord.file_id == existing_file.file_id).count()
            existing_file_id = str(getattr(existing_file, "file_id", ""))
            raw_score = getattr(existing_file, "quality_score", None)
            score_val = float(raw_score) if raw_score is not None else 0.8
            return {
                "success": True,
                "file_id": existing_file_id,
                "document_count": doc_count,
                "dataset_name": request.dataset_name,
                "avg_quality_score": round(score_val * 100.0, 1),
                "message": f"Bu veri kümesi daha önce aktarılmış ({existing_file_id}). {doc_count} doküman hazır."
            }

        # 4. Ortalama kalite skoru hesapla
        total_score = sum(float(s.get("metrics", {}).get("composite_score", 80.0)) for s in request.samples)
        avg_score = total_score / (sample_count * 100.0) if sample_count > 0 else 0.85

        # 5. FileRecord oluştur
        file_record = FileRecord(
            file_id=file_id,
            original_name=filename,
            relative_path=str(relative_path),
            mime_type="application/x-jsonlines",
            size_bytes=file_size,
            sha256=calculated_sha256,
            parser_name="synthetic_lab_ingest",
            parser_version="1.0.0",
            schema_version="1.0.0",
            security_level="INTERNAL",
            pii_detected=False,
            training_allowed=request.training_allowed,
            source=f"synthetic_lab:{request.paradigm or 'custom'}",
            language="tr",
            quality_score=min(1.0, max(0.0, avg_score))
        )
        db.add(file_record)

        # 6. DocumentRecord kayıtlarını oluştur
        document_records: List[DocumentRecord] = []
        for s in request.samples:
            doc_id = f"doc_synth_{uuid4().hex[:10]}"
            instr = str(s.get("instruction", "")).strip()
            resp = str(s.get("response", "")).strip()
            ctx = str(s.get("input_context", "")).strip()

            if ctx:
                doc_text = f"### Bağlam:\n{ctx}\n\n### Talimat:\n{instr}\n\n### Yanıt:\n{resp}"
            else:
                doc_text = f"### Talimat:\n{instr}\n\n### Yanıt:\n{resp}"

            sample_score = float(s.get("metrics", {}).get("composite_score", 80.0)) / 100.0

            doc_record = DocumentRecord(
                document_id=doc_id,
                file_id=file_id,
                title=(instr or "Sentetik Talimat Metni")[:120],
                text=doc_text,
                language="tr",
                char_count=len(doc_text),
                word_count=len(doc_text.split()),
                line_count=len(doc_text.splitlines()),
                parser_name="synthetic_lab_ingest",
                parser_version="1.0.0",
                schema_version="1.0.0",
                is_empty=False,
                is_duplicate=False,
                quality_score=min(1.0, max(0.0, sample_score))
            )
            document_records.append(doc_record)

        db.add_all(document_records)
        db.commit()
        db.refresh(file_record)

        logger.info(f"Synthetic dataset ingested: {file_id} with {len(document_records)} documents")

        rec_file_id = str(getattr(file_record, "file_id", file_id))
        return {
            "success": True,
            "file_id": rec_file_id,
            "document_count": len(document_records),
            "dataset_name": request.dataset_name,
            "avg_quality_score": round(avg_score * 100.0, 1),
            "message": f"{len(document_records)} sentetik doküman başarıyla derleyici havuzuna aktarıldı ({rec_file_id})."
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Sentetik veri aktarım hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Sentetik veri aktarım hatası: {str(e)}")

