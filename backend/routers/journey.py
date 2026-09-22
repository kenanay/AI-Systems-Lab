"""
backend/routers/journey.py

Guided Learning Journey, Knowledge Map & AI Glossary API Router
Local-First AI Research Lab - Section 56 Architecture
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from src.learning.journey_engine import JourneyEngine

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/journey",
    tags=["journey"]
)

_engine = JourneyEngine()


# Request / Response Schemas

class CheckQuestionRequest(BaseModel):
    """Aşama kontrol sorusu cevap request."""
    question_id: str = Field(..., description="Soru kimliği (örn. q_stage_0_1)")
    selected_option: int = Field(..., ge=0, description="Kullanıcının seçtiği opsiyon indeksi (0-3)")


class CheckQuestionResponse(BaseModel):
    """Soru kontrol sonucu ve pedagojik geri bildirim."""
    question_id: str
    is_correct: bool
    selected_option: int
    correct_index: int
    explanation: str
    math_intuition: Optional[str] = None


# Endpoints

@router.get("/curriculum")
async def get_curriculum() -> List[Dict[str, Any]]:
    """
    Tüm 13 aşamalı rehberli öğrenme yolculuğunu, hedeflerini,
    laboratuvar bağlantılarını ve kontrol sorularını döner.
    """
    logger.info("Fetching learning curriculum")
    return _engine.get_curriculum()


@router.get("/curriculum/{stage_id}")
async def get_stage(stage_id: str) -> Dict[str, Any]:
    """Belirli bir aşamanın ayrıntılarını döner."""
    stage = _engine.get_stage(stage_id)
    if not stage:
        raise HTTPException(status_code=404, detail=f"Stage not found: {stage_id}")
    return stage


@router.get("/graph")
async def get_knowledge_graph() -> Dict[str, Any]:
    """
    Ön bilgi grafı (Knowledge DAG) düğüm ve kenarlarını döner.
    Hangi kavramın hangi kavrama ön koşul olduğunu gösterir.
    """
    logger.info("Fetching knowledge graph DAG")
    return _engine.get_knowledge_graph()


@router.get("/glossary")
async def get_glossary(
    search: Optional[str] = Query(None, description="Terim veya açıklamada arama"),
    category: Optional[str] = Query(None, description="Kategori filtresi")
) -> List[Dict[str, Any]]:
    """
    AI & LLM Terimler Sözlüğünü döner. Arama ve kategori filtrelerini destekler.
    """
    return _engine.get_glossary(search=search, category=category)


@router.post("/check-question", response_model=CheckQuestionResponse)
async def check_question(request: CheckQuestionRequest) -> CheckQuestionResponse:
    """
    Aşama sonu mini kontrol sorusunu değerlendirir ve pedagojik açıklama sunar.
    """
    logger.info(f"Checking answer for question: {request.question_id}")
    result = _engine.check_question(request.question_id, request.selected_option)
    return CheckQuestionResponse(**result)
