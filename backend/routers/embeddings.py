"""
backend/routers/embeddings.py

Embedding Lab & Vector Projection API Endpoints
Local AI Research Lab - Developed by Kenan AY

Bu modül kelime ve token embedding vektörlerini inceleme,
PCA ile 2D/3D izdüşürme, Cosine Benzerliği ve Vektör Aritmetiği
(Analogy) işlemlerini sağlar:
- POST /api/v1/embeddings/project: PCA boyut indirgeme (2D veya 3D)
- POST /api/v1/embeddings/similarity: İki kelime arası Cosine Benzerliği ve açı hesabı
- POST /api/v1/embeddings/analogy: Vektör aritmetiği (A - B + C = ?)
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import math
import logging
import torch
import numpy as np

from src.server.inference_server import ModelManager
from src.model.gpt import GPTModel, GPTConfig
from src.rag.embedder import stable_text_seed

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/embeddings",
    tags=["embeddings"]
)

# Global ModelManager singleton import from inference router if available
try:
    from backend.routers.inference import manager
except ImportError:
    manager = ModelManager()


# ============================================================================
# Request / Response Schemas
# ============================================================================

class EmbeddingProjectRequest(BaseModel):
    words: List[str] = Field(..., min_length=2, max_length=120, description="İzdüşürülecek kelimeler listesi")
    dimensions: int = Field(2, ge=2, le=3, description="İzdüşüm boyutu (2 veya 3)")
    model_name: Optional[str] = Field(None, description="Model adı")
    normalize: bool = Field(True, description="Vektörleri L2 normalize et")


class ProjectedPoint(BaseModel):
    text: str
    x: float
    y: float
    z: Optional[float] = None
    norm: float


class EmbeddingProjectResponse(BaseModel):
    points: List[ProjectedPoint]
    dimensions: int
    explained_variance_ratio: List[float]
    d_model: int
    model_name: str


class SimilarityRequest(BaseModel):
    word_a: str = Field(..., min_length=1, description="İlk kelime")
    word_b: str = Field(..., min_length=1, description="İkinci kelime")
    model_name: Optional[str] = Field(None, description="Model adı")


class SimilarityResponse(BaseModel):
    word_a: str
    word_b: str
    cosine_similarity: float
    angle_degrees: float
    euclidean_distance: float
    dot_product: float
    model_name: str


class AnalogyCandidateResult(BaseModel):
    word: str
    similarity: float


class AnalogyRequest(BaseModel):
    word_a: str = Field(..., min_length=1, description="Örnek: kral")
    word_b: str = Field(..., min_length=1, description="Örnek: erkek")
    word_c: str = Field(..., min_length=1, description="Örnek: kadın")
    candidates: Optional[List[str]] = Field(None, description="Aday kelimeler")
    model_name: Optional[str] = Field(None, description="Model adı")


class AnalogyResponse(BaseModel):
    word_a: str
    word_b: str
    word_c: str
    formula: str
    top_matches: List[AnalogyCandidateResult]
    model_name: str


# ============================================================================
# Helpers
# ============================================================================

def _get_active_model_and_tokenizer(requested_model_name: Optional[str] = None):
    """Aktif modeli ve tokenizer'ı döndürür, yoksa registry'den veya fallback oluşturur."""
    model = manager.model
    tokenizer = manager.tokenizer
    model_name = manager.model_name or "active-model"

    if requested_model_name and manager.model_name != requested_model_name:
        try:
            manager.load_model(registry_dir="models", model_name=requested_model_name)
            model = manager.model
            tokenizer = manager.tokenizer
            model_name = requested_model_name
        except Exception as e:
            logger.warning(f"Could not load requested model '{requested_model_name}': {e}")

    if model is None:
        try:
            from src.registry.model_registry import ModelRegistry
            registry = ModelRegistry("models")
            avail = registry.list_models()
            if avail:
                manager.load_model(registry_dir="models", model_name=avail[0].model_name)
                model = manager.model
                tokenizer = manager.tokenizer
                model_name = manager.model_name or avail[0].model_name
        except Exception as e:
            logger.warning(f"Could not auto-load model from registry: {e}")

    if model is None:
        cfg = GPTConfig(
            vocab_size=300,
            max_seq_len=128,
            d_model=64,
            n_layers=4,
            n_heads=4,
            d_ff=128,
            dropout=0.0
        )
        model = GPTModel(cfg)
        model.eval()
        model_name = "demo-transformer-decoder"

    return model, tokenizer, model_name


def _get_word_vector(word: str, model: GPTModel, tokenizer: Any) -> torch.Tensor:
    """Bir kelime veya ifade için model embedding katmanından temsil vektörü çıkarır."""
    device = next(model.parameters()).device
    d_model = model.d_model
    vocab_size = getattr(model, "vocab_size", 300)

    # Tokenize
    token_ids = []
    if tokenizer is not None:
        try:
            token_ids = tokenizer.encode(word)
        except Exception:
            token_ids = []

    if not token_ids:
        token_ids = [ord(c) % vocab_size for c in word[:16]]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Extract token embedding table
    with torch.no_grad():
        embs = None
        if hasattr(model, "embeddings"):
            emb_layer = getattr(model, "embeddings")
            if hasattr(emb_layer, "token_embedding"):
                tok_emb = getattr(emb_layer, "token_embedding")
                if callable(tok_emb):
                    embs = tok_emb(input_ids)
                elif hasattr(tok_emb, "embedding") and callable(getattr(tok_emb, "embedding")):
                    embs = getattr(tok_emb, "embedding")(input_ids)
            elif callable(emb_layer):
                embs = emb_layer(input_ids)
        if not isinstance(embs, torch.Tensor):
            torch.manual_seed(stable_text_seed(word))
            return torch.randn(d_model, device=device)

    # Mean pool across sequence length
    vector: torch.Tensor = embs[0].mean(dim=0)
    return vector


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/project", response_model=EmbeddingProjectResponse)
async def project_embeddings(request: EmbeddingProjectRequest) -> EmbeddingProjectResponse:
    """
    Verilen kelime listesini model embedding uzayından alıp
    SVD tabanlı PCA ile 2D veya 3D koordinatlara izdüşürür.
    """
    clean_words = [w.strip() for w in request.words if w.strip()]
    if len(clean_words) < 2:
        raise HTTPException(status_code=422, detail="İzdüşüm için en az 2 kelime gereklidir.")

    model, tokenizer, model_name = _get_active_model_and_tokenizer(request.model_name)
    d_model = model.d_model

    # Extract vectors
    vectors = []
    raw_norms = []
    for word in clean_words:
        vec = _get_word_vector(word, model, tokenizer)
        norm = float(torch.norm(vec).item())
        raw_norms.append(round(norm, 4))
        if request.normalize and norm > 1e-8:
            vec = vec / norm
        vectors.append(vec)

    # Matrix shape: [N, d_model]
    X = torch.stack(vectors, dim=0).float() # [N, D]
    N = X.size(0)
    k = request.dimensions # 2 or 3

    # Centering
    mean = X.mean(dim=0, keepdim=True)
    X_centered = X - mean

    # SVD for exact PCA
    # X_centered = U @ diag(S) @ Vh
    # Principal directions are rows of Vh (or columns of V)
    try:
        _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        # S: [min(N, D)]
        # Vh: [min(N, D), D]
        
        # Explained variance
        variances = (S ** 2) / max(1, N - 1)
        total_variance = variances.sum().item()
        if total_variance > 1e-8:
            explained_ratios = [
                round(float((variances[i].item() / total_variance)), 4)
                for i in range(min(k, len(variances)))
            ]
        else:
            explained_ratios = [round(1.0 / k, 4)] * k

        # Projection: Z = X_centered @ Vh[:k].T
        # Shape: [N, k]
        components = Vh[:k].T # [D, k]
        Z = torch.matmul(X_centered, components) # [N, k]
    except Exception as e:
        logger.warning(f"SVD calculation fallback: {e}")
        Z = X_centered[:, :k]
        explained_ratios = [0.5, 0.5] if k == 2 else [0.4, 0.3, 0.3]

    coords = Z.cpu().numpy()

    # Scale coordinates to [-100, 100] for standard visualization
    max_abs = float(np.max(np.abs(coords))) if np.max(np.abs(coords)) > 0 else 1.0
    scaled_coords = (coords / max_abs) * 85.0

    points: List[ProjectedPoint] = []
    for i, word in enumerate(clean_words):
        x = round(float(scaled_coords[i, 0]), 2)
        y = round(float(scaled_coords[i, 1]), 2)
        z = round(float(scaled_coords[i, 2]), 2) if k >= 3 else None

        points.append(
            ProjectedPoint(
                text=word,
                x=x,
                y=y,
                z=z,
                norm=raw_norms[i]
            )
        )

    return EmbeddingProjectResponse(
        points=points,
        dimensions=k,
        explained_variance_ratio=explained_ratios,
        d_model=d_model,
        model_name=model_name
    )


@router.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest) -> SimilarityResponse:
    """İki kelime arasındaki Cosine Benzerliği, Euclidean mesafe ve açıyı hesaplar."""
    model, tokenizer, model_name = _get_active_model_and_tokenizer(request.model_name)

    vec_a = _get_word_vector(request.word_a.strip(), model, tokenizer)
    vec_b = _get_word_vector(request.word_b.strip(), model, tokenizer)

    dot = float(torch.dot(vec_a, vec_b).item())
    norm_a = float(torch.norm(vec_a).item())
    norm_b = float(torch.norm(vec_b).item())

    if request.word_a.strip() == request.word_b.strip():
        cosine_sim = 1.0
        angle_deg = 0.0
        euclidean_dist = 0.0
        dot = norm_a * norm_b
    elif norm_a > 1e-8 and norm_b > 1e-8:
        cosine_sim = dot / (norm_a * norm_b)
        cosine_sim = max(-1.0, min(1.0, cosine_sim))
        if abs(cosine_sim - 1.0) < 1e-5:
            cosine_sim = 1.0
            angle_deg = 0.0
        else:
            angle_rad = math.acos(cosine_sim)
            angle_deg = angle_rad * (180.0 / math.pi)
        euclidean_dist = float(torch.norm(vec_a - vec_b).item())
    else:
        cosine_sim = 0.0
        angle_deg = 90.0
        euclidean_dist = float(torch.norm(vec_a - vec_b).item())

    return SimilarityResponse(
        word_a=request.word_a,
        word_b=request.word_b,
        cosine_similarity=round(cosine_sim, 4),
        angle_degrees=round(angle_deg, 2),
        euclidean_distance=round(euclidean_dist, 4),
        dot_product=round(dot, 4),
        model_name=model_name
    )


@router.post("/analogy", response_model=AnalogyResponse)
async def compute_analogy(request: AnalogyRequest) -> AnalogyResponse:
    """
    Vektör aritmetiği benzetimi: word_a - word_b + word_c = ?
    Örn: kral - erkek + kadın = kraliçe
    """
    model, tokenizer, model_name = _get_active_model_and_tokenizer(request.model_name)

    vec_a = _get_word_vector(request.word_a.strip(), model, tokenizer)
    vec_b = _get_word_vector(request.word_b.strip(), model, tokenizer)
    vec_c = _get_word_vector(request.word_c.strip(), model, tokenizer)

    # Arithmetic
    target_vec = vec_a - vec_b + vec_c
    target_norm = float(torch.norm(target_vec).item())
    if target_norm > 1e-8:
        target_normed = target_vec / target_norm
    else:
        target_normed = target_vec

    # Candidates
    default_candidates = [
        "kraliçe", "prenses", "kadın", "saray", "yönetici", "lider",
        "taht", "soylu", "imparatoriçe", "anne", "kız", "insan"
    ]
    candidate_list = request.candidates if request.candidates else default_candidates
    # Ensure inputs themselves don't bias top results
    filter_set = {request.word_a.lower(), request.word_b.lower(), request.word_c.lower()}

    results = []
    for cand in candidate_list:
        cand_clean = cand.strip()
        if not cand_clean or cand_clean.lower() in filter_set:
            continue
        c_vec = _get_word_vector(cand_clean, model, tokenizer)
        c_norm = float(torch.norm(c_vec).item())
        if c_norm > 1e-8:
            sim = float(torch.dot(target_normed, c_vec / c_norm).item())
        else:
            sim = 0.0
        results.append(AnalogyCandidateResult(word=cand_clean, similarity=round(sim, 4)))

    results.sort(key=lambda x: x.similarity, reverse=True)

    formula = f"{request.word_a} - {request.word_b} + {request.word_c} = ?"

    return AnalogyResponse(
        word_a=request.word_a,
        word_b=request.word_b,
        word_c=request.word_c,
        formula=formula,
        top_matches=results[:6],
        model_name=model_name
    )
