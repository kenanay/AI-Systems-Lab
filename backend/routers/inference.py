"""
backend/routers/inference.py

Model Inference API Endpoints

Bu modül eğitilmiş modeller ile metin üretimi (Text Generation) ve
çıkarım (Inference) için REST ve Streaming API sağlar:
- Model yükleme/boşaltma
- Durum kontrolü (Model yüklü mü?)
- Metin tamamlama / üretim (Greedy, Top-k, Top-p, Temperature)
- Streaming metin üretimi
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any, List, Generator
from pydantic import BaseModel, Field
import logging
import json
import time
import torch
import torch.nn.functional as F

from src.server.inference_server import ModelManager
from src.inference.streaming_generation import stream_generate
from src.inference.beam_search import beam_search, BeamHypothesis
from src.model.gpt import GPTModel, GPTConfig

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/inference",
    tags=["inference"]
)

# Global ModelManager singleton
manager = ModelManager()


def _ensure_model_loaded():
    """Yüklü modeli, yoksa registry'deki en güncel modeli, o da yoksa demo modelini döndürür."""
    if manager.model is not None:
        return manager.model, manager.tokenizer, manager.model_name or "local-model"
    try:
        from src.registry.model_registry import ModelRegistry
        registry = ModelRegistry("models")
        avail = registry.list_models()
        if avail:
            manager.load_model(registry_dir="models", model_name=avail[0].model_name)
            if manager.model is not None:
                return manager.model, manager.tokenizer, manager.model_name or avail[0].model_name
    except Exception as e:
        logger.warning(f"Could not auto-load model from registry: {e}")

    # Fallback demo model
    cfg = GPTConfig(
        vocab_size=300,
        max_seq_len=128,
        d_model=64,
        n_layers=4,
        n_heads=4,
        d_ff=128,
        dropout=0.0
    )
    demo_model = GPTModel(cfg)
    demo_model.eval()
    return demo_model, manager.tokenizer, "demo-transformer-decoder"


def _encode_text(tokenizer, text: str, vocab_size: int = 300) -> List[int]:
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text)
            if ids:
                return ids
        except Exception:
            pass
    return [ord(c) % vocab_size for c in text]


def _decode_tokens(tokenizer, token_ids: List[int]) -> str:
    if tokenizer is not None:
        try:
            decoded = tokenizer.decode(token_ids)
            if decoded:
                return decoded
        except Exception:
            pass
    return "".join(chr(tid) if 32 <= tid < 127 else f"[{tid}]" for tid in token_ids)


class LoadModelRequest(BaseModel):
    model_name: str = Field(..., description="Yüklenecek model adı")
    version: Optional[str] = Field(None, description="Versiyon (varsayılan: en son)")
    registry_dir: str = Field("models", description="Registry dizini")


class InferenceGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Giriş metni / prompt")
    max_length: int = Field(50, ge=1, le=512, description="Üretilecek maksimum token sayısı")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Örnekleme sıcaklığı")
    top_k: Optional[int] = Field(50, ge=1, description="Top-k örnekleme")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Nucleus (top-p) örnekleme")


class InferenceGenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    tokens_generated: int
    generation_time_ms: float
    model_name: str


class BeamSearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Başlangıç metni / prompt")
    beam_width: int = Field(3, ge=1, le=10, description="Işın genişliği (k)")
    max_length: int = Field(40, ge=1, le=256, description="Üretilecek maksimum token sayısı")
    length_penalty: float = Field(1.0, ge=0.1, le=3.0, description="Uzunluk cezalandırma/ödüllendirme katsayısı")
    num_return_sequences: int = Field(3, ge=1, le=10, description="Döndürülecek hipotez sayısı")


class BeamHypothesisResponse(BaseModel):
    hypothesis_id: int
    text: str
    score: float
    normalized_score: float
    length: int
    tokens: List[str]


class BeamSearchResponse(BaseModel):
    prompt: str
    beam_width: int
    hypotheses: List[BeamHypothesisResponse]
    model_name: str
    execution_time_ms: float


class NextTokenProbsRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Analiz edilecek metin")
    top_k: int = Field(5, ge=1, le=20, description="Gösterilecek aday sayısı")
    temperature: float = Field(1.0, ge=0.1, le=3.0, description="Sıcaklık katsayısı")


class NextTokenCandidate(BaseModel):
    rank: int
    token_id: int
    token_text: str
    probability: float
    percentage: float
    raw_logit: float


class NextTokenProbsResponse(BaseModel):
    prompt: str
    temperature: float
    candidates: List[NextTokenCandidate]
    model_name: str

@router.post("/load")
def load_model(request: LoadModelRequest) -> Dict[str, Any]:
    """
    Belirtilen modeli belleğe yükler.
    """
    try:
        manager.load_model(
            registry_dir=request.registry_dir,
            model_name=request.model_name,
            version=request.version
        )
        return {
            "status": "success",
            "message": f"Model {request.model_name} başarıyla yüklendi",
            "model_name": manager.model_name
        }
    except Exception as e:
        logger.error(f"Failed to load model {request.model_name}: {e}")
        raise HTTPException(status_code=400, detail=f"Model yüklenemedi: {str(e)}")


@router.get("/status")
def get_inference_status() -> Dict[str, Any]:
    """
    Çıkarım motorunun ve aktif modelin durumunu döndürür.
    """
    is_loaded = manager.model is not None
    return {
        "ready": is_loaded,
        "model_loaded": is_loaded,
        "model_name": manager.model_name,
        "device": str(manager.device)
    }


@router.post("/generate", response_model=InferenceGenerateResponse)
def generate_text(request: InferenceGenerateRequest) -> InferenceGenerateResponse:
    """
    Yüklü model ile metin üretir.
    """
    if manager.model is None:
        raise HTTPException(
            status_code=400,
            detail="Henüz hiçbir model belleğe yüklenmedi. Önce /load ile bir model yükleyin."
        )

    try:
        generated_text, tokens_generated, time_ms = manager.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )

        return InferenceGenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            tokens_generated=tokens_generated,
            generation_time_ms=time_ms,
            model_name=manager.model_name or "unknown"
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Metin üretimi hatası: {str(e)}")


@router.post("/generate/stream")
def generate_stream(request: InferenceGenerateRequest) -> StreamingResponse:
    """
    Metin üretimini SSE (Server-Sent Events) akışı olarak iletir.
    """
    model, tokenizer, model_name = _ensure_model_loaded()

    def event_generator() -> Generator[str, None, None]:
        try:
            vocab_size = getattr(getattr(model, "config", None), "vocab_size", 300)
            input_ids_list = _encode_text(tokenizer, request.prompt, vocab_size)
            if not input_ids_list:
                input_ids_list = [1]
            
            device = next(model.parameters()).device
            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
            
            # Stream generate tokens
            generated_tokens = 0
            for token_id, full_sequence in stream_generate(
                model=model,
                input_ids=input_ids,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k if request.top_k and request.top_k > 0 else None,
                top_p=request.top_p if request.top_p and request.top_p < 1.0 else None,
                eos_token_id=None,
                pad_token_id=None
            ):
                generated_tokens += 1
                token_text = _decode_tokens(tokenizer, [token_id])
                
                # Yield token as SSE
                yield f"data: {json.dumps({'token': token_text, 'done': False})}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'done': True, 'tokens_generated': generated_tokens, 'model_name': model_name})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming generation error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/beam-search", response_model=BeamSearchResponse)
def run_beam_search(request: BeamSearchRequest) -> BeamSearchResponse:
    """
    Belirtilen başlangıç metni için çoklu ışın araması (Beam Search) gerçekleştirir.
    Paralel K adet hipotez arasından en yüksek kümülatif log-olasılığa sahip çıktıları üretir.
    """
    start_time = time.time()
    model, tokenizer, model_name = _ensure_model_loaded()
    model.eval()
    device = next(model.parameters()).device

    vocab_size = getattr(getattr(model, "config", None), "vocab_size", 300)
    input_ids = _encode_text(tokenizer, request.prompt, vocab_size)
    if not input_ids:
        input_ids = [1]

    # Truncate if exceeding max_seq_len
    max_seq_len = getattr(model, "max_seq_len", 128)
    if len(input_ids) > max_seq_len - 10:
        input_ids = input_ids[-(max_seq_len - 10):]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    try:
        hyps = beam_search(
            model=model,
            input_ids=input_tensor,
            beam_width=request.beam_width,
            max_new_tokens=request.max_length,
            length_penalty=request.length_penalty,
            num_return_sequences=min(request.num_return_sequences, request.beam_width),
            early_stopping=True
        )

        responses: List[BeamHypothesisResponse] = []
        for i, hyp in enumerate(hyps):
            gen_ids = hyp.token_ids[len(input_ids):]
            if not gen_ids:
                gen_ids = hyp.token_ids
            decoded_text = _decode_tokens(tokenizer, gen_ids)
            token_list = [_decode_tokens(tokenizer, [t]) for t in gen_ids]
            
            seq_len = max(1, len(gen_ids))
            norm_score = hyp.score / (seq_len ** request.length_penalty)

            responses.append(
                BeamHypothesisResponse(
                    hypothesis_id=i + 1,
                    text=decoded_text,
                    score=round(float(hyp.score), 3),
                    normalized_score=round(float(norm_score), 3),
                    length=len(gen_ids),
                    tokens=token_list
                )
            )

        elapsed_ms = (time.time() - start_time) * 1000.0

        return BeamSearchResponse(
            prompt=request.prompt,
            beam_width=request.beam_width,
            hypotheses=responses,
            model_name=model_name,
            execution_time_ms=round(elapsed_ms, 2)
        )
    except Exception as e:
        logger.error(f"Beam search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Işın araması hatası: {str(e)}")


@router.post("/next-token-probs", response_model=NextTokenProbsResponse)
def get_next_token_probabilities(request: NextTokenProbsRequest) -> NextTokenProbsResponse:
    """
    Verilen metnin son pozisyonundaki model çıktı logitlerini hesaplar ve
    en olası Top-K aday tokenın softmax olasılık dağılımını döndürür.
    """
    model, tokenizer, model_name = _ensure_model_loaded()
    model.eval()
    device = next(model.parameters()).device

    vocab_size = getattr(getattr(model, "config", None), "vocab_size", 300)
    input_ids = _encode_text(tokenizer, request.prompt, vocab_size)
    if not input_ids:
        input_ids = [1]

    max_seq_len = getattr(model, "max_seq_len", 128)
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[-max_seq_len:]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(input_tensor)

    # Son pozisyondaki logitler: shape [vocab_size]
    last_logits = logits[0, -1, :]

    temp = max(0.01, request.temperature)
    scaled_logits = last_logits / temp
    probs = F.softmax(scaled_logits, dim=-1)

    top_k = min(request.top_k, probs.size(-1))
    topk_probs, topk_indices = torch.topk(probs, k=top_k)

    candidates: List[NextTokenCandidate] = []
    for rank in range(top_k):
        tid = int(topk_indices[rank].item())
        p_val = float(topk_probs[rank].item())
        token_str = _decode_tokens(tokenizer, [tid])
        raw_l = float(last_logits[tid].item())

        candidates.append(
            NextTokenCandidate(
                rank=rank + 1,
                token_id=tid,
                token_text=token_str if token_str.strip() else f"[{tid}]",
                probability=round(p_val, 4),
                percentage=round(p_val * 100.0, 2),
                raw_logit=round(raw_l, 3)
            )
        )

    return NextTokenProbsResponse(
        prompt=request.prompt,
        temperature=request.temperature,
        candidates=candidates,
        model_name=model_name
    )

    

class AttentionInspectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="Analiz edilecek metin")
    layer_idx: Optional[int] = Field(None, ge=0, description="Katman indeksi (None ise tüm katmanların ortalaması)")
    head_idx: Optional[int] = Field(None, ge=0, description="Head indeksi (None ise tüm head'lerin ortalaması)")
    model_name: Optional[str] = Field(None, description="Model adı (None ise yüklü model veya varsayılan)")


class AttentionInspectResponse(BaseModel):
    tokens: List[str]
    token_ids: List[int]
    num_layers: int
    num_heads: int
    selected_layer: Optional[int]
    selected_head: Optional[int]
    matrix: List[List[float]]
    all_heads_matrix: Optional[Dict[str, List[List[float]]]] = None
    model_name: str


@router.post("/attention", response_model=AttentionInspectResponse)
def inspect_attention(request: AttentionInspectRequest) -> AttentionInspectResponse:
    """
    Girilen metin için modelin Self-Attention ağırlıklarını hesaplar ve döndürür.
    """
    model = manager.model
    tokenizer = manager.tokenizer
    model_name = manager.model_name or "local-gpt-research"

    # Eğer model yüklü değilse, registry'den en son modeli dene veya fallback research modeli oluştur
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

    # Fallback model (her zaman çalışabilen hazır eğitsel GPT modeli)
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

    # Tokenize input
    if tokenizer is not None:
        try:
            token_ids = tokenizer.encode(request.text)
            if not token_ids:
                token_ids = [ord(c) % 300 for c in request.text]
                tokens = [c for c in request.text]
            else:
                tokens = []
                for tid in token_ids:
                    try:
                        decoded = tokenizer.decode([tid])
                        tokens.append(decoded if decoded.strip() else f"[{tid}]")
                    except Exception:
                        tokens.append(f"[{tid}]")
        except Exception:
            token_ids = [ord(c) % 300 for c in request.text]
            tokens = [c for c in request.text]
    else:
        token_ids = [ord(c) % 300 for c in request.text[:64]]
        tokens = [c for c in request.text[:64]]

    # Truncate to max_seq_len
    max_len = getattr(model, "max_seq_len", 128)
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
        tokens = tokens[:max_len]

    # Forward pass in eval mode (disables attention dropout)
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        _, attention_weights = model(input_ids, need_weights=True)

    # attention_weights: List of [1, n_heads, seq_len, seq_len] for each layer
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].size(1)

    # Layer selection
    chosen_layer = request.layer_idx
    if chosen_layer is not None and chosen_layer >= num_layers:
        chosen_layer = num_layers - 1

    if chosen_layer is not None:
        # Specific layer: shape [num_heads, seq_len, seq_len]
        layer_attn = attention_weights[chosen_layer][0]
    else:
        # Mean across all layers: shape [num_heads, seq_len, seq_len]
        layer_attn = torch.stack([w[0] for w in attention_weights]).mean(dim=0)

    # Head selection
    chosen_head = request.head_idx
    if chosen_head is not None and chosen_head >= num_heads:
        chosen_head = num_heads - 1

    if chosen_head is not None:
        # Specific head: shape [seq_len, seq_len]
        matrix_tensor = layer_attn[chosen_head]
    else:
        # Average across all heads
        matrix_tensor = layer_attn.mean(dim=0)

    matrix = [[round(float(val), 4) for val in row] for row in matrix_tensor.cpu().numpy()]

    all_heads_matrix: Dict[str, List[List[float]]] = {}
    for h in range(num_heads):
        all_heads_matrix[f"Head {h}"] = [
            [round(float(val), 4) for val in row] for row in layer_attn[h].cpu().numpy()
        ]

    return AttentionInspectResponse(
        tokens=tokens,
        token_ids=token_ids,
        num_layers=num_layers,
        num_heads=num_heads,
        selected_layer=chosen_layer,
        selected_head=chosen_head,
        matrix=matrix,
        all_heads_matrix=all_heads_matrix,
        model_name=model_name
    )

