"""
backend/routers/rag.py

RAG (Retrieval-Augmented Generation) API Router
Local AI Research Lab - Developed by Kenan AY

Bu modül RAG altyapısı için kapsamlı REST API endpoint'leri sunar:
- POST /api/v1/rag/chunk: Canlı metin parçalama (Chunking görselleştirme ve test)
- POST /api/v1/rag/index: Özel metinleri veya DB'deki dokümanları vektör dizinine ekleme
- GET  /api/v1/rag/collections: Vektör koleksiyonları ve istatistikleri
- POST /api/v1/rag/search: Yoğun (Vektör), Seyrek (BM25) veya Hibrit (RRF) arama
- POST /api/v1/rag/query: Uçtan uca kaynak referanslı soru-cevap (RAG Generation + Citations)
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import logging

from backend.database import get_db
from backend.models import DocumentRecord
from src.rag.chunking import get_chunker, TextChunk
from src.rag.vector_store import VectorStore, SearchResult
from src.rag.retriever import BM25Index, HybridRetriever
from src.rag.pipeline import RAGPipeline, RAGResponse, Citation
from src.rag.embedder import get_embedder

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/rag",
    tags=["rag"]
)

# Global in-memory / persistent collection registries
_COLLECTIONS: Dict[str, RAGPipeline] = {}


def _get_or_create_pipeline(collection_name: str = "default") -> RAGPipeline:
    """Belirtilen koleksiyon için RAGPipeline örneğini getirir veya oluşturur."""
    if collection_name not in _COLLECTIONS:
        # Yeni embedder ile initialize et
        embedder = get_embedder(
            embedder_type="local",
            model_name="all-MiniLM-L6-v2",
            use_cache=True,
            cache_dir=f".embeddings_cache/{collection_name}"
        )
        
        v_store = VectorStore(
            collection_name=collection_name,
            d_model=embedder.d_model,
            use_faiss_if_available=True
        )
        b_index = BM25Index()
        pipeline = RAGPipeline(
            vector_store=v_store,
            bm25_index=b_index,
            embedder=embedder,
            d_model=embedder.d_model
        )
        
        # Eğer default koleksiyon boşsa, kullanıcıya hazır zengin bir örnek külliyat ilklendir
        if collection_name == "default" and v_store.count == 0:
            _seed_default_knowledge_base(pipeline)
            
        _COLLECTIONS[collection_name] = pipeline

    return _COLLECTIONS[collection_name]


def _seed_default_knowledge_base(pipeline: RAGPipeline) -> None:
    """RAG Lab'ın ilk açılışında arama ve soru-cevap yapabilmesi için örnek külliyat yükler."""
    sample_docs = [
        {
            "title": "Transformer Mimarisi ve Self-Attention",
            "text": (
                "Transformer mimarisi, 2017 yılında 'Attention Is All You Need' makalesiyle duyurulmuştur. "
                "Geleneksel RNN ve LSTM ağlarının aksine sıralı veri bağımlılığını ortadan kaldırarak paralel hesaplamaya izin verir. "
                "Mimarinin kalbinde Query (Q), Key (K) ve Value (V) matrislerinin çarpımıyla hesaplanan Scaled Dot-Product Attention mekanizması yer alır. "
                "Attention formülü: softmax((Q * K^T) / sqrt(d_k)) * V şeklindedir."
            )
        },
        {
            "title": "KV Cache Optimizasyonu",
            "text": (
                "Büyük dil modellerinde metin üretiminde her yeni token üretildiğinde geçmişteki tüm token'ların Query, Key ve Value değerleri yeniden hesaplanabilir. "
                "Ancak autoregressive üretimde geçmiş token'ların Key ve Value temsilleri değişmez. "
                "KV Cache mekanizması, bu K ve V tensörlerini bellekte saklayarak sonraki adımlarda fazladan matris çarpımı yapılmasını engeller. "
                "Bu optimizasyon çıkarım süresini O(N^2)'den O(N) karmaşıklığına indirir ve çıkarım hızını katlar."
            )
        },
        {
            "title": "LoRA (Low-Rank Adaptation) İnce Ayarı",
            "text": (
                "LoRA, büyük dil modellerinin parametrelerini dondurup ağırlık matrislerine düşük ranklı iki ayrışık matris (A ve B) ekleyen parametre verimli bir ince ayar tekniğidir. "
                "Model ağırlık değişimi W + delta_W olarak modellenir ve delta_W = B * A şeklinde çarpanlarına ayrılır. "
                "Bu sayede yüz milyonlarca parametre yerine modelin yalnızca %0.1 ila %1'i kadar ek parametre eğitilir, GPU VRAM gereksinimi dramatik şekilde düşer."
            )
        },
        {
            "title": "Retrieval-Augmented Generation (RAG) İlkeleri",
            "text": (
                "RAG sistemi, dil modellerinin eğitim verisinde bulunmayan güncel veya özel şirket verilerini dinamik olarak sorgulayıp yanıt üretmesini sağlar. "
                "Sistem üç ana aşamadan oluşur: Parçalama (Chunking), İndeksleme (Dense/Sparse Retrieval) ve Zenginleştirilmiş Üretim (Prompt Grounding). "
                "RAG sayesinde dil modellerinin halüsinasyon görme olasılığı azalır ve üretilen yanıtların hangi kaynaktan alındığı alıntılarla (citations) şeffaf şekilde doğrulanabilir."
            )
        },
        {
            "title": "BPE (Byte-Pair Encoding) Tokenizasyon",
            "text": (
                "Byte-Pair Encoding (BPE), metinleri karakter veya kelime düzeyinde değil, sık tekrarlanan alt kelime (subword) parçalarına bölen bir sıkıştırma ve tokenizasyon algoritmasıdır. "
                "Eğitim sırasında en sık yan yana gelen bayt veya karakter çiftleri birleştirilerek kelime dağarcığına (vocabulary) eklenir. "
                "Türkçe gibi sondan eklemeli zengin dillerde kök ve ekleri başarıyla ayrıştırarak OOV (Out-of-Vocabulary) sorununu kökten çözer."
            )
        }
    ]

    chunker = get_chunker("recursive", chunk_size=400, chunk_overlap=80)
    all_chunks: List[TextChunk] = []
    
    for doc_idx, doc in enumerate(sample_docs):
        chunks = chunker.chunk_text(
            text=doc["text"],
            metadata={"title": doc["title"], "source": "sample_seed"},
            document_id=f"seed_{doc_idx + 1}"
        )
        all_chunks.extend(chunks)

    pipeline.index_chunks(all_chunks)
    logger.info(f"RAG varsayılan külliyatı {len(all_chunks)} parça ile ilklendirildi.")


# ============================================================================
# Request / Response Schemas
# ============================================================================

class ChunkRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Parçalanacak ham metin")
    strategy: str = Field("recursive", description="Bölme stratejisi: 'recursive', 'sentence', 'fixed'")
    chunk_size: int = Field(400, ge=50, le=4000, description="Hedef parça boyutu (karakter)")
    chunk_overlap: int = Field(80, ge=0, le=1000, description="Parçalar arası örtüşme (karakter)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Ek metaveri")


class ChunkItemResponse(BaseModel):
    chunk_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict[str, Any]


class ChunkResponse(BaseModel):
    chunks: List[ChunkItemResponse]
    total_chunks: int
    strategy: str
    total_characters: int


class IndexDocumentRequest(BaseModel):
    collection_name: str = Field("default", description="Hedef koleksiyon adı")
    document_ids: Optional[List[str]] = Field(None, description="DB'deki DocumentRecord id listesi")
    custom_texts: Optional[List[Dict[str, str]]] = Field(None, description="Doğrudan eklenecek metinler listesi [{'title': '...', 'text': '...'}]")
    chunk_strategy: str = Field("recursive", description="Bölme stratejisi")
    chunk_size: int = Field(400, ge=50, le=2000)
    chunk_overlap: int = Field(80, ge=0, le=500)


class IndexDocumentResponse(BaseModel):
    collection_name: str
    indexed_chunks: int
    total_vectors_in_collection: int
    message: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Arama sorgusu")
    collection_name: str = Field("default", description="Aranacak koleksiyon adı")
    mode: str = Field("hybrid", description="Arama modu: 'hybrid', 'dense', 'sparse'")
    top_k: int = Field(5, ge=1, le=20, description="Döndürülecek sonuç sayısı")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Hibrit aramada Dense ağırlığı (0.0 = BM25, 1.0 = Vektör)")


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    mode: str
    collection_name: str
    results: List[SearchResultItem]
    count: int
    latency_ms: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Kullanıcı sorusu")
    collection_name: str = Field("default", description="Koleksiyon adı")
    retrieval_mode: str = Field("hybrid", description="'hybrid', 'dense', 'sparse'")
    top_k: int = Field(3, ge=1, le=10, description="Bağlama eklenecek parça sayısı")
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    max_new_tokens: int = Field(150, ge=20, le=500)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class CitationItem(BaseModel):
    citation_id: int
    chunk_id: str
    source_title: str
    snippet: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[SearchResultItem]
    citations: List[CitationItem]
    model_name: str
    prompt_used: str
    stats: Dict[str, Any]


class CollectionInfo(BaseModel):
    collection_name: str
    vector_count: int
    d_model: int
    backend: str


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/chunk", response_model=ChunkResponse)
async def chunk_text_api(request: ChunkRequest) -> ChunkResponse:
    """
    Canlı metin parçalama endpoint'i.
    Kullanıcının girdiği metni seçilen stratejiyle anında bölerek görselleştirir.
    """
    try:
        chunker = get_chunker(
            strategy=request.strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        chunks = chunker.chunk_text(
            text=request.text,
            metadata=request.metadata or {},
            document_id="preview"
        )
        
        items = [
            ChunkItemResponse(
                chunk_id=c.chunk_id,
                text=c.text,
                chunk_index=c.chunk_index,
                start_char=c.start_char,
                end_char=c.end_char,
                token_count=c.token_count,
                metadata=c.metadata
            )
            for c in chunks
        ]
        
        return ChunkResponse(
            chunks=items,
            total_chunks=len(items),
            strategy=request.strategy,
            total_characters=len(request.text)
        )
    except Exception as e:
        logger.error(f"Chunking hatası: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/index", response_model=IndexDocumentResponse)
async def index_documents_api(
    request: IndexDocumentRequest,
    db: Session = Depends(get_db)
) -> IndexDocumentResponse:
    """
    Özel metinleri veya veritabanındaki kayıtlı dokümanları vektör koleksiyonuna indeksler.
    """
    pipeline = _get_or_create_pipeline(request.collection_name)
    chunker = get_chunker(
        strategy=request.chunk_strategy,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )

    all_chunks: List[TextChunk] = []

    # 1. Custom texts
    if request.custom_texts:
        for idx, item in enumerate(request.custom_texts):
            txt = item.get("text", "")
            title = item.get("title", f"Metin {idx + 1}")
            if txt.strip():
                chunks = chunker.chunk_text(
                    text=txt,
                    metadata={"title": title, "source": "custom_upload"},
                    document_id=f"cust_{idx + 1}"
                )
                all_chunks.extend(chunks)

    # 2. Database DocumentRecord'ları
    if request.document_ids:
        docs = db.query(DocumentRecord).filter(DocumentRecord.document_id.in_(request.document_ids)).all()
        for doc in docs:
            d: Any = doc
            txt = str(d.text or "")
            if txt.strip():
                doc_id_str = str(d.document_id)
                title = str(d.title or doc_id_str)
                chunks = chunker.chunk_text(
                    text=txt,
                    metadata={"title": title, "source": "database", "document_id": doc_id_str},
                    document_id=doc_id_str
                )
                all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="İndekslenecek geçerli metin veya doküman bulunamadı."
        )

    added_count = pipeline.index_chunks(all_chunks)

    return IndexDocumentResponse(
        collection_name=request.collection_name,
        indexed_chunks=added_count,
        total_vectors_in_collection=pipeline.vector_store.count,
        message=f"{added_count} parça '{request.collection_name}' koleksiyonuna başarıyla eklendi."
    )


@router.get("/collections", response_model=List[CollectionInfo])
async def list_collections_api() -> List[CollectionInfo]:
    """Tüm aktif RAG koleksiyonlarını ve vektör sayılarını listeler."""
    # En azından default'u ilklendir
    _get_or_create_pipeline("default")
    
    result: List[CollectionInfo] = []
    for name, pipe in _COLLECTIONS.items():
        result.append(
            CollectionInfo(
                collection_name=name,
                vector_count=pipe.vector_store.count,
                d_model=pipe.d_model,
                backend="faiss" if pipe.vector_store.use_faiss else "pytorch_native"
            )
        )
    return result


@router.post("/search", response_model=SearchResponse)
async def search_api(request: SearchRequest) -> SearchResponse:
    """
    Yoğun (Dense), Seyrek (BM25) veya Hibrit (RRF) benzerlik araması çalıştırır.
    """
    t0 = time.time()
    pipeline = _get_or_create_pipeline(request.collection_name)

    query_vec = pipeline.embed_text(request.query)
    results = pipeline.hybrid_retriever.search(
        query=request.query,
        query_vector=query_vec,
        mode=request.mode,
        top_k=request.top_k,
        alpha=request.alpha
    )

    latency_ms = round((time.time() - t0) * 1000, 2)

    items = [
        SearchResultItem(
            chunk_id=r.chunk_id,
            text=r.text,
            score=round(float(r.score), 4),
            rank=r.rank,
            metadata=r.metadata
        )
        for r in results
    ]

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        collection_name=request.collection_name,
        results=items,
        count=len(items),
        latency_ms=latency_ms
    )


@router.post("/query", response_model=QueryResponse)
async def query_rag_api(request: QueryRequest) -> QueryResponse:
    """
    Uçtan uca RAG sorgusu:
    1. İlgili bağlam parçalarını getirir (Hybrid Retrieval).
    2. Grounded prompt oluşturur.
    3. Model yanıtını üretir.
    4. Kaynak atıflarını (citations) çıkarır.
    """
    pipeline = _get_or_create_pipeline(request.collection_name)
    rag_resp = pipeline.query(
        question=request.question,
        top_k=request.top_k,
        retrieval_mode=request.retrieval_mode,
        alpha=request.alpha,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature
    )

    chunks_converted = [
        SearchResultItem(
            chunk_id=c.chunk_id,
            text=c.text,
            score=round(float(c.score), 4),
            rank=c.rank,
            metadata=c.metadata
        )
        for c in rag_resp.retrieved_chunks
    ]

    citations_converted = [
        CitationItem(
            citation_id=cit.citation_id,
            chunk_id=cit.chunk_id,
            source_title=cit.source_title,
            snippet=cit.snippet,
            score=round(float(cit.score), 4)
        )
        for cit in rag_resp.citations
    ]

    return QueryResponse(
        query=rag_resp.query,
        answer=rag_resp.answer,
        retrieved_chunks=chunks_converted,
        citations=citations_converted,
        model_name=rag_resp.model_name,
        prompt_used=rag_resp.prompt_used,
        stats=rag_resp.stats
    )
