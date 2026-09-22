"""
tests/test_rag.py

Comprehensive Unit and Integration Tests for RAG Subsystem:
- Chunking strategies (Recursive, Sentence, Fixed)
- Dense VectorStore (cosine similarity, top-k, save/load persistence)
- BM25Index (inverted index, IDF calculation, lexical matching)
- HybridRetriever (Reciprocal Rank Fusion RRF)
- RAGPipeline (grounded prompt building, citations)
- FastAPI REST endpoints (/chunk, /index, /collections, /search, /query)
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from src.rag.chunking import (
    RecursiveCharacterChunker,
    SentenceChunker,
    FixedSizeChunker,
    get_chunker,
    TextChunk
)
from src.rag.vector_store import VectorStore, SearchResult
from src.rag.retriever import BM25Index, HybridRetriever
from src.rag.pipeline import RAGPipeline
from backend.main import app

client = TestClient(app)


# ============================================================================
# 1. Chunking Tests
# ============================================================================

def test_recursive_character_chunker_basic():
    text = (
        "Paragraf bir burasıdır. Bu paragraf yapay zekayı anlatır.\n\n"
        "İkinci paragraf burasıdır. Bu da derin öğrenme üzerine kuruludur.\n\n"
        "Üçüncü paragraf ise transformatör mimarilerini açıklar."
    )
    chunker = RecursiveCharacterChunker(chunk_size=70, chunk_overlap=15)
    chunks = chunker.chunk_text(text, metadata={"doc_type": "article"}, document_id="doc1")

    assert len(chunks) >= 3
    for c in chunks:
        assert isinstance(c, TextChunk)
        assert c.chunk_id.startswith("chk_doc1_")
        assert len(c.text) <= 90  # chunk_size around 70 with margins
        assert c.start_char >= 0
        assert c.end_char > c.start_char
        assert c.token_count > 0
        assert c.metadata["doc_type"] == "article"


def test_sentence_chunker():
    text = (
        "Türkiye'nin başkenti Ankara'dır. İstanbul ise en kalabalık şehridir! "
        "İzmir Ege Bölgesi'nin incisidir? Antalya turizmin merkezidir."
    )
    chunker = SentenceChunker(max_chunk_size=120, sentences_per_chunk=2, sentence_overlap=1)
    chunks = chunker.chunk_text(text, document_id="geo_doc")

    assert len(chunks) >= 2
    assert "Ankara'dır" in chunks[0].text
    assert chunks[0].metadata["strategy"] == "sentence"


def test_fixed_size_chunker():
    text = "A" * 250
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text(text)

    # 250 characters with 100 size and 20 overlap: step=80 -> 0:100, 80:180, 160:250 -> 3-4 chunks
    assert len(chunks) in (3, 4)
    assert len(chunks[0].text) == 100


def test_get_chunker_factory():
    c1 = get_chunker("recursive", chunk_size=300)
    assert isinstance(c1, RecursiveCharacterChunker)
    assert c1.chunk_size == 300

    c2 = get_chunker("sentence")
    assert isinstance(c2, SentenceChunker)

    c3 = get_chunker("fixed", chunk_size=200, chunk_overlap=50)
    assert isinstance(c3, FixedSizeChunker)


# ============================================================================
# 2. Vector Store & Similarity Tests
# ============================================================================

def test_vector_store_add_and_search():
    store = VectorStore(collection_name="test_col", d_model=4, use_faiss_if_available=False)

    chunk_ids = ["c1", "c2", "c3"]
    texts = [
        "Yapay zeka modelleri",
        "Derin öğrenme ağları",
        "Bahçede elma ağaçları"
    ]
    # Orthogonal or distinct vectors
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    added = store.add_vectors(chunk_ids, texts, vectors, metadatas=[{"k": 1}, {"k": 2}, {"k": 3}])
    assert added == 3
    assert store.count == 3

    # Query matching c1 / c2 closely
    query_vec = [1.0, 0.0, 0.0, 0.0]
    results = store.search(query_vec, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert pytest.approx(results[0].score, 0.01) == 1.0
    assert results[1].chunk_id == "c2"
    assert results[0].rank == 1
    assert results[1].rank == 2


def test_vector_store_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir) / "test_store"
        store = VectorStore(collection_name="persist_col", d_model=3, use_faiss_if_available=False)
        store.add_vectors(["p1", "p2"], ["Metin bir", "Metin iki"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        store.save(dir_path)

        loaded_store = VectorStore.load(dir_path)
        assert loaded_store.count == 2
        assert loaded_store.d_model == 3
        assert loaded_store.chunk_ids == ["p1", "p2"]

        res = loaded_store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(res) == 1
        assert res[0].chunk_id == "p1"


# ============================================================================
# 3. BM25 and Hybrid Retriever Tests
# ============================================================================

def test_bm25_index_search():
    bm25 = BM25Index()
    bm25.add_documents(
        chunk_ids=["b1", "b2", "b3"],
        texts=[
            "Python programlama dili veri bilimi için popülerdir.",
            "JavaScript web geliştirme ve frontend arayüzlerinde kullanılır.",
            "Python makine öğrenmesi kütüphaneleri TensorFlow ve PyTorch içerir."
        ],
        metadatas=[{"topic": "python"}, {"topic": "js"}, {"topic": "ml"}]
    )

    results = bm25.search("Python veri bilimi", top_k=2)
    assert len(results) >= 1
    # b1 has both 'Python' and 'veri bilimi'
    assert results[0].chunk_id == "b1"
    assert results[0].score > 0
    assert results[0].metadata["retrieval_type"] == "bm25"


def test_hybrid_retriever():
    store = VectorStore(d_model=2, use_faiss_if_available=False)
    store.add_vectors(["h1", "h2"], ["Kedi ve köpek", "Uzay gemisi"], [[1.0, 0.0], [0.0, 1.0]])

    bm25 = BM25Index()
    bm25.add_documents(["h1", "h2"], ["Kedi ve köpek", "Uzay gemisi"])

    hybrid = HybridRetriever(store, bm25)

    # Search hybrid
    results = hybrid.search(query="Kedi", query_vector=[1.0, 0.0], mode="hybrid", top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "h1"
    assert results[0].metadata["retrieval_type"] == "hybrid_rrf"
    assert "rrf_score" in results[0].metadata


# ============================================================================
# 4. RAG Pipeline Tests
# ============================================================================

def test_rag_pipeline_prompt_building():
    pipeline = RAGPipeline(d_model=16)
    chunks = [
        TextChunk("c1", "KV Cache çıkarım hızını katlar.", 0, 0, 30, 6, {"title": "KV Cache Belgesi"}),
        TextChunk("c2", "LoRA düşük rankli matris adaptasyonudur.", 1, 31, 70, 7, {"title": "LoRA Belgesi"})
    ]
    pipeline.index_chunks(chunks)

    full_prompt, context = pipeline.build_grounded_prompt(
        "KV Cache nedir?",
        pipeline.hybrid_retriever.search("KV Cache", pipeline.embed_text("KV Cache"), top_k=2)
    )

    assert "[Kaynak 1]" in full_prompt
    assert "KV Cache" in full_prompt
    assert "KULLANICI SORUSU: KV Cache nedir?" in full_prompt


def test_rag_pipeline_query_execution():
    pipeline = RAGPipeline(d_model=16)
    chunker = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_text(
        "Attention mekanizması Q, K ve V tensörlerinin çarpımıyla hesaplanır. "
        "Formülü softmax(QK^T / sqrt(d_k))V şeklindedir.",
        metadata={"title": "Attention Makalesi"}
    )
    pipeline.index_chunks(chunks)

    resp = pipeline.query("Attention formülü nedir?", top_k=1)
    assert resp.query == "Attention formülü nedir?"
    assert len(resp.retrieved_chunks) >= 1
    assert len(resp.citations) >= 1
    assert resp.citations[0].citation_id == 1
    assert "Attention" in resp.citations[0].snippet
    assert "latency_ms" in resp.stats


# ============================================================================
# 5. FastAPI RAG API Endpoint Tests
# ============================================================================

def test_api_chunk_endpoint():
    response = client.post(
        "/api/v1/rag/chunk",
        json={
            "text": "Birinci cümle burasıdır. İkinci cümle ise daha uzundur. Üçüncü cümle buradadır.",
            "strategy": "sentence",
            "chunk_size": 80,
            "chunk_overlap": 20
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] >= 1
    assert data["strategy"] == "sentence"
    assert len(data["chunks"]) == data["total_chunks"]
    assert "chunk_id" in data["chunks"][0]


def test_api_collections_and_search_endpoint():
    # 1. Collections endpoint
    res_col = client.get("/api/v1/rag/collections")
    assert res_col.status_code == 200
    collections = res_col.json()
    assert len(collections) >= 1
    assert collections[0]["collection_name"] == "default"
    assert collections[0]["vector_count"] > 0

    # 2. Search endpoint
    res_search = client.post(
        "/api/v1/rag/search",
        json={
            "query": "Transformer attention formülü",
            "collection_name": "default",
            "mode": "hybrid",
            "top_k": 3,
            "alpha": 0.5
        }
    )
    assert res_search.status_code == 200
    search_data = res_search.json()
    assert search_data["count"] > 0
    assert len(search_data["results"]) <= 3
    assert search_data["results"][0]["score"] > 0
    assert "latency_ms" in search_data


def test_api_query_rag_endpoint():
    response = client.post(
        "/api/v1/rag/query",
        json={
            "question": "LoRA ince ayarı nasıl çalışır?",
            "collection_name": "default",
            "retrieval_mode": "hybrid",
            "top_k": 2,
            "alpha": 0.5
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "LoRA ince ayarı nasıl çalışır?"
    assert len(data["answer"]) > 0
    assert len(data["retrieved_chunks"]) >= 1
    assert len(data["citations"]) >= 1
    assert "stats" in data
    assert "latency_ms" in data["stats"]


def test_api_index_custom_text_endpoint():
    response = client.post(
        "/api/v1/rag/index",
        json={
            "collection_name": "test_custom_api",
            "custom_texts": [
                {
                    "title": "Quantum Computing",
                    "text": "Kuantum bilgisayarlar kubitleri süperpozisyon ve dolanıklık ilkeleriyle işler."
                }
            ],
            "chunk_strategy": "recursive",
            "chunk_size": 200,
            "chunk_overlap": 50
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["collection_name"] == "test_custom_api"
    assert data["indexed_chunks"] >= 1
    assert data["total_vectors_in_collection"] >= 1
