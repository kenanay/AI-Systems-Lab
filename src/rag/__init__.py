"""
src/rag/__init__.py

RAG (Retrieval-Augmented Generation) Subsystem
Local AI Research Lab - Developed by Kenan AY
"""

from src.rag.chunking import (
    TextChunk,
    BaseChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
    FixedSizeChunker,
    get_chunker,
)
from src.rag.vector_store import (
    VectorStore,
    SearchResult,
)
from src.rag.retriever import (
    BM25Index,
    HybridRetriever,
)
from src.rag.pipeline import (
    RAGPipeline,
    Citation,
    RAGResponse,
)

__all__ = [
    "TextChunk",
    "BaseChunker",
    "RecursiveCharacterChunker",
    "SentenceChunker",
    "FixedSizeChunker",
    "get_chunker",
    "VectorStore",
    "SearchResult",
    "BM25Index",
    "HybridRetriever",
    "RAGPipeline",
    "Citation",
    "RAGResponse",
]
