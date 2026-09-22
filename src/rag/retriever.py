"""
src/rag/retriever.py

Sparse (BM25) and Hybrid Retrieval for RAG
Local AI Research Lab - Developed by Kenan AY

Bu modül iki temel arama motorunu birleştirir:
1. BM25Index: İstatistiki ters indeks (inverted index) ve TF-IDF temelli kelime araması.
2. HybridRetriever: Yoğun (Vektör/Semantik) ve Seyrek (BM25/Leksikal) aramayı
   Reciprocal Rank Fusion (RRF) veya Ağırlıklı Skor Kombinasyonu ile harmanlayan hibrit arama motoru.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Set
from collections import Counter, defaultdict
import math
import re
import logging

from src.rag.vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


def _tokenize_text(text: str) -> List[str]:
    """Türkçe karakterleri gözeten ve noktalama işaretlerini temizleyen tokenizasyon."""
    clean = text.lower()
    clean = re.sub(r'[\r\n\t]', ' ', clean)
    clean = re.sub(r'[^\w\s]', ' ', clean)
    tokens = [t.strip() for t in clean.split() if len(t.strip()) > 1]
    return tokens


class BM25Index:
    """
    Okapi BM25 sıralama algoritması.
    Anahtar kelime ve tam terim eşleşmelerinde yüksek hassasiyet sunar.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        # Dizin alanları
        self.chunk_ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # term -> [(doc_idx, tf)]
        self.term_doc_frequencies: Dict[str, int] = defaultdict(int)

    @property
    def corpus_size(self) -> int:
        return len(self.documents)

    def add_documents(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Külliyata yeni belgeler ekler ve ters dizini günceller."""
        if not texts:
            return 0

        start_idx = len(self.documents)
        total_len = sum(self.doc_lengths)

        for i, (cid, txt) in enumerate(zip(chunk_ids, texts)):
            idx = start_idx + i
            tokens = _tokenize_text(txt)
            doc_len = len(tokens)
            
            self.chunk_ids.append(cid)
            self.documents.append(txt)
            self.metadatas.append(metadatas[i] if metadatas and i < len(metadatas) else {})
            self.doc_lengths.append(doc_len)
            total_len += doc_len

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.inverted_index[term].append((idx, count))
                self.term_doc_frequencies[term] += 1

        self.avg_doc_length = total_len / max(1, self.corpus_size)
        return len(texts)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """BM25 skorlarına göre en iyi belgeleri getirir."""
        if self.corpus_size == 0 or not query.strip():
            return []

        query_tokens = _tokenize_text(query)
        if not query_tokens:
            return []

        scores: Dict[int, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            df = self.term_doc_frequencies[token]
            # IDF hesabı (BM25 standart logaritmik formülü)
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

            for doc_idx, tf in self.inverted_index[token]:
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-5, self.avg_doc_length)))
                score_term = idf * (numerator / max(1e-5, denominator))
                scores[doc_idx] += score_term

        if not scores:
            return []

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[SearchResult] = []
        for rank, (doc_idx, raw_score) in enumerate(sorted_docs):
            results.append(
                SearchResult(
                    chunk_id=self.chunk_ids[doc_idx],
                    text=self.documents[doc_idx],
                    score=raw_score,
                    rank=rank + 1,
                    metadata={
                        **self.metadatas[doc_idx],
                        "retrieval_type": "bm25",
                        "bm25_raw_score": round(raw_score, 4)
                    }
                )
            )

        return results


class HybridRetriever:
    """
    Yoğun (Dense/Vector) ve Seyrek (Sparse/BM25) aramayı birleştiren hibrit motor.
    Reciprocal Rank Fusion (RRF) kullanarak iki farklı arama alanının skorlarını dengeler.
    """

    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index):
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def search(
        self,
        query: str,
        query_vector: Optional[Union[Any, List[float]]] = None,
        mode: str = "hybrid",
        top_k: int = 5,
        alpha: float = 0.5,
        rrf_k: int = 60
    ) -> List[SearchResult]:
        """
        Sorgu aramasını gerçekleştirir.
        
        Args:
            query: Doğal dil sorgusu
            query_vector: İsteğe bağlı embedding vektörü
            mode: 'hybrid' | 'dense' | 'sparse'
            top_k: Döndürülecek sonuç sayısı
            alpha: Hibrit skor ağırlığı (1.0 = sadece dense, 0.0 = sadece sparse)
            rrf_k: RRF yumuşatma sabiti (varsayılan: 60)
        """
        mode = mode.lower().strip()

        # 1. Yalnızca Seyrek (Sparse - BM25)
        if mode in ("sparse", "bm25"):
            return self.bm25_index.search(query=query, top_k=top_k)

        # 2. Yalnızca Yoğun (Dense - Vector)
        if mode in ("dense", "vector"):
            if query_vector is None:
                logger.warning("Dense arama için query_vector sağlanmadı, BM25'e düşülüyor.")
                return self.bm25_index.search(query=query, top_k=top_k)
            return self.vector_store.search(query_vector=query_vector, top_k=top_k)

        # 3. Hibrit Arama (RRF / Weighted Fusion)
        fetch_k = max(top_k * 3, 20)
        sparse_results = self.bm25_index.search(query=query, top_k=fetch_k)
        
        dense_results: List[SearchResult] = []
        if query_vector is not None:
            dense_results = self.vector_store.search(query_vector=query_vector, top_k=fetch_k)

        # Eğer yoğun arama yapılamıyorsa sparse dön
        if not dense_results:
            return sparse_results[:top_k]

        # Reciprocal Rank Fusion (RRF)
        # RRF_Score(d) = sum(1 / (k + rank))
        rrf_scores: Dict[str, float] = defaultdict(float)
        all_chunks: Dict[str, SearchResult] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        for res in dense_results:
            all_chunks[res.chunk_id] = res
            dense_ranks[res.chunk_id] = res.rank
            rrf_scores[res.chunk_id] += alpha * (1.0 / (rrf_k + res.rank))

        for res in sparse_results:
            all_chunks[res.chunk_id] = res
            sparse_ranks[res.chunk_id] = res.rank
            rrf_scores[res.chunk_id] += (1.0 - alpha) * (1.0 / (rrf_k + res.rank))

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        final_results: List[SearchResult] = []
        for rank, (cid, fusion_score) in enumerate(sorted_chunks):
            base_item = all_chunks[cid]
            d_rank = dense_ranks.get(cid)
            s_rank = sparse_ranks.get(cid)
            
            combined_meta = dict(base_item.metadata)
            combined_meta["retrieval_type"] = "hybrid_rrf"
            combined_meta["dense_rank"] = d_rank
            combined_meta["sparse_rank"] = s_rank
            combined_meta["rrf_score"] = round(fusion_score, 6)

            final_results.append(
                SearchResult(
                    chunk_id=cid,
                    text=base_item.text,
                    score=fusion_score,
                    rank=rank + 1,
                    metadata=combined_meta
                )
            )

        return final_results
