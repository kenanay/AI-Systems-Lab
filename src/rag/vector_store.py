"""
src/rag/vector_store.py

Dense Vector Storage and Similarity Search for RAG
Local AI Research Lab - Developed by Kenan AY

Bu modül, metin parçalarının dense embedding vektörlerini saklar,
vektör indekslemesini yönetir ve Top-K benzerlik araması gerçekleştirir:
- PyTorch GPU/CPU tensör tabanlı L2-normalized Cosine Similarity
- Opsiyonel FAISS IndexFlatIP hızlandırması (varsa otomatik geçiş)
- Diske kalıcı kayıt ve geri yükleme (vektörler + metadata)
- Çoklu koleksiyon (collection) desteği
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

# Opsiyonel FAISS kontrolü
faiss_module: Any = None
try:
    import faiss as _faiss  # type: ignore
    faiss_module = _faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@dataclass
class SearchResult:
    """Vektör veya hibrit arama sonucu."""
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(float(self.score), 4),
            "rank": self.rank,
            "metadata": self.metadata,
        }


class VectorStore:
    """
    Yerel ve yüksek performanslı dense vektör deposu.
    PyTorch vektörize inner product ve isteğe bağlı FAISS backend'i sunar.
    """

    def __init__(
        self,
        collection_name: str = "default",
        d_model: Optional[int] = None,
        use_faiss_if_available: bool = True
    ):
        self.collection_name = collection_name
        self.d_model = d_model
        self.use_faiss = use_faiss_if_available and HAS_FAISS
        
        # Depolama yapıları
        self.vectors: Optional[torch.Tensor] = None  # Shape: [N, d_model]
        self.metadata_records: List[Dict[str, Any]] = []
        self.chunk_ids: List[str] = []
        
        # FAISS index (varsa)
        self.faiss_index: Any = None
        if self.use_faiss and self.d_model is not None:
            self._init_faiss(self.d_model)

    def _init_faiss(self, dim: int) -> None:
        """FAISS IndexFlatIP (Inner Product) indeksini ilklendirir."""
        if HAS_FAISS and faiss_module is not None:
            self.faiss_index = faiss_module.IndexFlatIP(dim)
            logger.info(f"FAISS IndexFlatIP başlatıldı (dim={dim})")

    @property
    def count(self) -> int:
        """Koleksiyondaki vektör sayısı."""
        return len(self.chunk_ids)

    def add_vectors(
        self,
        chunk_ids: List[str],
        texts: List[str],
        vectors: Union[torch.Tensor, np.ndarray, List[List[float]]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Koleksiyona yeni vektörler ve metaverilerini ekler.
        Vektörler otomatik olarak L2 normalize edilir.
        """
        if not chunk_ids:
            return 0

        # Tensöre dönüştür
        if isinstance(vectors, np.ndarray):
            t_vecs = torch.from_numpy(vectors).float()
        elif isinstance(vectors, torch.Tensor):
            t_vecs = vectors.float().detach().cpu()
        else:
            t_vecs = torch.tensor(vectors, dtype=torch.float32)

        if t_vecs.ndim == 1:
            t_vecs = t_vecs.unsqueeze(0)

        n_new, dim = t_vecs.shape
        if self.d_model is None:
            self.d_model = dim
            if self.use_faiss and self.faiss_index is None:
                self._init_faiss(dim)
        elif self.d_model != dim:
            raise ValueError(f"Vektör boyutu uyuşmuyor: Beklenen {self.d_model}, Gelen {dim}")

        # L2 Normalization (Cosine similarity için birim vektör)
        norms = torch.norm(t_vecs, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        norm_vecs = t_vecs / norms

        # PyTorch havuzuna ekle
        if self.vectors is None:
            self.vectors = norm_vecs
        else:
            self.vectors = torch.cat([self.vectors, norm_vecs], dim=0)

        # FAISS havuzuna ekle
        if self.use_faiss and self.faiss_index is not None:
            np_data = norm_vecs.numpy().astype(np.float32)
            self.faiss_index.add(np_data)

        # Metaveri kayıtlarını tut
        for idx in range(n_new):
            c_id = chunk_ids[idx]
            txt = texts[idx]
            meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            record = {
                "chunk_id": c_id,
                "text": txt,
                "metadata": meta,
            }
            self.chunk_ids.append(c_id)
            self.metadata_records.append(record)

        logger.info(f"{n_new} vektör '{self.collection_name}' koleksiyonuna eklendi (Toplam: {self.count}).")
        return n_new

    def search(
        self,
        query_vector: Union[torch.Tensor, np.ndarray, List[float]],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Sorgu vektörüne en yakın top_k parçayı döndürür.
        Cosine similarity aralığı: [-1.0, 1.0].
        """
        if self.count == 0 or self.vectors is None:
            return []

        top_k = min(max(1, top_k), self.count)

        # Sorguyu hazırla
        if isinstance(query_vector, np.ndarray):
            q_t = torch.from_numpy(query_vector).float()
        elif isinstance(query_vector, torch.Tensor):
            q_t = query_vector.float().detach().cpu()
        else:
            q_t = torch.tensor(query_vector, dtype=torch.float32)

        if q_t.ndim > 1:
            q_t = q_t.squeeze()

        # Normalize et
        q_norm = q_t / torch.norm(q_t, p=2).clamp_min(1e-12)

        results: List[SearchResult] = []

        # FAISS ile ara
        if self.use_faiss and self.faiss_index is not None:
            q_np = q_norm.unsqueeze(0).numpy().astype(np.float32)
            scores, indices = self.faiss_index.search(q_np, top_k)
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < 0 or idx >= len(self.metadata_records):
                    continue
                score_f = float(score)
                if score_threshold is not None and score_f < score_threshold:
                    continue
                rec = self.metadata_records[idx]
                results.append(
                    SearchResult(
                        chunk_id=rec["chunk_id"],
                        text=rec["text"],
                        score=score_f,
                        rank=rank + 1,
                        metadata=rec["metadata"],
                    )
                )
            return results

        # PyTorch tensör matris çarpımıyla ara (Cosine Similarity)
        # self.vectors shape: [N, d], q_norm shape: [d] -> scores shape: [N]
        scores_t = torch.mv(self.vectors, q_norm)
        top_scores, top_indices = torch.topk(scores_t, k=top_k)

        for rank, (score_val, idx_val) in enumerate(zip(top_scores.tolist(), top_indices.tolist())):
            score_f = float(score_val)
            if score_threshold is not None and score_f < score_threshold:
                continue
            rec = self.metadata_records[idx_val]
            results.append(
                SearchResult(
                    chunk_id=rec["chunk_id"],
                    text=rec["text"],
                    score=score_f,
                    rank=rank + 1,
                    metadata=rec["metadata"],
                )
            )

        return results

    def save(self, dir_path: Union[str, Path]) -> Path:
        """Koleksiyonu diske kaydeder."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)

        info = {
            "collection_name": self.collection_name,
            "d_model": self.d_model,
            "count": self.count,
            "backend": "faiss" if self.use_faiss else "pytorch",
        }
        with open(path / "collection_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata_records, f, indent=2, ensure_ascii=False)

        if self.vectors is not None:
            torch.save(self.vectors, path / "vectors.pt")

        logger.info(f"VectorStore kaydedildi: {path} ({self.count} vektör)")
        return path

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> "VectorStore":
        """Diskteki koleksiyonu yükler."""
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Dizin bulunamadı: {path}")

        with open(path / "collection_info.json", "r", encoding="utf-8") as f:
            info = json.load(f)

        store = cls(
            collection_name=info.get("collection_name", "loaded"),
            d_model=info.get("d_model"),
        )

        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            store.metadata_records = json.load(f)
            store.chunk_ids = [r["chunk_id"] for r in store.metadata_records]

        vec_path = path / "vectors.pt"
        if vec_path.exists():
            store.vectors = torch.load(vec_path, map_location="cpu")
            if store.use_faiss and store.d_model is not None and store.vectors is not None:
                store._init_faiss(store.d_model)
                if store.faiss_index is not None:
                    store.faiss_index.add(store.vectors.numpy().astype(np.float32))

        logger.info(f"VectorStore yüklendi: {path} ({store.count} vektör)")
        return store
