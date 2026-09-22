"""
src/rag/chunking.py

Text Chunking Strategies for RAG (Retrieval-Augmented Generation)
Local AI Research Lab - Developed by Kenan AY

Bu modül, büyük metin ve belgeleri bilgi kaybını en aza indirerek,
semantik bütünlüğü koruyarak parçalara (chunks) ayıran stratejileri içerir:
- RecursiveCharacterChunker: Paragraf, satır, cümle ve kelime sınırlarında hiyerarşik bölme
- SentenceChunker: Cümle sınırlarını kesin koruyan dilbilgisi duyarlı bölme
- FixedSizeChunker: Sabit karakter/token uzunluğunda kayan pencereli bölme
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Sequence
import re
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Bir metin parçasını ve metaverilerini temsil eden veri sınıfı."""
    chunk_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


class BaseChunker(ABC):
    """Chunking stratejileri için temel soyut sınıf."""

    @abstractmethod
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> List[TextChunk]:
        """Metni parçalara ayırır."""
        pass

    def estimate_token_count(self, text: str) -> int:
        """Basit ve hızlı token tahmini (yaklaşık 1 token ~ 4 karakter)."""
        words = text.split()
        return max(1, int(len(words) * 1.3)) if words else 0


class RecursiveCharacterChunker(BaseChunker):
    """
    Hiyerarşik ayırıcılar listesiyle metni bölmeye çalışan recursive chunker.
    Öncelikle paragraflara, sığmazsa satırlara, cümlelere ve en son kelimelere böler.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[Sequence[str]] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap, chunk_size'dan küçük olmalıdır.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators) if separators is not None else self.DEFAULT_SEPARATORS

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> List[TextChunk]:
        if not text or not text.strip():
            return []

        doc_meta = dict(metadata or {})
        if document_id:
            doc_meta["document_id"] = document_id

        raw_splits = self._split_text(text, self.separators)
        merged_chunks = self._merge_splits(raw_splits, text)

        chunks: List[TextChunk] = []
        for idx, (chunk_text, start_char, end_char) in enumerate(merged_chunks):
            chunk_id = f"chk_{document_id or 'doc'}_{idx:04d}_{uuid.uuid4().hex[:6]}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    chunk_index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=self.estimate_token_count(chunk_text),
                    metadata={**doc_meta, "strategy": "recursive", "chunk_size": len(chunk_text)}
                )
            )

        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Ayırıcı hiyerarşisine göre metni özyinelemeli parçalar."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits: List[str] = []
        for s in splits:
            if separator and s != splits[-1]:
                s = s + separator

            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    final_chunks.extend(good_splits)
                    good_splits = []
                if new_separators:
                    other_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(other_chunks)
                else:
                    final_chunks.append(s)

        if good_splits:
            final_chunks.extend(good_splits)

        return [c for c in final_chunks if c]

    def _merge_splits(self, splits: List[str], full_text: str) -> List[tuple[str, int, int]]:
        """Ayrılmış parçaları chunk_size ve chunk_overlap dikkate alarak birleştirir."""
        docs: List[tuple[str, int, int]] = []
        current_doc: List[str] = []
        total = 0

        for piece in splits:
            piece_len = len(piece)
            if total + piece_len > self.chunk_size and current_doc:
                combined = "".join(current_doc).strip()
                if combined:
                    start_idx = full_text.find(combined[:40]) if len(combined) >= 40 else full_text.find(combined)
                    start_idx = max(0, start_idx)
                    docs.append((combined, start_idx, start_idx + len(combined)))

                # Overlap için sondaki parçaları koru
                while total > self.chunk_overlap and current_doc:
                    removed = current_doc.pop(0)
                    total -= len(removed)

            current_doc.append(piece)
            total += piece_len

        if current_doc:
            combined = "".join(current_doc).strip()
            if combined:
                start_idx = full_text.find(combined[:40]) if len(combined) >= 40 else full_text.find(combined)
                start_idx = max(0, start_idx)
                docs.append((combined, start_idx, start_idx + len(combined)))

        return docs


class SentenceChunker(BaseChunker):
    """
    Cümle sınırlarını kesin koruyarak çalışan chunker.
    Türkçe noktalama işaretlerini ve kısaltmaları gözetir.
    """

    SENTENCE_SPLIT_REGEX = re.compile(
        r'(?<=[.?!])\s+(?=[A-ZÇĞİÖŞÜa-zçğıöşü0-9"\'\(\[])'
    )

    def __init__(
        self,
        max_chunk_size: int = 500,
        sentences_per_chunk: int = 3,
        sentence_overlap: int = 1
    ):
        self.max_chunk_size = max_chunk_size
        self.sentences_per_chunk = max(1, sentences_per_chunk)
        self.sentence_overlap = max(0, min(sentence_overlap, self.sentences_per_chunk - 1))

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> List[TextChunk]:
        if not text or not text.strip():
            return []

        doc_meta = dict(metadata or {})
        if document_id:
            doc_meta["document_id"] = document_id

        # Cümleleri ayır
        raw_sentences = self.SENTENCE_SPLIT_REGEX.split(text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            sentences = [text.strip()]

        chunks: List[TextChunk] = []
        i = 0
        chunk_idx = 0

        while i < len(sentences):
            current_sentences = sentences[i : i + self.sentences_per_chunk]
            chunk_str = " ".join(current_sentences)

            # Boyut aşımı varsa azalt
            while len(chunk_str) > self.max_chunk_size and len(current_sentences) > 1:
                current_sentences.pop()
                chunk_str = " ".join(current_sentences)

            start_char = text.find(chunk_str[:40]) if len(chunk_str) >= 40 else text.find(chunk_str)
            start_char = max(0, start_char)
            end_char = start_char + len(chunk_str)

            chunk_id = f"chk_{document_id or 'doc'}_{chunk_idx:04d}_{uuid.uuid4().hex[:6]}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_str,
                    chunk_index=chunk_idx,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=self.estimate_token_count(chunk_str),
                    metadata={
                        **doc_meta,
                        "strategy": "sentence",
                        "num_sentences": len(current_sentences),
                        "chunk_size": len(chunk_str)
                    }
                )
            )

            chunk_idx += 1
            step = max(1, len(current_sentences) - self.sentence_overlap)
            i += step

        return chunks


class FixedSizeChunker(BaseChunker):
    """
    Belirli karakter uzunluğu ve örtüşme ile sabit pencereli chunker.
    Hızlı ve deterministik dil bağımsız bölme sağlar.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap, chunk_size'dan küçük olmalıdır.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> List[TextChunk]:
        if not text or not text.strip():
            return []

        doc_meta = dict(metadata or {})
        if document_id:
            doc_meta["document_id"] = document_id

        clean_text = text.strip()
        step = self.chunk_size - self.chunk_overlap
        chunks: List[TextChunk] = []
        chunk_idx = 0

        for start in range(0, len(clean_text), step):
            end = min(start + self.chunk_size, len(clean_text))
            chunk_slice = clean_text[start:end]

            if not chunk_slice.strip():
                continue

            chunk_id = f"chk_{document_id or 'doc'}_{chunk_idx:04d}_{uuid.uuid4().hex[:6]}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_slice,
                    chunk_index=chunk_idx,
                    start_char=start,
                    end_char=end,
                    token_count=self.estimate_token_count(chunk_slice),
                    metadata={**doc_meta, "strategy": "fixed", "chunk_size": len(chunk_slice)}
                )
            )
            chunk_idx += 1
            if end >= len(clean_text):
                break

        return chunks


def get_chunker(strategy: str = "recursive", **kwargs: Any) -> BaseChunker:
    """Verilen strateji adına göre uygun chunker nesnesini döndürür."""
    strategy_clean = strategy.lower().strip()
    if strategy_clean in ("recursive", "recursive_character"):
        return RecursiveCharacterChunker(
            chunk_size=kwargs.get("chunk_size", 500),
            chunk_overlap=kwargs.get("chunk_overlap", 100)
        )
    elif strategy_clean in ("sentence", "sentences"):
        return SentenceChunker(
            max_chunk_size=kwargs.get("max_chunk_size", kwargs.get("chunk_size", 500)),
            sentences_per_chunk=kwargs.get("sentences_per_chunk", 3),
            sentence_overlap=kwargs.get("sentence_overlap", 1)
        )
    elif strategy_clean in ("fixed", "fixed_size"):
        return FixedSizeChunker(
            chunk_size=kwargs.get("chunk_size", 500),
            chunk_overlap=kwargs.get("chunk_overlap", 100)
        )
    else:
        logger.warning(f"Bilinmeyen chunking stratejisi: '{strategy}', varsayılan 'recursive' kullanılıyor.")
        return RecursiveCharacterChunker()
