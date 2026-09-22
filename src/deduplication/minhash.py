"""
src/deduplication/minhash.py

MinHash ve LSH (Locality-Sensitive Hashing) implementasyonu

Bu modül büyük metin koleksiyonlarında near-duplicate detection için
MinHash algoritması ve LSH banding technique kullanır.

Teori:
- MinHash: Jaccard similarity'yi efficiently tahmin eder
- LSH: O(n²) karşılaştırma yerine O(n) sublinear search
- Banding: Similar signatures'ı aynı bucket'a hash'ler

Kullanım:
    >>> deduplicator = MinHashDeduplicator(num_perm=128, threshold=0.85)
    >>> documents = ["metin 1", "metin 2", ...]
    >>> duplicates = deduplicator.find_duplicates(documents)

Kaynaklar:
    - Mining of Massive Datasets (Leskovec, Rajaraman, Ullman)
    - https://en.wikipedia.org/wiki/MinHash
    - https://en.wikipedia.org/wiki/Locality-sensitive_hashing

Version: 1.0.0
"""

import hashlib
import re
from typing import List, Set, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DuplicateGroup:
    """
    Duplicate document grubu.
    
    Attributes:
        representative_id: Grup representative'ı (genelde ilk döküman)
        document_ids: Gruptaki tüm döküman ID'leri
        similarity_score: Ortalama similarity skoru
        size: Gruptaki döküman sayısı
    """
    representative_id: str
    document_ids: List[str]
    similarity_score: float
    size: int = field(init=False)
    
    def __post_init__(self) -> None:
        self.size = len(self.document_ids)
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict formatına dönüştür."""
        return {
            'representative_id': self.representative_id,
            'document_ids': self.document_ids,
            'similarity_score': self.similarity_score,
            'size': self.size
        }


class MinHashDeduplicator:
    """
    MinHash + LSH tabanlı near-duplicate detector.
    
    Algorithm:
    1. Text → Shingles (n-grams)
    2. Shingles → MinHash signature (hash functions)
    3. Signatures → LSH bands (banding technique)
    4. Candidate pairs → Jaccard similarity verification
    
    Args:
        num_perm: Hash function sayısı (permutations)
        threshold: Similarity threshold (0.0-1.0)
        shingle_size: N-gram boyutu (default: 3)
        num_bands: LSH band sayısı (default: auto-computed)
        
    Attributes:
        prime: Large prime number for hash functions
        max_hash: Maximum hash value
    """
    
    # Large prime number for hash functions
    PRIME = 2**61 - 1
    MAX_HASH = 2**32 - 1
    
    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.85,
        shingle_size: int = 3,
        num_bands: Optional[int] = None
    ) -> None:
        """
        Initialize MinHash deduplicator.
        
        Args:
            num_perm: Hash function sayısı (daha fazla = daha accurate ama yavaş)
            threshold: Similarity threshold (0.85 = %85 benzerlik)
            shingle_size: N-gram boyutu (3 = trigrams)
            num_bands: LSH band sayısı (None = auto-compute)
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        
        # Auto-compute optimal band count
        if num_bands is None:
            # Optimal: b * r = num_perm, where b=bands, r=rows per band
            # For threshold t, we want (1/b)^(1/r) ≈ t
            self.num_bands = self._compute_optimal_bands(num_perm, threshold)
        else:
            self.num_bands = num_bands
        
        self.rows_per_band = num_perm // self.num_bands
        
        # Hash function parameters
        self._hash_params = self._generate_hash_params(num_perm)
        
        logger.info(
            f"MinHashDeduplicator initialized: "
            f"num_perm={num_perm}, threshold={threshold}, "
            f"bands={self.num_bands}, rows_per_band={self.rows_per_band}"
        )
    
    def _compute_optimal_bands(self, num_perm: int, threshold: float) -> int:
        """
        Optimal band sayısını hesapla.
        
        Target: (1/b)^(1/r) ≈ threshold
        where b=bands, r=rows_per_band
        
        Args:
            num_perm: Permutation sayısı
            threshold: Similarity threshold
            
        Returns:
            Optimal band sayısı
        """
        # Try different band counts
        best_bands = 1
        best_diff = float('inf')
        
        for b in range(1, num_perm + 1):
            if num_perm % b == 0:
                r = num_perm // b
                # Probability of being candidate: 1 - (1 - t^r)^b
                # For threshold t, we want this close to 1
                estimated_threshold = (1.0 / b) ** (1.0 / r)
                diff = abs(estimated_threshold - threshold)
                
                if diff < best_diff:
                    best_diff = diff
                    best_bands = b
        
        return best_bands
    
    def _generate_hash_params(self, num_perm: int) -> List[Tuple[int, int]]:
        """
        Hash function parametrelerini üret.
        
        Hash function: h(x) = (a * x + b) mod prime
        
        Args:
            num_perm: Kaç hash function
            
        Returns:
            List of (a, b) tuples
        """
        params = []
        
        # Use deterministic seed for reproducibility
        seed = 42
        
        for i in range(num_perm):
            # Generate pseudo-random a and b
            a = (seed * (i + 1) + 17) % self.PRIME
            b = (seed * (i + 2) + 31) % self.PRIME
            
            # Ensure a is not 0
            if a == 0:
                a = 1
            
            params.append((a, b))
        
        return params
    
    def _text_to_shingles(self, text: str) -> Set[str]:
        """
        Metni shingle'lara (n-grams) dönüştür.
        
        Args:
            text: Input text
            
        Returns:
            Set of shingles
        """
        # Normalize text
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Generate character-level shingles
        shingles: Set[str] = set()
        
        if len(text) < self.shingle_size:
            # Text too short, use as single shingle
            shingles.add(text)
        else:
            for i in range(len(text) - self.shingle_size + 1):
                shingle = text[i:i + self.shingle_size]
                shingles.add(shingle)
        
        return shingles
    
    def _hash_shingle(self, shingle: str) -> int:
        """
        Shingle'ı hash'le.
        
        Args:
            shingle: Shingle string
            
        Returns:
            Hash value
        """
        # Use SHA-256 and take first 4 bytes
        hash_bytes = hashlib.sha256(shingle.encode('utf-8')).digest()
        hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
        return hash_int % self.MAX_HASH
    
    def compute_signature(self, text: str) -> List[int]:
        """
        Text için MinHash signature hesapla.
        
        Args:
            text: Input text
            
        Returns:
            MinHash signature (list of hash values)
        """
        # Get shingles
        shingles = self._text_to_shingles(text)
        
        if not shingles:
            # Empty text, return max hashes
            return [self.MAX_HASH] * self.num_perm
        
        # Initialize signature with max values
        signature = [self.MAX_HASH] * self.num_perm
        
        # For each shingle
        for shingle in shingles:
            shingle_hash = self._hash_shingle(shingle)
            
            # Apply each hash function
            for i, (a, b) in enumerate(self._hash_params):
                # h(x) = (a * x + b) mod prime
                h = (a * shingle_hash + b) % self.PRIME
                h = h % self.MAX_HASH
                
                # MinHash: keep minimum
                signature[i] = min(signature[i], h)
        
        return signature
    
    def _signature_to_bands(self, signature: List[int]) -> List[Tuple[int, ...]]:
        """
        Signature'ı band'lere böl.
        
        Args:
            signature: MinHash signature
            
        Returns:
            List of band tuples
        """
        bands = []
        
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(signature[start:end])
            bands.append(band)
        
        return bands
    
    def _compute_jaccard_similarity(
        self,
        sig1: List[int],
        sig2: List[int]
    ) -> float:
        """
        İki signature arasındaki Jaccard similarity'yi hesapla.
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Jaccard similarity (0.0-1.0)
        """
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have same length")
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        similarity = matches / len(sig1)
        
        return similarity
    
    def find_duplicates(
        self,
        documents: List[Tuple[str, str]]
    ) -> List[DuplicateGroup]:
        """
        Document listesinde near-duplicate'leri bul.
        
        Args:
            documents: List of (doc_id, text) tuples
            
        Returns:
            List of DuplicateGroup objects
        """
        logger.info(f"Finding duplicates in {len(documents)} documents")
        
        # Step 1: Compute signatures
        signatures: Dict[str, List[int]] = {}
        
        for doc_id, text in documents:
            signatures[doc_id] = self.compute_signature(text)
        
        logger.info(f"Computed {len(signatures)} signatures")
        
        # Step 2: LSH - Hash into buckets
        buckets: Dict[Tuple[int, Tuple[int, ...]], List[str]] = defaultdict(list)
        
        for doc_id, signature in signatures.items():
            bands = self._signature_to_bands(signature)
            
            for band_idx, band in enumerate(bands):
                # Hash band into bucket
                bucket_key = (band_idx, band)
                buckets[bucket_key].append(doc_id)
        
        logger.info(f"Created {len(buckets)} LSH buckets")
        
        # Step 3: Find candidate pairs
        candidate_pairs: Set[Tuple[str, str]] = set()
        
        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                # All pairs in this bucket are candidates
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        doc_id1 = bucket_docs[i]
                        doc_id2 = bucket_docs[j]
                        
                        # Ensure consistent ordering
                        pair = (doc_id1, doc_id2) if doc_id1 <= doc_id2 else (doc_id2, doc_id1)
                        candidate_pairs.add(pair)
        
        logger.info(f"Found {len(candidate_pairs)} candidate pairs")
        
        # Step 4: Verify candidates with Jaccard similarity
        verified_pairs: List[Tuple[str, str, float]] = []
        
        for doc_id1, doc_id2 in candidate_pairs:
            similarity = self._compute_jaccard_similarity(
                signatures[doc_id1],
                signatures[doc_id2]
            )
            
            if similarity >= self.threshold:
                verified_pairs.append((doc_id1, doc_id2, similarity))
        
        logger.info(f"Verified {len(verified_pairs)} duplicate pairs")
        
        # Step 5: Group duplicates using Union-Find
        duplicate_groups = self._group_duplicates(verified_pairs)
        
        logger.info(f"Created {len(duplicate_groups)} duplicate groups")
        
        return duplicate_groups
    
    def _group_duplicates(
        self,
        pairs: List[Tuple[str, str, float]]
    ) -> List[DuplicateGroup]:
        """
        Duplicate pair'leri group'lara dönüştür (Union-Find).
        
        Args:
            pairs: List of (doc_id1, doc_id2, similarity)
            
        Returns:
            List of DuplicateGroup
        """
        # Union-Find data structure
        parent: Dict[str, str] = {}
        
        def find(doc_id: str) -> str:
            """Find root of doc_id."""
            if doc_id not in parent:
                parent[doc_id] = doc_id
            
            if parent[doc_id] != doc_id:
                # Path compression
                parent[doc_id] = find(parent[doc_id])
            
            return parent[doc_id]
        
        def union(doc_id1: str, doc_id2: str) -> None:
            """Union two sets."""
            root1 = find(doc_id1)
            root2 = find(doc_id2)
            
            if root1 != root2:
                parent[root2] = root1
        
        # Union all pairs
        similarity_map: Dict[Tuple[str, str], float] = {}
        
        for doc_id1, doc_id2, similarity in pairs:
            union(doc_id1, doc_id2)
            similarity_map[(doc_id1, doc_id2)] = similarity
        
        # Group documents by root
        groups: Dict[str, List[str]] = defaultdict(list)
        
        for doc_id in parent.keys():
            root = find(doc_id)
            groups[root].append(doc_id)
        
        # Create DuplicateGroup objects
        duplicate_groups = []
        
        for representative_id, doc_ids in groups.items():
            if len(doc_ids) > 1:
                # Compute average similarity
                similarities = []
                for i in range(len(doc_ids)):
                    for j in range(i + 1, len(doc_ids)):
                        d1, d2 = doc_ids[i], doc_ids[j]
                        pair = (d1, d2) if d1 <= d2 else (d2, d1)
                        if pair in similarity_map:
                            similarities.append(similarity_map[pair])
                
                avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
                
                group = DuplicateGroup(
                    representative_id=representative_id,
                    document_ids=sorted(doc_ids),
                    similarity_score=avg_similarity
                )
                duplicate_groups.append(group)
        
        return duplicate_groups
    
    def deduplicate(
        self,
        documents: List[Tuple[str, str]],
        keep_representative: bool = True
    ) -> List[str]:
        """
        Duplicate'leri tespit et ve unique document ID listesi döndür.
        
        Args:
            documents: List of (doc_id, text) tuples
            keep_representative: Her gruptan representative'ı tut
            
        Returns:
            Unique document ID listesi
        """
        duplicate_groups = self.find_duplicates(documents)
        
        # Document ID'lerinin tümü
        all_doc_ids = {doc_id for doc_id, _ in documents}
        
        # Duplicate'leri çıkar
        duplicates_to_remove: Set[str] = set()
        
        for group in duplicate_groups:
            if keep_representative:
                # Representative hariç hepsini kaldır
                for doc_id in group.document_ids:
                    if doc_id != group.representative_id:
                        duplicates_to_remove.add(doc_id)
            else:
                # Tüm grubu kaldır
                duplicates_to_remove.update(group.document_ids)
        
        # Unique document ID'leri
        unique_doc_ids = list(all_doc_ids - duplicates_to_remove)
        
        logger.info(
            f"Deduplication: {len(unique_doc_ids)}/{len(all_doc_ids)} unique "
            f"({len(duplicates_to_remove)} removed)"
        )
        
        return unique_doc_ids


def find_near_duplicates(
    documents: List[Tuple[str, str]],
    threshold: float = 0.85,
    num_perm: int = 128
) -> List[DuplicateGroup]:
    """
    Convenience function: Near-duplicate'leri bul.
    
    Args:
        documents: List of (doc_id, text) tuples
        threshold: Similarity threshold (0.0-1.0)
        num_perm: Hash function sayısı
        
    Returns:
        List of DuplicateGroup
    """
    deduplicator = MinHashDeduplicator(
        num_perm=num_perm,
        threshold=threshold
    )
    
    return deduplicator.find_duplicates(documents)
