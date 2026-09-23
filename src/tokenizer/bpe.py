"""
src/tokenizer/bpe.py

Byte-Pair Encoding (BPE) Tokenizer

Bu modül BPE algoritmasını implement eder. BPE, text'i
subword units'lere bölmek için kullanılan bir algoritmadır.

Kaynaklar:
    - Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016)
    - https://arxiv.org/abs/1508.07909

Kullanım:
    >>> from src.tokenizer.bpe import BPETokenizer
    >>> tokenizer = BPETokenizer(vocab_size=1000)
    >>> tokenizer.train(["Merhaba dünya", "Tokenizer örneği"])
    >>> tokens = tokenizer.encode("Merhaba")
"""

from typing import Dict, List, Tuple, Optional, Set, Union, Callable, Any
from collections import Counter, defaultdict
import re
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer.
    
    BPE algoritması:
    1. Text'i karakterlere böl
    2. En sık görülen character pair'i bul
    3. Bu pair'i tek bir token olarak birleştir
    4. Vocab size'a ulaşana kadar tekrarla
    
    Args:
        vocab_size: Hedef vocabulary boyutu
        special_tokens: Özel tokenler (PAD, UNK, BOS, EOS)
        min_frequency: Merge için minimum frequency
    
    Attributes:
        vocab: Token → ID mapping
        merges: Merge operasyonları listesi
        byte_encoder: Byte → karakter mapping
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self, 
        vocab_size: int = 8000,
        special_tokens: Optional[List[str]] = None,
        min_frequency: int = 2
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        if special_tokens is None:
            special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.special_tokens = special_tokens
        
        # Vocabulary: token -> id
        self.vocab: Dict[str, int] = {}
        
        # Reverse vocabulary: id -> token
        self.id_to_token: Dict[int, str] = {}
        
        # Merge operations: (pair) -> merged_token
        self.merges: List[Tuple[str, str]] = []
        
        # Training durumu
        self.is_trained = False
        
        # Byte-level encoding için
        self.byte_encoder = self._build_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    
    def _build_byte_encoder(self) -> Dict[int, str]:
        """
        Byte-level encoding için mapping oluştur.
        
        Her byte'ı printable bir karaktere map eder.
        GPT-2 yaklaşımı kullanılır.
        
        Returns:
            Dict[int, str]: Byte → karakter mapping
        """
        # Printable ASCII characters
        bs = list(range(ord("!"), ord("~")+1)) + \
             list(range(ord("¡"), ord("¬")+1)) + \
             list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        
        # Non-printable bytes için unique karakterler
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def train(
        self, 
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        BPE tokenizer'ı train et.
        
        Training akışı:
        1. Text'leri byte'lara encode et
        2. Base vocabulary oluştur (her byte bir token)
        3. Iterative olarak en sık pair'leri merge et
        4. Vocab size'a ulaşana kadar devam et
        
        Args:
            texts: Training text listesi
            progress_callback: Progress callback function(current, total)
            
        Raises:
            ValueError: texts boş ise
        """
        if not texts:
            raise ValueError("Training texts boş olamaz")
        
        logger.info(f"BPE training başladı: {len(texts)} doküman, target vocab_size={self.vocab_size}")
        
        # Step 1: Special tokens'ı vocabulary'ye ekle
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.special_tokens)}
        
        # Step 2: Text'leri byte'lara encode et ve word-level split yap
        word_freqs = Counter()
        for text in texts:
            # Kelime bazında tokenize et (space, punctuation ile split)
            words = self._pre_tokenize(text)
            word_freqs.update(words)
        
        logger.info(f"Unique word count: {len(word_freqs)}")
        
        # Step 3: Her kelimeyi character'lere böl
        # splits: {word: [char1, char2, ...]}
        splits = {}
        for word in word_freqs.keys():
            # Byte-level encoding
            byte_encoded = ''.join([self.byte_encoder[b] for b in word.encode('utf-8')])
            # Her karakteri ayır, EOW (end-of-word) ekle
            splits[word] = list(byte_encoded) + ['</w>']
        
        # Step 4: Base vocabulary - tüm unique karakterler
        base_vocab = set()
        for split in splits.values():
            base_vocab.update(split)
        
        for char in sorted(base_vocab):
            if char not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[char] = token_id
                self.id_to_token[token_id] = char
        
        logger.info(f"Base vocabulary size: {len(self.vocab)}")
        
        # Step 5: BPE merges
        num_merges = self.vocab_size - len(self.vocab)
        
        for merge_idx in range(num_merges):
            # En sık görülen pair'i bul
            pairs = self._get_pair_frequencies(splits, word_freqs)
            
            if not pairs:
                logger.warning(f"Merge #{merge_idx}: Daha fazla pair bulunamadı, erken durdu")
                break
            
            # En sık pair
            best_pair = max(pairs, key=lambda p: pairs[p])
            
            if pairs[best_pair] < self.min_frequency:
                logger.info(f"Merge #{merge_idx}: Frequency threshold'a ulaşıldı ({pairs[best_pair]} < {self.min_frequency})")
                break
            
            # Merge işlemi
            merged_token = ''.join(best_pair)
            self.merges.append(best_pair)
            
            # Vocabulary'ye ekle
            if merged_token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[merged_token] = token_id
                self.id_to_token[token_id] = merged_token
            
            # Splits'i güncelle
            splits = self._merge_pair(best_pair, splits)
            
            # Progress callback
            if progress_callback:
                progress_callback(merge_idx + 1, num_merges)
            
            # Log
            if (merge_idx + 1) % 100 == 0:
                logger.info(f"Merge #{merge_idx + 1}/{num_merges}: {best_pair} -> {merged_token} (freq={pairs[best_pair]})")
        
        self.is_trained = True
        logger.info(f"BPE training tamamlandı: final vocab_size={len(self.vocab)}, merges={len(self.merges)}")
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Text'i word-level'da tokenize et.
        
        GPT-2 pattern kullanır: kelime, sayı, punctuation'ı ayırır.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: Word listesi
        """
        # GPT-2 regex pattern
        pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        # Simplified version (Python regex doesn't support \p{L})
        pattern = r"'s|'t|'re|'ve|'m|'ll|'d|\w+|[^\w\s]+|\s+"
        
        words = re.findall(pattern, text)
        return words
    
    def _get_pair_frequencies(
        self, 
        splits: Dict[str, List[str]], 
        word_freqs: Counter
    ) -> Dict[Tuple[str, str], int]:
        """
        Tüm pair'lerin frequency'sini hesapla.
        
        Args:
            splits: {word: [token1, token2, ...]}
            word_freqs: Word frequency'leri
            
        Returns:
            Dict: (token1, token2) -> frequency
        """
        pair_freqs = defaultdict(int)
        
        for word, split in splits.items():
            freq = word_freqs[word]
            
            # Consecutive pairs
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        
        return pair_freqs
    
    def _merge_pair(
        self, 
        pair: Tuple[str, str], 
        splits: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Verilen pair'i tüm splits'de merge et.
        
        Args:
            pair: (token1, token2) tuple
            splits: {word: [token1, token2, ...]}
            
        Returns:
            Dict: Updated splits
        """
        new_splits = {}
        merged = ''.join(pair)
        
        for word, split in splits.items():
            new_split = []
            i = 0
            
            while i < len(split):
                # Pair bulundu mu?
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            
            new_splits[word] = new_split
        
        return new_splits
    
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        **kwargs: Any
    ) -> List[int]:
        """
        Text'i token ID'lerine encode et.
        
        Args:
            text: Input text
            add_bos: BOS token ekle (<BOS>)
            add_eos: EOS token ekle (<EOS>)
            
        Returns:
            List[int]: Token ID listesi
            
        Raises:
            RuntimeError: Tokenizer train edilmemişse
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer henüz train edilmemiş. Önce train() çağırın.")
        
        # Pre-tokenize
        words = self._pre_tokenize(text)
        
        token_ids = []
        if add_bos and "<BOS>" in self.vocab:
            token_ids.append(self.vocab["<BOS>"])
        
        for word in words:
            # Byte-level encoding
            byte_encoded = ''.join([self.byte_encoder[b] for b in word.encode('utf-8')])
            # Character split
            tokens = list(byte_encoded) + ['</w>']
            
            # Apply merges
            for merge in self.merges:
                i = 0
                new_tokens = []
                
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merge:
                        new_tokens.append(''.join(merge))
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                
                tokens = new_tokens
            
            # Convert to IDs
            for token in tokens:
                token_id = self.vocab.get(token, self.vocab.get("<UNK>", 1))
                token_ids.append(token_id)
        
        if add_eos and "<EOS>" in self.vocab:
            token_ids.append(self.vocab["<EOS>"])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Token ID'lerini text'e decode et.
        
        Args:
            token_ids: Token ID listesi
            
        Returns:
            str: Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, "<UNK>")
            tokens.append(token)
        
        # Tokens'ı birleştir
        text = ''.join(tokens)
        
        # </w> işaretlerini space'e çevir
        text = text.replace('</w>', ' ')
        
        # Byte decode
        try:
            # Byte characters'ı geri çevir
            byte_list = [self.byte_decoder[c] for c in text if c in self.byte_decoder]
            text = bytes(byte_list).decode('utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"Decode error: {e}")
        
        return text.strip()
    
    def save_vocab(self, output_path: Path) -> None:
        """
        Vocabulary'yi dosyaya kaydet.
        
        Format: JSON
        - vocab.json: {token: id}
        - merges.txt: merge operations
        
        Args:
            output_path: Output directory
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Vocab
        vocab_file = output_path / "vocab.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Merges
        merges_file = output_path / "merges.txt"
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write(f"#version: {self.VERSION}\n")
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}\n")
        
        # Config
        config_file = output_path / "tokenizer_config.json"
        config = {
            "tokenizer_type": "BPE",
            "version": self.VERSION,
            "vocab_size": len(self.vocab),
            "special_tokens": self.special_tokens,
            "min_frequency": self.min_frequency
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Vocabulary saved to {output_path}")

    def save(self, output_path: Union[str, Path]) -> None:
        """Alias for save_vocab for API compatibility."""
        self.save_vocab(Path(output_path))
    
    def load_vocab(self, input_path: Path) -> None:
        """
        Vocabulary'yi dosyadan yükle.
        
        Args:
            input_path: Input directory
            
        Raises:
            FileNotFoundError: Vocab dosyaları bulunamazsa
        """
        input_path = Path(input_path)
        
        # Vocab
        vocab_file = input_path / "vocab.json"
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        self.id_to_token = {int(idx): token for token, idx in self.vocab.items()}
        
        # Merges
        merges_file = input_path / "merges.txt"
        if merges_file.exists():
            self.merges = []
            with open(merges_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) == 2:
                            self.merges.append((parts[0], parts[1]))
        
        self.is_trained = True
        logger.info(f"Vocabulary loaded from {input_path}: vocab_size={len(self.vocab)}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        """
        Vocabulary ve merge kurallarını dosyadan veya dizinden yükleyerek tokenizer nesnesi döndürür.
        """
        p = Path(path)
        tokenizer = cls()
        if p.is_file():
            p = p.parent
        tokenizer.load_vocab(p)
        return tokenizer

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "BPETokenizer":
        """Alias for load()."""
        return cls.load(path)

    def get_vocab_size(self) -> int:
        """Returns vocabulary size."""
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)
    
    def get_vocab_stats(self) -> Dict[str, Any]:
        """
        Vocabulary istatistikleri.
        
        Returns:
            Dict: İstatistikler
        """
        return {
            "vocab_size": len(self.vocab),
            "num_merges": len(self.merges),
            "special_tokens": self.special_tokens,
            "is_trained": self.is_trained,
            "version": self.VERSION
        }
