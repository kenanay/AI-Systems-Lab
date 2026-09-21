"""
src/tokenizer/sentencepiece_tokenizer.py

Production-Grade SentencePiece Tokenizer Implementation

SentencePiece, Google tarafından geliştirilen dil-bağımsız subword tokenizer.
BPE ve Unigram Language Model algoritmalarını destekler.

Features:
- Language-agnostic: Dil-bağımsız, Türkçe için ideal
- Pre-tokenization gerekmez: Whitespace'i token olarak işler
- Reversible: Decode tam olarak orijinal metni döndürür
- Efficient: C++ backend ile hızlı
- Production-ready: Google production sistemlerinde kullanılıyor

Kaynaklar:
    - SentencePiece: A simple and language independent approach to Subword Text Tokenization
    - https://github.com/google/sentencepiece
    - https://arxiv.org/abs/1808.06226

Usage:
    >>> from src.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
    >>> tokenizer = SentencePieceTokenizer()
    >>> tokenizer.train(corpus_path="data/train.txt", vocab_size=8000)
    >>> tokens = tokenizer.encode("Merhaba dünya")
    >>> text = tokenizer.decode(tokens)
"""

import sentencepiece as spm
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """
    SentencePiece tokenizer konfigürasyonu.
    
    Attributes:
        vocab_size: Vocabulary boyutu (special tokens dahil)
        model_type: Algorithm type: 'bpe' veya 'unigram'
        character_coverage: Karakter kapsama oranı (0.9995-1.0 arası)
        normalization_rule_name: Text normalization rule ('nmt_nfkc' önerilen)
        add_dummy_prefix: Sentence başına dummy prefix ekle (space için)
        remove_extra_whitespaces: Fazla whitespace temizle
        max_sentence_length: Maksimum sentence uzunluğu (byte)
        num_threads: Training için thread sayısı
        unk_piece: Unknown token
        bos_piece: Beginning of sequence token
        eos_piece: End of sequence token
        pad_piece: Padding token
        unk_id: Unknown token ID
        bos_id: BOS token ID
        eos_id: EOS token ID
        pad_id: Padding token ID
    """
    vocab_size: int = 8000
    model_type: str = "bpe"  # 'bpe' or 'unigram'
    character_coverage: float = 0.9995
    normalization_rule_name: str = "nmt_nfkc"
    add_dummy_prefix: bool = True
    remove_extra_whitespaces: bool = True
    max_sentence_length: int = 16384
    num_threads: int = 16
    
    # Special tokens
    unk_piece: str = "<unk>"
    bos_piece: str = "<s>"
    eos_piece: str = "</s>"
    pad_piece: str = "<pad>"
    
    # Special token IDs
    unk_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 3
    
    def to_dict(self) -> Dict[str, Union[int, float, str, bool]]:
        """
        Config'i dictionary'ye dönüştür.
        
        Returns:
            Dict içinde config parametreleri
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Union[int, float, str, bool]]) -> 'TokenizerConfig':
        """
        Dictionary'den config oluştur.
        
        Args:
            data: Config parametrelerini içeren dict
            
        Returns:
            TokenizerConfig instance
        """
        return cls(**data)


class SentencePieceTokenizer:
    """
    Production-grade SentencePiece tokenizer.
    
    SentencePiece özellikleri:
    - Dil-bağımsız: Whitespace'i token olarak işler
    - Reversible: Lossless encode/decode
    - Subword: Rare words için etkili
    - Fast: C++ implementation
    
    Args:
        config: TokenizerConfig instance
        model_path: Eğitilmiş model path (optional)
    
    Example:
        >>> # Training
        >>> tokenizer = SentencePieceTokenizer()
        >>> tokenizer.train(
        ...     corpus_path="data/train.txt",
        ...     vocab_size=8000,
        ...     model_prefix="turkish_tokenizer"
        ... )
        >>> 
        >>> # Loading
        >>> tokenizer = SentencePieceTokenizer(model_path="turkish_tokenizer.model")
        >>> 
        >>> # Encoding
        >>> tokens = tokenizer.encode("Kütahya'da eğitim")
        >>> # [1, 234, 45, 67, 89, 2]  # [<s>, Kü, tah, ya, 'da, eğitim, </s>]
        >>> 
        >>> # Decoding
        >>> text = tokenizer.decode(tokens)
        >>> # "Kütahya'da eğitim"
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        config: Optional[TokenizerConfig] = None,
        model_path: Optional[str] = None
    ):
        """
        Tokenizer initialize.
        
        Args:
            config: TokenizerConfig instance (training için gerekli)
            model_path: Eğitilmiş model path (inference için gerekli)
        """
        self.config = config or TokenizerConfig()
        self.model_path = model_path
        self.sp_model = None
        
        # Eğer model path verilmişse, yükle
        if model_path and Path(model_path).exists():
            self.load(model_path)
            logger.info(f"Tokenizer loaded from {model_path}")
    
    def train(
        self,
        corpus_path: str,
        vocab_size: Optional[int] = None,
        model_prefix: str = "tokenizer",
        model_type: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        SentencePiece model'i train et.
        
        Args:
            corpus_path: Training corpus dosya path
            vocab_size: Vocabulary size (None ise config'den alınır)
            model_prefix: Output model dosya prefix
            model_type: 'bpe' veya 'unigram' (None ise config'den alınır)
            
        Returns:
            Tuple[str, Dict]: (model_path, training_stats)
            
        Raises:
            FileNotFoundError: Corpus dosyası bulunamazsa
            ValueError: Invalid parameters
            
        Example:
            >>> tokenizer = SentencePieceTokenizer()
            >>> model_path, stats = tokenizer.train(
            ...     corpus_path="data/turkish_corpus.txt",
            ...     vocab_size=8000,
            ...     model_prefix="turkish_tokenizer"
            ... )
            >>> print(f"Model: {model_path}, Vocab: {stats['vocab_size']}")
        """
        # Corpus kontrolü
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus dosyası bulunamadı: {corpus_path}")
        
        # Config güncelle
        if vocab_size:
            self.config.vocab_size = vocab_size
        if model_type:
            self.config.model_type = model_type
        
        logger.info(f"SentencePiece training başlıyor...")
        logger.info(f"  Corpus: {corpus_path}")
        logger.info(f"  Vocab size: {self.config.vocab_size}")
        logger.info(f"  Model type: {self.config.model_type}")
        logger.info(f"  Character coverage: {self.config.character_coverage}")
        
        # Training parametreleri
        train_params = {
            'input': str(corpus_path),
            'model_prefix': model_prefix,
            'vocab_size': self.config.vocab_size,
            'model_type': self.config.model_type,
            'character_coverage': self.config.character_coverage,
            'normalization_rule_name': self.config.normalization_rule_name,
            'add_dummy_prefix': self.config.add_dummy_prefix,
            'remove_extra_whitespaces': self.config.remove_extra_whitespaces,
            'max_sentence_length': self.config.max_sentence_length,
            'num_threads': self.config.num_threads,
            'unk_piece': self.config.unk_piece,
            'bos_piece': self.config.bos_piece,
            'eos_piece': self.config.eos_piece,
            'pad_piece': self.config.pad_piece,
            'unk_id': self.config.unk_id,
            'bos_id': self.config.bos_id,
            'eos_id': self.config.eos_id,
            'pad_id': self.config.pad_id,
        }
        
        # Train
        spm.SentencePieceTrainer.train(**train_params)
        
        model_path = f"{model_prefix}.model"
        vocab_path = f"{model_prefix}.vocab"
        
        logger.info(f"✓ Training tamamlandı: {model_path}")
        
        # Model'i yükle
        self.load(model_path)
        
        # Training stats
        stats = {
            'vocab_size': self.vocab_size,
            'model_type': self.config.model_type,
            'model_path': model_path,
            'vocab_path': vocab_path,
            'character_coverage': self.config.character_coverage,
            'version': self.VERSION
        }
        
        # Config'i kaydet
        config_path = f"{model_prefix}.config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Config saved: {config_path}")
        logger.info(f"  Vocab size: {stats['vocab_size']}")
        
        return model_path, stats
    
    def load(self, model_path: str):
        """
        Eğitilmiş model'i yükle.
        
        Args:
            model_path: .model dosya path
            
        Raises:
            FileNotFoundError: Model dosyası bulunamazsa
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(str(model_path))
        self.model_path = str(model_path)
        
        # Config varsa yükle
        config_path = model_path.with_suffix('.config.json')
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                self.config = TokenizerConfig.from_dict(config_dict)
        
        logger.info(f"Model loaded: {model_path} (vocab_size={self.vocab_size})")
    
    def save(self, output_dir: str, prefix: str = "tokenizer"):
        """
        Model ve config'i kaydet.
        
        Args:
            output_dir: Output directory
            prefix: Dosya prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model_path:
            # Model dosyasını kopyala
            import shutil
            src = Path(self.model_path)
            dst = output_dir / f"{prefix}.model"
            shutil.copy(src, dst)
            
            # Vocab dosyası varsa kopyala
            vocab_src = src.with_suffix('.vocab')
            if vocab_src.exists():
                vocab_dst = output_dir / f"{prefix}.vocab"
                shutil.copy(vocab_src, vocab_dst)
        
        # Config kaydet
        config_path = output_dir / f"{prefix}.config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Tokenizer saved to {output_dir}/{prefix}.*")
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        out_type: str = "id"
    ) -> Union[List[int], List[str]]:
        """
        Text'i token'lara encode et.
        
        Args:
            text: Input text
            add_bos: BOS token ekle
            add_eos: EOS token ekle
            out_type: 'id' (int) veya 'piece' (str)
            
        Returns:
            Token ID listesi veya token piece listesi
            
        Example:
            >>> tokens = tokenizer.encode("Merhaba dünya")
            >>> # [1, 234, 45, 67, 2]  # [<s>, Mer, haba, dünya, </s>]
            >>> 
            >>> pieces = tokenizer.encode("Merhaba dünya", out_type="piece")
            >>> # ['<s>', '▁Mer', 'haba', '▁dünya', '</s>']
        """
        if self.sp_model is None:
            raise RuntimeError("Model yüklenmemiş. train() veya load() çağır.")
        
        if out_type == "id":
            tokens = self.sp_model.encode(text, out_type=int)
        elif out_type == "piece":
            tokens = self.sp_model.encode(text, out_type=str)
        else:
            raise ValueError(f"Invalid out_type: {out_type}. 'id' veya 'piece' olmalı.")
        
        # BOS/EOS ekleme (sadece ID mode'da)
        if out_type == "id":
            if add_bos and (not tokens or tokens[0] != self.config.bos_id):
                tokens = [self.config.bos_id] + tokens
            if add_eos and (not tokens or tokens[-1] != self.config.eos_id):
                tokens = tokens + [self.config.eos_id]
        
        return tokens
    
    def decode(
        self,
        tokens: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Token ID'leri text'e decode et.
        
        Args:
            tokens: Token ID listesi
            skip_special_tokens: Special token'ları skip et
            
        Returns:
            Decoded text
            
        Example:
            >>> text = tokenizer.decode([1, 234, 45, 67, 2])
            >>> # "Merhaba dünya"
        """
        if self.sp_model is None:
            raise RuntimeError("Model yüklenmemiş. train() veya load() çağır.")
        
        # Special token'ları filtrele
        if skip_special_tokens:
            special_ids = {
                self.config.unk_id,
                self.config.bos_id,
                self.config.eos_id,
                self.config.pad_id
            }
            tokens = [t for t in tokens if t not in special_ids]
        
        text = self.sp_model.decode(tokens)
        return text
    
    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = True,
        add_eos: bool = True
    ) -> List[List[int]]:
        """
        Batch text encode.
        
        Args:
            texts: Text listesi
            add_bos: BOS token ekle
            add_eos: EOS token ekle
            
        Returns:
            Token ID listelerinin listesi
        """
        return [self.encode(text, add_bos, add_eos) for text in texts]
    
    def decode_batch(
        self,
        token_lists: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Batch token decode.
        
        Args:
            token_lists: Token ID listelerinin listesi
            skip_special_tokens: Special token'ları skip et
            
        Returns:
            Decoded text listesi
        """
        return [self.decode(tokens, skip_special_tokens) for tokens in token_lists]
    
    def id_to_piece(self, token_id: int) -> str:
        """Token ID'yi piece string'e çevir."""
        if self.sp_model is None:
            raise RuntimeError("Model yüklenmemiş.")
        return self.sp_model.id_to_piece(token_id)
    
    def piece_to_id(self, piece: str) -> int:
        """Piece string'i token ID'ye çevir."""
        if self.sp_model is None:
            raise RuntimeError("Model yüklenmemiş.")
        return self.sp_model.piece_to_id(piece)
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary boyutu."""
        if self.sp_model is None:
            return self.config.vocab_size
        return self.sp_model.get_piece_size()
    
    @property
    def bos_id(self) -> int:
        """BOS token ID."""
        return self.config.bos_id
    
    @property
    def eos_id(self) -> int:
        """EOS token ID."""
        return self.config.eos_id
    
    @property
    def pad_id(self) -> int:
        """Padding token ID."""
        return self.config.pad_id
    
    @property
    def unk_id(self) -> int:
        """Unknown token ID."""
        return self.config.unk_id
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Vocabulary dict'i al.
        
        Returns:
            {piece: id} dictionary
        """
        if self.sp_model is None:
            raise RuntimeError("Model yüklenmemiş.")
        
        vocab = {}
        for i in range(self.vocab_size):
            piece = self.sp_model.id_to_piece(i)
            vocab[piece] = i
        
        return vocab
    
    def get_vocab_list(self) -> List[str]:
        """
        Vocabulary listesi al (ID sırasıyla).
        
        Returns:
            Token piece listesi
        """
        if self.sp_model is None:
            raise RuntimeError("Model yüklenmemiş.")
        
        return [self.sp_model.id_to_piece(i) for i in range(self.vocab_size)]
    
    def tokenize_stats(self, text: str) -> Dict:
        """
        Text tokenization istatistikleri.
        
        Args:
            text: Input text
            
        Returns:
            Dict ile stats: chars, tokens, compression_ratio, avg_token_len
        """
        tokens = self.encode(text, add_bos=False, add_eos=False)
        pieces = self.encode(text, add_bos=False, add_eos=False, out_type="piece")
        
        char_count = len(text)
        token_count = len(tokens)
        
        # Compression ratio: kaç karakter 1 token'a sıkıştırılıyor
        compression_ratio = char_count / token_count if token_count > 0 else 0
        
        # Average token length (chars)
        avg_token_len = sum(len(p.replace('▁', '')) for p in pieces) / token_count if token_count > 0 else 0
        
        return {
            'char_count': char_count,
            'token_count': token_count,
            'compression_ratio': compression_ratio,
            'avg_token_len': avg_token_len,
            'sample_tokens': tokens[:10],
            'sample_pieces': pieces[:10]
        }
    
    def __repr__(self) -> str:
        """String representation."""
        model_status = "loaded" if self.sp_model else "not loaded"
        return (
            f"SentencePieceTokenizer(vocab_size={self.vocab_size}, "
            f"model_type={self.config.model_type}, status={model_status})"
        )


def main() -> None:
    """
    Demo ve test fonksiyonu.
    
    SentencePiece tokenizer'ın temel özelliklerini gösterir:
    - Corpus oluşturma
    - Model training
    - Encoding/decoding
    - Tokenization istatistikleri
    - Vocabulary inceleme
    """
    # Logger setup
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 80)
    print("SentencePiece Tokenizer Demo")
    print("=" * 80)
    
    # Sample corpus oluştur
    print("\n1. Sample corpus oluşturuluyor...")
    corpus_path = Path("sample_corpus.txt")
    
    sample_texts = [
        "Kütahya'da eğitim ve öğretim faaliyetleri devam ediyor.",
        "Türkiye'nin başkenti Ankara'dır.",
        "Yapay zeka ve makine öğrenmesi günümüzde çok önemli.",
        "SentencePiece dil-bağımsız bir tokenizer'dır.",
        "Transformer mimarisi self-attention kullanır.",
        "Model eğitimi için büyük veri setleri gereklidir.",
        "GPU'lar derin öğrenme için optimize edilmiştir.",
        "Tokenization doğal dil işlemede temel bir adımdır.",
    ] * 100  # 800 cümle
    
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sample_texts))
    
    print(f"  ✓ Corpus oluşturuldu: {len(sample_texts)} satır")
    
    # Tokenizer oluştur ve train et
    print("\n2. Tokenizer training...")
    tokenizer = SentencePieceTokenizer(
        config=TokenizerConfig(
            vocab_size=500,  # Sample corpus için küçük vocab
            model_type='bpe',
            character_coverage=0.9995
        )
    )
    
    model_path, stats = tokenizer.train(
        corpus_path=str(corpus_path),
        model_prefix="demo_tokenizer"
    )
    
    print(f"\n  Training Stats:")
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    # Encoding test
    print("\n3. Encoding test...")
    test_text = "Kütahya'da yapay zeka laboratuvarı"
    tokens = tokenizer.encode(test_text)
    pieces = tokenizer.encode(test_text, out_type="piece")
    
    print(f"  Text: {test_text}")
    print(f"  Tokens: {tokens}")
    print(f"  Pieces: {pieces}")
    
    # Decoding test
    print("\n4. Decoding test...")
    decoded = tokenizer.decode(tokens)
    print(f"  Decoded: {decoded}")
    print(f"  Match: {decoded.strip() == test_text}")
    
    # Stats
    print("\n5. Tokenization stats...")
    stats = tokenizer.tokenize_stats(test_text)
    print(f"  Characters: {stats['char_count']}")
    print(f"  Tokens: {stats['token_count']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f} chars/token")
    print(f"  Avg token length: {stats['avg_token_len']:.2f} chars")
    
    # Vocab sample
    print("\n6. Vocabulary sample...")
    vocab_list = tokenizer.get_vocab_list()
    print(f"  Total vocab: {len(vocab_list)} tokens")
    print(f"  First 20: {vocab_list[:20]}")
    
    # Special tokens
    print("\n7. Special tokens...")
    print(f"  BOS: {tokenizer.bos_id} = '{tokenizer.id_to_piece(tokenizer.bos_id)}'")
    print(f"  EOS: {tokenizer.eos_id} = '{tokenizer.id_to_piece(tokenizer.eos_id)}'")
    print(f"  PAD: {tokenizer.pad_id} = '{tokenizer.id_to_piece(tokenizer.pad_id)}'")
    print(f"  UNK: {tokenizer.unk_id} = '{tokenizer.id_to_piece(tokenizer.unk_id)}'")
    
    # Cleanup
    print("\n8. Cleanup...")
    corpus_path.unlink()
    Path("demo_tokenizer.model").unlink()
    Path("demo_tokenizer.vocab").unlink()
    Path("demo_tokenizer.config.json").unlink()
    print("  ✓ Temporary files removed")
    
    print("\n" + "=" * 80)
    print("✓ Demo tamamlandı!")
    print("=" * 80)


if __name__ == "__main__":
    main()
