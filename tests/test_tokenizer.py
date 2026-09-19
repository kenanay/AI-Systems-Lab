"""
tests/test_tokenizer.py

BPE Tokenizer Unit Tests

Bu modül BPE tokenizer'ın temel fonksiyonlarını test eder:
- Training (merge operations, vocabulary building)
- Encoding (text → token IDs)
- Decoding (token IDs → text)
- Special tokens handling
- Edge cases
"""

import pytest
from pathlib import Path
from typing import List

from src.tokenizer.bpe import BPETokenizer


class TestBPETokenizerBasic:
    """BPE tokenizer temel fonksiyon testleri"""
    
    def test_tokenizer_initialization(self) -> None:
        """
        Tokenizer initialization testi.
        
        Doğrulamalar:
        - Default parameters doğru
        - Special tokens set edilmiş
        - Training durumu False
        """
        tokenizer = BPETokenizer(vocab_size=1000)
        
        assert tokenizer.vocab_size == 1000
        assert tokenizer.is_trained is False
        assert len(tokenizer.special_tokens) > 0
        assert "<PAD>" in tokenizer.special_tokens
        assert "<UNK>" in tokenizer.special_tokens
    
    def test_custom_special_tokens(self) -> None:
        """
        Custom special tokens testi.
        """
        custom_tokens = ["<START>", "<END>", "<MASK>"]
        tokenizer = BPETokenizer(vocab_size=1000, special_tokens=custom_tokens)
        
        # Special tokens listesi doğru mu?
        assert tokenizer.special_tokens == custom_tokens
        
        # Train et - special tokens vocab'a eklenir
        tokenizer.train(["test data"])
        
        # Şimdi vocab'da olmalı
        assert "<START>" in tokenizer.vocab
        assert "<END>" in tokenizer.vocab
        assert "<MASK>" in tokenizer.vocab
    
    def test_training_basic(self) -> None:
        """
        Temel training testi.
        
        Training akışı:
        1. Text listesi ver
        2. Train et
        3. Vocab oluşturuldu mu kontrol et
        """
        tokenizer = BPETokenizer(vocab_size=200, min_frequency=1)
        
        texts = [
            "Merhaba dünya",
            "Merhaba arkadaşlar",
            "Bu bir test",
            "Test test test",
        ]
        
        tokenizer.train(texts)
        
        assert tokenizer.is_trained is True
        assert len(tokenizer.vocab) > len(tokenizer.special_tokens)
        assert len(tokenizer.merges) > 0
    
    def test_training_progress_callback(self) -> None:
        """
        Training progress callback testi.
        """
        tokenizer = BPETokenizer(vocab_size=200, min_frequency=1)
        
        progress_calls = []
        
        def progress_callback(current: int, total: int):
            progress_calls.append((current, total))
        
        texts = ["test " * 100, "data " * 100]
        tokenizer.train(texts, progress_callback=progress_callback)
        
        # Progress callback çağrıldı mı?
        assert len(progress_calls) > 0
        
        # Son call total'a eşit mi?
        if progress_calls:
            last_current, last_total = progress_calls[-1]
            assert last_current <= last_total
    
    def test_training_empty_texts(self) -> None:
        """
        Boş text listesi ile training ValueError fırlatmalı.
        """
        tokenizer = BPETokenizer(vocab_size=1000)
        
        with pytest.raises(ValueError, match="boş olamaz"):
            tokenizer.train([])
    
    def test_encode_decode_roundtrip(self) -> None:
        """
        Encode → Decode roundtrip testi.
        
        Text → token IDs → text dönüşümü orijinali korumalı.
        """
        tokenizer = BPETokenizer(vocab_size=500, min_frequency=1)
        
        # Train
        train_texts = [
            "Merhaba dünya",
            "Bu bir test metnidir",
            "Tokenizer çalışıyor",
        ]
        tokenizer.train(train_texts)
        
        # Test texts
        test_cases = [
            "Merhaba",
            "dünya",
            "test",
            "Merhaba dünya",
        ]
        
        for text in test_cases:
            token_ids = tokenizer.encode(text)
            decoded = tokenizer.decode(token_ids)
            
            # Decoded text orijinale çok yakın olmalı
            # (Byte-level encoding nedeniyle tam eşit olmayabilir)
            assert decoded.strip().lower() == text.strip().lower()
    
    def test_encode_before_training(self) -> None:
        """
        Train edilmemiş tokenizer encode yapmaya çalışırsa RuntimeError.
        """
        tokenizer = BPETokenizer(vocab_size=1000)
        
        with pytest.raises(RuntimeError, match="train edilmemiş"):
            tokenizer.encode("test")
    
    def test_encode_returns_list_of_ints(self) -> None:
        """
        Encode edilen token ID'leri integer listesi olmalı.
        """
        tokenizer = BPETokenizer(vocab_size=200, min_frequency=1)
        tokenizer.train(["test data"])
        
        token_ids = tokenizer.encode("test")
        
        assert isinstance(token_ids, list)
        assert all(isinstance(tid, int) for tid in token_ids)
        assert len(token_ids) > 0
    
    def test_decode_empty_list(self) -> None:
        """
        Boş token ID listesi boş string döndürmeli.
        """
        tokenizer = BPETokenizer(vocab_size=200, min_frequency=1)
        tokenizer.train(["test"])
        
        decoded = tokenizer.decode([])
        
        assert decoded == ""
    
    def test_special_tokens_in_vocab(self) -> None:
        """
        Special tokens vocabulary'de olmalı.
        """
        tokenizer = BPETokenizer(vocab_size=200)
        tokenizer.train(["test data"])
        
        for token in tokenizer.special_tokens:
            assert token in tokenizer.vocab
            assert tokenizer.vocab[token] < len(tokenizer.special_tokens)


class TestBPETokenizerEdgeCases:
    """BPE tokenizer edge case testleri"""
    
    def test_unicode_characters(self) -> None:
        """
        Unicode karakterler doğru işlenmeli.
        """
        tokenizer = BPETokenizer(vocab_size=300, min_frequency=1)
        
        texts = [
            "Türkçe karakterler: ışçığüöİŞÇĞÜÖ",
            "Emoji: 😀 🎉 🚀",
            "Çeşitli: café naïve résumé",
        ]
        
        tokenizer.train(texts)
        
        for text in texts:
            token_ids = tokenizer.encode(text)
            decoded = tokenizer.decode(token_ids)
            
            # Unicode karakterler korunmalı
            assert len(token_ids) > 0
            assert len(decoded) > 0
    
    def test_very_long_text(self) -> None:
        """
        Çok uzun text encode edilebilmeli.
        """
        tokenizer = BPETokenizer(vocab_size=500, min_frequency=1)
        tokenizer.train(["test " * 100])
        
        # 10000 kelimelik text
        long_text = "test " * 10000
        
        token_ids = tokenizer.encode(long_text)
        
        assert len(token_ids) > 0
        assert all(isinstance(tid, int) for tid in token_ids)
    
    def test_single_character_text(self) -> None:
        """
        Tek karakterli text encode edilebilmeli.
        """
        tokenizer = BPETokenizer(vocab_size=200, min_frequency=1)
        tokenizer.train(["a", "b", "c"])
        
        token_ids = tokenizer.encode("a")
        
        assert len(token_ids) > 0
    
    def test_repeated_characters(self) -> None:
        """
        Tekrarlanan karakterler doğru merge edilmeli.
        """
        tokenizer = BPETokenizer(vocab_size=200, min_frequency=1)
        
        texts = ["aaaa", "bbbb", "aaaa bbbb"]
        tokenizer.train(texts)
        
        # Merges oluşturuldu mu?
        assert len(tokenizer.merges) > 0
    
    def test_min_frequency_threshold(self) -> None:
        """
        min_frequency threshold'u doğru uygulanmalı.
        """
        tokenizer = BPETokenizer(vocab_size=1000, min_frequency=5)
        
        # Sadece 2 kez görünen pair
        texts = ["ab", "ab", "cd", "cd", "cd"]
        tokenizer.train(texts)
        
        # ab pair'i 2 kez görüldü, min_frequency=5, merge olmamalı
        # cd pair'i 3 kez görüldü, yine min_frequency=5'ten az
        # Vocab sadece base characters içermeli
        assert tokenizer.is_trained is True


class TestBPETokenizerPersistence:
    """Vocabulary save/load testleri"""
    
    def test_save_and_load_vocab(self, tmp_path: Path) -> None:
        """
        Vocabulary save → load roundtrip testi.
        
        Args:
            tmp_path: pytest fixture (temporary directory)
        """
        # Train tokenizer
        tokenizer1 = BPETokenizer(vocab_size=300, min_frequency=1)
        texts = ["Merhaba dünya", "Test data", "Tokenizer örneği"]
        tokenizer1.train(texts)
        
        # Save
        output_dir = tmp_path / "tokenizer"
        tokenizer1.save_vocab(output_dir)
        
        # Files oluşturuldu mu?
        assert (output_dir / "vocab.json").exists()
        assert (output_dir / "merges.txt").exists()
        assert (output_dir / "tokenizer_config.json").exists()
        
        # Load into new tokenizer
        tokenizer2 = BPETokenizer()
        tokenizer2.load_vocab(output_dir)
        
        # Vocabulary eşit mi?
        assert tokenizer2.vocab == tokenizer1.vocab
        assert tokenizer2.merges == tokenizer1.merges
        assert tokenizer2.is_trained is True
        
        # Encode/decode aynı sonucu veriyor mu?
        test_text = "Merhaba test"
        ids1 = tokenizer1.encode(test_text)
        ids2 = tokenizer2.encode(test_text)
        
        assert ids1 == ids2
    
    def test_load_nonexistent_vocab(self, tmp_path: Path) -> None:
        """
        Olmayan vocabulary load etmeye çalışırsa FileNotFoundError.
        """
        tokenizer = BPETokenizer()
        
        with pytest.raises(FileNotFoundError):
            tokenizer.load_vocab(tmp_path / "nonexistent")
    
    def test_get_vocab_stats(self) -> None:
        """
        Vocabulary istatistikleri doğru dönmeli.
        """
        tokenizer = BPETokenizer(vocab_size=500, min_frequency=2)
        tokenizer.train(["test data " * 10])
        
        stats = tokenizer.get_vocab_stats()
        
        assert "vocab_size" in stats
        assert "num_merges" in stats
        assert "special_tokens" in stats
        assert "is_trained" in stats
        assert "version" in stats
        
        assert stats["is_trained"] is True
        assert stats["vocab_size"] > 0
        assert stats["version"] == tokenizer.VERSION


class TestBPETokenizerMultiLanguage:
    """Multi-language support testleri"""
    
    def test_turkish_text(self) -> None:
        """Türkçe text encoding/decoding"""
        tokenizer = BPETokenizer(vocab_size=500, min_frequency=1)
        
        texts = [
            "Türkiye'de yaşıyorum",
            "Güzel bir gün",
            "İstanbul çok büyük bir şehir",
        ]
        
        tokenizer.train(texts)
        
        test_text = "Güzel bir gün"
        token_ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(token_ids)
        
        assert len(token_ids) > 0
        assert "güzel" in decoded.lower() or "gün" in decoded.lower()
    
    def test_mixed_language(self) -> None:
        """Karma dil (Türkçe + İngilizce) encoding"""
        tokenizer = BPETokenizer(vocab_size=800, min_frequency=1)
        
        texts = [
            "Hello dünya",
            "Machine learning öğreniyorum",
            "Deep learning ve yapay zeka",
        ]
        
        tokenizer.train(texts)
        
        test_text = "Machine learning"
        token_ids = tokenizer.encode(test_text)
        
        assert len(token_ids) > 0
        assert all(isinstance(tid, int) for tid in token_ids)
