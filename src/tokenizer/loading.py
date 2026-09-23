"""Strict shared tokenizer loading and content fingerprints."""
from pathlib import Path
import hashlib

def artifact_hash(path):
    path = Path(path)
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for item in files:
        if path.is_dir():
            digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()

def load_tokenizer(path):
    path = Path(path)
    if path.is_dir() or path.suffix == ".json":
        from src.tokenizer.bpe import BPETokenizer
        return BPETokenizer.load(path)
    from src.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
    tok = SentencePieceTokenizer()
    tok.load(str(path))
    return tok
