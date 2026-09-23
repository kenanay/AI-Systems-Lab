"""Stable content-based splits shared by tokenizer training and compilation."""
import hashlib
import re
import unicodedata

def content_hash(text):
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()

def split_for_text(text):
    bucket = int(content_hash(text)[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"
