"""
API Key Yönetimi ve Güvenliği

Python SDK, harici servisler, CI/CD ve CLI araçları için güvenli API Key üretimi
ve SHA-256 hash doğrulaması.

Author: Kenan AY
Version: 1.0.0
"""

import hashlib
import secrets
from typing import Tuple


def hash_api_key(raw_key: str) -> str:
    """API anahtarının SHA-256 özetini üretir."""
    if not raw_key or not isinstance(raw_key, str):
        raise ValueError("API anahtarı boş olamaz.")
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


def generate_api_key(prefix: str = "sk_live") -> Tuple[str, str, str]:
    """
    Yeni bir güvenli API anahtarı üretir.

    Returns:
        (raw_key, key_prefix, key_hash)
        - raw_key: İstemciye yalnızca bir defa gösterilecek olan tam gizli anahtar.
        - key_prefix: Veritabanında ve UI listelerinde gösterilecek ön ek (örn: sk_live_7a8b...).
        - key_hash: Veritabanında güvenle saklanacak SHA-256 özeti.
    """
    random_part = secrets.token_hex(20)
    raw_key = f"{prefix}_{random_part}"
    key_prefix = f"{raw_key[:12]}..."
    key_hash = hash_api_key(raw_key)
    return raw_key, key_prefix, key_hash
