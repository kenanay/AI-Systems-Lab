"""
Parola Güvenliği ve Hashleme Modülü

PBKDF2-HMAC-SHA256 algoritması ile FIPS uyumlu, zamanlama saldırılarına
dayanıklı (constant-time) ve harici C bağımlılığı gerektirmeyen güvenli parola yönetimi.

Author: Kenan AY
Version: 1.0.0
"""

import hashlib
import hmac
import secrets
from typing import Tuple

ITERATIONS = 100_000
HASH_NAME = "sha256"
SALT_BYTES = 16


def hash_password(password: str) -> str:
    """
    Düz metin parolayı PBKDF2-HMAC-SHA256 ile hashler.

    Args:
        password: Düz metin kullanıcı parolası.

    Returns:
        Biçimlendirilmiş hash dizgisi: pbkdf2_sha256$<iter>$<salt_hex>$<hash_hex>
    """
    if not isinstance(password, str) or not password:
        raise ValueError("Parola boş bırakılamaz.")

    salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac(
        HASH_NAME,
        password.encode("utf-8"),
        salt,
        ITERATIONS
    )
    return f"pbkdf2_{HASH_NAME}${ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Düz metin parolanın saklanan hash ile eşleştiğini doğrular.

    Args:
        plain_password: Kullanıcının girdiği düz metin parola.
        hashed_password: Veritabanında saklanan biçimlendirilmiş hash dizgisi.

    Returns:
        Eşleşiyorsa True, aksi halde False.
    """
    if not plain_password or not hashed_password:
        return False

    try:
        parts = hashed_password.split("$")
        if len(parts) != 4 or not parts[0].startswith("pbkdf2_"):
            return False

        algo = parts[0].replace("pbkdf2_", "")
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected_hash = parts[3]

        computed_dk = hashlib.pbkdf2_hmac(
            algo,
            plain_password.encode("utf-8"),
            salt,
            iterations
        )
        return hmac.compare_digest(computed_dk.hex(), expected_hash)
    except Exception:
        return False
