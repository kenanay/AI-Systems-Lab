"""
JWT (JSON Web Token) Güvenlik Modülü

RFC 7519 uyumlu HS256 HMAC-SHA256 token oluşturma, imzalama ve doğrulama.
PyJWT mevcutsa kullanır, aksi takdirde standart kütüphane tabanlı temiz
ve sıfır bağımlılıklı fallback uygular.

Author: Kenan AY
Version: 1.0.0
"""

import base64
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from backend.config import settings

ALGORITHM = "HS256"
DEFAULT_ACCESS_EXPIRE_MINUTES = 60
DEFAULT_REFRESH_EXPIRE_DAYS = 14


def _b64url_encode(data: bytes) -> str:
    """URL-safe base64 encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(s: str) -> bytes:
    """URL-safe base64 decode with auto-padding."""
    padding = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


def create_token(
    data: Dict[str, Any],
    expires_delta: timedelta,
    token_type: str = "access",
    secret_key: Optional[str] = None
) -> str:
    """
    Belirtilen payload ve geçerlilik süresiyle JWT token üretir.
    """
    secret = secret_key or settings.jwt_secret_key or settings.secret_key
    to_encode = data.copy()

    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    to_encode.update({
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": uuid.uuid4().hex,
        "type": token_type
    })

    header = {"alg": ALGORITHM, "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(to_encode, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    secret_key: Optional[str] = None
) -> str:
    """Kısa ömürlü Access Token oluşturur."""
    delta = expires_delta or timedelta(minutes=DEFAULT_ACCESS_EXPIRE_MINUTES)
    return create_token(data, delta, token_type="access", secret_key=secret_key)


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    secret_key: Optional[str] = None
) -> str:
    """Uzun ömürlü Refresh Token oluşturur."""
    delta = expires_delta or timedelta(days=DEFAULT_REFRESH_EXPIRE_DAYS)
    return create_token(data, delta, token_type="refresh", secret_key=secret_key)


def decode_token(token: str, secret_key: Optional[str] = None, verify_exp: bool = True) -> Dict[str, Any]:
    """
    JWT token'ı çözer, imzasını doğrular ve süresini kontrol eder.

    Raises:
        ValueError: Token geçersiz, manipüle edilmiş veya süresi dolmuşsa.
    """
    secret = secret_key or settings.jwt_secret_key or settings.secret_key
    if not token or not isinstance(token, str):
        raise ValueError("Geçersiz token formatı.")

    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("JWT token 3 parçadan (header.payload.sig) oluşmalıdır.")

    header_b64, payload_b64, sig_b64 = parts

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    expected_sig_b64 = _b64url_encode(expected_sig)

    if not hmac.compare_digest(expected_sig_b64, sig_b64):
        raise ValueError("Token imzası doğrulanamadı (geçersiz gizli anahtar veya manipülasyon).")

    try:
        payload_json = _b64url_decode(payload_b64).decode("utf-8")
        payload = json.loads(payload_json)
    except Exception as e:
        raise ValueError(f"Token içeriği çözülemedi: {str(e)}")

    try:
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg") != ALGORITHM or header.get("typ") != "JWT":
            raise ValueError("Unsupported token header")
        if not payload.get("sub") or not isinstance(payload.get("exp"), (int, float)):
            raise ValueError("Token requires sub and exp")
    except (TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid token") from exc

    if verify_exp:
        now_ts = int(time.time())
        if now_ts >= payload["exp"]:
            raise ValueError("Token kullanım süresi dolmuş (expired).")

    return payload
