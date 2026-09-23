"""
Security Modülü

JWT, Parola Güvenliği, API Key Yönetimi ve Rate Limiting.
"""

from backend.security.password import hash_password, verify_password
from backend.security.jwt import create_access_token, create_refresh_token, decode_token
from backend.security.api_keys import generate_api_key, hash_api_key
from backend.security.rate_limiter import rate_limit, global_rate_limiter, SlidingWindowRateLimiter
from backend.security.dependencies import get_current_user, get_optional_current_user, require_role

__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "generate_api_key",
    "hash_api_key",
    "rate_limit",
    "global_rate_limiter",
    "SlidingWindowRateLimiter",
    "get_current_user",
    "get_optional_current_user",
    "require_role",
]
