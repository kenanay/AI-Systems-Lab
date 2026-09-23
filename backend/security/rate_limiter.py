"""
Sliding Window (Kayan Pencere) API Rate Limiter

DoS saldırılarını ve kaba kuvvet (brute-force) denemelerini engellemek amacıyla
IP veya kullanıcı bazlı bellek içi (in-memory) istek sınırlandırma sistemi.

Author: Kenan AY
Version: 1.0.0
"""

import math
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Optional
from fastapi import HTTPException, Request, status


class SlidingWindowRateLimiter:
    """Thread-safe kayan pencere algoritması ile istek sınırlayıcı."""

    def __init__(self):
        self._records: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> tuple[bool, int, int]:
        """
        Verilen anahtarın istek hakkı olup olmadığını kontrol eder.

        Returns:
            (allowed: bool, remaining: int, retry_after: int)
        """
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            queue = self._records[key]

            # Süresi dolmuş eski zaman damgalarını temizle
            while queue and queue[0] < cutoff:
                queue.popleft()

            # Kullanım kotası dolduysa
            if len(queue) >= max_requests:
                oldest = queue[0]
                retry_after = max(1, math.ceil(oldest + window_seconds - now))
                return False, 0, retry_after

            # Yeni isteği kaydet
            queue.append(now)
            remaining = max_requests - len(queue)
            return True, remaining, 0

    def cleanup(self, max_idle_seconds: int = 3600) -> None:
        """Kullanılmayan eski anahtarları bellekten temizler."""
        now = time.time()
        with self._lock:
            stale_keys = [
                k for k, q in self._records.items()
                if not q or (now - q[-1] > max_idle_seconds)
            ]
            for k in stale_keys:
                del self._records[k]

    def reset(self) -> None:
        """Tüm kayıtları sıfırlar (özellikle birim testler için)."""
        with self._lock:
            self._records.clear()


# Global tekil rate limiter örneği
global_rate_limiter = SlidingWindowRateLimiter()


def get_client_identifier(request: Request) -> str:
    """İstemci IP adresini veya tekil kimliğini çıkarır."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    elif request.client and request.client.host:
        client_ip = request.client.host
    else:
        client_ip = "127.0.0.1"

    path = request.url.path
    return f"{client_ip}:{path}"


def rate_limit(
    max_requests: int = 60,
    window_seconds: int = 60,
    key_func: Optional[Callable[[Request], str]] = None
):
    """
    FastAPI endpoint bağımlılığı olarak kullanılacak Rate Limit fabrikası.

    Örnek:
        @router.post("/login", dependencies=[Depends(rate_limit(max_requests=10, window_seconds=60))])
    """
    def dependency(request: Request):
        from backend.config import settings
        # Rate limit konfigürasyonla devre dışı bırakılmışsa atla
        if not getattr(settings, "rate_limit_enabled", True):
            return

        identifier = key_func(request) if key_func else get_client_identifier(request)
        allowed, remaining, retry_after = global_rate_limiter.is_allowed(
            identifier, max_requests, window_seconds
        )

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Çok fazla istek gönderildi. Lütfen {retry_after} saniye sonra tekrar deneyin.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + retry_after)),
                }
            )

    return dependency
