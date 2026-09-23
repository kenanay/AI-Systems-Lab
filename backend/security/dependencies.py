"""
FastAPI Güvenlik ve Kimlik Doğrulama Bağımlılıkları

JWT Bearer ve API Key kimlik doğrulaması, rol tabanlı erişim kontrolü (RBAC).

Author: Kenan AY
Version: 1.0.0
"""

from typing import List, Optional, Union
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Header, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import UserRecord, APIKeyRecord
from backend.security.jwt import decode_token
from backend.security.api_keys import hash_api_key

# HTTP Bearer şeması (auto_error=False sayesinde hem opsiyonel hem zorunlu auth'u destekler)
bearer_scheme = HTTPBearer(auto_error=False)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_current_user(
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db)
) -> UserRecord:
    """
    İsteği gönderen geçerli kullanıcıyı doğrular ve döndürür.
    Bearer JWT veya API Key (X-API-Key / Bearer sk_live_...) kabul eder.

    Raises:
        HTTPException 401: Kimlik bilgisi eksik, hatalı veya süresi dolmuşsa.
    """
    token_str: Optional[str] = None

    if auth_header and auth_header.credentials:
        token_str = auth_header.credentials.strip()
    elif x_api_key:
        token_str = x_api_key.strip()

    if not token_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kimlik doğrulaması gerekli. Lütfen Bearer token veya X-API-Key sağlayın.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # 1. API Key Denetimi (sk_live_ ile başlıyorsa)
    if token_str.startswith("sk_live_"):
        key_hash = hash_api_key(token_str)
        api_key_rec = db.query(APIKeyRecord).filter(
            APIKeyRecord.key_hash == key_hash,
            APIKeyRecord.is_active == True
        ).first()

        if not api_key_rec:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Geçersiz veya iptal edilmiş API anahtarı.",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # Son kullanım zamanını güncelle
        api_key_rec.last_used_at = _utc_now()
        db.commit()

        # Anahtara bağlı kullanıcıyı al
        user = db.query(UserRecord).filter(UserRecord.user_id == api_key_rec.user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API anahtarına ait kullanıcı hesabı aktif değil.",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return user

    # 2. Standart JWT Token Denetimi
    try:
        payload = decode_token(token_str)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Geçersiz veya süresi dolmuş oturum token'ı: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token içinde geçerli bir kullanıcı kimliği (sub) bulunamadı.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = db.query(UserRecord).filter(UserRecord.user_id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı hesabı bulunamadı.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı hesabı devre dışı bırakılmış.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return user


def get_optional_current_user(
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db)
) -> Optional[UserRecord]:
    """
    Opsiyonel kimlik doğrulama.
    Kimlik bilgisi verilmişse doğrular ve kullanıcıyı döner, verilmemişse None döner.
    """
    token_str = None
    if auth_header and auth_header.credentials:
        token_str = auth_header.credentials.strip()
    elif x_api_key:
        token_str = x_api_key.strip()

    if not token_str:
        return None

    try:
        return get_current_user(auth_header=auth_header, x_api_key=x_api_key, db=db)
    except HTTPException:
        return None


def require_role(*roles: str):
    """
    Rol tabanlı erişim kontrolü (RBAC) bağımlılık fabrikası.

    'admin' rolü daima tam yetkilidir.

    Örnek:
        @router.delete("/model/{name}", dependencies=[Depends(require_role("admin"))])
        @router.post("/train", dependencies=[Depends(require_role("admin", "researcher"))])
    """
    def role_checker(user: UserRecord = Depends(get_current_user)) -> UserRecord:
        allowed_roles = [r.lower() for r in roles]
        user_role = (user.role or "").lower()

        # Admin her zaman geçiş hakkına sahiptir
        if user_role == "admin" or user_role in allowed_roles:
            return user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Bu işlem için yetkiniz yetersiz. "
                f"Gerekli roller: {', '.join(roles)} (Mevcut rolünüz: {user.role})"
            )
        )

    return role_checker
