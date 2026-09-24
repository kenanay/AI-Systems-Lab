"""
FastAPI Güvenlik ve Kimlik Doğrulama Bağımlılıkları

JWT Bearer ve API Key kimlik doğrulaması, rol tabanlı erişim kontrolü (RBAC).

Author: Kenan AY
Version: 1.0.0
"""

from typing import List, Optional, Union, Any
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

# A key may narrow a user's permissions, but it must never widen them.  Keep
# the hierarchy in one place so cookie/JWT auth, API keys and endpoint-level
# RBAC all evaluate the same effective role.
ROLE_RANKS = {"viewer": 0, "researcher": 1, "admin": 2}


def _effective_role(user_role: str, credential_role: Optional[str] = None) -> Optional[str]:
    """Return the least-privileged valid role represented by a credential."""
    normalized_user_role = (user_role or "").strip().lower()
    if normalized_user_role not in ROLE_RANKS:
        return None

    if credential_role is None:
        return normalized_user_role

    normalized_credential_role = (credential_role or "").strip().lower()
    if normalized_credential_role not in ROLE_RANKS:
        return None

    return min(
        (normalized_user_role, normalized_credential_role),
        key=lambda role: ROLE_RANKS[role],
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_current_user(
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
    request: Request = None,
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

    if not token_str and request is not None:
        token_str = request.cookies.get("ailab_access")

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

        if api_key_rec.expires_at and api_key_rec.expires_at.replace(tzinfo=timezone.utc) <= _utc_now():
            raise HTTPException(status_code=401, detail="API key expired")

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
        effective_role = _effective_role(user.role, api_key_rec.role)
        if effective_role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API anahtarı geçersiz bir rol kapsamına sahip.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if request is not None:
            request.state.effective_role = effective_role
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

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Access token required")
    from backend.models import RevokedToken
    if payload.get("jti") and db.get(RevokedToken, payload["jti"]):
        raise HTTPException(status_code=401, detail="Token revoked")

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

    if request is not None:
        request.state.effective_role = _effective_role(user.role)
        if request.state.effective_role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Kullanıcı hesabında geçersiz rol tanımlı.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return user


def get_optional_current_user(
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
    request: Request = None,
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

    if not token_str and request is not None:
        token_str = request.cookies.get("ailab_access")

    if not token_str:
        return None

    try:
        return get_current_user(auth_header=auth_header, x_api_key=x_api_key, db=db, request=request)
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
    def role_checker(
        request: Request,
        user: UserRecord = Depends(get_current_user),
    ) -> UserRecord:
        allowed_roles = [r.lower() for r in roles]
        # `get_current_user` stores the role narrowed by an API key here.
        # Checking `user.role` alone would let a scoped admin key perform
        # admin-only operations.
        user_role = getattr(request.state, "effective_role", user.role or "").lower()

        # Admin her zaman geçiş hakkına sahiptir
        if user_role == "admin" or user_role in allowed_roles:
            return user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Bu işlem için yetkiniz yetersiz. "
                f"Gerekli roller: {', '.join(roles)} (Mevcut rolünüz: {user_role})"
            )
        )

    return role_checker


def check_resource_access(
    resource: Any,
    current_user: UserRecord,
    allow_unowned: bool = True
) -> bool:
    """
    Kullanıcının verilen kaynağa erişim yetkisini doğrular.
    - Admin kullanıcılar tüm kaynaklara erişebilir.
    - Kaynak sahibi (owner_id) oturum açmış kullanıcı ile eşleşiyorsa erişim verilir.
    - allow_unowned=True ise sahipsiz (sistem/demo/tohum) kaynaklar herkes tarafından görülebilir.
    - DocumentRecord gibi alt kaynaklarda üst kaynağın (FileRecord) sahipliği de kontrol edilir.
    """
    if not current_user:
        return False

    if (current_user.role or "").lower() == "admin":
        return True

    owner_id = getattr(resource, "owner_id", None)
    if owner_id is not None:
        return str(owner_id) == str(current_user.user_id)

    # DocumentRecord için parent file sahipliğini kontrol et
    file_rel = getattr(resource, "file", None)
    if file_rel is not None and getattr(file_rel, "owner_id", None) is not None:
        return str(file_rel.owner_id) == str(current_user.user_id)

    return allow_unowned


def filter_by_owner(
    query: Any,
    model: Any,
    current_user: UserRecord,
    allow_unowned: bool = True
) -> Any:
    """
    SQLAlchemy sorgusuna satır düzeyinde sahiplik/yetkilendirme filtresi ekler.
    Admin kullanıcılar tüm kayıtları görür.
    Standart kullanıcılar kendi kaynaklarını (ve allow_unowned=True ise paylaşımlı/örnek kaynakları) görür.
    """
    if not current_user or (current_user.role or "").lower() == "admin":
        return query

    if hasattr(model, "owner_id"):
        if allow_unowned:
            return query.filter((model.owner_id == current_user.user_id) | (model.owner_id == None))
        return query.filter(model.owner_id == current_user.user_id)

    return query
