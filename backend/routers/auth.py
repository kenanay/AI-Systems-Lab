"""
Auth & Security Router

Kullanıcı kaydı, JWT ile oturum açma, profil yönetimi, API anahtarı (API Key) üretimi
ve RBAC rol yönetimi endpoint'leri.

Author: Kenan AY
Version: 1.0.0
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db
from backend.models import APIKeyRecord, UserRecord, utc_now
from backend.security.api_keys import generate_api_key
from backend.security.dependencies import get_current_user, require_role
from backend.security.jwt import create_access_token, create_refresh_token, decode_token
from backend.security.password import hash_password, verify_password
from backend.security.rate_limiter import rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/auth",
    tags=["Authentication & Security"],
)

VALID_ROLES = {"admin", "researcher", "viewer"}


# ============================================================================
# Schemas
# ============================================================================

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Kullanıcı adı (benzersiz)")
    email: str = Field(..., min_length=5, max_length=120, description="E-posta adresi")
    password: str = Field(..., min_length=6, max_length=100, description="Güçlü parola")
    full_name: Optional[str] = Field(None, max_length=100, description="Ad Soyad")
    role: Optional[str] = Field("researcher", description="Talep edilen rol (researcher, viewer)")


class LoginRequest(BaseModel):
    username_or_email: str = Field(..., description="Kullanıcı adı veya e-posta adresi")
    password: str = Field(..., description="Parola")


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., description="Geçerli Refresh Token")


class RefreshTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = Field(None, max_length=100)
    current_password: Optional[str] = None
    new_password: Optional[str] = Field(None, min_length=6, max_length=100)


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Anahtar etiketi (örn: Python SDK, Kaggle)")
    role: Optional[str] = Field(None, description="Yetki rolü (boş bırakılırsa kullanıcının rolü atanır)")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Geçerlilik süresi (gün)")


class APIKeyResponse(BaseModel):
    key_id: str
    user_id: str
    name: str
    key_prefix: str
    role: str
    is_active: bool
    created_at: str
    expires_at: Optional[str] = None
    last_used_at: Optional[str] = None
    raw_key: Optional[str] = None  # Sadece üretim anında tek defa döner


class UpdateRoleRequest(BaseModel):
    role: str = Field(..., description="Yeni kullanıcı rolü: admin, researcher, viewer")


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/register",
    response_model=Dict[str, Any],
    status_code=status.HTTP_201_CREATED,
    summary="Yeni Kullanıcı Kaydı",
    dependencies=[Depends(rate_limit(max_requests=10, window_seconds=60))]
)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Yeni bir kullanıcı hesabı oluşturur."""
    username_clean = req.username.strip().lower()
    email_clean = req.email.strip().lower()

    if db.query(UserRecord).filter(UserRecord.username == username_clean).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bu kullanıcı adı zaten kullanılıyor."
        )

    if db.query(UserRecord).filter(UserRecord.email == email_clean).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bu e-posta adresi ile kayıtlı bir hesap zaten var."
        )

    target_role = (req.role or "researcher").strip().lower()
    # Güvenlik: Normal kayıt ile doğrudan 'admin' hesabı oluşturulamaz
    if target_role not in {"researcher", "viewer"}:
        target_role = "researcher"

    new_user = UserRecord(
        username=username_clean,
        email=email_clean,
        hashed_password=hash_password(req.password),
        role=target_role,
        full_name=req.full_name.strip() if req.full_name else None,
        is_active=True
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info(f"Yeni kullanıcı kaydı başarılı: {new_user.username} (Rol: {new_user.role})")
    return {
        "message": "Kayıt başarılı. Giriş yapabilirsiniz.",
        "user": new_user.to_dict()
    }


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Oturum Açma (JWT Al)",
    dependencies=[Depends(rate_limit(max_requests=15, window_seconds=60))]
)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """
    Kullanıcı adı veya e-posta ve parola ile oturum açar.
    Access Token ve Refresh Token döner.
    """
    identifier = req.username_or_email.strip().lower()

    user = db.query(UserRecord).filter(
        (UserRecord.username == identifier) | (UserRecord.email == identifier)
    ).first()

    if not user or not verify_password(req.password, str(user.hashed_password)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı adı veya parola hatalı.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hesabınız askıya alınmış veya devre dışı bırakılmış."
        )

    # Son giriş tarihini güncelle
    user.last_login = utc_now()
    db.commit()

    # Token payload
    payload = {
        "sub": user.user_id,
        "username": user.username,
        "email": user.email,
        "role": user.role
    }

    access_delta = timedelta(minutes=settings.access_token_expire_minutes)
    refresh_delta = timedelta(days=settings.refresh_token_expire_days)

    access_token = create_access_token(payload, expires_delta=access_delta)
    refresh_token = create_refresh_token(payload, expires_delta=refresh_delta)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=int(access_delta.total_seconds()),
        user=user.to_dict()
    )


@router.post(
    "/refresh",
    response_model=RefreshTokenResponse,
    summary="Access Token Yenileme",
    dependencies=[Depends(rate_limit(max_requests=30, window_seconds=60))]
)
def refresh_token(req: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Refresh token kullanarak yeni bir Access Token üretir."""
    try:
        payload = decode_token(req.refresh_token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Geçersiz veya süresi dolmuş refresh token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz token türü. Refresh token bekleniyordu.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user_id = payload.get("sub")
    user = db.query(UserRecord).filter(UserRecord.user_id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı bulunamadı veya hesabı aktif değil.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    new_payload = {
        "sub": user.user_id,
        "username": user.username,
        "email": user.email,
        "role": user.role
    }
    access_delta = timedelta(minutes=settings.access_token_expire_minutes)
    new_access_token = create_access_token(new_payload, expires_delta=access_delta)

    return RefreshTokenResponse(
        access_token=new_access_token,
        token_type="bearer",
        expires_in=int(access_delta.total_seconds())
    )


@router.get(
    "/me",
    summary="Aktif Kullanıcı Profili"
)
def get_me(current_user: UserRecord = Depends(get_current_user)):
    """Giriş yapmış olan kullanıcının profil bilgilerini ve rollerini döndürür."""
    return {
        "user": current_user.to_dict(),
        "permissions": {
            "can_train": current_user.role in {"admin", "researcher"},
            "can_delete_models": current_user.role == "admin",
            "can_manage_users": current_user.role == "admin",
            "can_create_api_keys": True,
            "can_run_benchmarks": current_user.role in {"admin", "researcher"},
            "can_view_metrics": True
        }
    }


@router.put(
    "/profile",
    summary="Profil Güncelleme"
)
def update_profile(
    req: UpdateProfileRequest,
    current_user: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Kullanıcının profil adı veya parolasını günceller."""
    if req.full_name is not None:
        current_user.full_name = req.full_name.strip() or None

    if req.new_password:
        if not req.current_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parola değiştirmek için mevcut parolanızı girmelisiniz."
            )
        if not verify_password(req.current_password, str(current_user.hashed_password)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mevcut parolanız hatalı."
            )
        current_user.hashed_password = hash_password(req.new_password)

    current_user.updated_at = utc_now()
    db.commit()
    db.refresh(current_user)

    return {
        "message": "Profil bilgileri güncellendi.",
        "user": current_user.to_dict()
    }


# ============================================================================
# API Key Management
# ============================================================================

@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Yeni API Key Oluştur"
)
def create_user_api_key(
    req: CreateAPIKeyRequest,
    current_user: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Python SDK veya harici istemciler için yeni bir API anahtarı üretir.
    DİKKAT: raw_key yalnızca bu yanıtla birlikte TEK SEFERLİK görüntülenir!
    """
    raw_key, key_prefix, key_hash = generate_api_key()

    target_role = (req.role or current_user.role).strip().lower()
    # Kullanıcı kendi rolünden daha yüksek bir rol atayamaz (admin hariç)
    if current_user.role != "admin" and target_role == "admin":
        target_role = current_user.role

    expires_at = None
    if req.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=req.expires_in_days)

    api_key_rec = APIKeyRecord(
        user_id=current_user.user_id,
        name=req.name.strip(),
        key_hash=key_hash,
        key_prefix=key_prefix,
        role=target_role,
        is_active=True,
        expires_at=expires_at
    )
    db.add(api_key_rec)
    db.commit()
    db.refresh(api_key_rec)

    data = api_key_rec.to_dict()
    data["raw_key"] = raw_key
    return data


@router.get(
    "/api-keys",
    response_model=List[APIKeyResponse],
    summary="Kullanıcının API Anahtarlarını Listele"
)
def list_user_api_keys(
    current_user: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Giriş yapmış kullanıcının oluşturduğu tüm API anahtarlarını listeler."""
    keys = db.query(APIKeyRecord).filter(
        APIKeyRecord.user_id == current_user.user_id
    ).order_by(APIKeyRecord.created_at.desc()).all()

    return [k.to_dict() for k in keys]


@router.delete(
    "/api-keys/{key_id}",
    summary="API Anahtarını İptal Et / Sil"
)
def revoke_api_key(
    key_id: str,
    current_user: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Belirtilen API anahtarını iptal eder."""
    query = db.query(APIKeyRecord).filter(APIKeyRecord.key_id == key_id)
    if current_user.role != "admin":
        query = query.filter(APIKeyRecord.user_id == current_user.user_id)

    api_key_rec = query.first()
    if not api_key_rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API anahtarı bulunamadı."
        )

    db.delete(api_key_rec)
    db.commit()
    return {"message": f"'{api_key_rec.name}' ({api_key_rec.key_prefix}) API anahtarı başarıyla silindi."}


# ============================================================================
# Admin User Management
# ============================================================================

@router.get(
    "/users",
    summary="Tüm Kullanıcıları Listele (Admin Only)",
    dependencies=[Depends(require_role("admin"))]
)
def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Sistemdeki tüm kayıtlı kullanıcıları listeler (Yalnızca Admin yetkisi gerektirir)."""
    total = db.query(UserRecord).count()
    users = db.query(UserRecord).order_by(UserRecord.created_at.desc()).offset(skip).limit(limit).all()

    return {
        "total": total,
        "users": [u.to_dict() for u in users]
    }


@router.patch(
    "/users/{user_id}/role",
    summary="Kullanıcı Rolünü Güncelle (Admin Only)",
    dependencies=[Depends(require_role("admin"))]
)
def update_user_role(
    user_id: str,
    req: UpdateRoleRequest,
    current_admin: UserRecord = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Bir kullanıcının yetki rolünü günceller (Admin Only)."""
    target_role = req.role.strip().lower()
    if target_role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Geçersiz rol: {target_role}. Geçerli roller: {', '.join(VALID_ROLES)}"
        )

    user = db.query(UserRecord).filter(UserRecord.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kullanıcı bulunamadı.")

    user.role = target_role
    db.commit()
    db.refresh(user)

    logger.info(f"Admin '{current_admin.username}' kullanıcısı, '{user.username}' rolünü '{target_role}' yaptı.")
    return {
        "message": f"'{user.username}' kullanıcısının rolü '{target_role}' olarak güncellendi.",
        "user": user.to_dict()
    }
