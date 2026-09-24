"""Request ownership is propagated to ORM queries and filesystem registries."""
from contextvars import ContextVar
from dataclasses import dataclass
from fastapi import Depends, HTTPException, Request
from sqlalchemy import event, or_
from sqlalchemy.orm import Session, with_loader_criteria
from backend.security.dependencies import get_current_user
from backend.models import OwnedResource, UserRecord


from src.security.context import Principal, principal


async def require_access(request: Request, user: UserRecord = Depends(get_current_user)):
    role = getattr(request.state, "effective_role", user.role).lower()
    if request.method not in {"GET", "HEAD", "OPTIONS"} and role == "viewer":
        raise HTTPException(403, "Read-only account")
    # Cookie-authenticated mutations must originate from our frontend.
    if request.method not in {"GET", "HEAD", "OPTIONS"} and request.cookies.get("ailab_access"):
        from backend.config import settings
        origin = request.headers.get("origin")
        if origin and origin not in settings.allowed_origins:
            raise HTTPException(403, "Untrusted origin")
    from src.registry.model_registry import ModelRegistry
    for key in ("model_name", "version"):
        value = request.path_params.get(key) or request.query_params.get(key)
        if value:
            try:
                ModelRegistry._validate_identifier(value)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
    token = principal.set(Principal(str(user.user_id), str(role)))
    try:
        yield user
    finally:
        principal.reset(token)


@event.listens_for(Session, "do_orm_execute")
def filter_owned_resources(state):
    actor = principal.get()
    if actor is None or not state.is_select:
        return
    owner = actor.user_id
    if actor.role == "admin":
        return
    rule = with_loader_criteria(
        OwnedResource,
        lambda cls: or_(cls.owner_id == owner, cls.owner_id.is_(None)),
        include_aliases=True
    )
    state.statement = state.statement.options(rule)


@event.listens_for(Session, "before_flush")
def stamp_and_check_owner(session, context, instances):
    actor = principal.get()
    if actor is None:
        return

    # 1. Viewer rolü sadece okuma yapabilir; oluşturma, güncelleme veya silme yapamaz
    if actor.role == "viewer" and (session.new or session.dirty or session.deleted):
        raise PermissionError("Viewer role cannot create, modify, or delete resources")

    # 2. Yeni oluşturulan kaynaklarda owner_id yoksa aktörün ID'sini ata
    for obj in session.new:
        if isinstance(obj, OwnedResource):
            if getattr(obj, "owner_id", None) is None and actor.role != "admin":
                obj.owner_id = actor.user_id

    # 3. Admin rolü tüm kaynakları (başka kullanıcılara ait veya genel demo/unowned)
    #    oluşturma, güncelleme ve silme konusunda tam yetkiye sahiptir.
    if actor.role == "admin":
        return

    # 4. Standart kullanıcılar (researcher vb.):
    #    Yalnızca kendi sahip oldukları kaynakları güncelleyebilir veya silebilir.
    #    Başka kullanıcılara ait kayıtları veya genel demo kayıtlarını (owner_id is None) değiştiremez/silemez.
    for obj in session.dirty | session.deleted:
        if isinstance(obj, OwnedResource):
            if obj.owner_id != actor.user_id:
                raise PermissionError("Resource belongs to another user or is a protected demo resource")
