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
    role = getattr(request.state, "effective_role", user.role)
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
    for obj in session.new:
        if isinstance(obj, OwnedResource):
            obj.owner_id = actor.user_id
    for obj in session.dirty | session.deleted:
        if isinstance(obj, OwnedResource) and obj.owner_id != actor.user_id:
            if actor.role != "admin" and not (actor.role == "admin" and obj.owner_id is None):
                raise PermissionError("Resource belongs to another user")
