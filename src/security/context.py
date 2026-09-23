"""Identity context shared by API, workers, and artifact storage."""
from contextvars import ContextVar
from dataclasses import dataclass

@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str

principal: ContextVar[Principal | None] = ContextVar("principal", default=None)
