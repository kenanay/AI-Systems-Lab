"""Shared content-store coordination and durable physical-file cleanup."""

import threading
from typing import Tuple

from sqlalchemy.orm import Session

from backend.models import ContentRecord
from backend.storage import storage_manager
from src.security.context import Principal, principal


# Protects threads in one API process. Database row locks/transactions protect
# separate processes; both layers are intentionally used.
content_store_lock = threading.Lock()


def cleanup_deleting_content(db: Session) -> Tuple[int, int]:
    """Retry physical deletion and remove only completed tombstones.

    A ContentRecord in DELETING state is durable work, so a transient filesystem
    failure cannot make the database forget which path still needs cleanup.
    """
    cleaned = 0
    pending = 0
    with content_store_lock:
        token = principal.set(Principal("system", "admin"))
        try:
            records = db.query(ContentRecord).filter(
                ContentRecord.status == "DELETING"
            ).with_for_update().all()
            for record in records:
                removed = storage_manager.delete_file(record.relative_path)
                if removed or not storage_manager.file_exists(record.relative_path):
                    db.delete(record)
                    cleaned += 1
                else:
                    pending += 1
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            principal.reset(token)
    return cleaned, pending
