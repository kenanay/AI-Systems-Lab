"""Independent local worker for training, tokenizer and compilation jobs."""

import contextlib
import sys
from pathlib import Path

import backend.security.scope  # install ownership rules in worker processes
from backend.database import SessionLocal
from backend.models import CompilationJob, TokenizerJob, UserRecord
from src.security.context import Principal, principal
from backend.services.dataset_service import DatasetCompilationService
from backend.services.tokenizer_service import TokenizerTrainingService
from backend.services.training_service import TrainingService


@contextlib.contextmanager
def acquire_worker_lock(lock_file_path: Path):
    """Platform-independent file lock for worker processes."""
    lock_file = open(lock_file_path, "a")
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield lock_file
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        except Exception:
            pass
        lock_file.close()


def _run_owned_job(job_type: str, job_id: str) -> None:
    """Load ownership before installing the worker's principal scope."""
    model = {"tokenizer": TokenizerJob, "compilation": CompilationJob}[job_type]
    with SessionLocal() as db:
        bootstrap = principal.set(Principal("worker", "admin"))
        try:
            job = db.query(model).filter_by(job_id=job_id).first()
            if not job:
                return
            owner_id = getattr(job, "owner_id", None)
            owner = db.query(UserRecord).filter_by(user_id=owner_id).first() if owner_id else None
        finally:
            principal.reset(bootstrap)

        role = str(owner.role) if owner else "admin"
        scope = principal.set(Principal(str(owner_id or "worker"), role))
        try:
            if job_type == "tokenizer":
                TokenizerTrainingService(db).run_training_job(job_id)
            else:
                DatasetCompilationService(db).run_compilation_job(job_id)
        finally:
            principal.reset(scope)


def main() -> None:
    # One positional argument was the historical training-worker interface.
    if len(sys.argv) == 2:
        job_type, job_id = "training", sys.argv[1]
    elif len(sys.argv) == 3 and sys.argv[1] in {"training", "tokenizer", "compilation"}:
        job_type, job_id = sys.argv[1], sys.argv[2]
    else:
        raise SystemExit("usage: python -m backend.worker [training|tokenizer|compilation] JOB_ID")

    Path("checkpoints").mkdir(exist_ok=True)
    with acquire_worker_lock(Path("checkpoints/.worker.lock")):
        if job_type == "training":
            TrainingService._run_training_worker(job_id)
        else:
            _run_owned_job(job_type, job_id)


if __name__ == "__main__":
    main()
