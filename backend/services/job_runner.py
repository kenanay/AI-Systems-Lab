"""Process-backed job launching for jobs that must survive API restarts."""

import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy.orm import Session

from backend.config import settings


def spawn_job_worker(job_type: str, job_id: str, db: Session, job) -> int:
    """Start an independent worker and persist its PID on the job record."""
    log_root = (settings.log_file.parent if settings.log_file else Path("./logs")) / "jobs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{job_type}-{job_id}.log"
    env = dict(os.environ, DATABASE_URL=settings.database_url)

    try:
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "backend.worker", job_type, job_id],
                stdout=log,
                stderr=log,
                env=env,
                start_new_session=True,
            )
    except OSError as exc:
        job.status = "FAILED"
        job.error = f"Worker başlatılamadı: {exc}"
        db.commit()
        raise

    job.config = {**(job.config or {}), "worker_pid": process.pid}
    db.commit()
    return process.pid
