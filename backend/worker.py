"""Independent local worker. A cross-platform process lock serializes CPU/GPU jobs across API processes."""
import sys
import contextlib
from pathlib import Path
import backend.security.scope  # install ownership rules in worker processes
from backend.services.training_service import TrainingService


@contextlib.contextmanager
def acquire_worker_lock(lock_file_path: Path):
    """Platform-independent file lock for worker processes (Linux, macOS, Windows)."""
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


def main():
    Path("checkpoints").mkdir(exist_ok=True)
    with acquire_worker_lock(Path("checkpoints/.worker.lock")):
        TrainingService._run_training_worker(sys.argv[1])


if __name__ == "__main__":
    main()

