"""Independent local worker. A process lock serializes CPU/GPU jobs across API processes."""
import sys
import backend.security.scope  # install ownership rules in worker processes
import fcntl
from pathlib import Path
from backend.services.training_service import TrainingService


def main():
    Path('checkpoints').mkdir(exist_ok=True)
    with open('checkpoints/.worker.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        TrainingService._run_training_worker(sys.argv[1])


if __name__ == '__main__':
    main()
