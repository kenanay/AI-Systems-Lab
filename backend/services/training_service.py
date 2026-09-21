"""
backend/services/training_service.py

Model Training Service

Bu modül dil modeli eğitimi (Pre-training ve LoRA/SFT) süreçlerini yönetir:
- Job oluşturma, başlatma ve durdurma
- Arka planda (threading) eğitim döngüsü
- Gerçek zamanlı metrik (loss, lr, perplexity) kaydı
- Checkpoint yönetimi
- Tamamlandığında Model Registry'ye otomatik kayıt
"""

import threading
import time
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.models import TrainingJob, TokenizerRecord, DatasetVersion, DocumentRecord
from src.model.gpt import GPTModel, GPTConfig
from src.tokenizer.bpe import BPETokenizer
from src.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
from src.registry.model_registry import ModelRegistry
from src.training.lora import LoRAConfig, add_lora_to_model, merge_lora_weights

logger = logging.getLogger(__name__)

# Global registry of active threads and cancel flags
ACTIVE_TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}


class SimpleTokenDataset(Dataset):
    """Eğitim için basit sequence dataset."""
    def __init__(self, token_ids: List[int], seq_len: int = 64) -> None:
        self.seq_len = seq_len
        # Chunk into sequences of seq_len + 1 (for input and target)
        self.chunks: List[List[int]] = []
        stride = max(1, seq_len // 2)
        for i in range(0, len(token_ids) - seq_len - 1, stride):
            self.chunks.append(token_ids[i:i + seq_len + 1])
        if not self.chunks and len(token_ids) > 1:
            # Pad if too short
            pad_len = seq_len + 1 - len(token_ids)
            padded = token_ids + [0] * pad_len
            self.chunks.append(padded)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.tensor(self.chunks[index], dtype=torch.long)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class TrainingService:
    """
    Model Training Orchestration Service.
    """
    def __init__(self, db: Session) -> None:

        self.db = db
        self.checkpoints_base_dir = Path("checkpoints")
        self.checkpoints_base_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry(registry_dir=str(self.models_dir))

    def create_job(
        self,
        job_name: str,
        model_name: str,
        job_type: str = "PRETRAIN",
        dataset_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> TrainingJob:
        """Yeni bir training job oluşturur."""
        job_id = f"TRN-{uuid.uuid4().hex[:8].upper()}"
        config = config or {}

        # Default hyperparameters
        epochs = config.get("epochs", 3)
        
        job = TrainingJob(
            job_id=job_id,
            job_name=job_name,
            job_type=job_type,
            status="PENDING",
            model_name=model_name,
            dataset_id=dataset_id,
            tokenizer_id=tokenizer_id,
            config=config,
            progress=0.0,
            current_epoch=0,
            total_epochs=epochs,
            current_step=0,
            total_steps=0,
            metrics=[],
            output_dir=str(self.checkpoints_base_dir / job_id),
            created_at=datetime.now(timezone.utc)
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def start_training(self, job_id: str) -> None:
        """Eğitimi arka plan iş parçacığında başlatır."""
        with JOBS_LOCK:
            # Check if already running
            if job_id in ACTIVE_TRAINING_JOBS:
                raise ValueError(f"Training job {job_id} is already running")
            
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run_training_worker,
                args=(job_id, stop_event),
                daemon=True
            )
            ACTIVE_TRAINING_JOBS[job_id] = {
                "thread": thread,
                "stop_event": stop_event,
                "started_at": datetime.now(timezone.utc)
            }
            thread.start()
            
        logger.info(f"Training thread started for job {job_id}")

    def cancel_job(self, job_id: str) -> bool:
        """Çalışan eğitimi durdurur."""
        if job_id in ACTIVE_TRAINING_JOBS:
            ACTIVE_TRAINING_JOBS[job_id]["stop_event"].set()
            logger.info(f"Stop signal sent to job {job_id}")
            return True
        return False

    @staticmethod
    def _run_training_worker(job_id: str, stop_event: threading.Event) -> None:
        """Arka plan eğitim fonksiyonu."""
        with SessionLocal() as db:
            try:
            job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if not job:
                logger.error(f"Job {job_id} not found in worker")
                return

            job.status = "RUNNING"
            job.started_at = datetime.now(timezone.utc)
            db.commit()

            raw_cfg = getattr(job, "config", {}) or {}
            cfg: Dict[str, Any] = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}
            d_model = cfg.get("d_model", 128)
            n_layers = cfg.get("n_layers", 4)
            n_heads = cfg.get("n_heads", 4)
            d_ff = cfg.get("d_ff", d_model * 4)
            max_seq_len = cfg.get("max_seq_len", 128)
            batch_size = cfg.get("batch_size", 4)
            lr = cfg.get("lr", 1e-3)
            epochs = cfg.get("epochs", 3)
            use_lora = job.job_type in ["SFT", "SFT_LORA"] or cfg.get("use_lora", False)
            lora_r = cfg.get("lora_r", 8)
            lora_alpha = cfg.get("lora_alpha", 16)

            # 1. Metin verilerini topla
            text_corpus: List[str] = []
            if job.dataset_id:
                # Check compiled dataset or documents
                ds = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == str(job.dataset_id)).first()
                if ds and ds.storage_path and Path(str(ds.storage_path)).exists():
                    import pyarrow.parquet as pq
                    table = pq.read_table(str(ds.storage_path))
                    text_corpus = [str(t) for t in table.column("text").to_pylist()]
            
            if not text_corpus:
                # Fallback to database documents
                docs = db.query(DocumentRecord).filter(DocumentRecord.is_empty == False).all()
                text_corpus = [str(d.text) for d in docs if d.text]

            if not text_corpus:
                text_corpus = [
                    "Yapay zeka sistemleri yerel bilgisayarlarda eğitilebilir ve çalıştırılabilir.",
                    "Transformer mimarisi attention mekanizmasına dayanır ve dil modellerinde temeldir.",
                    "Derin öğrenme modelleri büyük veri kümeleri üzerinde optimize edilir.",
                    "Doğal dil işleme metinleri sayılara dönüştürerek analiz eder.",
                    "Local AI Research Lab uçtan uca araştırma platformudur."
                ] * 10

            full_text: str = "\n\n".join(text_corpus)

            # 2. Tokenizer'ı hazırla
            tokenizer_id = str(job.tokenizer_id) if job.tokenizer_id else None
            tokens: List[int] = []
            vocab_size: int = 500

            actual_tokenizer_path: Optional[str] = None
            if tokenizer_id:
                tok_rec = db.query(TokenizerRecord).filter(TokenizerRecord.tokenizer_id == tokenizer_id).first()
                if tok_rec and tok_rec.model_path and Path(str(tok_rec.model_path)).exists():
                    actual_tokenizer_path = str(tok_rec.model_path)
                    vocab_size = int(getattr(tok_rec, "vocab_size", 500))
                    try:
                        if str(tok_rec.tokenizer_type) == "SENTENCEPIECE":
                            sp_tok = SentencePieceTokenizer()
                            sp_tok.load(str(tok_rec.model_path))
                            tokens = [int(t) for t in sp_tok.encode(full_text)]
                        else:
                            bpe_tok = BPETokenizer()
                            bpe_tok.load_vocab(Path(str(tok_rec.model_path)))
                            tokens = list(bpe_tok.encode(full_text))
                    except Exception as e:
                        logger.warning(f"Tokenizer load failed, using fallback: {e}")

            if not tokens:
                # Karakter seviyesi fallback
                tokens = [ord(c) % vocab_size for c in full_text]

            # 3. Model & Dataset oluştur
            dataset = SimpleTokenDataset(tokens, seq_len=min(int(max_seq_len), 64))
            dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

            model_config = GPTConfig(
                vocab_size=max(vocab_size, 300),
                max_seq_len=int(max_seq_len),
                d_model=int(d_model),
                n_layers=int(n_layers),
                n_heads=int(n_heads),
                d_ff=int(d_ff),
                dropout=0.1
            )
            model = GPTModel(model_config)

            # LoRA eklenecekse uygula
            if use_lora:
                lora_cfg = LoRAConfig(rank=int(lora_r), alpha=float(lora_alpha), r=int(lora_r), lora_alpha=float(lora_alpha))
                model = add_lora_to_model(model, lora_cfg)
                logger.info(f"Applied LoRA (r={lora_r}, alpha={lora_alpha}) to model")

            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            model = model.to(device)
            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
            criterion = nn.CrossEntropyLoss()

            total_steps = int(epochs) * len(dataloader)
            job.total_steps = total_steps
            job.total_epochs = int(epochs)
            db.commit()

            global_step = 0
            loss_val: float = 0.0
            perplexity: float = 1.0
            metrics_history: List[Dict[str, Any]] = []
            output_dir = Path(str(job.output_dir))
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Starting training job {job_id}: {epochs} epochs, {total_steps} total steps")

            for epoch in range(int(epochs)):
                if stop_event.is_set():
                    job.status = "CANCELLED"
                    db.commit()
                    return

                epoch_loss = 0.0
                for step, (x_batch, y_batch) in enumerate(dataloader):
                    if stop_event.is_set():
                        job.status = "CANCELLED"
                        db.commit()
                        return

                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    optimizer.zero_grad()
                    logits, _ = model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    loss_val = float(loss.item())
                    epoch_loss += loss_val
                    global_step += 1

                    # Log every 2 steps or on end
                    if global_step % 2 == 0 or global_step == total_steps:
                        perplexity = float(torch.exp(loss).item()) if loss_val < 20 else 999.0
                        metric_entry = {
                            "step": global_step,
                            "epoch": epoch + 1,
                            "loss": round(loss_val, 4),
                            "perplexity": round(perplexity, 2),
                            "lr": lr
                        }
                        metrics_history.append(metric_entry)

                        job.progress = round(global_step / max(1, total_steps), 4)
                        job.current_step = global_step
                        job.current_epoch = epoch + 1
                        job.metrics = metrics_history
                        db.commit()

                    time.sleep(0.02)  # Yield CPU lightly

            # Checkpoint kaydet
            checkpoint_path = output_dir / "model_final.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model_config.to_dict(),
                "epoch": epochs,
                "step": global_step
            }, checkpoint_path)

            # Model Registry'ye kaydet
            registry = ModelRegistry(registry_dir="models")
            version = "1.0.0"
            try:
                registry.register_model(
                    model_name=str(job.model_name),
                    version=version,
                    checkpoint_path=str(checkpoint_path),
                    tokenizer_path=actual_tokenizer_path,
                    metrics={
                        "loss": round(loss_val, 4),
                        "perplexity": round(perplexity, 2)
                    },
                    training_config=cfg,
                    description=f"Trained via TrainingJob {str(job.job_id)} ({str(job.job_type)})"
                )
            except Exception as reg_err:
                logger.warning(f"Registry registration note: {reg_err}")

            job.status = "COMPLETED"
            job.progress = 1.0
            job.best_checkpoint = str(checkpoint_path)
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            logger.info(f"Training job {job_id} completed successfully!")

        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
            db.rollback()
            job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if job:
                job.status = "FAILED"
                job.error = str(e)
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
            finally:
                if job_id in ACTIVE_TRAINING_JOBS:
                    with JOBS_LOCK:
                        if job_id in ACTIVE_TRAINING_JOBS:
                            del ACTIVE_TRAINING_JOBS[job_id]
