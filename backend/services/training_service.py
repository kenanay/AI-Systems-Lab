"""Durable, strict training orchestration. No implicit data or tokenizer fallback."""
import copy
import json
import logging
import math
import os
import random
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.models import TrainingJob, TokenizerRecord, DatasetVersion, FileRecord
from src.model.gpt import GPTModel, GPTConfig
from src.registry.model_registry import ModelRegistry
from src.tokenizer.loading import load_tokenizer, artifact_hash
from src.security.context import principal, Principal
from src.training.lora import LoRAConfig, add_lora_to_model, LinearWithLoRA
from src.training.sft_trainer import InstructionExample, InstructionDataset, TEMPLATES

logger = logging.getLogger(__name__)
# Compatibility exports; execution state lives in the database, not here.
ACTIVE_TRAINING_JOBS: Dict[str, Any] = {}


class SimpleCharTokenizer:
    """Explicit demo-only reversible UTF-8 byte tokenizer."""
    def __init__(self, vocab_size=260):
        self.vocab_size = max(260, vocab_size)
    def encode(self, text, add_bos=False, add_eos=False, **kwargs):
        return ([1] if add_bos else []) + [b + 4 for b in text.encode('utf-8')] + ([2] if add_eos else [])
    def decode(self, ids, **kwargs):
        return bytes(i - 4 for i in ids if 4 <= i < 260).decode('utf-8', errors='replace')


class SimpleTokenDataset(Dataset):
    def __init__(self, token_ids, seq_len=64):
        self.chunks = [token_ids[i:i + seq_len + 1] for i in range(0, len(token_ids) - 1, seq_len)]
        self.seq_len = seq_len
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, index):
        chunk = self.chunks[index]
        x, y = chunk[:-1], chunk[1:]
        return (torch.tensor(x + [0] * (self.seq_len - len(x)), dtype=torch.long),
                torch.tensor(y + [-100] * (self.seq_len - len(y)), dtype=torch.long))


def load_artifacts(db, dataset_id, tokenizer_id):
    if not dataset_id or not tokenizer_id:
        raise ValueError('Select a compiled dataset and tokenizer')
    ds = db.query(DatasetVersion).filter_by(dataset_id=dataset_id, is_active=True).first()
    tok = db.query(TokenizerRecord).filter_by(tokenizer_id=tokenizer_id, is_active=True).first()
    if ds is None or tok is None:
        raise ValueError('Dataset or tokenizer not found or inaccessible')

    # Preflight Yetki Kontrolü: Kullanıcı erişimini doğrula
    actor = principal.get()
    if actor and actor.role != 'admin':
        if getattr(ds, 'owner_id', None) and ds.owner_id != actor.user_id:
            raise ValueError('Bu veri kümesine erişim yetkiniz bulunmuyor.')
        if getattr(tok, 'owner_id', None) and tok.owner_id != actor.user_id:
            raise ValueError('Bu tokenizer kaydına erişim yetkiniz bulunmuyor.')

    if ds.tokenizer_id != tokenizer_id:
        raise ValueError('Dataset was compiled with a different tokenizer')
    fingerprints = ds.custom_metadata or {}
    for key, path in [('dataset_sha256', ds.storage_path), ('tokenizer_sha256', tok.storage_path)]:
        if not fingerprints.get(key) or artifact_hash(path) != fingerprints[key]:
            raise ValueError(f'{key} missing or mismatched; recompile dataset')
    if not ds.source_file_ids:
        raise ValueError('Dataset has no source permission lineage')
    files = db.query(FileRecord).filter(FileRecord.file_id.in_(ds.source_file_ids)).all()
    if len(files) != len(set(ds.source_file_ids)) or any(not f.training_allowed for f in files):
        raise ValueError('Source training permission missing or revoked')
    tokenizer = load_tokenizer(tok.storage_path)
    return ds, tok, tokenizer, fingerprints


class TrainingService:
    def __init__(self, db: Session):
        self.db = db
        self.checkpoints_base_dir = Path('checkpoints')
        self.checkpoints_base_dir.mkdir(exist_ok=True)

    def create_job(self, job_name, model_name, job_type='PRETRAIN', dataset_id=None, tokenizer_id=None, config=None, user_id=None):
        config = dict(config or {})
        job_type = {'SFT': 'FULL_SFT', 'SFT_LORA': 'LORA_SFT'}.get(job_type, job_type)
        if job_type not in {'PRETRAIN', 'FULL_SFT', 'LORA_SFT'}:
            raise ValueError('Unsupported training type')
        ModelRegistry._validate_identifier(model_name)
        ds, tok, tokenizer, fingerprints = load_artifacts(self.db, dataset_id, tokenizer_id)
        config.update(fingerprints)

        # Preflight Kontrolü: Tokenizer ve Model Sözlük Boyutu Sınır Aşımı (Index Overflow Önleme)
        tok_vocab = getattr(tokenizer, "vocab_size", None) or (tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else None) or tok.vocab_size
        model_vocab = config.get("vocab_size") or tok_vocab
        if config.get("vocab_size") and config["vocab_size"] < tok_vocab:
            raise ValueError(f"Model sözlük boyutu ({config['vocab_size']}), tokenizer sözlük boyutundan ({tok_vocab}) küçük olamaz!")

        # Preflight Kontrolü: Derlenmiş veri kümesindeki token kimliklerinin model sınırları içinde kaldığını doğrula
        if ds.storage_path and Path(ds.storage_path).exists():
            try:
                import pyarrow.parquet as pq
                schema = pq.read_schema(ds.storage_path)
                if "token_ids" in schema.names:
                    import pandas as pd
                    df = pd.read_parquet(ds.storage_path, columns=["token_ids"])
                    for token_list in df["token_ids"].dropna():
                        if len(token_list) > 0:
                            max_id = max(token_list)
                            if max_id >= model_vocab:
                                raise ValueError(f"Derlenmiş veri kümesindeki maksimum token kimliği ({max_id}), model sözlük sınırını ({model_vocab}) aşıyor!")
            except ValueError as e:
                if "sözlük sınırını" in str(e):
                    raise e
                logger.warning(f"Could not check parquet token boundaries: {e}")
            except Exception as e:
                logger.warning(f"Could not check parquet token boundaries: {e}")

        # Preflight Manifesti: Artefakt sürümlerini ve özetlerini dondur
        config["dataset_version"] = getattr(ds, "version", None) or ds.dataset_id
        config["tokenizer_name"] = tok.name
        config["tokenizer_vocab_size"] = tok_vocab
        config["model_vocab_size"] = model_vocab
        config["manifest_frozen"] = True
        config["preflight_verified_at"] = datetime.now(timezone.utc).isoformat()

        if job_type != 'PRETRAIN':
            if not config.get('base_model') or not config.get('base_version'):
                raise ValueError('Fine-tuning requires a base model and explicit version')
            info = ModelRegistry('models').load_model(config['base_model'], config['base_version'], verify_integrity=True)
            if not info.get('tokenizer_path') or artifact_hash(info['tokenizer_path']) != fingerprints['tokenizer_sha256']:
                raise ValueError('Base model tokenizer does not match dataset')
            config['base_checkpoint_sha256'] = artifact_hash(info['checkpoint_path'])
        actor = principal.get()
        effective_user_id = user_id or (actor.user_id if actor else None)
        config['owner_role'] = actor.role if actor else 'admin'
        config['owner_id'] = effective_user_id
        config['mode'] = 'real'
        job_id = f'TRN-{uuid.uuid4().hex[:12]}'
        job = TrainingJob(job_id=job_id, job_name=job_name, model_name=model_name, job_type=job_type,
                          dataset_id=dataset_id, tokenizer_id=tokenizer_id, config=config,
                          total_epochs=config.get('epochs', 3), status='PENDING', metrics=[],
                          output_dir=str(self.checkpoints_base_dir / job_id),
                          owner_id=effective_user_id)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def start_training(self, job_id):
        job = self.db.query(TrainingJob).filter_by(job_id=job_id).first()
        if not job or job.status not in {'PENDING', 'INTERRUPTED', 'CANCELLED', 'FAILED'}:
            raise ValueError('Job cannot be started in its current state')
        job.status, job.error = 'QUEUED', None
        cfg = dict(job.config)
        cfg['cancel_requested'] = False
        job.config = cfg
        self.db.commit()
        log_dir = Path(job.output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        from backend.config import settings
        env = dict(os.environ, DATABASE_URL=settings.database_url)
        with (log_dir / 'worker.log').open('ab') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'backend.worker', job_id],
                                    stdout=log, stderr=log, env=env, start_new_session=True)
        job.config = {**job.config, 'worker_pid': proc.pid}
        self.db.commit()

    def cancel_job(self, job_id):
        job = self.db.query(TrainingJob).filter_by(job_id=job_id).first()
        if not job or job.status not in {'RUNNING', 'QUEUED'}:
            return False
        job.config = {**job.config, 'cancel_requested': True}
        self.db.commit()
        return True

    def recover_interrupted_jobs(self):
        for job in self.db.query(TrainingJob).filter(TrainingJob.status.in_(["RUNNING", "QUEUED"])).all():
            pid = (job.config or {}).get("worker_pid")
            if not pid:
                continue
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                job.status = "INTERRUPTED"
                job.error = "Worker stopped; resume from the last checkpoint"
            except PermissionError:
                pass
        self.db.commit()

    def resume_job(self, job_id, user_id=None, is_admin=False):
        job = self.db.query(TrainingJob).filter_by(job_id=job_id).first()
        if not job:
            raise ValueError('Training job not found')
        if user_id is not None and not is_admin and job.user_id != user_id:
            raise PermissionError('Bu işi devam ettirme yetkiniz yok')
        if not (Path(job.output_dir) / 'resume.pt').exists():
            raise ValueError('No resumable checkpoint exists')
        job.config = {**job.config, 'resume': True}
        self.db.commit()
        self.start_training(job_id)
        return job

    @staticmethod
    def _run_training_worker(job_id, stop_event=None):
        with SessionLocal() as db:
            job = db.query(TrainingJob).filter_by(job_id=job_id).first()
            if not job:
                return
            scope = principal.set(Principal(job.owner_id, job.config.get('owner_role', 'researcher'))) if job.owner_id else None
            try:
                TrainingService._train(db, job, stop_event)
            except Exception as exc:
                logger.exception('Training failed')
                db.rollback()
                job = db.query(TrainingJob).filter_by(job_id=job_id).one()
                job.status, job.error = 'FAILED', str(exc)
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
            finally:
                if scope is not None:
                    principal.reset(scope)

    @staticmethod
    def _train(db, job, stop_event):
        import pyarrow.parquet as pq
        import numpy as np
        cfg = dict(job.config)
        seed = int(cfg.get('seed', 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        ds, tok, tokenizer, fingerprints = load_artifacts(db, job.dataset_id, job.tokenizer_id)
        for key in ['dataset_sha256', 'tokenizer_sha256']:
            if fingerprints[key] != cfg[key]:
                raise ValueError('Artifact changed after job creation')
        rows = pq.read_table(ds.storage_path).to_pylist()
        if any(row.get('split') not in {'train', 'validation', 'test'} for row in rows):
            raise ValueError('Dataset requires explicit train/validation/test splits')
        train_rows = [r for r in rows if r['split'] == 'train']
        val_rows = [r for r in rows if r['split'] == 'validation']
        if not train_rows or not val_rows:
            raise ValueError('Training and validation splits must both contain documents')
        sft = job.job_type != 'PRETRAIN'
        vocab = tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size') else tokenizer.vocab_size
        config = GPTConfig(vocab_size=vocab, d_model=int(cfg.get('d_model',128)), n_layers=int(cfg.get('n_layers',4)),
                           n_heads=int(cfg.get('n_heads',4)), d_ff=int(cfg.get('d_ff',cfg.get('d_model',128)*4)),
                           max_seq_len=int(cfg.get('max_seq_len',128)), dropout=float(cfg.get('dropout',0.1)))
        if sft:
            from src.inference.pipeline import InferencePipeline
            info = ModelRegistry('models').load_model(cfg['base_model'], cfg['base_version'], verify_integrity=True)
            if artifact_hash(info['checkpoint_path']) != cfg['base_checkpoint_sha256']:
                raise ValueError('Base checkpoint changed')
            model = InferencePipeline.from_pretrained(info['checkpoint_path'], info['tokenizer_path'], device='cpu').model
            config = model.config
        else:
            model = GPTModel(config)
        if job.job_type == 'LORA_SFT':
            for param in model.parameters():
                param.requires_grad = False
            target_modules = cfg.get('target_modules') or ['q_proj', 'v_proj', 'w_q', 'w_v']
            model = add_lora_to_model(model, LoRAConfig(rank=cfg.get('lora_r',8), alpha=cfg.get('lora_alpha',16), target_modules=target_modules))
            if not any(p.requires_grad for p in model.parameters()):
                raise ValueError('No LoRA target layers found')
        device = cfg.get('device', 'cpu')
        if device not in {'cpu','cuda','mps'}:
            raise ValueError('Unsupported execution device')
        model.to(device)
        def dataset(part):
            if sft:
                examples = []
                for row in part:
                    entry = row if 'instruction' in row else json.loads(row['text'])
                    examples.append(InstructionExample(instruction=entry['instruction'], response=entry['response'], input=entry.get('input'), system=entry.get('system')))
                return InstructionDataset(examples, tokenizer, max_length=config.max_seq_len, template=TEMPLATES['simple'], mask_instruction=True)
            if any('token_ids' not in row for row in part):
                raise ValueError('Compiled token IDs missing')
            return SimpleTokenDataset([int(t) for row in part for t in row['token_ids']], config.max_seq_len)
        train_ds, val_ds = dataset(train_rows), dataset(val_rows)
        if not len(train_ds) or not len(val_ds):
            raise ValueError('Insufficient tokens for training/validation')
        collate = InstructionDataset.collate_fn if sft else None
        # Fixed order is recorded by seed and artifact fingerprint; supports exact batch resume.
        loader = DataLoader(train_ds, batch_size=cfg.get('batch_size',4), shuffle=False, collate_fn=collate, generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_ds, batch_size=cfg.get('batch_size',4), shuffle=False, collate_fn=collate, generator=torch.Generator().manual_seed(seed))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.get('lr',1e-3))
        total_steps = int(cfg.get('epochs',3)) * len(loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1-step/max(1,total_steps)))
        output = Path(job.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        resume_path = output / 'resume.pt'
        start_epoch, start_batch, global_step = 0, 0, 0
        history = []
        if cfg.get('resume'):
            state = torch.load(resume_path, map_location='cpu', weights_only=False)
            if state['lineage'] != fingerprints:
                raise ValueError('Resume artifact lineage mismatch')
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
            start_epoch, start_batch, global_step = state['epoch'], state['batch'], state['step']
            history = state['metrics']
            random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'])
            if device == 'cuda' and state.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all(state['cuda_rng'])
        def save(epoch, batch):
            state = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                         scheduler_state_dict=scheduler.state_dict(), config=config.to_dict(), epoch=epoch, batch=batch,
                         step=global_step, metrics=history, lineage=fingerprints, python_rng=random.getstate(),
                         numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None)
            temp = resume_path.with_suffix('.tmp')
            torch.save(state,temp)
            temp.replace(resume_path)
            job.best_checkpoint = str(resume_path)
        def loss_for(batch):
            if sft:
                x, y = batch['input_ids'].to(device), batch['labels'].to(device)
                logits, _ = model(x)
                logits, y = logits[:,:-1,:], y[:,1:]
            else:
                x, y = [t.to(device) for t in batch]
                logits, _ = model(x)
            count = int((y != -100).sum())
            if not count:
                raise ValueError('Batch has no supervised target tokens')
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)), y.reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            return loss, count
        job.status, job.started_at = 'RUNNING', datetime.now(timezone.utc)
        job.total_steps = total_steps
        db.commit()
        for epoch in range(start_epoch, int(cfg.get('epochs',3))):
            model.train()
            for batch_index, batch in enumerate(loader):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                db.refresh(job)
                if job.config.get('cancel_requested') or (stop_event and stop_event.is_set()):
                    save(epoch,batch_index)
                    job.status, job.completed_at = 'CANCELLED', datetime.now(timezone.utc)
                    db.commit()
                    return
                optimizer.zero_grad()
                loss, _ = loss_for(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1
                history.append({'step':global_step,'epoch':epoch+1,'loss':loss.item(),
                                'perplexity':math.exp(loss.item()),'lr':optimizer.param_groups[0]['lr'],'mode':'real'})
                job.current_epoch, job.current_step = epoch+1, global_step
                job.progress, job.metrics = global_step/total_steps, copy.deepcopy(history)
                job.config = {**job.config,'heartbeat':datetime.now(timezone.utc).isoformat()}
                save(epoch,batch_index+1)
                db.commit()
            model.eval()
            nll, count = 0.0, 0
            with torch.inference_mode():
                for batch in val_loader:
                    loss, n = loss_for(batch)
                    nll += loss.item()*n
                    count += n
            history[-1].update(val_loss=nll/count, val_perplexity=math.exp(nll/count))
            job.metrics = copy.deepcopy(history)
            save(epoch+1,0)
            db.commit()
        # Export a plain GPT checkpoint; resume retains unmerged adapters and optimizer.
        exported = copy.deepcopy(model).cpu()
        for name, module in list(exported.named_modules()):
            if isinstance(module, LinearWithLoRA):
                module.merge()
                parent_name, _, attr = name.rpartition('.')
                parent = exported.get_submodule(parent_name) if parent_name else exported
                setattr(parent,attr,module.base_layer)
        final_path = output / 'model_final.pt'
        lineage = {**fingerprints,'dataset_id':job.dataset_id,'tokenizer_id':job.tokenizer_id,'experiment_id':job.job_id}
        torch.save({'model_state_dict':exported.state_dict(),'config':config.to_dict(),
                    'tokenizer_sha256':fingerprints['tokenizer_sha256'],'lineage':lineage}, final_path)
        version = cfg.get('version') or f'1.0.{int(datetime.now().timestamp())}'
        try:
            ModelRegistry('models').register_model(job.model_name,version,final_path,tok.storage_path,
                metrics={'val_loss':nll/count,'val_perplexity':math.exp(nll/count)},
                training_config={**config.to_dict(),**lineage},description=f'Experiment {job.job_id}')
        except Exception as exc:
            job.status, job.error = 'REGISTRY_FAILED', str(exc)
        else:
            job.status = 'COMPLETED'
        job.config = {**job.config,'model_version':version,'checkpoint_sha256':artifact_hash(final_path)}
        job.best_checkpoint, job.progress = str(final_path),1.0
        job.completed_at = datetime.now(timezone.utc)
        (output/'experiment.json').write_text(json.dumps({**lineage,'config':job.config,'metrics':history,'status':job.status},indent=2))
        db.commit()
