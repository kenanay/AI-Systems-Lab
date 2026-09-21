"""
src/training/trainer.py

Training Loop & Trainer

Bu modül language model training için complete training loop'u içerir:
- Forward/backward pass
- Gradient accumulation
- Mixed precision training (optional)
- Validation loop
- Metrics tracking & logging
- Checkpoint management
- Progress monitoring

Trainer, tüm training component'lerini orchestrate eder ve
training process'i yönetir.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Callable
import logging
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple progress indicator if tqdm not available
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.initial = kwargs.get('initial', 0)
            self.n = self.initial
        def update(self, n=1):
            self.n += n
        def set_postfix(self, *args, **kwargs):
            pass
        def close(self):
            pass

from src.training.loss import LanguageModelLoss
from src.training.optimizer import create_optimizer, get_warmup_cosine_schedule, clip_gradients
from src.training.checkpoint import CheckpointManager
from src.training.metrics_logger import TrainingLogger
from src.training.tensorboard_logger import create_tensorboard_logger
from src.training.sample_tracker import SampleTracker

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Training Configuration
    
    Tüm training hyperparameter'larını içerir.
    
    Attributes:
        # Optimization
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        adam_beta1: Adam beta1 parameter
        adam_beta2: Adam beta2 parameter
        adam_eps: Adam epsilon
        
        # Learning rate schedule
        warmup_steps: Number of warmup steps
        max_steps: Maximum training steps
        min_lr_ratio: Minimum LR as ratio of peak
        
        # Training
        batch_size: Training batch size
        gradient_accumulation_steps: Steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        
        # Loss
        label_smoothing: Label smoothing factor
        
        # Logging & checkpointing
        log_interval: Steps between logging
        eval_interval: Steps between evaluation
        checkpoint_interval: Steps between checkpoints
        max_checkpoints: Maximum checkpoints to keep
        
        # Mixed precision
        use_amp: Use automatic mixed precision
        
        # Validation
        eval_steps: Number of eval steps per evaluation
    """
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    
    # LR schedule
    warmup_steps: int = 1000
    max_steps: int = 10000
    min_lr_ratio: float = 0.1
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Loss
    label_smoothing: float = 0.0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    max_checkpoints: int = 3
    
    # Mixed precision
    use_amp: bool = False
    
    # Validation
    eval_steps: int = 100
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.max_steps > 0, "max_steps must be positive"
        assert 0 <= self.min_lr_ratio <= 1, "min_lr_ratio must be in [0, 1]"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"
        assert 0 <= self.label_smoothing < 1, "label_smoothing must be in [0, 1)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class Trainer:
    """
    Language Model Trainer
    
    Complete training loop implementation.
    
    Args:
        model: PyTorch model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        config: Training configuration
        checkpoint_dir: Checkpoint directory
        device: Device to train on
        resume_from: Checkpoint path to resume from (optional)
    
    Example:
        >>> from src.model.gpt import GPTModel, GPTConfig
        >>> 
        >>> model_config = GPTConfig(vocab_size=8000, d_model=256, n_layers=6)
        >>> model = GPTModel(model_config)
        >>> 
        >>> train_config = TrainingConfig(
        ...     learning_rate=3e-4,
        ...     max_steps=10000,
        ...     batch_size=8
        ... )
        >>> 
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     config=train_config
        ... )
        >>> 
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: TrainingConfig = None,
        checkpoint_dir: str = 'checkpoints',
        device: str = 'cpu',
        resume_from: Optional[str] = None,
        # Dashboard components
        enable_tensorboard: bool = True,
        enable_sample_tracking: bool = False,
        test_prompts: Optional[list] = None,
        tokenizer: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or TrainingConfig()
        self.device = device
        
        # Get vocab size from model
        vocab_size = model.config.vocab_size if hasattr(model, 'config') else 8000
        
        # Loss function
        self.loss_fn = LanguageModelLoss(
            vocab_size=vocab_size,
            padding_idx=0,
            label_smoothing=self.config.label_smoothing
        )
        
        # Optimizer
        self.optimizer = create_optimizer(
            model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps
        )
        
        # LR Scheduler
        self.scheduler = get_warmup_cosine_schedule(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
            min_lr_ratio=self.config.min_lr_ratio
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=self.config.max_checkpoints,
            save_best=True,
            best_metric='val_loss' if val_dataloader else 'train_loss',
            best_mode='min'
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking (legacy)
        self.train_metrics = []
        self.val_metrics = []
        
        # === Dashboard Components ===
        
        # Metrics Logger
        self.metrics_logger = TrainingLogger(
            log_dir=checkpoint_dir,
            experiment_name='training_run'
        )
        # Log hyperparameters
        self.metrics_logger.log_hyperparameters(self.config.to_dict())
        
        # TensorBoard Logger
        self.tb_logger = create_tensorboard_logger(
            log_dir=f"{checkpoint_dir}/tensorboard",
            experiment_name='training',
            enabled=enable_tensorboard
        )
        if self.tb_logger.enabled:
            # Log hyperparameters to TensorBoard
            self.tb_logger.log_hyperparameters(self.config.to_dict())
            # Log model graph (if possible)
            try:
                max_seq_len = model.config.max_seq_len if hasattr(model, 'config') else 512
                self.tb_logger.log_model_graph(model, input_shape=(max_seq_len,))
            except Exception as e:
                logger.warning(f"Could not log model graph: {e}")
        
        # Sample Tracker
        self.sample_tracker = None
        if enable_sample_tracking and tokenizer and test_prompts:
            self.sample_tracker = SampleTracker(
                model=model,
                tokenizer=tokenizer,
                test_prompts=test_prompts,
                log_dir=f"{checkpoint_dir}/samples",
                log_to_tensorboard=enable_tensorboard,
                tensorboard_logger=self.tb_logger
            )
            logger.info(f"Sample tracking enabled with {len(test_prompts)} prompts")
        
        # Resume from checkpoint if provided
        if resume_from:
            self._resume_from_checkpoint(resume_from)
        
        logger.info(
            f"Trainer initialized: device={device}, "
            f"lr={self.config.learning_rate}, "
            f"max_steps={self.config.max_steps}"
        )
        logger.info(f"Dashboard: TensorBoard={'ON' if self.tb_logger.enabled else 'OFF'}, "
                   f"SampleTracking={'ON' if self.sample_tracker else 'OFF'}")
    
    def train(self):
        """
        Main training loop.
        
        Runs training for max_steps with logging, evaluation, and checkpointing.
        """
        logger.info("Starting training...")
        logger.info(f"  Total steps: {self.config.max_steps}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        
        self.model.train()
        train_iterator = iter(self.train_dataloader)
        
        # Progress bar
        pbar = tqdm(total=self.config.max_steps, initial=self.global_step, desc="Training")
        
        start_time = time.time()
        
        while self.global_step < self.config.max_steps:
            # Training step
            metrics = self._training_step(train_iterator)
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                self._log_metrics(metrics, prefix="train")
                pbar.set_postfix(metrics)
            
            # Validation
            if self.val_dataloader and self.global_step % self.config.eval_interval == 0:
                val_metrics = self.evaluate()
                self._log_metrics(val_metrics, prefix="val")
                
                # Add val_loss to metrics for checkpoint
                metrics['val_loss'] = val_metrics['loss']
                
                # Check if best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
            
            # Checkpointing
            if self.global_step % self.config.checkpoint_interval == 0 and self.global_step > 0:
                self._save_checkpoint(metrics)
            
            # Sample Generation (if enabled)
            if self.sample_tracker and self.global_step % self.config.checkpoint_interval == 0:
                logger.info(f"Generating samples at step {self.global_step}...")
                self.sample_tracker.generate_samples(
                    step=self.global_step,
                    epoch=self.epoch,
                    max_new_tokens=50
                )
            
            pbar.update(1)
            self.global_step += 1
        
        pbar.close()
        
        # Final checkpoint
        self._save_checkpoint(metrics)
        
        # Save dashboard logs
        logger.info("Saving dashboard logs...")
        self.metrics_logger.save_json()
        self.metrics_logger.save_csv()
        if self.sample_tracker:
            self.sample_tracker.save_history()
        
        # Close TensorBoard
        if self.tb_logger.enabled:
            self.tb_logger.close()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time/60:.2f} minutes")
        logger.info(f"Final train loss: {metrics.get('loss', 'N/A'):.4f}")
        if self.best_val_loss < float('inf'):
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("Training Summary")
        logger.info("="*80)
        self.metrics_logger.print_summary()
        if self.sample_tracker:
            self.sample_tracker.print_summary()
    
    def _training_step(self, train_iterator) -> Dict[str, float]:
        """
        Execute one training step with gradient accumulation.
        
        Args:
            train_iterator: Training data iterator
        
        Returns:
            Dictionary of metrics
        """
        total_loss = 0.0
        
        for accum_step in range(self.config.gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Restart iterator
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
            
            # Move batch to device
            # input_ids shape: [batch_size, seq_len]
            input_ids = batch['input_ids'].to(self.device)
            # labels shape: [batch_size, seq_len]
            labels = batch['labels'].to(self.device) if 'labels' in batch else input_ids
            
            # Forward pass
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    # logits shape: [batch_size, seq_len, vocab_size]
                    logits, _ = self.model(input_ids)
                    # loss shape: scalar
                    loss = self.loss_fn(logits, labels)
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                # logits shape: [batch_size, seq_len, vocab_size]
                logits, _ = self.model(input_ids)
                # loss shape: scalar
                loss = self.loss_fn(logits, labels)
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item()
        
        # Gradient clipping
        if self.config.use_amp:
            self.scaler.unscale_(self.optimizer)
        
        grad_norm = clip_gradients(self.model, max_norm=self.config.max_grad_norm)
        
        # Optimizer step
        if self.config.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Compute metrics
        current_lr = self.optimizer.param_groups[0]['lr']
        
        metrics = {
            'loss': total_loss,
            'lr': current_lr,
            'grad_norm': grad_norm,
            'step': self.global_step
        }
        
        self.train_metrics.append(metrics)
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        logger.info("Running validation...")
        
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        
        val_iterator = iter(self.val_dataloader)
        
        for _ in range(min(self.config.eval_steps, len(self.val_dataloader))):
            try:
                batch = next(val_iterator)
            except StopIteration:
                break
            
            # input_ids shape: [batch_size, seq_len]
            input_ids = batch['input_ids'].to(self.device)
            # labels shape: [batch_size, seq_len]
            labels = batch['labels'].to(self.device) if 'labels' in batch else input_ids
            
            # Forward pass
            # logits shape: [batch_size, seq_len, vocab_size]
            logits, _ = self.model(input_ids)
            # loss shape: scalar
            loss, batch_metrics = self.loss_fn(logits, labels, return_metrics=True)
            
            total_loss += loss.item()
            
            # Accuracy tracking
            # predictions shape: [batch_size, seq_len]
            predictions = logits.argmax(dim=-1)
            # mask shape: [batch_size, seq_len]
            mask = labels != 0  # Assuming 0 is padding
            # correct shape: [batch_size, seq_len]
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            num_batches += 1
        
        # Compute averages
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': avg_accuracy
        }
        
        self.val_metrics.append(metrics)
        
        self.model.train()
        
        return metrics
    
    def _save_checkpoint(self, metrics: Dict[str, float]):
        """Save training checkpoint."""
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.epoch,
            global_step=self.global_step,
            metrics=metrics,
            scheduler=self.scheduler,
            config=self.config.to_dict()
        )
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """
        Log metrics to all dashboard components.
        
        Args:
            metrics: Dictionary of metric name → value
            prefix: Prefix for metric names (e.g., 'train', 'val')
        """
        prefix_str = f"{prefix}/" if prefix else ""
        
        # Console logging
        log_str = f"Step {self.global_step} | "
        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f"{prefix_str}{key}: {value:.4f} | "
        logger.info(log_str)
        
        # Metrics Logger (structured storage)
        phase = prefix if prefix else 'train'
        self.metrics_logger.log_metrics(
            step=self.global_step,
            epoch=self.epoch,
            phase=phase,
            metrics=metrics
        )
        
        # TensorBoard Logger
        if self.tb_logger.enabled:
            self.tb_logger.log_metrics_dict(
                metrics=metrics,
                step=self.global_step,
                prefix=prefix_str
            )
            
            # Log model parameters periodically
            if self.global_step % (self.config.checkpoint_interval * 2) == 0:
                self.tb_logger.log_model_parameters(
                    self.model,
                    step=self.global_step,
                    log_gradients=True
                )


if __name__ == "__main__":
    # Simple training test
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Trainer Test (Dummy Data)")
    print("=" * 80)
    
    from src.model.gpt import GPTModel, GPTConfig
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create small model
    model_config = GPTConfig(
        vocab_size=100,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512
    )
    
    model = GPTModel(model_config)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create dummy dataset
    num_samples = 100
    seq_len = 32
    
    dummy_data = torch.randint(0, model_config.vocab_size, (num_samples, seq_len))
    dataset = TensorDataset(dummy_data)
    
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: {'input_ids': torch.stack([item[0] for item in x])}
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: {'input_ids': torch.stack([item[0] for item in x])}
    )
    
    # Create training config
    train_config = TrainingConfig(
        learning_rate=1e-3,
        max_steps=50,
        warmup_steps=10,
        batch_size=4,
        gradient_accumulation_steps=2,
        log_interval=5,
        eval_interval=20,
        checkpoint_interval=25,
        eval_steps=5
    )
    
    print(f"\nTraining config:")
    print(f"  Max steps: {train_config.max_steps}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    
    # Create trainer
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=train_config,
            checkpoint_dir=temp_dir,
            device='cpu'
        )
        
        print(f"\n{'='*80}")
        print("Running Training...")
        print(f"{'='*80}\n")
        
        # Train
        trainer.train()
        
        print(f"\n{'='*80}")
        print("✅ Training completed successfully!")
        print(f"{'='*80}")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n🗑️  Temporary directory cleaned up")
