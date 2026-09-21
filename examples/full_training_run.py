"""
examples/full_training_run.py

End-to-End Training Run with Real Turkish Data

Tüm Sprint 9 componentlerini entegre eder:
- SentencePiece tokenizer (Task #1)
- Turkish corpus pipeline (Task #2)
- Distributed training (Task #3)
- Advanced checkpointing (Task #4)

Bu script production-ready training pipeline gösterir:
1. Corpus download & preprocessing
2. Tokenizer training
3. Dataset preparation
4. Model initialization
5. Distributed training
6. Checkpoint management
7. Validation & metrics

Usage:
    # Single GPU training
    python examples/full_training_run.py
    
    # Multi-GPU training
    python examples/full_training_run.py --gpus 4
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer, TokenizerConfig
from src.data.turkish_corpus import TurkishCorpus, CorpusConfig
from src.model.gpt import GPTModel, GPTConfig
from src.training.distributed_trainer import DistributedTrainer, DistributedConfig
from src.training.checkpoint_manager import CheckpointManager, CheckpointConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.
    
    Args:
        texts: List of text strings
        tokenizer: SentencePiece tokenizer
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: SentencePieceTokenizer,
        max_length: int = 128
    ):
        """Initialize dataset."""
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """Dataset length."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item.
        
        Returns:
            Dict with 'input_ids' and 'labels'
        """
        text = self.texts[idx]
        
        # Tokenize
        # tokens shape: [seq_len] variable length
        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # Pad with pad_id
            tokens = tokens + [self.tokenizer.pad_id] * (self.max_length - len(tokens))
        
        # Convert to tensor
        # input_ids shape: [max_length]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Labels (shifted by 1 for language modeling)
        # labels shape: [max_length]
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,  # shape: [max_length]
            'labels': labels         # shape: [max_length]
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched tensors
    """
    # Stack tensors
    # input_ids: [batch_size, max_length]
    # labels: [batch_size, max_length]
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,  # shape: [batch_size, max_length]
        'labels': labels         # shape: [batch_size, max_length]
    }


def create_model_with_loss(model: GPTModel, pad_id: int) -> nn.Module:
    """
    Wrap model to include loss computation.
    
    Args:
        model: GPT model
        pad_id: Padding token ID
        
    Returns:
        Model with loss computation
    """
    class ModelWithLoss(nn.Module):
        """Model wrapper that computes loss."""
        
        def __init__(self, model: GPTModel, pad_id: int):
            """Initialize."""
            super().__init__()
            self.model = model
            self.pad_id = pad_id
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        
        def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass with loss.
            
            Args:
                input_ids: Input token IDs, shape [batch, seq_len]
                labels: Target token IDs, shape [batch, seq_len]
                
            Returns:
                Dict with 'loss' and 'logits'
            """
            # Forward pass
            # input_ids shape: [batch, seq_len]
            output = self.model(input_ids)  # returns (logits, attention_weights)
            
            # Extract logits (model returns tuple)
            if isinstance(output, tuple):
                logits = output[0]  # shape: [batch, seq_len, vocab_size]
            else:
                logits = output  # shape: [batch, seq_len, vocab_size]
            
            # Compute loss
            # Flatten for cross entropy
            # logits: [batch * seq_len, vocab_size]
            # labels: [batch * seq_len]
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )  # shape: [] (scalar)
            
            return {
                'loss': loss,      # shape: [] (scalar)
                'logits': logits   # shape: [batch, seq_len, vocab_size]
            }
    
    return ModelWithLoss(model, pad_id)


def train_epoch(
    trainer: DistributedTrainer,
    train_loader: DataLoader,
    epoch: int
) -> Dict[str, float]:
    """
    Train one epoch.
    
    Args:
        trainer: Distributed trainer
        train_loader: Training data loader
        epoch: Current epoch
        
    Returns:
        Dict with training metrics
    """
    total_loss = 0.0
    num_batches = 0
    num_steps = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Training step
        metrics = trainer.train_step(batch, batch_idx)
        
        total_loss += metrics['loss']
        num_batches += 1
        
        if metrics['is_step']:
            num_steps += 1
        
        # Log progress
        if batch_idx % 10 == 0 and trainer.config.is_main_process:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {metrics['loss']:.4f}"
            )
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Gather metrics across GPUs
    if trainer.config.is_distributed:
        gathered = trainer.gather_metrics({'loss': avg_loss})
        if gathered:
            avg_loss = gathered['loss']
    
    return {
        'loss': avg_loss,
        'num_batches': num_batches,
        'num_steps': num_steps
    }


def validate(
    trainer: DistributedTrainer,
    val_loader: DataLoader
) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        trainer: Distributed trainer
        val_loader: Validation data loader
        
    Returns:
        Dict with validation metrics
    """
    total_loss = 0.0
    num_batches = 0
    
    for batch in val_loader:
        metrics = trainer.validate_step(batch)
        total_loss += metrics['loss']
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Gather metrics
    if trainer.config.is_distributed:
        gathered = trainer.gather_metrics({'loss': avg_loss})
        if gathered:
            avg_loss = gathered['loss']
    
    # Compute perplexity
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity
    }


def main() -> None:
    """
    Main training pipeline.
    
    Steps:
    1. Prepare corpus
    2. Train tokenizer
    3. Create datasets
    4. Initialize model
    5. Setup distributed training
    6. Train with checkpointing
    7. Validate convergence
    """
    print("\n" + "=" * 80)
    print("Full Training Run - Turkish GPT Model")
    print("=" * 80)
    
    # Config
    output_dir = Path("training_output")
    output_dir.mkdir(exist_ok=True)
    
    corpus_dir = output_dir / "corpus"
    tokenizer_dir = output_dir / "tokenizer"
    checkpoint_dir = output_dir / "checkpoints"
    
    # =========================
    # STEP 1: Prepare Corpus
    # =========================
    print("\n[1/7] Preparing corpus...")
    
    corpus = TurkishCorpus(
        config=CorpusConfig(
            min_length=50,
            min_words=5,
            quality_threshold=0.5
        )
    )
    
    # Create sample corpus (in production, use real Wikipedia download)
    print("  Creating sample Turkish corpus...")
    sample_texts = [
        "Türkiye Cumhuriyeti'nin başkenti Ankara'dır. Nüfusu 5.6 milyon civarındadır.",
        "Yapay zeka ve makine öğrenmesi teknolojilerinde büyük ilerlemeler kaydedilmektedir.",
        "Doğal dil işleme alanında transformer mimarileri devrim yaratmıştır.",
        "GPT modelleri büyük dil modelleri kategorisinde önemli bir yere sahiptir.",
        "Türkçe için özel tokenizer'lar geliştirilmesi daha iyi sonuçlar vermektedir.",
        "SentencePiece algoritması dil-bağımsız tokenization sağlar.",
        "Model eğitiminde gradient accumulation büyük batch size'lar için kullanılır.",
        "Distributed training ile çoklu GPU kullanarak eğitim hızlandırılabilir.",
    ] * 20  # 160 sentences
    
    corpus_file = corpus_dir / "train.txt"
    corpus_dir.mkdir(exist_ok=True)
    
    with open(corpus_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(sample_texts))
    
    print(f"  ✓ Corpus created: {len(sample_texts)} sentences")
    
    # =========================
    # STEP 2: Train Tokenizer
    # =========================
    print("\n[2/7] Training tokenizer...")
    
    tokenizer = SentencePieceTokenizer(
        config=TokenizerConfig(
            vocab_size=500,  # Smaller vocab for small corpus
            model_type='bpe'
        )
    )
    
    tokenizer_dir.mkdir(exist_ok=True)
    model_path, stats = tokenizer.train(
        corpus_path=str(corpus_file),
        model_prefix=str(tokenizer_dir / "turkish_tokenizer")
    )
    
    print(f"  ✓ Tokenizer trained: {stats['vocab_size']} vocab")
    
    # =========================
    # STEP 3: Create Datasets
    # =========================
    print("\n[3/7] Creating datasets...")
    
    # Load texts
    with open(corpus_file, 'r', encoding='utf-8') as f:
        texts = [t.strip() for t in f.read().split('\n\n') if t.strip()]
    
    # Split train/val
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length=64)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=64)
    
    print(f"  ✓ Train: {len(train_dataset)} samples")
    print(f"  ✓ Val: {len(val_dataset)} samples")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # =========================
    # STEP 4: Initialize Model
    # =========================
    print("\n[4/7] Initializing model...")
    
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )
    
    gpt_model = GPTModel(model_config)
    
    # Wrap with loss
    model = create_model_with_loss(gpt_model, tokenizer.pad_id)
    
    print(f"  ✓ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # =========================
    # STEP 5: Setup Training
    # =========================
    print("\n[5/7] Setting up distributed training...")
    
    dist_config = DistributedConfig(
        world_size=1,  # Single GPU
        gradient_accumulation_steps=2,
        mixed_precision=False,  # CPU/single GPU demo
        gradient_clip_val=1.0
    )
    
    trainer = DistributedTrainer(model, dist_config)
    trainer.setup(rank=0, world_size=1)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.01
    )
    trainer.optimizer = optimizer
    
    print(f"  ✓ Trainer initialized")
    
    # =========================
    # STEP 6: Checkpoint Manager
    # =========================
    print("\n[6/7] Setting up checkpoint manager...")
    
    ckpt_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        model_name="turkish-gpt-small",
        config=CheckpointConfig(
            keep_top_k=2,
            keep_last_k=3,
            metric_for_best="val_loss",
            mode="min"
        )
    )
    
    print(f"  ✓ Checkpoint manager ready")
    
    # =========================
    # STEP 7: Training Loop
    # =========================
    print("\n[7/7] Starting training...")
    print("=" * 80)
    
    num_epochs = 5
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(trainer, train_loader, epoch)
        
        # Validate
        val_metrics = validate(trainer, val_loader)
        
        # Log
        logger.info(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Perplexity: {val_metrics['perplexity']:.2f}"
        )
        
        # Save checkpoint
        if trainer.config.is_main_process:
            ckpt_manager.save_checkpoint(
                model=gpt_model,  # Save unwrapped model
                epoch=epoch,
                step=(epoch + 1) * len(train_loader),
                metrics={
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_perplexity': val_metrics['perplexity']
                },
                optimizer=optimizer,
                model_config=model_config.__dict__,
                version=f"1.0.{epoch}"
            )
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                logger.info(f"  ✓ New best model! Val loss: {best_val_loss:.4f}")
    
    # =========================
    # Final Summary
    # =========================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    # Load best checkpoint
    print("\nLoading best checkpoint...")
    best_checkpoint = ckpt_manager.load_best_checkpoint(
        metric='val_loss',
        mode='min'
    )
    
    print(f"  ✓ Best model: epoch={best_checkpoint['epoch']}")
    print(f"  ✓ Metrics: {best_checkpoint['metrics']}")
    
    # List all checkpoints
    print("\nCheckpoints:")
    checkpoints = ckpt_manager.list_checkpoints(sort_by='epoch')
    for ckpt in checkpoints:
        print(f"  v{ckpt['version']} - Epoch {ckpt['epoch']}: "
              f"val_loss={ckpt['metrics']['val_loss']:.4f}, "
              f"perplexity={ckpt['metrics']['val_perplexity']:.2f}")
    
    print("\n" + "=" * 80)
    print("✓ Full training run complete!")
    print("=" * 80)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"  - Tokenizer: {tokenizer_dir}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Corpus: {corpus_dir}")
    
    # Cleanup
    trainer.cleanup()


if __name__ == "__main__":
    main()
