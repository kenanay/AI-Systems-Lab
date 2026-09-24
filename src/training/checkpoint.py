"""
src/training/checkpoint.py

Training Checkpoint Manager

Bu modül training checkpoint'lerini yönetir:
- Model state save/load
- Optimizer state save/load
- Training metadata (epoch, step, metrics)
- Best model tracking
- Checkpoint rotation (disk space yönetimi)

Checkpoint'ler training'i resume etmek ve best model'i saklamak için kullanılır.

Checkpoint Format:
    {
        'epoch': int,
        'global_step': int,
        'model_state_dict': OrderedDict,
        'optimizer_state_dict': dict,
        'scheduler_state_dict': dict (optional),
        'metrics': dict,
        'config': dict,
        'timestamp': str
    }
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Checkpoint Manager
    
    Training checkpoint'lerini save/load/manage eder.
    
    Args:
        checkpoint_dir: Checkpoint'lerin kaydedileceği dizin
        max_checkpoints: Maximum checkpoint sayısı (eski olanlar silinir)
        save_best: Best model'i ayrıca kaydet
        best_metric: Best model için metric ('loss', 'perplexity', 'accuracy')
        best_mode: Best metric modu ('min' veya 'max')
    
    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir='checkpoints',
        ...     max_checkpoints=3,
        ...     save_best=True,
        ...     best_metric='loss',
        ...     best_mode='min'
        ... )
        >>> manager.save_checkpoint(model, optimizer, epoch=0, metrics={'loss': 2.5})
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        max_checkpoints: int = 3,
        save_best: bool = True,
        best_metric: str = 'loss',
        best_mode: str = 'min'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        
        # Track best metric value
        self.best_metric_value = float('inf') if best_mode == 'min' else float('-inf')
        
        # Track saved checkpoints for rotation
        self.checkpoint_files = []
        
        logger.info(
            f"CheckpointManager initialized: dir={checkpoint_dir}, "
            f"max={max_checkpoints}, best_metric={best_metric} ({best_mode})"
        )
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        prefix: str = 'checkpoint'
    ) -> str:
        """
        Save training checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            global_step: Global training step
            metrics: Training metrics dict
            scheduler: LR scheduler (optional)
            config: Model/training config (optional)
            prefix: Checkpoint file prefix
        
        Returns:
            Checkpoint file path
        """
        # Create checkpoint dict
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add config if provided
        if config is not None:
            checkpoint['config'] = config
        
        # Generate checkpoint filename
        checkpoint_filename = f"{prefix}_epoch{epoch}_step{global_step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(
            f"Checkpoint saved: {checkpoint_path} "
            f"(epoch={epoch}, step={global_step}, metrics={metrics})"
        )
        
        # Track checkpoint for rotation
        self.checkpoint_files.append(checkpoint_path)
        self._rotate_checkpoints()
        
        # Check if this is best model
        if self.save_best and self.best_metric in metrics:
            self._maybe_save_best(checkpoint, metrics[self.best_metric])
        
        return str(checkpoint_path)
    
    def _maybe_save_best(self, checkpoint: Dict, metric_value: float):
        """
        Save checkpoint if it's the best so far.
        
        Args:
            checkpoint: Checkpoint dict
            metric_value: Current metric value
        """
        is_best = False
        
        if self.best_mode == 'min':
            is_best = metric_value < self.best_metric_value
        else:  # max
            is_best = metric_value > self.best_metric_value
        
        if is_best:
            self.best_metric_value = metric_value
            
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
            logger.info(
                f"Best model saved: {best_path} "
                f"({self.best_metric}={metric_value:.4f})"
            )
    
    def _rotate_checkpoints(self):
        """
        Remove old checkpoints to maintain max_checkpoints limit.
        """
        if len(self.checkpoint_files) > self.max_checkpoints:
            # Remove oldest checkpoints
            num_to_remove = len(self.checkpoint_files) - self.max_checkpoints
            
            for _ in range(num_to_remove):
                old_checkpoint = self.checkpoint_files.pop(0)
                
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore model/optimizer states.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: PyTorch model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: LR scheduler to load state into (optional)
            device: Device to load checkpoint to
        
        Returns:
            Checkpoint dict with metadata
        
        Example:
            >>> checkpoint = manager.load_checkpoint(
            ...     'checkpoints/checkpoint_epoch5.pt',
            ...     model=model,
            ...     optimizer=optimizer
            ... )
            >>> start_epoch = checkpoint['epoch'] + 1
        """
        path = Path(checkpoint_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model state loaded from {path}")
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded")
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Scheduler state loaded")
        
        logger.info(
            f"Checkpoint loaded: epoch={checkpoint.get('epoch', 'N/A')}, "
            f"step={checkpoint.get('global_step', 'N/A')}"
        )
        
        return checkpoint
    
    def load_best_model(
        self,
        model: nn.Module,
        device: str = 'cpu'
    ) -> Optional[Dict[str, Any]]:
        """
        Load best model checkpoint.
        
        Args:
            model: PyTorch model to load state into
            device: Device to load checkpoint to
        
        Returns:
            Checkpoint dict or None if best model doesn't exist
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        
        if not best_path.exists():
            logger.warning("Best model checkpoint not found")
            return None
        
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(
            f"Best model loaded: {self.best_metric}="
            f"{checkpoint['metrics'].get(self.best_metric, 'N/A')}"
        )
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to latest checkpoint.
        
        Returns:
            Latest checkpoint path or None
        """
        if not self.checkpoint_files:
            # Check directory for existing checkpoints
            checkpoint_pattern = self.checkpoint_dir.glob('checkpoint_*.pt')
            checkpoints = sorted(checkpoint_pattern, key=lambda p: p.stat().st_mtime)
            
            if checkpoints:
                return str(checkpoints[-1])
            
            return None
        
        return str(self.checkpoint_files[-1])
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint paths
        """
        checkpoint_pattern = self.checkpoint_dir.glob('checkpoint_*.pt')
        checkpoints = sorted(checkpoint_pattern, key=lambda p: p.stat().st_mtime)
        
        return [str(p) for p in checkpoints]
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str = 'metadata.json'):
        """
        Save training metadata to JSON file.
        
        Args:
            metadata: Metadata dict
            filename: Metadata filename
        """
        metadata_path = self.checkpoint_dir / filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved: {metadata_path}")
    
    def load_metadata(self, filename: str = 'metadata.json') -> Optional[Dict[str, Any]]:
        """
        Load training metadata from JSON file.
        
        Args:
            filename: Metadata filename
        
        Returns:
            Metadata dict or None
        """
        metadata_path = self.checkpoint_dir / filename
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Metadata loaded: {metadata_path}")
        
        return metadata


def save_model_only(
    model: nn.Module,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save only model weights (no optimizer, no training state).
    
    Useful for inference deployment.
    
    Args:
        model: PyTorch model
        save_path: Path to save model
        metadata: Optional metadata to include
    
    Example:
        >>> save_model_only(model, 'models/gpt_turkish.pt')
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, save_path)
    logger.info(f"Model saved: {save_path}")


def load_model_only(
    model: nn.Module,
    load_path: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load only model weights.
    
    Args:
        model: PyTorch model
        load_path: Path to load model from
        device: Device to load to
    
    Returns:
        Loaded dict with metadata
    
    Example:
        >>> load_model_only(model, 'models/gpt_turkish.pt')
    """
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Model loaded: {load_path}")
    
    return checkpoint


if __name__ == "__main__":
    # Test checkpoint manager
    logging.basicConfig(level=logging.INFO)
    
    from src.model.gpt import GPTModel, GPTConfig
    from src.training.optimizer import create_optimizer, get_warmup_cosine_schedule
    
    print("=" * 80)
    print("Checkpoint Manager Test")
    print("=" * 80)
    
    # Create temporary checkpoint directory
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = os.path.join(temp_dir, 'test_checkpoints')
    
    print(f"\nTemporary checkpoint directory: {checkpoint_dir}")
    
    try:
        # Create model and optimizer
        config = GPTConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4
        )
        
        model = GPTModel(config)
        optimizer = create_optimizer(model, learning_rate=3e-4)
        scheduler = get_warmup_cosine_schedule(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=1000
        )
        
        # Test 1: Create checkpoint manager
        print(f"\n{'='*80}")
        print("[1] Checkpoint Manager Creation")
        print(f"{'='*80}")
        
        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=2,
            save_best=True,
            best_metric='loss',
            best_mode='min'
        )
        
        print(f"✅ Manager created")
        
        # Test 2: Save checkpoints
        print(f"\n{'='*80}")
        print("[2] Save Checkpoints")
        print(f"{'='*80}")
        
        checkpoint_paths = []
        
        for epoch in range(3):
            metrics = {
                'loss': 3.0 - epoch * 0.5,  # Decreasing loss
                'perplexity': 20.0 - epoch * 2.0,
                'accuracy': 0.1 + epoch * 0.05
            }
            
            path = manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=epoch * 100,
                metrics=metrics,
                scheduler=scheduler,
                config=config.to_dict()
            )
            
            checkpoint_paths.append(path)
            print(f"  Epoch {epoch}: {os.path.basename(path)}")
        
        print(f"✅ Saved 3 checkpoints")
        
        # Test 3: Checkpoint rotation
        print(f"\n{'='*80}")
        print("[3] Checkpoint Rotation")
        print(f"{'='*80}")
        
        checkpoints = manager.list_checkpoints()
        print(f"  Total checkpoints: {len(checkpoints)}")
        print(f"  Max checkpoints: {manager.max_checkpoints}")
        
        assert len(checkpoints) <= manager.max_checkpoints
        print(f"✅ Rotation working (kept {len(checkpoints)} checkpoints)")
        
        # Test 4: Load checkpoint
        print(f"\n{'='*80}")
        print("[4] Load Checkpoint")
        print(f"{'='*80}")
        
        # Create new model and optimizer
        model2 = GPTModel(config)
        optimizer2 = create_optimizer(model2, learning_rate=3e-4)
        scheduler2 = get_warmup_cosine_schedule(
            optimizer2,
            num_warmup_steps=100,
            num_training_steps=1000
        )
        
        # Load latest checkpoint
        latest = manager.get_latest_checkpoint()
        assert latest is not None, "Latest checkpoint should not be None"
        print(f"  Latest checkpoint: {os.path.basename(latest)}")
        
        checkpoint = manager.load_checkpoint(
            latest,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2
        )
        
        print(f"  Loaded epoch: {checkpoint['epoch']}")
        print(f"  Loaded step: {checkpoint['global_step']}")
        print(f"  Loaded metrics: {checkpoint['metrics']}")
        print(f"✅ Checkpoint loaded successfully")
        
        # Test 5: Best model
        print(f"\n{'='*80}")
        print("[5] Best Model")
        print(f"{'='*80}")
        
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        print(f"  Best model exists: {os.path.exists(best_path)}")
        
        if os.path.exists(best_path):
            model3 = GPTModel(config)
            best_checkpoint = manager.load_best_model(model3)
            assert best_checkpoint is not None
            print(f"  Best loss: {best_checkpoint['metrics']['loss']:.4f}")
            print(f"  Best epoch: {best_checkpoint['epoch']}")
            print(f"✅ Best model loaded")
        
        # Test 6: Model-only save/load
        print(f"\n{'='*80}")
        print("[6] Model-only Save/Load")
        print(f"{'='*80}")
        
        model_path = os.path.join(temp_dir, 'model_only.pt')
        
        save_model_only(
            model,
            model_path,
            metadata={'description': 'Test model', 'version': '1.0'}
        )
        
        print(f"  Model saved: {os.path.basename(model_path)}")
        
        model4 = GPTModel(config)
        loaded = load_model_only(model4, model_path)
        
        print(f"  Model loaded")
        print(f"  Metadata: {loaded.get('metadata', 'None')}")
        print(f"✅ Model-only save/load working")
        
        # Test 7: Metadata save/load
        print(f"\n{'='*80}")
        print("[7] Metadata Save/Load")
        print(f"{'='*80}")
        
        metadata = {
            'model_name': 'GPT-Turkish',
            'training_steps': 10000,
            'final_loss': 1.5,
            'vocab_size': 8000
        }
        
        manager.save_metadata(metadata)
        loaded_metadata = manager.load_metadata()
        assert loaded_metadata is not None
        print(f"  Metadata saved and loaded")
        print(f"  Model name: {loaded_metadata['model_name']}")
        print(f"✅ Metadata working")
        
        print(f"\n" + "=" * 80)
        print("✅ All tests PASSED!")
        print("=" * 80)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n🗑️  Temporary directory cleaned up")
