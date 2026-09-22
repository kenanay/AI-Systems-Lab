"""
src/training/checkpoint_manager.py

Production Checkpoint Management System

Model versioning, artifact management, metadata tracking ve resume capability.

Bu modül production-ready checkpoint management sağlar:
- Model Versioning: Semantic versioning (v1.0.0)
- Artifact Management: Checkpoint organization ve cleanup
- Metadata Tracking: Training metrics, config, provenance
- Resume Capability: Training state restoration
- Best Model Tracking: Metric-based best checkpoint selection

Features:
- Semantic versioning with tags
- Checkpoint lifecycle management (save, load, delete, list)
- Metadata rich checkpoints (config, metrics, git commit, timestamp)
- Best model tracking with custom metrics
- Automatic cleanup policies (keep top-k, age-based)
- Artifact integrity validation

Kaynaklar:
    - MLOps Best Practices for Model Management
    - PyTorch Save/Load Best Practices
    - Semantic Versioning 2.0.0: https://semver.org

Usage:
    >>> from src.training.checkpoint_manager import CheckpointManager, CheckpointConfig
    >>> 
    >>> manager = CheckpointManager(
    ...     checkpoint_dir="checkpoints",
    ...     model_name="gpt-turkish"
    ... )
    >>> 
    >>> # Save checkpoint
    >>> manager.save_checkpoint(
    ...     model=model,
    ...     optimizer=optimizer,
    ...     epoch=10,
    ...     metrics={'loss': 2.5, 'perplexity': 12.2},
    ...     version="1.0.0"
    ... )
    >>> 
    >>> # Load best checkpoint
    >>> checkpoint = manager.load_best_checkpoint(metric='loss', mode='min')
    >>> model.load_state_dict(checkpoint['model_state_dict'])
"""

import torch
import logging
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib
import re

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """
    Checkpoint metadata.
    
    Attributes:
        version: Model version (semantic versioning: v1.0.0)
        model_name: Model name/identifier
        epoch: Training epoch
        step: Training step
        timestamp: Checkpoint creation timestamp
        metrics: Training metrics dict
        config: Model/training configuration
        git_commit: Git commit hash (optional)
        tags: Custom tags list
        checkpoint_hash: Checkpoint file hash (for integrity)
        file_size_mb: Checkpoint file size in MB
    """
    version: str
    model_name: str
    epoch: int
    step: int
    timestamp: str
    metrics: Dict[str, float]
    config: Dict[str, Any]
    git_commit: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    checkpoint_hash: Optional[str] = None
    file_size_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e dönüştür."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Dict'ten oluştur."""
        return cls(**data)


@dataclass
class CheckpointConfig:
    """
    Checkpoint manager konfigürasyonu.
    
    Attributes:
        keep_top_k: Top-k checkpoint'leri sakla (None = hepsini sakla)
        keep_last_k: Son k checkpoint'i sakla
        metric_for_best: Best model için kullanılacak metric
        mode: Metric mode ('min' veya 'max')
        save_optimizer: Optimizer state'i kaydet
        save_scheduler: Scheduler state'i kaydet
        save_scaler: AMP scaler state'i kaydet
        auto_cleanup: Otomatik cleanup aktif
    """
    keep_top_k: Optional[int] = 3
    keep_last_k: Optional[int] = 5
    metric_for_best: str = "loss"
    mode: str = "min"  # 'min' or 'max'
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_scaler: bool = True
    auto_cleanup: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e dönüştür."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointConfig':
        """Dict'ten oluştur."""
        return cls(**data)


class CheckpointManager:
    """
    Production checkpoint management system.
    
    Model versioning, artifact management ve metadata tracking.
    
    Features:
    - Semantic versioning: v1.0.0, v1.0.1, v1.1.0
    - Metadata tracking: Metrics, config, git commit, timestamp
    - Best model tracking: Metric-based selection
    - Cleanup policies: Keep top-k, keep last-k
    - Resume capability: Full training state restoration
    
    Args:
        checkpoint_dir: Checkpoint directory path
        model_name: Model name/identifier
        config: CheckpointConfig instance
    
    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir="checkpoints",
        ...     model_name="gpt-turkish"
        ... )
        >>> 
        >>> # Save
        >>> manager.save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     metrics={'loss': 2.5},
        ...     version="1.0.0"
        ... )
        >>> 
        >>> # Load best
        >>> checkpoint = manager.load_best_checkpoint(metric='loss')
        >>> 
        >>> # List all
        >>> checkpoints = manager.list_checkpoints()
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str,
        config: Optional[CheckpointConfig] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Checkpoint directory
            model_name: Model identifier
            config: CheckpointConfig (None = default)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.config = config or CheckpointConfig()
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.checkpoint_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        logger.info(f"CheckpointManager initialized")
        logger.info(f"  Directory: {self.checkpoint_dir}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Keep top-k: {self.config.keep_top_k}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        model_config: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Tuple[str, CheckpointMetadata]:
        """
        Checkpoint kaydet.
        
        Args:
            model: PyTorch model
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            optimizer: Optimizer (optional)
            scheduler: LR scheduler (optional)
            scaler: AMP scaler (optional)
            model_config: Model configuration (optional)
            version: Model version (None = auto-generate)
            tags: Custom tags (optional)
            
        Returns:
            Tuple[str, CheckpointMetadata]: (checkpoint_path, metadata)
            
        Example:
            >>> path, metadata = manager.save_checkpoint(
            ...     model=model,
            ...     epoch=10,
            ...     step=1000,
            ...     metrics={'loss': 2.5, 'accuracy': 0.85},
            ...     optimizer=optimizer,
            ...     version="1.0.0"
            ... )
        """
        # Version
        if version is None:
            version = self._generate_version()
        
        # Validate version format
        if not self._validate_version(version):
            logger.warning(f"Invalid version format: {version}, using auto-generated")
            version = self._generate_version()
        
        # Checkpoint filename
        checkpoint_name = f"{self.model_name}_v{version}_epoch{epoch}_step{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        logger.info(f"Saving checkpoint: {checkpoint_name}")
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': model_config or {},
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'manager_version': self.VERSION
        }
        
        # Optional states
        if optimizer and self.config.save_optimizer:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler and self.config.save_scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        if scaler and self.config.save_scaler:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Calculate file hash
        checkpoint_hash = self._calculate_file_hash(checkpoint_path)
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
        # Create metadata
        metadata = CheckpointMetadata(
            version=version,
            model_name=self.model_name,
            epoch=epoch,
            step=step,
            timestamp=checkpoint_data['timestamp'],
            metrics=metrics,
            config=model_config or {},
            git_commit=self._get_git_commit(),
            tags=tags or [],
            checkpoint_hash=checkpoint_hash,
            file_size_mb=round(file_size_mb, 2)
        )
        
        # Save metadata
        metadata_path = self.metadata_dir / f"{checkpoint_name}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Checkpoint saved: {checkpoint_path}")
        logger.info(f"  Size: {file_size_mb:.2f} MB")
        logger.info(f"  Metrics: {metrics}")
        
        # Auto cleanup
        if self.config.auto_cleanup:
            self._cleanup_old_checkpoints()
        
        return str(checkpoint_path), metadata
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Checkpoint yükle.
        
        Args:
            checkpoint_path: Checkpoint dosya yolu
            model: Model (state dict buraya yüklenir)
            optimizer: Optimizer (state dict buraya yüklenir)
            scheduler: Scheduler to load state into (optional)
            scaler: GradScaler to load state into (optional)
            strict: Strict loading mode for model state dict
            
        Returns:
            Dict: Checkpoint içeriği
            
        Example:
            >>> checkpoint = manager.load_checkpoint(
            ...     "checkpoints/model_v1.0.0_epoch10.pt",
            ...     model=model,
            ...     optimizer=optimizer
            ... )
            >>> print(f"Resumed from epoch {checkpoint['epoch']}")
        """
        path = Path(checkpoint_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        logger.info(f"Loading checkpoint: {path.name}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location='cpu')
        
        # Verify integrity
        metadata_path = self.metadata_dir / f"{path.name}.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Hash verification
            current_hash = self._calculate_file_hash(path)
            if metadata.get('checkpoint_hash') != current_hash:
                logger.warning(f"Checkpoint hash mismatch! File may be corrupted.")
        
        # Load states
        if model and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            logger.info(f"  Model state loaded")
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"  Optimizer state loaded")
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"  Scheduler state loaded")
        
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"  Scaler state loaded")
        
        logger.info(f"  Checkpoint loaded: epoch={checkpoint.get('epoch')}, step={checkpoint.get('step')}")
        
        return checkpoint
    
    def load_best_checkpoint(
        self,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Best checkpoint'i yükle.
        
        Args:
            metric: Metric name (None = use config default)
            mode: 'min' or 'max' (None = use config default)
            model: Model to load into (optional)
            optimizer: Optimizer to load into (optional)
            **kwargs: Additional args for load_checkpoint
            
        Returns:
            Dict: Checkpoint data
            
        Example:
            >>> # Load best by validation loss (minimize)
            >>> checkpoint = manager.load_best_checkpoint(
            ...     metric='val_loss',
            ...     mode='min',
            ...     model=model
            ... )
        """
        metric = metric or self.config.metric_for_best
        mode = mode or self.config.mode
        
        # Find best checkpoint
        best_checkpoint_path = self._find_best_checkpoint(metric, mode)
        
        if best_checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found with metric: {metric}")
        
        logger.info(f"Loading best checkpoint (metric={metric}, mode={mode})")
        
        return self.load_checkpoint(
            best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            **kwargs
        )
    
    def list_checkpoints(
        self,
        sort_by: str = "timestamp",
        reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Checkpoint listesi.
        
        Args:
            sort_by: Sort field ('timestamp', 'epoch', 'step', metric name)
            reverse: Reverse sort order
            
        Returns:
            List[Dict]: Checkpoint metadata list
            
        Example:
            >>> checkpoints = manager.list_checkpoints(sort_by='epoch')
            >>> for ckpt in checkpoints:
            >>>     print(f"{ckpt['version']}: {ckpt['metrics']}")
        """
        checkpoints = []
        
        # Load all metadata files
        for metadata_file in self.metadata_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                checkpoints.append(metadata)
        
        # Sort
        if sort_by in ['epoch', 'step', 'timestamp']:
            checkpoints.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
        elif sort_by in checkpoints[0].get('metrics', {}) if checkpoints else {}:
            checkpoints.sort(
                key=lambda x: x.get('metrics', {}).get(sort_by, float('inf')),
                reverse=reverse
            )
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Checkpoint sil.
        
        Args:
            checkpoint_path: Checkpoint file path
        """
        path = Path(checkpoint_path)
        
        if path.exists():
            path.unlink()
            logger.info(f"Deleted checkpoint: {path.name}")
        
        # Delete metadata
        metadata_path = self.metadata_dir / f"{path.name}.json"
        if metadata_path.exists():
            metadata_path.unlink()
    
    def _cleanup_old_checkpoints(self) -> None:
        """
        Eski checkpoint'leri temizle.
        
        Keep top-k ve keep last-k politikalarını uygular.
        """
        checkpoints = self.list_checkpoints(sort_by='timestamp', reverse=True)
        
        if not checkpoints:
            return
        
        # Keep last-k
        if self.config.keep_last_k:
            checkpoints_to_keep = set()
            for ckpt in checkpoints[:self.config.keep_last_k]:
                ckpt_name = f"{ckpt['model_name']}_v{ckpt['version']}_epoch{ckpt['epoch']}_step{ckpt['step']}.pt"
                checkpoints_to_keep.add(ckpt_name)
        else:
            checkpoints_to_keep = set()
        
        # Keep top-k by metric
        if self.config.keep_top_k and self.config.metric_for_best:
            metric = self.config.metric_for_best
            mode = self.config.mode
            
            # Sort by metric
            checkpoints_by_metric = sorted(
                [c for c in checkpoints if metric in c.get('metrics', {})],
                key=lambda x: x['metrics'][metric],
                reverse=(mode == 'max')
            )
            
            for ckpt in checkpoints_by_metric[:self.config.keep_top_k]:
                ckpt_name = f"{ckpt['model_name']}_v{ckpt['version']}_epoch{ckpt['epoch']}_step{ckpt['step']}.pt"
                checkpoints_to_keep.add(ckpt_name)
        
        # Delete checkpoints not in keep set
        deleted_count = 0
        for ckpt in checkpoints:
            ckpt_name = f"{ckpt['model_name']}_v{ckpt['version']}_epoch{ckpt['epoch']}_step{ckpt['step']}.pt"
            ckpt_path = self.checkpoint_dir / ckpt_name
            
            if ckpt_name not in checkpoints_to_keep and ckpt_path.exists():
                self.delete_checkpoint(str(ckpt_path))
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old checkpoints")
    
    def _find_best_checkpoint(
        self,
        metric: str,
        mode: str = "min"
    ) -> Optional[str]:
        """
        Best checkpoint'i bul.
        
        Args:
            metric: Metric name
            mode: 'min' or 'max'
            
        Returns:
            Optional[str]: Best checkpoint path
        """
        checkpoints = self.list_checkpoints()
        
        # Filter checkpoints with metric
        valid_checkpoints = [
            c for c in checkpoints
            if metric in c.get('metrics', {})
        ]
        
        if not valid_checkpoints:
            return None
        
        # Find best
        if mode == 'min':
            best = min(valid_checkpoints, key=lambda x: x['metrics'][metric])
        else:
            best = max(valid_checkpoints, key=lambda x: x['metrics'][metric])
        
        # Construct path
        checkpoint_name = f"{best['model_name']}_v{best['version']}_epoch{best['epoch']}_step{best['step']}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        return str(checkpoint_path) if checkpoint_path.exists() else None
    
    def _generate_version(self) -> str:
        """
        Auto-generate version.
        
        Returns:
            str: Version string (v1.0.0 format)
        """
        # Find latest version
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return "1.0.0"
        
        # Extract versions
        versions = [c['version'] for c in checkpoints]
        
        # Parse semantic versions
        parsed_versions = []
        for v in versions:
            match = re.match(r'(\d+)\.(\d+)\.(\d+)', v)
            if match:
                parsed_versions.append(tuple(map(int, match.groups())))
        
        if not parsed_versions:
            return "1.0.0"
        
        # Increment patch version
        latest = max(parsed_versions)
        new_version = f"{latest[0]}.{latest[1]}.{latest[2] + 1}"
        
        return new_version
    
    def _validate_version(self, version: str) -> bool:
        """
        Validate semantic version format.
        
        Args:
            version: Version string
            
        Returns:
            bool: Valid or not
        """
        pattern = r'^\d+\.\d+\.\d+$'
        return bool(re.match(pattern, version))
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate file MD5 hash.
        
        Args:
            file_path: File path
            
        Returns:
            str: MD5 hash
        """
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_git_commit(self) -> Optional[str]:
        """
        Get current git commit hash.
        
        Returns:
            Optional[str]: Git commit hash
        """
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        return None


def main() -> None:
    """
    Demo ve test fonksiyonu.
    
    Checkpoint manager özelliklerini gösterir:
    - Checkpoint save/load
    - Best model tracking
    - Metadata management
    - Cleanup policies
    """
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 80)
    print("Checkpoint Manager Demo")
    print("=" * 80)
    
    # Simple model
    print("\n1. Creating model...")
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Checkpoint manager
    print("\n2. Initializing checkpoint manager...")
    manager = CheckpointManager(
        checkpoint_dir="demo_checkpoints",
        model_name="demo_model",
        config=CheckpointConfig(
            keep_top_k=2,
            keep_last_k=3,
            metric_for_best="loss",
            mode="min"
        )
    )
    
    # Save multiple checkpoints
    print("\n3. Saving checkpoints...")
    
    for epoch in range(5):
        loss = 5.0 - epoch * 0.5  # Decreasing loss
        accuracy = 0.5 + epoch * 0.1  # Increasing accuracy
        
        path, metadata = manager.save_checkpoint(
            model=model,
            epoch=epoch,
            step=epoch * 100,
            metrics={'loss': loss, 'accuracy': accuracy},
            optimizer=optimizer,
            model_config={'d_model': 64, 'vocab_size': 1000},
            version=f"1.0.{epoch}",
            tags=['demo', f'epoch_{epoch}']
        )
        
        print(f"  Epoch {epoch}: loss={loss:.2f}, accuracy={accuracy:.2f}")
    
    # List checkpoints
    print("\n4. Listing checkpoints...")
    checkpoints = manager.list_checkpoints(sort_by='epoch')
    
    print(f"  Total checkpoints: {len(checkpoints)}")
    for ckpt in checkpoints:
        print(f"    v{ckpt['version']} - Epoch {ckpt['epoch']}: {ckpt['metrics']}")
    
    # Load best checkpoint
    print("\n5. Loading best checkpoint (by loss)...")
    best_checkpoint = manager.load_best_checkpoint(
        metric='loss',
        mode='min',
        model=model,
        optimizer=optimizer
    )
    
    print(f"  Best checkpoint: epoch={best_checkpoint['epoch']}")
    print(f"  Metrics: {best_checkpoint['metrics']}")
    
    # Cleanup
    print("\n6. Cleanup...")
    import shutil
    shutil.rmtree("demo_checkpoints")
    print("  ✓ Demo checkpoints removed")
    
    print("\n" + "=" * 80)
    print("✓ Demo tamamlandı!")
    print("=" * 80)
    
    print("\nProduction usage:")
    print("  manager = CheckpointManager(checkpoint_dir='checkpoints', model_name='gpt')")
    print("  manager.save_checkpoint(model, epoch=10, metrics={'loss': 2.5})")
    print("  checkpoint = manager.load_best_checkpoint(metric='loss', model=model)")


if __name__ == "__main__":
    main()
