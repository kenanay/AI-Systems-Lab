"""
src/registry/model_registry.py

Production Model Registry

Trained model'lerin central repository'si - storage, versioning, metadata, discovery.

Bu modül production model management sağlar:
- Model Registration: Trained model'leri kaydet
- Versioning: Semantic versioning with tags
- Metadata: Model info, metrics, provenance
- Discovery: Model search ve listing
- Loading: Model download ve loading utilities

Features:
- Centralized model storage
- Rich metadata tracking
- Version management
- Tag-based organization
- Model comparison utilities
- Production/staging environments

Kaynaklar:
    - MLflow Model Registry
    - Hugging Face Model Hub
    - AWS SageMaker Model Registry

Usage:
    >>> from src.registry.model_registry import ModelRegistry
    >>> 
    >>> registry = ModelRegistry(registry_dir="models")
    >>> 
    >>> # Register model
    >>> registry.register_model(
    ...     model_name="turkish-gpt",
    ...     version="1.0.0",
    ...     checkpoint_path="checkpoints/model.pt",
    ...     tokenizer_path="tokenizer/tokenizer.model",
    ...     metrics={'val_loss': 0.025, 'perplexity': 1.03},
    ...     tags=['production', 'turkish']
    ... )
    >>> 
    >>> # List models
    >>> models = registry.list_models(tag='production')
    >>> 
    >>> # Load model
    >>> model_info = registry.load_model("turkish-gpt", version="1.0.0")
"""

import torch
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """
    Registered model metadata.
    
    Attributes:
        model_name: Model identifier
        version: Semantic version (v1.0.0)
        description: Model description
        architecture: Model architecture name
        parameters: Parameter count
        metrics: Performance metrics
        training_config: Training configuration
        tokenizer_info: Tokenizer information
        tags: Organizational tags
        environment: 'development', 'staging', 'production'
        created_at: Registration timestamp
        created_by: User/system identifier
        model_hash: Model file hash
        file_size_mb: Model file size
    """
    model_name: str
    version: str
    description: str
    architecture: str
    parameters: int
    metrics: Dict[str, float]
    training_config: Dict[str, Any]
    tokenizer_info: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    environment: str = "development"
    created_at: str = ""
    created_by: str = "system"
    model_hash: Optional[str] = None
    file_size_mb: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Post-initialization."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e dönüştür."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Dict'ten oluştur."""
        return cls(**data)


class ModelRegistry:
    """
    Production model registry.
    
    Central repository for trained models with versioning and metadata.
    
    Features:
    - Model registration with rich metadata
    - Semantic versioning
    - Tag-based organization
    - Environment management (dev/staging/prod)
    - Model discovery and search
    - Loading utilities
    
    Args:
        registry_dir: Registry root directory
    
    Example:
        >>> registry = ModelRegistry("models")
        >>> 
        >>> # Register
        >>> registry.register_model(
        ...     model_name="gpt-turkish",
        ...     version="1.0.0",
        ...     checkpoint_path="checkpoint.pt",
        ...     tokenizer_path="tokenizer.model",
        ...     metrics={'loss': 0.025}
        ... )
        >>> 
        >>> # List
        >>> models = registry.list_models(environment='production')
        >>> 
        >>> # Load
        >>> model_info = registry.load_model("gpt-turkish", "1.0.0")
    """
    
    VERSION = "1.0.0"
    index: Dict[str, Any]
    
    def __init__(self, registry_dir: Union[str, Path] = "model_registry") -> None:
        """
        Initialize model registry.
        
        Args:
            registry_dir: Registry directory path
        """
        from src.security.context import principal
        actor = principal.get()
        self.registry_dir = Path(registry_dir)
        if actor is not None:
            self.registry_dir = Path("models") if actor.role == "admin" else Path("models") / "users" / actor.user_id
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        
        # Registry structure
        self.models_dir = self.registry_dir / "models"
        self.metadata_dir = self.registry_dir / "metadata"
        self.index_file = self.registry_dir / "index.json"
        
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Load or create index
        self._load_index()
        
        logger.info(f"ModelRegistry initialized: {self.registry_dir}")
    
    @staticmethod
    def _validate_identifier(value):
        import re
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) or ".." in value:
            raise ValueError("Invalid model name or version")

    def register_model(
        self,
        model_name: str,
        version: str,
        checkpoint_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        description: str = "",
        architecture: str = "GPT",
        parameters: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        environment: str = "development"
    ) -> ModelMetadata:
        """
        Register a trained model.
        
        Args:
            model_name: Model identifier
            version: Semantic version
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer (optional)
            description: Model description
            architecture: Architecture name
            parameters: Parameter count
            metrics: Performance metrics
            training_config: Training configuration
            tags: Organization tags
            environment: 'development', 'staging', 'production'
            
        Returns:
            ModelMetadata: Registered model metadata
            
        Example:
            >>> metadata = registry.register_model(
            ...     model_name="turkish-gpt-small",
            ...     version="1.0.0",
            ...     checkpoint_path="checkpoints/model.pt",
            ...     metrics={'val_loss': 0.025, 'perplexity': 1.03},
            ...     tags=['production', 'turkish']
            ... )
        """
        self._validate_identifier(model_name)
        self._validate_identifier(version)
        if (self.metadata_dir / f"{model_name}_{version}.json").exists():
            raise ValueError("Model version already exists; choose a new version")
        cp_path = Path(checkpoint_path)
        
        if not cp_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {cp_path}")
        
        logger.info(f"Registering model: {model_name} v{version}")
        
        # Create model directory
        model_dir = self.models_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoint
        checkpoint_dest = model_dir / "model.pt"
        shutil.copy(cp_path, checkpoint_dest)
        
        # Calculate hash and size
        model_hash = self._calculate_hash(checkpoint_dest)
        file_size_mb = checkpoint_dest.stat().st_size / (1024 * 1024)
        
        # Copy tokenizer if provided
        tokenizer_info: Dict[str, Any] = {}
        if tokenizer_path is not None:
            tok_path = Path(tokenizer_path)
            if tok_path.exists():
                tokenizer_dest = model_dir / ("tokenizer" if tok_path.is_dir() else "tokenizer" + tok_path.suffix)
                if tok_path.is_dir():
                    shutil.copytree(tok_path, tokenizer_dest)
                else:
                    shutil.copy(tok_path, tokenizer_dest)
                tokenizer_info['path'] = str(tokenizer_dest.relative_to(self.registry_dir))
                tokenizer_info['type'] = 'bpe' if tok_path.is_dir() or tok_path.suffix == '.json' else 'sentencepiece'
                
                # Copy vocab if exists
                vocab_path = tok_path.with_suffix('.vocab')
                if vocab_path.exists():
                    vocab_dest = model_dir / "tokenizer.vocab"
                    shutil.copy(vocab_path, vocab_dest)
                
                # Copy config if exists
                config_path = tok_path.with_suffix('.config.json')
                if config_path.exists():
                    config_dest = model_dir / "tokenizer.config.json"
                    shutil.copy(config_path, config_dest)
                    
                    with open(config_path, 'r') as f:
                        tokenizer_config = json.load(f)
                        tokenizer_info['vocab_size'] = tokenizer_config.get('vocab_size')
                        tokenizer_info['model_type'] = tokenizer_config.get('model_type')
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            description=description,
            architecture=architecture,
            parameters=parameters or 0,
            metrics=metrics or {},
            training_config=training_config or {},
            tokenizer_info=tokenizer_info,
            tags=tags or [],
            environment=environment,
            model_hash=model_hash,
            file_size_mb=round(file_size_mb, 2)
        )
        
        # Save metadata
        metadata_path = self.metadata_dir / f"{model_name}_{version}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Update index
        self._update_index(model_name, version, metadata)
        
        logger.info(f"  ✓ Model registered: {model_name} v{version}")
        logger.info(f"  Size: {file_size_mb:.2f} MB")
        logger.info(f"  Location: {model_dir}")
        
        return metadata
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        tag: Optional[str] = None,
        environment: Optional[str] = None
    ) -> List[ModelMetadata]:
        """
        List registered models.
        
        Args:
            model_name: Filter by model name (optional)
            tag: Filter by tag (optional)
            environment: Filter by environment (optional)
            
        Returns:
            List[ModelMetadata]: Matching models
            
        Example:
            >>> # All production models
            >>> prod_models = registry.list_models(environment='production')
            >>> 
            >>> # Specific model versions
            >>> versions = registry.list_models(model_name='turkish-gpt')
        """
        models = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                metadata = ModelMetadata.from_dict(data)
                
                # Apply filters
                if model_name and metadata.model_name != model_name:
                    continue
                
                if tag and tag not in metadata.tags:
                    continue
                
                if environment and metadata.environment != environment:
                    continue
                
                models.append(metadata)
        
        # Sort by creation date
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        return models
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        load_weights: bool = False,
        verify_integrity: bool = False
    ) -> Dict[str, Any]:
        """
        Load model metadata and optionally weights.
        
        Args:
            model_name: Model identifier
            version: Version (None = latest)
            load_weights: Load model weights into memory
            verify_integrity: Verify SHA-256 hash before loading
            
        Returns:
            Dict with model info and paths
            
        Example:
            >>> # Load metadata only
            >>> info = registry.load_model("turkish-gpt", "1.0.0")
            >>> print(info['checkpoint_path'])
            >>> 
            >>> # Load weights
            >>> info = registry.load_model("turkish-gpt", load_weights=True)
            >>> model.load_state_dict(info['state_dict'])
        """
        self._validate_identifier(model_name)
        if version is not None:
            self._validate_identifier(version)
        # Find model
        if version is None:
            # Get latest version
            models = self.list_models(model_name=model_name)
            if not models:
                raise ValueError(f"Model not found: {model_name}")
            metadata = models[0]  # Already sorted by date
            version = metadata.version
        else:
            # Load specific version
            metadata_path = self.metadata_dir / f"{model_name}_{version}.json"
            if not metadata_path.exists():
                raise ValueError(f"Model not found: {model_name} v{version}")
            
            with open(metadata_path, 'r') as f:
                metadata = ModelMetadata.from_dict(json.load(f))
        
        # Model paths
        model_dir = self.models_dir / model_name / version
        checkpoint_path = model_dir / "model.pt"
        tokenizer_path = self.registry_dir / metadata.tokenizer_info.get("path", str(model_dir.relative_to(self.registry_dir) / "tokenizer.model"))
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Integrity verification if requested
        if verify_integrity:
            expected_hash = metadata.model_hash
            if expected_hash:
                algo = "md5" if len(expected_hash) == 32 else "sha256"
                actual_hash = self._calculate_hash(checkpoint_path, algorithm=algo)
                if actual_hash.lower() != expected_hash.lower():
                    raise ValueError(
                        f"Model bütünlük doğrulaması başarısız ({model_name} v{version}): "
                        f"beklenen {expected_hash}, hesaplanan {actual_hash}"
                    )
        
        # Build response
        result = {
            'metadata': metadata.to_dict(),
            'checkpoint_path': str(checkpoint_path),
            'tokenizer_path': str(tokenizer_path) if tokenizer_path.exists() else None,
            'model_dir': str(model_dir)
        }
        
        # Load weights if requested
        if load_weights:
            logger.info(f"Loading weights: {model_name} v{version}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            result['state_dict'] = checkpoint.get('model_state_dict', checkpoint)
            result['full_checkpoint'] = checkpoint
        
        logger.info(f"Loaded model: {model_name} v{version}")
        
        return result

    def verify_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Model checkpoint dosyasının SHA-256 bütünlüğünü doğrular.
        
        Diskteki .pt dosyasının hash'ini hesaplar ve metadata'daki model_hash ile karşılaştırır.
        
        Args:
            model_name: Model adı
            version: Versiyon (None = en son versiyon)
            
        Returns:
            Dict: Doğrulama raporu
                - model_name: str
                - version: str
                - verified: bool
                - status: "VALID" | "CORRUPTED"
                - expected_hash: str
                - actual_hash: str
                - file_size_mb: float
                - checkpoint_path: str
                - verified_at: str
        """
        info = self.load_model(model_name=model_name, version=version, load_weights=False)
        metadata = info['metadata']
        checkpoint_path = Path(info['checkpoint_path'])
        
        expected_hash = metadata.get('model_hash') or ""
        algo = "md5" if len(expected_hash) == 32 else "sha256"
        actual_hash = self._calculate_hash(checkpoint_path, algorithm=algo)
        is_valid = bool(expected_hash and actual_hash.lower() == expected_hash.lower())
        
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
        return {
            "model_name": metadata.get('model_name', model_name),
            "version": metadata.get('version', version or "latest"),
            "verified": is_valid,
            "status": "VALID" if is_valid else "CORRUPTED",
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
            "file_size_mb": round(file_size_mb, 2),
            "checkpoint_path": str(checkpoint_path),
            "verified_at": datetime.now(timezone.utc).isoformat()
        }
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """
        Get latest version of a model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Optional[str]: Latest version or None
        """
        models = self.list_models(model_name=model_name)
        if models:
            return models[0].version
        return None

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all registered versions for a model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            List[Dict[str, Any]]: List of version entries
        """
        models_dict: Dict[str, Any] = self.index.get('models', {})
        if model_name in models_dict:
            entry = models_dict[model_name]
            if isinstance(entry, dict):
                return entry.get('versions', [])
        return []
    
    def compare_models(
        self,
        model_specs: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Compare multiple model versions.
        
        Args:
            model_specs: List of (model_name, version) tuples
            
        Returns:
            Dict with comparison data
            
        Example:
            >>> comparison = registry.compare_models([
            ...     ('turkish-gpt', '1.0.0'),
            ...     ('turkish-gpt', '1.0.1')
            ... ])
        """
        comparisons = []
        
        for model_name, version in model_specs:
            info = self.load_model(model_name, version)
            metadata = info['metadata']
            
            comparisons.append({
                'model': f"{model_name} v{version}",
                'parameters': metadata['parameters'],
                'metrics': metadata['metrics'],
                'size_mb': metadata['file_size_mb'],
                'environment': metadata['environment'],
                'created_at': metadata['created_at']
            })
        
        return {
            'models': comparisons,
            'count': len(comparisons)
        }
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        to_environment: str
    ) -> None:
        """
        Promote model to different environment.
        
        Args:
            model_name: Model identifier
            version: Version
            to_environment: Target environment ('staging' or 'production')
            
        Example:
            >>> # Promote to production
            >>> registry.promote_model('turkish-gpt', '1.0.0', 'production')
        """
        metadata_path = self.metadata_dir / f"{model_name}_{version}.json"
        
        if not metadata_path.exists():
            raise ValueError(f"Model not found: {model_name} v{version}")
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        old_env = data['environment']
        data['environment'] = to_environment
        
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Update index
        metadata = ModelMetadata.from_dict(data)
        self._update_index(model_name, version, metadata)
        
        logger.info(f"Promoted {model_name} v{version}: {old_env} → {to_environment}")
    
    def _load_index(self) -> None:
        """Load registry index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {'models': {}, 'version': self.VERSION}
            self._save_index()
    
    def _save_index(self) -> None:
        """Save registry index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)
    
    def _update_index(
        self,
        model_name: str,
        version: str,
        metadata: ModelMetadata
    ) -> None:
        """Update index with new model."""
        models_dict: Dict[str, Any] = self.index.setdefault('models', {})
        if model_name not in models_dict:
            models_dict[model_name] = {'versions': []}
        
        # Add or update version
        model_entry: Dict[str, Any] = models_dict[model_name]
        versions: List[Dict[str, Any]] = model_entry.get('versions', [])
        version_entry: Dict[str, Any] = {
            'version': version,
            'environment': metadata.environment,
            'created_at': metadata.created_at,
            'metrics': metadata.metrics
        }
        
        # Remove if exists (update)
        updated_versions = [v for v in versions if isinstance(v, dict) and v.get('version') != version]
        updated_versions.append(version_entry)
        
        model_entry['versions'] = updated_versions
        self._save_index()

    
    def delete_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Model'i registry'den siler.
        
        Args:
            model_name: Model adı
            version: Spesifik versiyon (None ise tüm versiyonlar)
            
        Returns:
            bool: Silme başarılı mı
            
        Raises:
            ValueError: Model bulunamadığında
        """
        logger.info(f"Deleting model: {model_name}, version: {version or 'all'}")
        
        models_dict = self.index.setdefault('models', {})
        if model_name not in models_dict:
            # Check if directory exists on disk as fallback
            model_disk_dir = self.models_dir / model_name
            if not model_disk_dir.exists():
                raise ValueError(f"Model bulunamadı: {model_name}")
            models_dict[model_name] = {'versions': []}
        
        model_entry = models_dict[model_name]
        existing_versions = [
            v['version'] for v in model_entry.get('versions', [])
            if isinstance(v, dict) and 'version' in v
        ]
        
        # Check disk for versions as well
        model_disk_dir = self.models_dir / model_name
        if model_disk_dir.exists():
            for p in model_disk_dir.iterdir():
                if p.is_dir() and p.name not in existing_versions:
                    existing_versions.append(p.name)
        
        deleted_count = 0
        if version is None:
            versions_to_delete = list(existing_versions)
        else:
            if version not in existing_versions:
                raise ValueError(f"Version bulunamadı: {model_name}@{version}")
            versions_to_delete = [version]
        
        for ver in versions_to_delete:
            # 1. Model dosyalarını sil (self.models_dir / model_name / ver)
            ver_dir = self.models_dir / model_name / ver
            if ver_dir.exists():
                shutil.rmtree(ver_dir, ignore_errors=True)
                logger.info(f"  Deleted model version directory: {ver_dir}")
            
            # 2. Metadata dosyasını sil (self.metadata_dir / f"{model_name}_{ver}.json")
            metadata_file = self.metadata_dir / f"{model_name}_{ver}.json"
            if metadata_file.exists():
                metadata_file.unlink()
                logger.info(f"  Deleted metadata: {metadata_file}")
            
            # 3. Index'teki versiyon listesini güncelle
            model_entry['versions'] = [
                v for v in model_entry.get('versions', [])
                if isinstance(v, dict) and v.get('version') != ver
            ]
            deleted_count += 1
        
        # Model'in hiç versiyonu kalmadıysa model'i tamamen kaldır
        if not model_entry.get('versions'):
            if model_name in models_dict:
                del models_dict[model_name]
            if model_disk_dir.exists():
                shutil.rmtree(model_disk_dir, ignore_errors=True)
                logger.info(f"  Removed model directory: {model_disk_dir}")
        
        # Index'i kaydet
        self._save_index()
        logger.info(f"✅ Deleted {deleted_count} version(s) of {model_name}")
        return True

    def _calculate_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate file hash (default SHA-256, supports MD5)."""
        hasher = hashlib.sha256() if algorithm == "sha256" else hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


def main() -> None:
    """
    Demo ve test fonksiyonu.
    
    Model registry özelliklerini gösterir:
    - Model registration
    - Listing & filtering
    - Model loading
    - Environment promotion
    """
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 80)
    print("Model Registry Demo")
    print("=" * 80)
    
    # Initialize registry
    print("\n1. Initializing registry...")
    registry = ModelRegistry("demo_registry")
    
    # Create dummy checkpoint
    print("\n2. Creating dummy model...")
    import torch.nn as nn
    
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    checkpoint_path = Path("demo_model.pt")
    torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
    
    # Register models
    print("\n3. Registering models...")
    
    metadata1 = registry.register_model(
        model_name="demo-gpt",
        version="1.0.0",
        checkpoint_path=str(checkpoint_path),
        description="Demo GPT model v1",
        parameters=sum(p.numel() for p in model.parameters()),
        metrics={'val_loss': 0.5, 'perplexity': 1.65},
        tags=['demo', 'v1'],
        environment='development'
    )
    
    metadata2 = registry.register_model(
        model_name="demo-gpt",
        version="1.1.0",
        checkpoint_path=str(checkpoint_path),
        description="Demo GPT model v1.1",
        parameters=sum(p.numel() for p in model.parameters()),
        metrics={'val_loss': 0.3, 'perplexity': 1.35},
        tags=['demo', 'v1.1'],
        environment='staging'
    )
    
    # List models
    print("\n4. Listing models...")
    all_models = registry.list_models()
    print(f"  Total models: {len(all_models)}")
    
    for m in all_models:
        print(f"    {m.model_name} v{m.version} ({m.environment}): {m.metrics}")
    
    # Load model
    print("\n5. Loading model...")
    info = registry.load_model("demo-gpt", "1.1.0")
    print(f"  Loaded: {info['metadata']['model_name']} v{info['metadata']['version']}")
    print(f"  Checkpoint: {info['checkpoint_path']}")
    
    # Compare models
    print("\n6. Comparing models...")
    comparison = registry.compare_models([
        ('demo-gpt', '1.0.0'),
        ('demo-gpt', '1.1.0')
    ])
    
    for model in comparison['models']:
        print(f"  {model['model']}: loss={model['metrics'].get('val_loss')}")
    
    # Promote
    print("\n7. Promoting model...")
    registry.promote_model('demo-gpt', '1.1.0', 'production')
    
    prod_models = registry.list_models(environment='production')
    print(f"  Production models: {len(prod_models)}")
    
    # Cleanup
    print("\n8. Cleanup...")
    checkpoint_path.unlink()
    shutil.rmtree("demo_registry")
    print("  ✓ Demo files removed")
    
    print("\n" + "=" * 80)
    print("✓ Demo tamamlandı!")
    print("=" * 80)


if __name__ == "__main__":
    main()
