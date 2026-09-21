"""
src/training/config.py

Training Configuration System

Bu modül model training için configuration yönetimini sağlar:
- Model architecture configs (Transformer, GPT-style)
- Training hyperparameters
- YAML-based configuration
- Validation
- Config versioning

Her config dosyası şu format'ta:
```yaml
version: "1.0.0"
model:
  architecture: "transformer_decoder"
  vocab_size: 8000
  context_length: 512
  d_model: 256
  n_layers: 6
  n_heads: 8
  d_ff: 1024
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.0003
  max_steps: 100000
  warmup_steps: 1000
  gradient_clip: 1.0
  weight_decay: 0.01
```
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelArchitectureConfig:
    """
    Model architecture configuration.
    
    Transformer decoder (GPT-style) parametreleri.
    
    Attributes:
        architecture: Model tipi (örn: "transformer_decoder")
        vocab_size: Vocabulary boyutu
        context_length: Maximum sequence length (context window)
        d_model: Model dimension (embedding size)
        n_layers: Transformer layer sayısı
        n_heads: Attention head sayısı
        d_ff: Feed-forward layer dimension
        dropout: Dropout rate
        activation: Activation function ("gelu", "relu", "swish")
        use_bias: Bias kullan mı
        tie_embeddings: Input/output embedding'leri tie et mi
    """
    
    architecture: str = "transformer_decoder"
    vocab_size: int = 8000
    context_length: int = 512
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    use_bias: bool = True
    tie_embeddings: bool = True
    
    def validate(self):
        """
        Configuration'ı validate et.
        
        Raises:
            ValueError: Geçersiz parametre varsa
        """
        # d_model n_heads'e bölünebilir olmalı
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) n_heads ({self.n_heads})'e bölünebilir olmalı"
            )
        
        # Positive values
        if self.vocab_size <= 0:
            raise ValueError("vocab_size pozitif olmalı")
        
        if self.context_length <= 0:
            raise ValueError("context_length pozitif olmalı")
        
        if self.n_layers <= 0:
            raise ValueError("n_layers pozitif olmalı")
        
        if self.n_heads <= 0:
            raise ValueError("n_heads pozitif olmalı")
        
        # Dropout range
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout [0, 1) aralığında olmalı")
        
        # Activation
        valid_activations = ["gelu", "relu", "swish", "silu"]
        if self.activation not in valid_activations:
            raise ValueError(
                f"activation {valid_activations} içinden biri olmalı"
            )
        
        logger.info(f"Model config validated: {self.architecture}")
    
    @property
    def d_head(self) -> int:
        """Attention head dimension"""
        return self.d_model // self.n_heads
    
    @property
    def num_parameters(self) -> int:
        """
        Approximate model parameter count.
        
        Returns:
            int: Yaklaşık parametre sayısı
        """
        # Token + Position embeddings
        embedding_params = self.vocab_size * self.d_model + self.context_length * self.d_model
        
        # Transformer layers
        # - Attention: Q, K, V projections + output projection
        # - Feed-forward: 2 linear layers
        # - Layer norms: 2 per layer
        layer_params = (
            # Attention
            4 * (self.d_model * self.d_model) +  # Q, K, V, O projections
            # Feed-forward
            (self.d_model * self.d_ff) + (self.d_ff * self.d_model) +
            # Layer norms (approximate)
            2 * self.d_model * 2  # weight + bias per norm
        )
        
        transformer_params = self.n_layers * layer_params
        
        # Output layer (LM head)
        if self.tie_embeddings:
            output_params = 0  # Tied with input embedding
        else:
            output_params = self.vocab_size * self.d_model
        
        total = embedding_params + transformer_params + output_params
        return total


@dataclass
class TrainingConfig:
    """
    Training hyperparameters configuration.
    
    Attributes:
        batch_size: Training batch size
        learning_rate: Initial learning rate
        max_steps: Maximum training steps
        warmup_steps: Learning rate warmup steps
        gradient_clip: Gradient clipping threshold
        weight_decay: Weight decay (L2 regularization)
        eval_interval: Evaluation interval (steps)
        save_interval: Checkpoint save interval (steps)
        max_grad_norm: Maximum gradient norm (clipping)
        beta1: Adam optimizer beta1
        beta2: Adam optimizer beta2
        epsilon: Adam optimizer epsilon
        scheduler: Learning rate scheduler ("cosine", "linear", "constant")
        min_lr: Minimum learning rate (for scheduler)
    """
    
    batch_size: int = 32
    learning_rate: float = 0.0003
    max_steps: int = 100000
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    eval_interval: int = 1000
    save_interval: int = 5000
    max_grad_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    scheduler: str = "cosine"
    min_lr: float = 1e-5
    
    def validate(self):
        """
        Configuration'ı validate et.
        
        Raises:
            ValueError: Geçersiz parametre varsa
        """
        # Positive values
        if self.batch_size <= 0:
            raise ValueError("batch_size pozitif olmalı")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate pozitif olmalı")
        
        if self.max_steps <= 0:
            raise ValueError("max_steps pozitif olmalı")
        
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps negatif olamaz")
        
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps max_steps'ten büyük olamaz")
        
        # Scheduler
        valid_schedulers = ["cosine", "linear", "constant"]
        if self.scheduler not in valid_schedulers:
            raise ValueError(
                f"scheduler {valid_schedulers} içinden biri olmalı"
            )
        
        # Learning rate range
        if self.min_lr >= self.learning_rate:
            raise ValueError("min_lr learning_rate'ten küçük olmalı")
        
        logger.info("Training config validated")


@dataclass
class DataConfig:
    """
    Dataset configuration for training.
    
    Attributes:
        dataset_path: Parquet dataset path
        tokenizer_id: Tokenizer ID
        sequence_length: Training sequence length
        stride: Stride for sequence creation (overlap)
        shuffle: Shuffle dataset
        num_workers: DataLoader worker sayısı
        pin_memory: Pin memory for GPU transfer
    """
    
    dataset_path: str = ""
    tokenizer_id: str = ""
    sequence_length: int = 512
    stride: Optional[int] = None  # Default: same as sequence_length (no overlap)
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    def validate(self):
        """
        Configuration'ı validate et.
        
        Raises:
            ValueError: Geçersiz parametre varsa
        """
        if not self.dataset_path:
            raise ValueError("dataset_path gerekli")
        
        if not self.tokenizer_id:
            raise ValueError("tokenizer_id gerekli")
        
        if self.sequence_length <= 0:
            raise ValueError("sequence_length pozitif olmalı")
        
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride pozitif olmalı veya None")
        
        if self.num_workers < 0:
            raise ValueError("num_workers negatif olamaz")
        
        logger.info("Data config validated")


@dataclass
class TrainingRunConfig:
    """
    Complete training run configuration.
    
    Bir training run'ı tanımlayan tüm parametreler.
    
    Attributes:
        version: Config version (semantic versioning)
        name: Config name/description
        model: Model architecture config
        training: Training hyperparameters
        data: Dataset config
        metadata: Additional metadata
    """
    
    version: str = "1.0.0"
    name: str = ""
    model: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self, skip_data: bool = False):
        """
        Tüm configuration'ları validate et.
        
        Args:
            skip_data: Data config validation'ı atla (template creation için)
        
        Raises:
            ValueError: Herhangi bir config geçersizse
        """
        if not self.name:
            raise ValueError("Config name gerekli")
        
        # Sub-configs validate
        self.model.validate()
        self.training.validate()
        
        if not skip_data:
            self.data.validate()
            
            # Cross-config validation
            # Context length consistency
            if self.data.sequence_length > self.model.context_length:
                raise ValueError(
                    f"data.sequence_length ({self.data.sequence_length}) "
                    f"model.context_length ({self.model.context_length})'ten büyük olamaz"
                )
        
        logger.info(f"Training run config validated: {self.name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Configuration'ı dict'e çevir.
        
        Returns:
            Dict: Serializable dict
        """
        return {
            "version": self.version,
            "name": self.name,
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "metadata": self.metadata
        }
    
    def to_yaml(self, path: Path):
        """
        Configuration'ı YAML dosyasına kaydet.
        
        Args:
            path: Output YAML path
        """
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Config saved to YAML: {path}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingRunConfig":
        """
        Dict'ten configuration oluştur.
        
        Args:
            data: Configuration dict
            
        Returns:
            TrainingRunConfig: Configuration instance
        """
        return cls(
            version=data.get("version", "1.0.0"),
            name=data.get("name", ""),
            model=ModelArchitectureConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(**data.get("data", {})),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_yaml(cls, path: Path, validate: bool = True, skip_data: bool = False) -> "TrainingRunConfig":
        """
        YAML dosyasından configuration yükle.
        
        Args:
            path: YAML file path
            validate: Validation yap mı
            skip_data: Data validation'ı atla (template configs için)
            
        Returns:
            TrainingRunConfig: Configuration instance
            
        Raises:
            FileNotFoundError: Dosya bulunamazsa
            ValueError: YAML parse edilemezse veya geçersizse
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = cls.from_dict(data)
        
        if validate:
            config.validate(skip_data=skip_data)
        
        logger.info(f"Config loaded from YAML: {path}")
        return config


# Predefined configs
def create_tiny_config(
    vocab_size: int = 8000,
    context_length: int = 256
) -> TrainingRunConfig:
    """
    Tiny model config (test/debug için).
    
    ~5M parameters
    """
    return TrainingRunConfig(
        name="tiny_model",
        version="1.0.0",
        model=ModelArchitectureConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=16,
            learning_rate=0.001,
            max_steps=10000,
            warmup_steps=500
        )
    )


def create_small_config(
    vocab_size: int = 8000,
    context_length: int = 512
) -> TrainingRunConfig:
    """
    Small model config (educational/prototype).
    
    ~25M parameters
    """
    return TrainingRunConfig(
        name="small_model",
        version="1.0.0",
        model=ModelArchitectureConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=32,
            learning_rate=0.0003,
            max_steps=100000,
            warmup_steps=1000
        )
    )


def create_medium_config(
    vocab_size: int = 8000,
    context_length: int = 1024
) -> TrainingRunConfig:
    """
    Medium model config (production-ready small model).
    
    ~125M parameters
    """
    return TrainingRunConfig(
        name="medium_model",
        version="1.0.0",
        model=ModelArchitectureConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=512,
            n_layers=12,
            n_heads=8,
            d_ff=2048,
            dropout=0.1
        ),
        training=TrainingConfig(
            batch_size=64,
            learning_rate=0.0002,
            max_steps=500000,
            warmup_steps=2000
        )
    )


if __name__ == "__main__":
    # Test configs
    logging.basicConfig(level=logging.INFO)
    
    # Create and validate configs (skip data validation for templates)
    tiny = create_tiny_config()
    tiny.validate(skip_data=True)
    print(f"Tiny model: ~{tiny.model.num_parameters / 1e6:.1f}M parameters")
    
    small = create_small_config()
    small.validate(skip_data=True)
    print(f"Small model: ~{small.model.num_parameters / 1e6:.1f}M parameters")
    
    medium = create_medium_config()
    medium.validate(skip_data=True)
    print(f"Medium model: ~{medium.model.num_parameters / 1e6:.1f}M parameters")
    
    # Save to YAML
    output_dir = Path("configs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tiny.to_yaml(output_dir / "tiny_model.yaml")
    small.to_yaml(output_dir / "small_model.yaml")
    medium.to_yaml(output_dir / "medium_model.yaml")
    
    print("\nConfigs saved to configs/models/")
    
    # Test load
    loaded = TrainingRunConfig.from_yaml(output_dir / "small_model.yaml", skip_data=True)
    print(f"\nLoaded config: {loaded.name}")
    print(f"Model: {loaded.model.architecture}")
    print(f"d_model: {loaded.model.d_model}")
    print(f"n_layers: {loaded.model.n_layers}")
