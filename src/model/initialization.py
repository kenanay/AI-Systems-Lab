"""
src/model/initialization.py

Model Weight Initialization Strategies

Bu modül transformer model'ler için weight initialization stratejilerini içerir.
Doğru initialization training stability ve convergence için kritiktir.

Desteklenen stratejiler:
- GPT-style: Xavier for Linear, Normal(0, 0.02) for Embeddings
- BERT-style: Normal(0, 0.02) for all weights
- Megatron-style: Scaled initialization for deep networks
- Custom: Configurable initialization

Kaynaklar:
    - Delving Deep into Rectifiers (He et al., 2015)
    - Understanding the difficulty of training deep feedforward neural networks (Glorot & Bengio, 2010)
    - Megatron-LM: Training Multi-Billion Parameter Language Models (Shoeybi et al., 2019)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Callable, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InitStrategy(Enum):
    """Weight initialization strategy enum."""
    GPT = "gpt"           # GPT-style: Xavier + Normal
    BERT = "bert"         # BERT-style: Normal for all
    MEGATRON = "megatron" # Megatron: Scaled init for depth
    XAVIER = "xavier"     # Xavier/Glorot uniform
    KAIMING = "kaiming"   # Kaiming/He initialization
    NORMAL = "normal"     # Normal distribution
    CUSTOM = "custom"     # Custom initialization


class WeightInitializer:
    """
    Weight Initialization Manager
    
    Farklı initialization stratejilerini uygular ve yönetir.
    
    Args:
        strategy: Initialization strategy (InitStrategy enum)
        std: Standard deviation for normal init
        scale: Scaling factor for depth-dependent init
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        strategy: InitStrategy = InitStrategy.GPT,
        std: float = 0.02,
        scale: float = 1.0,
        seed: Optional[int] = None
    ):
        self.strategy = strategy
        self.std = std
        self.scale = scale
        
        if seed is not None:
            torch.manual_seed(seed)
            logger.info(f"Random seed set to {seed}")
        
        logger.info(f"WeightInitializer created with strategy: {strategy.value}")
    
    def initialize_model(
        self,
        model: nn.Module,
        n_layers: Optional[int] = None
    ):
        """
        Initialize all weights in model.
        
        Args:
            model: PyTorch model to initialize
            n_layers: Number of layers (for depth-dependent scaling)
        """
        if self.strategy == InitStrategy.GPT:
            self._gpt_init(model)
        elif self.strategy == InitStrategy.BERT:
            self._bert_init(model)
        elif self.strategy == InitStrategy.MEGATRON:
            assert n_layers is not None, "n_layers required for Megatron init"
            self._megatron_init(model, n_layers)
        elif self.strategy == InitStrategy.XAVIER:
            self._xavier_init(model)
        elif self.strategy == InitStrategy.KAIMING:
            self._kaiming_init(model)
        elif self.strategy == InitStrategy.NORMAL:
            self._normal_init(model)
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using GPT init")
            self._gpt_init(model)
        
        logger.info(f"Model initialized with {self.strategy.value} strategy")
    
    def _gpt_init(self, model: nn.Module):
        """
        GPT-style initialization.
        
        - Linear layers: Xavier uniform
        - Embeddings: Normal(0, 0.02)
        - LayerNorm: weight=1, bias=0
        """
        def _init_module(module):
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Normal initialization
                nn.init.normal_(module.weight, mean=0.0, std=self.std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        model.apply(_init_module)
        logger.debug("GPT-style init: Xavier for Linear, Normal(0, 0.02) for Embeddings")
    
    def _bert_init(self, model: nn.Module):
        """
        BERT-style initialization.
        
        - All weights: Normal(0, 0.02)
        - All biases: Zero
        - LayerNorm: weight=1, bias=0
        """
        def _init_module(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Normal initialization for all weights
                nn.init.normal_(module.weight, mean=0.0, std=self.std)
                if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        model.apply(_init_module)
        logger.debug("BERT-style init: Normal(0, 0.02) for all weights")
    
    def _megatron_init(self, model: nn.Module, n_layers: int):
        """
        Megatron-style initialization with depth scaling.
        
        Output layers scaled by 1/sqrt(2*n_layers) to account for residual connections.
        
        Args:
            model: Model to initialize
            n_layers: Number of transformer layers
        """
        # Scaling factor based on depth
        output_scale = 1.0 / math.sqrt(2.0 * n_layers)
        
        def _init_module(module):
            if isinstance(module, nn.Linear):
                # Normal init with std based on fan_in
                fan_in = module.weight.size(1)
                std = math.sqrt(1.0 / fan_in)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                
                # Scale output projection layers
                if hasattr(module, '_is_output_projection') and module._is_output_projection:
                    module.weight.data *= output_scale
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        model.apply(_init_module)
        logger.debug(f"Megatron-style init: depth scaling with factor {output_scale:.4f}")
    
    def _xavier_init(self, model: nn.Module):
        """
        Xavier/Glorot uniform initialization for all linear layers.
        """
        def _init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=self.scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Use Xavier for embeddings too
                nn.init.xavier_uniform_(module.weight, gain=self.scale)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        model.apply(_init_module)
        logger.debug(f"Xavier uniform init with gain={self.scale}")
    
    def _kaiming_init(self, model: nn.Module):
        """
        Kaiming/He initialization for all linear layers.
        
        Optimized for ReLU activations but works well with GELU too.
        """
        def _init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        model.apply(_init_module)
        logger.debug("Kaiming/He uniform init")
    
    def _normal_init(self, model: nn.Module):
        """
        Simple normal initialization for all weights.
        """
        def _init_module(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=self.std)
                if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        model.apply(_init_module)
        logger.debug(f"Normal init with std={self.std}")


def initialize_model(
    model: nn.Module,
    strategy: str = "gpt",
    std: float = 0.02,
    n_layers: Optional[int] = None,
    seed: Optional[int] = None
) -> nn.Module:
    """
    Convenience function to initialize model weights.
    
    Args:
        model: PyTorch model to initialize
        strategy: Initialization strategy name
        std: Standard deviation for normal init
        n_layers: Number of layers (for depth-dependent init)
        seed: Random seed for reproducibility
    
    Returns:
        Initialized model (same object, modified in-place)
    
    Example:
        >>> model = GPTModel(config)
        >>> model = initialize_model(model, strategy="gpt")
        >>> # Model weights are now initialized
    """
    # Convert string to enum
    strategy_map = {
        "gpt": InitStrategy.GPT,
        "bert": InitStrategy.BERT,
        "megatron": InitStrategy.MEGATRON,
        "xavier": InitStrategy.XAVIER,
        "kaiming": InitStrategy.KAIMING,
        "normal": InitStrategy.NORMAL,
    }
    
    strategy_enum = strategy_map.get(strategy.lower(), InitStrategy.GPT)
    
    # Create initializer and apply
    initializer = WeightInitializer(
        strategy=strategy_enum,
        std=std,
        seed=seed
    )
    initializer.initialize_model(model, n_layers=n_layers)
    
    return model


def get_init_stats(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Get statistics about initialized weights.
    
    Useful for verifying initialization worked correctly.
    
    Args:
        model: Initialized model
    
    Returns:
        Dictionary with statistics per layer type
        
    Example:
        >>> stats = get_init_stats(model)
        >>> print(stats['Linear']['mean'])
        0.0001
        >>> print(stats['Embedding']['std'])
        0.0199
    """
    stats = {}
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        if module_type not in stats:
            stats[module_type] = {
                'mean': [],
                'std': [],
                'min': [],
                'max': [],
                'count': 0
            }
        
        # Get weights
        weight = getattr(module, 'weight', None)
        if weight is not None and isinstance(weight, torch.Tensor):
            stats[module_type]['mean'].append(float(weight.mean().item()))
            stats[module_type]['std'].append(float(weight.std().item()))
            stats[module_type]['min'].append(float(weight.min().item()))
            stats[module_type]['max'].append(float(weight.max().item()))
            stats[module_type]['count'] += 1
    
    # Aggregate statistics
    aggregated = {}
    for module_type, values in stats.items():
        if values['count'] > 0:
            aggregated[module_type] = {
                'mean': sum(values['mean']) / len(values['mean']),
                'std': sum(values['std']) / len(values['std']),
                'min': min(values['min']),
                'max': max(values['max']),
                'count': values['count']
            }
    
    return aggregated


if __name__ == "__main__":
    # Test weight initialization
    logging.basicConfig(level=logging.INFO)
    
    from src.model.gpt import GPTModel, GPTConfig
    
    print("=" * 80)
    print("Weight Initialization Test")
    print("=" * 80)
    
    # Create test configuration
    config = GPTConfig(
        vocab_size=1000,
        max_seq_len=128,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        dropout=0.0  # No dropout for init test
    )
    
    print(f"\nModel Configuration:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    
    # Test different initialization strategies
    strategies = ["gpt", "bert", "xavier", "kaiming", "normal"]
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Testing {strategy.upper()} Initialization")
        print(f"{'='*80}")
        
        # Create model (without default init)
        model = GPTModel(config)
        
        # Apply initialization strategy
        initialize_model(
            model,
            strategy=strategy,
            std=0.02,
            n_layers=config.n_layers,
            seed=42  # For reproducibility
        )
        
        # Get initialization statistics
        stats = get_init_stats(model)
        
        print(f"\nWeight Statistics:")
        for module_type, values in stats.items():
            if module_type in ['Linear', 'Embedding']:
                print(f"  {module_type}:")
                print(f"    Mean: {values['mean']:.6f}")
                print(f"    Std:  {values['std']:.6f}")
                print(f"    Range: [{values['min']:.6f}, {values['max']:.6f}]")
                print(f"    Count: {values['count']} layers")
    
    # Test Megatron initialization
    print(f"\n{'='*80}")
    print(f"Testing MEGATRON Initialization (with depth scaling)")
    print(f"{'='*80}")
    
    model_megatron = GPTModel(config)
    
    # Mark output projections for scaling
    for name, module in model_megatron.named_modules():
        if 'w_o' in name or 'output_projection' in name:
            setattr(module, '_is_output_projection', True)
    
    initialize_model(
        model_megatron,
        strategy="megatron",
        std=0.02,
        n_layers=config.n_layers,
        seed=42
    )
    
    stats_megatron = get_init_stats(model_megatron)
    
    print(f"\nMegatron Weight Statistics:")
    for module_type, values in stats_megatron.items():
        if module_type == 'Linear':
            print(f"  {module_type}:")
            print(f"    Mean: {values['mean']:.6f}")
            print(f"    Std:  {values['std']:.6f}")
            print(f"    Depth scaling factor: {1.0 / math.sqrt(2.0 * config.n_layers):.4f}")
    
    # Test reproducibility
    print(f"\n{'='*80}")
    print(f"Testing Reproducibility (with seed)")
    print(f"{'='*80}")
    
    model1 = GPTModel(config)
    model2 = GPTModel(config)
    
    initialize_model(model1, strategy="gpt", seed=123)
    initialize_model(model2, strategy="gpt", seed=123)
    
    # Check if weights are identical
    weights_match = True
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.allclose(p1, p2):
            weights_match = False
            break
    
    if weights_match:
        print("✅ Reproducibility test PASSED (weights identical with same seed)")
    else:
        print("❌ Reproducibility test FAILED")
    
    print(f"\n" + "=" * 80)
    print("✅ All initialization tests PASSED!")
    print("=" * 80)
