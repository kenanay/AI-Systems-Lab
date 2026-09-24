"""
LoRA (Low-Rank Adaptation) Implementation

Bu modül, büyük dil modellerini parameter-efficient şekilde fine-tune etmek için
LoRA (Low-Rank Adaptation) tekniğini implement eder.

LoRA Prensibi:
----------------
Original weight matrix W ∈ R^(d×k) yerine, iki küçük matris kullanır:
    ΔW = B × A
    where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)

Forward pass:
    h = W₀x + ΔWx = W₀x + BAx
    
Burada:
- W₀: Frozen pretrained weights (güncellenmiyor)
- B, A: Trainable LoRA adapters (sadece bunlar güncelleniyor)
- r: LoRA rank (tipik 4, 8, 16, 32, 64)

Avantajlar:
-----------
1. Memory efficient: Sadece ΔW parametreleri gradient hesabı yapıyor
2. Storage efficient: Her task için sadece B,A kayıt edilir (W₀ paylaşılır)
3. Inference: ΔW, W₀'a merge edilebilir → ekstra compute yok
4. Modular: LoRA'lar kolayca swap edilebilir

Örnek Parametre Sayısı:
-----------------------
Original Linear layer: d=4096, k=4096 → 16M params
LoRA with r=8: (4096×8) + (8×4096) = 65K params
Reduction: 250x daha az parametre!

Matematiksel Detay:
------------------
ΔW = B × A
B: [d, r] - down-projection (genelde zeros ile init)
A: [r, k] - up-projection (genelde Gaussian ile init)

Forward:
    result = Wx = W₀x + s·(B(Ax))
    where s = α/r (scaling factor)

Burada:
- α: LoRA alpha (tipik r ile aynı veya 2r)
- s: scaling to keep magnitude similar to pretrained weights

Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Set
import logging
import math

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    LoRA adapter layer.
    
    Bu layer, existing linear layer'a eklenerek low-rank adaptation sağlar.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (r)
        alpha: LoRA alpha (scaling factor)
        dropout: Dropout probability
    
    Math:
        output = W₀x + (α/r) × B(Ax)
        B shape: [out_features, rank]
        A shape: [rank, in_features]
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # LoRA scaling factor
        self.scaling = alpha / rank
        
        # LoRA matrices
        # A: [rank, in_features] - initialized with Gaussian
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        
        # B: [out_features, rank] - initialized with zeros
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        logger.debug(
            f"LoRA layer created: in={in_features}, out={out_features}, "
            f"rank={rank}, alpha={alpha}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA adapter.
        
        Args:
            x: Input tensor [batch, seq_len, in_features]
            
        Returns:
            LoRA output [batch, seq_len, out_features]
            
        Math:
            result = (α/r) × B(Ax)
        """
        # x shape: [batch, seq_len, in_features]
        # lora_A shape: [rank, in_features]
        # lora_B shape: [out_features, rank]
        
        # Apply dropout to input
        # x shape remains: [batch, seq_len, in_features]
        x = self.dropout(x)
        
        # A projection (down-projection):
        # x @ lora_A.T: [batch, seq_len, in_features] @ [in_features, rank]
        #            → [batch, seq_len, rank]
        result = x @ self.lora_A.T  # [batch, seq_len, rank]
        
        # B projection (up-projection):
        # result @ lora_B.T: [batch, seq_len, rank] @ [rank, out_features]
        #                  → [batch, seq_len, out_features]
        result = result @ self.lora_B.T  # [batch, seq_len, out_features]
        
        # Scale by α/r
        # result shape: [batch, seq_len, out_features]
        result = result * self.scaling
        
        return result
    
    def extra_repr(self) -> str:
        """Extra representation for print."""
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'rank={self.rank}, '
            f'alpha={self.alpha}, '
            f'scaling={self.scaling:.4f}'
        )


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adapter.
    
    Bu module, pretrained linear layer'ı LoRA ile wrap eder.
    
    Args:
        base_layer: Original frozen linear layer
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        merge_weights: Merge LoRA into base weights
    
    Forward:
        if merged:
            output = (W₀ + ΔW)x
        else:
            output = W₀x + ΔWx
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False
    ):
        super().__init__()
        
        # Store base layer (frozen)
        self.base_layer = base_layer
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Create LoRA adapter
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Merge state
        self.merged = False
        self.merge_weights = merge_weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, in_features]
            
        Returns:
            Output [batch, seq_len, out_features]
        """
        if self.merged:
            # Weights are already merged
            # base_layer weight contains W₀ + ΔW
            return self.base_layer(x)
        else:
            # Separate computation
            # base_output: W₀x shape: [batch, seq_len, out_features]
            base_output = self.base_layer(x)
            
            # lora_output: ΔWx shape: [batch, seq_len, out_features]
            lora_output = self.lora(x)
            
            # Combined output: [batch, seq_len, out_features]
            return base_output + lora_output
    
    def merge(self):
        """
        Merge LoRA weights into base layer.
        
        After merge: W = W₀ + (α/r)BA
        
        This is useful for inference to avoid extra computation.
        """
        if self.merged:
            logger.warning("LoRA weights already merged")
            return
        
        # Compute ΔW = (α/r) × B @ A
        # lora_B shape: [out_features, rank]
        # lora_A shape: [rank, in_features]
        # B @ A: [out_features, rank] @ [rank, in_features]
        #     → [out_features, in_features]
        delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling  # [out_features, in_features]
        
        # Add to base weight
        # base_layer.weight shape: [out_features, in_features]
        # delta_w shape: [out_features, in_features]
        # Result: element-wise addition → [out_features, in_features]
        self.base_layer.weight.data += delta_w
        
        self.merged = True
        logger.info("LoRA weights merged into base layer")
    
    def unmerge(self):
        """
        Unmerge LoRA weights from base layer.
        
        Restore: W = W₀ (remove ΔW)
        """
        if not self.merged:
            logger.warning("LoRA weights not merged, nothing to unmerge")
            return
        
        # Compute ΔW = (α/r) × B @ A
        # lora_B shape: [out_features, rank]
        # lora_A shape: [rank, in_features]
        # B @ A: [out_features, rank] @ [rank, in_features]
        #     → [out_features, in_features]
        delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling  # [out_features, in_features]
        
        # Subtract from base weight
        # base_layer.weight shape: [out_features, in_features]
        # delta_w shape: [out_features, in_features]
        # Result: element-wise subtraction → [out_features, in_features]
        self.base_layer.weight.data -= delta_w
        
        self.merged = False
        logger.info("LoRA weights unmerged from base layer")


class LoRAConfig:
    """
    LoRA configuration.
    
    Attributes:
        rank: LoRA rank (r)
        alpha: LoRA alpha for scaling
        dropout: Dropout probability
        target_modules: Which modules to apply LoRA (e.g., ['q_proj', 'v_proj'])
        merge_weights: Whether to merge weights for inference
    """
    
    def __init__(
        self,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        merge_weights: bool = False,
        r: Optional[int] = None,
        lora_alpha: Optional[float] = None
    ):
        self.rank = rank if rank is not None else (r if r is not None else 8)
        self.alpha = alpha if alpha is not None else (lora_alpha if lora_alpha is not None else 16.0)
        self.dropout = dropout
        self.target_modules = target_modules or ['q_proj', 'v_proj']
        self.merge_weights = merge_weights
    
    def __repr__(self) -> str:
        return (
            f"LoRAConfig(rank={self.rank}, alpha={self.alpha}, "
            f"dropout={self.dropout}, target_modules={self.target_modules})"
        )


def add_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
    verbose: bool = True
) -> nn.Module:
    """
    Add LoRA adapters to model.
    
    Bu fonksiyon, model içindeki target linear layer'ları bulup
    LoRA adapter ekler.
    
    Args:
        model: Base model
        config: LoRA configuration
        verbose: Log added LoRA layers
        
    Returns:
        Model with LoRA adapters
        
    Example:
        >>> from src.model.gpt import GPTModel, GPTConfig
        >>> model = GPTModel(GPTConfig(vocab_size=1000))
        >>> lora_config = LoRAConfig(rank=8, target_modules=['q_proj', 'v_proj'])
        >>> model = add_lora_to_model(model, lora_config)
    """
    lora_count = 0
    
    # Iterate through all modules
    for name, module in model.named_modules():
        # Check if module name matches target
        should_add_lora = any(
            target in name for target in config.target_modules
        )
        
        if should_add_lora and isinstance(module, nn.Linear):
            # Get parent module and attribute name
            *parent_path, attr_name = name.split('.')
            
            if parent_path:
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)
            else:
                parent = model
            
            # Replace with LoRA version
            lora_layer = LinearWithLoRA(
                base_layer=module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                merge_weights=config.merge_weights
            )
            
            setattr(parent, attr_name, lora_layer)
            lora_count += 1
            
            if verbose:
                logger.info(f"Added LoRA to {name}")
    
    logger.info(f"Total LoRA layers added: {lora_count}")
    
    # Print parameter statistics
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Trainable params: {trainable_params:,}")
    logger.info(f"Total params: {total_params:,}")
    logger.info(
        f"Trainable %: {100 * trainable_params / total_params:.2f}%"
    )
    
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from model.
    
    Args:
        model: Model with LoRA adapters
        
    Returns:
        List of LoRA parameters (only trainable ones)
    """
    lora_params = []
    
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            lora_params.extend([
                module.lora.lora_A,
                module.lora.lora_B
            ])
    
    return lora_params


def merge_lora_weights(model: nn.Module):
    """
    Merge all LoRA weights in model.
    
    Useful for inference to avoid extra computation.
    
    Args:
        model: Model with LoRA adapters
    """
    merge_count = 0
    
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            module.merge()
            merge_count += 1
    
    logger.info(f"Merged {merge_count} LoRA layers")


def unmerge_lora_weights(model: nn.Module):
    """
    Unmerge all LoRA weights in model.
    
    Restore original weights for continued training.
    
    Args:
        model: Model with LoRA adapters
    """
    unmerge_count = 0
    
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            module.unmerge()
            unmerge_count += 1
    
    logger.info(f"Unmerged {unmerge_count} LoRA layers")


def save_lora_weights(model: nn.Module, path: str):
    """
    Save only LoRA weights (not base model).
    
    This is very storage efficient - only saves adapter weights.
    
    Args:
        model: Model with LoRA adapters
        path: Save path
    """
    lora_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            lora_state_dict[f"{name}.lora_A"] = module.lora.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora.lora_B.data
    
    torch.save(lora_state_dict, path)
    
    # Calculate size
    total_params = sum(p.numel() for p in lora_state_dict.values())
    logger.info(f"Saved LoRA weights: {path}")
    logger.info(f"LoRA parameters: {total_params:,}")


def load_lora_weights(model: nn.Module, path: str):
    """
    Load LoRA weights into model.
    
    Args:
        model: Model with LoRA adapters
        path: Load path
    """
    lora_state_dict = torch.load(path, weights_only=True)
    
    loaded_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            
            if a_key in lora_state_dict and b_key in lora_state_dict:
                module.lora.lora_A.data = lora_state_dict[a_key]
                module.lora.lora_B.data = lora_state_dict[b_key]
                loaded_count += 1
    
    logger.info(f"Loaded LoRA weights from {path}")
    logger.info(f"LoRA layers loaded: {loaded_count}")


def print_trainable_parameters(model: nn.Module):
    """
    Print trainable parameter statistics.
    
    Useful to verify LoRA is working correctly.
    
    Args:
        model: Model to analyze
    """
    trainable_params = 0
    all_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            logger.debug(f"Trainable: {name} | {param.numel():,} params")
    
    print(f"\n{'='*60}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"All params: {all_params:,}")
    print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")
    print(f"{'='*60}\n")


def main():
    """Demo LoRA usage."""
    print("LoRA module loaded successfully!")
    print(f"Available functions:")
    print(f"  - add_lora_to_model()")
    print(f"  - merge_lora_weights()")
    print(f"  - save_lora_weights()")
    print(f"  - load_lora_weights()")


if __name__ == "__main__":
    main()
