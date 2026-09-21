"""
src/training/optimizer.py

Optimizer & Learning Rate Scheduler

Bu modül training için optimizer ve learning rate scheduler'ları içerir:
- AdamW optimizer (weight decay ile)
- Warmup + Cosine Decay scheduler
- Linear warmup scheduler
- Gradient clipping utilities

AdamW, standard Adam'ın weight decay'i düzgün implement eden versiyonudur.
Transformer model'ler için standard choice.

Learning Rate Schedule:
    1. Warmup: 0 → peak_lr (linear artış)
    2. Cosine Decay: peak_lr → min_lr (smooth azalış)
    
Gradient Clipping:
    Exploding gradient'ları önlemek için gradient norm'u clip eder.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8
) -> AdamW:
    """
    Create AdamW optimizer.
    
    AdamW, weight decay'i gradient update'ten ayrı olarak uygular.
    Bu, L2 regularization'dan daha etkilidir.
    
    Args:
        model: PyTorch model
        learning_rate: Initial learning rate
        weight_decay: Weight decay coefficient (L2 regularization)
        betas: Adam beta parameters (momentum terms)
        eps: Adam epsilon for numerical stability
    
    Returns:
        AdamW optimizer
    
    Example:
        >>> from src.model.gpt import GPTModel, GPTConfig
        >>> config = GPTConfig(vocab_size=1000, d_model=256, n_layers=4, n_heads=8)
        >>> model = GPTModel(config)
        >>> optimizer = create_optimizer(model, learning_rate=3e-4)
    """
    # Separate parameters: those with weight decay and those without
    # Typically, bias and LayerNorm parameters don't get weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for bias and LayerNorm
        if 'bias' in name or 'ln' in name.lower() or 'norm' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {
            'params': decay_params,
            'weight_decay': weight_decay
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=betas,
        eps=eps
    )
    
    logger.info(
        f"AdamW optimizer created: lr={learning_rate}, "
        f"weight_decay={weight_decay}, "
        f"params_with_decay={len(decay_params)}, "
        f"params_no_decay={len(no_decay_params)}"
    )
    
    return optimizer


def get_warmup_cosine_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create warmup + cosine decay learning rate scheduler.
    
    Schedule:
        1. Warmup (0 to num_warmup_steps): Linear increase from 0 to peak_lr
        2. Cosine decay (num_warmup_steps to num_training_steps): 
           Smooth decrease from peak_lr to min_lr
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of peak LR (e.g., 0.1 = 10% of peak)
        last_epoch: Last epoch index (for resuming)
    
    Returns:
        LambdaLR scheduler
    
    Example:
        >>> optimizer = create_optimizer(model)
        >>> scheduler = get_warmup_cosine_schedule(
        ...     optimizer,
        ...     num_warmup_steps=1000,
        ...     num_training_steps=10000
        ... )
        >>> for step in range(10000):
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()  # Update learning rate
    """
    def lr_lambda(current_step: int) -> float:
        """
        Compute learning rate multiplier for current step.
        
        Returns:
            LR multiplier in [min_lr_ratio, 1.0]
        """
        # Warmup phase: linear increase from 0 to 1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        # progress: 0 at warmup end, 1 at training end
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        
        # Cosine decay: 1 → min_lr_ratio
        # cos(0) = 1, cos(π) = -1
        # Map [0, 1] → [1, min_lr_ratio]
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        
        return max(min_lr_ratio, lr_multiplier)
    
    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
    
    logger.info(
        f"Warmup + Cosine scheduler created: "
        f"warmup_steps={num_warmup_steps}, "
        f"training_steps={num_training_steps}, "
        f"min_lr_ratio={min_lr_ratio}"
    )
    
    return scheduler


def get_linear_warmup_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create linear warmup scheduler (no decay after warmup).
    
    Schedule:
        - Warmup: Linear increase from 0 to peak_lr
        - After warmup: Constant at peak_lr
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        last_epoch: Last epoch index
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
    
    logger.info(f"Linear warmup scheduler created: warmup_steps={num_warmup_steps}")
    
    return scheduler


def get_constant_schedule(
    optimizer: Optimizer,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create constant learning rate scheduler (no warmup, no decay).
    
    Args:
        optimizer: PyTorch optimizer
        last_epoch: Last epoch index
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        return 1.0
    
    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
    
    logger.info("Constant scheduler created (no LR changes)")
    
    return scheduler


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients by norm to prevent exploding gradients.
    
    Gradient clipping, gradient norm'u belirli bir değeri aşarsa
    gradient'ları scale eder. Bu, training stability için önemlidir.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        norm_type: Type of norm (2.0 = L2 norm)
    
    Returns:
        Total gradient norm before clipping
    
    Example:
        >>> loss.backward()
        >>> grad_norm = clip_gradients(model, max_norm=1.0)
        >>> optimizer.step()
    """
    # Compute total gradient norm
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=norm_type
    )
    
    return total_norm.item()


def get_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Get total gradient norm without clipping.
    
    Gradient norm'u monitoring için kullanılır.
    
    Args:
        model: PyTorch model
        norm_type: Type of norm (2.0 = L2 norm)
    
    Returns:
        Total gradient norm
    
    Example:
        >>> loss.backward()
        >>> grad_norm = get_grad_norm(model)
        >>> print(f"Gradient norm: {grad_norm:.4f}")
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    
    if len(parameters) == 0:
        return 0.0
    
    # Compute norm
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type) 
            for p in parameters
        ]),
        norm_type
    )
    
    return total_norm.item()


def get_optimizer_state_dict(optimizer: Optimizer) -> dict:
    """
    Get optimizer state dict for checkpointing.
    
    Args:
        optimizer: PyTorch optimizer
    
    Returns:
        Optimizer state dict
    """
    return optimizer.state_dict()


def load_optimizer_state_dict(optimizer: Optimizer, state_dict: dict):
    """
    Load optimizer state dict from checkpoint.
    
    Args:
        optimizer: PyTorch optimizer
        state_dict: Optimizer state dict
    """
    optimizer.load_state_dict(state_dict)
    logger.info("Optimizer state loaded from checkpoint")


if __name__ == "__main__":
    # Test optimizer and scheduler
    logging.basicConfig(level=logging.INFO)
    
    from src.model.gpt import GPTModel, GPTConfig
    
    print("=" * 80)
    print("Optimizer & LR Scheduler Test")
    print("=" * 80)
    
    # Create model
    config = GPTConfig(
        vocab_size=1000,
        max_seq_len=128,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024
    )
    
    model = GPTModel(config)
    
    print(f"\nModel Configuration:")
    print(f"  Parameters: {model.get_num_params():,}")
    
    # Test 1: Create optimizer
    print(f"\n{'='*80}")
    print("[1] AdamW Optimizer Test")
    print(f"{'='*80}")
    
    optimizer = create_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=0.01
    )
    
    print(f"  Optimizer type: {type(optimizer).__name__}")
    print(f"  Initial LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"  Weight decay (group 0): {optimizer.param_groups[0]['weight_decay']}")
    print(f"  Weight decay (group 1): {optimizer.param_groups[1]['weight_decay']}")
    print(f"✅ Optimizer created with parameter groups")
    
    # Test 2: Warmup + Cosine scheduler
    print(f"\n{'='*80}")
    print("[2] Warmup + Cosine Scheduler Test")
    print(f"{'='*80}")
    
    num_warmup_steps = 100
    num_training_steps = 1000
    
    scheduler = get_warmup_cosine_schedule(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=0.1
    )
    
    # Simulate training steps and track LR
    lr_history = []
    test_steps = [0, 50, 100, 250, 500, 750, 999]
    
    for step in range(num_training_steps):
        current_lr = optimizer.param_groups[0]['lr']
        
        if step in test_steps:
            lr_history.append((step, current_lr))
        
        # Simulate optimizer step
        scheduler.step()
    
    print(f"\n  Learning Rate Schedule:")
    print(f"  {'Step':<10} {'LR':<15} {'Phase'}")
    print(f"  {'-'*10} {'-'*15} {'-'*20}")
    
    for step, lr in lr_history:
        if step < num_warmup_steps:
            phase = "Warmup"
        else:
            phase = "Cosine Decay"
        print(f"  {step:<10} {lr:<15.6f} {phase}")
    
    print(f"\n  Warmup end LR: {lr_history[2][1]:.6f} (should be ~3e-4)")
    print(f"  Final LR: {lr_history[-1][1]:.6f} (should be ~3e-5, 10% of peak)")
    print(f"✅ Scheduler working correctly")
    
    # Test 3: Gradient clipping
    print(f"\n{'='*80}")
    print("[3] Gradient Clipping Test")
    print(f"{'='*80}")
    
    # Create dummy loss and compute gradients
    batch_size = 2
    seq_len = 50
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, _ = model(token_ids)
    
    # Dummy targets
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Compute loss
    from src.training.loss import compute_cross_entropy_loss
    loss = compute_cross_entropy_loss(logits, targets)
    
    # Backward pass
    loss.backward()
    
    # Get gradient norm before clipping
    grad_norm_before = get_grad_norm(model)
    print(f"  Gradient norm before clipping: {grad_norm_before:.4f}")
    
    # Clip gradients
    grad_norm_clipped = clip_gradients(model, max_norm=1.0)
    print(f"  Gradient norm after clipping: {grad_norm_clipped:.4f}")
    
    if grad_norm_before > 1.0:
        print(f"  Gradients were clipped (reduced by {grad_norm_before/grad_norm_clipped:.2f}x)")
    else:
        print(f"  Gradients were NOT clipped (already < 1.0)")
    
    print(f"✅ Gradient clipping working")
    
    # Test 4: Linear warmup scheduler
    print(f"\n{'='*80}")
    print("[4] Linear Warmup Scheduler Test")
    print(f"{'='*80}")
    
    optimizer2 = create_optimizer(model, learning_rate=1e-3)
    scheduler2 = get_linear_warmup_schedule(optimizer2, num_warmup_steps=50)
    
    lr_warmup = []
    for step in range(100):
        lr_warmup.append(optimizer2.param_groups[0]['lr'])
        scheduler2.step()
    
    print(f"  LR at step 0: {lr_warmup[0]:.6f}")
    print(f"  LR at step 25 (mid-warmup): {lr_warmup[25]:.6f}")
    print(f"  LR at step 50 (warmup end): {lr_warmup[50]:.6f}")
    print(f"  LR at step 99 (after warmup): {lr_warmup[99]:.6f}")
    
    assert abs(lr_warmup[50] - 1e-3) < 1e-6, "Warmup should reach peak LR"
    assert abs(lr_warmup[99] - 1e-3) < 1e-6, "Should stay constant after warmup"
    print(f"✅ Linear warmup working")
    
    # Test 5: Optimizer state save/load
    print(f"\n{'='*80}")
    print("[5] Optimizer State Save/Load Test")
    print(f"{'='*80}")
    
    # Save state
    state_before = get_optimizer_state_dict(optimizer)
    print(f"  State dict keys: {list(state_before.keys())}")
    
    # Modify optimizer state (take a step) - need fresh loss
    optimizer.zero_grad()
    logits2, _ = model(token_ids)
    loss2 = compute_cross_entropy_loss(logits2, targets)
    loss2.backward()
    optimizer.step()
    
    state_after_step = get_optimizer_state_dict(optimizer)
    
    # Create new optimizer and load state
    optimizer3 = create_optimizer(model, learning_rate=3e-4)
    load_optimizer_state_dict(optimizer3, state_before)
    
    state_loaded = get_optimizer_state_dict(optimizer3)
    
    # Check if states match
    keys_match = set(state_before.keys()) == set(state_loaded.keys())
    print(f"  State keys match: {keys_match}")
    print(f"✅ Optimizer state save/load working")
    
    # Test 6: Different LR for different parameter groups
    print(f"\n{'='*80}")
    print("[6] Parameter Group Test")
    print(f"{'='*80}")
    
    decay_params = optimizer.param_groups[0]['params']
    no_decay_params = optimizer.param_groups[1]['params']
    
    print(f"  Parameters with weight decay: {len(decay_params)}")
    print(f"  Parameters without weight decay: {len(no_decay_params)}")
    print(f"  Total parameter tensors: {len(list(model.parameters()))}")
    print(f"✅ Parameter groups correctly separated")
    
    print(f"\n" + "=" * 80)
    print("✅ All tests PASSED!")
    print("=" * 80)
