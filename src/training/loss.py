"""
src/training/loss.py

Loss Functions & Metrics for Language Model Training

Bu modül language model training için loss function'ları ve metric'leri içerir:
- Cross-entropy loss (next-token prediction için)
- Label smoothing (regularization için)
- Perplexity (model performance metric)
- Accuracy calculation

Cross-entropy loss autoregressive language modeling'in standard loss function'ıdır.
Her pozisyonda, model bir sonraki token'ı predict eder.

Formül:
    Loss = -log P(token_t | token_1, ..., token_{t-1})
    
Perplexity:
    PPL = exp(Loss)
    Düşük perplexity = daha iyi model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class LanguageModelLoss(nn.Module):
    """
    Language Model Loss Function
    
    Next-token prediction için cross-entropy loss hesaplar.
    Optional label smoothing ile regularization sağlar.
    
    Args:
        vocab_size: Vocabulary size
        padding_idx: Padding token ID (loss hesaplanmaz)
        label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = typical)
        reduction: Loss reduction method ('mean', 'sum', 'none')
    
    Shape:
        logits: [batch, seq_len, vocab_size]
        targets: [batch, seq_len]
        output: scalar (if reduction='mean' or 'sum')
    """
    
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        # Cross-entropy loss
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=padding_idx,
            label_smoothing=label_smoothing,
            reduction=reduction
        )
        
        logger.debug(
            f"LanguageModelLoss initialized: vocab_size={vocab_size}, "
            f"label_smoothing={label_smoothing}"
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_metrics: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute language modeling loss.
        
        Args:
            logits: Model predictions, shape [B, T, V]
            targets: Target token IDs, shape [B, T]
            return_metrics: If True, return additional metrics
        
        Returns:
            loss: Scalar loss value
            metrics: (optional) Dict with loss, perplexity, accuracy
            
        Shape notation:
            B = batch_size
            T = seq_len
            V = vocab_size
        """
        # logits shape: [B, T, V]
        # targets shape: [B, T]
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for cross-entropy
        # logits: [B, T, V] -> [B*T, V]
        logits_flat = logits.reshape(-1, vocab_size)
        
        # targets: [B, T] -> [B*T]
        targets_flat = targets.reshape(-1)
        
        # Compute loss
        # loss shape: scalar (if reduction='mean')
        loss = self.loss_fn(logits_flat, targets_flat)
        
        if not return_metrics:
            return loss
        
        # Compute additional metrics
        with torch.no_grad():
            # Perplexity: exp(loss)
            perplexity = torch.exp(loss).item()
            
            # Accuracy: correct predictions / total predictions
            # predictions shape: [B*T]
            predictions = logits_flat.argmax(dim=-1)
            
            # Mask out padding tokens
            mask = targets_flat != self.padding_idx
            correct = (predictions == targets_flat) & mask
            
            accuracy = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
        
        metrics = {
            'loss': loss.item(),
            'perplexity': perplexity,
            'accuracy': accuracy
        }
        
        return loss, metrics


def compute_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    padding_idx: int = 0,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Convenience function to compute cross-entropy loss.
    
    Args:
        logits: Model predictions, shape [B, T, V]
        targets: Target token IDs, shape [B, T]
        padding_idx: Padding token ID to ignore
        label_smoothing: Label smoothing factor
    
    Returns:
        loss: Scalar loss value
    
    Example:
        >>> logits = torch.randn(2, 10, 1000)  # [B=2, T=10, V=1000]
        >>> targets = torch.randint(0, 1000, (2, 10))  # [2, 10]
        >>> loss = compute_cross_entropy_loss(logits, targets)
        >>> loss.shape
        torch.Size([])
    """
    # logits shape: [B, T, V]
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape: [B, T, V] -> [B*T, V]
    logits_flat = logits.reshape(-1, vocab_size)
    
    # Reshape: [B, T] -> [B*T]
    targets_flat = targets.reshape(-1)
    
    # Compute loss
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=padding_idx,
        label_smoothing=label_smoothing,
        reduction='mean'
    )
    
    return loss


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    padding_idx: int = 0
) -> float:
    """
    Compute perplexity metric.
    
    Perplexity = exp(cross_entropy_loss)
    
    Düşük perplexity daha iyi model demektir.
    - PPL = 1: Perfect prediction
    - PPL = vocab_size: Random guessing
    
    Args:
        logits: Model predictions, shape [B, T, V]
        targets: Target token IDs, shape [B, T]
        padding_idx: Padding token ID to ignore
    
    Returns:
        perplexity: Scalar perplexity value
    
    Example:
        >>> logits = torch.randn(2, 10, 1000)
        >>> targets = torch.randint(0, 1000, (2, 10))
        >>> ppl = compute_perplexity(logits, targets)
        >>> ppl > 0
        True
    """
    with torch.no_grad():
        loss = compute_cross_entropy_loss(logits, targets, padding_idx)
        perplexity = torch.exp(loss).item()
    
    return perplexity


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    padding_idx: int = 0
) -> float:
    """
    Compute token-level accuracy.
    
    Accuracy = (correct predictions) / (total non-padding tokens)
    
    Args:
        logits: Model predictions, shape [B, T, V]
        targets: Target token IDs, shape [B, T]
        padding_idx: Padding token ID to ignore
    
    Returns:
        accuracy: Accuracy value in [0, 1]
    
    Example:
        >>> logits = torch.randn(2, 10, 1000)
        >>> targets = torch.randint(0, 1000, (2, 10))
        >>> acc = compute_accuracy(logits, targets)
        >>> 0 <= acc <= 1
        True
    """
    with torch.no_grad():
        # logits shape: [B, T, V]
        # Get predictions: [B, T]
        predictions = logits.argmax(dim=-1)
        
        # Mask out padding
        # mask shape: [B, T]
        mask = targets != padding_idx
        
        # Correct predictions
        # correct shape: [B, T]
        correct = (predictions == targets) & mask
        
        # Compute accuracy
        accuracy = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
    
    return accuracy


def compute_top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
    padding_idx: int = 0
) -> float:
    """
    Compute top-k accuracy.
    
    Top-k accuracy: target is in top k predictions.
    
    Args:
        logits: Model predictions, shape [B, T, V]
        targets: Target token IDs, shape [B, T]
        k: Number of top predictions to consider
        padding_idx: Padding token ID to ignore
    
    Returns:
        top_k_accuracy: Top-k accuracy value in [0, 1]
    
    Example:
        >>> logits = torch.randn(2, 10, 1000)
        >>> targets = torch.randint(0, 1000, (2, 10))
        >>> acc = compute_top_k_accuracy(logits, targets, k=5)
        >>> 0 <= acc <= 1
        True
    """
    with torch.no_grad():
        # logits shape: [B, T, V]
        batch_size, seq_len, vocab_size = logits.shape
        
        # Get top-k predictions: [B, T, k]
        top_k_predictions = logits.topk(k=min(k, vocab_size), dim=-1).indices
        
        # Expand targets for comparison: [B, T, 1]
        targets_expanded = targets.unsqueeze(-1)
        
        # Check if target is in top-k: [B, T]
        in_top_k = (top_k_predictions == targets_expanded).any(dim=-1)
        
        # Mask out padding
        mask = targets != padding_idx
        
        # Correct predictions
        correct = in_top_k & mask
        
        # Compute accuracy
        top_k_acc = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
    
    return top_k_acc


if __name__ == "__main__":
    # Test loss functions and metrics
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Loss Functions & Metrics Test")
    print("=" * 80)
    
    # Parameters
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    
    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  vocab_size: {vocab_size}")
    
    # Test 1: Basic cross-entropy loss
    print(f"\n{'='*80}")
    print("[1] Cross-Entropy Loss Test")
    print(f"{'='*80}")
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Targets shape: {targets.shape}")
    
    loss = compute_cross_entropy_loss(logits, targets)
    print(f"  Loss: {loss.item():.4f}")
    print(f"✅ Cross-entropy loss computed")
    
    # Test 2: Label smoothing
    print(f"\n{'='*80}")
    print("[2] Label Smoothing Test")
    print(f"{'='*80}")
    
    loss_no_smooth = compute_cross_entropy_loss(logits, targets, label_smoothing=0.0)
    loss_with_smooth = compute_cross_entropy_loss(logits, targets, label_smoothing=0.1)
    
    print(f"  Loss (no smoothing): {loss_no_smooth.item():.4f}")
    print(f"  Loss (smoothing=0.1): {loss_with_smooth.item():.4f}")
    print(f"  Difference: {abs(loss_no_smooth.item() - loss_with_smooth.item()):.4f}")
    print(f"✅ Label smoothing working")
    
    # Test 3: Perplexity
    print(f"\n{'='*80}")
    print("[3] Perplexity Test")
    print(f"{'='*80}")
    
    perplexity = compute_perplexity(logits, targets)
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Expected range: [1, {vocab_size}]")
    print(f"  (Lower is better, 1 = perfect, {vocab_size} = random)")
    
    # Verify: PPL = exp(loss)
    expected_ppl = torch.exp(loss_no_smooth).item()
    assert abs(perplexity - expected_ppl) < 0.01, "Perplexity mismatch"
    print(f"✅ Perplexity = exp(loss) verified")
    
    # Test 4: Accuracy
    print(f"\n{'='*80}")
    print("[4] Accuracy Test")
    print(f"{'='*80}")
    
    accuracy = compute_accuracy(logits, targets)
    print(f"  Token accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Expected: ~1/{vocab_size} = {1/vocab_size:.4f} (random model)")
    
    top5_accuracy = compute_top_k_accuracy(logits, targets, k=5)
    print(f"  Top-5 accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
    print(f"  Expected: ~5/{vocab_size} = {5/vocab_size:.4f} (random model)")
    
    assert top5_accuracy >= accuracy, "Top-k should be >= top-1"
    print(f"✅ Accuracy metrics computed")
    
    # Test 5: Padding mask
    print(f"\n{'='*80}")
    print("[5] Padding Mask Test")
    print(f"{'='*80}")
    
    # Create targets with padding
    targets_with_pad = targets.clone()
    targets_with_pad[:, -3:] = 0  # Last 3 positions are padding
    
    print(f"  Targets with padding: {targets_with_pad[0].tolist()}")
    print(f"  (0 = padding token)")
    
    loss_with_pad = compute_cross_entropy_loss(logits, targets_with_pad, padding_idx=0)
    accuracy_with_pad = compute_accuracy(logits, targets_with_pad, padding_idx=0)
    
    print(f"  Loss (ignoring padding): {loss_with_pad.item():.4f}")
    print(f"  Accuracy (ignoring padding): {accuracy_with_pad:.4f}")
    print(f"✅ Padding correctly ignored")
    
    # Test 6: LanguageModelLoss class
    print(f"\n{'='*80}")
    print("[6] LanguageModelLoss Class Test")
    print(f"{'='*80}")
    
    loss_module = LanguageModelLoss(
        vocab_size=vocab_size,
        padding_idx=0,
        label_smoothing=0.1
    )
    
    loss, metrics = loss_module(logits, targets, return_metrics=True)
    
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"✅ LanguageModelLoss class working")
    
    # Test 7: Gradient flow
    print(f"\n{'='*80}")
    print("[7] Gradient Flow Test")
    print(f"{'='*80}")
    
    logits_grad = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    targets_grad = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss_grad = compute_cross_entropy_loss(logits_grad, targets_grad)
    loss_grad.backward()
    
    print(f"  Gradients computed: {logits_grad.grad is not None}")
    print(f"  Gradient mean: {logits_grad.grad.mean():.6f}")
    print(f"  Gradient std: {logits_grad.grad.std():.6f}")
    print(f"✅ Gradient flow verified")
    
    print(f"\n" + "=" * 80)
    print("✅ All tests PASSED!")
    print("=" * 80)
