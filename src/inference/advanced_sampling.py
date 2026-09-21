"""
src/inference/advanced_sampling.py

Advanced Sampling Strategies

Bu modül gelişmiş text generation sampling stratejilerini içerir:
- Contrastive Search: Degenerate repetition prevention
- Typical Sampling: Information-theoretic sampling
- Mirostat: Adaptive perplexity control
- Min-P Sampling: Minimum probability threshold
- Locally Typical Sampling: Entropy-based

Advanced Strategies Benefits:
- Better quality: Reduce repetition and incoherence
- More control: Fine-grained control over randomness
- Adaptive: Adjust based on model confidence
- Theory-grounded: Based on information theory principles

References:
- Contrastive Search: "A Contrastive Framework for Neural Text Generation" (Su et al., 2022)
- Typical Sampling: "Typical Decoding for Natural Language Generation" (Meister et al., 2022)
- Mirostat: "Mirostat: A Neural Text Decoding Algorithm..." (Basu et al., 2020)
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def contrastive_search(
    logits: torch.Tensor,
    past_tokens: torch.Tensor,
    past_embeddings: torch.Tensor,
    current_embedding: torch.Tensor,
    alpha: float = 0.6,
    top_k: int = 4
) -> torch.Tensor:
    """
    Contrastive search sampling.
    
    Balances model confidence with degeneration penalty.
    Penalizes tokens similar to previously generated tokens.
    
    Score(token) = (1-α) * P(token) - α * max_similarity(token, past)
    
    Args:
        logits: Next token logits, shape [batch, vocab_size]
        past_tokens: Previously generated tokens, shape [batch, past_len]
        past_embeddings: Embeddings of past tokens, shape [batch, past_len, d_model]
        current_embedding: Embedding for candidate tokens, shape [vocab_size, d_model]
        alpha: Balance parameter (0-1), higher = more penalty for repetition
        top_k: Number of top candidates to consider
    
    Returns:
        Selected token, shape [batch]
    
    Example:
        >>> logits = model(input_ids)[:, -1, :]  # [batch, vocab]
        >>> token = contrastive_search(
        ...     logits, past_tokens, past_embeddings,
        ...     model.embeddings.weight, alpha=0.6
        ... )
    """
    # logits shape: [batch, vocab_size]
    batch_size, vocab_size = logits.shape
    device = logits.device
    
    # Get top-k candidates by probability
    # probs shape: [batch, vocab_size]
    probs = F.softmax(logits, dim=-1)
    
    # top_k_probs shape: [batch, top_k]
    # top_k_indices shape: [batch, top_k]
    top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
    
    # Compute degeneration penalty
    if past_embeddings is not None and past_embeddings.size(1) > 0:
        # For each candidate, compute max similarity with past tokens
        # current_embedding shape: [vocab_size, d_model]
        # We need embeddings for top-k candidates
        
        # top_k_embeddings shape: [batch, top_k, d_model]
        top_k_embeddings = current_embedding[top_k_indices]
        
        # Normalize embeddings for cosine similarity
        # top_k_embeddings_norm shape: [batch, top_k, d_model]
        top_k_embeddings_norm = F.normalize(top_k_embeddings, p=2, dim=-1)
        
        # past_embeddings_norm shape: [batch, past_len, d_model]
        past_embeddings_norm = F.normalize(past_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        # similarity shape: [batch, top_k, past_len]
        similarity = torch.bmm(
            top_k_embeddings_norm,
            past_embeddings_norm.transpose(1, 2)
        )
        
        # Max similarity for each candidate
        # max_similarity shape: [batch, top_k]
        max_similarity, _ = similarity.max(dim=-1)
    else:
        # No past tokens, no penalty
        # max_similarity shape: [batch, top_k]
        max_similarity = torch.zeros(batch_size, top_k, device=device)
    
    # Compute contrastive scores
    # score = (1 - alpha) * prob - alpha * max_similarity
    # scores shape: [batch, top_k]
    scores = (1 - alpha) * top_k_probs - alpha * max_similarity
    
    # Select token with highest score
    # best_idx shape: [batch]
    best_idx = scores.argmax(dim=-1)
    
    # Get corresponding token
    # selected_token shape: [batch]
    selected_token = torch.gather(top_k_indices, dim=1, index=best_idx.unsqueeze(-1)).squeeze(-1)
    
    return selected_token


def typical_sampling(
    logits: torch.Tensor,
    tau: float = 0.95,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Typical sampling (locally typical sampling).
    
    Samples from tokens with entropy close to conditional entropy.
    Based on information theory - select "typical" tokens.
    
    Args:
        logits: Next token logits, shape [batch, vocab_size]
        tau: Threshold for cumulative mass (default: 0.95)
        temperature: Temperature scaling
    
    Returns:
        Sampled token, shape [batch]
    
    Reference:
        "Typical Decoding for Natural Language Generation" (Meister et al., 2022)
    
    Example:
        >>> logits = model(input_ids)[:, -1, :]
        >>> token = typical_sampling(logits, tau=0.95)
    """
    # logits shape: [batch, vocab_size]
    if temperature != 1.0:
        logits = logits / temperature
    
    # probs shape: [batch, vocab_size]
    probs = F.softmax(logits, dim=-1)
    
    # Compute entropy (negative log probability)
    # entropy shape: [batch, vocab_size]
    entropy = -torch.log(probs + 1e-10)
    
    # Compute conditional entropy (expected entropy)
    # H(X) = -sum(p(x) * log(p(x)))
    # conditional_entropy shape: [batch]
    conditional_entropy = (probs * entropy).sum(dim=-1, keepdim=True)
    
    # Compute deviation from conditional entropy
    # deviation shape: [batch, vocab_size]
    deviation = torch.abs(entropy - conditional_entropy)
    
    # Sort by deviation (ascending - closer to typical)
    # sorted_deviation shape: [batch, vocab_size]
    # sorted_indices shape: [batch, vocab_size]
    sorted_deviation, sorted_indices = torch.sort(deviation, dim=-1)
    
    # Get corresponding probabilities
    # sorted_probs shape: [batch, vocab_size]
    sorted_probs = torch.gather(probs, dim=1, index=sorted_indices)
    
    # Compute cumulative probabilities
    # cumsum_probs shape: [batch, vocab_size]
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find tokens within tau threshold
    # mask shape: [batch, vocab_size]
    mask = cumsum_probs < tau
    
    # Ensure at least one token is selected
    mask[:, 0] = True
    
    # Zero out probabilities outside mask
    # filtered_probs shape: [batch, vocab_size]
    filtered_probs = sorted_probs * mask.float()
    
    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    # Sample from filtered distribution
    # sampled_idx shape: [batch]
    sampled_idx = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
    
    # Get original token index
    # selected_token shape: [batch]
    selected_token = torch.gather(sorted_indices, dim=1, index=sampled_idx.unsqueeze(-1)).squeeze(-1)
    
    return selected_token


def mirostat_sampling(
    logits: torch.Tensor,
    target_surprise: float = 5.0,
    learning_rate: float = 0.1,
    current_surprise: Optional[float] = None
) -> Tuple[torch.Tensor, float]:
    """
    Mirostat sampling with adaptive perplexity control.
    
    Dynamically adjusts sampling threshold to maintain target perplexity.
    Prevents both repetition (low perplexity) and incoherence (high perplexity).
    
    Args:
        logits: Next token logits, shape [batch, vocab_size]
        target_surprise: Target surprise (bits) - typically 3-7
        learning_rate: Adaptation rate for threshold
        current_surprise: Current surprise estimate (None for first call)
    
    Returns:
        Tuple of (sampled_token, new_surprise):
            sampled_token: Selected token, shape [batch]
            new_surprise: Updated surprise estimate
    
    Reference:
        "Mirostat: A Neural Text Decoding Algorithm..." (Basu et al., 2020)
    
    Example:
        >>> surprise = None
        >>> for step in range(max_tokens):
        ...     logits = model(input_ids)[:, -1, :]
        ...     token, surprise = mirostat_sampling(
        ...         logits, target_surprise=5.0, current_surprise=surprise
        ...     )
    """
    # logits shape: [batch, vocab_size]
    batch_size = logits.size(0)
    
    if batch_size != 1:
        logger.warning(f"Mirostat designed for batch_size=1, got {batch_size}")
    
    # probs shape: [batch, vocab_size]
    probs = F.softmax(logits, dim=-1)
    
    # Compute surprises (negative log probabilities in bits)
    # surprise shape: [batch, vocab_size]
    surprise = -torch.log2(probs + 1e-10)
    
    # Initialize current surprise if not provided
    if current_surprise is None:
        # Use expected surprise (entropy)
        current_surprise = (probs * surprise).sum(dim=-1).mean().item()
    
    # Compute error
    error = target_surprise - current_surprise
    
    # Adaptive threshold: adjust based on error
    # Higher threshold -> more tokens considered -> higher perplexity
    # Lower threshold -> fewer tokens -> lower perplexity
    threshold = current_surprise + learning_rate * error
    
    # Select tokens with surprise below threshold
    # mask shape: [batch, vocab_size]
    mask = surprise <= threshold
    
    # Ensure at least one token
    if not mask.any():
        # If no tokens pass, select token with minimum surprise
        min_surprise_idx = surprise.argmin(dim=-1)
        mask = torch.zeros_like(mask, dtype=torch.bool)
        mask.scatter_(1, min_surprise_idx.unsqueeze(-1), True)
    
    # Filter probabilities
    # filtered_probs shape: [batch, vocab_size]
    filtered_probs = probs * mask.float()
    filtered_probs = filtered_probs / (filtered_probs.sum(dim=-1, keepdim=True) + 1e-10)
    
    # Sample
    # sampled_token shape: [batch]
    sampled_token = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
    
    # Update surprise estimate
    # Get surprise of sampled token
    sampled_surprise = surprise.gather(1, sampled_token.unsqueeze(-1)).squeeze(-1).mean().item()
    
    # Moving average
    new_surprise = learning_rate * sampled_surprise + (1 - learning_rate) * current_surprise
    
    return sampled_token, new_surprise


def min_p_sampling(
    logits: torch.Tensor,
    min_p: float = 0.05,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Min-P sampling: Filter tokens below minimum probability threshold.
    
    Unlike top-p, uses absolute probability threshold relative to max prob.
    More stable than top-p for varying distributions.
    
    Args:
        logits: Next token logits, shape [batch, vocab_size]
        min_p: Minimum probability ratio relative to max (0.0-1.0)
        temperature: Temperature scaling
    
    Returns:
        Sampled token, shape [batch]
    
    Example:
        >>> logits = model(input_ids)[:, -1, :]
        >>> token = min_p_sampling(logits, min_p=0.05)
    """
    # logits shape: [batch, vocab_size]
    if temperature != 1.0:
        logits = logits / temperature
    
    # probs shape: [batch, vocab_size]
    probs = F.softmax(logits, dim=-1)
    
    # Find max probability
    # max_prob shape: [batch, 1]
    max_prob, _ = probs.max(dim=-1, keepdim=True)
    
    # Compute threshold
    threshold = min_p * max_prob
    
    # Mask tokens below threshold
    # mask shape: [batch, vocab_size]
    mask = probs >= threshold
    
    # Filter probabilities
    # filtered_probs shape: [batch, vocab_size]
    filtered_probs = probs * mask.float()
    filtered_probs = filtered_probs / (filtered_probs.sum(dim=-1, keepdim=True) + 1e-10)
    
    # Sample
    # sampled_token shape: [batch]
    sampled_token = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
    
    return sampled_token


if __name__ == "__main__":
    # Test advanced sampling strategies
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Advanced Sampling Test")
    print("=" * 80)
    
    # Create test logits
    batch_size = 2
    vocab_size = 100
    
    # Test 1: Typical Sampling
    print("\n1. Testing typical sampling...")
    
    # logits shape: [2, 100]
    logits = torch.randn(batch_size, vocab_size)
    
    tokens = typical_sampling(logits, tau=0.95)
    print(f"   Logits: {logits.shape}")
    print(f"   Sampled tokens: {tokens}")
    print(f"   Token values: {tokens.tolist()}")
    
    # Test 2: Min-P Sampling
    print("\n2. Testing min-p sampling...")
    
    tokens_minp = min_p_sampling(logits, min_p=0.05)
    print(f"   Sampled tokens: {tokens_minp.tolist()}")
    
    # Compare with different min_p values
    print("\n   Testing different min_p values:")
    for min_p_val in [0.01, 0.05, 0.1, 0.2]:
        tokens = min_p_sampling(logits, min_p=min_p_val)
        print(f"     min_p={min_p_val}: tokens={tokens.tolist()}")
    
    # Test 3: Mirostat Sampling
    print("\n3. Testing mirostat sampling...")
    
    # Single batch for mirostat
    # logits_single shape: [1, 100]
    logits_single = logits[:1]
    
    surprise = None
    surprises = []
    
    print("   Running 10 steps:")
    for step in range(10):
        token, surprise = mirostat_sampling(
            logits_single,
            target_surprise=5.0,
            learning_rate=0.1,
            current_surprise=surprise
        )
        surprises.append(surprise)
        print(f"     Step {step}: token={token.item()}, surprise={surprise:.3f}")
    
    print(f"   Surprise values: {[f'{s:.3f}' for s in surprises[:5]]}...")
    print(f"   Final surprise: {surprises[-1]:.3f} (target: 5.0)")
    
    # Test 4: Contrastive Search (simplified)
    print("\n4. Testing contrastive search (simplified)...")
    
    # Create mock embeddings
    d_model = 64
    past_len = 5
    
    # past_tokens shape: [1, 5]
    past_tokens = torch.randint(0, vocab_size, (1, past_len))
    
    # past_embeddings shape: [1, 5, 64]
    past_embeddings = torch.randn(1, past_len, d_model)
    
    # current_embedding shape: [100, 64] (embedding matrix)
    current_embedding = torch.randn(vocab_size, d_model)
    
    # logits_single shape: [1, 100]
    token = contrastive_search(
        logits_single,
        past_tokens,
        past_embeddings,
        current_embedding,
        alpha=0.6,
        top_k=4
    )
    
    print(f"   Past tokens: {past_tokens[0].tolist()}")
    print(f"   Selected token: {token.item()}")
    
    # Test with different alpha values
    print("\n   Testing different alpha values:")
    for alpha_val in [0.0, 0.3, 0.6, 0.9]:
        token = contrastive_search(
            logits_single, past_tokens, past_embeddings,
            current_embedding, alpha=alpha_val
        )
        print(f"     alpha={alpha_val}: token={token.item()}")
    
    # Test 5: Compare strategies
    print("\n5. Comparing sampling strategies...")
    
    # logits shape: [1, 100]
    test_logits = torch.randn(1, vocab_size)
    
    strategies = {
        'typical': typical_sampling(test_logits),
        'min_p': min_p_sampling(test_logits),
        'mirostat': mirostat_sampling(test_logits, current_surprise=5.0)[0],
        'contrastive': contrastive_search(
            test_logits, past_tokens, past_embeddings, current_embedding
        )
    }
    
    print("   Tokens from different strategies:")
    for name, token in strategies.items():
        print(f"     {name:12s}: {token.item()}")
    
    print("\n" + "=" * 80)
    print("✅ Advanced sampling tests completed!")
    print("=" * 80)
