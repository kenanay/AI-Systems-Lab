"""
src/inference/generation.py

Text Generation & Sampling Strategies

Bu modül autoregressive text generation için gerekli fonksiyonları içerir:
- Next-token prediction loop
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Greedy decoding

Generation process:
1. Prompt → tokens
2. Model forward pass → logits
3. Sample next token (with strategy)
4. Append to sequence
5. Repeat until stopping condition

Sampling stratejileri model çıktısının çeşitliliğini ve kalitesini kontrol eder.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Callable
import logging

logger = logging.getLogger(__name__)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:
    """
    Sample next token from logits using various strategies.
    
    Args:
        logits: Model output logits, shape [vocab_size] or [batch, vocab_size]
        temperature: Sampling temperature (higher = more random)
            - temperature < 1.0: more confident/deterministic
            - temperature = 1.0: standard sampling
            - temperature > 1.0: more diverse/random
        top_k: Keep only top k tokens (optional)
        top_p: Nucleus sampling - keep tokens with cumulative prob >= p (optional)
    
    Returns:
        Sampled token ID(s), shape [] or [batch]
    
    Example:
        >>> logits = model(input_ids)[:, -1, :]  # [batch, vocab_size]
        >>> next_token = sample_next_token(logits, temperature=0.8, top_k=50)
    """
    # Handle batch dimension
    # logits shape: [vocab_size] or [batch, vocab_size]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)  # [1, vocab_size]
    
    batch_size = logits.size(0)
    
    # Apply temperature
    # Higher temperature → flatter distribution (more random)
    # Lower temperature → sharper distribution (more deterministic)
    if temperature != 1.0:
        # logits shape: [batch, vocab_size]
        logits = logits / temperature
    
    # Top-k filtering
    if top_k is not None and top_k > 0:
        # Keep only top k tokens
        # values shape: [batch, top_k], indices shape: [batch, top_k]
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k, dim=-1)
        
        # Create mask for top-k tokens
        # mask shape: [batch, vocab_size]
        mask = torch.full_like(logits, float('-inf'))
        # scatter top-k values back
        mask.scatter_(dim=-1, index=indices, src=values)
        logits = mask
    
    # Convert to probabilities
    # probs shape: [batch, vocab_size]
    probs = F.softmax(logits, dim=-1)
    
    # Top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        # Sort probabilities in descending order
        # sorted_probs shape: [batch, vocab_size]
        # sorted_indices shape: [batch, vocab_size]
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Compute cumulative probabilities
        # cumsum_probs shape: [batch, vocab_size]
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        # Keep first token that exceeds threshold (nucleus)
        # mask shape: [batch, vocab_size]
        sorted_mask = cumsum_probs > top_p
        # Keep at least one token
        sorted_mask[..., 0] = False
        
        # Set removed tokens to 0 probability
        sorted_probs[sorted_mask] = 0.0
        
        # Renormalize
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        # Scatter back to original order
        # probs shape: [batch, vocab_size]
        probs = torch.zeros_like(probs).scatter_(
            dim=-1, 
            index=sorted_indices, 
            src=sorted_probs
        )
    
    # Sample from distribution
    # sampled shape: [batch]
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return sampled


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy sampling - always pick most likely token.
    
    Args:
        logits: Model output logits, shape [vocab_size] or [batch, vocab_size]
    
    Returns:
        Token ID(s) with highest probability, shape [] or [batch]
    """
    # logits shape: [batch, vocab_size] or [vocab_size]
    # argmax returns indices of maximum values
    # result shape: [batch] or []
    return logits.argmax(dim=-1)


def generate_text(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    repetition_penalty: float = 1.0
) -> torch.Tensor:
    """
    Generate text autoregressively.
    
    Autoregressive generation:
    1. Start with prompt tokens
    2. Model predicts next token
    3. Append to sequence
    4. Use updated sequence as new input
    5. Repeat until stopping condition
    
    Args:
        model: Language model
        input_ids: Prompt token IDs, shape [batch, seq_len]
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        eos_token_id: End-of-sequence token ID (stops generation)
        pad_token_id: Padding token ID
        repetition_penalty: Penalty for repeating tokens (> 1.0 discourages repetition)
    
    Returns:
        Generated token IDs, shape [batch, seq_len + new_tokens]
    
    Example:
        >>> from src.model.gpt import GPTModel
        >>> model = GPTModel.from_pretrained("checkpoints/model.pt")
        >>> prompt_ids = torch.tensor([[1, 2, 3, 4]])
        >>> generated = generate_text(
        ...     model, prompt_ids, 
        ...     max_new_tokens=50,
        ...     temperature=0.8,
        ...     top_k=50
        ... )
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Move input to device
    # input_ids shape: [batch, seq_len]
    input_ids = input_ids.to(device)
    batch_size = input_ids.size(0)
    
    # Track which sequences are done (encountered EOS)
    # done shape: [batch]
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Get model's max sequence length
    config = getattr(model, 'config', None)
    max_seq_len: int = int(getattr(config, 'max_seq_len', 512))
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Check if we've exceeded max sequence length
            if input_ids.size(1) >= max_seq_len:
                logger.warning(f"Reached max sequence length: {max_seq_len}")
                break
            
            # Forward pass
            # Only use last max_seq_len tokens if sequence is too long
            input_for_model = input_ids[:, -max_seq_len:]
            
            # logits shape: [batch, current_seq_len, vocab_size]
            # cache is ignored for now (can be added for efficiency)
            logits, _ = model(input_for_model)
            
            # Get logits for last position (next token prediction)
            # next_token_logits shape: [batch, vocab_size]
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                # Penalize tokens that already appeared
                for i in range(batch_size):
                    # Get unique tokens in sequence
                    # unique_tokens shape: [num_unique]
                    unique_tokens = input_ids[i].unique()
                    # Reduce logits for repeated tokens
                    next_token_logits[i, unique_tokens] /= repetition_penalty
            
            # Sample next token
            # next_token shape: [batch]
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # For sequences that are done, use pad token
            if eos_token_id is not None:
                if pad_token_id is not None:
                    next_token = torch.where(done, pad_token_id, next_token)
            
            # Append to sequence
            # input_ids shape: [batch, seq_len + 1]
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            
            # Check for EOS token
            if eos_token_id is not None:
                # done shape: [batch]
                done = done | (next_token == eos_token_id)
                
                # Stop if all sequences are done
                if done.all():
                    break
    
    return input_ids


def generate_with_prompt(
    model: torch.nn.Module,
    prompt: str,
    tokenizer,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    return_only_new_text: bool = False
) -> str:
    """
    Generate text from string prompt.
    
    Complete pipeline: text → tokens → generation → text
    
    Args:
        model: Language model
        prompt: Input text prompt
        tokenizer: Tokenizer with encode/decode methods
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        return_only_new_text: If True, return only generated text (exclude prompt)
    
    Returns:
        Generated text string
    
    Example:
        >>> model = GPTModel.from_pretrained("model.pt")
        >>> tokenizer = BPETokenizer.from_file("tokenizer.json")
        >>> text = generate_with_prompt(
        ...     model, 
        ...     "Bir varmış bir yokmuş",
        ...     tokenizer,
        ...     max_new_tokens=100,
        ...     temperature=0.8
        ... )
    """
    # Encode prompt
    # token_ids: List[int]
    token_ids = tokenizer.encode(prompt)
    
    # Convert to tensor
    # input_ids shape: [1, prompt_len]
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    
    prompt_length = input_ids.size(1)
    
    # Generate
    # output_ids shape: [1, prompt_len + new_tokens]
    output_ids = generate_text(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=getattr(tokenizer, 'eos_token_id', None),
        pad_token_id=getattr(tokenizer, 'pad_token_id', None)
    )
    
    # Decode
    if return_only_new_text:
        # Only decode generated tokens
        generated_ids = output_ids[0, prompt_length:].tolist()
    else:
        # Decode full sequence
        generated_ids = output_ids[0].tolist()
    
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


if __name__ == "__main__":
    # Simple generation test
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Text Generation Test")
    print("=" * 80)
    
    # Test sampling functions
    print("\n1. Testing sampling strategies...")
    
    # Create dummy logits
    vocab_size = 100
    logits = torch.randn(vocab_size)
    
    # Greedy
    greedy_token = greedy_sample(logits)
    print(f"   Greedy: {greedy_token.item()}")
    
    # Temperature sampling
    temp_token = sample_next_token(logits, temperature=0.5)
    print(f"   Temperature (0.5): {temp_token.item()}")
    
    # Top-k
    topk_token = sample_next_token(logits, temperature=1.0, top_k=10)
    print(f"   Top-k (k=10): {topk_token.item()}")
    
    # Top-p
    topp_token = sample_next_token(logits, temperature=1.0, top_p=0.9)
    print(f"   Top-p (p=0.9): {topp_token.item()}")
    
    # Test generation with dummy model
    print("\n2. Testing generation loop...")
    
    from src.model.gpt import GPTModel, GPTConfig
    
    # Create small model
    config = GPTConfig(
        vocab_size=100,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4
    )
    
    model = GPTModel(config)
    model.eval()
    
    # Generate from random prompt
    prompt_ids = torch.randint(0, 100, (1, 10))
    print(f"   Prompt length: {prompt_ids.size(1)}")
    
    generated = generate_text(
        model,
        prompt_ids,
        max_new_tokens=20,
        temperature=1.0
    )
    
    print(f"   Generated length: {generated.size(1)}")
    print(f"   New tokens: {generated.size(1) - prompt_ids.size(1)}")
    
    print("\n3. Testing repetition penalty...")
    
    # Generate with repetition penalty
    generated_with_penalty = generate_text(
        model,
        prompt_ids,
        max_new_tokens=20,
        temperature=1.0,
        repetition_penalty=1.5
    )
    
    print(f"   Generated with penalty: {generated_with_penalty.size(1)}")
    
    print("\n" + "=" * 80)
    print("✅ Generation tests completed!")
    print("=" * 80)
