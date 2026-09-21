"""
src/inference/batch_generation.py

Batch Text Generation

Bu modül birden fazla prompt'u aynı anda (batch olarak) process eder:
- Dynamic padding for variable-length sequences
- Attention mask handling for padded positions
- Parallel generation across batch
- Memory-efficient batching strategies

Batch Inference Benefits:
- Throughput: Process multiple requests simultaneously
- GPU Utilization: Better hardware utilization
- Latency: Individual request latency may increase but overall throughput improves

Trade-offs:
- Padding overhead: Shorter sequences waste computation
- Memory: Batch size limited by available memory
- Synchronization: All sequences must reach same length or EOS

Real-world usage:
- Serving multiple users simultaneously
- Offline batch processing
- Dataset evaluation
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging

from src.inference.generation import sample_next_token

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """
    Configuration for batch generation.
    
    Attributes:
        max_batch_size: Maximum number of sequences in batch
        max_seq_len: Maximum sequence length
        pad_token_id: Token ID for padding
        eos_token_id: Token ID for end-of-sequence
        dynamic_batching: Use dynamic batching (group similar lengths)
    """
    max_batch_size: int = 32
    max_seq_len: int = 512
    pad_token_id: int = 0
    eos_token_id: Optional[int] = None
    dynamic_batching: bool = True


def pad_sequences(
    sequences: List[List[int]],
    pad_token_id: int,
    max_length: Optional[int] = None,
    padding_side: str = 'right'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad sequences to same length.
    
    Args:
        sequences: List of token ID sequences, each of variable length
        pad_token_id: Token ID to use for padding
        max_length: Maximum length (default: longest sequence in batch)
        padding_side: 'right' or 'left' padding
    
    Returns:
        padded: Padded sequences, shape [batch, max_len]
        attention_mask: Attention mask (1 = real token, 0 = padding), shape [batch, max_len]
    
    Example:
        >>> sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        >>> padded, mask = pad_sequences(sequences, pad_token_id=0)
        >>> # padded: [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]]
        >>> # mask: [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
    """
    batch_size = len(sequences)
    
    # Find max length in batch
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    # Initialize padded tensor
    # padded shape: [batch, max_length]
    padded = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    
    # Initialize attention mask
    # attention_mask shape: [batch, max_length]
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
    
    # Fill in sequences and masks
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_length)
        
        if padding_side == 'right':
            # Right padding: [tokens, pad, pad, ...]
            # padded[i, :seq_len] shape: [seq_len]
            padded[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)
            # attention_mask[i, :seq_len] shape: [seq_len]
            attention_mask[i, :seq_len] = 1
        else:
            # Left padding: [pad, pad, ..., tokens]
            start_idx = max_length - seq_len
            # padded[i, start_idx:] shape: [seq_len]
            padded[i, start_idx:] = torch.tensor(seq[:seq_len], dtype=torch.long)
            # attention_mask[i, start_idx:] shape: [seq_len]
            attention_mask[i, start_idx:] = 1
    
    return padded, attention_mask


def create_attention_mask_for_generation(
    attention_mask: torch.Tensor,
    current_length: int
) -> torch.Tensor:
    """
    Create causal attention mask for generation with padding support.
    
    Combines:
    1. Causal mask (token i can only attend to j <= i)
    2. Padding mask (don't attend to pad tokens)
    
    Args:
        attention_mask: Padding mask, shape [batch, seq_len]
                       1 = real token, 0 = padding
        current_length: Current sequence length for causal masking
    
    Returns:
        Combined mask, shape [batch, 1, current_length, current_length]
        True positions are masked (no attention)
    
    Example:
        >>> # attention_mask: [batch, seq_len]
        >>> # [[1, 1, 1, 0],  # 3 tokens + 1 pad
        >>> #  [1, 1, 0, 0]]  # 2 tokens + 2 pads
        >>> mask = create_attention_mask_for_generation(attention_mask, 4)
        >>> # mask shape: [2, 1, 4, 4]
    """
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
    
    # 1. Create causal mask (lower triangular)
    # causal_mask shape: [current_length, current_length]
    causal_mask = torch.triu(
        torch.ones(current_length, current_length, device=device, dtype=torch.bool),
        diagonal=1
    )
    
    # 2. Create padding mask
    # attention_mask shape: [batch, seq_len]
    # We want shape [batch, 1, 1, seq_len] for broadcasting
    # Positions with 0 should be masked (True)
    padding_mask = (attention_mask == 0)
    # padding_mask shape: [batch, 1, 1, seq_len]
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    
    # 3. Combine masks
    # Expand causal mask for batch
    # causal_mask shape: [1, 1, current_length, current_length]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # Combine: mask if either causal or padding mask is True
    # combined_mask shape: [batch, 1, current_length, current_length]
    combined_mask = causal_mask | padding_mask[:, :, :, :current_length]
    
    return combined_mask


def batch_generate(
    model: nn.Module,
    prompt_sequences: List[List[int]],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    pad_token_id: int = 0,
    eos_token_id: Optional[int] = None,
    padding_side: str = 'left'
) -> Tuple[List[List[int]], Dict]:
    """
    Generate text for multiple prompts in batch.
    
    Args:
        model: Language model
        prompt_sequences: List of prompt token ID lists (variable length)
        max_new_tokens: Maximum tokens to generate per sequence
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        pad_token_id: Padding token ID
        eos_token_id: End-of-sequence token ID (stops generation)
        padding_side: 'left' or 'right' (left recommended for generation)
    
    Returns:
        generated_sequences: List of generated token ID lists
        stats: Generation statistics
    
    Example:
        >>> prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        >>> generated, stats = batch_generate(
        ...     model, prompts,
        ...     max_new_tokens=10,
        ...     pad_token_id=0
        ... )
        >>> print(f"Throughput: {stats['tokens_per_sec']:.1f} tokens/sec")
    """
    model.eval()
    device = next(model.parameters()).device
    
    batch_size = len(prompt_sequences)
    
    # Get prompt lengths
    prompt_lengths = [len(seq) for seq in prompt_sequences]
    max_prompt_len = max(prompt_lengths)
    
    # Pad prompts
    # input_ids shape: [batch, max_prompt_len]
    # attention_mask shape: [batch, max_prompt_len]
    input_ids, attention_mask = pad_sequences(
        prompt_sequences,
        pad_token_id=pad_token_id,
        padding_side=padding_side
    )
    
    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Track which sequences are done (encountered EOS)
    # done shape: [batch]
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Track original lengths for output extraction
    original_lengths = torch.tensor(prompt_lengths, device=device)
    
    logger.info(
        f"Batch generation: {batch_size} sequences, "
        f"prompt lengths {min(prompt_lengths)}-{max(prompt_lengths)}"
    )
    
    # Statistics
    import time
    start_time = time.time()
    tokens_generated = 0
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            current_length = input_ids.size(1)
            
            # Create attention mask
            # full_mask shape: [batch, 1, current_length, current_length]
            full_mask = create_attention_mask_for_generation(
                attention_mask,
                current_length
            )
            
            # Forward pass
            # logits shape: [batch, current_length, vocab_size]
            logits, _ = model(input_ids, mask=full_mask)
            
            # Get last position logits
            # next_token_logits shape: [batch, vocab_size]
            next_token_logits = logits[:, -1, :]
            
            # Sample next token
            # next_token shape: [batch]
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # For done sequences, use pad token
            if eos_token_id is not None:
                next_token = torch.where(done, pad_token_id, next_token)
            
            # Append to sequences
            # input_ids shape: [batch, current_length + 1]
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            
            # Update attention mask (1 for all new positions, even if done)
            # new_mask shape: [batch, 1]
            new_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            # Set to 0 for done sequences
            new_mask[done] = 0
            # attention_mask shape: [batch, current_length + 1]
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            
            tokens_generated += batch_size
            
            # Check for EOS
            if eos_token_id is not None:
                # done shape: [batch]
                done = done | (next_token == eos_token_id)
                
                # Stop if all sequences are done
                if done.all():
                    logger.info(f"All sequences done at step {step}")
                    break
    
    elapsed_time = time.time() - start_time
    
    # Extract generated sequences (remove padding)
    generated_sequences = []
    for i in range(batch_size):
        # Get sequence
        # seq shape: [seq_len]
        seq = input_ids[i]
        
        # Find first pad token (after original content)
        mask = attention_mask[i]
        # valid_length: scalar
        valid_length = mask.sum().item()
        
        # Extract valid tokens
        valid_seq = seq[:valid_length].cpu().tolist()
        generated_sequences.append(valid_seq)
    
    # Statistics
    stats = {
        'batch_size': batch_size,
        'prompt_lengths': prompt_lengths,
        'total_tokens_generated': tokens_generated,
        'elapsed_time': elapsed_time,
        'tokens_per_sec': tokens_generated / elapsed_time if elapsed_time > 0 else 0,
        'sequences_per_sec': batch_size / elapsed_time if elapsed_time > 0 else 0
    }
    
    logger.info(
        f"Batch generation complete: {batch_size} sequences, "
        f"{tokens_generated} tokens, {elapsed_time:.2f}s, "
        f"{stats['tokens_per_sec']:.1f} tokens/sec"
    )
    
    return generated_sequences, stats


if __name__ == "__main__":
    # Test batch generation
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Batch Generation Test")
    print("=" * 80)
    
    # Test padding
    print("\n1. Testing sequence padding...")
    
    sequences = [
        [1, 2, 3],
        [4, 5],
        [6, 7, 8, 9]
    ]
    
    padded, mask = pad_sequences(sequences, pad_token_id=0, padding_side='right')
    
    print(f"   Original sequences:")
    for i, seq in enumerate(sequences):
        print(f"     {i}: {seq}")
    
    print(f"\n   Padded (right):")
    print(f"     {padded}")
    print(f"   Attention mask:")
    print(f"     {mask}")
    
    # Test left padding
    padded_left, mask_left = pad_sequences(sequences, pad_token_id=0, padding_side='left')
    
    print(f"\n   Padded (left):")
    print(f"     {padded_left}")
    print(f"   Attention mask:")
    print(f"     {mask_left}")
    
    # Test attention mask creation
    print("\n2. Testing attention mask creation...")
    
    # attention_mask shape: [2, 4]
    test_mask = torch.tensor([
        [1, 1, 1, 0],  # 3 real tokens, 1 pad
        [1, 1, 0, 0]   # 2 real tokens, 2 pads
    ])
    
    # combined_mask shape: [2, 1, 4, 4]
    combined = create_attention_mask_for_generation(test_mask, current_length=4)
    
    print(f"   Input mask: {test_mask.shape}")
    print(f"     {test_mask}")
    print(f"   Combined mask: {combined.shape}")
    print(f"   First sequence attention pattern:")
    print(f"     {combined[0, 0]}")
    
    # Test batch generation
    print("\n3. Testing batch generation...")
    
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
    
    # Create test prompts (variable length)
    prompt_sequences = [
        [1, 2, 3, 4, 5],           # 5 tokens
        [10, 11, 12],               # 3 tokens
        [20, 21, 22, 23, 24, 25, 26]  # 7 tokens
    ]
    
    print(f"   Prompt lengths: {[len(p) for p in prompt_sequences]}")
    
    # Generate
    generated, stats = batch_generate(
        model,
        prompt_sequences,
        max_new_tokens=10,
        temperature=1.0,
        pad_token_id=0,
        padding_side='left'
    )
    
    print(f"\n   Generated sequences:")
    for i, seq in enumerate(generated):
        print(f"     {i}: length={len(seq)}, tokens={seq[:15]}...")
    
    print(f"\n   Statistics:")
    print(f"     Batch size: {stats['batch_size']}")
    print(f"     Total tokens: {stats['total_tokens_generated']}")
    print(f"     Time: {stats['elapsed_time']:.3f}s")
    print(f"     Throughput: {stats['tokens_per_sec']:.1f} tokens/sec")
    print(f"     Sequences/sec: {stats['sequences_per_sec']:.1f}")
    
    print("\n" + "=" * 80)
    print("✅ Batch generation tests completed!")
    print("=" * 80)
