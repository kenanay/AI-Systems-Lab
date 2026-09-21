"""
src/inference/streaming_generation.py

Streaming Text Generation

Bu modül token'ları generate edilirken real-time yield eder:
- Generator-based streaming
- Token-by-token callbacks
- Partial text decoding
- Streaming with batches
- Cancellation support

Streaming Benefits:
- Better UX: Users see output immediately
- Early stopping: Stop generation mid-stream
- Live processing: Process tokens as they arrive
- Memory efficient: Don't need to wait for full completion

Use Cases:
- Chat interfaces (ChatGPT-style)
- Live translation
- Real-time transcription
- Interactive applications

Example:
    for token_id, partial_text in stream_generate(model, prompt):
        print(partial_text, end='', flush=True)
"""

import torch
import torch.nn as nn
from typing import Generator, Optional, Callable, List, Tuple, Dict, Any
from dataclasses import dataclass
import logging

from src.inference.generation import sample_next_token

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """
    Configuration for streaming generation.
    
    Attributes:
        buffer_tokens: Number of tokens to buffer before yielding (1 = immediate)
        yield_incomplete_utf8: Whether to yield incomplete UTF-8 sequences
        callback_every_n: Call callback every N tokens
    """
    buffer_tokens: int = 1
    yield_incomplete_utf8: bool = False
    callback_every_n: int = 1


def stream_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    callback: Optional[Callable[[int, int], bool]] = None
) -> Generator[Tuple[int, torch.Tensor], None, None]:
    """
    Stream generate tokens one-by-one.
    
    Yields tokens as they are generated, enabling real-time output.
    
    Args:
        model: Language model
        input_ids: Prompt token IDs, shape [batch, seq_len] (batch=1 for streaming)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        eos_token_id: Stop generation at this token
        pad_token_id: Padding token ID
        callback: Optional callback function(step, token_id) -> continue_generation
                 Return False to stop generation early
    
    Yields:
        Tuple of (token_id, full_sequence):
            token_id: Newly generated token ID (scalar)
            full_sequence: Current full sequence, shape [batch, current_len]
    
    Example:
        >>> model = GPTModel.from_pretrained("model.pt")
        >>> prompt_ids = torch.tensor([[1, 2, 3]])
        >>> for step, (token_id, sequence) in enumerate(stream_generate(
        ...     model, prompt_ids, max_new_tokens=50
        ... )):
        ...     print(f"Step {step}: generated token {token_id}")
        ...     # Process token in real-time
    """
    model.eval()
    device = next(model.parameters()).device
    
    # input_ids shape: [batch, seq_len]
    input_ids = input_ids.to(device)
    batch_size = input_ids.size(0)
    
    if batch_size != 1:
        logger.warning(f"Streaming with batch_size={batch_size}, only first sequence will be monitored")
    
    # Track done state
    # done shape: [batch]
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Get model's max sequence length
    max_seq_len = model.config.max_seq_len if hasattr(model, 'config') else 512
    
    logger.debug(f"Starting streaming generation: max_new_tokens={max_new_tokens}")
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Check max length
            if input_ids.size(1) >= max_seq_len:
                logger.warning(f"Reached max sequence length: {max_seq_len}")
                break
            
            # Forward pass
            # Limit context window
            input_for_model = input_ids[:, -max_seq_len:]
            
            # logits shape: [batch, current_len, vocab_size]
            logits, _ = model(input_for_model)
            
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
            if eos_token_id is not None and pad_token_id is not None:
                next_token = torch.where(done, pad_token_id, next_token)
            
            # Append to sequence
            # input_ids shape: [batch, seq_len + 1]
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            
            # Extract token ID for yielding (first batch element)
            token_id = next_token[0].item()
            
            # Yield token and current sequence
            yield token_id, input_ids.clone()
            
            # Call callback if provided
            if callback is not None:
                should_continue = callback(step, token_id)
                if not should_continue:
                    logger.info(f"Generation stopped by callback at step {step}")
                    break
            
            # Check for EOS
            if eos_token_id is not None:
                done = done | (next_token == eos_token_id)
                if done.all():
                    logger.debug(f"EOS reached at step {step}")
                    break


def stream_generate_text(
    model: nn.Module,
    prompt: str,
    tokenizer,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    callback: Optional[Callable[[str, str], bool]] = None
) -> Generator[str, None, None]:
    """
    Stream generate text with automatic tokenization/detokenization.
    
    Yields decoded text incrementally as tokens are generated.
    
    Args:
        model: Language model
        prompt: Input text prompt
        tokenizer: Tokenizer with encode/decode methods
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        callback: Optional callback function(token_text, full_text) -> continue_generation
    
    Yields:
        Partial generated text after each new token
    
    Example:
        >>> for partial_text in stream_generate_text(
        ...     model, "Hello", tokenizer, max_new_tokens=20
        ... ):
        ...     print(partial_text, end='', flush=True)
        Hello world! How are you doing today?
    """
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    # input_ids shape: [1, prompt_len]
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    
    prompt_length = len(prompt_tokens)
    
    # Stream generation
    for token_id, full_sequence in stream_generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=getattr(tokenizer, 'eos_token_id', None),
        pad_token_id=getattr(tokenizer, 'pad_token_id', None)
    ):
        # Decode full sequence
        # full_sequence shape: [1, current_len]
        full_tokens = full_sequence[0].tolist()
        full_text = tokenizer.decode(full_tokens)
        
        # Decode just the new token
        token_text = tokenizer.decode([token_id])
        
        # Yield full text
        yield full_text
        
        # Call callback if provided
        if callback is not None:
            should_continue = callback(token_text, full_text)
            if not should_continue:
                break


def stream_generate_with_delta(
    model: nn.Module,
    prompt: str,
    tokenizer,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> Generator[Tuple[str, str], None, None]:
    """
    Stream generate with both delta (new) and full text.
    
    Useful for incremental UI updates where you need both the new
    text chunk and the complete text.
    
    Args:
        model: Language model
        prompt: Input text prompt
        tokenizer: Tokenizer
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
    
    Yields:
        Tuple of (delta_text, full_text):
            delta_text: Newly generated text chunk
            full_text: Complete generated text so far
    
    Example:
        >>> for delta, full in stream_generate_with_delta(
        ...     model, "Hello", tokenizer
        ... ):
        ...     print(f"New: '{delta}'")
        ...     print(f"Full: '{full}'")
        New: ' world'
        Full: 'Hello world'
        New: '!'
        Full: 'Hello world!'
    """
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    # input_ids shape: [1, prompt_len]
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    
    previous_text = prompt
    
    # Stream generation
    for token_id, full_sequence in stream_generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=getattr(tokenizer, 'eos_token_id', None),
        pad_token_id=getattr(tokenizer, 'pad_token_id', None)
    ):
        # Decode full sequence
        # full_sequence shape: [1, current_len]
        full_tokens = full_sequence[0].tolist()
        current_text = tokenizer.decode(full_tokens)
        
        # Calculate delta (new text)
        if current_text.startswith(previous_text):
            delta_text = current_text[len(previous_text):]
        else:
            # Fallback: decode just the token
            delta_text = tokenizer.decode([token_id])
        
        # Yield both delta and full
        yield delta_text, current_text
        
        previous_text = current_text


def batch_stream_generate(
    model: nn.Module,
    prompt_sequences: List[List[int]],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    pad_token_id: int = 0,
    eos_token_id: Optional[int] = None
) -> Generator[Tuple[int, List[int]], None, None]:
    """
    Stream generate for multiple sequences in batch.
    
    Yields after each generation step with status for all sequences.
    
    Args:
        model: Language model
        prompt_sequences: List of prompt token sequences
        max_new_tokens: Maximum tokens per sequence
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        pad_token_id: Padding token ID
        eos_token_id: End-of-sequence token ID
    
    Yields:
        Tuple of (step, new_tokens):
            step: Current generation step
            new_tokens: List of newly generated tokens for each sequence
    
    Example:
        >>> prompts = [[1, 2, 3], [4, 5]]
        >>> for step, tokens in batch_stream_generate(
        ...     model, prompts, max_new_tokens=10
        ... ):
        ...     print(f"Step {step}: {tokens}")
    """
    from src.inference.batch_generation import pad_sequences, create_attention_mask_for_generation
    
    model.eval()
    device = next(model.parameters()).device
    
    batch_size = len(prompt_sequences)
    
    # Pad prompts
    # input_ids shape: [batch, max_prompt_len]
    # attention_mask shape: [batch, max_prompt_len]
    input_ids, attention_mask = pad_sequences(
        prompt_sequences,
        pad_token_id=pad_token_id,
        padding_side='left'
    )
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Track done sequences
    # done shape: [batch]
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    logger.debug(f"Batch streaming: {batch_size} sequences")
    
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
            
            # Sample next tokens
            # next_tokens shape: [batch]
            next_tokens = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # For done sequences, use pad token
            if eos_token_id is not None:
                next_tokens = torch.where(done, pad_token_id, next_tokens)
            
            # Append to sequences
            # input_ids shape: [batch, current_length + 1]
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention mask
            # new_mask shape: [batch, 1]
            new_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            new_mask[done] = 0
            # attention_mask shape: [batch, current_length + 1]
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            
            # Yield step and new tokens
            # next_tokens: List[int]
            yield step, next_tokens.cpu().tolist()
            
            # Check EOS
            if eos_token_id is not None:
                done = done | (next_tokens == eos_token_id)
                if done.all():
                    logger.debug(f"All sequences done at step {step}")
                    break


if __name__ == "__main__":
    # Test streaming generation
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Streaming Generation Test")
    print("=" * 80)
    
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
    
    # Test 1: Basic token streaming
    print("\n1. Testing token streaming...")
    
    # input_ids shape: [1, 5]
    prompt_ids = torch.randint(0, 100, (1, 5))
    print(f"   Prompt: {prompt_ids[0].tolist()}")
    
    generated_tokens = []
    for step, (token_id, sequence) in enumerate(stream_generate(
        model, prompt_ids, max_new_tokens=10, temperature=1.0
    )):
        generated_tokens.append(token_id)
        print(f"   Step {step}: token={token_id}, seq_len={sequence.size(1)}")
        
        if step >= 4:  # Show first 5
            break
    
    print(f"   Generated tokens: {generated_tokens}")
    
    # Test 2: Streaming with callback
    print("\n2. Testing streaming with callback...")
    
    stop_at_step = 7
    
    def early_stop_callback(step: int, token_id: int) -> bool:
        """Stop generation at specific step."""
        print(f"   Callback: step={step}, token={token_id}")
        return step < stop_at_step
    
    count = 0
    for step, (token_id, sequence) in enumerate(stream_generate(
        model, prompt_ids,
        max_new_tokens=15,
        callback=early_stop_callback
    )):
        count += 1
    
    print(f"   Generated {count} tokens (stopped by callback)")
    assert count == stop_at_step + 1, f"Expected {stop_at_step + 1}, got {count}"
    
    # Test 3: Batch streaming
    print("\n3. Testing batch streaming...")
    
    prompt_sequences = [
        [1, 2, 3],
        [4, 5],
        [6, 7, 8, 9]
    ]
    
    print(f"   Prompts: {len(prompt_sequences)} sequences")
    
    for step, tokens in batch_stream_generate(
        model, prompt_sequences,
        max_new_tokens=5,
        pad_token_id=0
    ):
        print(f"   Step {step}: tokens={tokens}")
    
    # Test 4: Streaming with text (mock tokenizer)
    print("\n4. Testing text streaming (simulated)...")
    
    class MockTokenizer:
        """Simple mock tokenizer for testing."""
        def __init__(self):
            self.vocab = {i: f"tok{i}" for i in range(100)}
            self.eos_token_id = 99
            self.pad_token_id = 0
        
        def encode(self, text: str) -> List[int]:
            # Simple: return [1, 2, 3] for any text
            return [1, 2, 3]
        
        def decode(self, tokens: List[int]) -> str:
            # Simple: concatenate token representations
            return " ".join(self.vocab.get(t, f"<{t}>") for t in tokens)
    
    tokenizer = MockTokenizer()
    
    print("   Streaming text:")
    print("   ", end='', flush=True)
    
    count = 0
    for partial_text in stream_generate_text(
        model, "test prompt", tokenizer,
        max_new_tokens=5
    ):
        # In real app: print(partial_text[-10:], end='', flush=True)
        count += 1
    
    print(f"\n   Streamed {count} updates")
    
    # Test 5: Delta streaming
    print("\n5. Testing delta streaming...")
    
    count = 0
    for delta, full in stream_generate_with_delta(
        model, "test", tokenizer, max_new_tokens=3
    ):
        print(f"   Delta: '{delta[:20]}...' | Full length: {len(full)}")
        count += 1
    
    print(f"   Generated {count} deltas")
    
    print("\n" + "=" * 80)
    print("✅ Streaming generation tests completed!")
    print("=" * 80)
