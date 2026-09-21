"""
src/inference/beam_search.py

Beam Search Implementation

Beam search, text generation için alternatif bir stratejidir.
Greedy search sadece en yüksek olasılıklı token'ı seçerken,
beam search birden fazla hypothesis'i parallel olarak takip eder.

Beam Search Süreci:
1. Her step'te top-k en olasılıklı hypothesis'i tut (beam width = k)
2. Her hypothesis için tüm olası next token'ları dene
3. En yüksek cumulative score'a sahip k hypothesis'i seç
4. Bitene kadar tekrarla

Avantajları:
- Greedy'den daha iyi sonuç (global optimization)
- Sampling'den daha deterministic ve tutarlı

Dezavantajları:
- Daha yavaş (k parallel sequence)
- Repetitive text üretme eğilimi
- Diversity düşük olabilir
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BeamHypothesis:
    """
    Single beam hypothesis.
    
    Bir beam search hypothesis'i:
    - token_ids: Şimdiye kadar generate edilen token'lar
    - score: Log probability toplamı (cumulative score)
    - finished: EOS token'a ulaşıldı mı
    
    Attributes:
        token_ids: Token sequence, List[int]
        score: Cumulative log probability (higher is better)
        finished: Whether sequence ended with EOS
    """
    token_ids: List[int]
    score: float
    finished: bool = False
    
    def __len__(self) -> int:
        """Sequence length."""
        return len(self.token_ids)
    
    def __repr__(self) -> str:
        return f"BeamHypothesis(len={len(self)}, score={self.score:.3f}, finished={self.finished})"


def beam_search(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    beam_width: int = 5,
    max_new_tokens: int = 50,
    length_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    num_return_sequences: int = 1,
    early_stopping: bool = True
) -> List[BeamHypothesis]:
    """
    Beam search text generation.
    
    Beam search maintains k (beam_width) best hypotheses at each step.
    
    Args:
        model: Language model
        input_ids: Prompt tokens, shape [1, seq_len] (batch=1 for beam search)
        beam_width: Number of beams (hypotheses) to maintain
        max_new_tokens: Maximum tokens to generate
        length_penalty: Length normalization factor
            - > 1.0: favor longer sequences
            - < 1.0: favor shorter sequences
            - = 1.0: no length penalty
        eos_token_id: End-of-sequence token
        pad_token_id: Padding token
        num_return_sequences: Number of sequences to return (≤ beam_width)
        early_stopping: Stop when beam_width sequences finish
    
    Returns:
        List of BeamHypothesis, sorted by score (best first)
    
    Example:
        >>> model = GPTModel.from_pretrained("model.pt")
        >>> prompt = torch.tensor([[1, 2, 3]])
        >>> hypotheses = beam_search(
        ...     model, prompt,
        ...     beam_width=5,
        ...     max_new_tokens=50
        ... )
        >>> best = hypotheses[0]
        >>> print(f"Best score: {best.score:.3f}")
    """
    assert input_ids.size(0) == 1, "Beam search requires batch size = 1"
    assert num_return_sequences <= beam_width, "num_return_sequences must be ≤ beam_width"
    
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Get model config
    max_seq_len = model.config.max_seq_len if hasattr(model, 'config') else 512
    vocab_size = model.config.vocab_size if hasattr(model, 'config') else 50000
    
    # Initialize beams
    # Start with single hypothesis (the prompt)
    initial_tokens = input_ids[0].tolist()
    beams = [BeamHypothesis(token_ids=initial_tokens, score=0.0)]
    
    # Finished hypotheses
    finished_beams: List[BeamHypothesis] = []
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Stop if all beams are finished and early stopping enabled
            if early_stopping and len(finished_beams) >= beam_width:
                break
            
            # Collect all candidate hypotheses
            all_candidates: List[BeamHypothesis] = []
            
            # For each beam, compute next token probabilities
            for beam in beams:
                if beam.finished:
                    # Keep finished beams as-is
                    all_candidates.append(beam)
                    continue
                
                # Check max sequence length
                if len(beam.token_ids) >= max_seq_len:
                    beam.finished = True
                    all_candidates.append(beam)
                    continue
                
                # Prepare input
                # current_input shape: [1, current_len]
                current_input = torch.tensor([beam.token_ids], device=device)
                
                # Only use last max_seq_len tokens
                if current_input.size(1) > max_seq_len:
                    current_input = current_input[:, -max_seq_len:]
                
                # Forward pass
                # logits shape: [1, current_len, vocab_size]
                logits, _ = model(current_input)
                
                # Get logits for last position
                # next_logits shape: [vocab_size]
                next_logits = logits[0, -1, :]
                
                # Convert to log probabilities
                # log_probs shape: [vocab_size]
                log_probs = F.log_softmax(next_logits, dim=-1)
                
                # Get top beam_width tokens
                # top_log_probs shape: [beam_width]
                # top_indices shape: [beam_width]
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                
                # Create new hypotheses
                for log_prob, token_id in zip(top_log_probs.tolist(), top_indices.tolist()):
                    new_tokens = beam.token_ids + [token_id]
                    new_score = beam.score + log_prob
                    
                    # Check if finished
                    is_finished = (eos_token_id is not None and token_id == eos_token_id)
                    
                    new_beam = BeamHypothesis(
                        token_ids=new_tokens,
                        score=new_score,
                        finished=is_finished
                    )
                    
                    all_candidates.append(new_beam)
            
            # Select top beam_width hypotheses
            # Sort by normalized score (length penalty)
            def get_normalized_score(beam: BeamHypothesis) -> float:
                """Compute length-normalized score."""
                if length_penalty == 1.0:
                    return beam.score
                # Length normalization: score / (length ** alpha)
                length = len(beam.token_ids)
                return beam.score / (length ** length_penalty)
            
            # Sort candidates by normalized score
            all_candidates.sort(key=get_normalized_score, reverse=True)
            
            # Split into finished and ongoing
            beams = []
            for candidate in all_candidates:
                if candidate.finished:
                    # Add to finished if not already there
                    if candidate not in finished_beams:
                        finished_beams.append(candidate)
                else:
                    beams.append(candidate)
                
                # Keep only beam_width ongoing beams
                if len(beams) >= beam_width:
                    break
            
            # If no ongoing beams left, we're done
            if len(beams) == 0:
                break
            
            # Keep only beam_width beams
            beams = beams[:beam_width]
    
    # Combine finished and ongoing beams
    all_beams = finished_beams + beams
    
    # Sort by normalized score
    all_beams.sort(key=lambda b: get_normalized_score(b), reverse=True)
    
    # Return top num_return_sequences
    return all_beams[:num_return_sequences]


def beam_search_generate(
    model: torch.nn.Module,
    prompt: str,
    tokenizer,
    beam_width: int = 5,
    max_new_tokens: int = 50,
    length_penalty: float = 1.0,
    num_return_sequences: int = 1
) -> List[str]:
    """
    Beam search text generation from string prompt.
    
    Args:
        model: Language model
        prompt: Input text
        tokenizer: Tokenizer
        beam_width: Number of beams
        max_new_tokens: Max tokens to generate
        length_penalty: Length normalization
        num_return_sequences: Number of outputs to return
    
    Returns:
        List of generated text strings (best first)
    
    Example:
        >>> model = GPTModel.from_pretrained("model.pt")
        >>> tokenizer = BPETokenizer.from_file("tokenizer.json")
        >>> results = beam_search_generate(
        ...     model,
        ...     "Bir varmış bir yokmuş",
        ...     tokenizer,
        ...     beam_width=5,
        ...     num_return_sequences=3
        ... )
        >>> for i, text in enumerate(results):
        ...     print(f"{i+1}. {text}")
    """
    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    
    # Beam search
    hypotheses = beam_search(
        model,
        input_ids,
        beam_width=beam_width,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
        eos_token_id=getattr(tokenizer, 'eos_token_id', None),
        pad_token_id=getattr(tokenizer, 'pad_token_id', None),
        num_return_sequences=num_return_sequences
    )
    
    # Decode hypotheses
    results = []
    for hyp in hypotheses:
        text = tokenizer.decode(hyp.token_ids)
        results.append(text)
    
    return results


if __name__ == "__main__":
    # Beam search test
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Beam Search Test")
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
    
    # Test beam search
    print("\n1. Testing beam search...")
    
    prompt_ids = torch.randint(0, 100, (1, 10))
    print(f"   Prompt length: {prompt_ids.size(1)}")
    
    hypotheses = beam_search(
        model,
        prompt_ids,
        beam_width=3,
        max_new_tokens=20,
        num_return_sequences=3
    )
    
    print(f"   Generated {len(hypotheses)} hypotheses:")
    for i, hyp in enumerate(hypotheses):
        print(f"      {i+1}. {hyp}")
    
    # Test with different beam widths
    print("\n2. Testing different beam widths...")
    
    for width in [1, 3, 5]:
        hypotheses = beam_search(
            model,
            prompt_ids,
            beam_width=width,
            max_new_tokens=15,
            num_return_sequences=1
        )
        best = hypotheses[0]
        print(f"   Beam width={width}: score={best.score:.3f}, length={len(best)}")
    
    # Test length penalty
    print("\n3. Testing length penalty...")
    
    for penalty in [0.5, 1.0, 1.5]:
        hypotheses = beam_search(
            model,
            prompt_ids,
            beam_width=3,
            max_new_tokens=20,
            length_penalty=penalty,
            num_return_sequences=1
        )
        best = hypotheses[0]
        print(f"   Length penalty={penalty}: score={best.score:.3f}, length={len(best)}")
    
    print("\n" + "=" * 80)
    print("✅ Beam search tests completed!")
    print("=" * 80)
