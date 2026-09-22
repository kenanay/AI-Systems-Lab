"""
src/training/sample_tracker.py

Sample Generation Tracking During Training

Bu modül training sırasında text generation samples'larını track eder:
- Periodic sample generation at checkpoints
- Quality metrics tracking (diversity, repetition, length)
- Sample history logging
- Comparison across training steps
- TensorBoard integration for visualization

Training sırasında model'in generation quality'sini izlemek,
model improvement'ı anlamak için kritik öneme sahiptir.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationSample:
    """
    Single generation sample with metadata.
    
    Bir training step'inde generate edilen text sample:
    - step: Training step number
    - epoch: Training epoch
    - prompt: Input prompt text
    - generated: Generated text
    - metrics: Quality metrics (diversity, repetition, etc.)
    
    Attributes:
        step: Global training step
        epoch: Training epoch
        prompt: Input prompt string
        generated: Generated text string
        tokens: Number of tokens generated
        metrics: Dictionary of quality metrics
        timestamp: ISO timestamp
    """
    step: int
    epoch: int
    prompt: str
    generated: str
    tokens: int
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step': self.step,
            'epoch': self.epoch,
            'prompt': self.prompt,
            'generated': self.generated,
            'tokens': self.tokens,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationSample":
        """Create from dictionary."""
        return cls(**data)


def compute_diversity_metrics(text: str) -> Dict[str, float]:
    """
    Compute text diversity metrics.
    
    Metrics:
    - unique_tokens: Ratio of unique tokens to total tokens
    - repetition_2gram: Ratio of repeated 2-grams
    - repetition_3gram: Ratio of repeated 3-grams
    - avg_token_length: Average token length (chars)
    
    Args:
        text: Generated text string
    
    Returns:
        Dictionary of diversity metrics
    
    Example:
        >>> metrics = compute_diversity_metrics("hello world hello")
        >>> print(metrics['unique_tokens'])  # 2/3 = 0.67
    """
    tokens = text.split()
    
    if len(tokens) == 0:
        return {
            'unique_tokens': 0.0,
            'repetition_2gram': 0.0,
            'repetition_3gram': 0.0,
            'avg_token_length': 0.0
        }
    
    # Unique token ratio
    unique_ratio = len(set(tokens)) / len(tokens)
    
    # N-gram repetition
    def ngram_repetition(n: int) -> float:
        """Calculate n-gram repetition ratio."""
        if len(tokens) < n:
            return 0.0
        
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if len(ngrams) == 0:
            return 0.0
        
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        
        # Repetition ratio: 1 - (unique / total)
        return 1.0 - (unique_ngrams / total_ngrams)
    
    rep_2gram = ngram_repetition(2)
    rep_3gram = ngram_repetition(3)
    
    # Average token length
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    
    return {
        'unique_tokens': unique_ratio,
        'repetition_2gram': rep_2gram,
        'repetition_3gram': rep_3gram,
        'avg_token_length': avg_len
    }


class SampleTracker:
    """
    Track generation samples during training.
    
    Periodically generates text samples and tracks quality metrics.
    Useful for monitoring model improvement and detecting issues
    (e.g., repetition, mode collapse).
    
    Features:
    - Periodic generation at specified intervals
    - Multiple test prompts support
    - Diversity and quality metrics
    - History tracking
    - TensorBoard integration
    - JSON export
    
    Args:
        model: Language model
        tokenizer: Tokenizer with encode/decode
        test_prompts: List of test prompts to use
        generation_config: Optional generation config
        log_dir: Directory to save samples
        log_to_tensorboard: Enable TensorBoard logging
    
    Example:
        >>> tracker = SampleTracker(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     test_prompts=["Once upon", "The quick"],
        ...     log_dir='samples'
        ... )
        >>> tracker.generate_samples(step=100, epoch=0)
        >>> tracker.save_history()
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        test_prompts: List[str],
        generation_config: Optional[Any] = None,
        log_dir: Optional[Union[str, Path]] = None,
        log_to_tensorboard: bool = False,
        tensorboard_logger: Optional[Any] = None
    ):
        """Initialize sample tracker."""
        self.model = model
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.generation_config = generation_config
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_to_tensorboard = log_to_tensorboard
        self.tb_logger = tensorboard_logger
        
        # Sample history
        self.history: List[GenerationSample] = []
        
        # Create log directory
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"SampleTracker initialized: {len(test_prompts)} prompts, "
            f"tensorboard={log_to_tensorboard}"
        )
    
    def generate_samples(
        self,
        step: int,
        epoch: int,
        max_new_tokens: int = 50
    ) -> List[GenerationSample]:
        """
        Generate samples for all test prompts.
        
        Args:
            step: Current training step
            epoch: Current epoch
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            List of GenerationSample objects
        
        Example:
            >>> samples = tracker.generate_samples(step=100, epoch=0)
            >>> for sample in samples:
            ...     print(f"Prompt: {sample.prompt}")
            ...     print(f"Generated: {sample.generated}")
        """
        from datetime import datetime
        
        samples = []
        
        # Set model to eval mode
        was_training = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            for prompt in self.test_prompts:
                try:
                    # Encode prompt
                    prompt_ids = self.tokenizer.encode(prompt)
                    # input_ids shape: [1, prompt_len]
                    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
                    
                    # Move to model device
                    device = next(self.model.parameters()).device
                    input_ids = input_ids.to(device)
                    
                    # Generate
                    if self.generation_config:
                        # Use custom config
                        from src.inference.generation import generate_text
                        output_ids = generate_text(
                            self.model,
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            temperature=getattr(self.generation_config, 'temperature', 1.0),
                            top_k=getattr(self.generation_config, 'top_k', None),
                            top_p=getattr(self.generation_config, 'top_p', None)
                        )
                    else:
                        # Simple greedy generation
                        from src.inference.generation import generate_text
                        output_ids = generate_text(
                            self.model,
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            temperature=0.8
                        )
                    
                    # Decode
                    generated_ids = output_ids[0].tolist()
                    generated_text = self.tokenizer.decode(generated_ids)
                    
                    # Compute metrics
                    metrics = compute_diversity_metrics(generated_text)
                    
                    # Create sample
                    sample = GenerationSample(
                        step=step,
                        epoch=epoch,
                        prompt=prompt,
                        generated=generated_text,
                        tokens=len(generated_ids),
                        metrics=metrics,
                        timestamp=datetime.now().isoformat()
                    )
                    
                    samples.append(sample)
                    
                    logger.info(
                        f"Step {step} | Prompt: '{prompt[:20]}...' | "
                        f"Generated: {sample.tokens} tokens | "
                        f"Diversity: {metrics['unique_tokens']:.2f}"
                    )
                    
                except Exception as e:
                    logger.error(f"Generation failed for prompt '{prompt}': {e}")
        
        # Restore training mode
        if was_training:
            self.model.train()
        
        # Add to history
        self.history.extend(samples)
        
        # Log to TensorBoard
        if self.log_to_tensorboard and self.tb_logger:
            self._log_to_tensorboard(samples, step)
        
        return samples
    
    def _log_to_tensorboard(
        self,
        samples: List[GenerationSample],
        step: int
    ):
        """
        Log samples to TensorBoard.
        
        Args:
            samples: List of generation samples
            step: Training step
        """
        if not self.tb_logger or not self.tb_logger.enabled:
            return
        
        for i, sample in enumerate(samples):
            # Log generated text
            self.tb_logger.log_text(
                f'samples/prompt_{i}',
                f"**Prompt:** {sample.prompt}\n\n**Generated:**\n{sample.generated}",
                step=step
            )
            
            # Log metrics
            for metric_name, metric_value in sample.metrics.items():
                self.tb_logger.log_scalar(
                    f'generation_quality/{metric_name}',
                    metric_value,
                    step=step
                )
    
    def get_history(
        self,
        prompt: Optional[str] = None
    ) -> List[GenerationSample]:
        """
        Get generation history.
        
        Args:
            prompt: Filter by specific prompt (optional)
        
        Returns:
            List of GenerationSample objects
        """
        if prompt is None:
            return self.history
        
        return [s for s in self.history if s.prompt == prompt]
    
    def get_metric_progression(
        self,
        metric_name: str,
        prompt: Optional[str] = None
    ) -> List[tuple]:
        """
        Get progression of a metric over training.
        
        Args:
            metric_name: Name of metric (e.g., 'unique_tokens')
            prompt: Filter by specific prompt (optional)
        
        Returns:
            List of (step, value) tuples
        
        Example:
            >>> progression = tracker.get_metric_progression('unique_tokens')
            >>> steps, values = zip(*progression)
            >>> # Plot: plt.plot(steps, values)
        """
        history = self.get_history(prompt)
        
        result = []
        for sample in history:
            if metric_name in sample.metrics:
                result.append((sample.step, sample.metrics[metric_name]))
        
        return result
    
    def save_history(self, filename: str = "generation_samples.json"):
        """
        Save generation history to JSON.
        
        Args:
            filename: Output filename
        """
        if not self.log_dir:
            logger.warning("No log_dir specified, skipping save")
            return
        
        filepath = self.log_dir / filename
        
        data = {
            'test_prompts': self.test_prompts,
            'total_samples': len(self.history),
            'samples': [s.to_dict() for s in self.history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Generation history saved: {filepath} ({len(self.history)} samples)")
    
    def load_history(self, filename: str = "generation_samples.json"):
        """
        Load generation history from JSON.
        
        Args:
            filename: Input filename
        """
        if not self.log_dir:
            logger.error("No log_dir specified")
            return
        
        filepath = self.log_dir / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.test_prompts = data.get('test_prompts', [])
        self.history = [
            GenerationSample.from_dict(s)
            for s in data.get('samples', [])
        ]
        
        logger.info(f"Generation history loaded: {filepath} ({len(self.history)} samples)")
    
    def print_summary(self):
        """Print summary of generation samples."""
        if not self.history:
            print("No samples generated yet")
            return
        
        print("=" * 80)
        print("Generation Sample Summary")
        print("=" * 80)
        print(f"\nTotal samples: {len(self.history)}")
        print(f"Test prompts: {len(self.test_prompts)}")
        
        # Latest samples
        print("\nLatest samples:")
        latest_step = max(s.step for s in self.history)
        latest_samples = [s for s in self.history if s.step == latest_step]
        
        for sample in latest_samples:
            print(f"\n  Prompt: '{sample.prompt}'")
            print(f"  Generated ({sample.tokens} tokens): '{sample.generated[:100]}...'")
            print(f"  Metrics:")
            for name, value in sample.metrics.items():
                print(f"    {name}: {value:.3f}")
        
        # Metric averages
        print("\nAverage metrics across all samples:")
        if self.history[0].metrics:
            metric_names = self.history[0].metrics.keys()
            for metric_name in metric_names:
                values = [
                    s.metrics[metric_name]
                    for s in self.history
                    if metric_name in s.metrics
                ]
                if values:
                    avg = sum(values) / len(values)
                    print(f"  {metric_name}: {avg:.3f}")


if __name__ == "__main__":
    # Test sample tracker
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Sample Tracker Test")
    print("=" * 80)
    
    # Test diversity metrics
    print("\n1. Testing diversity metrics...")
    
    test_texts = [
        "hello world hello world",  # High repetition
        "the quick brown fox jumps over the lazy dog",  # Low repetition
        "test test test test test"  # Very high repetition
    ]
    
    for text in test_texts:
        metrics = compute_diversity_metrics(text)
        print(f"\n   Text: '{text}'")
        print(f"   Unique tokens: {metrics['unique_tokens']:.2f}")
        print(f"   2-gram repetition: {metrics['repetition_2gram']:.2f}")
        print(f"   3-gram repetition: {metrics['repetition_3gram']:.2f}")
    
    # Test sample tracker with dummy model
    print("\n2. Testing sample tracker...")
    
    from src.model.gpt import GPTModel, GPTConfig
    import tempfile
    
    # Create dummy tokenizer
    class DummyTokenizer:
        def encode(self, text: str) -> List[int]:
            return [ord(c) % 100 for c in text[:10]]
        
        def decode(self, ids: List[int]) -> str:
            return ''.join([chr(i + 65) if i < 26 else str(i) for i in ids[:30]])
    
    config = GPTConfig(vocab_size=100, max_seq_len=32, d_model=64, n_layers=2, n_heads=2)
    model = GPTModel(config)
    tokenizer = DummyTokenizer()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = SampleTracker(
            model=model,
            tokenizer=tokenizer,
            test_prompts=["Test prompt 1", "Test prompt 2"],
            log_dir=tmpdir
        )
        
        print("\n3. Generating samples...")
        samples = tracker.generate_samples(step=0, epoch=0, max_new_tokens=20)
        print(f"   Generated {len(samples)} samples")
        
        samples = tracker.generate_samples(step=10, epoch=0, max_new_tokens=20)
        print(f"   Generated {len(samples)} more samples")
        
        print("\n4. Getting metric progression...")
        progression = tracker.get_metric_progression('unique_tokens')
        print(f"   Progression: {progression}")
        
        print("\n5. Saving history...")
        tracker.save_history()
        
        print("\n6. Summary:")
        tracker.print_summary()
    
    print("\n" + "=" * 80)
    print("✅ Sample tracker tests completed!")
    print("=" * 80)
