"""
src/inference/pipeline.py

End-to-End Inference Pipeline

Bu modül complete inference pipeline'ı sağlar:
- Model loading (checkpoint'tan)
- Tokenizer loading
- Text → tokens → model → tokens → text pipeline
- GenerationConfig integration
- Batch inference support

Pipeline, tüm inference component'lerini bir araya getirir ve
kullanımı kolay bir interface sunar.

Pipeline Süreci:
1. Load model + tokenizer
2. Encode input text → token IDs
3. Generate with model (sampling/beam search)
4. Decode token IDs → output text
5. Return result(s)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import logging
import json

from src.inference.config import GenerationConfig, SamplingStrategy
from src.inference.generation import generate_text, sample_next_token
from src.inference.beam_search import beam_search

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    End-to-end inference pipeline.
    
    Pipeline integrates:
    - Model loading from checkpoint
    - Tokenizer loading
    - Text generation with various strategies
    - Batch processing
    
    Args:
        model: Loaded PyTorch model
        tokenizer: Loaded tokenizer
        device: Device to run inference on ('cpu', 'cuda', 'mps')
        default_config: Default generation config
    
    Example:
        >>> # Create pipeline
        >>> pipeline = InferencePipeline.from_pretrained(
        ...     model_path="checkpoints/model.pt",
        ...     tokenizer_path="tokenizer.json"
        ... )
        >>> 
        >>> # Generate text
        >>> output = pipeline("Bir varmış bir yokmuş", max_new_tokens=50)
        >>> print(output)
        >>> 
        >>> # Generate with custom config
        >>> config = GenerationConfig.creative(temperature=1.2)
        >>> output = pipeline("Bir varmış bir yokmuş", config=config)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cpu",
        default_config: Optional[GenerationConfig] = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model: PyTorch model
            tokenizer: Tokenizer with encode/decode methods
            device: Device for inference
            default_config: Default generation config
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.default_config = default_config or GenerationConfig.sampling()
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        logger.info(
            f"InferencePipeline initialized: device={device}, "
            f"model_params={self._count_parameters():,}"
        )
    
    def _count_parameters(self) -> int:
        """Count model parameters."""
        return sum(p.numel() for p in self.model.parameters())
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        device: Optional[str] = None,
        default_config: Optional[GenerationConfig] = None
    ) -> "InferencePipeline":
        """
        Load pipeline from saved model and tokenizer.
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            tokenizer_path: Path to tokenizer file (.json)
            device: Device to use (auto-detect if None)
            default_config: Default generation config
        
        Returns:
            Initialized pipeline
        
        Example:
            >>> pipeline = InferencePipeline.from_pretrained(
            ...     "checkpoints/best_model.pt",
            ...     "tokenizer.json"
            ... )
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        logger.info(f"Loading model from {model_path}...")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                config_dict = checkpoint.get('config', None)
            else:
                state_dict = checkpoint
                config_dict = None
        else:
            state_dict = checkpoint
            config_dict = None
        
        # Create model from config or infer from state dict
        if config_dict is not None:
            # Model config available
            from src.model.gpt import GPTModel, GPTConfig
            
            # Convert config dict to GPTConfig
            model_config = GPTConfig(**config_dict) if isinstance(config_dict, dict) else config_dict
            model = GPTModel(model_config)
        else:
            # Try to infer from state dict
            logger.warning("Model config not found in checkpoint, attempting to infer...")
            from src.model.gpt import GPTModel, GPTConfig
            
            # Infer vocab size and d_model from embeddings
            embed_weight = state_dict.get('embeddings.token_embedding.weight', None)
            if embed_weight is not None:
                vocab_size, d_model = embed_weight.shape
                logger.info(f"Inferred: vocab_size={vocab_size}, d_model={d_model}")
                
                # Create default config
                model_config = GPTConfig(
                    vocab_size=vocab_size,
                    d_model=d_model
                )
                model = GPTModel(model_config)
            else:
                raise ValueError("Cannot infer model config from checkpoint")
        
        # Load state dict
        model.load_state_dict(state_dict)
        logger.info(f"Model loaded: {model.get_num_params():,} parameters")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}...")
        
        # Check tokenizer type based on file extension
        tokenizer_path = Path(tokenizer_path)
        if tokenizer_path.suffix == '.json':
            from src.tokenizer.bpe import BPETokenizer
            tokenizer = BPETokenizer.from_file(str(tokenizer_path))
        else:
            raise ValueError(f"Unsupported tokenizer format: {tokenizer_path.suffix}")
        
        logger.info(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")
        
        # Create pipeline
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            default_config=default_config
        )
    
    def __call__(
        self,
        prompt: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).
        
        Args:
            prompt: Input text or list of texts
            config: Generation config (uses default if None)
            **kwargs: Override config parameters
        
        Returns:
            Generated text or list of texts
        
        Example:
            >>> output = pipeline("Once upon a time")
            >>> outputs = pipeline(["Prompt 1", "Prompt 2"])
        """
        # Handle single vs batch input
        is_batch = isinstance(prompt, list)
        if not is_batch:
            prompt = [prompt]
        
        # Get config
        if config is None:
            config = self.default_config
        
        # Override config with kwargs
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = GenerationConfig.from_dict(config_dict)
        
        # Generate for each prompt
        results = []
        for p in prompt:
            result = self._generate_single(p, config)
            results.append(result)
        
        # Return single result or list
        return results[0] if not is_batch else results

    def generate(
        self,
        prompt: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s). Alias for __call__.
        """
        return self(prompt, config=config, **kwargs)
    
    def _generate_single(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> str:
        """
        Generate text for single prompt.
        
        Args:
            prompt: Input text
            config: Generation config
        
        Returns:
            Generated text
        """
        # Encode prompt
        # token_ids: List[int]
        token_ids = self.tokenizer.encode(prompt)
        
        # Convert to tensor
        # input_ids shape: [1, prompt_len]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        
        prompt_length = input_ids.size(1)
        
        # Generate based on strategy
        if config.strategy == SamplingStrategy.BEAM_SEARCH:
            # Beam search
            hypotheses = beam_search(
                self.model,
                input_ids,
                beam_width=config.beam_width,
                max_new_tokens=config.max_new_tokens,
                length_penalty=config.length_penalty,
                eos_token_id=config.eos_token_id or getattr(self.tokenizer, 'eos_token_id', None),
                pad_token_id=config.pad_token_id or getattr(self.tokenizer, 'pad_token_id', None),
                num_return_sequences=config.num_return_sequences,
                early_stopping=config.early_stopping
            )
            
            # Get best hypothesis
            best = hypotheses[0]
            output_ids = best.token_ids
        else:
            # Sampling or greedy
            # output_ids shape: [1, total_len]
            output_ids_tensor = generate_text(
                self.model,
                input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_k=config.top_k if config.do_sample else None,
                top_p=config.top_p if config.do_sample else None,
                eos_token_id=config.eos_token_id or getattr(self.tokenizer, 'eos_token_id', None),
                pad_token_id=config.pad_token_id or getattr(self.tokenizer, 'pad_token_id', None),
                repetition_penalty=config.repetition_penalty
            )
            
            output_ids = output_ids_tensor[0].tolist()
        
        # Decode
        generated_text = self.tokenizer.decode(output_ids)
        
        return generated_text
    
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ):
        """
        Generate text with streaming (token-by-token).
        
        Yields tokens as they are generated.
        
        Args:
            prompt: Input text
            config: Generation config
            **kwargs: Override config parameters
        
        Yields:
            Generated tokens (one at a time)
        
        Example:
            >>> for token_text in pipeline.generate_stream("Once upon"):
            ...     print(token_text, end="", flush=True)
        """
        # Get config
        if config is None:
            config = self.default_config
        
        # Override config with kwargs
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = GenerationConfig.from_dict(config_dict)
        
        # Encode prompt
        token_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        
        # Get model config
        model_cfg = getattr(self.model, 'config', None)
        max_seq_len: int = int(getattr(model_cfg, 'max_seq_len', 512)) if model_cfg is not None else 512
        
        # Generate token by token
        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                # Check max length
                if input_ids.size(1) >= max_seq_len:
                    break
                
                # Forward pass
                input_for_model = input_ids[:, -max_seq_len:]
                logits, _ = self.model(input_for_model)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :]
                
                # Sample next token
                if config.do_sample:
                    next_token = sample_next_token(
                        next_token_logits,
                        temperature=config.temperature,
                        top_k=config.top_k if config.top_k > 0 else None,
                        top_p=config.top_p if config.top_p < 1.0 else None
                    )
                else:
                    # Greedy
                    next_token = next_token_logits.argmax(dim=-1)
                
                # Check for EOS
                eos_token_id = config.eos_token_id or getattr(self.tokenizer, 'eos_token_id', None)
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                
                # Decode and yield single token
                token_text = self.tokenizer.decode([next_token.item()])
                yield token_text
    
    def save_config(self, path: Union[str, Path]):
        """Save default config to file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.default_config.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path}")
    
    def load_config(self, path: Union[str, Path]):
        """Load default config from file."""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        self.default_config = GenerationConfig.from_dict(config_dict)
        logger.info(f"Config loaded from {path}")


if __name__ == "__main__":
    # Pipeline test
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Inference Pipeline Test")
    print("=" * 80)
    
    from src.model.gpt import GPTModel, GPTConfig
    from src.tokenizer.bpe import BPETokenizer
    import tempfile
    
    # Create test model and tokenizer
    print("\n1. Creating test model and tokenizer...")
    
    model_config = GPTConfig(
        vocab_size=100,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4
    )
    
    model = GPTModel(model_config)
    
    # Create dummy tokenizer
    class DummyTokenizer:
        def __init__(self, vocab_size=100):
            self.vocab_size = vocab_size
            self.eos_token_id = 2
            self.pad_token_id = 0
        
        def encode(self, text: str) -> List[int]:
            # Simple char-based encoding
            return [ord(c) % self.vocab_size for c in text[:10]]
        
        def decode(self, token_ids: List[int]) -> str:
            # Simple decoding
            return ''.join([chr(t + 65) if t < 26 else str(t) for t in token_ids[:20]])
        
        def get_vocab_size(self) -> int:
            return self.vocab_size
    
    tokenizer = DummyTokenizer()
    
    print(f"   Model: {model.get_num_params():,} parameters")
    print(f"   Tokenizer: vocab_size={tokenizer.vocab_size}")
    
    # Test pipeline creation
    print("\n2. Testing pipeline creation...")
    
    pipeline = InferencePipeline(
        model=model,
        tokenizer=tokenizer,
        device="cpu"
    )
    
    print(f"   Pipeline created: device={pipeline.device}")
    
    # Test text generation
    print("\n3. Testing text generation...")
    
    prompt = "Test prompt"
    output = pipeline(prompt, max_new_tokens=10, temperature=1.0)
    
    print(f"   Prompt: '{prompt}'")
    print(f"   Output: '{output}'")
    
    # Test streaming generation
    print("\n4. Testing streaming generation...")
    
    print("   Generated (streaming): ", end="")
    for token in pipeline.generate_stream(prompt, max_new_tokens=10):
        print(token, end="", flush=True)
    print()
    
    # Test config save/load
    print("\n5. Testing config save/load...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_path = f.name
    
    try:
        pipeline.save_config(config_path)
        pipeline.load_config(config_path)
        print(f"   Config saved and loaded successfully")
    finally:
        Path(config_path).unlink()
    
    print("\n" + "=" * 80)
    print("✅ Pipeline tests completed!")
    print("=" * 80)
