"""
src/inference/config.py

Generation Configuration & Stopping Criteria

Bu modül text generation için configuration management sağlar:
- GenerationConfig: Tüm generation parametrelerini içerir
- StoppingCriteria: Generation'ı durdurma koşulları
- Preset configs: Farklı use case'ler için hazır ayarlar

Configuration kullanımı generation sürecini standardize eder ve
farklı generation stratejilerini kolayca karşılaştırmayı sağlar.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """
    Sampling strategy types.
    
    - GREEDY: Always pick most likely token (deterministic)
    - SAMPLING: Sample from probability distribution
    - BEAM_SEARCH: Maintain multiple hypotheses
    """
    GREEDY = "greedy"
    SAMPLING = "sampling"
    BEAM_SEARCH = "beam_search"


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.
    
    Bu config generation sürecinin tüm parametrelerini içerir:
    - Sampling strategy (greedy, sampling, beam search)
    - Sampling parameters (temperature, top-k, top-p)
    - Stopping criteria (max length, EOS token, stop sequences)
    - Special tokens (EOS, PAD, BOS)
    - Output control (num_return_sequences, etc.)
    
    Attributes:
        # Strategy
        strategy: Sampling strategy (greedy/sampling/beam_search)
        
        # Sampling parameters
        temperature: Sampling temperature (0.0 = greedy, >1.0 = more random)
        top_k: Keep only top k tokens (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        
        # Beam search parameters
        beam_width: Number of beams for beam search
        length_penalty: Length normalization for beam search
        
        # Stopping criteria
        max_new_tokens: Maximum tokens to generate
        max_length: Maximum total sequence length (prompt + generated)
        min_length: Minimum total sequence length
        eos_token_id: End-of-sequence token ID
        stop_sequences: List of token sequences that stop generation
        
        # Special tokens
        pad_token_id: Padding token ID
        bos_token_id: Beginning-of-sequence token ID
        
        # Output control
        num_return_sequences: Number of sequences to return
        early_stopping: Stop when num_return_sequences beams finish (beam search)
        
        # Other
        do_sample: Whether to use sampling (vs greedy)
    
    Example:
        >>> # Greedy decoding
        >>> config = GenerationConfig(strategy=SamplingStrategy.GREEDY)
        >>> 
        >>> # Nucleus sampling
        >>> config = GenerationConfig(
        ...     strategy=SamplingStrategy.SAMPLING,
        ...     temperature=0.8,
        ...     top_p=0.9
        ... )
        >>> 
        >>> # Beam search
        >>> config = GenerationConfig(
        ...     strategy=SamplingStrategy.BEAM_SEARCH,
        ...     beam_width=5,
        ...     length_penalty=1.0
        ... )
    """
    
    # Strategy
    strategy: SamplingStrategy = SamplingStrategy.SAMPLING
    
    # Sampling parameters
    temperature: float = 1.0
    top_k: int = 0  # 0 = disabled
    top_p: float = 1.0  # 1.0 = disabled
    repetition_penalty: float = 1.0
    
    # Beam search parameters
    beam_width: int = 1
    length_penalty: float = 1.0
    
    # Stopping criteria
    max_new_tokens: int = 50
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    eos_token_id: Optional[int] = None
    stop_sequences: List[List[int]] = field(default_factory=list)
    
    # Special tokens
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    
    # Output control
    num_return_sequences: int = 1
    early_stopping: bool = True
    
    # Other
    do_sample: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        # Temperature validation
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        
        # Top-k validation
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        
        # Top-p validation
        if not 0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        
        # Repetition penalty validation
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")
        
        # Beam width validation
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {self.beam_width}")
        
        # Max tokens validation
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")
        
        # Num return sequences validation
        if self.num_return_sequences < 1:
            raise ValueError(f"num_return_sequences must be >= 1, got {self.num_return_sequences}")
        
        if self.strategy == SamplingStrategy.BEAM_SEARCH:
            if self.num_return_sequences > self.beam_width:
                raise ValueError(
                    f"num_return_sequences ({self.num_return_sequences}) must be "
                    f"<= beam_width ({self.beam_width})"
                )
        
        # Auto-set strategy based on parameters
        if self.strategy == SamplingStrategy.SAMPLING:
            if self.temperature == 0.0 or not self.do_sample:
                self.strategy = SamplingStrategy.GREEDY
                logger.info("Auto-switching to greedy decoding (temperature=0 or do_sample=False)")
            elif self.beam_width > 1:
                self.strategy = SamplingStrategy.BEAM_SEARCH
                logger.info(f"Auto-switching to beam search (beam_width={self.beam_width})")
    
    def to_dict(self):
        """Convert to dictionary."""
        result = asdict(self)
        result['strategy'] = self.strategy.value
        return result
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary."""
        config_dict = config_dict.copy()
        if 'strategy' in config_dict and isinstance(config_dict['strategy'], str):
            config_dict['strategy'] = SamplingStrategy(config_dict['strategy'])
        return cls(**config_dict)
    
    @classmethod
    def greedy(cls, max_new_tokens: int = 50, **kwargs) -> "GenerationConfig":
        """
        Greedy decoding preset.
        
        Always picks most likely token. Deterministic and fast.
        """
        return cls(
            strategy=SamplingStrategy.GREEDY,
            temperature=1.0,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    @classmethod
    def sampling(
        cls,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        max_new_tokens: int = 50,
        **kwargs
    ) -> "GenerationConfig":
        """
        Sampling preset with top-k and top-p.
        
        Good balance of diversity and quality.
        """
        return cls(
            strategy=SamplingStrategy.SAMPLING,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    @classmethod
    def nucleus(
        cls,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_new_tokens: int = 50,
        **kwargs
    ) -> "GenerationConfig":
        """
        Nucleus (top-p) sampling preset.
        
        More diverse than greedy, but coherent.
        """
        return cls(
            strategy=SamplingStrategy.SAMPLING,
            temperature=temperature,
            top_k=0,  # disabled
            top_p=top_p,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    @classmethod
    def beam_search(
        cls,
        beam_width: int = 5,
        length_penalty: float = 1.0,
        max_new_tokens: int = 50,
        num_return_sequences: int = 1,
        **kwargs
    ) -> "GenerationConfig":
        """
        Beam search preset.
        
        More coherent than sampling, but less diverse.
        Good for translations and structured outputs.
        """
        return cls(
            strategy=SamplingStrategy.BEAM_SEARCH,
            beam_width=beam_width,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=False,
            **kwargs
        )
    
    @classmethod
    def creative(
        cls,
        temperature: float = 1.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 100,
        **kwargs
    ) -> "GenerationConfig":
        """
        Creative text generation preset.
        
        High temperature and repetition penalty for diverse,
        creative outputs. Good for story generation.
        """
        return cls(
            strategy=SamplingStrategy.SAMPLING,
            temperature=temperature,
            top_k=0,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    @classmethod
    def deterministic(
        cls,
        temperature: float = 0.1,
        top_k: int = 10,
        max_new_tokens: int = 50,
        **kwargs
    ) -> "GenerationConfig":
        """
        Deterministic generation preset.
        
        Low temperature and small top-k for consistent,
        predictable outputs. Good for code generation.
        """
        return cls(
            strategy=SamplingStrategy.SAMPLING,
            temperature=temperature,
            top_k=top_k,
            top_p=1.0,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            **kwargs
        )


class StoppingCriteria:
    """
    Stopping criteria for text generation.
    
    Checks whether generation should stop based on:
    - Maximum length reached
    - EOS token generated
    - Stop sequence encountered
    - Custom stopping function
    
    Example:
        >>> criteria = StoppingCriteria(
        ...     max_length=100,
        ...     eos_token_id=2,
        ...     stop_sequences=[[13, 14]]  # "\n\n" tokens
        ... )
        >>> 
        >>> should_stop = criteria(input_ids, scores)
    """
    
    def __init__(
        self,
        max_length: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        stop_sequences: Optional[List[List[int]]] = None,
        custom_stop_fn: Optional[Callable] = None
    ):
        """
        Initialize stopping criteria.
        
        Args:
            max_length: Maximum sequence length
            eos_token_id: End-of-sequence token
            stop_sequences: List of token sequences that trigger stop
            custom_stop_fn: Custom function(input_ids) -> bool
        """
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.stop_sequences = stop_sequences or []
        self.custom_stop_fn = custom_stop_fn
    
    def __call__(self, input_ids: List[int], scores: Optional[List[float]] = None) -> bool:
        """
        Check if generation should stop.
        
        Args:
            input_ids: Current token sequence
            scores: Token scores (optional)
        
        Returns:
            True if generation should stop
        """
        # Check max length
        if self.max_length is not None and len(input_ids) >= self.max_length:
            return True
        
        # Check EOS token
        if self.eos_token_id is not None and len(input_ids) > 0:
            if input_ids[-1] == self.eos_token_id:
                return True
        
        # Check stop sequences
        for stop_seq in self.stop_sequences:
            if len(input_ids) >= len(stop_seq):
                # Check if sequence ends with stop sequence
                if input_ids[-len(stop_seq):] == stop_seq:
                    return True
        
        # Check custom function
        if self.custom_stop_fn is not None:
            if self.custom_stop_fn(input_ids):
                return True
        
        return False


if __name__ == "__main__":
    # Config test
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Generation Config Test")
    print("=" * 80)
    
    # Test presets
    print("\n1. Testing config presets...")
    
    configs = {
        "greedy": GenerationConfig.greedy(),
        "sampling": GenerationConfig.sampling(),
        "nucleus": GenerationConfig.nucleus(),
        "beam_search": GenerationConfig.beam_search(),
        "creative": GenerationConfig.creative(),
        "deterministic": GenerationConfig.deterministic()
    }
    
    for name, config in configs.items():
        print(f"\n   {name}:")
        print(f"      Strategy: {config.strategy.value}")
        print(f"      Temperature: {config.temperature}")
        print(f"      Top-k: {config.top_k}")
        print(f"      Top-p: {config.top_p}")
        if config.strategy == SamplingStrategy.BEAM_SEARCH:
            print(f"      Beam width: {config.beam_width}")
    
    # Test validation
    print("\n2. Testing validation...")
    
    try:
        bad_config = GenerationConfig(temperature=0.0, do_sample=False)
        print(f"   Config created, auto-switched to: {bad_config.strategy.value}")
    except ValueError as e:
        print(f"   ❌ Validation error: {e}")
    
    try:
        bad_config = GenerationConfig(temperature=-1.0)
        print(f"   ❌ Should have failed!")
    except ValueError as e:
        print(f"   ✅ Caught invalid temperature: {e}")
    
    # Test to_dict / from_dict
    print("\n3. Testing serialization...")
    
    config = GenerationConfig.sampling(temperature=0.7, top_k=40)
    config_dict = config.to_dict()
    print(f"   Serialized: {list(config_dict.keys())[:5]}...")
    
    restored = GenerationConfig.from_dict(config_dict)
    print(f"   Restored: strategy={restored.strategy.value}, temp={restored.temperature}")
    
    # Test stopping criteria
    print("\n4. Testing stopping criteria...")
    
    criteria = StoppingCriteria(
        max_length=10,
        eos_token_id=2,
        stop_sequences=[[5, 6, 7]]
    )
    
    # Test cases
    test_cases = [
        ([1, 2, 3], False, "short sequence"),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], True, "max length"),
        ([1, 2], True, "EOS token"),
        ([1, 3, 4, 5, 6, 7], True, "stop sequence"),
        ([1, 3, 4, 5], False, "no stop condition"),
    ]
    
    for tokens, expected, description in test_cases:
        result = criteria(tokens)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {description}: {result}")
    
    print("\n" + "=" * 80)
    print("✅ Config tests completed!")
    print("=" * 80)
