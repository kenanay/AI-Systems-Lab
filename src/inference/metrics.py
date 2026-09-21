"""
src/inference/metrics.py

Generation Performance Metrics

Bu modül text generation performance metriklerini ölçer ve track eder:
- TTFT (Time To First Token): Prefill phase latency
- TPOT (Time Per Output Token): Per-token decode latency
- Throughput: Tokens/sec, requests/sec
- Latency: Total generation time
- Memory usage: Peak memory, cache overhead
- Token statistics: Generated tokens, prompt length

Metrics Categories:
1. Latency Metrics: TTFT, TPOT, total time
2. Throughput Metrics: tokens/sec, requests/sec
3. Resource Metrics: memory, GPU utilization
4. Quality Metrics: perplexity, diversity

These metrics are essential for:
- Performance optimization
- SLA monitoring
- Cost estimation
- Capacity planning
- A/B testing different strategies
"""

import torch
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """
    Comprehensive generation metrics.
    
    Attributes:
        prompt_length: Number of tokens in prompt
        generated_tokens: Number of tokens generated
        total_tokens: Total tokens (prompt + generated)
        
        # Timing metrics
        prefill_time: Time to process prompt (TTFT), seconds
        decode_time: Time to generate tokens, seconds
        total_time: Total generation time, seconds
        
        # Per-token metrics
        time_to_first_token: TTFT in seconds
        time_per_output_token: Average TPOT in seconds
        
        # Throughput metrics
        tokens_per_second: Overall throughput (tokens/sec)
        requests_per_second: Request throughput (for batch)
        
        # Memory metrics
        peak_memory_mb: Peak memory usage in MB
        cache_memory_mb: KV cache memory in MB
        
        # Metadata
        timestamp: When generation started
        model_name: Model identifier
        batch_size: Number of sequences
        
        # Token-level timing (optional detailed tracking)
        token_times: List of per-token generation times
    """
    # Input/output sizes
    prompt_length: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0
    
    # Timing metrics (seconds)
    prefill_time: float = 0.0
    decode_time: float = 0.0
    total_time: float = 0.0
    
    # Derived metrics
    time_to_first_token: float = 0.0  # TTFT
    time_per_output_token: float = 0.0  # TPOT
    
    # Throughput
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    
    # Memory
    peak_memory_mb: float = 0.0
    cache_memory_mb: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    model_name: str = ""
    batch_size: int = 1
    
    # Detailed tracking
    token_times: List[float] = field(default_factory=list)
    
    def compute_derived_metrics(self):
        """Compute derived metrics from raw measurements."""
        # TTFT = prefill time
        self.time_to_first_token = self.prefill_time
        
        # TPOT = average decode time per token
        if self.generated_tokens > 0:
            self.time_per_output_token = self.decode_time / self.generated_tokens
        
        # Total tokens
        self.total_tokens = self.prompt_length + self.generated_tokens
        
        # Throughput
        if self.total_time > 0:
            self.tokens_per_second = self.total_tokens / self.total_time
            self.requests_per_second = self.batch_size / self.total_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'prompt_length': self.prompt_length,
            'generated_tokens': self.generated_tokens,
            'total_tokens': self.total_tokens,
            'prefill_time': round(self.prefill_time, 4),
            'decode_time': round(self.decode_time, 4),
            'total_time': round(self.total_time, 4),
            'time_to_first_token_ms': round(self.time_to_first_token * 1000, 2),
            'time_per_output_token_ms': round(self.time_per_output_token * 1000, 2),
            'tokens_per_second': round(self.tokens_per_second, 2),
            'requests_per_second': round(self.requests_per_second, 2),
            'peak_memory_mb': round(self.peak_memory_mb, 2),
            'cache_memory_mb': round(self.cache_memory_mb, 2),
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'batch_size': self.batch_size
        }
    
    def print_summary(self):
        """Print formatted metrics summary."""
        print("\n" + "=" * 70)
        print("Generation Metrics")
        print("=" * 70)
        
        print(f"\nTokens:")
        print(f"  Prompt length:     {self.prompt_length:,}")
        print(f"  Generated tokens:  {self.generated_tokens:,}")
        print(f"  Total tokens:      {self.total_tokens:,}")
        
        print(f"\nLatency:")
        print(f"  TTFT (prefill):    {self.time_to_first_token*1000:.2f} ms")
        print(f"  TPOT (decode avg): {self.time_per_output_token*1000:.2f} ms")
        print(f"  Total time:        {self.total_time:.3f} s")
        
        print(f"\nThroughput:")
        print(f"  Tokens/sec:        {self.tokens_per_second:.2f}")
        print(f"  Requests/sec:      {self.requests_per_second:.2f}")
        
        if self.peak_memory_mb > 0:
            print(f"\nMemory:")
            print(f"  Peak memory:       {self.peak_memory_mb:.2f} MB")
            if self.cache_memory_mb > 0:
                print(f"  Cache memory:      {self.cache_memory_mb:.2f} MB")
        
        if self.model_name:
            print(f"\nMetadata:")
            print(f"  Model:             {self.model_name}")
            print(f"  Batch size:        {self.batch_size}")
        
        print("=" * 70 + "\n")


class MetricsTracker:
    """
    Tracks metrics during generation.
    
    Features:
    - Start/stop timing
    - Token-level timing
    - Memory tracking
    - Automatic metric computation
    
    Example:
        >>> tracker = MetricsTracker(model_name="gpt-small")
        >>> tracker.start_prefill()
        >>> # ... process prompt ...
        >>> tracker.end_prefill(prompt_length=10)
        >>> 
        >>> tracker.start_decode()
        >>> for token in generate():
        >>>     tracker.record_token()
        >>> tracker.end_decode()
        >>> 
        >>> metrics = tracker.get_metrics()
        >>> metrics.print_summary()
    """
    
    def __init__(
        self,
        model_name: str = "",
        batch_size: int = 1,
        track_memory: bool = True
    ):
        """
        Initialize metrics tracker.
        
        Args:
            model_name: Model identifier
            batch_size: Number of sequences in batch
            track_memory: Whether to track memory usage
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.track_memory = track_memory
        
        # Timing
        self.start_time: Optional[float] = None
        self.prefill_start: Optional[float] = None
        self.prefill_end: Optional[float] = None
        self.decode_start: Optional[float] = None
        self.decode_end: Optional[float] = None
        
        # Counters
        self.prompt_length: int = 0
        self.tokens_generated: int = 0
        
        # Token-level timing
        self.token_times: List[float] = []
        self.last_token_time: Optional[float] = None
        
        # Memory
        self.initial_memory: float = 0.0
        self.peak_memory: float = 0.0
        
        logger.debug(f"MetricsTracker initialized: {model_name}, batch={batch_size}")
    
    def start_prefill(self):
        """Start prefill phase timing."""
        self.start_time = time.time()
        self.prefill_start = time.time()
        
        if self.track_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        
        logger.debug("Prefill phase started")
    
    def end_prefill(self, prompt_length: int):
        """
        End prefill phase.
        
        Args:
            prompt_length: Number of tokens in prompt
        """
        self.prefill_end = time.time()
        self.prompt_length = prompt_length
        
        logger.debug(f"Prefill phase ended: {prompt_length} tokens")
    
    def start_decode(self):
        """Start decode phase timing."""
        self.decode_start = time.time()
        self.last_token_time = time.time()
        
        logger.debug("Decode phase started")
    
    def record_token(self):
        """Record generation of one token."""
        current_time = time.time()
        
        if self.last_token_time is not None:
            token_time = current_time - self.last_token_time
            self.token_times.append(token_time)
        
        self.last_token_time = current_time
        self.tokens_generated += 1
        
        # Track memory
        if self.track_memory and torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            self.peak_memory = max(self.peak_memory, current_memory)
    
    def end_decode(self):
        """End decode phase."""
        self.decode_end = time.time()
        
        logger.debug(f"Decode phase ended: {self.tokens_generated} tokens")
    
    def get_metrics(self) -> GenerationMetrics:
        """
        Get computed metrics.
        
        Returns:
            GenerationMetrics with all measurements
        """
        metrics = GenerationMetrics(
            model_name=self.model_name,
            batch_size=self.batch_size,
            prompt_length=self.prompt_length,
            generated_tokens=self.tokens_generated,
            token_times=self.token_times.copy()
        )
        
        # Compute timing
        if self.prefill_start and self.prefill_end:
            metrics.prefill_time = self.prefill_end - self.prefill_start
        
        if self.decode_start and self.decode_end:
            metrics.decode_time = self.decode_end - self.decode_start
        
        if self.start_time and self.decode_end:
            metrics.total_time = self.decode_end - self.start_time
        
        # Memory
        if self.track_memory and torch.cuda.is_available():
            metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        # Compute derived metrics
        metrics.compute_derived_metrics()
        
        return metrics


def benchmark_generation(
    generate_fn,
    num_runs: int = 5,
    warmup_runs: int = 1
) -> Dict[str, Any]:
    """
    Benchmark generation function.
    
    Runs generation multiple times and computes statistics.
    
    Args:
        generate_fn: Function that generates text and returns metrics
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs (not counted)
    
    Returns:
        Dictionary with aggregated statistics
    
    Example:
        >>> def gen():
        ...     # Generate text
        ...     return metrics
        >>> stats = benchmark_generation(gen, num_runs=10)
        >>> print(f"Avg TTFT: {stats['ttft_avg_ms']:.2f} ms")
    """
    logger.info(f"Running benchmark: {warmup_runs} warmup + {num_runs} measured runs")
    
    # Warmup
    for i in range(warmup_runs):
        logger.debug(f"Warmup run {i+1}/{warmup_runs}")
        generate_fn()
    
    # Benchmark runs
    all_metrics: List[GenerationMetrics] = []
    
    for i in range(num_runs):
        logger.debug(f"Benchmark run {i+1}/{num_runs}")
        metrics = generate_fn()
        all_metrics.append(metrics)
    
    # Aggregate statistics
    ttfts = [m.time_to_first_token for m in all_metrics]
    tpots = [m.time_per_output_token for m in all_metrics]
    throughputs = [m.tokens_per_second for m in all_metrics]
    total_times = [m.total_time for m in all_metrics]
    
    stats = {
        'num_runs': num_runs,
        'ttft_avg_ms': sum(ttfts) / len(ttfts) * 1000,
        'ttft_min_ms': min(ttfts) * 1000,
        'ttft_max_ms': max(ttfts) * 1000,
        'tpot_avg_ms': sum(tpots) / len(tpots) * 1000,
        'tpot_min_ms': min(tpots) * 1000,
        'tpot_max_ms': max(tpots) * 1000,
        'throughput_avg': sum(throughputs) / len(throughputs),
        'throughput_min': min(throughputs),
        'throughput_max': max(throughputs),
        'total_time_avg': sum(total_times) / len(total_times),
        'prompt_length': all_metrics[0].prompt_length,
        'generated_tokens': all_metrics[0].generated_tokens
    }
    
    logger.info(
        f"Benchmark complete: TTFT={stats['ttft_avg_ms']:.2f}ms, "
        f"TPOT={stats['tpot_avg_ms']:.2f}ms, "
        f"Throughput={stats['throughput_avg']:.2f} tok/s"
    )
    
    return stats


if __name__ == "__main__":
    # Test metrics tracking
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Generation Metrics Test")
    print("=" * 80)
    
    # Test 1: Basic metrics tracking
    print("\n1. Testing metrics tracker...")
    
    tracker = MetricsTracker(model_name="test-model", batch_size=1)
    
    # Simulate prefill
    tracker.start_prefill()
    time.sleep(0.01)  # Simulate prefill work
    tracker.end_prefill(prompt_length=10)
    
    # Simulate decode
    tracker.start_decode()
    for i in range(20):
        time.sleep(0.002)  # Simulate token generation
        tracker.record_token()
    tracker.end_decode()
    
    # Get metrics
    metrics = tracker.get_metrics()
    
    print(f"\n   Measured metrics:")
    print(f"     Prompt length: {metrics.prompt_length}")
    print(f"     Generated tokens: {metrics.generated_tokens}")
    print(f"     TTFT: {metrics.time_to_first_token*1000:.2f} ms")
    print(f"     TPOT: {metrics.time_per_output_token*1000:.2f} ms")
    print(f"     Throughput: {metrics.tokens_per_second:.2f} tokens/sec")
    
    # Test 2: Print formatted summary
    print("\n2. Testing formatted output...")
    
    metrics.print_summary()
    
    # Test 3: Metrics to dict
    print("\n3. Testing serialization...")
    
    metrics_dict = metrics.to_dict()
    print(f"   Dictionary keys: {list(metrics_dict.keys())[:5]}...")
    print(f"   TTFT (ms): {metrics_dict['time_to_first_token_ms']}")
    print(f"   TPOT (ms): {metrics_dict['time_per_output_token_ms']}")
    
    # Test 4: Benchmark function
    print("\n4. Testing benchmark function...")
    
    def mock_generate() -> GenerationMetrics:
        """Mock generation function."""
        tracker = MetricsTracker(model_name="mock")
        tracker.start_prefill()
        time.sleep(0.005)
        tracker.end_prefill(prompt_length=5)
        tracker.start_decode()
        for _ in range(10):
            time.sleep(0.001)
            tracker.record_token()
        tracker.end_decode()
        return tracker.get_metrics()
    
    stats = benchmark_generation(mock_generate, num_runs=3, warmup_runs=1)
    
    print(f"\n   Benchmark results ({stats['num_runs']} runs):")
    print(f"     TTFT: {stats['ttft_avg_ms']:.2f} ms (±{stats['ttft_max_ms']-stats['ttft_min_ms']:.2f})")
    print(f"     TPOT: {stats['tpot_avg_ms']:.2f} ms (±{stats['tpot_max_ms']-stats['tpot_min_ms']:.2f})")
    print(f"     Throughput: {stats['throughput_avg']:.2f} tokens/sec")
    
    # Test 5: Token-level timing
    print("\n5. Testing token-level timing...")
    
    if metrics.token_times:
        print(f"   Recorded {len(metrics.token_times)} token times")
        print(f"   First 5 tokens: {[f'{t*1000:.2f}ms' for t in metrics.token_times[:5]]}")
        print(f"   Average: {sum(metrics.token_times)/len(metrics.token_times)*1000:.2f} ms")
        print(f"   Min: {min(metrics.token_times)*1000:.2f} ms")
        print(f"   Max: {max(metrics.token_times)*1000:.2f} ms")
    
    print("\n" + "=" * 80)
    print("✅ Metrics tests completed!")
    print("=" * 80)
