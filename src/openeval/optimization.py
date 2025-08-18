"""Advanced optimization and performance monitoring for OpenEval Lab."""

import time
import threading
import statistics
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import resource
from functools import wraps

from .logging import get_logger
from .core import Adapter, Dataset, Example

T = TypeVar('T')


@dataclass
class PerformanceMetric:
    """A single performance measurement."""
    
    name: str
    value: float
    unit: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """System resource usage snapshot."""
    
    memory_used_mb: float
    thread_count: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class OptimizationSuggestion:
    """Performance optimization suggestion."""
    
    category: str  # memory, cpu, io, network, concurrency
    severity: str  # low, medium, high, critical
    title: str
    description: str
    recommendation: str
    impact: str  # estimated improvement
    effort: str  # implementation effort (low, medium, high)


class PerformanceMonitor:
    """Real-time performance monitoring and optimization."""
    
    def __init__(self, sample_interval: float = 1.0):
        """Initialize performance monitor."""
        self.logger = get_logger()
        self.sample_interval = sample_interval
        self.monitoring = False
        self.metrics: List[PerformanceMetric] = []
        self.snapshots: List[SystemSnapshot] = []
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance thresholds
        self.thresholds = {
            "response_time_warning": 5.0,    # seconds
            "response_time_critical": 10.0,
            "throughput_warning": 0.5,       # samples/sec
            "error_rate_warning": 0.05,      # 5%
            "error_rate_critical": 0.10,     # 10%
        }
    
    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Started performance monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Stopped performance monitoring")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # Create snapshot with available info
                snapshot = SystemSnapshot(
                    memory_used_mb=self._get_memory_usage(),
                    thread_count=threading.active_count()
                )
                
                self.snapshots.append(snapshot)
                
                # Keep only recent snapshots (last hour)
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                self.snapshots = [
                    s for s in self.snapshots 
                    if datetime.fromisoformat(s.timestamp) > cutoff_time
                ]
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                self.logger.warning(f"Performance monitoring error: {e}")
                time.sleep(self.sample_interval)
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in MB using resource module."""
        try:
            # Use resource module for basic memory info
            mem_info = resource.getrusage(resource.RUSAGE_SELF)
            # Convert to MB (getrusage returns in KB on Linux, bytes on macOS)
            return mem_info.ru_maxrss / 1024  # Assume KB for simplicity
        except Exception:
            return 0.0
    
    @contextmanager
    def measure_operation(self, operation_name: str):
        """Context manager to measure operation performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Force garbage collection before measurement
        gc.collect()
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            memory_delta = self._get_memory_usage() - start_memory
            
            # Record metrics
            self.record_metric("operation_duration", duration, "seconds", {"operation": operation_name})
            self.record_metric("operation_memory_delta", memory_delta, "mb", {"operation": operation_name})
            
            self.logger.debug(f"Operation '{operation_name}' took {duration:.2f}s")
    
    def record_metric(self, name: str, value: float, unit: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Keep only recent metrics (last hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        self.metrics = [
            m for m in self.metrics 
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        if not self.snapshots:
            return {"error": "No performance data available"}
        
        recent_snapshots = self.snapshots[-10:]  # Last 10 measurements
        
        summary = {
            "current": {
                "memory_used_mb": recent_snapshots[-1].memory_used_mb,
                "thread_count": recent_snapshots[-1].thread_count
            },
            "averages": {
                "memory_used_mb": statistics.mean(s.memory_used_mb for s in recent_snapshots)
            },
            "peaks": {
                "max_memory_used_mb": max(s.memory_used_mb for s in recent_snapshots)
            }
        }
        
        return summary


class AdaptiveConcurrencyController:
    """Dynamically adjust concurrency based on system performance."""
    
    def __init__(self, initial_concurrency: int = 4, min_concurrency: int = 1, max_concurrency: int = 16):
        """Initialize adaptive concurrency controller."""
        self.logger = get_logger()
        self.current_concurrency = initial_concurrency
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        
        # Performance tracking
        self.throughput_history: List[float] = []
        self.error_rate_history: List[float] = []
        self.response_time_history: List[float] = []
        
        # Adjustment parameters
        self.adjustment_interval = 10  # measurements between adjustments
        self.measurement_count = 0
        self.last_adjustment_time = time.time()
    
    def record_performance(self, throughput: float, error_rate: float, response_time: float) -> None:
        """Record performance metrics for concurrency adjustment."""
        self.throughput_history.append(throughput)
        self.error_rate_history.append(error_rate)
        self.response_time_history.append(response_time)
        
        # Keep only recent history
        max_history = 20
        if len(self.throughput_history) > max_history:
            self.throughput_history = self.throughput_history[-max_history:]
            self.error_rate_history = self.error_rate_history[-max_history:]
            self.response_time_history = self.response_time_history[-max_history:]
    
    def get_current_concurrency(self) -> int:
        """Get current optimal concurrency level."""
        return self.current_concurrency


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 32
    max_concurrent: int = 4
    timeout_per_batch: Optional[float] = None
    retry_failed: bool = True


class BatchProcessor:
    """Efficient batch processing for model evaluation."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize batch processor."""
        self.config = config or BatchConfig()
    
    def create_batches(self, items: List[T], batch_size: Optional[int] = None) -> List[List[T]]:
        """Split items into batches."""
        batch_size = batch_size or self.config.batch_size
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    async def process_batch_async(
        self, 
        batch: List[T], 
        processor_func: Callable[[T], Any],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """Process a batch asynchronously."""
        timeout = timeout or self.config.timeout_per_batch
        
        async def process_item(item: T) -> Any:
            # Run in thread pool for CPU-bound operations
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, processor_func, item)
        
        # Process batch with timeout
        try:
            tasks = [process_item(item) for item in batch]
            if timeout:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks), 
                    timeout=timeout
                )
            else:
                results = await asyncio.gather(*tasks)
            return results
        except asyncio.TimeoutError:
            if self.config.retry_failed:
                # Retry with smaller batch or individual items
                return await self._retry_batch(batch, processor_func)
            else:
                raise
    
    async def _retry_batch(
        self, 
        batch: List[T], 
        processor_func: Callable[[T], Any]
    ) -> List[Any]:
        """Retry failed batch with fallback strategy."""
        # Try processing items individually
        results = []
        for item in batch:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, processor_func, item
                )
                results.append(result)
            except Exception as e:
                # Log error and use None as placeholder
                print(f"Warning: Failed to process item: {e}")
                results.append(None)
        return results
    
    def process_batches(
        self, 
        items: List[T], 
        processor_func: Callable[[T], Any]
    ) -> List[Any]:
        """Process all items in batches."""
        batches = self.create_batches(items)
        results = []
        
        async def process_all_batches():
            semaphore = asyncio.Semaphore(self.config.max_concurrent)
            
            async def process_single_batch(batch):
                async with semaphore:
                    return await self.process_batch_async(batch, processor_func)
            
            batch_tasks = [process_single_batch(batch) for batch in batches]
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Flatten results
            for batch_result in batch_results:
                results.extend(batch_result)
        
        # Run the async processing
        try:
            asyncio.run(process_all_batches())
        except Exception as e:
            # Fallback to synchronous processing
            print(f"Async processing failed, falling back to sync: {e}")
            for batch in batches:
                batch_results = [processor_func(item) for item in batch]
                results.extend(batch_results)
        
        return results


class CacheManager:
    """Advanced caching with TTL and size limits."""
    
    def __init__(self, max_size: int = 10000, default_ttl: float = 3600):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cached items
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._cache:
            return True
        
        entry = self._cache[key]
        if entry.get("ttl") is None:
            return False
        
        return time.time() > entry["timestamp"] + entry["ttl"]
    
    def _evict_lru(self):
        """Evict least recently used items if cache is full."""
        if len(self._cache) >= self.max_size:
            # Find LRU item
            lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            del self._cache[lru_key]
            del self._access_times[lru_key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if self._is_expired(key):
            self.delete(key)
            return None
        
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]["value"]
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set item in cache."""
        self._evict_lru()
        
        self._cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl or self.default_ttl
        }
        self._access_times[key] = time.time()
    
    def delete(self, key: str) -> None:
        """Delete item from cache."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        expired_entries = sum(1 for key in self._cache if self._is_expired(key))
        
        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "valid_entries": total_entries - expired_entries,
            "cache_size": total_entries,
            "max_size": self.max_size,
            "hit_rate": 0.0,  # Would need request tracking for this
        }


def memoize_with_ttl(ttl: float = 3600):
    """Decorator for memoizing function results with TTL."""
    cache = CacheManager(default_ttl=ttl)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str(hash((args, tuple(sorted(kwargs.items())))))
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        # Add cache management methods
        # Store cache methods
        wrapper._cache = cache  # type: ignore
        
        return wrapper
    
    return decorator


class ProgressTracker:
    """Track progress for long-running evaluations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """Initialize progress tracker."""
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        current_time = time.time()
        
        # Update every second or at completion
        if current_time - self.last_update >= 1.0 or self.current >= self.total:
            self._print_progress()
            self.last_update = current_time
    
    def _print_progress(self) -> None:
        """Print progress bar."""
        if self.total == 0:
            return
        
        progress = self.current / self.total
        elapsed = time.time() - self.start_time
        
        # Estimate time remaining
        if progress > 0:
            eta = elapsed / progress - elapsed
            eta_str = f"ETA: {eta:.1f}s" if eta > 0 else "ETA: 0s"
        else:
            eta_str = "ETA: --"
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Print progress
        print(f"\r{self.description}: {bar} {progress:.1%} ({self.current}/{self.total}) {eta_str}", end="")
        
        if self.current >= self.total:
            print()  # New line when complete


class StreamingDataset:
    """Memory-efficient streaming dataset wrapper."""
    
    def __init__(self, dataset: Dataset, chunk_size: int = 1000):
        """
        Initialize streaming dataset.
        
        Args:
            dataset: Base dataset to stream from
            chunk_size: Number of examples to load at once
        """
        self.dataset = dataset
        self.chunk_size = chunk_size
        self._current_chunk = []
        self._chunk_index = 0
        self._total_processed = 0
    
    def __iter__(self):
        """Iterate over dataset in chunks."""
        chunk = []
        for example in self.dataset:
            chunk.append(example)
            self._total_processed += 1
            
            if len(chunk) >= self.chunk_size:
                yield from chunk
                chunk = []
        
        # Yield remaining examples
        if chunk:
            yield from chunk
    
    def get_stats(self) -> Dict[str, int]:
        """Get streaming statistics."""
        return {
            "total_processed": self._total_processed,
            "chunk_size": self.chunk_size,
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_performance(operation_name: str):
    """Decorator for monitoring function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with performance_monitor.measure_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def profile_evaluation(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to profile evaluation performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = _get_memory_usage()
            
            # Log performance stats
            elapsed = end_time - start_time
            memory_delta = end_memory - start_memory
            
            print(f"\n--- Performance Profile for {func.__name__} ---")
            print(f"Execution time: {elapsed:.2f}s")
            print(f"Memory usage: {memory_delta:.2f}MB")
            print(f"Peak memory: {end_memory:.2f}MB")
            
            return result
            
        except Exception as e:
            print(f"\n--- Error in {func.__name__} ---")
            print(f"Error: {e}")
            raise
    
    return wrapper


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        # Use resource module for basic memory info
        mem_info = resource.getrusage(resource.RUSAGE_SELF)
        # Convert to MB (getrusage returns in KB on Linux, bytes on macOS)
        return mem_info.ru_maxrss / 1024  # Assume KB for simplicity
    except Exception:
        return 0.0  # Fallback if not available
