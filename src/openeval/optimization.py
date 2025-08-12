"""Performance optimization utilities for OpenEval."""

import asyncio
import concurrent.futures
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass
from functools import wraps

from openeval.core import Adapter, Dataset, Example

T = TypeVar('T')


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
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.stats
        
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
            
            # Print performance stats
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
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0  # Fallback if psutil not available
