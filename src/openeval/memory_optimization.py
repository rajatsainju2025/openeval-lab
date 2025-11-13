"""
Memory Optimization Utilities for Large-Scale Evaluations

Provides utilities for efficient memory management during large evaluations,
including streaming processing, memory pooling, and garbage collection hints.
"""

from __future__ import annotations

import gc
import sys
from typing import Iterator, List, TypeVar, Generic, Callable, Any, Set
import logging
import weakref

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MemoryPool(Generic[T]):
    """Object pool for reducing allocation overhead."""

    def __init__(self, factory: Callable[[], T], initial_size: int = 100):
        """Initialize memory pool.

        Args:
            factory: Function to create new objects
            initial_size: Initial pool size
        """
        self.factory = factory
        self.available: List[T] = []
        self.in_use: Set[Any] = weakref.WeakSet()  # type: ignore

        # Pre-allocate
        for _ in range(initial_size):
            self.available.append(factory())

    def acquire(self) -> T:
        """Acquire an object from the pool."""
        if self.available:
            obj = self.available.pop()
        else:
            obj = self.factory()

        self.in_use.add(obj)
        return obj

    def release(self, obj: T) -> None:
        """Release object back to pool."""
        self.available.append(obj)

    def get_status(self) -> dict:
        """Get pool status."""
        return {
            "available": len(self.available),
            "in_use": len(self.in_use),
            "total": len(self.available) + len(self.in_use),
        }


class StreamingProcessor(Generic[T]):
    """Process items in a memory-efficient streaming fashion."""

    def __init__(self, process_fn: Callable[[T], Any], batch_size: int = 1000):
        """Initialize streaming processor.

        Args:
            process_fn: Function to process each item
            batch_size: Batch size for garbage collection
        """
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.processed_count = 0

    def process_stream(self, items: Iterator[T], gc_interval: int = 1000) -> Iterator[Any]:
        """Process items from a stream.

        Args:
            items: Iterator of items to process
            gc_interval: Call gc.collect() every N items

        Yields:
            Processed items
        """
        for i, item in enumerate(items):
            result = self.process_fn(item)
            yield result

            self.processed_count += 1

            # Periodically collect garbage
            if i % gc_interval == 0 and i > 0:
                gc.collect()

    def reset(self) -> None:
        """Reset processing count."""
        self.processed_count = 0


class MemoryMonitor:
    """Monitor memory usage during processing."""

    def __init__(self, threshold_mb: int = 100, sample_interval: int = 1000):
        """Initialize memory monitor.

        Args:
            threshold_mb: Warning threshold in MB
            sample_interval: Sample memory every N items
        """
        self.threshold_mb = threshold_mb
        self.sample_interval = sample_interval
        self.peak_memory_mb = 0
        self.samples = []

    def get_current_memory_mb(self) -> int:
        """Get current memory usage in MB."""
        return int(sys.getsizeof(gc.get_objects()) / (1024 * 1024))

    def check_memory(self, item_count: int) -> None:
        """Check memory usage and log if threshold exceeded.

        Args:
            item_count: Current item count
        """
        if item_count % self.sample_interval != 0:
            return

        current_mb = self.get_current_memory_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, current_mb)
        self.samples.append(current_mb)

        if current_mb > self.threshold_mb:
            logger.warning(f"High memory usage: {current_mb}MB after {item_count} items")

            # Try to reduce memory
            gc.collect()

    def get_report(self) -> dict:
        """Get memory monitoring report."""
        return {
            "peak_memory_mb": self.peak_memory_mb,
            "samples": len(self.samples),
            "average_memory_mb": (sum(self.samples) / len(self.samples) if self.samples else 0),
        }


def enable_gc_optimization() -> None:
    """Enable garbage collection optimization for batch processing."""
    gc.set_debug(0)  # Disable debug mode
    # Increase collection thresholds for better performance
    gc.set_threshold(10000, 5, 5)


def disable_gc_during_critical(fn: Callable) -> Callable:
    """Decorator to disable GC during critical section."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        gc_was_enabled = gc.isenabled()
        try:
            gc.disable()
            return fn(*args, **kwargs)
        finally:
            if gc_was_enabled:
                gc.enable()

    return wrapper


def trim_memory() -> int:
    """Trim memory by forcing garbage collection.

    Returns:
        Memory freed in bytes (approximate)
    """
    before = sys.getsizeof(gc.get_objects())
    gc.collect()
    after = sys.getsizeof(gc.get_objects())
    return before - after
