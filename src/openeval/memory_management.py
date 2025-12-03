"""
Memory Management Utilities for OpenEval Lab.

This module provides memory-efficient utilities for handling large datasets
and optimizing memory usage during evaluation runs.

Key Features:
- Memory usage tracking and monitoring
- Automatic garbage collection triggers
- Memory-efficient iterators for large datasets
- Context managers for memory-intensive operations
- Memory profiling utilities

Design Goals:
- Enable evaluation of arbitrarily large datasets
- Provide early warning for memory pressure
- Support streaming processing patterns
"""

from __future__ import annotations

import gc
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterator, Optional, TypeVar

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None  # type: ignore


T = TypeVar("T")


@dataclass
class MemorySnapshot:
    """Snapshot of current memory usage."""

    process_mb: float
    system_percent: float
    available_mb: float

    @classmethod
    def capture(cls) -> "MemorySnapshot":
        """Capture current memory state."""
        if not HAS_PSUTIL or psutil is None:
            return cls(process_mb=0.0, system_percent=0.0, available_mb=0.0)

        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()

        return cls(
            process_mb=memory_info.rss / (1024 * 1024),
            system_percent=system_memory.percent,
            available_mb=system_memory.available / (1024 * 1024),
        )

    def __str__(self) -> str:
        return (
            f"Process: {self.process_mb:.1f}MB | "
            f"System: {self.system_percent:.1f}% | "
            f"Available: {self.available_mb:.1f}MB"
        )


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB.

    Returns:
        Memory usage in megabytes, or 0.0 if psutil is not available.
    """
    if not HAS_PSUTIL or psutil is None:
        return 0.0

    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def get_available_memory_mb() -> float:
    """Get available system memory in MB.

    Returns:
        Available memory in megabytes, or 0.0 if psutil is not available.
    """
    if not HAS_PSUTIL or psutil is None:
        return 0.0

    return psutil.virtual_memory().available / (1024 * 1024)


def force_gc() -> int:
    """Force garbage collection and return objects collected.

    Returns:
        Number of objects collected.
    """
    return gc.collect()


@contextmanager
def memory_tracked_operation(
    operation_name: str = "operation",
    threshold_mb: float = 100.0,
    auto_gc: bool = True,
) -> Generator[MemorySnapshot, None, None]:
    """Context manager to track memory usage of an operation.

    Args:
        operation_name: Name of the operation for logging.
        threshold_mb: Memory increase threshold to trigger warning.
        auto_gc: Whether to run garbage collection after operation.

    Yields:
        Initial memory snapshot.

    Example:
        >>> with memory_tracked_operation("data_loading", threshold_mb=50.0) as snapshot:
        ...     data = load_large_dataset()
    """
    before = MemorySnapshot.capture()
    yield before

    if auto_gc:
        force_gc()

    after = MemorySnapshot.capture()
    delta_mb = after.process_mb - before.process_mb

    if delta_mb > threshold_mb:
        import logging

        logging.warning(
            f"[Memory] {operation_name} increased memory by {delta_mb:.1f}MB "
            f"(threshold: {threshold_mb:.1f}MB)"
        )


def chunked_iterator(
    items: Iterator[T],
    chunk_size: int = 1000,
    gc_every_n_chunks: int = 10,
) -> Generator[list[T], None, None]:
    """Memory-efficient chunked iterator with periodic garbage collection.

    Args:
        items: Iterator to chunk.
        chunk_size: Number of items per chunk.
        gc_every_n_chunks: Run GC every N chunks (0 to disable).

    Yields:
        Chunks of items as lists.

    Example:
        >>> for chunk in chunked_iterator(range(10000), chunk_size=100):
        ...     process_batch(chunk)
    """
    chunk: list[T] = []
    chunk_count = 0

    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
            chunk_count += 1

            if gc_every_n_chunks > 0 and chunk_count % gc_every_n_chunks == 0:
                gc.collect()

    if chunk:
        yield chunk


def memory_guard(
    threshold_percent: float = 90.0,
    on_threshold: Optional[Callable[[], None]] = None,
) -> bool:
    """Check if memory usage is within safe limits.

    Args:
        threshold_percent: System memory threshold (0-100).
        on_threshold: Optional callback when threshold exceeded.

    Returns:
        True if memory is within limits, False if threshold exceeded.
    """
    if not HAS_PSUTIL or psutil is None:
        return True  # Can't check, assume OK

    system_memory = psutil.virtual_memory()

    if system_memory.percent > threshold_percent:
        if on_threshold:
            on_threshold()
        return False

    return True


def estimate_object_size(obj: Any) -> int:
    """Estimate memory size of an object in bytes.

    This is a rough estimate using sys.getsizeof and doesn't account
    for nested objects by default.

    Args:
        obj: Object to measure.

    Returns:
        Estimated size in bytes.
    """
    return sys.getsizeof(obj)


def estimate_list_memory_mb(items: list[Any]) -> float:
    """Estimate memory usage of a list of items in MB.

    Args:
        items: List to estimate.

    Returns:
        Estimated size in megabytes.
    """
    if not items:
        return 0.0

    # Sample first few items for estimation
    sample_size = min(len(items), 100)
    sample = items[:sample_size]
    avg_item_size = sum(sys.getsizeof(item) for item in sample) / sample_size

    # List overhead + estimated item sizes
    total_bytes = sys.getsizeof(items) + (avg_item_size * len(items))
    return total_bytes / (1024 * 1024)


class MemoryEfficientAccumulator:
    """Memory-efficient accumulator for large result sets.

    Automatically writes to disk when memory threshold is reached.
    """

    def __init__(
        self,
        max_memory_mb: float = 100.0,
        flush_callback: Optional[Callable[[list[Any]], None]] = None,
    ):
        """Initialize accumulator.

        Args:
            max_memory_mb: Maximum memory for buffer before flush.
            flush_callback: Optional callback to handle flushed data.
        """
        self._buffer: list[Any] = []
        self._max_memory_mb = max_memory_mb
        self._flush_callback = flush_callback
        self._flush_count = 0
        self._total_items = 0

    def add(self, item: Any) -> None:
        """Add item to accumulator."""
        self._buffer.append(item)
        self._total_items += 1

        # Check if we need to flush
        if estimate_list_memory_mb(self._buffer) > self._max_memory_mb:
            self.flush()

    def flush(self) -> list[Any]:
        """Flush buffer and return contents."""
        if self._flush_callback and self._buffer:
            self._flush_callback(self._buffer)

        result = self._buffer
        self._buffer = []
        self._flush_count += 1
        gc.collect()
        return result

    def get_buffer(self) -> list[Any]:
        """Get current buffer contents without flushing."""
        return self._buffer

    @property
    def stats(self) -> dict[str, int]:
        """Get accumulator statistics."""
        return {
            "total_items": self._total_items,
            "buffer_size": len(self._buffer),
            "flush_count": self._flush_count,
        }
