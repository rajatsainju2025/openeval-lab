"""
Unified Batch Operations Module for OpenEval Lab

Consolidates batch processing logic across the codebase into a single,
optimized implementation with adaptive batching, rate limiting, and
performance monitoring.

Features:
- Dynamic batch size calculation
- Rate-aware batching
- Batch processing with callbacks
- Memory-aware batching
- Performance metrics
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Callable, TypeVar, Generic, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchMetrics:
    """Metrics for batch processing performance."""

    total_items_processed: int = 0
    total_batches: int = 0
    average_batch_size: float = 0.0
    total_processing_time: float = 0.0
    min_batch_size: int = 0
    max_batch_size: int = 0
    failed_items: int = 0
    success_items: int = 0

    @property
    def average_time_per_item_ms(self) -> float:
        """Average processing time per item in milliseconds."""
        if self.total_items_processed == 0:
            return 0.0
        return (self.total_processing_time / self.total_items_processed) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items_processed": self.total_items_processed,
            "total_batches": self.total_batches,
            "average_batch_size": self.average_batch_size,
            "total_processing_time_s": self.total_processing_time,
            "average_time_per_item_ms": self.average_time_per_item_ms,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "failed_items": self.failed_items,
            "success_items": self.success_items,
        }


class AdaptiveBatcher(Generic[T, R]):
    """Adaptive batch processor with dynamic sizing.

    Automatically adjusts batch size based on performance and system resources.
    """

    def __init__(
        self,
        process_batch: Callable[[List[T]], List[R]],
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 512,
        max_memory_mb: Optional[int] = None,
        target_time_per_batch_ms: float = 100.0,
    ):
        """Initialize adaptive batcher.

        Args:
            process_batch: Function to process a batch
            initial_batch_size: Initial batch size
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            max_memory_mb: Maximum memory for batch (None for unlimited)
            target_time_per_batch_ms: Target time for batch processing
        """
        self.process_batch = process_batch
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_memory_mb = max_memory_mb
        self.target_time_per_batch_ms = target_time_per_batch_ms
        self.metrics = BatchMetrics()
        self.lock = threading.RLock()

    def _estimate_memory_usage(self, items: List[T]) -> int:
        """Estimate memory usage of items in MB."""
        if not items:
            return 0
        try:
            import sys

            total_size = sum(sys.getsizeof(item) for item in items)
            return total_size // (1024 * 1024)
        except Exception:
            return 0

    def _adjust_batch_size(self, last_batch_time_ms: float) -> None:
        """Adjust batch size based on performance."""
        with self.lock:
            # If processing is too fast, increase batch size
            if last_batch_time_ms > 0 and last_batch_time_ms < self.target_time_per_batch_ms * 0.5:
                self.batch_size = min(int(self.batch_size * 1.2), self.max_batch_size)

            # If processing is too slow, decrease batch size
            elif last_batch_time_ms > self.target_time_per_batch_ms * 1.5:
                self.batch_size = max(int(self.batch_size * 0.8), self.min_batch_size)

    def process(self, items: List[T]) -> List[R]:
        """Process items in adaptive batches.

        Args:
            items: Items to process

        Returns:
            List of results
        """
        results = []
        current_batch_size = self.batch_size

        for i in range(0, len(items), current_batch_size):
            batch = items[i : i + current_batch_size]

            # Check memory constraints
            if self.max_memory_mb:
                memory_used = self._estimate_memory_usage(batch)
                while memory_used > self.max_memory_mb and len(batch) > self.min_batch_size:
                    current_batch_size = max(int(current_batch_size * 0.8), self.min_batch_size)
                    batch = items[i : i + current_batch_size]
                    memory_used = self._estimate_memory_usage(batch)

            # Process batch
            start_time = time.time()
            try:
                batch_results = self.process_batch(batch)
                results.extend(batch_results)
                self.metrics.success_items += len(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                self.metrics.failed_items += len(batch)

            # Update metrics
            batch_time_ms = (time.time() - start_time) * 1000
            with self.lock:
                self.metrics.total_items_processed += len(batch)
                self.metrics.total_batches += 1
                self.metrics.total_processing_time += batch_time_ms / 1000

                if self.metrics.total_batches == 1:
                    self.metrics.min_batch_size = len(batch)
                    self.metrics.max_batch_size = len(batch)
                else:
                    self.metrics.min_batch_size = min(self.metrics.min_batch_size, len(batch))
                    self.metrics.max_batch_size = max(self.metrics.max_batch_size, len(batch))

                self.metrics.average_batch_size = (
                    self.metrics.total_items_processed / self.metrics.total_batches
                )

            # Adjust batch size
            self._adjust_batch_size(batch_time_ms)

        return results

    def get_metrics(self) -> BatchMetrics:
        """Get processing metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self.metrics = BatchMetrics()


class RateLimitedBatcher(Generic[T, R]):
    """Batch processor with rate limiting.

    Respects rate limits and implements backoff strategies.
    """

    def __init__(
        self,
        process_batch: Callable[[List[T]], List[R]],
        batch_size: int = 32,
        requests_per_second: float = 10.0,
        backoff_factor: float = 2.0,
        max_retries: int = 3,
    ):
        """Initialize rate-limited batcher.

        Args:
            process_batch: Function to process a batch
            batch_size: Batch size
            requests_per_second: Rate limit in requests/second
            backoff_factor: Backoff multiplier for retries
            max_retries: Maximum retry attempts
        """
        self.process_batch = process_batch
        self.batch_size = batch_size
        self.min_interval = 1.0 / requests_per_second
        self.backoff_factor = backoff_factor
        self.max_retries = max_retries
        self.metrics = BatchMetrics()
        self.last_request_time = 0.0
        self.lock = threading.RLock()

    def process(self, items: List[T]) -> List[R]:
        """Process items with rate limiting.

        Args:
            items: Items to process

        Returns:
            List of results
        """
        results = []
        backoff_time = self.min_interval

        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]

            for attempt in range(self.max_retries):
                try:
                    # Rate limiting
                    with self.lock:
                        elapsed = time.time() - self.last_request_time
                        if elapsed < self.min_interval:
                            time.sleep(self.min_interval - elapsed)

                        self.last_request_time = time.time()

                    # Process batch
                    start_time = time.time()
                    batch_results = self.process_batch(batch)
                    results.extend(batch_results)

                    # Update metrics
                    with self.lock:
                        self.metrics.total_items_processed += len(batch_results)
                        self.metrics.total_batches += 1
                        self.metrics.success_items += len(batch_results)
                        self.metrics.total_processing_time += time.time() - start_time

                    backoff_time = self.min_interval  # Reset backoff
                    break

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            f"Batch processing failed (attempt {attempt + 1}), "
                            f"retrying in {backoff_time:.2f}s: {e}"
                        )
                        time.sleep(backoff_time)
                        backoff_time *= self.backoff_factor
                    else:
                        logger.error(
                            f"Batch processing failed after {self.max_retries} attempts: {e}"
                        )
                        with self.lock:
                            self.metrics.failed_items += len(batch)

        return results

    def get_metrics(self) -> BatchMetrics:
        """Get processing metrics."""
        return self.metrics


def batch_items(items: List[T], batch_size: int) -> List[List[T]]:
    """Split items into batches.

    Args:
        items: Items to batch
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def calculate_optimal_batch_size(
    total_items: int,
    avg_item_size_bytes: int,
    max_memory_mb: int = 256,
    target_batch_time_ms: float = 100.0,
) -> int:
    """Calculate optimal batch size for processing.

    Args:
        total_items: Total number of items
        avg_item_size_bytes: Average size of each item
        max_memory_mb: Maximum memory available
        target_batch_time_ms: Target time per batch

    Returns:
        Recommended batch size
    """
    max_items_by_memory = (max_memory_mb * 1024 * 1024) // max(avg_item_size_bytes, 1)
    max_items_by_time = max(1, int((target_batch_time_ms / 1000) * 100))  # Rough estimate
    optimal_size = min(max_items_by_memory, max_items_by_time)
    return max(1, min(optimal_size, total_items))
