"""Batch processor for optimized multi-element explanation.

Provides concurrent batch processing with configurable concurrency limits.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from .types import CodeElement, ExplainLevel, ExplanationResult

T = TypeVar("T")


class BatchStatus(str, Enum):
    """Status of a batch operation."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BatchItem(Generic[T]):
    """A single item in a batch."""

    index: int
    input_data: Any
    result: Optional[T] = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def success(self) -> bool:
        """Check if item was processed successfully."""
        return self.error is None and self.result is not None

    @property
    def duration_ms(self) -> float:
        """Get processing duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


@dataclass
class BatchResult(Generic[T]):
    """Result of a batch processing operation."""

    status: BatchStatus
    items: List[BatchItem[T]]
    total_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def success_count(self) -> int:
        """Count of successful items."""
        return sum(1 for item in self.items if item.success)

    @property
    def failure_count(self) -> int:
        """Count of failed items."""
        return sum(1 for item in self.items if not item.success)

    @property
    def successful_results(self) -> List[T]:
        """Get list of successful results."""
        return [item.result for item in self.items if item.success and item.result]

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per item in milliseconds."""
        durations = [item.duration_ms for item in self.items if item.duration_ms > 0]
        return sum(durations) / len(durations) if durations else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "total_items": len(self.items),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_time_ms": self.total_time * 1000,
            "avg_latency_ms": self.avg_latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class ConcurrencyLimiter:
    """Limits concurrent operations using a semaphore."""

    def __init__(self, max_concurrent: int = 10) -> None:
        """Initialize concurrency limiter.

        Args:
            max_concurrent: Maximum concurrent operations.
        """
        self.max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    async def acquire(self) -> None:
        """Acquire a slot."""
        await self._get_semaphore().acquire()

    def release(self) -> None:
        """Release a slot."""
        self._get_semaphore().release()

    async def __aenter__(self) -> "ConcurrencyLimiter":
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        self.release()


class BatchProcessor(Generic[T]):
    """Process items in batches with configurable concurrency.

    Supports both sync and async processing functions.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        batch_size: int = 50,
        timeout_per_item: float = 30.0,
        continue_on_error: bool = True,
    ) -> None:
        """Initialize batch processor.

        Args:
            max_concurrent: Maximum concurrent operations.
            batch_size: Maximum items per batch.
            timeout_per_item: Timeout per item in seconds.
            continue_on_error: Whether to continue processing on errors.
        """
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.timeout_per_item = timeout_per_item
        self.continue_on_error = continue_on_error
        self._limiter = ConcurrencyLimiter(max_concurrent)

    async def process_async(
        self,
        items: List[Any],
        process_func: Callable[[Any], T],
    ) -> BatchResult[T]:
        """Process items asynchronously with concurrency limit.

        Args:
            items: Items to process.
            process_func: Function to apply to each item.

        Returns:
            BatchResult with all outcomes.
        """
        start_time = time.time()
        batch_items: List[BatchItem[T]] = []

        async def process_item(index: int, item: Any) -> BatchItem[T]:
            batch_item = BatchItem[T](index=index, input_data=item)
            batch_item.start_time = time.time()

            async with self._limiter:
                try:
                    # Run in executor if sync function
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, process_func, item),
                        timeout=self.timeout_per_item,
                    )
                    batch_item.result = result
                except asyncio.TimeoutError:
                    batch_item.error = TimeoutError(
                        f"Item {index} timed out after {self.timeout_per_item}s"
                    )
                except Exception as e:
                    batch_item.error = e
                    if not self.continue_on_error:
                        raise
                finally:
                    batch_item.end_time = time.time()

            return batch_item

        # Process in chunks to respect batch_size
        for chunk_start in range(0, len(items), self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, len(items))
            chunk = items[chunk_start:chunk_end]

            tasks = [process_item(chunk_start + i, item) for i, item in enumerate(chunk)]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in chunk_results:
                if isinstance(result, BatchItem):
                    batch_items.append(result)
                elif isinstance(result, Exception):
                    # Handle gather exceptions
                    batch_items.append(
                        BatchItem(
                            index=len(batch_items),
                            input_data=None,
                            error=result,
                        )
                    )

        total_time = time.time() - start_time

        # Determine status
        if all(item.success for item in batch_items):
            status = BatchStatus.COMPLETED
        elif all(not item.success for item in batch_items):
            status = BatchStatus.FAILED
        else:
            status = BatchStatus.PARTIAL

        return BatchResult(
            status=status,
            items=batch_items,
            total_time=total_time,
        )

    def process_sync(
        self,
        items: List[Any],
        process_func: Callable[[Any], T],
    ) -> BatchResult[T]:
        """Process items synchronously (sequential).

        Args:
            items: Items to process.
            process_func: Function to apply to each item.

        Returns:
            BatchResult with all outcomes.
        """
        start_time = time.time()
        batch_items: List[BatchItem[T]] = []

        for index, item in enumerate(items):
            batch_item = BatchItem[T](index=index, input_data=item)
            batch_item.start_time = time.time()

            try:
                batch_item.result = process_func(item)
            except Exception as e:
                batch_item.error = e
                if not self.continue_on_error:
                    break
            finally:
                batch_item.end_time = time.time()

            batch_items.append(batch_item)

        total_time = time.time() - start_time

        # Determine status
        if all(item.success for item in batch_items):
            status = BatchStatus.COMPLETED
        elif all(not item.success for item in batch_items):
            status = BatchStatus.FAILED
        else:
            status = BatchStatus.PARTIAL

        return BatchResult(
            status=status,
            items=batch_items,
            total_time=total_time,
        )


class ExplainerBatchProcessor:
    """Batch processor specialized for CodeExplainer operations."""

    def __init__(
        self,
        explainer: Any,  # CodeExplainer
        max_concurrent: int = 5,
        timeout_per_item: float = 60.0,
    ) -> None:
        """Initialize explainer batch processor.

        Args:
            explainer: CodeExplainer to use.
            max_concurrent: Maximum concurrent explanations.
            timeout_per_item: Timeout per explanation in seconds.
        """
        self.explainer = explainer
        self.processor: BatchProcessor[ExplanationResult] = BatchProcessor(
            max_concurrent=max_concurrent,
            timeout_per_item=timeout_per_item,
        )

    async def explain_batch_async(
        self,
        elements: List[CodeElement],
        level: ExplainLevel = ExplainLevel.DETAILED,
        context: Optional[Dict[str, Any]] = None,
    ) -> BatchResult[ExplanationResult]:
        """Explain multiple elements asynchronously.

        Args:
            elements: Code elements to explain.
            level: Explanation detail level.
            context: Additional context.

        Returns:
            BatchResult with explanation results.
        """

        def explain_one(element: CodeElement) -> ExplanationResult:
            return self.explainer.explain(element, level, context)

        return await self.processor.process_async(elements, explain_one)

    def explain_batch_sync(
        self,
        elements: List[CodeElement],
        level: ExplainLevel = ExplainLevel.DETAILED,
        context: Optional[Dict[str, Any]] = None,
    ) -> BatchResult[ExplanationResult]:
        """Explain multiple elements synchronously.

        Args:
            elements: Code elements to explain.
            level: Explanation detail level.
            context: Additional context.

        Returns:
            BatchResult with explanation results.
        """

        def explain_one(element: CodeElement) -> ExplanationResult:
            return self.explainer.explain(element, level, context)

        return self.processor.process_sync(elements, explain_one)

    def get_successful_explanations(
        self,
        batch_result: BatchResult[ExplanationResult],
    ) -> List[ExplanationResult]:
        """Extract successful explanations from batch result.

        Args:
            batch_result: Result from batch processing.

        Returns:
            List of successful ExplanationResults.
        """
        return batch_result.successful_results


def run_batch_async(
    items: List[Any],
    process_func: Callable[[Any], T],
    max_concurrent: int = 10,
) -> BatchResult[T]:
    """Convenience function to run batch processing.

    Args:
        items: Items to process.
        process_func: Function to apply.
        max_concurrent: Maximum concurrency.

    Returns:
        BatchResult.
    """
    processor: BatchProcessor[T] = BatchProcessor(max_concurrent=max_concurrent)
    return asyncio.run(processor.process_async(items, process_func))
