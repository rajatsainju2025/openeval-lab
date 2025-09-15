"""
Async Evaluation Engine for OpenEval Lab

This module provides an asynchronous evaluation engine that replaces the thread-based
approach with asyncio for better concurrency, reduced overhead, and improved performance.

Optimizations:
- Connection pooling and reuse
- Adaptive batching based on load
- Circuit breaker pattern for fault tolerance
- Memory-efficient streaming with backpressure
- Priority-based task scheduling
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Union, AsyncIterator, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import asynccontextmanager
from collections import deque
import statistics
import heapq

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    aiofiles = None  # type: ignore
    HAS_AIOFILES = False

from .enhanced_logging import get_logger
from .cache import PredictionCache, CacheStats
from .utils import set_seed, hash_prompt

logger = get_logger(__name__)


@dataclass
class AsyncTaskConfig:
    """Configuration for async task execution."""
    max_concurrent_requests: int = 10
    request_timeout: Optional[float] = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    semaphore_limit: Optional[int] = None
    enable_progress_tracking: bool = True
    # New optimization parameters
    adaptive_batching: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 50
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    priority_levels: int = 3


@dataclass
class CircuitBreakerState:
    """State for circuit breaker pattern."""
    failures: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half-open


@dataclass
class AsyncEvaluationResult:
    """Result of an async evaluation."""
    index: int
    prediction: Any
    latency: float
    error: Optional[str] = None
    cached: bool = False
    retry_count: int = 0
    priority: int = 1


@dataclass(order=True)
class PrioritizedTask:
    """Priority queue item for task scheduling."""
    priority: int
    index: int
    coro: asyncio.Task = field(compare=False)


class ConnectionPool:
    """Connection pool for efficient resource reuse."""

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.available_connections = asyncio.Queue(maxsize=max_connections)
        self._connection_count = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> Any:
        """Acquire a connection from the pool."""
        try:
            return self.available_connections.get_nowait()
        except asyncio.QueueEmpty:
            async with self._lock:
                if self._connection_count < self.max_connections:
                    self._connection_count += 1
                    return await self._create_connection()
                else:
                    return await self.available_connections.get()

    async def release(self, connection: Any) -> None:
        """Release a connection back to the pool."""
        try:
            self.available_connections.put_nowait(connection)
        except asyncio.QueueFull:
            # Pool is full, close the connection
            await self._close_connection(connection)

    async def _create_connection(self) -> Any:
        """Create a new connection."""
        # This should be overridden by subclasses
        return object()

    async def _close_connection(self, connection: Any) -> None:
        """Close a connection."""
        # This should be overridden by subclasses
        pass


class AsyncAdapterWrapper:
    """
    Wrapper that provides async interface for any adapter.
    """

    def __init__(self, adapter: Any, thread_pool: Optional[ThreadPoolExecutor] = None):
        self.adapter = adapter
        self.thread_pool = thread_pool or ThreadPoolExecutor(max_workers=4)
        self._loop = None

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Async generate method with automatic fallback."""
        if hasattr(self.adapter, 'agenerate') and asyncio.iscoroutinefunction(self.adapter.agenerate):
            return await self.adapter.agenerate(prompt, **kwargs)

        # Fallback to sync method in thread pool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: self.adapter.generate(prompt, **kwargs)
        )

    async def agenerate_with_logprobs(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Async generate with logprobs method."""
        if hasattr(self.adapter, 'agenerate_with_logprobs') and asyncio.iscoroutinefunction(self.adapter.agenerate_with_logprobs):
            return await self.adapter.agenerate_with_logprobs(prompt, **kwargs)

        # Fallback to sync method
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: self.adapter.generate_with_logprobs(prompt, **kwargs)
        )


class AsyncEvaluationEngine:
    """
    High-performance async evaluation engine with advanced optimizations.
    """

    def __init__(self, config: Optional[AsyncTaskConfig] = None):
        self.config = config or AsyncTaskConfig()
        self.semaphore = asyncio.Semaphore(self.config.semaphore_limit or self.config.max_concurrent_requests)
        self.cache: Optional[PredictionCache] = None
        self.cache_stats = CacheStats()
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)

        # New optimization components
        self.connection_pool = ConnectionPool(self.config.max_concurrent_requests)
        self.circuit_breaker = CircuitBreakerState()
        self.latency_history: deque[float] = deque(maxlen=100)
        self.task_queue: List[PrioritizedTask] = []
        self._adaptive_batch_size = self.config.min_batch_size

    def set_cache(self, cache: PredictionCache) -> None:
        """Set the prediction cache."""
        self.cache = cache

    def _update_adaptive_batch_size(self) -> None:
        """Update batch size based on recent performance."""
        if not self.config.adaptive_batching or len(self.latency_history) < 10:
            return

        avg_latency = statistics.mean(self.latency_history)
        if avg_latency < 1.0:  # Fast responses, can increase batch size
            self._adaptive_batch_size = min(self._adaptive_batch_size + 5, self.config.max_batch_size)
        elif avg_latency > 5.0:  # Slow responses, reduce batch size
            self._adaptive_batch_size = max(self._adaptive_batch_size - 2, self.config.min_batch_size)

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should allow requests."""
        current_time = time.time()

        if self.circuit_breaker.state == "open":
            if current_time - self.circuit_breaker.last_failure_time > self.config.circuit_breaker_timeout:
                self.circuit_breaker.state = "half-open"
                return True
            return False

        return True

    def _record_failure(self) -> None:
        """Record a failure for circuit breaker."""
        self.circuit_breaker.failures += 1
        self.circuit_breaker.last_failure_time = time.time()

        if self.circuit_breaker.failures >= self.config.circuit_breaker_threshold:
            self.circuit_breaker.state = "open"
            logger.warning(f"Circuit breaker opened after {self.circuit_breaker.failures} failures")

    def _record_success(self) -> None:
        """Record a success for circuit breaker."""
        if self.circuit_breaker.state == "half-open":
            self.circuit_breaker.state = "closed"
            self.circuit_breaker.failures = 0
            logger.info("Circuit breaker closed - service recovered")

    async def _execute_with_retry(
        self,
        func: Callable[[], Any],
        max_retries: int,
        retry_delay: float
    ) -> Any:
        """Execute a function with retry logic."""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")

        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Function failed after all retries")

    async def _cached_generate(
        self,
        adapter: AsyncAdapterWrapper,
        prompt: str,
        cache_key: str,
        **kwargs: Any
    ) -> Tuple[str, bool]:
        """Generate with caching support."""
        loop = asyncio.get_running_loop()

        # Try cache first
        if self.cache is not None:
            try:
                cached_result = await loop.run_in_executor(
                    self._thread_pool,
                    lambda: self.cache.get(cache_key)  # type: ignore
                )
                if cached_result is not None:
                    self.cache_stats.hits += 1
                    return cached_result, True
            except Exception as e:
                logger.debug(f"Cache read error: {e}")

        self.cache_stats.misses += 1

        # Generate new result
        result = await adapter.agenerate(prompt, **kwargs)

        # Cache the result
        if self.cache is not None:
            try:
                await loop.run_in_executor(
                    self._thread_pool,
                    lambda: self.cache.set(cache_key, result)  # type: ignore
                )
            except Exception as e:
                logger.debug(f"Cache write error: {e}")

        return result, False

    async def evaluate_batch(
        self,
        adapter: Any,
        prompts: List[str],
        cache_keys: Optional[List[str]] = None,
        priorities: Optional[List[int]] = None,
        **kwargs: Any
    ) -> List[AsyncEvaluationResult]:
        """
        Evaluate a batch of prompts asynchronously with priority scheduling.

        Args:
            adapter: The model adapter to use
            prompts: List of prompts to evaluate
            cache_keys: Optional cache keys for each prompt
            priorities: Optional priority levels for each prompt (1-3, higher = more important)
            **kwargs: Additional arguments for generation

        Returns:
            List of evaluation results
        """
        async_adapter = AsyncAdapterWrapper(adapter, self._thread_pool)
        cache_keys = cache_keys or [hash_prompt([prompt]) for prompt in prompts]
        priorities = priorities or [1] * len(prompts)

        # Update adaptive batch size
        self._update_adaptive_batch_size()

        # Create prioritized tasks
        for i, (prompt, cache_key, priority) in enumerate(zip(prompts, cache_keys, priorities)):
            task = self._evaluate_single(
                async_adapter, prompt, cache_key, i, priority=priority, **kwargs
            )
            heapq.heappush(self.task_queue, PrioritizedTask(priority, i, asyncio.create_task(task)))

        # Execute tasks by priority
        results: List[AsyncEvaluationResult] = []
        completed_tasks = set()

        while self.task_queue and len(completed_tasks) < len(prompts):
            # Get highest priority task
            prioritized_task = heapq.heappop(self.task_queue)
            task = prioritized_task.coro
            index = prioritized_task.index

            if index in completed_tasks:
                continue

            try:
                result = await task
                results.append(result)
                completed_tasks.add(index)

                # Record latency for adaptive batching
                self.latency_history.append(result.latency)

                # Record success for circuit breaker
                self._record_success()

            except Exception as e:
                # Record failure for circuit breaker
                self._record_failure()

                # Create error result
                error_result = AsyncEvaluationResult(
                    index=index,
                    prediction="",
                    latency=0.0,
                    error=str(e),
                    cached=False
                )
                results.append(error_result)
                completed_tasks.add(index)

        # Clear the task queue
        self.task_queue.clear()

        # Sort results by index to maintain original order
        results.sort(key=lambda r: r.index)

        return results

    async def _evaluate_single(
        self,
        adapter: AsyncAdapterWrapper,
        prompt: str,
        cache_key: str,
        index: int,
        priority: int = 1,
        **kwargs: Any
    ) -> AsyncEvaluationResult:
        """Evaluate a single prompt."""
        start_time = time.perf_counter()

        try:
            # Circuit breaker check
            if not self._check_circuit_breaker():
                raise RuntimeError("Circuit breaker is open, request denied")

            # Execute with retry logic
            async def _generate():
                return await self._cached_generate(adapter, prompt, cache_key, **kwargs)

            result, cached = await self._execute_with_retry(
                _generate,
                self.config.max_retries,
                self.config.retry_delay
            )

            latency = time.perf_counter() - start_time

            return AsyncEvaluationResult(
                index=index,
                prediction=result,
                latency=latency,
                cached=cached,
                priority=priority
            )

        except Exception as e:
            latency = time.perf_counter() - start_time
            return AsyncEvaluationResult(
                index=index,
                prediction="",
                latency=latency,
                error=str(e),
                cached=False,
                priority=priority
            )

    async def evaluate_streaming(
        self,
        adapter: Any,
        prompt_iterator: AsyncIterator[str],
        cache_key_iterator: Optional[AsyncIterator[str]] = None,
        batch_size: int = 10,
        **kwargs: Any
    ) -> AsyncIterator[AsyncEvaluationResult]:
        """
        Evaluate prompts from async iterators in streaming fashion.

        Args:
            adapter: The model adapter to use
            prompt_iterator: Async iterator of prompts
            cache_key_iterator: Optional async iterator of cache keys
            batch_size: Size of batches to process
            **kwargs: Additional arguments for generation

        Yields:
            Evaluation results as they complete
        """
        async_adapter = AsyncAdapterWrapper(adapter, self._thread_pool)

        # Collect batches
        current_batch = []
        current_cache_keys = []

        async for prompt in prompt_iterator:
            current_batch.append(prompt)

            # Get cache key
            if cache_key_iterator:
                try:
                    cache_key = await cache_key_iterator.__anext__()
                except StopAsyncIteration:
                    cache_key = hash_prompt([prompt])
            else:
                cache_key = hash_prompt([prompt])

            current_cache_keys.append(cache_key)

            # Process batch when it reaches the target size
            if len(current_batch) >= batch_size:
                async for result in self._process_batch_streaming(
                    async_adapter, current_batch, current_cache_keys, **kwargs
                ):
                    yield result

                current_batch = []
                current_cache_keys = []

        # Process remaining batch
        if current_batch:
            async for result in self._process_batch_streaming(
                async_adapter, current_batch, current_cache_keys, **kwargs
            ):
                yield result

    async def _process_batch_streaming(
        self,
        adapter: AsyncAdapterWrapper,
        prompts: List[str],
        cache_keys: List[str],
        **kwargs: Any
    ) -> AsyncIterator[AsyncEvaluationResult]:
        """Process a batch and yield results as they complete."""
        # Create tasks for the batch
        tasks = []
        for i, (prompt, cache_key) in enumerate(zip(prompts, cache_keys)):
            task = self._evaluate_single(adapter, prompt, cache_key, i, **kwargs)
            tasks.append(task)

        # Execute tasks and yield results as they complete
        for coro in asyncio.as_completed(tasks):
            result = await coro
            yield result

    @asynccontextmanager
    async def session(self):
        """Async context manager for evaluation sessions."""
        try:
            yield self
        finally:
            # Cleanup resources
            self._thread_pool.shutdown(wait=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "cache_hits": self.cache_stats.hits,
            "cache_misses": self.cache_stats.misses,
            "cache_hit_rate": self.cache_stats.hit_rate,
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "thread_pool_workers": self._thread_pool._max_workers
        }


# Utility functions for easy integration
async def evaluate_with_async_engine(
    adapter: Any,
    prompts: List[str],
    config: Optional[AsyncTaskConfig] = None,
    cache: Optional[PredictionCache] = None
) -> List[AsyncEvaluationResult]:
    """
    Convenience function for async evaluation.

    Args:
        adapter: Model adapter
        prompts: List of prompts
        config: Async configuration
        cache: Optional cache

    Returns:
        List of evaluation results
    """
    engine = AsyncEvaluationEngine(config)
    if cache:
        engine.set_cache(cache)

    async with engine.session():
        return await engine.evaluate_batch(adapter, prompts)


def create_async_iterator_from_list(items: List[Any]) -> AsyncIterator[Any]:
    """Create an async iterator from a list."""
    async def _aiter():
        for item in items:
            yield item
    return _aiter()