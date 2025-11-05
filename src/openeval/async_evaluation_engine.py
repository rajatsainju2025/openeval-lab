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

import time
from typing import Any, Dict, List, Optional, Callable, AsyncIterator, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from collections import deque
import statistics
import heapq

# Lazy import asyncio for faster startup
_asyncio = None


def _get_asyncio():
    """Lazy import asyncio."""
    global _asyncio
    if _asyncio is None:
        import asyncio

        _asyncio = asyncio
    return _asyncio


try:
    import aiofiles

    HAS_AIOFILES = True
except ImportError:
    aiofiles = None  # type: ignore
    HAS_AIOFILES = False

from .logging import get_logger
from .cache import PredictionCache, CacheStats
from .utils import hash_prompt

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
    # Adaptive concurrency parameters
    enable_adaptive_concurrency: bool = True
    adaptive_concurrency_config: Optional[AdaptiveConcurrencyConfig] = None


@dataclass
class CircuitBreakerState:
    """State for circuit breaker pattern."""

    failures: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half-open


@dataclass
class AdaptiveConcurrencyConfig:
    """Configuration for adaptive concurrency control."""

    min_concurrency: int = 1
    max_concurrency: int = 50
    target_response_time: float = 2.0  # seconds
    adaptation_interval: float = 10.0  # seconds
    cpu_threshold_high: float = 0.8  # scale down when CPU > 80%
    cpu_threshold_low: float = 0.3   # scale up when CPU < 30%
    memory_threshold_high: float = 0.85  # scale down when memory > 85%
    memory_threshold_low: float = 0.4   # scale up when memory < 40%
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    stabilization_window: int = 5  # measurements to consider stable
    enable_load_balancing: bool = True


class AdaptiveConcurrencyController:
    """Adaptive concurrency controller that adjusts based on system load and performance."""

    def __init__(self, config: AdaptiveConcurrencyConfig):
        self.config = config
        self.current_concurrency = config.min_concurrency
        self.last_adaptation = time.time()
        self.response_times: deque[float] = deque(maxlen=100)
        self.cpu_measurements: deque[float] = deque(maxlen=20)
        self.memory_measurements: deque[float] = deque(maxlen=20)
        self.concurrency_history: deque[int] = deque(maxlen=10)
        self._lock = _get_asyncio().Lock()

    async def get_optimal_concurrency(self) -> int:
        """Calculate optimal concurrency based on current conditions."""
        async with self._lock:
            current_time = time.time()

            # Check if we should adapt
            if current_time - self.last_adaptation < self.config.adaptation_interval:
                return self.current_concurrency

            # Update system metrics
            await self._update_system_metrics()

            # Calculate optimal concurrency
            optimal = self._calculate_optimal_concurrency()

            # Apply bounds and stabilization
            optimal = self._apply_bounds_and_stabilization(optimal)

            # Update state
            self.current_concurrency = optimal
            self.last_adaptation = current_time
            self.concurrency_history.append(optimal)

            return optimal

    def record_response_time(self, response_time: float) -> None:
        """Record a response time measurement."""
        self.response_times.append(response_time)

    async def _update_system_metrics(self) -> None:
        """Update CPU and memory measurements."""
        try:
            import psutil

            # CPU measurement (average over short period)
            cpu_percent = psutil.cpu_percent(interval=0.5)
            self.cpu_measurements.append(cpu_percent / 100.0)

            # Memory measurement
            memory = psutil.virtual_memory()
            self.memory_measurements.append(memory.percent / 100.0)

        except ImportError:
            # Fallback estimates
            self.cpu_measurements.append(0.5)  # Default CPU estimate
            self.memory_measurements.append(0.5)  # Default memory estimate

    def _calculate_optimal_concurrency(self) -> int:
        """Calculate optimal concurrency based on all factors."""
        if len(self.response_times) < 5:
            return self.current_concurrency

        # Base calculation from response times
        avg_response_time = statistics.mean(self.response_times)
        response_time_ratio = self.config.target_response_time / avg_response_time

        optimal = int(self.current_concurrency * response_time_ratio)

        # Apply system resource constraints
        optimal = self._apply_resource_constraints(optimal)

        # Apply load balancing considerations
        if self.config.enable_load_balancing:
            optimal = self._apply_load_balancing(optimal)

        return optimal

    def _apply_resource_constraints(self, base_concurrency: int) -> int:
        """Apply CPU and memory constraints to concurrency."""
        if not self.cpu_measurements or not self.memory_measurements:
            return base_concurrency

        avg_cpu = statistics.mean(self.cpu_measurements)
        avg_memory = statistics.mean(self.memory_measurements)

        # CPU-based adjustment
        if avg_cpu > self.config.cpu_threshold_high:
            cpu_factor = self.config.scale_down_factor
        elif avg_cpu < self.config.cpu_threshold_low:
            cpu_factor = self.config.scale_up_factor
        else:
            cpu_factor = 1.0

        # Memory-based adjustment
        if avg_memory > self.config.memory_threshold_high:
            memory_factor = self.config.scale_down_factor
        elif avg_memory < self.config.memory_threshold_low:
            memory_factor = self.config.scale_up_factor
        else:
            memory_factor = 1.0

        # Combine factors (take the more restrictive one)
        adjustment_factor = min(cpu_factor, memory_factor)

        return int(base_concurrency * adjustment_factor)

    def _apply_load_balancing(self, base_concurrency: int) -> int:
        """Apply load balancing considerations."""
        if len(self.concurrency_history) < 3:
            return base_concurrency

        # Check for oscillations (rapid changes indicate instability)
        recent_changes = []
        history_list = list(self.concurrency_history)
        for i in range(1, len(history_list)):
            change = abs(history_list[i] - history_list[i-1])
            recent_changes.append(change)

        if recent_changes:
            avg_change = statistics.mean(recent_changes)
            max_recent = max(history_list[-3:]) if len(history_list) >= 3 else max(history_list)
            min_recent = min(history_list[-3:]) if len(history_list) >= 3 else min(history_list)

            # If oscillating too much, stabilize
            if avg_change > max_recent * 0.3:  # More than 30% change on average
                # Return to middle of recent range for stability
                return (max_recent + min_recent) // 2

        return base_concurrency

    def _apply_bounds_and_stabilization(self, optimal: int) -> int:
        """Apply bounds and stabilization logic."""
        # Apply absolute bounds
        optimal = max(self.config.min_concurrency, min(self.config.max_concurrency, optimal))

        # Stabilization: don't change by more than 50% at once
        max_change = max(1, self.current_concurrency // 2)
        if abs(optimal - self.current_concurrency) > max_change:
            if optimal > self.current_concurrency:
                optimal = self.current_concurrency + max_change
            else:
                optimal = self.current_concurrency - max_change

        return optimal

    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "current_concurrency": self.current_concurrency,
            "avg_response_time": statistics.mean(self.response_times) if self.response_times else 0.0,
            "avg_cpu_usage": statistics.mean(self.cpu_measurements) if self.cpu_measurements else 0.0,
            "avg_memory_usage": statistics.mean(self.memory_measurements) if self.memory_measurements else 0.0,
            "response_time_count": len(self.response_times),
            "concurrency_history": list(self.concurrency_history),
        }


class DynamicSemaphore:
    """Semaphore that can dynamically adjust its limit."""

    def __init__(self, initial_limit: int):
        self._limit = initial_limit
        self._current = 0
        self._waiters: deque = deque()
        self._lock = _get_asyncio().Lock()

    async def acquire(self) -> None:
        """Acquire the semaphore."""
        async with self._lock:
            if self._current < self._limit:
                self._current += 1
                return

            # Create a future for waiting
            future = _get_asyncio().Future()
            self._waiters.append(future)

        # Wait for the semaphore
        await future

    def release(self) -> None:
        """Release the semaphore."""
        # Note: This is a simplified implementation. In production, use asyncio.Lock properly.
        self._current -= 1

        # Wake up a waiter if any (simplified)
        if self._waiters:
            waiter = self._waiters.popleft()
            self._current += 1
            waiter.set_result(None)

    def set_limit(self, new_limit: int) -> None:
        """Set a new limit for the semaphore."""
        if new_limit < 1:
            new_limit = 1

        self._limit = new_limit

        # Wake up waiters if the limit increased
        while self._waiters and self._current < self._limit:
            waiter = self._waiters.popleft()
            self._current += 1
            waiter.set_result(None)

    @property
    def limit(self) -> int:
        """Get current limit."""
        return self._limit

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()
        return False


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
    coro: Any = field(compare=False)


class ConnectionPool:
    """Connection pool for efficient resource reuse."""

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.available_connections = _get_asyncio().Queue(maxsize=max_connections)
        self._connection_count = 0
        self._lock = _get_asyncio().Lock()

    async def acquire(self) -> Any:
        """Acquire a connection from the pool."""
        try:
            return self.available_connections.get_nowait()
        except _get_asyncio().QueueEmpty:
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
        except _get_asyncio().QueueFull:
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
        if hasattr(self.adapter, "agenerate") and _get_asyncio().iscoroutinefunction(
            self.adapter.agenerate
        ):
            return await self.adapter.agenerate(prompt, **kwargs)

        # Fallback to sync method in thread pool
        loop = _get_asyncio().get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool, lambda: self.adapter.generate(prompt, **kwargs)
        )

    async def agenerate_with_logprobs(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Async generate with logprobs method."""
        if hasattr(self.adapter, "agenerate_with_logprobs") and _get_asyncio().iscoroutinefunction(
            self.adapter.agenerate_with_logprobs
        ):
            return await self.adapter.agenerate_with_logprobs(prompt, **kwargs)

        # Fallback to sync method
        loop = _get_asyncio().get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool, lambda: self.adapter.generate_with_logprobs(prompt, **kwargs)
        )


class AsyncEvaluationEngine:
    """
    High-performance async evaluation engine with advanced optimizations.
    """

    def __init__(self, config: Optional[AsyncTaskConfig] = None):
        self.config = config or AsyncTaskConfig()

        # Initialize adaptive concurrency
        if self.config.enable_adaptive_concurrency:
            adaptive_config = self.config.adaptive_concurrency_config or AdaptiveConcurrencyConfig()
            self.concurrency_controller = AdaptiveConcurrencyController(adaptive_config)
            initial_concurrency = adaptive_config.min_concurrency
        else:
            self.concurrency_controller = None
            initial_concurrency = self.config.semaphore_limit or self.config.max_concurrent_requests

        # Use dynamic semaphore if adaptive concurrency is enabled
        if self.concurrency_controller:
            self.semaphore = DynamicSemaphore(initial_concurrency)
        else:
            self.semaphore = _get_asyncio().Semaphore(initial_concurrency)

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
            self._adaptive_batch_size = min(
                self._adaptive_batch_size + 5, self.config.max_batch_size
            )
        elif avg_latency > 5.0:  # Slow responses, reduce batch size
            self._adaptive_batch_size = max(
                self._adaptive_batch_size - 2, self.config.min_batch_size
            )

    async def _adapt_concurrency(self) -> None:
        """Adapt concurrency based on system conditions and performance."""
        if not self.concurrency_controller or not isinstance(self.semaphore, DynamicSemaphore):
            return

        # Record response times for controller
        if self.latency_history:
            avg_response_time = statistics.mean(self.latency_history)
            self.concurrency_controller.record_response_time(avg_response_time)

        # Get optimal concurrency
        optimal_concurrency = await self.concurrency_controller.get_optimal_concurrency()

        # Update semaphore limit
        self.semaphore.set_limit(optimal_concurrency)

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should allow requests."""
        current_time = time.time()

        if self.circuit_breaker.state == "open":
            if (
                current_time - self.circuit_breaker.last_failure_time
                > self.config.circuit_breaker_timeout
            ):
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
        self, func: Callable[[], Any], max_retries: int, retry_delay: float
    ) -> Any:
        """Execute a function with retry logic."""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    await _get_asyncio().sleep(retry_delay * (2**attempt))  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")

        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Function failed after all retries")

    async def _cached_generate(
        self, adapter: AsyncAdapterWrapper, prompt: str, cache_key: str, **kwargs: Any
    ) -> Tuple[str, bool]:
        """Generate with caching support."""
        loop = _get_asyncio().get_running_loop()

        # Try cache first
        if self.cache is not None:
            try:
                cached_result = await loop.run_in_executor(
                    self._thread_pool, lambda: self.cache.get(cache_key)  # type: ignore
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
                    self._thread_pool, lambda: self.cache.set(cache_key, result)  # type: ignore
                )
            except Exception as e:
                logger.debug(f"Cache write error: {e}")

        return result, False

    async def _cached_generate_batch(
        self, adapter: AsyncAdapterWrapper, prompts: List[str], cache_keys: List[str], **kwargs: Any
    ) -> List[Tuple[str, bool]]:
        """Generate a batch with cache-aware lookups to minimize round trips."""

        loop = _get_asyncio().get_running_loop()
        results: List[Optional[Tuple[str, bool]]] = [None] * len(prompts)
        cached_results: Dict[str, Any] = {}

        if self.cache is not None:
            try:
                if hasattr(self.cache, "get_batch"):
                    cached_batch = await loop.run_in_executor(
                        self._thread_pool,
                        lambda: self.cache.get_batch(cache_keys),  # type: ignore[attr-defined]
                    )
                    cached_results = {
                        key: value
                        for key, value in zip(cache_keys, cached_batch)
                        if value is not None
                    }
                else:
                    for cache_key in cache_keys:
                        try:
                            cached_value = await loop.run_in_executor(
                                self._thread_pool,
                                lambda k=cache_key: self.cache.get(k),  # type: ignore[arg-type]
                            )
                            if cached_value is not None:
                                cached_results[cache_key] = cached_value
                        except Exception as exc:
                            logger.debug(f"Cache read error for {cache_key}: {exc}")
            except Exception as exc:
                logger.debug(f"Batch cache read error: {exc}")

        to_generate: List[Tuple[int, str, str]] = []
        for idx, (prompt, cache_key) in enumerate(zip(prompts, cache_keys)):
            cached_value = cached_results.get(cache_key)
            if cached_value is not None:
                results[idx] = (cached_value, True)
                self.cache_stats.hits += 1
            else:
                to_generate.append((idx, prompt, cache_key))
                self.cache_stats.misses += 1

        async def _generate_with_limit(prompt: str, **inner_kwargs: Any) -> str:
            async with self.semaphore:
                return await adapter.agenerate(prompt, **inner_kwargs)

        if to_generate:
            generated_outputs = await _get_asyncio().gather(
                *[_generate_with_limit(prompt, **kwargs) for _, prompt, _ in to_generate]
            )

            if self.cache is not None:
                try:
                    cache_updates = list(
                        zip([cache_key for _, _, cache_key in to_generate], generated_outputs)
                    )
                    if hasattr(self.cache, "set_batch"):
                        await loop.run_in_executor(
                            self._thread_pool,
                            lambda: self.cache.set_batch(cache_updates),  # type: ignore[attr-defined]
                        )
                    else:
                        for cache_key, output in cache_updates:
                            await loop.run_in_executor(
                                self._thread_pool,
                                lambda k=cache_key, v=output: self.cache.set(k, v),  # type: ignore[arg-type]
                            )
                except Exception as exc:
                    logger.debug(f"Batch cache write error: {exc}")

            for (original_index, _, _), output in zip(to_generate, generated_outputs):
                results[original_index] = (output, False)

        return [item if item is not None else ("", False) for item in results]

    async def evaluate_batch(
        self,
        adapter: Any,
        prompts: List[str],
        cache_keys: Optional[List[str]] = None,
        priorities: Optional[List[int]] = None,
        **kwargs: Any,
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

        # Adapt concurrency based on current conditions
        await self._adapt_concurrency()
        priorities = priorities or [1] * len(prompts)

        # Update adaptive batch size
        self._update_adaptive_batch_size()

        # Create prioritized tasks
        for i, (prompt, cache_key, priority) in enumerate(zip(prompts, cache_keys, priorities)):
            task = self._evaluate_single(
                async_adapter, prompt, cache_key, i, priority=priority, **kwargs
            )
            heapq.heappush(
                self.task_queue, PrioritizedTask(priority, i, _get_asyncio().create_task(task))
            )

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
                    index=index, prediction="", latency=0.0, error=str(e), cached=False
                )
                results.append(error_result)
                completed_tasks.add(index)

        # Clear the task queue
        self.task_queue.clear()

        # Sort results by index to maintain original order
        results.sort(key=lambda r: r.index)

        return results

    async def evaluate_batch_optimized(
        self,
        adapter: Any,
        prompts: List[str],
        cache_keys: Optional[List[str]] = None,
        priorities: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> List[AsyncEvaluationResult]:
        """
        Evaluate a batch of prompts with optimized batch caching.

        This method groups prompts by cache status and processes cached/uncached
        items in batches for better performance.
        """
        async_adapter = AsyncAdapterWrapper(adapter, self._thread_pool)
        cache_keys = cache_keys or [hash_prompt([prompt]) for prompt in prompts]
        priorities = priorities or [1] * len(prompts)

        # Update adaptive batch size
        self._update_adaptive_batch_size()

        start_time = time.perf_counter()

        try:
            # Use batch caching for better performance
            batch_results = await self._cached_generate_batch(
                async_adapter, prompts, cache_keys, **kwargs
            )

            # Convert to AsyncEvaluationResult format
            results = []
            total_latency = time.perf_counter() - start_time

            for i, (prediction, cached) in enumerate(batch_results):
                # Estimate latency per item (could be improved with per-item timing)
                estimated_latency = total_latency / len(prompts)

                result = AsyncEvaluationResult(
                    index=i,
                    prediction=prediction,
                    latency=estimated_latency,
                    cached=cached,
                    priority=priorities[i],
                )
                results.append(result)

                # Record latency for adaptive batching
                self.latency_history.append(estimated_latency)

            # Record success for circuit breaker
            self._record_success()

            return results

        except Exception as e:
            # Record failure for circuit breaker
            self._record_failure()

            # Create error results for all prompts
            error_results = []
            for i in range(len(prompts)):
                error_result = AsyncEvaluationResult(
                    index=i,
                    prediction="",
                    latency=0.0,
                    error=str(e),
                    cached=False,
                    priority=priorities[i],
                )
                error_results.append(error_result)

            return error_results

    async def _evaluate_single(
        self,
        adapter: AsyncAdapterWrapper,
        prompt: str,
        cache_key: str,
        index: int,
        priority: int = 1,
        **kwargs: Any,
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
                _generate, self.config.max_retries, self.config.retry_delay
            )

            latency = time.perf_counter() - start_time

            return AsyncEvaluationResult(
                index=index, prediction=result, latency=latency, cached=cached, priority=priority
            )

        except Exception as e:
            latency = time.perf_counter() - start_time
            return AsyncEvaluationResult(
                index=index,
                prediction="",
                latency=latency,
                error=str(e),
                cached=False,
                priority=priority,
            )

    async def evaluate_streaming(
        self,
        adapter: Any,
        prompt_iterator: AsyncIterator[str],
        cache_key_iterator: Optional[AsyncIterator[str]] = None,
        batch_size: int = 10,
        **kwargs: Any,
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
        self, adapter: AsyncAdapterWrapper, prompts: List[str], cache_keys: List[str], **kwargs: Any
    ) -> AsyncIterator[AsyncEvaluationResult]:
        """Process a batch and yield results as they complete."""
        # Create tasks for the batch
        tasks = []
        for i, (prompt, cache_key) in enumerate(zip(prompts, cache_keys)):
            task = self._evaluate_single(adapter, prompt, cache_key, i, **kwargs)
            tasks.append(task)

        # Execute tasks and yield results as they complete
        for coro in _get_asyncio().as_completed(tasks):
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
        """Get engine statistics including cache performance."""
        cache_stats = {
            "cache_hits": self.cache_stats.hits,
            "cache_misses": self.cache_stats.misses,
            "cache_hit_rate": (
                self.cache_stats.hits / (self.cache_stats.hits + self.cache_stats.misses)
                if (self.cache_stats.hits + self.cache_stats.misses) > 0
                else 0.0
            ),
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "thread_pool_workers": self._thread_pool._max_workers,
            "adaptive_batch_size": self._adaptive_batch_size,
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_failures": self.circuit_breaker.failures,
        }

        # Add concurrency controller stats if available
        if self.concurrency_controller:
            concurrency_stats = self.concurrency_controller.get_stats()
            cache_stats["concurrency_controller"] = concurrency_stats
            cache_stats["current_semaphore_limit"] = self.semaphore.limit if isinstance(self.semaphore, DynamicSemaphore) else self.config.max_concurrent_requests

        if len(self.latency_history) > 0:
            cache_stats["avg_latency"] = statistics.mean(self.latency_history)
            cache_stats["p95_latency"] = (
                statistics.quantiles(self.latency_history, n=20)[18]
                if len(self.latency_history) >= 20
                else max(self.latency_history)
            )

        return cache_stats


# Utility functions for easy integration
async def evaluate_with_async_engine(
    adapter: Any,
    prompts: List[str],
    config: Optional[AsyncTaskConfig] = None,
    cache: Optional[PredictionCache] = None,
    use_optimized_batch: bool = True,
) -> List[AsyncEvaluationResult]:
    """
    Convenience function for async evaluation.

    Args:
        adapter: Model adapter
        prompts: List of prompts
        config: Async configuration
        cache: Optional cache
        use_optimized_batch: Whether to use optimized batch processing

    Returns:
        List of evaluation results
    """
    engine = AsyncEvaluationEngine(config)
    if cache:
        engine.set_cache(cache)

    async with engine.session():
        if use_optimized_batch:
            return await engine.evaluate_batch_optimized(adapter, prompts)
        else:
            return await engine.evaluate_batch(adapter, prompts)


def create_async_iterator_from_list(items: List[Any]) -> AsyncIterator[Any]:
    """Create an async iterator from a list."""

    async def _aiter():
        for item in items:
            yield item

    return _aiter()
