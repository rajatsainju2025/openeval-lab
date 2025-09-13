"""
Async Evaluation Engine for OpenEval Lab

This module provides an asynchronous evaluation engine that replaces the thread-based
approach with asyncio for better concurrency, reduced overhead, and improved performance.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Union, AsyncIterator, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import asynccontextmanager

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
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


@dataclass
class AsyncEvaluationResult:
    """Result of an async evaluation."""
    index: int
    prediction: Any
    latency: float
    error: Optional[str] = None
    cached: bool = False
    retry_count: int = 0


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
    High-performance async evaluation engine.
    """

    def __init__(self, config: Optional[AsyncTaskConfig] = None):
        self.config = config or AsyncTaskConfig()
        self.semaphore = asyncio.Semaphore(self.config.semaphore_limit or self.config.max_concurrent_requests)
        self.cache: Optional[PredictionCache] = None
        self.cache_stats = CacheStats()
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)

    def set_cache(self, cache: PredictionCache) -> None:
        """Set the prediction cache."""
        self.cache = cache

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
                    lambda: self.cache.get(cache_key)
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
                    lambda: self.cache.set(cache_key, result)
                )
            except Exception as e:
                logger.debug(f"Cache write error: {e}")

        return result, False

    async def evaluate_batch(
        self,
        adapter: Any,
        prompts: List[str],
        cache_keys: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[AsyncEvaluationResult]:
        """
        Evaluate a batch of prompts asynchronously.

        Args:
            adapter: The model adapter to use
            prompts: List of prompts to evaluate
            cache_keys: Optional cache keys for each prompt
            **kwargs: Additional arguments for generation

        Returns:
            List of evaluation results
        """
        async_adapter = AsyncAdapterWrapper(adapter, self._thread_pool)
        cache_keys = cache_keys or [hash_prompt([prompt]) for prompt in prompts]

        # Create evaluation tasks
        tasks = []
        for i, (prompt, cache_key) in enumerate(zip(prompts, cache_keys)):
            task = self._evaluate_single(
                async_adapter, prompt, cache_key, i, **kwargs
            )
            tasks.append(task)

        # Execute with controlled concurrency
        results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        async def execute_with_semaphore(task_coro, index):
            async with semaphore:
                return await task_coro

        # Execute all tasks concurrently with semaphore control
        task_coros = [execute_with_semaphore(task, i) for i, task in enumerate(tasks)]
        results = await asyncio.gather(*task_coros, return_exceptions=True)

        # Handle exceptions and convert to AsyncEvaluationResult
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(AsyncEvaluationResult(
                    index=i,
                    prediction="",
                    latency=0.0,
                    error=str(result),
                    cached=False
                ))
            else:
                final_results.append(result)

        return final_results

    async def _evaluate_single(
        self,
        adapter: AsyncAdapterWrapper,
        prompt: str,
        cache_key: str,
        index: int,
        **kwargs: Any
    ) -> AsyncEvaluationResult:
        """Evaluate a single prompt."""
        start_time = time.perf_counter()

        try:
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
                cached=cached
            )

        except Exception as e:
            latency = time.perf_counter() - start_time
            return AsyncEvaluationResult(
                index=index,
                prediction="",
                latency=latency,
                error=str(e),
                cached=False
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