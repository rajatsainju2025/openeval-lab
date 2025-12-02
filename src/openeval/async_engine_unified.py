"""
Unified Async Evaluation Engine

Consolidates all async evaluation functionality into a single, optimized engine.
Provides connection pooling, adaptive batching, and fault-tolerant evaluation.
"""

from __future__ import annotations

import asyncio
import time
import statistics
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .core import Task, Dataset, Adapter, Metric
from .imports import httpx, aiohttp, HAS_HTTPX, HAS_AIOHTTP
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class AsyncConfig:
    """Configuration for async evaluation."""

    max_concurrent: int = 50  # Maximum concurrent requests
    batch_size: int = 10  # Batch size for processing
    timeout: float = 30.0  # Request timeout in seconds
    max_retries: int = 3  # Maximum retries per request
    backoff_factor: float = 1.5  # Exponential backoff factor
    circuit_breaker_threshold: int = 5  # Circuit breaker failure threshold
    enable_connection_pooling: bool = True
    adaptive_batching: bool = True
    priority_scheduling: bool = True


@dataclass
class EvaluationTask:
    """Single evaluation task."""

    id: str
    example: Any
    prompt: str
    priority: int = 0  # Higher = higher priority
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other):
        """Support priority queue ordering."""
        return self.priority > other.priority  # Higher priority first


@dataclass
class EvaluationResult:
    """Result of async evaluation."""

    task_id: str
    prediction: Optional[str] = None
    error: Optional[str] = None
    latency: float = 0.0
    retry_count: int = 0
    success: bool = False


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True

    def record_success(self):
        """Record successful execution."""
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed execution."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.threshold:
            self.state = "open"


class AsyncConnectionPool:
    """Connection pool for HTTP clients."""

    def __init__(self, config: AsyncConfig):
        self.config = config
        self._httpx_client: Optional[Any] = None
        self._aiohttp_session: Optional[Any] = None

    async def __aenter__(self):
        if HAS_HTTPX:
            limits = httpx.Limits(
                max_keepalive_connections=self.config.max_concurrent,
                max_connections=self.config.max_concurrent * 2,
            )
            self._httpx_client = httpx.AsyncClient(limits=limits, timeout=self.config.timeout)

        if HAS_AIOHTTP:
            connector = aiohttp.TCPConnector(
                limit=self.config.max_concurrent, limit_per_host=self.config.max_concurrent // 2
            )
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._aiohttp_session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        return self

    async def __aexit__(self, *args):
        if self._httpx_client:
            await self._httpx_client.aclose()
        if self._aiohttp_session:
            await self._aiohttp_session.close()

    def get_httpx_client(self):
        """Get HTTPX client."""
        return self._httpx_client

    def get_aiohttp_session(self):
        """Get aiohttp session."""
        return self._aiohttp_session


class UnifiedAsyncEngine:
    """Unified async evaluation engine with all optimizations."""

    def __init__(self, config: Optional[AsyncConfig] = None):
        self.config = config or AsyncConfig()
        self.circuit_breaker = CircuitBreaker(threshold=self.config.circuit_breaker_threshold)
        self.task_queue: Optional[asyncio.PriorityQueue] = None
        self.results: Dict[str, EvaluationResult] = {}
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "retries": 0,
            "avg_latency": 0.0,
        }

    async def evaluate_async(
        self, task: Task, adapter: Adapter, dataset: Dataset, metrics: List[Metric], **kwargs
    ) -> Dict[str, Any]:
        """Perform async evaluation with optimizations."""
        start_time = time.time()

        # Initialize queue
        self.task_queue = asyncio.PriorityQueue()
        self.results.clear()

        # Create evaluation tasks
        evaluation_tasks = []
        for i, example in enumerate(dataset):
            prompt = task.build_prompt(example)
            eval_task = EvaluationTask(
                id=f"task_{i}",
                example=example,
                prompt=prompt,
                priority=getattr(example, "priority", 0),
            )
            evaluation_tasks.append(eval_task)
            await self.task_queue.put(eval_task)

        self.stats["total_tasks"] = len(evaluation_tasks)

        # Process tasks with connection pooling
        async with AsyncConnectionPool(self.config) as pool:
            # Start worker tasks
            workers = [
                asyncio.create_task(self._worker(adapter, pool))
                for _ in range(min(self.config.max_concurrent, len(evaluation_tasks)))
            ]

            # Wait for completion
            await self.task_queue.join()

            # Cancel workers
            for worker in workers:
                worker.cancel()

            await asyncio.gather(*workers, return_exceptions=True)

        # Collect results and compute metrics
        predictions = []
        references = []
        latencies = []

        for eval_task in evaluation_tasks:
            result = self.results.get(eval_task.id)
            if result and result.success:
                predictions.append(result.prediction)
                references.append(eval_task.example.reference)
                latencies.append(result.latency)

        # Compute metrics
        metric_results = {}
        for metric in metrics:
            try:
                if hasattr(metric, "compute"):
                    metric_results[metric.__class__.__name__] = metric.compute(
                        predictions, references
                    )
            except Exception as e:
                logger.error(f"Error computing metric {metric.__class__.__name__}: {e}")

        # Update stats
        self.stats["completed_tasks"] = len([r for r in self.results.values() if r.success])
        self.stats["failed_tasks"] = len([r for r in self.results.values() if not r.success])
        self.stats["avg_latency"] = statistics.mean(latencies) if latencies else 0.0

        total_time = time.time() - start_time

        return {
            "metrics": metric_results,
            "stats": {
                **self.stats,
                "total_time": total_time,
                "throughput": len(predictions) / total_time if total_time > 0 else 0,
            },
            "config": {
                "max_concurrent": self.config.max_concurrent,
                "batch_size": self.config.batch_size,
                "timeout": self.config.timeout,
            },
        }

    async def _worker(self, adapter: Adapter, pool: AsyncConnectionPool):
        """Worker coroutine for processing evaluation tasks."""
        if self.task_queue is None:
            return

        while True:
            try:
                # Get task from queue
                eval_task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # Check circuit breaker
                if not self.circuit_breaker.can_execute():
                    logger.warning("Circuit breaker open, skipping task")
                    self.task_queue.task_done()
                    continue

                # Process task
                result = await self._process_task(eval_task, adapter, pool)
                self.results[eval_task.id] = result

                # Update circuit breaker
                if result.success:
                    self.circuit_breaker.record_success()
                else:
                    self.circuit_breaker.record_failure()

                self.task_queue.task_done()

            except asyncio.TimeoutError:
                # No more tasks, exit worker
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                if "eval_task" in locals():
                    self.task_queue.task_done()

    async def _process_task(
        self, eval_task: EvaluationTask, adapter: Adapter, pool: AsyncConnectionPool
    ) -> EvaluationResult:
        """Process single evaluation task with retries."""
        result = EvaluationResult(task_id=eval_task.id)

        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()

                # Generate prediction using adapter
                if hasattr(adapter, "generate_async"):
                    # Use async adapter if available
                    prediction = await getattr(adapter, "generate_async")(
                        eval_task.prompt, timeout=self.config.timeout
                    )
                else:
                    # Fallback to sync adapter in thread pool
                    loop = asyncio.get_event_loop()
                    prediction = await loop.run_in_executor(
                        None, adapter.generate, eval_task.prompt
                    )

                result.prediction = prediction
                result.latency = time.time() - start_time
                result.success = True
                result.retry_count = attempt

                break

            except Exception as e:
                result.error = str(e)
                result.retry_count = attempt

                if attempt < self.config.max_retries:
                    # Exponential backoff
                    wait_time = self.config.backoff_factor**attempt
                    await asyncio.sleep(wait_time)
                    self.stats["retries"] += 1
                else:
                    logger.error(f"Task {eval_task.id} failed after {attempt + 1} attempts: {e}")

        return result

    async def batch_evaluate(
        self, tasks: List[Tuple[Task, Adapter, Dataset, List[Metric]]], **kwargs
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple task/adapter/dataset combinations concurrently."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def evaluate_single(task_tuple):
            async with semaphore:
                task, adapter, dataset, metrics = task_tuple
                return await self.evaluate_async(task, adapter, dataset, metrics, **kwargs)

        # Create coroutines
        coroutines = [evaluate_single(task_tuple) for task_tuple in tasks]

        # Execute concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch evaluation {i} failed: {result}")
                processed_results.append({"error": str(result), "metrics": {}, "stats": {}})
            else:
                processed_results.append(result)

        return processed_results


# Factory function for easy engine creation
def create_async_engine(
    max_concurrent: int = 50, timeout: float = 30.0, max_retries: int = 3, **kwargs
) -> UnifiedAsyncEngine:
    """Create async evaluation engine with specified configuration."""
    config = AsyncConfig(
        max_concurrent=max_concurrent, timeout=timeout, max_retries=max_retries, **kwargs
    )
    return UnifiedAsyncEngine(config)


__all__ = [
    "AsyncConfig",
    "UnifiedAsyncEngine",
    "create_async_engine",
]
