"""
Async/Await Optimization Patterns

Provides best-practice patterns for async code in OpenEval Lab,
including structured concurrency, timeout handling, and resource cleanup.
"""

from __future__ import annotations

import asyncio
from typing import List, TypeVar, Awaitable, Optional, Callable, Any, Union, Coroutine
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def gather_with_limit(
    tasks: List[Union[Awaitable[T], Coroutine[Any, Any, T]]],
    concurrency: int = 10,
    timeout: Optional[float] = None,
) -> List[Any]:
    """Gather coroutines with concurrency limit.

    Args:
        tasks: List of awaitable tasks
        concurrency: Maximum concurrent tasks
        timeout: Timeout in seconds

    Returns:
        List of results
    """
    results: List[Any] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_task(task: Union[Awaitable[T], Coroutine[Any, Any, T]]) -> Any:
        async with semaphore:
            try:
                if timeout:
                    return await asyncio.wait_for(task, timeout=timeout)  # type: ignore
                else:
                    return await task  # type: ignore
            except asyncio.TimeoutError:
                logger.error("Task timed out")
                raise

    bounded_tasks = [bounded_task(task) for task in tasks]
    results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
    return results


async def timeout_handler(
    coro: Awaitable[T], timeout: float, default: Optional[T] = None
) -> Optional[T]:
    """Handle coroutine with timeout and default value.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        default: Default value if timeout

    Returns:
        Result or default
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default


class AsyncResourceManager:
    """Context manager for async resource cleanup."""

    def __init__(self):
        """Initialize resource manager."""
        self.resources: List[Any] = []

    async def __aenter__(self) -> "AsyncResourceManager":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context with cleanup."""
        # Clean up all resources
        for resource in reversed(self.resources):
            try:
                if hasattr(resource, "aclose"):
                    await resource.aclose()
                elif hasattr(resource, "close"):
                    resource.close()
            except Exception as e:
                logger.error(f"Error cleaning up resource: {e}")

    def add_resource(self, resource: Any) -> None:
        """Add resource for cleanup."""
        self.resources.append(resource)


class StructuredConcurrencyTaskGroup:
    """Task group for structured concurrency.

    Similar to Python 3.11's TaskGroup but compatible with earlier versions.
    """

    def __init__(self):
        """Initialize task group."""
        self.tasks: List[asyncio.Task] = []
        self.exceptions: List[Exception] = []

    async def create_task(self, coro: Union[Awaitable[T], Coroutine[Any, Any, T]]) -> asyncio.Task[T]:  # type: ignore
        """Create a task in the group.

        Args:
            coro: Coroutine to run

        Returns:
            Task object
        """
        task = asyncio.create_task(coro)  # type: ignore
        self.tasks.append(task)
        return task

    async def wait_all(self) -> None:
        """Wait for all tasks to complete."""
        if not self.tasks:
            return

        done, _ = await asyncio.wait(self.tasks)
        for task in done:
            try:
                await task
            except Exception as e:
                self.exceptions.append(e)
                logger.error(f"Task failed: {e}")

    async def __aenter__(self) -> "StructuredConcurrencyTaskGroup":
        """Enter context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context, waiting for all tasks."""
        await self.wait_all()
        if self.exceptions:
            raise RuntimeError(f"{len(self.exceptions)} tasks failed")


async def run_with_progress(
    tasks: List[Union[Awaitable[T], Coroutine[Any, Any, T]]], update_fn: Callable[[int, int], None]
) -> List[Any]:
    """Run tasks with progress updates.

    Args:
        tasks: List of tasks
        update_fn: Callback(completed, total)

    Returns:
        Results
    """
    results: List[Any] = []
    pending = set(asyncio.create_task(t) for t in tasks)  # type: ignore
    total = len(pending)

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                results.append(await task)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                results.append(None)

        update_fn(len(results), total)

    return results
