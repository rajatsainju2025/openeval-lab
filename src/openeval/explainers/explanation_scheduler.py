"""Explanation scheduler module for scheduling explanation generation tasks.

This module provides tools for scheduling, queuing, and managing explanation
generation tasks with support for priorities, dependencies, and execution control.
"""

from __future__ import annotations

import heapq
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class TaskState(Enum):
    """State of a scheduled task."""

    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    PAUSED = auto()
    TIMEOUT = auto()


class TaskPriority(Enum):
    """Priority levels for tasks."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class ScheduleType(Enum):
    """Types of task scheduling."""

    IMMEDIATE = auto()
    DELAYED = auto()
    RECURRING = auto()
    CRON = auto()
    DEPENDENCY = auto()


@dataclass(order=True)
class ScheduledTask:
    """A scheduled explanation task."""

    priority: int
    scheduled_time: float
    task_id: str = field(compare=False)
    name: str = field(compare=False)
    callable: Callable[..., Any] = field(compare=False)
    args: tuple[Any, ...] = field(default_factory=tuple, compare=False)
    kwargs: dict[str, Any] = field(default_factory=dict, compare=False)
    state: TaskState = field(default=TaskState.PENDING, compare=False)
    created_at: float = field(default_factory=time.time, compare=False)
    started_at: float | None = field(default=None, compare=False)
    completed_at: float | None = field(default=None, compare=False)
    result: Any = field(default=None, compare=False)
    error: Exception | None = field(default=None, compare=False)
    timeout: float | None = field(default=None, compare=False)
    retry_count: int = field(default=0, compare=False)
    max_retries: int = field(default=0, compare=False)
    dependencies: list[str] = field(default_factory=list, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def duration(self) -> float | None:
        """Get task execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def wait_time(self) -> float | None:
        """Get time spent waiting in queue."""
        if self.started_at:
            return self.started_at - self.created_at
        return None

    @property
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.state in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.TIMEOUT,
        )


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_id: str
    success: bool
    result: Any = None
    error: str | None = None
    duration: float = 0.0
    retry_count: int = 0


@dataclass
class SchedulerStats:
    """Statistics for the scheduler."""

    total_tasks: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    total_execution_time: float = 0.0
    average_wait_time: float = 0.0
    average_execution_time: float = 0.0


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    max_workers: int = 4
    max_queue_size: int = 1000
    default_timeout: float | None = 300.0
    default_priority: TaskPriority = TaskPriority.NORMAL
    enable_persistence: bool = False
    retry_delay: float = 1.0
    max_retries: int = 3


class TaskQueue(ABC):
    """Abstract base class for task queues."""

    @abstractmethod
    def push(self, task: ScheduledTask) -> None:
        """Add a task to the queue."""
        pass

    @abstractmethod
    def pop(self) -> ScheduledTask | None:
        """Remove and return the next task."""
        pass

    @abstractmethod
    def peek(self) -> ScheduledTask | None:
        """Return the next task without removing it."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get current queue size."""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        pass


class PriorityTaskQueue(TaskQueue):
    """Priority-based task queue using heap."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize priority queue."""
        self._heap: list[ScheduledTask] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def push(self, task: ScheduledTask) -> None:
        """Add a task to the queue."""
        with self._lock:
            if len(self._heap) >= self._max_size:
                raise RuntimeError("Queue is full")
            heapq.heappush(self._heap, task)

    def pop(self) -> ScheduledTask | None:
        """Remove and return the highest priority task."""
        with self._lock:
            if self._heap:
                return heapq.heappop(self._heap)
            return None

    def peek(self) -> ScheduledTask | None:
        """Return the highest priority task without removing it."""
        with self._lock:
            if self._heap:
                return self._heap[0]
            return None

    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._heap)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._heap) == 0


class FIFOTaskQueue(TaskQueue):
    """First-in-first-out task queue."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize FIFO queue."""
        self._queue: list[ScheduledTask] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def push(self, task: ScheduledTask) -> None:
        """Add a task to the queue."""
        with self._lock:
            if len(self._queue) >= self._max_size:
                raise RuntimeError("Queue is full")
            self._queue.append(task)

    def pop(self) -> ScheduledTask | None:
        """Remove and return the first task."""
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
            return None

    def peek(self) -> ScheduledTask | None:
        """Return the first task without removing it."""
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None

    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0


class TaskExecutor:
    """Executor for running scheduled tasks."""

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize executor."""
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running_tasks: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()

    def submit(self, task: ScheduledTask) -> Future[Any]:
        """Submit a task for execution."""
        future = self._executor.submit(self._execute_task, task)

        with self._lock:
            self._running_tasks[task.task_id] = future

        return future

    def _execute_task(self, task: ScheduledTask) -> TaskResult:
        """Execute a task and return result."""
        task.state = TaskState.RUNNING
        task.started_at = time.time()

        try:
            if task.timeout:
                # Execute with timeout
                result = self._execute_with_timeout(task)
            else:
                result = task.callable(*task.args, **task.kwargs)

            task.result = result
            task.state = TaskState.COMPLETED
            task.completed_at = time.time()

            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                duration=task.duration or 0.0,
                retry_count=task.retry_count,
            )

        except TimeoutError as e:
            task.state = TaskState.TIMEOUT
            task.error = e
            task.completed_at = time.time()

            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                duration=task.duration or 0.0,
            )

        except Exception as e:
            task.state = TaskState.FAILED
            task.error = e
            task.completed_at = time.time()

            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                duration=task.duration or 0.0,
                retry_count=task.retry_count,
            )

        finally:
            with self._lock:
                self._running_tasks.pop(task.task_id, None)

    def _execute_with_timeout(self, task: ScheduledTask) -> Any:
        """Execute task with timeout."""
        import concurrent.futures

        with ThreadPoolExecutor(max_workers=1) as timeout_executor:
            future = timeout_executor.submit(task.callable, *task.args, **task.kwargs)
            try:
                return future.result(timeout=task.timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Task {task.task_id} timed out after {task.timeout}s")

    def cancel(self, task_id: str) -> bool:
        """Cancel a running task."""
        with self._lock:
            future = self._running_tasks.get(task_id)
            if future:
                return future.cancel()
            return False

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=wait)


class ExplanationScheduler:
    """Main scheduler for explanation generation tasks."""

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        """Initialize scheduler."""
        self._config = config or SchedulerConfig()
        self._queue = PriorityTaskQueue(self._config.max_queue_size)
        self._executor = TaskExecutor(self._config.max_workers)
        self._tasks: dict[str, ScheduledTask] = {}
        self._callbacks: dict[str, list[Callable[[TaskResult], None]]] = {}
        self._running = False
        self._scheduler_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    def schedule(
        self,
        callable: Callable[..., Any],
        name: str | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        delay: float = 0.0,
        timeout: float | None = None,
        max_retries: int | None = None,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Schedule a new task."""
        task_id = str(uuid.uuid4())
        scheduled_time = time.time() + delay

        task = ScheduledTask(
            priority=priority.value,
            scheduled_time=scheduled_time,
            task_id=task_id,
            name=name or callable.__name__,
            callable=callable,
            args=args or (),
            kwargs=kwargs or {},
            timeout=timeout or self._config.default_timeout,
            max_retries=max_retries if max_retries is not None else self._config.max_retries,
            dependencies=dependencies or [],
            metadata=metadata or {},
        )

        with self._lock:
            self._tasks[task_id] = task
            task.state = TaskState.QUEUED
            self._queue.push(task)
            self._condition.notify()

        return task_id

    def schedule_many(
        self,
        tasks: list[dict[str, Any]],
    ) -> list[str]:
        """Schedule multiple tasks at once."""
        task_ids = []
        for task_config in tasks:
            task_id = self.schedule(**task_config)
            task_ids.append(task_id)
        return task_ids

    def cancel(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            if task.state == TaskState.QUEUED:
                task.state = TaskState.CANCELLED
                return True

            if task.state == TaskState.RUNNING:
                if self._executor.cancel(task_id):
                    task.state = TaskState.CANCELLED
                    return True

            return False

    def pause(self, task_id: str) -> bool:
        """Pause a queued task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.state == TaskState.QUEUED:
                task.state = TaskState.PAUSED
                return True
            return False

    def resume(self, task_id: str) -> bool:
        """Resume a paused task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.state == TaskState.PAUSED:
                task.state = TaskState.QUEUED
                self._queue.push(task)
                self._condition.notify()
                return True
            return False

    def get_task(self, task_id: str) -> ScheduledTask | None:
        """Get a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_result(
        self, task_id: str, block: bool = True, timeout: float | None = None
    ) -> TaskResult | None:
        """Get the result of a completed task."""
        task = self.get_task(task_id)
        if not task:
            return None

        if block and not task.is_terminal:
            # Wait for task to complete
            start_time = time.time()
            while not task.is_terminal:
                if timeout and (time.time() - start_time) > timeout:
                    return None
                time.sleep(0.1)

        if task.is_terminal:
            return TaskResult(
                task_id=task.task_id,
                success=task.state == TaskState.COMPLETED,
                result=task.result,
                error=str(task.error) if task.error else None,
                duration=task.duration or 0.0,
                retry_count=task.retry_count,
            )

        return None

    def on_complete(self, task_id: str, callback: Callable[[TaskResult], None]) -> None:
        """Register a callback for task completion."""
        with self._lock:
            if task_id not in self._callbacks:
                self._callbacks[task_id] = []
            self._callbacks[task_id].append(callback)

    def start(self) -> None:
        """Start the scheduler."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._scheduler_thread = threading.Thread(target=self._run, daemon=True)
            self._scheduler_thread.start()

    def stop(self, wait: bool = True) -> None:
        """Stop the scheduler."""
        with self._lock:
            self._running = False
            self._condition.notify_all()

        if wait and self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)

        self._executor.shutdown(wait=wait)

    def _run(self) -> None:
        """Main scheduler loop."""
        while self._running:
            with self._condition:
                while self._running and self._queue.is_empty():
                    self._condition.wait(timeout=1.0)

                if not self._running:
                    break

                task = self._queue.pop()
                if not task:
                    continue

                # Check if task is ready
                if task.state == TaskState.CANCELLED:
                    continue

                if task.state == TaskState.PAUSED:
                    continue

                # Check dependencies
                if not self._check_dependencies(task):
                    # Re-queue with slight delay
                    task.scheduled_time = time.time() + 0.5
                    self._queue.push(task)
                    continue

                # Check scheduled time
                now = time.time()
                if task.scheduled_time > now:
                    # Re-queue for later
                    self._queue.push(task)
                    self._condition.wait(timeout=task.scheduled_time - now)
                    continue

            # Submit task for execution
            future = self._executor.submit(task)
            future.add_done_callback(lambda f, t=task: self._on_task_done(t, f))

    def _check_dependencies(self, task: ScheduledTask) -> bool:
        """Check if all dependencies are completed."""
        for dep_id in task.dependencies:
            dep_task = self._tasks.get(dep_id)
            if not dep_task or dep_task.state != TaskState.COMPLETED:
                return False
        return True

    def _on_task_done(self, task: ScheduledTask, future: Future[Any]) -> None:
        """Handle task completion."""
        try:
            result = future.result()

            # Check if retry needed
            if not result.success and task.retry_count < task.max_retries:
                task.retry_count += 1
                task.state = TaskState.QUEUED
                task.scheduled_time = time.time() + self._config.retry_delay
                with self._lock:
                    self._queue.push(task)
                    self._condition.notify()
                return

            # Call callbacks
            with self._lock:
                callbacks = self._callbacks.pop(task.task_id, [])

            for callback in callbacks:
                try:
                    callback(result)
                except Exception:
                    pass  # Ignore callback errors

        except Exception:
            pass  # Task result error

    def get_stats(self) -> SchedulerStats:
        """Get scheduler statistics."""
        with self._lock:
            stats = SchedulerStats()
            stats.total_tasks = len(self._tasks)

            execution_times = []
            wait_times = []

            for task in self._tasks.values():
                if task.state == TaskState.PENDING or task.state == TaskState.QUEUED:
                    stats.pending_tasks += 1
                elif task.state == TaskState.RUNNING:
                    stats.running_tasks += 1
                elif task.state == TaskState.COMPLETED:
                    stats.completed_tasks += 1
                    if task.duration:
                        execution_times.append(task.duration)
                    if task.wait_time:
                        wait_times.append(task.wait_time)
                elif task.state == TaskState.FAILED:
                    stats.failed_tasks += 1
                elif task.state == TaskState.CANCELLED:
                    stats.cancelled_tasks += 1

            stats.total_execution_time = sum(execution_times)
            if execution_times:
                stats.average_execution_time = sum(execution_times) / len(execution_times)
            if wait_times:
                stats.average_wait_time = sum(wait_times) / len(wait_times)

            return stats

    def clear_completed(self) -> int:
        """Clear completed and failed tasks from history."""
        with self._lock:
            to_remove = [task_id for task_id, task in self._tasks.items() if task.is_terminal]
            for task_id in to_remove:
                del self._tasks[task_id]
            return len(to_remove)


# Global instance
_explanation_scheduler: ExplanationScheduler | None = None


def get_explanation_scheduler() -> ExplanationScheduler:
    """Get or create global explanation scheduler."""
    global _explanation_scheduler
    if _explanation_scheduler is None:
        _explanation_scheduler = ExplanationScheduler()
    return _explanation_scheduler


def reset_explanation_scheduler() -> None:
    """Reset global explanation scheduler."""
    global _explanation_scheduler
    if _explanation_scheduler:
        _explanation_scheduler.stop()
    _explanation_scheduler = None


# Convenience functions
def schedule_task(
    callable: Callable[..., Any],
    name: str | None = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    delay: float = 0.0,
    **kwargs: Any,
) -> str:
    """Schedule a task for execution."""
    scheduler = get_explanation_scheduler()
    if not scheduler._running:
        scheduler.start()
    return scheduler.schedule(callable, name=name, priority=priority, delay=delay, **kwargs)


def schedule_explanation(
    explainer: Callable[..., Any],
    code: str,
    priority: TaskPriority = TaskPriority.NORMAL,
    **kwargs: Any,
) -> str:
    """Schedule an explanation generation task."""
    return schedule_task(explainer, args=(code,), priority=priority, **kwargs)


def cancel_task(task_id: str) -> bool:
    """Cancel a scheduled task."""
    return get_explanation_scheduler().cancel(task_id)


def get_task_result(
    task_id: str, block: bool = True, timeout: float | None = None
) -> TaskResult | None:
    """Get the result of a task."""
    return get_explanation_scheduler().get_result(task_id, block=block, timeout=timeout)


def get_scheduler_stats() -> SchedulerStats:
    """Get scheduler statistics."""
    return get_explanation_scheduler().get_stats()


def start_scheduler() -> None:
    """Start the global scheduler."""
    get_explanation_scheduler().start()


def stop_scheduler(wait: bool = True) -> None:
    """Stop the global scheduler."""
    get_explanation_scheduler().stop(wait=wait)


def create_scheduler(config: SchedulerConfig | None = None) -> ExplanationScheduler:
    """Create a new scheduler instance."""
    return ExplanationScheduler(config)
