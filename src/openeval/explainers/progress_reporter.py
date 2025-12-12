"""Progress reporting system for long-running explanation tasks.

This module provides tools for tracking and reporting progress of explanation
generation, batch processing, and other long-running operations.
"""

import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, TextIO, TypeVar

T = TypeVar("T")


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class ProgressStatus(str, Enum):
    """Status of a progress task."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressStyle(str, Enum):
    """Style of progress display."""

    BAR = "bar"  # Progress bar
    SPINNER = "spinner"  # Spinning indicator
    PERCENTAGE = "percentage"  # Percentage only
    COUNTER = "counter"  # Item counter
    MINIMAL = "minimal"  # Minimal output
    VERBOSE = "verbose"  # Detailed output
    SILENT = "silent"  # No output


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ProgressUpdate:
    """A progress update event."""

    task_id: str
    current: int
    total: int
    message: str = ""
    status: ProgressStatus = ProgressStatus.RUNNING
    percentage: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def remaining(self) -> int:
        """Items remaining."""
        return max(0, self.total - self.current)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "status": self.status.value,
            "percentage": self.percentage,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "remaining": self.remaining,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class TaskInfo:
    """Information about a tracked task."""

    id: str
    name: str
    total: int
    current: int = 0
    status: ProgressStatus = ProgressStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def percentage(self) -> float:
        """Completion percentage."""
        if self.total == 0:
            return 100.0 if self.status == ProgressStatus.COMPLETED else 0.0
        return (self.current / self.total) * 100

    @property
    def eta(self) -> Optional[float]:
        """Estimated time remaining in seconds."""
        if self.current == 0 or self.elapsed == 0:
            return None
        rate = self.current / self.elapsed
        remaining = self.total - self.current
        return remaining / rate if rate > 0 else None


@dataclass
class ProgressReport:
    """A summary progress report."""

    tasks: List[TaskInfo]
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_items: int
    completed_items: int
    overall_percentage: float
    elapsed_time: float
    estimated_remaining: Optional[float]
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "overall_percentage": self.overall_percentage,
            "elapsed_time": self.elapsed_time,
            "estimated_remaining": self.estimated_remaining,
            "generated_at": self.generated_at,
        }


@dataclass
class ProgressConfig:
    """Configuration for progress reporting."""

    style: ProgressStyle = ProgressStyle.BAR
    show_eta: bool = True
    show_percentage: bool = True
    show_speed: bool = True
    update_interval: float = 0.1  # Minimum seconds between updates
    bar_width: int = 40
    spinner_chars: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    use_colors: bool = True
    output: TextIO = field(default_factory=lambda: sys.stderr)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Progress Renderer
# =============================================================================


class ProgressRenderer(ABC):
    """Abstract base class for progress renderers."""

    @abstractmethod
    def render(self, update: ProgressUpdate, config: ProgressConfig) -> str:
        """Render a progress update.

        Args:
            update: Progress update to render.
            config: Rendering configuration.

        Returns:
            Rendered string.
        """
        pass


class BarRenderer(ProgressRenderer):
    """Progress bar renderer."""

    def render(self, update: ProgressUpdate, config: ProgressConfig) -> str:
        """Render a progress bar."""
        filled = int(config.bar_width * update.percentage / 100)
        bar = "█" * filled + "░" * (config.bar_width - filled)

        parts = [f"[{bar}]"]

        if config.show_percentage:
            parts.append(f" {update.percentage:5.1f}%")

        parts.append(f" ({update.current}/{update.total})")

        if config.show_eta and update.eta_seconds is not None:
            eta_str = self._format_time(update.eta_seconds)
            parts.append(f" ETA: {eta_str}")

        if update.message:
            parts.append(f" - {update.message}")

        return "".join(parts)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours}h {mins}m"


class SpinnerRenderer(ProgressRenderer):
    """Spinner renderer for indeterminate progress."""

    def __init__(self):
        """Initialize spinner."""
        self._frame = 0

    def render(self, update: ProgressUpdate, config: ProgressConfig) -> str:
        """Render a spinner."""
        char = config.spinner_chars[self._frame % len(config.spinner_chars)]
        self._frame += 1

        parts = [char]

        if update.total > 0:
            parts.append(f" {update.current}/{update.total}")

        if update.message:
            parts.append(f" {update.message}")

        return " ".join(parts)


class PercentageRenderer(ProgressRenderer):
    """Simple percentage renderer."""

    def render(self, update: ProgressUpdate, config: ProgressConfig) -> str:
        """Render percentage."""
        return f"{update.percentage:.1f}% complete ({update.current}/{update.total})"


class VerboseRenderer(ProgressRenderer):
    """Verbose progress renderer with detailed information."""

    def render(self, update: ProgressUpdate, config: ProgressConfig) -> str:
        """Render verbose progress."""
        lines = [
            f"Task: {update.task_id}",
            f"Progress: {update.current}/{update.total} ({update.percentage:.1f}%)",
            f"Status: {update.status.value}",
            f"Elapsed: {self._format_time(update.elapsed_seconds)}",
        ]

        if update.eta_seconds is not None:
            lines.append(f"ETA: {self._format_time(update.eta_seconds)}")

        if update.message:
            lines.append(f"Message: {update.message}")

        return "\n".join(lines)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m {int(seconds % 60)}s"
        else:
            return f"{int(seconds / 3600)}h {int((seconds % 3600) / 60)}m"


# =============================================================================
# Progress Tracker
# =============================================================================


class ProgressTracker:
    """Tracks progress of tasks."""

    def __init__(self, config: Optional[ProgressConfig] = None):
        """Initialize tracker.

        Args:
            config: Optional progress configuration.
        """
        self.config = config or ProgressConfig()
        self._tasks: Dict[str, TaskInfo] = {}
        self._listeners: List[Callable[[ProgressUpdate], None]] = []
        self._renderers: Dict[ProgressStyle, ProgressRenderer] = {
            ProgressStyle.BAR: BarRenderer(),
            ProgressStyle.SPINNER: SpinnerRenderer(),
            ProgressStyle.PERCENTAGE: PercentageRenderer(),
            ProgressStyle.VERBOSE: VerboseRenderer(),
        }
        self._last_update: Dict[str, float] = {}
        self._lock = threading.Lock()

    def create_task(
        self,
        task_id: str,
        name: str,
        total: int,
        parent_id: Optional[str] = None,
    ) -> TaskInfo:
        """Create a new task.

        Args:
            task_id: Unique task identifier.
            name: Human-readable task name.
            total: Total items to process.
            parent_id: Optional parent task ID.

        Returns:
            Created TaskInfo.
        """
        with self._lock:
            task = TaskInfo(
                id=task_id,
                name=name,
                total=total,
                parent_id=parent_id,
            )
            self._tasks[task_id] = task

            if parent_id and parent_id in self._tasks:
                self._tasks[parent_id].children.append(task_id)

            return task

    def start_task(self, task_id: str) -> None:
        """Start a task.

        Args:
            task_id: Task identifier.
        """
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = ProgressStatus.RUNNING
                task.start_time = time.time()
                self._emit_update(task)

    def update_task(
        self,
        task_id: str,
        current: Optional[int] = None,
        increment: int = 0,
        message: str = "",
    ) -> None:
        """Update task progress.

        Args:
            task_id: Task identifier.
            current: Set current progress (absolute).
            increment: Increment current progress.
            message: Optional status message.
        """
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]

            if current is not None:
                task.current = current
            else:
                task.current += increment

            # Check rate limiting
            now = time.time()
            last = self._last_update.get(task_id, 0)
            if now - last >= self.config.update_interval:
                self._last_update[task_id] = now
                self._emit_update(task, message)

    def complete_task(self, task_id: str, message: str = "") -> None:
        """Mark a task as completed.

        Args:
            task_id: Task identifier.
            message: Optional completion message.
        """
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = ProgressStatus.COMPLETED
                task.current = task.total
                task.end_time = time.time()
                self._emit_update(task, message)

    def fail_task(self, task_id: str, message: str = "") -> None:
        """Mark a task as failed.

        Args:
            task_id: Task identifier.
            message: Optional failure message.
        """
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = ProgressStatus.FAILED
                task.end_time = time.time()
                self._emit_update(task, message or "Task failed")

    def cancel_task(self, task_id: str) -> None:
        """Cancel a task.

        Args:
            task_id: Task identifier.
        """
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = ProgressStatus.CANCELLED
                task.end_time = time.time()
                self._emit_update(task, "Cancelled")

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task information.

        Args:
            task_id: Task identifier.

        Returns:
            TaskInfo or None if not found.
        """
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[TaskInfo]:
        """Get all tasks.

        Returns:
            List of TaskInfo objects.
        """
        return list(self._tasks.values())

    def get_report(self) -> ProgressReport:
        """Generate a progress report.

        Returns:
            ProgressReport with summary information.
        """
        tasks = list(self._tasks.values())
        completed = sum(1 for t in tasks if t.status == ProgressStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == ProgressStatus.FAILED)

        total_items = sum(t.total for t in tasks)
        completed_items = sum(t.current for t in tasks)

        overall_pct = (completed_items / total_items * 100) if total_items > 0 else 0

        # Calculate elapsed and ETA
        start_times = [t.start_time for t in tasks if t.start_time]
        elapsed = time.time() - min(start_times) if start_times else 0

        # Estimate remaining time
        eta = None
        if completed_items > 0 and elapsed > 0:
            rate = completed_items / elapsed
            remaining = total_items - completed_items
            eta = remaining / rate if rate > 0 else None

        return ProgressReport(
            tasks=tasks,
            total_tasks=len(tasks),
            completed_tasks=completed,
            failed_tasks=failed,
            total_items=total_items,
            completed_items=completed_items,
            overall_percentage=overall_pct,
            elapsed_time=elapsed,
            estimated_remaining=eta,
        )

    def add_listener(self, listener: Callable[[ProgressUpdate], None]) -> None:
        """Add a progress listener.

        Args:
            listener: Callback function for updates.
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[ProgressUpdate], None]) -> None:
        """Remove a progress listener.

        Args:
            listener: Listener to remove.
        """
        if listener in self._listeners:
            self._listeners.remove(listener)

    def clear(self) -> None:
        """Clear all tasks."""
        with self._lock:
            self._tasks.clear()
            self._last_update.clear()

    def _emit_update(self, task: TaskInfo, message: str = "") -> None:
        """Emit a progress update."""
        update = ProgressUpdate(
            task_id=task.id,
            current=task.current,
            total=task.total,
            message=message,
            status=task.status,
            percentage=task.percentage,
            elapsed_seconds=task.elapsed,
            eta_seconds=task.eta,
        )

        # Render and display
        if self.config.style != ProgressStyle.SILENT:
            self._display_progress(update)

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(update)
            except Exception:
                pass

    def _display_progress(self, update: ProgressUpdate) -> None:
        """Display progress to output."""
        renderer = self._renderers.get(self.config.style)
        if renderer:
            rendered = renderer.render(update, self.config)
            self.config.output.write(f"\r{rendered}")
            self.config.output.flush()

            # Newline on completion
            if update.status in (
                ProgressStatus.COMPLETED,
                ProgressStatus.FAILED,
                ProgressStatus.CANCELLED,
            ):
                self.config.output.write("\n")
                self.config.output.flush()


# =============================================================================
# Progress Context Manager
# =============================================================================


class Progress:
    """Context manager for tracking progress."""

    def __init__(
        self,
        total: int,
        name: str = "Processing",
        tracker: Optional[ProgressTracker] = None,
        config: Optional[ProgressConfig] = None,
    ):
        """Initialize progress context.

        Args:
            total: Total items to process.
            name: Task name.
            tracker: Optional existing tracker.
            config: Optional progress configuration.
        """
        self.total = total
        self.name = name
        self.tracker = tracker or ProgressTracker(config)
        self._task_id = f"progress_{id(self)}_{time.time()}"
        self._task: Optional[TaskInfo] = None

    def __enter__(self) -> "Progress":
        """Enter context."""
        self._task = self.tracker.create_task(self._task_id, self.name, self.total)
        self.tracker.start_task(self._task_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context."""
        if exc_type is not None:
            self.tracker.fail_task(self._task_id, str(exc_val))
        else:
            self.tracker.complete_task(self._task_id)

    def update(self, increment: int = 1, message: str = "") -> None:
        """Update progress.

        Args:
            increment: Number of items completed.
            message: Optional status message.
        """
        self.tracker.update_task(self._task_id, increment=increment, message=message)

    def set_progress(self, current: int, message: str = "") -> None:
        """Set absolute progress.

        Args:
            current: Current progress value.
            message: Optional status message.
        """
        self.tracker.update_task(self._task_id, current=current, message=message)

    @property
    def current(self) -> int:
        """Get current progress."""
        return self._task.current if self._task else 0

    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        return self._task.percentage if self._task else 0.0


def track_progress(
    iterable: Iterator[T],
    total: Optional[int] = None,
    name: str = "Processing",
    config: Optional[ProgressConfig] = None,
) -> Iterator[T]:
    """Track progress of an iterable.

    Args:
        iterable: Items to iterate over.
        total: Total count (auto-detected if possible).
        name: Task name.
        config: Optional progress configuration.

    Yields:
        Items from the iterable.
    """
    # Try to get length
    if total is None:
        try:
            total = len(iterable)  # type: ignore
        except TypeError:
            total = 0

    with Progress(total, name, config=config) as progress:
        for item in iterable:
            yield item
            progress.update()


# =============================================================================
# Multi-Task Progress Reporter
# =============================================================================


class MultiTaskReporter:
    """Reporter for multiple concurrent tasks."""

    def __init__(self, config: Optional[ProgressConfig] = None):
        """Initialize reporter.

        Args:
            config: Optional progress configuration.
        """
        self.config = config or ProgressConfig()
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_task(self, task_id: str, name: str, total: int) -> None:
        """Add a task to track.

        Args:
            task_id: Unique task identifier.
            name: Task name.
            total: Total items.
        """
        with self._lock:
            self._tasks[task_id] = TaskInfo(
                id=task_id,
                name=name,
                total=total,
                status=ProgressStatus.PENDING,
            )

    def start_task(self, task_id: str) -> None:
        """Start a task."""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = ProgressStatus.RUNNING
                self._tasks[task_id].start_time = time.time()

    def update_task(self, task_id: str, current: int) -> None:
        """Update task progress."""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].current = current

    def complete_task(self, task_id: str) -> None:
        """Mark task as completed."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = ProgressStatus.COMPLETED
                task.current = task.total
                task.end_time = time.time()

    def start_reporting(self, interval: float = 0.5) -> None:
        """Start periodic reporting.

        Args:
            interval: Update interval in seconds.
        """
        self._running = True
        self._thread = threading.Thread(target=self._report_loop, args=(interval,), daemon=True)
        self._thread.start()

    def stop_reporting(self) -> None:
        """Stop periodic reporting."""
        self._running = False
        if self._thread:
            self._thread.join()

    def render(self) -> str:
        """Render current progress state.

        Returns:
            Rendered progress string.
        """
        with self._lock:
            lines = []
            for task in self._tasks.values():
                bar_width = 20
                filled = int(bar_width * task.percentage / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                status_icon = self._get_status_icon(task.status)
                lines.append(f"{status_icon} {task.name[:15]:<15} [{bar}] {task.percentage:5.1f}%")
            return "\n".join(lines)

    def _report_loop(self, interval: float) -> None:
        """Background reporting loop."""
        while self._running:
            output = self.render()
            # Clear previous lines and print
            num_lines = len(self._tasks)
            self.config.output.write(f"\033[{num_lines}A")  # Move up
            self.config.output.write("\033[J")  # Clear below
            self.config.output.write(output + "\n")
            self.config.output.flush()
            time.sleep(interval)

    def _get_status_icon(self, status: ProgressStatus) -> str:
        """Get icon for status."""
        icons = {
            ProgressStatus.PENDING: "⏸",
            ProgressStatus.RUNNING: "▶",
            ProgressStatus.COMPLETED: "✓",
            ProgressStatus.FAILED: "✗",
            ProgressStatus.CANCELLED: "⊘",
        }
        return icons.get(status, "?")


# =============================================================================
# Callback Progress Reporter
# =============================================================================


class CallbackReporter:
    """Progress reporter using callbacks."""

    def __init__(
        self,
        on_start: Optional[Callable[[str, int], None]] = None,
        on_update: Optional[Callable[[str, int, int, float], None]] = None,
        on_complete: Optional[Callable[[str, float], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ):
        """Initialize callback reporter.

        Args:
            on_start: Called when task starts (task_id, total).
            on_update: Called on progress (task_id, current, total, percentage).
            on_complete: Called on completion (task_id, elapsed_time).
            on_error: Called on error (task_id, exception).
        """
        self.on_start = on_start
        self.on_update = on_update
        self.on_complete = on_complete
        self.on_error = on_error
        self._start_times: Dict[str, float] = {}

    def start(self, task_id: str, total: int) -> None:
        """Report task start."""
        self._start_times[task_id] = time.time()
        if self.on_start:
            self.on_start(task_id, total)

    def update(self, task_id: str, current: int, total: int) -> None:
        """Report progress update."""
        if self.on_update:
            percentage = (current / total * 100) if total > 0 else 0
            self.on_update(task_id, current, total, percentage)

    def complete(self, task_id: str) -> None:
        """Report task completion."""
        if self.on_complete:
            start = self._start_times.get(task_id, time.time())
            elapsed = time.time() - start
            self.on_complete(task_id, elapsed)

    def error(self, task_id: str, exception: Exception) -> None:
        """Report task error."""
        if self.on_error:
            self.on_error(task_id, exception)


# =============================================================================
# Global Instance Management
# =============================================================================


_global_tracker: Optional[ProgressTracker] = None


def get_tracker() -> ProgressTracker:
    """Get the global progress tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProgressTracker()
    return _global_tracker


def reset_tracker() -> None:
    """Reset the global progress tracker."""
    global _global_tracker
    _global_tracker = None


def create_tracker(config: Optional[ProgressConfig] = None) -> ProgressTracker:
    """Create a new progress tracker."""
    return ProgressTracker(config)


def create_progress(
    total: int, name: str = "Processing", config: Optional[ProgressConfig] = None
) -> Progress:
    """Create a new progress context."""
    return Progress(total, name, config=config)


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"


def format_rate(items: int, seconds: float) -> str:
    """Format processing rate.

    Args:
        items: Number of items processed.
        seconds: Time taken.

    Returns:
        Formatted rate string.
    """
    if seconds == 0:
        return "∞/s"
    rate = items / seconds
    if rate >= 1:
        return f"{rate:.1f}/s"
    else:
        return f"{rate * 60:.1f}/min"
