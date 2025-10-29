"""Performance profiling utilities for OpenEval."""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from rich.console import Console

console = Console()


def profile_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure and log execution time of a function.

    Args:
        func: Function to profile

    Returns:
        Wrapped function that logs execution time

    Example:
        >>> @profile_time
        ... def slow_function():
        ...     time.sleep(1)
        ...     return "done"
        >>> result = slow_function()
        slow_function took 1.00s
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            console.print(f"[dim]{func.__name__} took {elapsed:.2f}s[/dim]")

    return wrapper


@contextmanager
def profile_block(name: str) -> Generator[None, None, None]:
    """Context manager to measure execution time of a code block.

    Args:
        name: Name of the code block for logging

    Yields:
        None

    Example:
        >>> with profile_block("data loading"):
        ...     data = load_large_dataset()
        data loading took 2.35s
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        console.print(f"[dim]{name} took {elapsed:.2f}s[/dim]")


class PerformanceTimer:
    """Simple timer for measuring performance of operations.

    Example:
        >>> timer = PerformanceTimer()
        >>> timer.start("loading")
        >>> data = load_data()
        >>> timer.stop("loading")
        >>> timer.start("processing")
        >>> process(data)
        >>> timer.stop("processing")
        >>> timer.report()
        Performance Report:
        loading: 1.23s
        processing: 0.45s
        Total: 1.68s
    """

    def __init__(self) -> None:
        """Initialize performance timer."""
        self.timings: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    def start(self, name: str) -> None:
        """Start timing an operation.

        Args:
            name: Name of the operation
        """
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        """Stop timing an operation.

        Args:
            name: Name of the operation

        Raises:
            ValueError: If operation was not started
        """
        if name not in self._starts:
            raise ValueError(f"Timer '{name}' was not started")
        elapsed = time.perf_counter() - self._starts[name]
        self.timings[name] = self.timings.get(name, 0.0) + elapsed
        del self._starts[name]

    def report(self) -> None:
        """Print a formatted report of all timings."""
        console.print("\n[bold]Performance Report:[/bold]")
        for name, elapsed in sorted(self.timings.items()):
            console.print(f"  {name}: [cyan]{elapsed:.2f}s[/cyan]")
        total = sum(self.timings.values())
        console.print(f"  [bold]Total: [cyan]{total:.2f}s[/cyan][/bold]\n")

    def reset(self) -> None:
        """Reset all timings."""
        self.timings.clear()
        self._starts.clear()
