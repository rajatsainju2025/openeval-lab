"""
Performance profiling decorators and utilities.

Provides decorators for measuring function execution time,
memory usage, and performance metrics.
"""

import functools
import time
import threading
from typing import Any, Callable, Dict
from collections import defaultdict

try:
    import tracemalloc

    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False
    tracemalloc = None  # type: ignore


class PerformanceMetrics:
    """Collect and report performance metrics."""

    def __init__(self):
        """Initialize performance metrics."""
        self._metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "calls": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "total_memory": 0,
            }
        )
        self._lock = threading.Lock()

    def record_call(
        self,
        name: str,
        duration: float,
        memory_delta: int = 0,
    ) -> None:
        """Record a function call.

        Args:
            name: Function name
            duration: Execution time in seconds
            memory_delta: Memory change in bytes
        """
        with self._lock:
            metric = self._metrics[name]
            metric["calls"] += 1
            metric["total_time"] += duration
            metric["min_time"] = min(metric["min_time"], duration)
            metric["max_time"] = max(metric["max_time"], duration)
            metric["total_memory"] += memory_delta

    def get_summary(self, name: str) -> Dict[str, Any]:
        """Get summary for a function.

        Args:
            name: Function name

        Returns:
            Dictionary with metrics summary
        """
        with self._lock:
            if name not in self._metrics:
                return {}

            metric = self._metrics[name]
            avg_time = metric["total_time"] / metric["calls"] if metric["calls"] > 0 else 0

            return {
                "calls": metric["calls"],
                "total_time_ms": metric["total_time"] * 1000,
                "avg_time_ms": avg_time * 1000,
                "min_time_ms": metric["min_time"] * 1000,
                "max_time_ms": metric["max_time"] * 1000,
                "total_memory_mb": metric["total_memory"] / (1024**2),
            }

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all tracked functions.

        Returns:
            Dictionary mapping function names to summaries
        """
        with self._lock:
            return {
                name: {
                    "calls": metric["calls"],
                    "total_time_ms": metric["total_time"] * 1000,
                    "avg_time_ms": (
                        metric["total_time"] / metric["calls"] * 1000 if metric["calls"] > 0 else 0
                    ),
                }
                for name, metric in self._metrics.items()
            }

    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()


# Global metrics instance
_metrics = PerformanceMetrics()


def profile_time(func: Callable) -> Callable:
    """Decorator to profile function execution time.

    Args:
        func: Function to profile

    Returns:
        Wrapped function
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            _metrics.record_call(func.__name__, elapsed)

    return wrapper


def profile_memory(func: Callable) -> Callable:
    """Decorator to profile function memory usage.

    Args:
        func: Function to profile

    Returns:
        Wrapped function
    """
    if not HAS_TRACEMALLOC:
        return func

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if HAS_TRACEMALLOC and tracemalloc is not None:
            tracemalloc.start()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            if HAS_TRACEMALLOC and tracemalloc is not None:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                _metrics.record_call(func.__name__, 0, memory_delta=peak)

    return wrapper


def profile_both(func: Callable) -> Callable:
    """Decorator to profile both time and memory.

    Args:
        func: Function to profile

    Returns:
        Wrapped function
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        if HAS_TRACEMALLOC and tracemalloc is not None:
            tracemalloc.start()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            memory_delta = 0

            if HAS_TRACEMALLOC and tracemalloc is not None:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_delta = peak

            _metrics.record_call(func.__name__, elapsed, memory_delta)

    return wrapper


class ProfileBlock:
    """Context manager for profiling code blocks."""

    def __init__(self, name: str = "block"):
        """Initialize profile block.

        Args:
            name: Name for this block
        """
        self.name = name
        self.start_time = 0.0

    def __enter__(self) -> "ProfileBlock":
        """Enter context."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context."""
        elapsed = time.perf_counter() - self.start_time
        _metrics.record_call(self.name, elapsed)


def get_metrics(func_name: str) -> Dict[str, Any]:
    """Get metrics for a function.

    Args:
        func_name: Function name

    Returns:
        Metrics dictionary
    """
    return _metrics.get_summary(func_name)


def get_all_metrics() -> Dict[str, Dict[str, Any]]:
    """Get all collected metrics.

    Returns:
        Dictionary of all metrics
    """
    return _metrics.get_all_summaries()


def clear_metrics() -> None:
    """Clear all metrics."""
    _metrics.clear()


def print_metrics(top_n: int = 10) -> None:
    """Print metrics for top N slowest functions.

    Args:
        top_n: Number of top functions to show
    """
    summaries = _metrics.get_all_summaries()

    # Sort by total time
    sorted_funcs = sorted(
        summaries.items(),
        key=lambda x: x[1]["total_time_ms"],
        reverse=True,
    )

    print(f"\n📊 Performance Metrics (Top {top_n}):\n")
    print(f"{'Function':<30} {'Calls':>8} {'Total':>12} {'Avg':>12}")
    print("-" * 65)

    for func_name, metrics in sorted_funcs[:top_n]:
        print(
            f"{func_name:<30} {metrics['calls']:>8} "
            f"{metrics['total_time_ms']:>10.1f}ms {metrics['avg_time_ms']:>10.1f}ms"
        )
