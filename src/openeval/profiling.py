"""Profiling utilities for OpenEval.

Consolidates profiling_decorators.py and performance_benchmarks.py.

Example:
    >>> from openeval.profiling import benchmark, profile_time
    >>>
    >>> # Quick benchmark
    >>> with benchmark("data_loading"):
    ...     data = load_large_dataset()
    >>> # Prints: data_loading completed in 1.234s
    >>>
    >>> # Profile a function
    >>> @profile_time
    ... def expensive_operation():
    ...     return compute_metrics()
"""

from .profiling_decorators import (
    profile_time,
    profile_memory,
    profile_calls,
)
import time
from contextlib import contextmanager


@contextmanager
def benchmark(name: str, verbose: bool = True):
    """Context manager for quick benchmarking of code blocks.

    Args:
        name: Name to identify this benchmark
        verbose: Whether to print timing information (default: True)

    Yields:
        dict with timing information

    Example:
        >>> with benchmark("matrix_multiply") as timer:
        ...     result = np.dot(matrix_a, matrix_b)
        >>> print(f"Took {timer['elapsed']:.3f} seconds")
    """
    start_time = time.perf_counter()
    timer_info = {"start": start_time, "elapsed": 0.0}

    try:
        yield timer_info
    finally:
        elapsed = time.perf_counter() - start_time
        timer_info["elapsed"] = elapsed

        if verbose:
            print(f"⏱️  {name} completed in {elapsed:.3f}s")


def compare_performance(func1, func2, *args, iterations: int = 100, **kwargs):
    """Compare the performance of two functions.

    Args:
        func1: First function to benchmark
        func2: Second function to benchmark
        *args: Positional arguments to pass to both functions
        iterations: Number of iterations to run (default: 100)
        **kwargs: Keyword arguments to pass to both functions

    Returns:
        Dictionary with comparison results

    Example:
        >>> results = compare_performance(
        ...     old_implementation,
        ...     new_implementation,
        ...     test_data,
        ...     iterations=1000
        ... )
        >>> print(f"Speedup: {results['speedup']:.2f}x")
    """
    # Benchmark func1
    start1 = time.perf_counter()
    for _ in range(iterations):
        func1(*args, **kwargs)
    elapsed1 = time.perf_counter() - start1

    # Benchmark func2
    start2 = time.perf_counter()
    for _ in range(iterations):
        func2(*args, **kwargs)
    elapsed2 = time.perf_counter() - start2

    avg1 = elapsed1 / iterations
    avg2 = elapsed2 / iterations
    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else float("inf")

    return {
        "func1_name": func1.__name__,
        "func2_name": func2.__name__,
        "func1_avg_ms": avg1 * 1000,
        "func2_avg_ms": avg2 * 1000,
        "speedup": speedup,
        "faster": func1.__name__ if elapsed1 < elapsed2 else func2.__name__,
    }


__all__ = ["profile_time", "profile_memory", "profile_calls", "benchmark", "compare_performance"]
