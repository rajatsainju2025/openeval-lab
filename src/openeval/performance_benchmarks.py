"""
Performance regression tests to ensure optimizations persist.

Provides benchmarks for key operations to detect performance regressions
in CI/CD pipeline.
"""

import time
from typing import Callable, Dict, Any


class PerformanceBenchmark:
    """Benchmark performance of operations."""

    def __init__(self, name: str, expected_time_ms: float):
        """Initialize benchmark.

        Args:
            name: Benchmark name
            expected_time_ms: Expected execution time in ms
        """
        self.name = name
        self.expected_time_ms = expected_time_ms
        self.results: Dict[str, Any] = {}

    def run(self, func: Callable, *args: Any, **kwargs: Any) -> float:
        """Run benchmark.

        Args:
            func: Function to benchmark
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Execution time in milliseconds
        """
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000

        self.results = {
            "elapsed_ms": elapsed,
            "expected_ms": self.expected_time_ms,
            "regression": elapsed > self.expected_time_ms * 1.2,  # 20% tolerance
            "speedup": self.expected_time_ms / elapsed if elapsed > 0 else 0,
        }

        return elapsed

    def check_regression(self) -> bool:
        """Check if performance regressed.

        Returns:
            True if regression detected
        """
        return self.results.get("regression", False)

    def report(self) -> str:
        """Generate benchmark report.

        Returns:
            Formatted report string
        """
        regression_status = "❌ REGRESSED" if self.check_regression() else "✅ PASS"

        return (
            f"{self.name}: {regression_status}\n"
            f"  Expected: {self.expected_time_ms:.1f}ms\n"
            f"  Actual: {self.results.get('elapsed_ms', 0):.1f}ms\n"
            f"  Speedup: {self.results.get('speedup', 0):.1f}x"
        )


class PerformanceSuite:
    """Suite of performance benchmarks."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.results: Dict[str, Dict] = {}

    def add_benchmark(
        self,
        name: str,
        func: Callable,
        expected_time_ms: float,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add benchmark to suite.

        Args:
            name: Benchmark name
            func: Function to benchmark
            expected_time_ms: Expected execution time
            *args: Function arguments
            **kwargs: Function keyword arguments
        """
        benchmark = PerformanceBenchmark(name, expected_time_ms)
        benchmark.run(func, *args, **kwargs)
        self.benchmarks[name] = benchmark
        self.results[name] = benchmark.results

    def run_all(self) -> bool:
        """Run all benchmarks.

        Returns:
            True if all passed, False if any regressed
        """
        passed = 0
        failed = 0

        for name, benchmark in self.benchmarks.items():
            if benchmark.check_regression():
                failed += 1
                print(f"❌ {benchmark.report()}")
            else:
                passed += 1
                print(f"✅ {benchmark.report()}")

        print(f"\n📊 Results: {passed} passed, {failed} failed")
        return failed == 0

    def get_summary(self) -> Dict[str, Dict]:
        """Get summary of all benchmarks.

        Returns:
            Dictionary of benchmark results
        """
        return self.results.copy()


# Common benchmark thresholds
BENCHMARK_THRESHOLDS = {
    "cache_get": 1.0,  # 1ms
    "cache_set": 1.5,  # 1.5ms
    "cache_batch_get_100": 5.0,  # 5ms for 100 items
    "cache_batch_set_100": 8.0,  # 8ms for 100 items
    "string_build_large": 10.0,  # 10ms for large report
    "validation_cache_hit": 0.5,  # 0.5ms
    "dataset_stream_1000": 5.0,  # 5ms for 1000 items
    "metrics_calculation": 2.0,  # 2ms
}


def create_benchmark_suite() -> PerformanceSuite:
    """Create standard benchmark suite.

    Returns:
        PerformanceSuite instance
    """
    suite = PerformanceSuite()
    return suite
