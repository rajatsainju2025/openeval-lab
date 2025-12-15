"""
Benchmark Runner for explanation performance testing.

This module provides tools for benchmarking code explanation performance,
comparing explainer implementations, and detecting regressions.
"""

from __future__ import annotations

import gc
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TypeVar
from uuid import uuid4

T = TypeVar("T")


class BenchmarkStatus(Enum):
    """Status of a benchmark run."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


class MetricType(Enum):
    """Types of metrics collected."""

    LATENCY = auto()  # Time in seconds
    THROUGHPUT = auto()  # Operations per second
    MEMORY = auto()  # Memory in bytes
    TOKEN_COUNT = auto()  # Number of tokens
    QUALITY_SCORE = auto()  # Quality score 0-1
    ERROR_RATE = auto()  # Error rate 0-1


@dataclass
class MetricResult:
    """Result of a single metric measurement."""

    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""

    iteration: int
    latency_seconds: float
    memory_bytes: int
    success: bool
    error: Optional[str] = None
    output_size: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "latency_seconds": self.latency_seconds,
            "memory_bytes": self.memory_bytes,
            "success": self.success,
            "error": self.error,
            "output_size": self.output_size,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BenchmarkStatistics:
    """Statistics for a benchmark run."""

    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    p50: float
    p90: float
    p95: float
    p99: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "min": self.min_value,
            "max": self.max_value,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
        }

    @staticmethod
    def from_values(values: List[float]) -> "BenchmarkStatistics":
        """Create statistics from a list of values."""
        if not values:
            return BenchmarkStatistics(
                count=0,
                mean=0,
                median=0,
                std_dev=0,
                min_value=0,
                max_value=0,
                p50=0,
                p90=0,
                p95=0,
                p99=0,
            )

        sorted_values = sorted(values)
        count = len(sorted_values)

        def percentile(p: float) -> float:
            idx = int(count * p / 100)
            return sorted_values[min(idx, count - 1)]

        return BenchmarkStatistics(
            count=count,
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if count > 1 else 0,
            min_value=min(values),
            max_value=max(values),
            p50=percentile(50),
            p90=percentile(90),
            p95=percentile(95),
            p99=percentile(99),
        )


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    description: str = ""
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    timeout_seconds: float = 30.0
    collect_memory: bool = True
    gc_between_iterations: bool = True
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "warmup_iterations": self.warmup_iterations,
            "benchmark_iterations": self.benchmark_iterations,
            "timeout_seconds": self.timeout_seconds,
            "collect_memory": self.collect_memory,
            "gc_between_iterations": self.gc_between_iterations,
            "tags": self.tags,
        }


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    id: str
    config: BenchmarkConfig
    status: BenchmarkStatus
    samples: List[BenchmarkSample]
    latency_stats: BenchmarkStatistics
    memory_stats: BenchmarkStatistics
    success_rate: float
    total_duration_seconds: float
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "config": self.config.to_dict(),
            "status": self.status.name,
            "samples": [s.to_dict() for s in self.samples],
            "latency_stats": self.latency_stats.to_dict(),
            "memory_stats": self.memory_stats.to_dict(),
            "success_rate": self.success_rate,
            "total_duration_seconds": self.total_duration_seconds,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark results."""

    baseline_id: str
    comparison_id: str
    baseline_name: str
    comparison_name: str
    latency_change_percent: float
    memory_change_percent: float
    success_rate_change: float
    is_regression: bool
    regression_threshold: float
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_id": self.baseline_id,
            "comparison_id": self.comparison_id,
            "baseline_name": self.baseline_name,
            "comparison_name": self.comparison_name,
            "latency_change_percent": self.latency_change_percent,
            "memory_change_percent": self.memory_change_percent,
            "success_rate_change": self.success_rate_change,
            "is_regression": self.is_regression,
            "regression_threshold": self.regression_threshold,
            "summary": self.summary,
            "details": self.details,
        }


@dataclass
class BenchmarkSuite:
    """A collection of benchmark configurations."""

    name: str
    description: str
    benchmarks: List[BenchmarkConfig]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SuiteResult:
    """Result of running a benchmark suite."""

    suite_name: str
    results: List[BenchmarkResult]
    total_duration_seconds: float
    passed: int
    failed: int
    skipped: int
    started_at: datetime
    completed_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_name": self.suite_name,
            "results": [r.to_dict() for r in self.results],
            "total_duration_seconds": self.total_duration_seconds,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
        }


class BenchmarkStorage(ABC):
    """Abstract storage for benchmark results."""

    @abstractmethod
    def save_result(self, result: BenchmarkResult) -> None:
        """Save a benchmark result."""
        pass

    @abstractmethod
    def get_result(self, result_id: str) -> Optional[BenchmarkResult]:
        """Get a result by ID."""
        pass

    @abstractmethod
    def list_results(
        self,
        benchmark_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[BenchmarkResult]:
        """List benchmark results."""
        pass

    @abstractmethod
    def get_baseline(self, benchmark_name: str) -> Optional[BenchmarkResult]:
        """Get the baseline result for a benchmark."""
        pass

    @abstractmethod
    def set_baseline(self, benchmark_name: str, result_id: str) -> None:
        """Set a result as the baseline for a benchmark."""
        pass


class InMemoryBenchmarkStorage(BenchmarkStorage):
    """In-memory storage for benchmark results."""

    def __init__(self):
        self.results: Dict[str, BenchmarkResult] = {}
        self.baselines: Dict[str, str] = {}

    def save_result(self, result: BenchmarkResult) -> None:
        """Save a benchmark result."""
        self.results[result.id] = result

    def get_result(self, result_id: str) -> Optional[BenchmarkResult]:
        """Get a result by ID."""
        return self.results.get(result_id)

    def list_results(
        self,
        benchmark_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[BenchmarkResult]:
        """List benchmark results."""
        results = list(self.results.values())

        if benchmark_name:
            results = [r for r in results if r.config.name == benchmark_name]

        results.sort(key=lambda r: r.started_at, reverse=True)
        return results[:limit]

    def get_baseline(self, benchmark_name: str) -> Optional[BenchmarkResult]:
        """Get the baseline result for a benchmark."""
        baseline_id = self.baselines.get(benchmark_name)
        if baseline_id:
            return self.results.get(baseline_id)
        return None

    def set_baseline(self, benchmark_name: str, result_id: str) -> None:
        """Set a result as the baseline for a benchmark."""
        self.baselines[benchmark_name] = result_id


class BenchmarkRunner:
    """Main class for running benchmarks."""

    def __init__(
        self,
        storage: Optional[BenchmarkStorage] = None,
        regression_threshold: float = 0.1,  # 10% regression threshold
    ):
        """Initialize the benchmark runner."""
        self.storage = storage or InMemoryBenchmarkStorage()
        self.regression_threshold = regression_threshold
        self._event_handlers: Dict[str, List[Callable]] = {}

    def run(
        self,
        config: BenchmarkConfig,
        target: Callable[[], Any],
        setup: Optional[Callable[[], None]] = None,
        teardown: Optional[Callable[[], None]] = None,
    ) -> BenchmarkResult:
        """Run a benchmark."""
        result_id = str(uuid4())
        started_at = datetime.now()
        samples: List[BenchmarkSample] = []

        self._emit_event("benchmark_started", config)

        try:
            # Setup
            if setup:
                setup()

            # Warmup
            for i in range(config.warmup_iterations):
                self._run_iteration(target, config)
                self._emit_event("warmup_iteration", i + 1, config.warmup_iterations)

            # Benchmark iterations
            for i in range(config.benchmark_iterations):
                if config.gc_between_iterations:
                    gc.collect()

                sample = self._run_iteration(target, config, iteration=i)
                samples.append(sample)
                self._emit_event("benchmark_iteration", i + 1, config.benchmark_iterations, sample)

            # Teardown
            if teardown:
                teardown()

            # Calculate statistics
            latencies = [s.latency_seconds for s in samples if s.success]
            memories = [s.memory_bytes for s in samples if s.success]
            success_count = sum(1 for s in samples if s.success)

            completed_at = datetime.now()

            result = BenchmarkResult(
                id=result_id,
                config=config,
                status=BenchmarkStatus.COMPLETED,
                samples=samples,
                latency_stats=BenchmarkStatistics.from_values(latencies),
                memory_stats=BenchmarkStatistics.from_values([float(m) for m in memories]),
                success_rate=success_count / len(samples) if samples else 0,
                total_duration_seconds=(completed_at - started_at).total_seconds(),
                started_at=started_at,
                completed_at=completed_at,
            )

            self.storage.save_result(result)
            self._emit_event("benchmark_completed", result)
            return result

        except Exception as e:
            completed_at = datetime.now()
            result = BenchmarkResult(
                id=result_id,
                config=config,
                status=BenchmarkStatus.FAILED,
                samples=samples,
                latency_stats=BenchmarkStatistics.from_values([]),
                memory_stats=BenchmarkStatistics.from_values([]),
                success_rate=0,
                total_duration_seconds=(completed_at - started_at).total_seconds(),
                started_at=started_at,
                completed_at=completed_at,
                error=str(e),
            )

            self.storage.save_result(result)
            self._emit_event("benchmark_failed", result, e)
            return result

    def _run_iteration(
        self,
        target: Callable[[], Any],
        config: BenchmarkConfig,
        iteration: int = 0,
    ) -> BenchmarkSample:
        """Run a single iteration."""
        import sys

        # Get initial memory (approximate)
        initial_memory = 0
        if config.collect_memory:
            try:
                import tracemalloc

                tracemalloc.start()
                initial_memory = tracemalloc.get_traced_memory()[0]
            except ImportError:
                pass

        start_time = time.perf_counter()
        success = True
        error = None
        output_size = 0

        try:
            result = target()
            if result is not None:
                output_size = sys.getsizeof(result)
        except Exception as e:
            success = False
            error = f"{type(e).__name__}: {str(e)}"

        end_time = time.perf_counter()
        latency = end_time - start_time

        # Get final memory
        final_memory = 0
        if config.collect_memory:
            try:
                import tracemalloc

                final_memory = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
            except (ImportError, RuntimeError):
                pass

        return BenchmarkSample(
            iteration=iteration,
            latency_seconds=latency,
            memory_bytes=final_memory - initial_memory,
            success=success,
            error=error,
            output_size=output_size,
        )

    def run_suite(
        self,
        suite: BenchmarkSuite,
        targets: Dict[str, Callable[[], Any]],
    ) -> SuiteResult:
        """Run a benchmark suite."""
        started_at = datetime.now()
        results: List[BenchmarkResult] = []
        passed = 0
        failed = 0
        skipped = 0

        self._emit_event("suite_started", suite)

        for config in suite.benchmarks:
            if config.name not in targets:
                skipped += 1
                continue

            result = self.run(config, targets[config.name])
            results.append(result)

            if result.status == BenchmarkStatus.COMPLETED:
                passed += 1
            else:
                failed += 1

        completed_at = datetime.now()

        suite_result = SuiteResult(
            suite_name=suite.name,
            results=results,
            total_duration_seconds=(completed_at - started_at).total_seconds(),
            passed=passed,
            failed=failed,
            skipped=skipped,
            started_at=started_at,
            completed_at=completed_at,
        )

        self._emit_event("suite_completed", suite_result)
        return suite_result

    def compare(
        self,
        baseline_id: str,
        comparison_id: str,
    ) -> ComparisonResult:
        """Compare two benchmark results."""
        baseline = self.storage.get_result(baseline_id)
        comparison = self.storage.get_result(comparison_id)

        if not baseline or not comparison:
            raise ValueError("Baseline or comparison result not found")

        # Calculate changes
        latency_change = 0.0
        if baseline.latency_stats.mean > 0:
            latency_change = (
                (comparison.latency_stats.mean - baseline.latency_stats.mean)
                / baseline.latency_stats.mean
            ) * 100

        memory_change = 0.0
        if baseline.memory_stats.mean > 0:
            memory_change = (
                (comparison.memory_stats.mean - baseline.memory_stats.mean)
                / baseline.memory_stats.mean
            ) * 100

        success_rate_change = comparison.success_rate - baseline.success_rate

        # Determine if this is a regression
        is_regression = (
            latency_change > self.regression_threshold * 100
            or memory_change > self.regression_threshold * 100
            or success_rate_change < -0.05  # 5% success rate drop
        )

        # Generate summary
        summary_parts = []
        if latency_change > 0:
            summary_parts.append(f"Latency increased by {latency_change:.1f}%")
        elif latency_change < 0:
            summary_parts.append(f"Latency decreased by {abs(latency_change):.1f}%")

        if memory_change > 0:
            summary_parts.append(f"Memory increased by {memory_change:.1f}%")
        elif memory_change < 0:
            summary_parts.append(f"Memory decreased by {abs(memory_change):.1f}%")

        if is_regression:
            summary_parts.append("REGRESSION DETECTED")

        summary = ". ".join(summary_parts) or "No significant changes"

        return ComparisonResult(
            baseline_id=baseline_id,
            comparison_id=comparison_id,
            baseline_name=baseline.config.name,
            comparison_name=comparison.config.name,
            latency_change_percent=latency_change,
            memory_change_percent=memory_change,
            success_rate_change=success_rate_change,
            is_regression=is_regression,
            regression_threshold=self.regression_threshold,
            summary=summary,
            details={
                "baseline_latency_mean": baseline.latency_stats.mean,
                "comparison_latency_mean": comparison.latency_stats.mean,
                "baseline_memory_mean": baseline.memory_stats.mean,
                "comparison_memory_mean": comparison.memory_stats.mean,
                "baseline_success_rate": baseline.success_rate,
                "comparison_success_rate": comparison.success_rate,
            },
        )

    def compare_to_baseline(self, result_id: str) -> Optional[ComparisonResult]:
        """Compare a result to its benchmark's baseline."""
        result = self.storage.get_result(result_id)
        if not result:
            return None

        baseline = self.storage.get_baseline(result.config.name)
        if not baseline:
            return None

        return self.compare(baseline.id, result_id)

    def set_baseline(self, result_id: str) -> None:
        """Set a result as the baseline for its benchmark."""
        result = self.storage.get_result(result_id)
        if result:
            self.storage.set_baseline(result.config.name, result_id)

    def get_history(
        self,
        benchmark_name: str,
        limit: int = 20,
    ) -> List[BenchmarkResult]:
        """Get benchmark history."""
        return self.storage.list_results(benchmark_name, limit)

    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        """Unregister an event handler."""
        if event in self._event_handlers:
            self._event_handlers[event] = [h for h in self._event_handlers[event] if h != handler]

    def _emit_event(self, event: str, *args, **kwargs) -> None:
        """Emit an event to all registered handlers."""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    handler(*args, **kwargs)
                except Exception:
                    pass


# Global benchmark runner instance
_global_runner: Optional[BenchmarkRunner] = None


def get_benchmark_runner() -> BenchmarkRunner:
    """Get the global benchmark runner instance."""
    global _global_runner
    if _global_runner is None:
        _global_runner = BenchmarkRunner()
    return _global_runner


def reset_benchmark_runner() -> None:
    """Reset the global benchmark runner instance."""
    global _global_runner
    _global_runner = None


def create_benchmark_runner(
    storage: Optional[BenchmarkStorage] = None,
    regression_threshold: float = 0.1,
) -> BenchmarkRunner:
    """Create a new benchmark runner instance."""
    return BenchmarkRunner(storage=storage, regression_threshold=regression_threshold)


# Convenience functions
def benchmark(
    name: str,
    target: Callable[[], Any],
    iterations: int = 10,
    warmup: int = 3,
    **kwargs,
) -> BenchmarkResult:
    """Run a quick benchmark."""
    config = BenchmarkConfig(
        name=name,
        benchmark_iterations=iterations,
        warmup_iterations=warmup,
        **kwargs,
    )
    runner = get_benchmark_runner()
    return runner.run(config, target)


def compare_benchmarks(
    baseline_id: str,
    comparison_id: str,
) -> ComparisonResult:
    """Compare two benchmark results."""
    runner = get_benchmark_runner()
    return runner.compare(baseline_id, comparison_id)


def get_benchmark_history(
    benchmark_name: str,
    limit: int = 20,
) -> List[BenchmarkResult]:
    """Get benchmark history."""
    runner = get_benchmark_runner()
    return runner.get_history(benchmark_name, limit)


def check_regression(result_id: str) -> Optional[ComparisonResult]:
    """Check if a benchmark result shows regression compared to baseline."""
    runner = get_benchmark_runner()
    return runner.compare_to_baseline(result_id)


def create_benchmark_config(
    name: str,
    description: str = "",
    iterations: int = 10,
    warmup: int = 3,
    timeout: float = 30.0,
    **kwargs,
) -> BenchmarkConfig:
    """Create a benchmark configuration."""
    return BenchmarkConfig(
        name=name,
        description=description,
        benchmark_iterations=iterations,
        warmup_iterations=warmup,
        timeout_seconds=timeout,
        **kwargs,
    )


def create_benchmark_suite(
    name: str,
    description: str = "",
    benchmarks: Optional[List[BenchmarkConfig]] = None,
) -> BenchmarkSuite:
    """Create a benchmark suite."""
    return BenchmarkSuite(
        name=name,
        description=description,
        benchmarks=benchmarks or [],
    )
