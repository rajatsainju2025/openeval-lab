"""Performance profiler for explanation generation.

This module provides comprehensive profiling capabilities for measuring
and optimizing explanation generation performance.

Example:
    >>> from openeval.explainers import PerformanceProfiler, profile_explanation
    >>> profiler = get_performance_profiler()
    >>> with profiler.profile("my_operation"):
    ...     result = generate_explanation(code)
    >>> report = profiler.generate_report()
"""

from __future__ import annotations

import cProfile
import functools
import gc
import io
import pstats
import sys
import threading
import time
import tracemalloc
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generator, TypeVar


T = TypeVar("T")


class ProfileLevel(Enum):
    """Profiling detail levels."""

    MINIMAL = "minimal"
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class MetricCategory(Enum):
    """Categories of performance metrics."""

    TIME = "time"
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    CUSTOM = "custom"


@dataclass
class TimingMetric:
    """A timing measurement."""

    name: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def stop(self) -> None:
        """Stop the timing."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


@dataclass
class MemoryMetric:
    """A memory measurement."""

    name: str
    start_memory: int
    peak_memory: int = 0
    end_memory: int = 0
    allocated: int = 0
    freed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileSample:
    """A single profile sample."""

    timestamp: datetime
    operation: str
    timing: TimingMetric | None
    memory: MemoryMetric | None
    call_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.timing:
            return self.timing.duration * 1000
        return 0.0

    @property
    def memory_delta_kb(self) -> float:
        """Get memory change in KB."""
        if self.memory:
            return (self.memory.end_memory - self.memory.start_memory) / 1024
        return 0.0


@dataclass
class ProfileStatistics:
    """Statistics for profiled operations."""

    operation: str
    count: int
    total_time: float
    min_time: float
    max_time: float
    avg_time: float
    std_dev: float
    total_memory: int
    peak_memory: int
    avg_memory: int
    percentiles: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_samples(cls, operation: str, samples: list[ProfileSample]) -> ProfileStatistics:
        """Create statistics from samples."""
        if not samples:
            return cls(
                operation=operation,
                count=0,
                total_time=0,
                min_time=0,
                max_time=0,
                avg_time=0,
                std_dev=0,
                total_memory=0,
                peak_memory=0,
                avg_memory=0,
            )

        times = [s.duration_ms for s in samples if s.timing]
        memories = [s.memory.end_memory - s.memory.start_memory for s in samples if s.memory]

        total_time = sum(times)
        avg_time = total_time / len(times) if times else 0
        variance = sum((t - avg_time) ** 2 for t in times) / len(times) if times else 0

        sorted_times = sorted(times)
        percentiles = {}
        if sorted_times:
            percentiles["p50"] = sorted_times[len(sorted_times) // 2]
            percentiles["p90"] = sorted_times[int(len(sorted_times) * 0.9)]
            percentiles["p99"] = sorted_times[int(len(sorted_times) * 0.99)]

        return cls(
            operation=operation,
            count=len(samples),
            total_time=total_time,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            avg_time=avg_time,
            std_dev=variance**0.5,
            total_memory=sum(memories),
            peak_memory=max(memories) if memories else 0,
            avg_memory=int(sum(memories) / len(memories)) if memories else 0,
            percentiles=percentiles,
        )


@dataclass
class ProfileReport:
    """A complete profiling report."""

    name: str
    start_time: datetime
    end_time: datetime
    statistics: dict[str, ProfileStatistics]
    samples: list[ProfileSample]
    total_operations: int
    total_duration: float
    hotspots: list[str]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_operations": self.total_operations,
            "total_duration_ms": self.total_duration,
            "statistics": {
                k: {
                    "count": v.count,
                    "total_time_ms": v.total_time,
                    "avg_time_ms": v.avg_time,
                    "min_time_ms": v.min_time,
                    "max_time_ms": v.max_time,
                    "percentiles": v.percentiles,
                }
                for k, v in self.statistics.items()
            },
            "hotspots": self.hotspots,
            "recommendations": self.recommendations,
        }

    def format_text(self) -> str:
        """Format as text report."""
        lines = [
            f"Performance Profile Report: {self.name}",
            "=" * 60,
            f"Duration: {self.total_duration:.2f}ms",
            f"Total Operations: {self.total_operations}",
            "",
            "Operation Statistics:",
            "-" * 60,
        ]

        for op, stats in sorted(
            self.statistics.items(), key=lambda x: x[1].total_time, reverse=True
        ):
            lines.append(f"\n{op}:")
            lines.append(f"  Count: {stats.count}")
            lines.append(f"  Total: {stats.total_time:.2f}ms")
            lines.append(f"  Avg: {stats.avg_time:.2f}ms")
            lines.append(f"  Min: {stats.min_time:.2f}ms")
            lines.append(f"  Max: {stats.max_time:.2f}ms")
            if stats.percentiles:
                lines.append(f"  P50: {stats.percentiles.get('p50', 0):.2f}ms")
                lines.append(f"  P99: {stats.percentiles.get('p99', 0):.2f}ms")

        if self.hotspots:
            lines.append("\n" + "-" * 60)
            lines.append("Hotspots:")
            for hotspot in self.hotspots:
                lines.append(f"  - {hotspot}")

        if self.recommendations:
            lines.append("\n" + "-" * 60)
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class ProfilerBackend(ABC):
    """Abstract base class for profiler backends."""

    @abstractmethod
    def start(self) -> None:
        """Start profiling."""
        pass

    @abstractmethod
    def stop(self) -> dict[str, Any]:
        """Stop profiling and return results."""
        pass


class CProfileBackend(ProfilerBackend):
    """cProfile-based profiler backend."""

    def __init__(self) -> None:
        """Initialize cProfile backend."""
        self._profiler: cProfile.Profile | None = None

    def start(self) -> None:
        """Start cProfile."""
        self._profiler = cProfile.Profile()
        self._profiler.enable()

    def stop(self) -> dict[str, Any]:
        """Stop and get stats."""
        if self._profiler is None:
            return {}

        self._profiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)

        return {
            "profile_output": stream.getvalue(),
            "total_calls": (
                self._profiler.getstats()[0].callcount if self._profiler.getstats() else 0
            ),
        }


class MemoryProfilerBackend(ProfilerBackend):
    """Memory profiler backend using tracemalloc."""

    def __init__(self) -> None:
        """Initialize memory profiler."""
        self._snapshot_start: tracemalloc.Snapshot | None = None

    def start(self) -> None:
        """Start memory tracking."""
        tracemalloc.start()
        self._snapshot_start = tracemalloc.take_snapshot()

    def stop(self) -> dict[str, Any]:
        """Stop and get memory stats."""
        if self._snapshot_start is None:
            return {}

        snapshot_end = tracemalloc.take_snapshot()
        top_stats = snapshot_end.compare_to(self._snapshot_start, "lineno")

        tracemalloc.stop()

        return {
            "top_allocations": [
                {"file": str(stat.traceback), "size": stat.size, "count": stat.count}
                for stat in top_stats[:10]
            ],
            "current_memory": tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0,
        }


class PerformanceProfiler:
    """Main performance profiler class."""

    def __init__(
        self,
        level: ProfileLevel = ProfileLevel.BASIC,
        enable_memory: bool = True,
    ) -> None:
        """Initialize the profiler.

        Args:
            level: Profiling detail level.
            enable_memory: Whether to track memory.
        """
        self.level = level
        self.enable_memory = enable_memory
        self._samples: list[ProfileSample] = []
        self._active_timings: dict[str, TimingMetric] = {}
        self._lock = threading.Lock()
        self._start_time: datetime | None = None
        self._backends: list[ProfilerBackend] = []

        if level == ProfileLevel.COMPREHENSIVE:
            self._backends.append(CProfileBackend())
        if enable_memory and level in [ProfileLevel.DETAILED, ProfileLevel.COMPREHENSIVE]:
            self._backends.append(MemoryProfilerBackend())

    def start(self) -> None:
        """Start the profiler."""
        self._start_time = datetime.now()
        self._samples.clear()

        for backend in self._backends:
            backend.start()

    def stop(self) -> ProfileReport:
        """Stop the profiler and generate report.

        Returns:
            ProfileReport with collected data.
        """
        end_time = datetime.now()
        backend_results = {}

        for backend in self._backends:
            results = backend.stop()
            backend_results[backend.__class__.__name__] = results

        return self.generate_report(
            end_time=end_time,
            backend_results=backend_results,
        )

    @contextmanager
    def profile(
        self,
        operation: str,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[ProfileSample, None, None]:
        """Context manager for profiling an operation.

        Args:
            operation: Name of the operation.
            metadata: Optional metadata.

        Yields:
            ProfileSample being recorded.
        """
        sample = self._start_sample(operation, metadata)
        try:
            yield sample
        finally:
            self._end_sample(sample)

    def _start_sample(
        self,
        operation: str,
        metadata: dict[str, Any] | None = None,
    ) -> ProfileSample:
        """Start a new sample."""
        timing = TimingMetric(
            name=operation,
            start_time=time.perf_counter(),
        )

        memory = None
        if self.enable_memory:
            gc.collect()
            memory = MemoryMetric(
                name=operation,
                start_memory=self._get_memory_usage(),
            )

        sample = ProfileSample(
            timestamp=datetime.now(),
            operation=operation,
            timing=timing,
            memory=memory,
            metadata=metadata or {},
        )

        with self._lock:
            self._active_timings[operation] = timing

        return sample

    def _end_sample(self, sample: ProfileSample) -> None:
        """End a sample."""
        if sample.timing:
            sample.timing.stop()

        if sample.memory and self.enable_memory:
            gc.collect()
            sample.memory.end_memory = self._get_memory_usage()
            sample.memory.allocated = max(0, sample.memory.end_memory - sample.memory.start_memory)

        with self._lock:
            self._samples.append(sample)
            self._active_timings.pop(sample.operation, None)

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        if sys.platform == "darwin":
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        else:
            try:
                import resource

                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            except ImportError:
                return 0

    def record_metric(
        self,
        operation: str,
        duration_ms: float,
        memory_bytes: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a custom metric.

        Args:
            operation: Operation name.
            duration_ms: Duration in milliseconds.
            memory_bytes: Memory usage in bytes.
            metadata: Optional metadata.
        """
        timing = TimingMetric(
            name=operation,
            start_time=0,
            end_time=duration_ms / 1000,
            duration=duration_ms / 1000,
        )

        memory = MemoryMetric(
            name=operation,
            start_memory=0,
            end_memory=memory_bytes,
        )

        sample = ProfileSample(
            timestamp=datetime.now(),
            operation=operation,
            timing=timing,
            memory=memory,
            metadata=metadata or {},
        )

        with self._lock:
            self._samples.append(sample)

    def generate_report(
        self,
        name: str = "Performance Report",
        end_time: datetime | None = None,
        backend_results: dict[str, Any] | None = None,
    ) -> ProfileReport:
        """Generate a profiling report.

        Args:
            name: Report name.
            end_time: End time for the report.
            backend_results: Results from profiler backends.

        Returns:
            ProfileReport with statistics.
        """
        end_time = end_time or datetime.now()
        start_time = self._start_time or datetime.now()

        # Group samples by operation
        by_operation: dict[str, list[ProfileSample]] = defaultdict(list)
        for sample in self._samples:
            by_operation[sample.operation].append(sample)

        # Calculate statistics
        statistics = {
            op: ProfileStatistics.from_samples(op, samples) for op, samples in by_operation.items()
        }

        # Identify hotspots
        hotspots = self._identify_hotspots(statistics)

        # Generate recommendations
        recommendations = self._generate_recommendations(statistics, hotspots)

        total_duration = sum(s.total_time for s in statistics.values())

        return ProfileReport(
            name=name,
            start_time=start_time,
            end_time=end_time,
            statistics=statistics,
            samples=self._samples.copy(),
            total_operations=len(self._samples),
            total_duration=total_duration,
            hotspots=hotspots,
            recommendations=recommendations,
            metadata={"backend_results": backend_results or {}},
        )

    def _identify_hotspots(self, statistics: dict[str, ProfileStatistics]) -> list[str]:
        """Identify performance hotspots."""
        hotspots = []

        if not statistics:
            return hotspots

        total_time = sum(s.total_time for s in statistics.values())

        for op, stats in statistics.items():
            # Operations taking >20% of total time
            if total_time > 0 and stats.total_time / total_time > 0.2:
                hotspots.append(
                    f"{op}: {stats.total_time:.1f}ms "
                    f"({stats.total_time / total_time * 100:.1f}% of total)"
                )

            # High variance operations
            if stats.avg_time > 0 and stats.std_dev / stats.avg_time > 0.5:
                hotspots.append(f"{op}: High variance (std_dev={stats.std_dev:.2f}ms)")

        return hotspots

    def _generate_recommendations(
        self,
        statistics: dict[str, ProfileStatistics],
        hotspots: list[str],
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        for op, stats in statistics.items():
            # Recommend caching for repeated operations
            if stats.count > 10 and stats.avg_time > 100:
                recommendations.append(
                    f"Consider caching results for '{op}' "
                    f"(called {stats.count} times, avg {stats.avg_time:.1f}ms)"
                )

            # Recommend batching
            if stats.count > 50 and stats.avg_time < 10:
                recommendations.append(
                    f"Consider batching '{op}' operations " f"(many small calls detected)"
                )

            # Recommend memory optimization
            if stats.peak_memory > 100 * 1024 * 1024:  # >100MB
                recommendations.append(
                    f"High memory usage in '{op}': " f"{stats.peak_memory / 1024 / 1024:.1f}MB peak"
                )

        return recommendations

    def clear(self) -> None:
        """Clear all collected samples."""
        with self._lock:
            self._samples.clear()
            self._active_timings.clear()


def profile_function(
    profiler: PerformanceProfiler | None = None,
    operation_name: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to profile a function.

    Args:
        profiler: Profiler to use (global if None).
        operation_name: Custom operation name.

    Returns:
        Decorated function.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            prof = profiler or get_performance_profiler()
            op_name = operation_name or func.__name__
            with prof.profile(op_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Global instance
_performance_profiler: PerformanceProfiler | None = None


def get_performance_profiler() -> PerformanceProfiler:
    """Get the global performance profiler.

    Returns:
        The global PerformanceProfiler instance.
    """
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler


def reset_performance_profiler() -> None:
    """Reset the global performance profiler."""
    global _performance_profiler
    _performance_profiler = None


def create_performance_profiler(
    level: ProfileLevel = ProfileLevel.BASIC,
    enable_memory: bool = True,
) -> PerformanceProfiler:
    """Create a new performance profiler.

    Args:
        level: Profiling detail level.
        enable_memory: Whether to track memory.

    Returns:
        New PerformanceProfiler instance.
    """
    return PerformanceProfiler(level=level, enable_memory=enable_memory)


@contextmanager
def profile_explanation(
    operation: str,
    **kwargs: Any,
) -> Generator[ProfileSample, None, None]:
    """Context manager for profiling explanation operations.

    Args:
        operation: Operation name.
        **kwargs: Additional metadata.

    Yields:
        ProfileSample being recorded.
    """
    profiler = get_performance_profiler()
    with profiler.profile(operation, metadata=kwargs) as sample:
        yield sample


def benchmark_operation(
    func: Callable[..., T],
    iterations: int = 100,
    warmup: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Benchmark an operation.

    Args:
        func: Function to benchmark.
        iterations: Number of iterations.
        warmup: Warmup iterations.
        **kwargs: Arguments to pass to func.

    Returns:
        Benchmark results.
    """
    # Warmup
    for _ in range(warmup):
        func(**kwargs)

    # Benchmark
    profiler = PerformanceProfiler(level=ProfileLevel.DETAILED)
    profiler.start()

    for i in range(iterations):
        with profiler.profile(f"iteration_{i}"):
            func(**kwargs)

    report = profiler.stop()

    times = [s.duration_ms for s in report.samples if s.timing]

    return {
        "iterations": iterations,
        "total_time_ms": sum(times),
        "avg_time_ms": sum(times) / len(times) if times else 0,
        "min_time_ms": min(times) if times else 0,
        "max_time_ms": max(times) if times else 0,
        "std_dev_ms": report.statistics.get(
            "iteration_0", ProfileStatistics("", 0, 0, 0, 0, 0, 0, 0, 0, 0)
        ).std_dev,
        "ops_per_second": iterations / (sum(times) / 1000) if times else 0,
    }
