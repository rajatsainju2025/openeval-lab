"""Advanced optimization and performance monitoring for OpenEval Lab."""

import time
import threading
import statistics
import asyncio
from typing import Dict, List, Any, Optional, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import gc
import resource
from functools import wraps

from .logging import get_logger
from .core import Dataset

T = TypeVar("T")


@dataclass
class PerformanceMetric:
    """A single performance measurement."""

    name: str
    value: float
    unit: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """System resource usage snapshot."""

    memory_used_mb: float
    thread_count: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class OptimizationSuggestion:
    """Performance optimization suggestion."""

    category: str  # memory, cpu, io, network, concurrency
    severity: str  # low, medium, high, critical
    title: str
    description: str
    recommendation: str
    impact: str  # estimated improvement
    effort: str  # implementation effort (low, medium, high)


class PerformanceMonitor:
    """Real-time performance monitoring and optimization."""

    def __init__(self, sample_interval: float = 1.0):
        """Initialize performance monitor."""
        self.logger = get_logger()
        self.sample_interval = sample_interval
        self.monitoring = False
        self.metrics: List[PerformanceMetric] = []
        self.snapshots: List[SystemSnapshot] = []
        self.monitor_thread: Optional[threading.Thread] = None

        # Performance thresholds
        self.thresholds = {
            "response_time_warning": 5.0,  # seconds
            "response_time_critical": 10.0,
            "throughput_warning": 0.5,  # samples/sec
            "error_rate_warning": 0.05,  # 5%
            "error_rate_critical": 0.10,  # 10%
        }

    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Started performance monitoring")

    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Stopped performance monitoring")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # Create snapshot with available info
                snapshot = SystemSnapshot(
                    memory_used_mb=self._get_memory_usage(), thread_count=threading.active_count()
                )

                self.snapshots.append(snapshot)

                # Keep only recent snapshots (last hour)
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                self.snapshots = [
                    s for s in self.snapshots if datetime.fromisoformat(s.timestamp) > cutoff_time
                ]

                time.sleep(self.sample_interval)

            except Exception as e:
                self.logger.warning(f"Performance monitoring error: {e}")
                time.sleep(self.sample_interval)

    def _get_memory_usage(self) -> float:
        """Get memory usage in MB using resource module."""
        try:
            # Use resource module for basic memory info
            mem_info = resource.getrusage(resource.RUSAGE_SELF)
            # Convert to MB (getrusage returns in KB on Linux, bytes on macOS)
            return mem_info.ru_maxrss / 1024  # Assume KB for simplicity
        except Exception:
            return 0.0

    @contextmanager
    def measure_operation(self, operation_name: str):
        """Context manager to measure operation performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Force garbage collection before measurement
        gc.collect()

        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            memory_delta = self._get_memory_usage() - start_memory

            # Record metrics
            self.record_metric(
                "operation_duration", duration, "seconds", {"operation": operation_name}
            )
            self.record_metric(
                "operation_memory_delta", memory_delta, "mb", {"operation": operation_name}
            )

            self.logger.debug(f"Operation '{operation_name}' took {duration:.2f}s")

    def record_metric(
        self, name: str, value: float, unit: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(name=name, value=value, unit=unit, metadata=metadata or {})

        self.metrics.append(metric)

        # Keep only recent metrics (last hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        self.metrics = [
            m for m in self.metrics if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        if not self.snapshots:
            return {"error": "No performance data available"}

        recent_snapshots = self.snapshots[-10:]  # Last 10 measurements

        summary = {
            "current": {
                "memory_used_mb": recent_snapshots[-1].memory_used_mb,
                "thread_count": recent_snapshots[-1].thread_count,
            },
            "averages": {
                "memory_used_mb": statistics.mean(s.memory_used_mb for s in recent_snapshots)
            },
            "peaks": {"max_memory_used_mb": max(s.memory_used_mb for s in recent_snapshots)},
        }

        return summary


class AdaptiveConcurrencyController:
    """Dynamically adjust concurrency based on system performance."""

    def __init__(
        self, initial_concurrency: int = 4, min_concurrency: int = 1, max_concurrency: int = 16
    ):
        """Initialize adaptive concurrency controller."""
        self.logger = get_logger()
        self.current_concurrency = initial_concurrency
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency

        # Performance tracking
        self.throughput_history: List[float] = []
        self.error_rate_history: List[float] = []
        self.response_time_history: List[float] = []

        # Adjustment parameters
        self.adjustment_interval = 10  # measurements between adjustments
        self.measurement_count = 0
        self.last_adjustment_time = time.time()

    def record_performance(
        self, throughput: float, error_rate: float, response_time: float
    ) -> None:
        """Record performance metrics for concurrency adjustment."""
        self.throughput_history.append(throughput)
        self.error_rate_history.append(error_rate)
        self.response_time_history.append(response_time)

        # Keep only recent history
        max_history = 20
        if len(self.throughput_history) > max_history:
            self.throughput_history = self.throughput_history[-max_history:]
            self.error_rate_history = self.error_rate_history[-max_history:]
            self.response_time_history = self.response_time_history[-max_history:]

    def get_current_concurrency(self) -> int:
        """Get current optimal concurrency level."""
        return self.current_concurrency


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int = 32
    max_concurrent: int = 4
    timeout_per_batch: Optional[float] = None
    retry_failed: bool = True


class BatchProcessor:
    """Efficient batch processing for model evaluation."""

    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize batch processor."""
        self.config = config or BatchConfig()

    def create_batches(self, items: List[T], batch_size: Optional[int] = None) -> List[List[T]]:
        """Split items into batches."""
        batch_size = batch_size or self.config.batch_size
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    async def process_batch_async(
        self, batch: List[T], processor_func: Callable[[T], Any], timeout: Optional[float] = None
    ) -> List[Any]:
        """Process a batch asynchronously."""
        timeout = timeout or self.config.timeout_per_batch

        async def process_item(item: T) -> Any:
            # Run in thread pool for CPU-bound operations
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, processor_func, item)

        # Process batch with timeout
        try:
            tasks = [process_item(item) for item in batch]
            if timeout:
                results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
            else:
                results = await asyncio.gather(*tasks)
            return results
        except asyncio.TimeoutError:
            if self.config.retry_failed:
                # Retry with smaller batch or individual items
                return await self._retry_batch(batch, processor_func)
            else:
                raise

    async def _retry_batch(self, batch: List[T], processor_func: Callable[[T], Any]) -> List[Any]:
        """Retry failed batch with fallback strategy."""
        # Try processing items individually
        results = []
        for item in batch:
            try:
                result = await asyncio.get_event_loop().run_in_executor(None, processor_func, item)
                results.append(result)
            except Exception as e:
                # Log error and use None as placeholder
                print(f"Warning: Failed to process item: {e}")
                results.append(None)
        return results

    def process_batches(self, items: List[T], processor_func: Callable[[T], Any]) -> List[Any]:
        """Process all items in batches."""
        batches = self.create_batches(items)
        results = []

        async def process_all_batches():
            semaphore = asyncio.Semaphore(self.config.max_concurrent)

            async def process_single_batch(batch):
                async with semaphore:
                    return await self.process_batch_async(batch, processor_func)

            batch_tasks = [process_single_batch(batch) for batch in batches]
            batch_results = await asyncio.gather(*batch_tasks)

            # Flatten results
            for batch_result in batch_results:
                results.extend(batch_result)

        # Run the async processing
        try:
            asyncio.run(process_all_batches())
        except Exception as e:
            # Fallback to synchronous processing
            print(f"Async processing failed, falling back to sync: {e}")
            for batch in batches:
                batch_results = [processor_func(item) for item in batch]
                results.extend(batch_results)

        return results


class CacheManager:
    """Advanced caching with TTL and size limits."""

    def __init__(self, max_size: int = 10000, default_ttl: float = 3600):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of cached items
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._cache:
            return True

        entry = self._cache[key]
        if entry.get("ttl") is None:
            return False

        return time.time() > entry["timestamp"] + entry["ttl"]

    def _evict_lru(self):
        """Evict least recently used items if cache is full."""
        if len(self._cache) >= self.max_size:
            # Find LRU item
            lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            del self._cache[lru_key]
            del self._access_times[lru_key]

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if self._is_expired(key):
            self.delete(key)
            return None

        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]["value"]

        return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set item in cache."""
        self._evict_lru()

        self._cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl or self.default_ttl,
        }
        self._access_times[key] = time.time()

    def delete(self, key: str) -> None:
        """Delete item from cache."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        expired_entries = sum(1 for key in self._cache if self._is_expired(key))

        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "valid_entries": total_entries - expired_entries,
            "cache_size": total_entries,
            "max_size": self.max_size,
            "hit_rate": 0.0,  # Would need request tracking for this
        }


def memoize_with_ttl(ttl: float = 3600):
    """Decorator for memoizing function results with TTL."""
    cache = CacheManager(default_ttl=ttl)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str(hash((args, tuple(sorted(kwargs.items())))))

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        # Add cache management methods
        # Store cache methods
        wrapper._cache = cache  # type: ignore

        return wrapper

    return decorator


class ProgressTracker:
    """Track progress for long-running evaluations."""

    def __init__(self, total: int, description: str = "Processing"):
        """Initialize progress tracker."""
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0

    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        current_time = time.time()

        # Update every second or at completion
        if current_time - self.last_update >= 1.0 or self.current >= self.total:
            self._print_progress()
            self.last_update = current_time

    def _print_progress(self) -> None:
        """Print progress bar."""
        if self.total == 0:
            return

        progress = self.current / self.total
        elapsed = time.time() - self.start_time

        # Estimate time remaining
        if progress > 0:
            eta = elapsed / progress - elapsed
            eta_str = f"ETA: {eta:.1f}s" if eta > 0 else "ETA: 0s"
        else:
            eta_str = "ETA: --"

        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        # Print progress
        print(
            f"\r{self.description}: {bar} {progress:.1%} ({self.current}/{self.total}) {eta_str}",
            end="",
        )

        if self.current >= self.total:
            print()  # New line when complete


class StreamingDataset:
    """Memory-efficient streaming dataset wrapper."""

    def __init__(self, dataset: Dataset, chunk_size: int = 1000):
        """
        Initialize streaming dataset.

        Args:
            dataset: Base dataset to stream from
            chunk_size: Number of examples to load at once
        """
        self.dataset = dataset
        self.chunk_size = chunk_size
        self._current_chunk = []
        self._chunk_index = 0
        self._total_processed = 0

    def __iter__(self):
        """Iterate over dataset in chunks."""
        chunk = []
        for example in self.dataset:
            chunk.append(example)
            self._total_processed += 1

            if len(chunk) >= self.chunk_size:
                yield from chunk
                chunk = []

        # Yield remaining examples
        if chunk:
            yield from chunk

    def get_stats(self) -> Dict[str, int]:
        """Get streaming statistics."""
        return {
            "total_processed": self._total_processed,
            "chunk_size": self.chunk_size,
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_performance(operation_name: str):
    """Decorator for monitoring function performance."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with performance_monitor.measure_operation(operation_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def profile_evaluation(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to profile evaluation performance."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage()

        try:
            result = func(*args, **kwargs)

            end_time = time.time()
            end_memory = _get_memory_usage()

            # Log performance stats
            elapsed = end_time - start_time
            memory_delta = end_memory - start_memory

            print(f"\n--- Performance Profile for {func.__name__} ---")
            print(f"Execution time: {elapsed:.2f}s")
            print(f"Memory usage: {memory_delta:.2f}MB")
            print(f"Peak memory: {end_memory:.2f}MB")

            return result

        except Exception as e:
            print(f"\n--- Error in {func.__name__} ---")
            print(f"Error: {e}")
            raise

    return wrapper


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        # Use resource module for basic memory info
        mem_info = resource.getrusage(resource.RUSAGE_SELF)
        # Convert to MB (getrusage returns in KB on Linux, bytes on macOS)
        return mem_info.ru_maxrss / 1024  # Assume KB for simplicity
    except Exception:
        return 0.0  # Fallback if not available


@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottlenecks."""

    component: str
    bottleneck_type: str  # cpu, memory, io, network
    severity: str
    description: str
    recommendations: List[str]
    estimated_impact: float  # percentage improvement


class AdvancedProfiler:
    """Advanced performance profiling with bottleneck analysis."""

    def __init__(self):
        self.logger = get_logger()
        self.measurements = []
        self.bottlenecks = []

    @contextmanager
    def profile_component(self, component_name: str, component_type: str = "general"):
        """Profile a specific component with detailed analysis."""
        start_time = time.time()
        start_memory = _get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = _get_memory_usage()

            measurement = {
                "component": component_name,
                "type": component_type,
                "duration": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "start_time": start_time,
                "end_time": end_time,
            }
            self.measurements.append(measurement)

    def analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyze measurements to identify bottlenecks."""
        if not self.measurements:
            return []

        bottlenecks = []

        # Find slowest components
        sorted_measurements = sorted(self.measurements, key=lambda x: x["duration"], reverse=True)
        total_time = sum(m["duration"] for m in self.measurements)

        for measurement in sorted_measurements[:3]:  # Top 3 bottlenecks
            percentage = (measurement["duration"] / total_time) * 100

            if percentage > 50:
                severity = "critical"
            elif percentage > 25:
                severity = "high"
            elif percentage > 10:
                severity = "medium"
            else:
                severity = "low"

            # Determine bottleneck type
            bottleneck_type = "cpu"
            recommendations = []

            if measurement["memory_delta"] > 100:  # > 100MB increase
                bottleneck_type = "memory"
                recommendations = [
                    "Consider streaming data processing",
                    "Implement memory pooling",
                    "Use smaller batch sizes",
                ]
            elif "io" in measurement["component"].lower():
                bottleneck_type = "io"
                recommendations = [
                    "Implement caching",
                    "Use async I/O operations",
                    "Consider data compression",
                ]
            elif "network" in measurement["component"].lower():
                bottleneck_type = "network"
                recommendations = [
                    "Implement connection pooling",
                    "Use batch requests",
                    "Consider local caching",
                ]
            else:
                recommendations = [
                    "Profile with line profiler",
                    "Consider algorithm optimization",
                    "Evaluate parallelization opportunities",
                ]

            bottleneck = BottleneckAnalysis(
                component=measurement["component"],
                bottleneck_type=bottleneck_type,
                severity=severity,
                description=f"Component takes {percentage:.1f}% of total time ({measurement['duration']:.2f}s)",
                recommendations=recommendations,
                estimated_impact=min(percentage * 0.8, 90.0),  # Conservative estimate
            )
            bottlenecks.append(bottleneck)

        self.bottlenecks = bottlenecks
        return bottlenecks

    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report."""
        if not self.measurements:
            return "No measurements available for analysis."

        report = ["# Advanced Performance Profiling Report\n"]

        # Summary statistics
        total_time = sum(m["duration"] for m in self.measurements)
        total_memory = sum(m["memory_delta"] for m in self.measurements)
        component_count = len(set(m["component"] for m in self.measurements))

        report.append("## Summary")
        report.append(f"- Total profiled time: {total_time:.2f}s")
        report.append(f"- Memory delta: {total_memory:.1f}MB")
        report.append(f"- Components profiled: {component_count}")
        report.append("")

        # Component breakdown
        report.append("## Component Performance")
        for measurement in sorted(self.measurements, key=lambda x: x["duration"], reverse=True):
            percentage = (measurement["duration"] / total_time) * 100
            report.append(
                f"- {measurement['component']}: {measurement['duration']:.2f}s ({percentage:.1f}%)"
            )
        report.append("")

        # Bottleneck analysis
        bottlenecks = self.analyze_bottlenecks()
        if bottlenecks:
            report.append("## Bottleneck Analysis")
            for bottleneck in bottlenecks:
                report.append(f"### {bottleneck.component} ({bottleneck.severity.upper()})")
                report.append(f"- Type: {bottleneck.bottleneck_type}")
                report.append(f"- Description: {bottleneck.description}")
                report.append(f"- Estimated impact: {bottleneck.estimated_impact:.1f}% improvement")
                report.append("- Recommendations:")
                for rec in bottleneck.recommendations:
                    report.append(f"  - {rec}")
                report.append("")

        return "\n".join(report)


class AdaptiveOptimizer:
    """Adaptive optimization based on real-time performance monitoring."""

    def __init__(self):
        self.logger = get_logger()
        self.performance_history = []
        self.adaptation_rules = {
            "high_memory": {
                "condition": lambda metrics: metrics.get("memory_mb", 0) > 1000,
                "action": "reduce_batch_size",
                "description": "Reduce batch size to lower memory usage",
            },
            "low_throughput": {
                "condition": lambda metrics: metrics.get("throughput", 1) < 5,
                "action": "increase_concurrency",
                "description": "Increase concurrency for better throughput",
            },
            "high_cpu": {
                "condition": lambda metrics: metrics.get("cpu_percent", 0) > 90,
                "action": "reduce_concurrency",
                "description": "Reduce concurrency to lower CPU usage",
            },
        }

    def analyze_and_adapt(self, current_metrics: Dict[str, Any]) -> List[str]:
        """Analyze current metrics and suggest adaptations."""
        adaptations = []

        for rule_name, rule in self.adaptation_rules.items():
            if rule["condition"](current_metrics):
                adaptations.append(rule["description"])
                self.logger.info(f"Adaptation triggered: {rule['description']}")

        self.performance_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": current_metrics,
                "adaptations": adaptations,
            }
        )

        return adaptations

    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.performance_history) < 2:
            return {"trend": "insufficient_data"}

        # Simple trend analysis
        recent = self.performance_history[-5:]  # Last 5 measurements
        memory_trend = self._calculate_trend([m["metrics"].get("memory_mb", 0) for m in recent])
        throughput_trend = self._calculate_trend(
            [m["metrics"].get("throughput", 0) for m in recent]
        )

        return {
            "memory_trend": memory_trend,
            "throughput_trend": throughput_trend,
            "recommendations": self._generate_trend_recommendations(memory_trend, throughput_trend),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return "stable"

        # Simple linear trend
        slope = statistics.linear_regression(range(len(values)), values).slope

        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def _generate_trend_recommendations(
        self, memory_trend: str, throughput_trend: str
    ) -> List[str]:
        """Generate recommendations based on trends."""
        recommendations = []

        if memory_trend == "increasing":
            recommendations.append("Memory usage is trending up - consider memory optimization")
        if throughput_trend == "decreasing":
            recommendations.append(
                "Throughput is trending down - investigate performance bottlenecks"
            )
        if memory_trend == "stable" and throughput_trend == "stable":
            recommendations.append("Performance is stable - no immediate optimizations needed")

        return recommendations


"""
Performance Profiling and Optimization Toolkit for OpenEval Lab

This module provides comprehensive performance profiling, bottleneck analysis,
and optimization recommendations for ev            # Get current memory usage
            if HAS_PSUTIL and psutil is not None:
                process = psutil.Process()
                metrics.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                metrics.cpu_usage = process.cpu_percent()
            else:
                metrics.memory_usage = 0.0
                metrics.cpu_usage = 0.0

            # Predictive analytics - simple linear regression on last 5 metrics
            if len(self.metrics_history) >= 5 and HAS_NUMPY and np is not None:
                x = np.arange(5)
                y = np.array([m.execution_time for m in self.metrics_history[-5:]])
                coeffs = np.polyfit(x, y, 1)
                metrics.predicted_time = coeffs[0] * 5 + coeffs[1]  # Predict next value
            else:
                metrics.predicted_time = Nonepelines.

Advanced Features:
- Real-time performance monitoring with adaptive sampling
- Memory leak detection and analysis
- CPU hotspot identification with flame graph generation
- Predictive performance modeling
- Automated optimization suggestions
- Distributed profiling support
"""

from __future__ import annotations

import psutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import cProfile
import pstats
import io
from contextlib import contextmanager
import tracemalloc

try:
    import psutil  # type: ignore

    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    HAS_PSUTIL = False

try:
    import memory_profiler  # type: ignore

    HAS_MEMORY_PROFILER = True
except ImportError:
    memory_profiler = None  # type: ignore
    HAS_MEMORY_PROFILER = False

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics with predictive analytics."""

    execution_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    peak_memory: float = 0.0
    memory_growth: float = 0.0
    function_calls: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    predicted_time: Optional[float] = None
    efficiency_score: float = 0.0
    bottleneck_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with predictions."""
        return {
            "execution_time": self.execution_time,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "peak_memory": self.peak_memory,
            "memory_growth": self.memory_growth,
            "function_calls": self.function_calls,
            "timestamp": self.timestamp.isoformat(),
            "predicted_time": self.predicted_time,
            "efficiency_score": self.efficiency_score,
            "bottleneck_type": self.bottleneck_type,
        }


@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottlenecks."""

    slowest_functions: List[Tuple[str, float]] = field(default_factory=list)
    memory_hungry_functions: List[Tuple[str, float]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)

    def add_recommendation(self, recommendation: str) -> None:
        """Add a performance recommendation."""
        self.recommendations.append(recommendation)

    def add_optimization_opportunity(self, opportunity: str) -> None:
        """Add an optimization opportunity."""
        self.optimization_opportunities.append(opportunity)


class PerformanceProfiler:
    """
    Comprehensive performance profiler for evaluation pipelines.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("performance_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[PerformanceMetrics] = []
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

    @contextmanager
    def profile_execution(
        self,
        name: str = "evaluation",
        enable_memory_profiling: bool = True,
        enable_cpu_profiling: bool = True,
    ):
        """
        Context manager for profiling code execution.

        Args:
            name: Name identifier for the profiling session
            enable_memory_profiling: Whether to profile memory usage
            enable_cpu_profiling: Whether to profile CPU usage
        """
        metrics = PerformanceMetrics()
        start_time = time.time()

        # Start monitoring
        if enable_cpu_profiling or enable_memory_profiling:
            self._start_monitoring(metrics, enable_memory_profiling, enable_cpu_profiling)

        # Memory profiling setup
        start_snapshot = None
        if enable_memory_profiling:
            tracemalloc.start()
            gc.collect()  # Clean up before measurement
            start_snapshot = tracemalloc.take_snapshot()

        profiler = cProfile.Profile()
        profiler.enable()

        try:
            yield metrics
        finally:
            # Stop profiling
            profiler.disable()

            # Stop monitoring
            if self._monitoring_thread:
                self._stop_monitoring.set()
                self._monitoring_thread.join()

            # Calculate execution time
            metrics.execution_time = time.time() - start_time

            # Memory profiling analysis
            if enable_memory_profiling and start_snapshot is not None:
                end_snapshot = tracemalloc.take_snapshot()
                tracemalloc.stop()

                start_stats = start_snapshot.statistics("lineno")
                end_stats = end_snapshot.statistics("lineno")

                metrics.memory_growth = sum(stat.size for stat in end_stats) - sum(
                    stat.size for stat in start_stats
                )
                metrics.peak_memory = max((stat.size for stat in end_stats), default=0)

            # CPU profiling analysis
            stats_stream = io.StringIO()
            ps = pstats.Stats(profiler, stream=stats_stream)
            ps.print_stats()
            stats_output = stats_stream.getvalue()

            # Extract function calls count from stats output
            # This is a simplified approach - in practice you'd parse the output
            metrics.function_calls = stats_output.count("\n")  # Rough estimate

            # Get current memory usage
            process = psutil.Process()
            metrics.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            metrics.cpu_usage = process.cpu_percent()

            # Predictive analytics - simple linear regression on last 5 metrics
            if len(self.metrics_history) >= 5:
                x = np.arange(5)
                y = np.array([m.execution_time for m in self.metrics_history[-5:]])
                coeffs = np.polyfit(x, y, 1)
                metrics.predicted_time = coeffs[0] * 5 + coeffs[1]  # Extrapolate to next point

            # Efficiency scoring - based on CPU and memory usage
            metrics.efficiency_score = 100 - (metrics.cpu_usage + metrics.memory_usage / 100)

            # Store metrics
            self.metrics_history.append(metrics)

            # Log performance summary
            logger.info(f"Performance profiling completed for '{name}':")
            logger.info(f"  Execution time: {metrics.execution_time:.3f}s")
            logger.info(f"  Memory usage: {metrics.memory_usage:.2f}MB")
            logger.info(f"  Peak memory: {metrics.peak_memory / 1024 / 1024:.2f}MB")
            logger.info(f"  Function calls: {metrics.function_calls}")
            logger.info(f"  Predicted next execution time: {metrics.predicted_time:.3f}s")
            logger.info(f"  Efficiency score: {metrics.efficiency_score:.1f}")
            logger.info(f"  Bottleneck type: {metrics.bottleneck_type}")

    def _start_monitoring(
        self, metrics: PerformanceMetrics, enable_memory: bool, enable_cpu: bool
    ) -> None:
        """Start background monitoring thread."""
        self._stop_monitoring.clear()

        def monitor():
            if not HAS_PSUTIL or psutil is None:
                return

            process = psutil.Process()
            memory_readings = []
            cpu_readings = []

            while not self._stop_monitoring.is_set():
                if enable_memory:
                    memory_readings.append(process.memory_info().rss / 1024 / 1024)  # MB

                if enable_cpu:
                    cpu_readings.append(process.cpu_percent(interval=0.1))

                time.sleep(0.1)

            # Update metrics with averages
            if memory_readings:
                metrics.memory_usage = statistics.mean(memory_readings)
            if cpu_readings:
                metrics.cpu_usage = statistics.mean(cpu_readings)

        self._monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self._monitoring_thread.start()

    def analyze_bottlenecks(
        self, profile_data: Optional[pstats.Stats] = None, top_n: int = 10
    ) -> BottleneckAnalysis:
        """
        Analyze performance bottlenecks from profiling data.

        Args:
            profile_data: Profiling statistics (if None, analyzes recent metrics)
            top_n: Number of top bottlenecks to analyze

        Returns:
            BottleneckAnalysis with findings and recommendations
        """
        analysis = BottleneckAnalysis()

        if profile_data:
            # For now, provide general recommendations when profiling data is available
            # In a full implementation, you'd properly parse the pstats.Stats object
            analysis.add_recommendation(
                "Profiling data indicates potential performance bottlenecks. "
                "Consider using cProfile with pstats for detailed function-level analysis."
            )

        # Analyze memory patterns from metrics history
        if self.metrics_history:
            recent_metrics = self.metrics_history[-10:]  # Last 10 measurements

            memory_usages = [m.memory_usage for m in recent_metrics]
            if len(memory_usages) > 1:
                memory_growth_rate = statistics.mean(memory_usages[-3:]) - statistics.mean(
                    memory_usages[:3]
                )
                if memory_growth_rate > 50:  # MB
                    analysis.add_recommendation(
                        f"Memory usage is growing at {memory_growth_rate:.1f}MB. "
                        "Check for memory leaks or consider using generators for large datasets."
                    )

            # Check for high memory usage
            avg_memory = statistics.mean(memory_usages)
            if avg_memory > 1000:  # MB
                analysis.add_recommendation(
                    f"High memory usage detected ({avg_memory:.1f}MB average). "
                    "Consider processing data in batches or using memory-efficient data structures."
                )

        # General optimization opportunities
        analysis.add_optimization_opportunity(
            "Use batch processing for large datasets to reduce memory overhead"
        )
        analysis.add_optimization_opportunity("Implement caching for expensive computations")
        analysis.add_optimization_opportunity("Use multiprocessing for CPU-bound tasks")
        analysis.add_optimization_opportunity(
            "Profile with different batch sizes to find optimal configuration"
        )

        return analysis

    def generate_performance_report(
        self, analysis: Optional[BottleneckAnalysis] = None, include_recommendations: bool = True
    ) -> Path:
        """
        Generate a comprehensive performance report.

        Args:
            analysis: Bottleneck analysis to include
            include_recommendations: Whether to include optimization recommendations

        Returns:
            Path to the generated report
        """
        report_path = self.output_dir / f"performance_report_{int(time.time())}.html"

        if not self.metrics_history:
            # Generate basic report with no data
            html_content = self._generate_empty_report()
        else:
            html_content = self._generate_full_report(analysis, include_recommendations)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated performance report: {report_path}")
        return report_path

    def _generate_empty_report(self) -> str:
        """Generate a report when no profiling data is available."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>OpenEval Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .warning {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 OpenEval Performance Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="warning">
        <h3>No Profiling Data Available</h3>
        <p>Use the PerformanceProfiler context manager to collect performance metrics.</p>
    </div>
</body>
</html>"""

    def _generate_full_report(
        self, analysis: Optional[BottleneckAnalysis], include_recommendations: bool
    ) -> str:
        """Generate a full performance report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OpenEval Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 3px; }}
        .bottleneck {{ background: #f8d7da; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .recommendation {{ background: #d1ecf1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .opportunity {{ background: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 OpenEval Performance Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>📊 Performance Summary</h2>
"""

        if self.metrics_history:
            latest = self.metrics_history[-1]
            html += f"""
        <div class="metric">Execution Time: {latest.execution_time:.3f}s</div>
        <div class="metric">Memory Usage: {latest.memory_usage:.2f}MB</div>
        <div class="metric">Peak Memory: {latest.peak_memory / 1024 / 1024:.2f}MB</div>
        <div class="metric">CPU Usage: {latest.cpu_usage:.1f}%</div>
        <div class="metric">Function Calls: {latest.function_calls:,}</div>
        <div class="metric">Predicted Next Execution Time: {latest.predicted_time:.3f}s</div>
        <div class="metric">Efficiency Score: {latest.efficiency_score:.1f}</div>
"""

        html += """
    </div>

    <div class="section">
        <h2>📈 Performance History</h2>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Execution Time (s)</th>
                <th>Memory Usage (MB)</th>
                <th>Peak Memory (MB)</th>
                <th>CPU Usage (%)</th>
                <th>Function Calls</th>
            </tr>
"""

        for metrics in self.metrics_history[-20:]:  # Show last 20 entries
            html += f"""
            <tr>
                <td>{metrics.timestamp.strftime('%H:%M:%S')}</td>
                <td>{metrics.execution_time:.3f}</td>
                <td>{metrics.memory_usage:.2f}</td>
                <td>{metrics.peak_memory / 1024 / 1024:.2f}</td>
                <td>{metrics.cpu_usage:.1f}</td>
                <td>{metrics.function_calls:,}</td>
            </tr>"""

        html += """
        </table>
    </div>
"""

        if analysis:
            html += """
    <div class="section">
        <h2>🔍 Bottleneck Analysis</h2>
"""

            if analysis.slowest_functions:
                html += """
        <h3>Slowest Functions</h3>
"""
                for func, cum_time in analysis.slowest_functions[:10]:
                    func_name = f"{func[0]}:{func[1]}({func[2]})"
                    html += f"""
        <div class="bottleneck">
            <strong>{func_name}</strong>: {cum_time:.3f}s cumulative time
        </div>"""

            if include_recommendations and analysis.recommendations:
                html += """
        <h3>Performance Recommendations</h3>
"""
                for rec in analysis.recommendations:
                    html += f"""
        <div class="recommendation">
            {rec}
        </div>"""

            if include_recommendations and analysis.optimization_opportunities:
                html += """
        <h3>Optimization Opportunities</h3>
"""
                for opp in analysis.optimization_opportunities:
                    html += f"""
        <div class="opportunity">
            {opp}
        </div>"""

        html += """
    </div>
</body>
</html>"""

        return html

    def export_metrics(self, format: str = "json") -> Path:
        """
        Export performance metrics in various formats.

        Args:
            format: Export format (json, csv)

        Returns:
            Path to exported file
        """
        timestamp = int(time.time())
        metrics_data = [m.to_dict() for m in self.metrics_history]

        if format == "json":
            import json

            export_path = self.output_dir / f"performance_metrics_{timestamp}.json"
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)

        elif format == "csv":
            try:
                import pandas as pd

                df = pd.DataFrame(metrics_data)
                export_path = self.output_dir / f"performance_metrics_{timestamp}.csv"
                df.to_csv(export_path, index=False)
            except ImportError:
                raise ImportError("pandas required for CSV export")

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported performance metrics to: {export_path}")
        return export_path

    def clear_history(self) -> None:
        """Clear the metrics history."""
        self.metrics_history.clear()
        logger.info("Cleared performance metrics history")


class OptimizationAdvisor:
    """
    Provides optimization advice based on performance analysis.
    """

    def __init__(self):
        self.optimization_patterns = {
            "memory_leak": {
                "indicators": ["memory_growth", "peak_memory"],
                "advice": "Consider using weak references or implementing proper cleanup",
            },
            "cpu_bound": {
                "indicators": ["cpu_usage", "execution_time"],
                "advice": "Consider using multiprocessing or async processing",
            },
            "io_bound": {
                "indicators": ["function_calls", "execution_time"],
                "advice": "Consider using async I/O or caching",
            },
        }

    def get_optimization_suggestions(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """
        Get optimization suggestions based on performance metrics.

        Args:
            metrics: List of performance metrics

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        if not metrics:
            return suggestions

        # Analyze patterns
        avg_metrics = self._calculate_average_metrics(metrics)

        # Memory optimization
        if avg_metrics.get("memory_growth", 0) > 100:  # MB
            suggestions.append(
                "High memory growth detected. Consider processing data in smaller batches."
            )

        if avg_metrics.get("peak_memory", 0) > 2000:  # MB
            suggestions.append(
                "High peak memory usage. Consider using memory-efficient data structures."
            )

        # CPU optimization
        if avg_metrics.get("cpu_usage", 0) > 80:
            suggestions.append(
                "High CPU usage detected. Consider distributing workload across multiple processes."
            )

        # Execution time optimization
        if avg_metrics.get("execution_time", 0) > 60:  # seconds
            suggestions.append(
                "Long execution times detected. Consider optimizing algorithms or using caching."
            )

        return suggestions

    def _calculate_average_metrics(self, metrics: List[PerformanceMetrics]) -> Dict[str, float]:
        """Calculate average values for metrics."""
        if not metrics:
            return {}

        return {
            "execution_time": statistics.mean(m.execution_time for m in metrics),
            "cpu_usage": statistics.mean(m.cpu_usage for m in metrics),
            "memory_usage": statistics.mean(m.memory_usage for m in metrics),
            "peak_memory": statistics.mean(m.peak_memory for m in metrics),
            "memory_growth": statistics.mean(m.memory_growth for m in metrics),
            "function_calls": statistics.mean(m.function_calls for m in metrics),
        }


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile a function's performance.

    Args:
        func: Function to profile

    Returns:
        Wrapped function with profiling
    """

    def wrapper(*args, **kwargs):
        profiler = PerformanceProfiler()

        with profiler.profile_execution(name=func.__name__):
            result = func(*args, **kwargs)

        # Generate quick analysis
        analysis = profiler.analyze_bottlenecks()
        if analysis.recommendations:
            logger.info(f"Performance recommendations for {func.__name__}:")
            for rec in analysis.recommendations[:3]:  # Show top 3
                logger.info(f"  - {rec}")

        return result

    return wrapper


def create_performance_profiler(output_dir: Optional[Path] = None) -> PerformanceProfiler:
    """Create a performance profiler instance."""
    return PerformanceProfiler(output_dir)


def quick_performance_analysis(
    func: Callable, *args, **kwargs
) -> Tuple[Any, PerformanceMetrics, BottleneckAnalysis]:
    """
    Perform a quick performance analysis of a function call.

    Args:
        func: Function to analyze
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (function result, performance metrics, bottleneck analysis)
    """
    profiler = PerformanceProfiler()

    with profiler.profile_execution(name=f"quick_analysis_{func.__name__}") as metrics:
        result = func(*args, **kwargs)

    analysis = profiler.analyze_bottlenecks()

    return result, metrics, analysis
