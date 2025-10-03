"""Performance benchmarking and optimization tools."""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
import statistics
import json
from pathlib import Path
import functools
from concurrent.futures import ThreadPoolExecutor

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import aiofiles

    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    aiofiles = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation."""

    # Timing metrics
    wall_time: float
    cpu_time: float

    # Memory metrics
    peak_memory_mb: float
    memory_delta_mb: float

    # System metrics
    cpu_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float

    # Operation metrics
    throughput: Optional[float] = None  # items per second
    latency_stats: Optional[Dict[str, float]] = None  # p50, p95, p99

    # Context
    operation_name: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor and collect performance metrics."""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self._start_time: Optional[float] = None
        self._start_cpu_time: Optional[float] = None
        self._start_memory: Optional[float] = None
        self._start_disk_io: Optional[Dict[str, int]] = None
        self._peak_memory: float = 0.0
        self._operation_name = ""
        self._metadata: Dict[str, Any] = {}

        # For latency tracking
        self._latencies: List[float] = []
        self._lock = threading.Lock()

    def start_monitoring(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Start monitoring performance."""
        self._operation_name = operation_name
        self._metadata = metadata or {}

        # Record start state
        self._start_time = time.perf_counter()
        self._start_cpu_time = time.process_time()

        # Memory tracking (if psutil available)
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                process = psutil.Process()
                self._start_memory = process.memory_info().rss / 1024 / 1024  # MB
                self._peak_memory = self._start_memory if self._start_memory is not None else 0.0
            except Exception:
                self._start_memory = 0.0
                self._peak_memory = 0.0
        else:
            self._start_memory = 0.0
            self._peak_memory = 0.0

        # Disk I/O tracking (if psutil available)
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self._start_disk_io = {
                        "read_bytes": disk_io.read_bytes,
                        "write_bytes": disk_io.write_bytes,
                    }
            except Exception:
                self._start_disk_io = None
        else:
            self._start_disk_io = None

        # Start memory monitoring thread
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and return metrics."""
        if self._start_time is None:
            raise ValueError("Monitoring not started")

        # Stop monitoring thread
        self._monitoring = False
        if hasattr(self, "_monitor_thread"):
            self._monitor_thread.join(timeout=1.0)

        # Calculate final metrics
        end_time = time.perf_counter()
        end_cpu_time = time.process_time()

        wall_time = end_time - self._start_time
        cpu_time = end_cpu_time - (self._start_cpu_time or 0)

        # Memory metrics
        end_memory = 0.0
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                end_memory = 0.0

        memory_delta = end_memory - (self._start_memory or 0)

        # CPU utilization
        cpu_percent = 0.0
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                cpu_percent = psutil.cpu_percent()
            except Exception:
                cpu_percent = 0.0

        # Disk I/O metrics
        disk_read_mb = 0.0
        disk_write_mb = 0.0
        if self._start_disk_io and PSUTIL_AVAILABLE and psutil is not None:
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    disk_read_mb = (
                        (disk_io.read_bytes - self._start_disk_io["read_bytes"]) / 1024 / 1024
                    )
                    disk_write_mb = (
                        (disk_io.write_bytes - self._start_disk_io["write_bytes"]) / 1024 / 1024
                    )
            except Exception:
                pass

        # Latency statistics
        latency_stats = None
        if self._latencies:
            latency_stats = {
                "mean": statistics.mean(self._latencies),
                "median": statistics.median(self._latencies),
                "p50": statistics.median(self._latencies),
                "p95": self._percentile(self._latencies, 95),
                "p99": self._percentile(self._latencies, 99),
                "min": min(self._latencies),
                "max": max(self._latencies),
                "std": statistics.stdev(self._latencies) if len(self._latencies) > 1 else 0,
            }

        # Throughput calculation
        throughput = None
        if "item_count" in self._metadata and wall_time > 0:
            throughput = self._metadata["item_count"] / wall_time

        metrics = PerformanceMetrics(
            wall_time=wall_time,
            cpu_time=cpu_time,
            peak_memory_mb=self._peak_memory,
            memory_delta_mb=memory_delta,
            cpu_percent=cpu_percent,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            throughput=throughput,
            latency_stats=latency_stats,
            operation_name=self._operation_name,
            metadata=self._metadata,
        )

        self.metrics_history.append(metrics)

        # Reset state
        self._start_time = None
        self._latencies.clear()

        return metrics

    def record_latency(self, latency: float):
        """Record a latency measurement."""
        with self._lock:
            self._latencies.append(latency)

    def _monitor_memory(self):
        """Monitor peak memory usage in background thread."""
        if not PSUTIL_AVAILABLE or psutil is None:
            return

        try:
            process = psutil.Process()
            while getattr(self, "_monitoring", False):
                try:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    self._peak_memory = max(self._peak_memory, current_memory)
                    time.sleep(0.1)  # Check every 100ms
                except Exception:
                    break
        except Exception:
            pass

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile / 100
        f = int(k)
        c = k - f

        if f + 1 < len(sorted_data):
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
        else:
            return sorted_data[f]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded metrics."""
        if not self.metrics_history:
            return {}

        wall_times = [m.wall_time for m in self.metrics_history]
        memory_peaks = [m.peak_memory_mb for m in self.metrics_history]
        throughputs = [m.throughput for m in self.metrics_history if m.throughput is not None]

        summary = {
            "total_operations": len(self.metrics_history),
            "total_wall_time": sum(wall_times),
            "avg_wall_time": statistics.mean(wall_times),
            "max_wall_time": max(wall_times),
            "peak_memory_mb": max(memory_peaks),
            "avg_memory_mb": statistics.mean(memory_peaks),
        }

        if throughputs:
            summary.update(
                {"avg_throughput": statistics.mean(throughputs), "max_throughput": max(throughputs)}
            )

        return summary


class AsyncPerformanceMonitor:
    """Async version of PerformanceMonitor for asyncio applications."""

    def __init__(self, thread_pool: Optional[ThreadPoolExecutor] = None):
        self.monitor = PerformanceMonitor()
        self.thread_pool = thread_pool or ThreadPoolExecutor(max_workers=4)
        self._active_monitors: Dict[str, PerformanceMonitor] = {}
        self._lock = threading.Lock()

    @asynccontextmanager
    async def monitor_async(self, operation_name: str, **metadata):
        """Async context manager for performance monitoring."""
        loop = asyncio.get_event_loop()
        monitor = PerformanceMonitor()

        # Start monitoring in thread pool
        await loop.run_in_executor(
            self.thread_pool, monitor.start_monitoring, operation_name, metadata
        )

        try:
            yield monitor
        finally:
            # Stop monitoring in thread pool
            metrics = await loop.run_in_executor(self.thread_pool, monitor.stop_monitoring)
            with self._lock:
                self._active_monitors[operation_name] = monitor

    async def benchmark_async_function(
        self, func: Callable, *args, iterations: int = 100, **kwargs
    ) -> Dict[str, Any]:
        """Benchmark an async function."""
        latencies = []

        # Warmup
        for _ in range(min(10, iterations // 10)):
            await func(*args, **kwargs)

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            await func(*args, **kwargs)
            latencies.append(time.perf_counter() - start)

        return {
            "avg_latency": statistics.mean(latencies),
            "p50_latency": statistics.median(latencies),
            "p95_latency": statistics.quantiles(latencies, n=20)[18],  # 95th percentile
            "p99_latency": statistics.quantiles(latencies, n=100)[98],  # 99th percentile
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "throughput": iterations / sum(latencies),
        }


@contextmanager
def performance_context(monitor: PerformanceMonitor, operation_name: str, **metadata):
    """Context manager for performance monitoring."""
    monitor.start_monitoring(operation_name, metadata)
    try:
        yield monitor
    finally:
        metrics = monitor.stop_monitoring()
        return metrics


def benchmark_function(iterations: int = 100, warmup: int = 10):
    """Decorator to benchmark a function."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            results = []

            # Warmup runs
            for _ in range(warmup):
                try:
                    func(*args, **kwargs)
                except Exception:
                    pass

            # Benchmark runs
            for i in range(iterations):
                monitor.start_monitoring(f"{func.__name__}_iter_{i}")
                try:
                    result = func(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    results.append(e)
                finally:
                    monitor.stop_monitoring()

            # Return results and benchmark summary
            return {
                "results": results,
                "benchmark_summary": monitor.get_summary(),
                "detailed_metrics": monitor.metrics_history,
            }

        return wrapper

    return decorator


class PerformanceBenchmark:
    """Comprehensive benchmarking suite."""

    def __init__(self, name: str):
        self.name = name
        self.monitor = PerformanceMonitor()
        self.benchmarks: Dict[str, Any] = {}

    def add_benchmark(self, name: str, func: Callable, *args, iterations: int = 100, **kwargs):
        """Add a benchmark function."""

        @benchmark_function(iterations=iterations)
        def wrapped_func():
            return func(*args, **kwargs)

        result = wrapped_func()
        self.benchmarks[name] = result
        return result

    def run_comparison_benchmark(
        self, functions: Dict[str, Callable], *args, iterations: int = 100, **kwargs
    ):
        """Run comparison benchmark between multiple functions."""
        results = {}

        for name, func in functions.items():
            results[name] = self.add_benchmark(name, func, *args, iterations=iterations, **kwargs)

        # Create comparison summary
        comparison = {}
        for name, result in results.items():
            summary = result["benchmark_summary"]
            comparison[name] = {
                "avg_time": summary.get("avg_wall_time", 0),
                "max_time": summary.get("max_wall_time", 0),
                "peak_memory": summary.get("peak_memory_mb", 0),
                "throughput": summary.get("avg_throughput"),
            }

        return {
            "detailed_results": results,
            "comparison": comparison,
            "fastest": min(comparison.keys(), key=lambda k: comparison[k]["avg_time"]),
            "most_memory_efficient": min(
                comparison.keys(), key=lambda k: comparison[k]["peak_memory"]
            ),
        }

    def save_report(self, filepath: Path):
        """Save benchmark report to file."""
        report = {
            "benchmark_name": self.name,
            "timestamp": time.time(),
            "system_info": {
                "cpu_count": psutil.cpu_count() if PSUTIL_AVAILABLE and psutil else "unknown",
                "memory_total_gb": (
                    (psutil.virtual_memory().total / 1024 / 1024 / 1024)
                    if PSUTIL_AVAILABLE and psutil
                    else "unknown"
                ),
                "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            },
            "benchmarks": self.benchmarks,
            "summary": self.monitor.get_summary(),
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)


class AdapterPerformanceProfiler:
    """Specialized profiler for adapter performance."""

    def __init__(self):
        self.profiles: Dict[str, List[PerformanceMetrics]] = {}

    def profile_adapter(
        self, adapter, examples: List[Any], batch_sizes: Optional[List[int]] = None
    ):
        """Profile adapter performance with different batch sizes."""
        if batch_sizes is None:
            batch_sizes = [1, 5, 10, 20, 50]

        monitor = PerformanceMonitor()

        for batch_size in batch_sizes:
            # Create batches
            batches = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]

            for i, batch in enumerate(batches):
                monitor.start_monitoring(
                    f"adapter_batch_{batch_size}_{i}",
                    {"batch_size": batch_size, "item_count": len(batch)},
                )

                start_time = time.perf_counter()
                try:
                    # Process batch
                    if hasattr(adapter, "predict_batch"):
                        results = adapter.predict_batch(batch)
                    else:
                        results = [adapter.predict(ex) for ex in batch]

                    # Record individual latencies for throughput calculation
                    end_time = time.perf_counter()
                    batch_time = end_time - start_time
                    avg_latency = batch_time / len(batch)

                    for _ in batch:
                        monitor.record_latency(avg_latency)

                except Exception:
                    # Record error but continue profiling
                    pass

                metrics = monitor.stop_monitoring()

            # Store metrics for this batch size
            adapter_name = getattr(adapter, "__class__", type(adapter)).__name__
            key = f"{adapter_name}_batch_{batch_size}"

            if key not in self.profiles:
                self.profiles[key] = []

            self.profiles[key].extend(monitor.metrics_history)
            monitor.metrics_history.clear()

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Analyze profiles and provide optimization recommendations."""
        recommendations = []

        # Analyze batch size efficiency
        batch_sizes = {}
        for key, metrics_list in self.profiles.items():
            if "_batch_" in key:
                parts = key.split("_batch_")
                if len(parts) == 2:
                    adapter_name = parts[0]
                    batch_size = int(parts[1])

                    if adapter_name not in batch_sizes:
                        batch_sizes[adapter_name] = {}

                    avg_throughput = statistics.mean(
                        [m.throughput for m in metrics_list if m.throughput]
                    )
                    batch_sizes[adapter_name][batch_size] = avg_throughput

        for adapter_name, sizes in batch_sizes.items():
            if sizes:
                optimal_batch = max(sizes.keys(), key=lambda k: sizes[k])
                recommendations.append(
                    {
                        "type": "batch_size_optimization",
                        "adapter": adapter_name,
                        "recommendation": f"Use batch size {optimal_batch} for optimal throughput",
                        "throughput_data": sizes,
                    }
                )

        # Memory efficiency analysis
        high_memory_operations = []
        for key, metrics_list in self.profiles.items():
            avg_memory = statistics.mean([m.peak_memory_mb for m in metrics_list])
            if avg_memory > 1000:  # > 1GB
                high_memory_operations.append(
                    {
                        "operation": key,
                        "avg_memory_mb": avg_memory,
                        "recommendation": "Consider memory optimization or smaller batch sizes",
                    }
                )

        if high_memory_operations:
            recommendations.append(
                {"type": "memory_optimization", "high_memory_operations": high_memory_operations}
            )

        return {
            "recommendations": recommendations,
            "detailed_profiles": self.profiles,
            "summary": {
                "total_profiles": len(self.profiles),
                "total_operations": sum(len(metrics) for metrics in self.profiles.values()),
            },
        }


def create_performance_suite():
    """Create a comprehensive performance monitoring suite."""
    return {
        "monitor": PerformanceMonitor(),
        "benchmark": PerformanceBenchmark,
        "adapter_profiler": AdapterPerformanceProfiler(),
        "context": performance_context,
        "benchmark_decorator": benchmark_function,
    }
