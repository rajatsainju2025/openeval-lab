"""Telemetry system for explainer performance monitoring.

This module provides comprehensive telemetry collection for tracking
explainer performance, latency, usage statistics, and resource consumption.
"""

import time
import threading
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class MetricType(Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class MetricValue:
    """A single metric measurement."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit,
        }


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    values: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        """Calculate mean latency."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def p50_ms(self) -> float:
        """Calculate 50th percentile (median)."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        return sorted_values[len(sorted_values) // 2]

    @property
    def p95_ms(self) -> float:
        """Calculate 95th percentile."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * 0.95)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    @property
    def p99_ms(self) -> float:
        """Calculate 99th percentile."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * 0.99)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    @property
    def std_dev_ms(self) -> float:
        """Calculate standard deviation."""
        if len(self.values) < 2:
            return 0.0
        return statistics.stdev(self.values)

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        self.values.append(latency_ms)

        # Keep only last 1000 values for percentile calculations
        if len(self.values) > 1000:
            self.values = self.values[-1000:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "total_ms": self.total_ms,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms if self.min_ms != float("inf") else 0.0,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "std_dev_ms": self.std_dev_ms,
        }


@dataclass
class ThroughputStats:
    """Statistics for throughput measurements."""

    window_seconds: float = 60.0
    timestamps: deque = field(default_factory=lambda: deque())

    def record(self) -> None:
        """Record an event."""
        now = time.time()
        self.timestamps.append(now)
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove old timestamps outside the window."""
        cutoff = time.time() - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    @property
    def requests_per_second(self) -> float:
        """Calculate current requests per second."""
        self._cleanup()
        if len(self.timestamps) < 2:
            return 0.0
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 0:
            return 0.0
        return len(self.timestamps) / duration

    @property
    def requests_per_minute(self) -> float:
        """Calculate requests per minute."""
        return self.requests_per_second * 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requests_per_second": self.requests_per_second,
            "requests_per_minute": self.requests_per_minute,
            "window_seconds": self.window_seconds,
            "total_in_window": len(self.timestamps),
        }


@dataclass
class UsageStats:
    """Usage statistics for explainers."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_processed: int = 0
    total_characters_processed: int = 0
    by_level: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_model: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_explainer: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return 1.0 - self.success_rate

    def record_request(
        self,
        success: bool,
        level: str = "summary",
        model: Optional[str] = None,
        explainer: Optional[str] = None,
        tokens: int = 0,
        characters: int = 0,
    ) -> None:
        """Record a request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_tokens_processed += tokens
        self.total_characters_processed += characters
        self.by_level[level] += 1

        if model:
            self.by_model[model] += 1
        if explainer:
            self.by_explainer[explainer] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "total_tokens_processed": self.total_tokens_processed,
            "total_characters_processed": self.total_characters_processed,
            "by_level": dict(self.by_level),
            "by_model": dict(self.by_model),
            "by_explainer": dict(self.by_explainer),
        }


class TelemetryBackend(ABC):
    """Abstract base class for telemetry backends."""

    @abstractmethod
    def record_metric(self, metric: MetricValue) -> None:
        """Record a metric value."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered metrics."""
        pass


class InMemoryBackend(TelemetryBackend):
    """In-memory telemetry backend for local storage."""

    def __init__(self, max_metrics: int = 10000) -> None:
        """Initialize in-memory backend.

        Args:
            max_metrics: Maximum metrics to retain.
        """
        self._metrics: deque = deque(maxlen=max_metrics)
        self._lock = threading.Lock()

    def record_metric(self, metric: MetricValue) -> None:
        """Record a metric value."""
        with self._lock:
            self._metrics.append(metric)

    def flush(self) -> None:
        """No-op for in-memory backend."""
        pass

    def get_metrics(
        self,
        name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[MetricValue]:
        """Query metrics.

        Args:
            name: Filter by metric name.
            start_time: Filter by start time.
            end_time: Filter by end time.

        Returns:
            List of matching metrics.
        """
        with self._lock:
            result = list(self._metrics)

        if name:
            result = [m for m in result if m.name == name]
        if start_time:
            result = [m for m in result if m.timestamp >= start_time]
        if end_time:
            result = [m for m in result if m.timestamp <= end_time]

        return result


class LoggingBackend(TelemetryBackend):
    """Telemetry backend that logs metrics."""

    def __init__(self, logger_name: str = "explainer.telemetry") -> None:
        """Initialize logging backend.

        Args:
            logger_name: Name for the logger.
        """
        import logging

        self._logger = logging.getLogger(logger_name)

    def record_metric(self, metric: MetricValue) -> None:
        """Log a metric value."""
        tags_str = ",".join(f"{k}={v}" for k, v in metric.tags.items())
        self._logger.info(
            f"METRIC {metric.name}={metric.value} "
            f"type={metric.metric_type.value} "
            f"tags=[{tags_str}]"
        )

    def flush(self) -> None:
        """No-op for logging backend."""
        pass


class ExplainerTelemetry:
    """Telemetry collector for explainer operations.

    Collects and aggregates performance metrics, latency data,
    throughput statistics, and usage information.
    """

    def __init__(
        self,
        backend: Optional[TelemetryBackend] = None,
        enable_detailed_latency: bool = True,
        enable_throughput: bool = True,
        throughput_window: float = 60.0,
    ) -> None:
        """Initialize telemetry collector.

        Args:
            backend: Telemetry backend for storage.
            enable_detailed_latency: Track detailed latency percentiles.
            enable_throughput: Track throughput statistics.
            throughput_window: Window size for throughput in seconds.
        """
        self._backend = backend or InMemoryBackend()
        self._enable_detailed_latency = enable_detailed_latency
        self._enable_throughput = enable_throughput

        # Latency tracking per operation
        self._latency: Dict[str, LatencyStats] = defaultdict(LatencyStats)

        # Throughput tracking per operation
        self._throughput: Dict[str, ThroughputStats] = defaultdict(
            lambda: ThroughputStats(window_seconds=throughput_window)
        )

        # Usage statistics
        self._usage = UsageStats()

        # Error tracking
        self._errors: Dict[str, int] = defaultdict(int)

        # Lock for thread safety
        self._lock = threading.RLock()

        # Session info
        self._start_time = datetime.now()
        self._enabled = True

    def enable(self) -> None:
        """Enable telemetry collection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable telemetry collection."""
        self._enabled = False

    @contextmanager
    def measure_latency(self, operation: str, tags: Optional[Dict[str, str]] = None):
        """Context manager to measure operation latency.

        Args:
            operation: Name of the operation being measured.
            tags: Optional tags for the metric.

        Yields:
            None
        """
        if not self._enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            with self._lock:
                self._latency[operation].record(elapsed_ms)

                if self._enable_throughput:
                    self._throughput[operation].record()

            self._backend.record_metric(
                MetricValue(
                    name=f"{operation}.latency",
                    value=elapsed_ms,
                    metric_type=MetricType.TIMER,
                    tags=tags or {},
                    unit="ms",
                )
            )

    def record_latency(
        self, operation: str, latency_ms: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Manually record latency for an operation.

        Args:
            operation: Name of the operation.
            latency_ms: Latency in milliseconds.
            tags: Optional tags for the metric.
        """
        if not self._enabled:
            return

        with self._lock:
            self._latency[operation].record(latency_ms)

            if self._enable_throughput:
                self._throughput[operation].record()

        self._backend.record_metric(
            MetricValue(
                name=f"{operation}.latency",
                value=latency_ms,
                metric_type=MetricType.TIMER,
                tags=tags or {},
                unit="ms",
            )
        )

    def record_counter(
        self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a counter metric.

        Args:
            name: Metric name.
            value: Value to add to counter.
            tags: Optional tags.
        """
        if not self._enabled:
            return

        self._backend.record_metric(
            MetricValue(
                name=name,
                value=value,
                metric_type=MetricType.COUNTER,
                tags=tags or {},
            )
        )

    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric.

        Args:
            name: Metric name.
            value: Current value.
            tags: Optional tags.
        """
        if not self._enabled:
            return

        self._backend.record_metric(
            MetricValue(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                tags=tags or {},
            )
        )

    def record_request(
        self,
        success: bool,
        level: str = "summary",
        model: Optional[str] = None,
        explainer: Optional[str] = None,
        tokens: int = 0,
        characters: int = 0,
        error_type: Optional[str] = None,
    ) -> None:
        """Record an explanation request.

        Args:
            success: Whether the request succeeded.
            level: Explanation level.
            model: Model used.
            explainer: Explainer type used.
            tokens: Number of tokens processed.
            characters: Number of characters processed.
            error_type: Type of error if failed.
        """
        if not self._enabled:
            return

        with self._lock:
            self._usage.record_request(
                success=success,
                level=level,
                model=model,
                explainer=explainer,
                tokens=tokens,
                characters=characters,
            )

            if not success and error_type:
                self._errors[error_type] += 1

        # Record as metrics
        self.record_counter("requests.total", tags={"level": level})
        if success:
            self.record_counter("requests.success", tags={"level": level})
        else:
            self.record_counter("requests.failure", tags={"level": level})

        if tokens > 0:
            self.record_counter("tokens.processed", value=tokens)
        if characters > 0:
            self.record_counter("characters.processed", value=characters)

    def get_latency_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get latency statistics.

        Args:
            operation: Specific operation to get stats for.

        Returns:
            Latency statistics dictionary.
        """
        with self._lock:
            if operation:
                return {operation: self._latency[operation].to_dict()}
            return {op: stats.to_dict() for op, stats in self._latency.items()}

    def get_throughput_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get throughput statistics.

        Args:
            operation: Specific operation to get stats for.

        Returns:
            Throughput statistics dictionary.
        """
        with self._lock:
            if operation:
                return {operation: self._throughput[operation].to_dict()}
            return {op: stats.to_dict() for op, stats in self._throughput.items()}

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Usage statistics dictionary.
        """
        with self._lock:
            return self._usage.to_dict()

    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics.

        Returns:
            Error counts by type.
        """
        with self._lock:
            return dict(self._errors)

    def get_summary(self) -> Dict[str, Any]:
        """Get a complete telemetry summary.

        Returns:
            Complete telemetry summary dictionary.
        """
        with self._lock:
            uptime = datetime.now() - self._start_time
            return {
                "start_time": self._start_time.isoformat(),
                "uptime_seconds": uptime.total_seconds(),
                "enabled": self._enabled,
                "usage": self._usage.to_dict(),
                "latency": {op: stats.to_dict() for op, stats in self._latency.items()},
                "throughput": {op: stats.to_dict() for op, stats in self._throughput.items()},
                "errors": dict(self._errors),
            }

    def reset(self) -> None:
        """Reset all telemetry data."""
        with self._lock:
            self._latency.clear()
            self._throughput.clear()
            self._usage = UsageStats()
            self._errors.clear()
            self._start_time = datetime.now()

    def flush(self) -> None:
        """Flush metrics to backend."""
        self._backend.flush()


def timed(telemetry: ExplainerTelemetry, operation: str) -> Callable[[F], F]:
    """Decorator to measure function execution time.

    Args:
        telemetry: Telemetry instance to record to.
        operation: Name of the operation.

    Returns:
        Decorated function.

    Example:
        @timed(telemetry, "explain_code")
        def explain_code(code: str) -> str:
            return generate_explanation(code)
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with telemetry.measure_latency(operation):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def async_timed(telemetry: ExplainerTelemetry, operation: str) -> Callable[[F], F]:
    """Decorator to measure async function execution time.

    Args:
        telemetry: Telemetry instance to record to.
        operation: Name of the operation.

    Returns:
        Decorated async function.

    Example:
        @async_timed(telemetry, "explain_code_async")
        async def explain_code_async(code: str) -> str:
            return await generate_explanation_async(code)
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with telemetry.measure_latency(operation):
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class TelemetryExporter:
    """Export telemetry data to various formats."""

    def __init__(self, telemetry: ExplainerTelemetry) -> None:
        """Initialize exporter.

        Args:
            telemetry: Telemetry instance to export from.
        """
        self._telemetry = telemetry

    def to_json(self, include_raw_metrics: bool = False) -> str:
        """Export telemetry as JSON.

        Args:
            include_raw_metrics: Include raw metric values.

        Returns:
            JSON string.
        """
        import json

        data = self._telemetry.get_summary()

        if include_raw_metrics and isinstance(self._telemetry._backend, InMemoryBackend):
            data["raw_metrics"] = [m.to_dict() for m in self._telemetry._backend.get_metrics()]

        return json.dumps(data, indent=2, default=str)

    def to_prometheus(self) -> str:
        """Export telemetry in Prometheus format.

        Returns:
            Prometheus-formatted metrics string.
        """
        lines = []
        summary = self._telemetry.get_summary()

        # Usage metrics
        usage = summary["usage"]
        lines.append("# HELP explainer_requests_total Total number of requests")
        lines.append("# TYPE explainer_requests_total counter")
        lines.append(f'explainer_requests_total {usage["total_requests"]}')

        lines.append("# HELP explainer_requests_success Successful requests")
        lines.append("# TYPE explainer_requests_success counter")
        lines.append(f'explainer_requests_success {usage["successful_requests"]}')

        lines.append("# HELP explainer_success_rate Success rate")
        lines.append("# TYPE explainer_success_rate gauge")
        lines.append(f'explainer_success_rate {usage["success_rate"]:.4f}')

        # Latency metrics
        for op, stats in summary["latency"].items():
            safe_op = op.replace(".", "_")
            lines.append(f"# HELP explainer_{safe_op}_latency_ms Latency in ms")
            lines.append(f"# TYPE explainer_{safe_op}_latency_ms summary")
            lines.append(f'explainer_{safe_op}_latency_ms{{quantile="0.5"}} {stats["p50_ms"]:.4f}')
            lines.append(f'explainer_{safe_op}_latency_ms{{quantile="0.95"}} {stats["p95_ms"]:.4f}')
            lines.append(f'explainer_{safe_op}_latency_ms{{quantile="0.99"}} {stats["p99_ms"]:.4f}')
            lines.append(f'explainer_{safe_op}_latency_ms_count {stats["count"]}')
            lines.append(f'explainer_{safe_op}_latency_ms_sum {stats["total_ms"]:.4f}')

        # Throughput metrics
        for op, stats in summary["throughput"].items():
            safe_op = op.replace(".", "_")
            lines.append(f"# HELP explainer_{safe_op}_rps Requests per second")
            lines.append(f"# TYPE explainer_{safe_op}_rps gauge")
            lines.append(f'explainer_{safe_op}_rps {stats["requests_per_second"]:.4f}')

        return "\n".join(lines)


# Global telemetry instance
_global_telemetry: Optional[ExplainerTelemetry] = None


def get_telemetry(
    backend: Optional[TelemetryBackend] = None,
) -> ExplainerTelemetry:
    """Get or create the global telemetry instance.

    Args:
        backend: Optional backend (only used on first call).

    Returns:
        Global ExplainerTelemetry instance.
    """
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = ExplainerTelemetry(backend=backend)
    return _global_telemetry


def reset_telemetry() -> None:
    """Reset the global telemetry instance."""
    global _global_telemetry
    if _global_telemetry is not None:
        _global_telemetry.reset()
    _global_telemetry = None
