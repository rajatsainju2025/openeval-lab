"""Enterprise observability platform with distributed tracing and structured logging.

This module provides comprehensive observability features including distributed tracing,
structured logging, metrics collection, and real-time monitoring for production-ready
evaluation deployments.
"""

import json
import time
import uuid
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from contextlib import contextmanager
import logging
import structlog
from collections import defaultdict, deque

try:
    import opentelemetry
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

from .enhanced_logging import get_logger
from .unified_config import ObservabilityConfig

logger = get_logger(__name__)


class LogLevel(Enum):
    """Log levels with structured logging support."""
    DEBUG = "debug"
    INFO = "info" 
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SpanKind(Enum):
    """Span kinds for tracing."""
    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class TraceContext:
    """Trace context for request correlation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    service: str
    version: str
    trace_context: Optional[TraceContext] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    duration: Optional[float] = None


@dataclass  
class MetricPoint:
    """Metric data point."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # counter, gauge, histogram


@dataclass
class Span:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    kind: SpanKind = SpanKind.INTERNAL


class StructuredLogger:
    """Enhanced structured logger with trace correlation."""
    
    def __init__(self, service_name: str, version: str = "1.0.0", 
                 log_level: LogLevel = LogLevel.INFO,
                 correlation_enabled: bool = True):
        self.service_name = service_name
        self.version = version
        self.log_level = log_level
        self.correlation_enabled = correlation_enabled
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, log_level.value.upper())
            ),
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True
        )
        
        self._logger = structlog.get_logger()
        
        # Local log storage for analysis
        self._log_buffer = deque(maxlen=10000)
        self._lock = threading.RLock()
    
    def _get_trace_context(self) -> Optional[TraceContext]:
        """Get current trace context if available."""
        if not self.correlation_enabled:
            return None
            
        # Try to get from thread-local storage or OpenTelemetry
        trace_context = getattr(_thread_local, 'trace_context', None)
        return trace_context
    
    def _create_log_entry(self, level: LogLevel, message: str, 
                         **kwargs) -> LogEntry:
        """Create structured log entry."""
        return LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            service=self.service_name,
            version=self.version,
            trace_context=self._get_trace_context(),
            fields=kwargs,
            exception=kwargs.get('exc_info')
        )
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method."""
        entry = self._create_log_entry(level, message, **kwargs)
        
        # Store in buffer
        with self._lock:
            self._log_buffer.append(entry)
        
        # Log using structlog
        log_data = asdict(entry)
        
        # Remove None values
        log_data = {k: v for k, v in log_data.items() if v is not None}
        
        # Convert datetime to ISO string
        log_data['timestamp'] = entry.timestamp.isoformat()
        
        # Log at appropriate level
        getattr(self._logger, level.value)(message, **log_data)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message.""" 
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def get_recent_logs(self, limit: int = 100) -> List[LogEntry]:
        """Get recent log entries."""
        with self._lock:
            return list(self._log_buffer)[-limit:]
    
    def search_logs(self, query: str, time_range: Optional[timedelta] = None) -> List[LogEntry]:
        """Search logs by message content or fields."""
        with self._lock:
            logs = list(self._log_buffer)
        
        # Filter by time range
        if time_range:
            cutoff = datetime.now() - time_range
            logs = [log for log in logs if log.timestamp >= cutoff]
        
        # Search in message and fields
        matching_logs = []
        for log in logs:
            if query.lower() in log.message.lower():
                matching_logs.append(log)
            elif any(query.lower() in str(v).lower() for v in log.fields.values()):
                matching_logs.append(log)
        
        return matching_logs


class DistributedTracer:
    """Distributed tracing implementation."""
    
    def __init__(self, service_name: str, jaeger_endpoint: Optional[str] = None,
                 sample_rate: float = 0.1):
        self.service_name = service_name
        self.sample_rate = sample_rate
        
        self._spans: Dict[str, Span] = {}
        self._active_spans: Dict[int, str] = {}  # thread_id -> span_id
        self._lock = threading.RLock()
        
        # Initialize OpenTelemetry if available
        if HAS_OTEL and jaeger_endpoint:
            self._init_opentelemetry(jaeger_endpoint)
        else:
            logger.warning("OpenTelemetry not available or no endpoint configured")
            self._tracer_provider = None
    
    def _init_opentelemetry(self, jaeger_endpoint: str):
        """Initialize OpenTelemetry tracing."""
        try:
            # Set up tracer provider
            trace.set_tracer_provider(TracerProvider())
            
            # Configure Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_endpoint.split(':')[0],
                agent_port=int(jaeger_endpoint.split(':')[1]),
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Instrument HTTP libraries
            RequestsInstrumentor().instrument()
            HTTPXClientInstrumentor().instrument()
            
            self._tracer = trace.get_tracer(self.service_name)
            logger.info(f"OpenTelemetry initialized with Jaeger endpoint: {jaeger_endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self._tracer = None
    
    def _should_sample(self) -> bool:
        """Determine if trace should be sampled."""
        import random
        return random.random() < self.sample_rate
    
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None,
                   tags: Optional[Dict[str, Any]] = None,
                   kind: SpanKind = SpanKind.INTERNAL) -> str:
        """Start a new span."""
        if not self._should_sample():
            return ""
        
        span_id = str(uuid.uuid4())
        
        # Get trace_id from parent span if available
        if parent_span_id and parent_span_id in self._spans:
            trace_id = self._spans[parent_span_id].trace_id
        else:
            trace_id = str(uuid.uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=datetime.now(),
            tags=tags or {},
            kind=kind
        )
        
        with self._lock:
            self._spans[span_id] = span
            self._active_spans[threading.get_ident()] = span_id
        
        # Set trace context in thread-local storage
        _thread_local.trace_context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id
        )
        
        return span_id
    
    def finish_span(self, span_id: str, status: str = "ok", 
                    tags: Optional[Dict[str, Any]] = None):
        """Finish a span."""
        if not span_id:
            return
            
        with self._lock:
            if span_id in self._spans:
                span = self._spans[span_id]
                span.end_time = datetime.now()
                span.duration = (span.end_time - span.start_time).total_seconds()
                span.status = status
                
                if tags:
                    span.tags.update(tags)
                
                # Remove from active spans
                thread_id = threading.get_ident()
                if self._active_spans.get(thread_id) == span_id:
                    del self._active_spans[thread_id]
    
    def add_span_log(self, span_id: str, message: str, **fields):
        """Add log entry to span."""
        if not span_id:
            return
            
        with self._lock:
            if span_id in self._spans:
                self._spans[span_id].logs.append({
                    'timestamp': datetime.now().isoformat(),
                    'message': message,
                    **fields
                })
    
    def set_span_tag(self, span_id: str, key: str, value: Any):
        """Set tag on span."""
        if not span_id:
            return
            
        with self._lock:
            if span_id in self._spans:
                self._spans[span_id].tags[key] = value
    
    @contextmanager
    def span(self, operation_name: str, tags: Optional[Dict[str, Any]] = None,
             kind: SpanKind = SpanKind.INTERNAL):
        """Context manager for spans."""
        span_id = self.start_span(operation_name, tags=tags, kind=kind)
        try:
            yield span_id
        except Exception as e:
            self.set_span_tag(span_id, 'error', True)
            self.add_span_log(span_id, f"Exception: {str(e)}")
            self.finish_span(span_id, status="error")
            raise
        else:
            self.finish_span(span_id)
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        with self._lock:
            return [span for span in self._spans.values() if span.trace_id == trace_id]
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """Get span by ID."""
        with self._lock:
            return self._spans.get(span_id)


class MetricsCollector:
    """Metrics collection and aggregation."""
    
    def __init__(self, service_name: str, registry: Optional[Any] = None):
        self.service_name = service_name
        
        if HAS_PROMETHEUS:
            self.registry = registry or CollectorRegistry()
            self._init_prometheus_metrics()
        else:
            logger.warning("Prometheus client not available")
            self.registry = None
            
        self._custom_metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        if not HAS_PROMETHEUS:
            return
            
        self.request_counter = Counter(
            'openeval_requests_total',
            'Total number of requests',
            ['service', 'method', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'openeval_request_duration_seconds',
            'Request duration in seconds',
            ['service', 'method'],
            registry=self.registry
        )
        
        self.evaluation_gauge = Gauge(
            'openeval_evaluations_active',
            'Number of active evaluations',
            ['service'],
            registry=self.registry
        )
        
        self.error_counter = Counter(
            'openeval_errors_total',
            'Total number of errors',
            ['service', 'error_type'],
            registry=self.registry
        )
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        if tags is None:
            tags = {}
            
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags,
            metric_type="counter"
        )
        
        with self._lock:
            self._custom_metrics[name].append(metric_point)
        
        # Update Prometheus if available
        if HAS_PROMETHEUS and hasattr(self, 'request_counter') and name == 'requests':
            self.request_counter.labels(
                service=self.service_name,
                method=tags.get('method', 'unknown'),
                status=tags.get('status', 'unknown')
            ).inc(value)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set gauge metric value."""
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type="gauge"
        )
        
        with self._lock:
            # For gauges, keep only the latest value
            self._custom_metrics[name] = [metric_point]
        
        # Update Prometheus if available
        if HAS_PROMETHEUS and hasattr(self, 'evaluation_gauge') and name == 'active_evaluations':
            self.evaluation_gauge.labels(service=self.service_name).set(value)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record histogram metric value."""
        if tags is None:
            tags = {}
            
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags,
            metric_type="histogram"
        )
        
        with self._lock:
            self._custom_metrics[name].append(metric_point)
        
        # Update Prometheus if available
        if HAS_PROMETHEUS and hasattr(self, 'request_duration') and name == 'request_duration':
            self.request_duration.labels(
                service=self.service_name,
                method=tags.get('method', 'unknown')
            ).observe(value)
    
    def get_metric_values(self, name: str, time_range: Optional[timedelta] = None) -> List[MetricPoint]:
        """Get metric values within time range."""
        with self._lock:
            metrics = self._custom_metrics.get(name, [])
        
        if time_range:
            cutoff = datetime.now() - time_range
            metrics = [m for m in metrics if m.timestamp >= cutoff]
        
        return metrics
    
    def get_all_metrics(self) -> Dict[str, List[MetricPoint]]:
        """Get all collected metrics."""
        with self._lock:
            return dict(self._custom_metrics)


class ObservabilityPlatform:
    """Main observability platform coordinating all components."""
    
    def __init__(self, config: ObservabilityConfig, service_name: str = "openeval",
                 version: str = "1.0.0"):
        self.config = config
        self.service_name = service_name
        self.version = version
        
        # Initialize components
        self.logger = StructuredLogger(
            service_name=service_name,
            version=version,
            correlation_enabled=config.log_correlation_id
        )
        
        self.tracer = DistributedTracer(
            service_name=service_name,
            jaeger_endpoint=config.tracing_endpoint if config.tracing_enabled else None,
            sample_rate=config.trace_sample_rate
        )
        
        self.metrics = MetricsCollector(service_name=service_name)
        
        # Start metrics server if enabled
        if config.metrics_enabled and HAS_PROMETHEUS:
            self._start_metrics_server()
        
        # Health check endpoints
        self._health_checks: Dict[str, Callable[[], bool]] = {}
        
        logger.info(f"ObservabilityPlatform initialized for service: {service_name}")
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server."""
        if not HAS_PROMETHEUS:
            return
            
        try:
            from prometheus_client import start_http_server
            start_http_server(self.config.metrics_port, registry=self.metrics.registry)
            self.logger.info(f"Metrics server started on port {self.config.metrics_port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
    
    def add_health_check(self, name: str, check_func: Callable[[], bool]):
        """Add a health check function."""
        self._health_checks[name] = check_func
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        health_status = {
            'service': self.service_name,
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'checks': {}
        }
        
        overall_healthy = True
        
        for name, check_func in self._health_checks.items():
            try:
                is_healthy = check_func()
                health_status['checks'][name] = {
                    'status': 'pass' if is_healthy else 'fail'
                }
                if not is_healthy:
                    overall_healthy = False
            except Exception as e:
                health_status['checks'][name] = {
                    'status': 'fail',
                    'error': str(e)
                }
                overall_healthy = False
        
        health_status['status'] = 'healthy' if overall_healthy else 'unhealthy'
        return health_status
    
    @contextmanager
    def trace_operation(self, operation_name: str, tags: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations."""
        with self.tracer.span(operation_name, tags=tags) as span_id:
            start_time = time.time()
            
            # Record operation start
            self.logger.info(f"Starting operation: {operation_name}", 
                           operation=operation_name, span_id=span_id)
            
            try:
                yield span_id
                
                # Record success metrics
                duration = time.time() - start_time
                self.metrics.record_histogram('operation_duration', duration, 
                                            {'operation': operation_name, 'status': 'success'})
                self.metrics.increment_counter('operations', 1, 
                                             {'operation': operation_name, 'status': 'success'})
                
                self.logger.info(f"Completed operation: {operation_name}", 
                               operation=operation_name, duration=duration)
                
            except Exception as e:
                # Record error metrics
                duration = time.time() - start_time
                self.metrics.increment_counter('operations', 1, 
                                             {'operation': operation_name, 'status': 'error'})
                
                self.logger.error(f"Failed operation: {operation_name}", 
                                operation=operation_name, duration=duration, error=str(e))
                raise
    
    def get_observability_summary(self) -> Dict[str, Any]:
        """Get comprehensive observability summary."""
        summary = {
            'service': self.service_name,
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'health': self.get_health_status(),
            'logging': {
                'recent_log_count': len(self.logger.get_recent_logs()),
                'error_log_count': len([log for log in self.logger.get_recent_logs() 
                                      if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]),
            },
            'tracing': {
                'active_spans': len(self.tracer._active_spans),
                'total_spans': len(self.tracer._spans),
            },
            'metrics': {
                'custom_metrics_count': len(self.metrics.get_all_metrics()),
                'prometheus_enabled': HAS_PROMETHEUS and self.config.metrics_enabled,
            }
        }
        
        return summary
    
    def export_observability_data(self, output_path: Path, include_logs: bool = True,
                                 include_traces: bool = True, include_metrics: bool = True):
        """Export observability data for analysis."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'service': self.service_name,
            'version': self.version,
            'config': asdict(self.config)
        }
        
        if include_logs:
            export_data['logs'] = [asdict(log) for log in self.logger.get_recent_logs()]
        
        if include_traces:
            export_data['traces'] = {
                span_id: asdict(span) for span_id, span in self.tracer._spans.items()
            }
        
        if include_metrics:
            metrics_data = {}
            for name, points in self.metrics.get_all_metrics().items():
                metrics_data[name] = [asdict(point) for point in points]
            export_data['metrics'] = metrics_data
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Observability data exported to: {output_path}")


# Thread-local storage for trace context
_thread_local = threading.local()

# Global observability platform instance
_observability_platform: Optional[ObservabilityPlatform] = None


def get_observability_platform(config: Optional[ObservabilityConfig] = None,
                               service_name: str = "openeval") -> ObservabilityPlatform:
    """Get global observability platform instance."""
    global _observability_platform
    if _observability_platform is None:
        if config is None:
            from .unified_config import load_config
            unified_config = load_config()
            config = unified_config.observability
        _observability_platform = ObservabilityPlatform(config, service_name)
    return _observability_platform


# Convenience functions
def trace_operation(operation_name: str, tags: Optional[Dict[str, Any]] = None):
    """Decorator for tracing operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            platform = get_observability_platform()
            with platform.trace_operation(operation_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_event(level: str, message: str, **kwargs):
    """Log structured event."""
    platform = get_observability_platform()
    getattr(platform.logger, level)(message, **kwargs)


def record_metric(name: str, value: Union[int, float], metric_type: str = "gauge",
                  tags: Optional[Dict[str, str]] = None):
    """Record custom metric."""
    platform = get_observability_platform()
    
    if metric_type == "counter":
        platform.metrics.increment_counter(name, int(value), tags)
    elif metric_type == "histogram":
        platform.metrics.record_histogram(name, float(value), tags)
    else:  # gauge
        platform.metrics.set_gauge(name, float(value), tags)