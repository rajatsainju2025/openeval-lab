"""
Structured logging with performance metrics integration.

Enhances logging with structured metrics, performance tracking,
and better observability for debugging and optimization.
"""

import json
import logging
import time
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON.

        Args:
            record: Log record to format

        Returns:
            JSON formatted log string
        """
        log_data: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "metric_name"):
            log_data["metric"] = getattr(record, "metric_name", None)
        if hasattr(record, "metric_value"):
            log_data["value"] = getattr(record, "metric_value", None)
        if hasattr(record, "performance"):
            log_data["performance"] = getattr(record, "performance", None)

        return json.dumps(log_data)


class MetricsLogger:
    """Logger with built-in performance metrics tracking."""

    def __init__(self, name: str):
        """Initialize metrics logger.

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self._metrics: Dict[str, Dict[str, Any]] = {}

    def log_metric(self, name: str, value: float, unit: str = "") -> None:
        """Log a performance metric.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
        """
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"Metric: {name}={value}{unit}",
            args=(),
            exc_info=None,
        )
        record.metric_name = name
        record.metric_value = value

        # Store metric
        if name not in self._metrics:
            self._metrics[name] = {
                "values": [],
                "min": value,
                "max": value,
            }
        else:
            self._metrics[name]["values"].append(value)
            self._metrics[name]["min"] = min(self._metrics[name]["min"], value)
            self._metrics[name]["max"] = max(self._metrics[name]["max"], value)

        self.logger.handle(record)

    def log_performance(self, operation: str, duration: float, success: bool = True) -> None:
        """Log a performance event.

        Args:
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation succeeded
        """
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO if success else logging.WARNING,
            pathname="",
            lineno=0,
            msg=f"Performance: {operation}={duration*1000:.1f}ms",
            args=(),
            exc_info=None,
        )
        record.performance = {
            "operation": operation,
            "duration_ms": duration * 1000,
            "success": success,
        }

        self.logger.handle(record)

    def get_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of collected metrics.

        Returns:
            Dictionary with metrics summaries
        """
        summary = {}
        for name, data in self._metrics.items():
            values = data["values"]
            if values:
                avg = sum(values) / len(values)
            else:
                avg = 0

            summary[name] = {
                "count": len(values),
                "min": data["min"],
                "max": data["max"],
                "avg": avg,
            }

        return summary


def setup_structured_logging(
    logger_name: str = "openeval",
    log_file: Optional[str] = None,
) -> MetricsLogger:
    """Setup structured logging for OpenEval.

    Args:
        logger_name: Logger name
        log_file: Optional log file path

    Returns:
        MetricsLogger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with structured formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    return MetricsLogger(logger_name)


# Global structured logger
_metrics_logger: Optional[MetricsLogger] = None


def get_metrics_logger(name: str = "openeval") -> MetricsLogger:
    """Get global metrics logger.

    Args:
        name: Logger name

    Returns:
        MetricsLogger instance
    """
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = setup_structured_logging(name)
    return _metrics_logger


def log_metric(name: str, value: float, unit: str = "") -> None:
    """Log a metric to global logger.

    Args:
        name: Metric name
        value: Metric value
        unit: Unit of measurement
    """
    logger = get_metrics_logger()
    logger.log_metric(name, value, unit)


def log_performance(operation: str, duration: float, success: bool = True) -> None:
    """Log performance to global logger.

    Args:
        operation: Operation name
        duration: Duration in seconds
        success: Whether operation succeeded
    """
    logger = get_metrics_logger()
    logger.log_performance(operation, duration, success)
