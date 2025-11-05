"""Enhanced logging and error handling for OpenEval Lab."""

import logging
import sys
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    structlog = None
    HAS_STRUCTLOG = False


class OpenEvalLogger:
    """Enterprise-grade logging system with structured logging and multiple handlers."""

    def __init__(
        self,
        name: str = "openeval",
        log_dir: Optional[Path] = None,
        log_level: str = "INFO",
        use_structlog: bool = True,
        enable_audit: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ):
        """Initialize enterprise logger with multiple handlers and structured logging."""
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.use_structlog = use_structlog and HAS_STRUCTLOG
        self.enable_audit = enable_audit
        self.audit_log: List[Dict[str, Any]] = []
        self.audit_lock = threading.Lock()

        # Create standard logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Avoid duplicate handlers
        if self.logger.handlers:
            return

        # Configure structured logging if available
        if self.use_structlog:
            self._configure_structlog()
        else:
            self._configure_standard_logging(max_file_size, backup_count)

    def _configure_structlog(self):
        """Configure structlog for structured logging."""
        import structlog

        # Configure structlog processors
        shared_processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        if sys.stderr.isatty():
            # Pretty printing for console
            shared_processors.append(structlog.dev.ConsoleRenderer())
        else:
            # JSON for production
            shared_processors.append(structlog.processors.JSONRenderer())

        structlog.configure(
            processors=shared_processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Add file handler for structured logs
        log_file = self.log_dir / f"openeval_structured_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(file_handler)

    def _configure_standard_logging(self, max_file_size: int, backup_count: int):
        """Configure standard logging with multiple handlers."""
        # Rotating file handler for general logs
        log_file = self.log_dir / f"openeval_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            )
        )

        # Timed rotating handler for errors
        error_log_file = self.log_dir / "openeval_errors.log"
        error_handler = TimedRotatingFileHandler(
            error_log_file, when="midnight", interval=1, backupCount=30
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s\n"
                "Exception: %(exc_text)s"
            )
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)

    def audit(self, action: str, user: str = "system", resource: str = "", **kwargs):
        """Log audit event for compliance and security tracking."""
        if not self.enable_audit:
            return

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user,
            "resource": resource,
            "details": kwargs,
        }

        with self.audit_lock:
            self.audit_log.append(audit_entry)

        # Also log to audit file
        audit_file = self.log_dir / "audit.log"
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")

        # Log to main logger
        self.logger.info(f"AUDIT: {action} by {user} on {resource}", **kwargs)

    def get_audit_trail(
        self, user: Optional[str] = None, action: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get audit trail filtered by user and/or action."""
        with self.audit_lock:
            trail = self.audit_log.copy()

        if user:
            trail = [entry for entry in trail if entry["user"] == user]
        if action:
            trail = [entry for entry in trail if entry["action"] == action]

        return trail

    def export_audit_log(self, filepath: Path) -> None:
        """Export audit log to file."""
        with self.audit_lock:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.audit_log, f, indent=2)

    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)

    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.debug(message)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception and structured data."""
        if exception:
            message = f"{message} | Exception: {str(exception)}"
            self.logger.error(message, exc_info=True)
        else:
            if kwargs:
                message = f"{message} | {json.dumps(kwargs)}"
            self.logger.error(message)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with optional exception and structured data."""
        if exception:
            message = f"{message} | Exception: {str(exception)}"
            self.logger.critical(message, exc_info=True)
        else:
            if kwargs:
                message = f"{message} | {json.dumps(kwargs)}"
            self.logger.critical(message)


class ErrorHandler:
    """Centralized error handling with context tracking."""

    def __init__(self, logger: Optional[OpenEvalLogger] = None):
        """Initialize error handler with optional logger."""
        self.logger = logger or OpenEvalLogger()
        self.error_count = 0
        self.errors: Dict[str, int] = {}

    def handle_error(
        self, error: Exception, context: str = "", critical: bool = False, **metadata
    ) -> None:
        """Handle error with logging and tracking."""
        self.error_count += 1
        error_type = type(error).__name__
        self.errors[error_type] = self.errors.get(error_type, 0) + 1

        error_info = {
            "error_type": error_type,
            "context": context,
            "total_errors": self.error_count,
            "error_count_by_type": self.errors[error_type],
            **metadata,
        }

        if critical:
            self.logger.critical(
                f"Critical error in {context}: {str(error)}", exception=error, **error_info
            )
        else:
            self.logger.error(f"Error in {context}: {str(error)}", exception=error, **error_info)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        return {
            "total_errors": self.error_count,
            "errors_by_type": self.errors,
            "most_common_error": (
                max(self.errors.items(), key=lambda x: x[1])[0] if self.errors else None
            ),
        }


# Global instances
_global_logger = None
_global_error_handler = None


def get_logger(name: str = "openeval") -> OpenEvalLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = OpenEvalLogger(name)
    return _global_logger


def get_error_handler() -> ErrorHandler:
    """Get or create global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(get_logger())
    return _global_error_handler


def log_execution_time(func):
    """Decorator to log function execution time."""

    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = datetime.now()

        try:
            logger.debug(f"Starting execution of {func.__name__}")
            result = func(*args, **kwargs)

            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Completed {func.__name__}", execution_time=execution_time, status="success"
            )
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            get_error_handler().handle_error(
                e, context=f"function:{func.__name__}", execution_time=execution_time
            )
            raise

    return wrapper


"""Enhanced logging and debugging tools for OpenEval Lab."""

import traceback
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
import functools


@dataclass
class LogContext:
    """Context information for structured logging."""

    request_id: str = ""
    session_id: str = ""
    user_id: str = ""
    operation: str = ""
    component: str = ""

    # Evaluation context
    task_name: str = ""
    adapter_name: str = ""
    dataset_name: str = ""
    example_id: str = ""

    # Performance context
    start_time: Optional[float] = None
    memory_usage: Optional[float] = None

    # Custom context
    custom: Dict[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, include_context: bool = True, redact_sensitive: bool = True):
        super().__init__()
        self.include_context = include_context
        self.redact_sensitive = redact_sensitive
        self.sensitive_fields = {
            "api_key",
            "token",
            "password",
            "secret",
            "auth",
            "authorization",
            "x-api-key",
            "bearer",
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""

        # Base log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add context if available
        if self.include_context and hasattr(record, "context"):
            context = getattr(record, "context", None)
            if isinstance(context, LogContext):
                context_dict = {
                    "request_id": context.request_id,
                    "session_id": context.session_id,
                    "operation": context.operation,
                    "component": context.component,
                    "task_name": context.task_name,
                    "adapter_name": context.adapter_name,
                    "dataset_name": context.dataset_name,
                    "example_id": context.example_id,
                    "custom": context.custom,
                }

                # Add performance metrics
                if context.start_time:
                    context_dict["duration_ms"] = (time.time() - context.start_time) * 1000

                if context.memory_usage:
                    context_dict["memory_mb"] = context.memory_usage

                log_entry["context"] = context_dict
            else:
                log_entry["context"] = context

        # Add custom fields from record
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "context",
            }:
                log_entry[key] = value

        # Redact sensitive information
        if self.redact_sensitive:
            log_entry = self._redact_sensitive_data(log_entry)

        return json.dumps(log_entry, default=str, ensure_ascii=False)

    def _redact_sensitive_data(self, data: Any) -> Any:
        """Redact sensitive information from log data."""
        if isinstance(data, dict):
            redacted = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                    redacted[key] = "***REDACTED***"
                else:
                    redacted[key] = self._redact_sensitive_data(value)
            return redacted

        elif isinstance(data, list):
            return [self._redact_sensitive_data(item) for item in data]

        elif isinstance(data, str):
            # Redact common patterns
            if any(sensitive in data.lower() for sensitive in self.sensitive_fields):
                return "***REDACTED***"
            return data

        return data


class ContextualLogger:
    """Logger with automatic context injection."""

    def __init__(self, name: str, context: Optional[LogContext] = None):
        self.logger = logging.getLogger(name)
        self.context = context or LogContext()
        self._local = threading.local()

    def set_context(self, **kwargs):
        """Update logging context."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.custom[key] = value

    def get_context(self) -> LogContext:
        """Get current context."""
        return getattr(self._local, "context", self.context)

    def push_context(self, **kwargs) -> LogContext:
        """Push new context frame."""
        current = self.get_context()
        new_context = LogContext(
            request_id=current.request_id,
            session_id=current.session_id,
            user_id=current.user_id,
            operation=current.operation,
            component=current.component,
            task_name=current.task_name,
            adapter_name=current.adapter_name,
            dataset_name=current.dataset_name,
            example_id=current.example_id,
            start_time=current.start_time,
            memory_usage=current.memory_usage,
            custom=current.custom.copy(),
        )

        # Update with new values
        for key, value in kwargs.items():
            if hasattr(new_context, key):
                setattr(new_context, key, value)
            else:
                new_context.custom[key] = value

        self._local.context = new_context
        return new_context

    def pop_context(self):
        """Pop context frame."""
        self._local.context = self.context

    def _log(self, level: int, message: str, *args, **kwargs):
        """Internal log method with context injection."""
        extra = kwargs.get("extra", {})
        extra["context"] = self.get_context()
        kwargs["extra"] = extra

        self.logger.log(level, message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        self._log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        self._log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        self._log(logging.ERROR, message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        self._log(logging.CRITICAL, message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs):
        kwargs["exc_info"] = True
        self._log(logging.ERROR, message, *args, **kwargs)


@contextmanager
def log_context(**kwargs):
    """Context manager for logging context."""
    logger = get_contextual_logger()
    logger.get_context()

    try:
        logger.push_context(**kwargs)
        yield logger
    finally:
        logger.pop_context()


class DebugTracer:
    """Advanced debugging and tracing functionality."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.traces: List[Dict[str, Any]] = []
        self.breakpoints: Dict[str, Callable] = {}
        self._trace_lock = threading.Lock()

    def trace(self, name: str, data: Optional[Dict[str, Any]] = None):
        """Add a trace point."""
        if not self.enabled:
            return

        trace_entry = {
            "timestamp": time.time(),
            "name": name,
            "thread_id": threading.get_ident(),
            "data": data or {},
        }

        with self._trace_lock:
            self.traces.append(trace_entry)

    def set_breakpoint(self, name: str, condition: Optional[Callable] = None):
        """Set a conditional breakpoint."""
        self.breakpoints[name] = condition or (lambda: True)

    def check_breakpoint(self, name: str, data: Optional[Dict[str, Any]] = None):
        """Check if breakpoint should trigger."""
        if not self.enabled or name not in self.breakpoints:
            return False

        condition = self.breakpoints[name]
        if condition():
            self.trace(f"breakpoint_{name}", data)
            return True

        return False

    def get_traces(self, name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get traces, optionally filtered by name."""
        with self._trace_lock:
            if name_filter:
                return [t for t in self.traces if name_filter in t["name"]]
            return self.traces.copy()

    def clear_traces(self):
        """Clear all traces."""
        with self._trace_lock:
            self.traces.clear()

    def save_traces(self, filepath: Path):
        """Save traces to file."""
        traces = self.get_traces()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(traces, f, indent=2, default=str)


class PerformanceProfiler:
    """Function-level performance profiling."""

    def __init__(self):
        self.profiles: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def profile(self, name: Optional[str] = None, include_args: bool = False):
        """Decorator for profiling functions."""

        def decorator(func: Callable) -> Callable:
            profile_name = name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()

                success = False
                error = None
                result = None

                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    end_time = time.perf_counter()
                    end_memory = self._get_memory_usage()

                    profile_data = {
                        "timestamp": start_time,
                        "duration": end_time - start_time,
                        "memory_delta": (
                            end_memory - start_memory if start_memory and end_memory else None
                        ),
                        "success": success,
                        "error": error,
                        "thread_id": threading.get_ident(),
                    }

                    if include_args:
                        profile_data["args"] = str(args)[:100]  # Truncate for safety
                        profile_data["kwargs"] = {k: str(v)[:100] for k, v in kwargs.items()}

                    with self._lock:
                        if profile_name not in self.profiles:
                            self.profiles[profile_name] = []
                        self.profiles[profile_name].append(profile_data)

                return result

            return wrapper

        return decorator

    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None

    def get_profile_summary(self, name: str) -> Dict[str, Any]:
        """Get profile summary for a function."""
        if name not in self.profiles:
            return {}

        profiles = self.profiles[name]
        durations = [p["duration"] for p in profiles]
        success_rate = sum(1 for p in profiles if p["success"]) / len(profiles)

        return {
            "call_count": len(profiles),
            "success_rate": success_rate,
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations),
        }

    def save_profiles(self, filepath: Path):
        """Save all profiles to file."""
        profiles_with_summary = {}

        for name, profile_list in self.profiles.items():
            profiles_with_summary[name] = {
                "summary": self.get_profile_summary(name),
                "profiles": profile_list,
            }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(profiles_with_summary, f, indent=2, default=str)


class LoggingManager:
    """Centralized logging management."""

    def __init__(self):
        self.loggers: Dict[str, ContextualLogger] = {}
        self.tracers: Dict[str, DebugTracer] = {}
        self.profilers: Dict[str, PerformanceProfiler] = {}
        self._configured = False

    def configure_logging(
        self,
        level: str = "INFO",
        format_type: str = "structured",
        log_file: Optional[Path] = None,
        console_output: bool = True,
        redact_sensitive: bool = True,
    ):
        """Configure global logging settings."""

        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Configure formatters
        if format_type == "structured":
            formatter = StructuredFormatter(redact_sensitive=redact_sensitive)
        else:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        self._configured = True

    def get_logger(self, name: str, context: Optional[LogContext] = None) -> ContextualLogger:
        """Get or create a contextual logger."""
        if name not in self.loggers:
            self.loggers[name] = ContextualLogger(name, context)

        return self.loggers[name]

    def get_tracer(self, name: str, enabled: bool = False) -> DebugTracer:
        """Get or create a debug tracer."""
        if name not in self.tracers:
            self.tracers[name] = DebugTracer(enabled)

        return self.tracers[name]

    def get_profiler(self, name: str) -> PerformanceProfiler:
        """Get or create a performance profiler."""
        if name not in self.profilers:
            self.profilers[name] = PerformanceProfiler()

        return self.profilers[name]

    def enable_debug_mode(self):
        """Enable debug mode with tracing."""
        self.configure_logging(level="DEBUG")

        for tracer in self.tracers.values():
            tracer.enabled = True

    def save_all_debug_data(self, output_dir: Path):
        """Save all debug data to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save traces
        for name, tracer in self.tracers.items():
            if tracer.traces:
                tracer.save_traces(output_dir / f"traces_{name}.json")

        # Save profiles
        for name, profiler in self.profilers.items():
            if profiler.profiles:
                profiler.save_profiles(output_dir / f"profiles_{name}.json")


# Global logging manager instance
_logging_manager = LoggingManager()


def configure_logging(**kwargs):
    """Configure global logging settings."""
    _logging_manager.configure_logging(**kwargs)


def get_logger(name: str, **context_kwargs) -> ContextualLogger:
    """Get a contextual logger."""
    context = LogContext(**context_kwargs) if context_kwargs else None
    return _logging_manager.get_logger(name, context)


def get_contextual_logger() -> ContextualLogger:
    """Get the default contextual logger."""
    return _logging_manager.get_logger("openeval")


def get_tracer(name: str = "default", enabled: bool = False) -> DebugTracer:
    """Get a debug tracer."""
    return _logging_manager.get_tracer(name, enabled)


def get_profiler(name: str = "default") -> PerformanceProfiler:
    """Get a performance profiler."""
    return _logging_manager.get_profiler(name)


def enable_debug_mode():
    """Enable comprehensive debugging."""
    _logging_manager.enable_debug_mode()


def save_debug_data(output_dir: Union[str, Path]):
    """Save all debug data."""
    _logging_manager.save_all_debug_data(Path(output_dir))


# Convenient decorators
def traced(name: Optional[str] = None, tracer_name: str = "default"):
    """Decorator to add tracing to a function."""

    def decorator(func: Callable) -> Callable:
        trace_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer(tracer_name)
            tracer.trace(
                f"{trace_name}_start", {"args_count": len(args), "kwargs_count": len(kwargs)}
            )

            try:
                result = func(*args, **kwargs)
                tracer.trace(f"{trace_name}_success")
                return result
            except Exception as e:
                tracer.trace(f"{trace_name}_error", {"error": str(e)})
                raise

        return wrapper

    return decorator


def profiled(
    name: Optional[str] = None, profiler_name: str = "default", include_args: bool = False
):
    """Decorator to add profiling to a function."""
    profiler = get_profiler(profiler_name)
    return profiler.profile(name, include_args)
