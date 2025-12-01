"""
Unified Error Handling System for OpenEval Lab.

Comprehensive error handling including:
- Typed exceptions with categories and error codes
- Retry mechanisms with exponential backoff
- Circuit breaker pattern for external services
- Graceful degradation strategies
- Error tracking and analytics
- Recovery strategies

This module consolidates:
- error_handling.py (original - 416 lines)
- error_context.py (deprecated wrapper)
- error_handling_unified.py (381 lines)
- error_handling_enhanced.py (559 lines)
- error_recovery.py (210 lines)
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

__all__ = [
    # Enums
    "ErrorCategory",
    "ErrorSeverity",
    # Error context
    "ErrorContext",
    # Base exception
    "OpenEvalError",
    # Specific exceptions
    "ConfigurationError",
    "DataError",
    "ModelError",
    "EvaluationError",
    "CacheError",
    "ValidationError",
    "ResourceError",
    "NetworkError",
    "TimeoutError",
    "RateLimitError",
    # Error tracking
    "ErrorTracker",
    # Retry mechanisms
    "RetryConfig",
    "retry_with_config",
    "retry",
    # Circuit breaker
    "CircuitBreaker",
    # Graceful degradation
    "GracefulDegradation",
    # Recovery
    "ErrorRecovery",  # Alias for ErrorRecoveryManager
    "ErrorRecoveryManager",
    "ErrorRecoveryStrategy",
    "attempt_error_recovery",
    # Context managers
    "error_context",
    "safe_operation",
    # Factory
    "ErrorContextFactory",
    # Helpers
    "create_robust_evaluation_context",
    "with_error_context",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")

# =============================================================================
# Enums
# =============================================================================


class ErrorCategory(Enum):
    """Categories of errors for classification and filtering."""

    CONFIGURATION = "configuration"
    DATA = "data"
    MODEL = "model"
    EVALUATION = "evaluation"
    CACHE = "cache"
    VALIDATION = "validation"
    RESOURCE = "resource"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    FILE_IO = "file_io"
    MEMORY = "memory"
    DEPENDENCY = "dependency"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels for prioritization."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# Error Context
# =============================================================================


class ErrorContext:
    """Rich contextual information about an error.

    Provides comprehensive error details including suggestions,
    documentation links, and debug information.
    """

    __slots__ = (
        "category",
        "severity",
        "message",
        "error_code",
        "details",
        "suggestions",
        "documentation_url",
        "component",
        "operation",
        "user_data",
        "stack_trace",
        "timestamp",
        "retry_count",
        "recoverable",
    )

    def __init__(
        self,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        message: str = "",
        error_code: str = "E0000",
        details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        documentation_url: str = "",
        component: Optional[str] = None,
        operation: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None,
        timestamp: Optional[float] = None,
        retry_count: int = 0,
        recoverable: bool = True,
    ):
        self.category = category
        self.severity = severity
        self.message = message
        self.error_code = error_code
        self.details = details
        self.suggestions = suggestions or []
        self.documentation_url = documentation_url
        self.component = component
        self.operation = operation
        self.user_data = user_data
        self.stack_trace = stack_trace
        self.timestamp = timestamp or time.time()
        self.retry_count = retry_count
        self.recoverable = recoverable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "suggestions": self.suggestions,
            "documentation_url": self.documentation_url,
            "component": self.component,
            "operation": self.operation,
            "user_data": self.user_data,
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
            "recoverable": self.recoverable,
        }


# =============================================================================
# Exception Classes
# =============================================================================


class OpenEvalError(Exception):
    """Base exception for OpenEval Lab with rich context.

    Provides enhanced error information including category, severity,
    suggestions, and documentation links.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "E0000",
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        documentation_url: str = "",
        component: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.category = category
        self.cause = cause

        self.context = ErrorContext(
            category=category,
            severity=severity,
            message=message,
            error_code=error_code,
            details=details,
            suggestions=suggestions,
            documentation_url=documentation_url,
            component=component,
            operation=operation,
            user_data=context,
            stack_trace=traceback.format_exc() if cause else None,
        )

        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def format_user_message(self) -> str:
        """Format a user-friendly error message."""
        lines = [f"❌ {self.context.message}"]

        if self.context.details:
            lines.append(f"📝 Details: {self.context.details}")

        if self.context.suggestions:
            lines.append("💡 Suggestions:")
            for suggestion in self.context.suggestions:
                lines.append(f"   • {suggestion}")

        if self.context.documentation_url:
            lines.append(f"📚 Documentation: {self.context.documentation_url}")

        if self.context.error_code:
            lines.append(f"🔍 Error Code: {self.context.error_code}")

        return "\n".join(lines)


class ConfigurationError(OpenEvalError):
    """Configuration-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "E1000")
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class DataError(OpenEvalError):
    """Data-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "E2000")
        kwargs.setdefault("category", ErrorCategory.DATA)
        super().__init__(message, **kwargs)


class ModelError(OpenEvalError):
    """Model-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "E3000")
        kwargs.setdefault("category", ErrorCategory.MODEL)
        super().__init__(message, **kwargs)


class EvaluationError(OpenEvalError):
    """Evaluation-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "E4000")
        kwargs.setdefault("category", ErrorCategory.EVALUATION)
        super().__init__(message, **kwargs)


class CacheError(OpenEvalError):
    """Cache-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "E5000")
        kwargs.setdefault("category", ErrorCategory.CACHE)
        super().__init__(message, **kwargs)


class ValidationError(OpenEvalError):
    """Validation-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "E6000")
        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        super().__init__(message, **kwargs)


class ResourceError(OpenEvalError):
    """Resource-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "E7000")
        kwargs.setdefault("category", ErrorCategory.RESOURCE)
        super().__init__(message, **kwargs)


class NetworkError(OpenEvalError):
    """Network-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "E8000")
        kwargs.setdefault("category", ErrorCategory.NETWORK)
        super().__init__(message, **kwargs)


class TimeoutError(OpenEvalError):
    """Timeout-related errors."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "E8100")
        kwargs.setdefault("category", ErrorCategory.TIMEOUT)
        super().__init__(message, **kwargs)


class RateLimitError(OpenEvalError):
    """Rate limit errors."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("error_code", "E8200")
        kwargs.setdefault("category", ErrorCategory.RATE_LIMIT)
        super().__init__(message, **kwargs)


# =============================================================================
# Error Tracking
# =============================================================================


class ErrorTracker:
    """Tracks and categorizes errors during evaluation.

    Thread-safe error tracking with analytics.
    """

    __slots__ = ("errors", "error_counts", "max_tracked", "_lock")

    def __init__(self, max_tracked: int = 1000):
        self.errors: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
        self.max_tracked = max_tracked
        self._lock = threading.RLock()

    def log_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        component: str = "",
        operation: str = "",
    ) -> ErrorContext:
        """Log an error with context."""
        error_type = type(error).__name__

        with self._lock:
            error_ctx = ErrorContext(
                category=ErrorCategory.UNKNOWN,
                severity=severity,
                message=str(error),
                stack_trace=traceback.format_exc(),
                user_data=context,
                recoverable=recoverable,
                component=component,
                operation=operation,
            )

            self.errors.append(error_ctx)
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

            # Keep size limited
            if len(self.errors) > self.max_tracked:
                self.errors.pop(0)

        # Log to standard logger
        log_level = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }[severity]

        logger.log(log_level, f"{error_type}: {error_ctx.message}")

        return error_ctx

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        with self._lock:
            return {
                "total_errors": len(self.errors),
                "error_types": dict(self.error_counts),
                "critical_errors": len(
                    [e for e in self.errors if e.severity == ErrorSeverity.CRITICAL]
                ),
                "recoverable_errors": len([e for e in self.errors if e.recoverable]),
                "recent_errors": [
                    {
                        "message": e.message,
                        "severity": e.severity.value,
                        "timestamp": e.timestamp,
                    }
                    for e in self.errors[-5:]
                ],
            }

    def get_error_histogram(self) -> Dict[str, int]:
        """Get histogram of error types."""
        with self._lock:
            return dict(self.error_counts)

    def clear(self) -> None:
        """Clear all tracked errors."""
        with self._lock:
            self.errors.clear()
            self.error_counts.clear()


# =============================================================================
# Retry Mechanisms
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry logic with exponential backoff."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retryable_errors: List[Type[Exception]] = field(
        default_factory=lambda: [ConnectionError, builtins_TimeoutError, OSError]
    )

    def is_retryable(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        return any(isinstance(error, e) for e in self.retryable_errors)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_delay

        delay = min(delay, self.max_delay)

        if self.jitter:
            import random

            delay *= 0.5 + random.random() * 0.5

        return delay


# Alias for builtin TimeoutError to avoid conflict
builtins_TimeoutError = (
    __builtins__["TimeoutError"]
    if isinstance(__builtins__, dict)
    else getattr(__builtins__, "TimeoutError", Exception)
)


def retry_with_config(
    config: RetryConfig, error_tracker: Optional[ErrorTracker] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with configuration."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Optional[Exception] = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as error:
                    last_error = error

                    if error_tracker:
                        severity = (
                            ErrorSeverity.HIGH
                            if attempt == config.max_attempts
                            else ErrorSeverity.MEDIUM
                        )
                        error_ctx = error_tracker.log_error(
                            error,
                            severity=severity,
                            context={
                                "function": func.__name__,
                                "attempt": attempt,
                                "max_attempts": config.max_attempts,
                            },
                        )
                        error_ctx.retry_count = attempt

                    if not config.is_retryable(error):
                        raise

                    if attempt == config.max_attempts:
                        raise

                    delay = config.get_delay(attempt)
                    time.sleep(delay)

            if last_error is not None:
                raise last_error
            raise RuntimeError("Unexpected error in retry wrapper")

        return wrapper

    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential: bool = True,
    exceptions: Optional[List[Type[Exception]]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Simple retry decorator.

    Args:
        max_attempts: Maximum number of attempts
        delay: Base delay between attempts
        exponential: Whether to use exponential backoff
        exceptions: List of exceptions to retry on (default: all)
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=delay,
        exponential_backoff=exponential,
        retryable_errors=exceptions or [Exception],
    )
    return retry_with_config(config)


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures.

    States:
    - closed: Normal operation, requests pass through
    - open: Failure threshold reached, requests blocked
    - half_open: Testing if service recovered
    """

    __slots__ = (
        "failure_threshold",
        "timeout",
        "expected_exception",
        "failure_count",
        "last_failure_time",
        "state",
        "_lock",
    )

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"
        self._lock = threading.RLock()

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with self._lock:
                if self.state == "open":
                    if (
                        self.last_failure_time is not None
                        and time.time() - self.last_failure_time >= self.timeout
                    ):
                        self.state = "half_open"
                    else:
                        raise RuntimeError(f"Circuit breaker OPEN. Blocked for {self.timeout}s")

            try:
                result = func(*args, **kwargs)

                with self._lock:
                    if self.state == "half_open":
                        self.state = "closed"
                        self.failure_count = 0

                return result

            except self.expected_exception as error:
                with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = time.time()

                    if self.failure_count >= self.failure_threshold:
                        self.state = "open"

                raise error

        return wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self.state = "closed"
            self.failure_count = 0
            self.last_failure_time = None


# =============================================================================
# Graceful Degradation
# =============================================================================


class GracefulDegradation:
    """Provides graceful degradation strategies."""

    @staticmethod
    def fallback_to_default(
        default_value: T,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Return default value on error."""

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    return default_value

            return wrapper

        return decorator

    @staticmethod
    def fallback_to_function(
        fallback_func: Callable[..., T],
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Call fallback function on error."""

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    return fallback_func(error, *args, **kwargs)

            return wrapper

        return decorator


# =============================================================================
# Error Recovery
# =============================================================================


@dataclass
class ErrorRecoveryStrategy:
    """Defines a strategy for recovering from specific errors."""

    exception_type: Type[Exception]
    recovery_function: Callable[[], Any]
    max_retries: int = 3
    backoff_multiplier: float = 1.5
    description: str = ""


class ErrorRecoveryManager:
    """Manages error recovery strategies for different error types."""

    __slots__ = ("strategies", "_lock", "_retry_counts")

    def __init__(self):
        self.strategies: Dict[Type[Exception], ErrorRecoveryStrategy] = {}
        self._lock = threading.RLock()
        self._retry_counts: Dict[Type[Exception], int] = {}

    def register_strategy(self, strategy: ErrorRecoveryStrategy) -> None:
        """Register a recovery strategy."""
        with self._lock:
            self.strategies[strategy.exception_type] = strategy

    def attempt_recovery(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Attempt to recover from an error."""
        error_type = type(error)

        with self._lock:
            strategy = self.strategies.get(error_type)
            if not strategy:
                return None

            retry_count = self._retry_counts.get(error_type, 0)
            if retry_count >= strategy.max_retries:
                return None

            try:
                logger.info(
                    f"Attempting recovery for {error_type.__name__} "
                    f"(attempt {retry_count + 1}/{strategy.max_retries})"
                )
                result = strategy.recovery_function()
                self._retry_counts[error_type] = 0
                return result
            except Exception as recovery_error:
                self._retry_counts[error_type] = retry_count + 1
                logger.warning(f"Recovery attempt failed: {recovery_error}")
                return None

    def reset_retry_counts(self) -> None:
        """Reset all retry counts."""
        with self._lock:
            self._retry_counts.clear()


# Global recovery manager
_global_recovery_manager = ErrorRecoveryManager()


def attempt_error_recovery(
    error: Exception, context: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """Attempt to recover from an error using registered strategies."""
    return _global_recovery_manager.attempt_recovery(error, context)


# =============================================================================
# Context Managers and Decorators
# =============================================================================


@contextmanager
def error_context(component: str, operation: str, error_tracker: Optional[ErrorTracker] = None):
    """Context manager for error handling with component and operation context.

    Example:
        with error_context("cache", "get", tracker):
            value = cache.get(key)
    """
    try:
        yield
    except Exception as e:
        if error_tracker:
            error_tracker.log_error(
                e,
                component=component,
                operation=operation,
                context={"component": component, "operation": operation},
            )
        raise


def with_error_context(
    component: str, operation: str
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that adds error context to a function.

    Example:
        @with_error_context("cache", "get")
        def get_from_cache(key):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {component}.{operation}: {e}")
                raise

        return wrapper

    return decorator


def safe_operation(
    operation_name: str = "operation",
    default: Any = None,
    reraise: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for safe operations with optional default return.

    Args:
        operation_name: Name of the operation for logging
        default: Default value to return on error
        reraise: Whether to re-raise the exception
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {operation_name}: {e}")
                if reraise:
                    raise
                return default

        return wrapper

    return decorator


# =============================================================================
# Error Context Factory
# =============================================================================


# Error templates with suggested fixes
_ERROR_TEMPLATES: Dict[str, ErrorContext] = {
    "E1000": ErrorContext(
        category=ErrorCategory.CONFIGURATION,
        message="Configuration file not found",
        error_code="E1000",
        suggestions=["Create config file", "Specify path with --config flag"],
        documentation_url="https://docs.openeval.lab/config",
    ),
    "E2000": ErrorContext(
        category=ErrorCategory.DATA,
        message="Dataset loading failed",
        error_code="E2000",
        suggestions=["Check dataset path", "Verify format is supported"],
        documentation_url="https://docs.openeval.lab/datasets",
    ),
    "E3000": ErrorContext(
        category=ErrorCategory.MODEL,
        message="Model initialization failed",
        error_code="E3000",
        suggestions=["Verify model name", "Check dependencies are installed"],
        documentation_url="https://docs.openeval.lab/models",
    ),
    "E4000": ErrorContext(
        category=ErrorCategory.EVALUATION,
        message="Evaluation failed",
        error_code="E4000",
        suggestions=["Check experiment specification", "Review logs for details"],
        documentation_url="https://docs.openeval.lab/evaluation",
    ),
    "E5000": ErrorContext(
        category=ErrorCategory.CACHE,
        message="Cache operation failed",
        error_code="E5000",
        suggestions=["Clear cache and retry", "Check disk space"],
        documentation_url="https://docs.openeval.lab/caching",
    ),
    "E6000": ErrorContext(
        category=ErrorCategory.VALIDATION,
        message="Validation failed",
        error_code="E6000",
        suggestions=["Verify input matches expected schema"],
        documentation_url="https://docs.openeval.lab/validation",
    ),
    "E7000": ErrorContext(
        category=ErrorCategory.RESOURCE,
        message="Resource limit exceeded",
        error_code="E7000",
        suggestions=["Increase limits", "Reduce batch size"],
        documentation_url="https://docs.openeval.lab/resources",
    ),
    "E8000": ErrorContext(
        category=ErrorCategory.NETWORK,
        message="Network error",
        error_code="E8000",
        suggestions=["Check network connection", "Verify firewall settings"],
        documentation_url="https://docs.openeval.lab/networking",
    ),
}


class ErrorContextFactory:
    """Factory for creating contextualized errors."""

    _templates = _ERROR_TEMPLATES

    @classmethod
    def create(
        cls,
        error_code: str,
        message: Optional[str] = None,
        component: Optional[str] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> ErrorContext:
        """Create an error context from template."""
        template = cls._templates.get(error_code)
        if template is None:
            return ErrorContext(
                category=ErrorCategory.UNKNOWN,
                message=message or "Unknown error",
                error_code=error_code,
                suggestions=["Check logs and documentation"],
                documentation_url="https://docs.openeval.lab/errors",
                user_data=debug_info,
            )

        return ErrorContext(
            category=template.category,
            message=message or template.message,
            error_code=error_code,
            suggestions=template.suggestions.copy(),
            documentation_url=template.documentation_url,
            component=component,
            user_data=debug_info,
        )

    @classmethod
    def add_template(cls, error_code: str, context: ErrorContext) -> None:
        """Add a new error template."""
        cls._templates[error_code] = context


# =============================================================================
# Helper Functions
# =============================================================================


def create_robust_evaluation_context() -> Dict[str, Any]:
    """Create a context with comprehensive error handling.

    Returns dictionary with pre-configured error handling components:
    - error_tracker: ErrorTracker instance
    - network_retry: RetryConfig for network operations
    - circuit_breaker: CircuitBreaker for external services
    - recovery_manager: ErrorRecoveryManager
    """
    error_tracker = ErrorTracker()

    network_retry = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        retryable_errors=[ConnectionError, builtins_TimeoutError, OSError],
    )

    service_breaker = CircuitBreaker(failure_threshold=3, timeout=30.0)

    return {
        "error_tracker": error_tracker,
        "network_retry": network_retry,
        "circuit_breaker": service_breaker,
        "recovery_manager": _global_recovery_manager,
    }


# Backward compatibility alias
ErrorRecovery = ErrorRecoveryManager
