"""Enhanced error handling and retry mechanisms."""

import time
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import traceback


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Contextual information about an error."""
    error_type: str
    severity: ErrorSeverity
    message: str
    traceback_str: str
    timestamp: float
    context: Dict[str, Any]
    retry_count: int = 0
    recoverable: bool = True


class ErrorTracker:
    """Tracks and categorizes errors during evaluation."""
    
    def __init__(self):
        self.errors: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
    
    def log_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ) -> ErrorContext:
        """Log an error with context."""
        
        error_type = type(error).__name__
        error_ctx = ErrorContext(
            error_type=error_type,
            severity=severity,
            message=str(error),
            traceback_str=traceback.format_exc(),
            timestamp=time.time(),
            context=context or {},
            recoverable=recoverable
        )
        
        self.errors.append(error_ctx)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log to standard logger
        logger = logging.getLogger(__name__)
        log_level = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        logger.log(log_level, f"{error_type}: {error_ctx.message}")
        
        return error_ctx
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        return {
            "total_errors": len(self.errors),
            "error_types": dict(self.error_counts),
            "critical_errors": len([e for e in self.errors if e.severity == ErrorSeverity.CRITICAL]),
            "recoverable_errors": len([e for e in self.errors if e.recoverable]),
            "recent_errors": [
                {
                    "type": e.error_type,
                    "message": e.message,
                    "severity": e.severity.value,
                    "timestamp": e.timestamp
                }
                for e in self.errors[-5:]  # Last 5 errors
            ]
        }


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
        retryable_errors: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
        self.retryable_errors = retryable_errors or [
            ConnectionError,
            TimeoutError,
            OSError,
            # Add more as needed
        ]
    
    def is_retryable(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        return any(isinstance(error, error_type) for error_type in self.retryable_errors)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_delay
        
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


def retry_with_config(
    config: RetryConfig,
    error_tracker: Optional[ErrorTracker] = None
):
    """Decorator for retrying functions with configuration."""
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            last_error: Optional[Exception] = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as error:
                    last_error = error
                    
                    # Log error
                    if error_tracker:
                        severity = ErrorSeverity.HIGH if attempt == config.max_attempts else ErrorSeverity.MEDIUM
                        error_ctx = error_tracker.log_error(
                            error,
                            severity=severity,
                            context={
                                "function": func.__name__,
                                "attempt": attempt,
                                "max_attempts": config.max_attempts
                            }
                        )
                        error_ctx.retry_count = attempt
                    
                    # Check if retryable
                    if not config.is_retryable(error):
                        if error_tracker:
                            error_tracker.log_error(
                                error,
                                severity=ErrorSeverity.HIGH,
                                context={"function": func.__name__, "reason": "non_retryable"},
                                recoverable=False
                            )
                        raise error
                    
                    # Final attempt
                    if attempt == config.max_attempts:
                        if error_tracker:
                            error_tracker.log_error(
                                error,
                                severity=ErrorSeverity.CRITICAL,
                                context={"function": func.__name__, "reason": "max_attempts_reached"},
                                recoverable=False
                            )
                        raise error
                    
                    # Wait before retry
                    delay = config.get_delay(attempt)
                    time.sleep(delay)
            
            # Shouldn't reach here, but just in case
            if last_error is not None:
                raise last_error
            else:
                raise RuntimeError("Unexpected error in retry wrapper")
        
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
    
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            if self.state == "open":
                if self.last_failure_time is not None and time.time() - self.last_failure_time >= self.timeout:
                    self.state = "half_open"
                else:
                    raise Exception(f"Circuit breaker is OPEN. Calls blocked for {self.timeout}s")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                
                return result
            
            except self.expected_exception as error:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                
                raise error
        
        return wrapper


class GracefulDegradation:
    """Provides graceful degradation strategies."""
    
    @staticmethod
    def fallback_to_default(default_value: Any):
        """Return default value on error."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    return default_value
            return wrapper
        return decorator
    
    @staticmethod
    def fallback_to_function(fallback_func: Callable):
        """Call fallback function on error."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    return fallback_func(error, *args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def partial_success(allow_partial: bool = True):
        """Allow partial success in batch operations."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs) -> Any:
                if allow_partial:
                    # Try to process in smaller chunks or individual items
                    try:
                        return func(*args, **kwargs)
                    except Exception:
                        # Implement chunked processing logic here
                        # This is a placeholder - actual implementation depends on function
                        return {"partial": True, "error": "Failed to process fully"}
                else:
                    return func(*args, **kwargs)
            return wrapper
        return decorator


class ErrorRecovery:
    """Provides error recovery mechanisms."""
    
    def __init__(self, error_tracker: ErrorTracker):
        self.error_tracker = error_tracker
    
    def auto_recover(self, func: Callable, max_recovery_attempts: int = 2):
        """Attempt automatic recovery from common errors."""
        def wrapper(*args, **kwargs) -> Any:
            recovery_attempts = 0
            
            while recovery_attempts <= max_recovery_attempts:
                try:
                    return func(*args, **kwargs)
                
                except ImportError as error:
                    if recovery_attempts < max_recovery_attempts:
                        self.error_tracker.log_error(
                            error,
                            context={"recovery_attempt": recovery_attempts + 1}
                        )
                        # Try to install missing package
                        self._attempt_package_install(str(error))
                        recovery_attempts += 1
                    else:
                        raise error
                
                except FileNotFoundError as error:
                    if recovery_attempts < max_recovery_attempts:
                        self.error_tracker.log_error(
                            error,
                            context={"recovery_attempt": recovery_attempts + 1}
                        )
                        # Try to create missing directories
                        self._attempt_create_directories(str(error))
                        recovery_attempts += 1
                    else:
                        raise error
                
                except Exception as error:
                    # For other errors, don't attempt recovery
                    self.error_tracker.log_error(error, recoverable=False)
                    raise error
        
        return wrapper
    
    def _attempt_package_install(self, error_msg: str):
        """Attempt to install missing package."""
        # This is a placeholder - actual implementation would need
        # to parse error message and try pip install
        pass
    
    def _attempt_create_directories(self, error_msg: str):
        """Attempt to create missing directories."""
        import re
        from pathlib import Path
        
        # Try to extract path from error message
        path_match = re.search(r"'([^']+)'", error_msg)
        if path_match:
            path = Path(path_match.group(1))
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass  # Recovery failed


def create_robust_evaluation_context():
    """Create a context with comprehensive error handling."""
    
    error_tracker = ErrorTracker()
    
    # Configure retry for network-related operations
    network_retry = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        retryable_errors=[ConnectionError, TimeoutError, OSError]
    )
    
    # Configure retry for temporary failures
    temp_failure_retry = RetryConfig(
        max_attempts=5,
        base_delay=0.5,
        exponential_backoff=True,
        retryable_errors=[RuntimeError, ValueError]  # Customize as needed
    )
    
    # Circuit breaker for external services
    service_breaker = CircuitBreaker(
        failure_threshold=3,
        timeout=30.0
    )
    
    # Error recovery
    recovery = ErrorRecovery(error_tracker)
    
    return {
        "error_tracker": error_tracker,
        "network_retry": network_retry,
        "temp_failure_retry": temp_failure_retry,
        "circuit_breaker": service_breaker,
        "recovery": recovery
    }
