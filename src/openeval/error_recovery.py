"""
Enhanced Error Handling Enhancement Module

Adds advanced error handling utilities to the existing error_handling_unified.py
including recovery strategies, error tracking, and better context management.
"""

from __future__ import annotations

import threading
from typing import Optional, Dict, List, Any, Callable, Type
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


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

    def __init__(self):
        """Initialize recovery manager."""
        self.strategies: Dict[Type[Exception], ErrorRecoveryStrategy] = {}
        self.lock = threading.RLock()
        self._retry_counts: Dict[Type[Exception], int] = {}

    def register_strategy(self, strategy: ErrorRecoveryStrategy) -> None:
        """Register a recovery strategy."""
        with self.lock:
            self.strategies[strategy.exception_type] = strategy

    def attempt_recovery(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Attempt to recover from an error.

        Args:
            error: The exception
            context: Additional context

        Returns:
            Recovery result or None if recovery failed
        """
        error_type = type(error)

        with self.lock:
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
                self._retry_counts[error_type] = 0  # Reset on success
                return result
            except Exception as recovery_error:
                self._retry_counts[error_type] = retry_count + 1
                logger.warning(f"Recovery attempt {retry_count + 1} failed: {recovery_error}")
                return None

    def reset_retry_counts(self) -> None:
        """Reset all retry counts."""
        with self.lock:
            self._retry_counts.clear()


# Global recovery manager
_global_recovery_manager = ErrorRecoveryManager()


def register_recovery_strategy(strategy: ErrorRecoveryStrategy) -> None:
    """Register a global recovery strategy."""
    _global_recovery_manager.register_strategy(strategy)


def attempt_error_recovery(
    error: Exception, context: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """Attempt to recover from an error using registered strategies."""
    return _global_recovery_manager.attempt_recovery(error, context)


class ErrorTracker:
    """Track and analyze errors for debugging and monitoring."""

    def __init__(self, max_tracked: int = 1000):
        """Initialize error tracker."""
        self.max_tracked = max_tracked
        self.errors: List[Dict[str, Any]] = []
        self.lock = threading.RLock()

    def track_error(
        self,
        error: Exception,
        operation: str = "",
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track an error.

        Args:
            error: The exception
            operation: Operation being performed
            extra_info: Additional information
        """
        with self.lock:
            error_entry = {
                "error_type": type(error).__name__,
                "message": str(error),
                "operation": operation,
                "extra": extra_info or {},
                "timestamp": __import__("time").time(),
            }
            self.errors.append(error_entry)

            # Keep size limited
            if len(self.errors) > self.max_tracked:
                self.errors.pop(0)

    def get_error_histogram(self) -> Dict[str, int]:
        """Get histogram of error types."""
        with self.lock:
            histogram = {}
            for entry in self.errors:
                error_type = entry["error_type"]
                histogram[error_type] = histogram.get(error_type, 0) + 1
            return histogram

    def clear(self) -> None:
        """Clear tracked errors."""
        with self.lock:
            self.errors.clear()


# Global error tracker
_global_error_tracker = ErrorTracker()


def track_error(
    error: Exception,
    operation: str = "",
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Track an error globally."""
    _global_error_tracker.track_error(error, operation, extra_info)


def get_error_histogram() -> Dict[str, int]:
    """Get histogram of tracked errors."""
    return _global_error_tracker.get_error_histogram()


class SafeOperation:
    """Context manager for safe error-handled operations."""

    def __init__(
        self,
        operation_name: str = "Operation",
        on_error: Optional[Callable[[Exception], None]] = None,
        reraise: bool = True,
    ):
        """Initialize safe operation context.

        Args:
            operation_name: Name of the operation
            on_error: Callback for error handling
            reraise: Whether to re-raise caught exceptions
        """
        self.operation_name = operation_name
        self.on_error = on_error
        self.reraise = reraise
        self.error: Optional[Exception] = None

    def __enter__(self) -> "SafeOperation":
        """Enter context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit context, handling any exceptions."""
        if exc_val:
            self.error = exc_val
            track_error(exc_val, operation=self.operation_name)

            if self.on_error:
                try:
                    self.on_error(exc_val)
                except Exception as callback_error:
                    logger.error(f"Error in error handler: {callback_error}")

            return not self.reraise  # Suppress exception if reraise=False

        return True
