"""Tests for error handling framework."""

import pytest
import time

from openeval.error_handling import (
    ErrorTracker,
    ErrorSeverity,
    ErrorContext,
    RetryConfig,
    retry_with_config,
    CircuitBreaker,
    ErrorRecovery,
    create_robust_evaluation_context,
)


class TestErrorTracker:
    """Test ErrorTracker functionality."""

    def test_error_tracker_creation(self):
        """Test creating an error tracker."""
        tracker = ErrorTracker()
        assert tracker.errors == []
        assert tracker.error_counts == {}

    def test_log_error(self):
        """Test logging an error."""
        tracker = ErrorTracker()
        error = ValueError("Test error")

        ErrorContext(
            error_type="ValueError",
            severity=ErrorSeverity.MEDIUM,
            message="Test error",
            traceback_str="",
            timestamp=time.time(),
            context={"test": True},
        )

        logged = tracker.log_error(error, context={"test": True})

        assert len(tracker.errors) == 1
        assert tracker.error_counts["ValueError"] == 1
        assert logged.error_type == "ValueError"
        assert logged.context["test"] is True

    def test_get_error_summary(self):
        """Test getting error summary."""
        tracker = ErrorTracker()

        # Log some errors
        tracker.log_error(ValueError("Error 1"), severity=ErrorSeverity.LOW)
        tracker.log_error(RuntimeError("Error 2"), severity=ErrorSeverity.CRITICAL)
        tracker.log_error(ValueError("Error 3"), severity=ErrorSeverity.MEDIUM)

        summary = tracker.get_error_summary()

        assert summary["total_errors"] == 3
        assert summary["error_types"]["ValueError"] == 2
        assert summary["error_types"]["RuntimeError"] == 1
        assert summary["critical_errors"] == 1
        assert len(summary["recent_errors"]) == 3


class TestRetryConfig:
    """Test RetryConfig functionality."""

    def test_retry_config_creation(self):
        """Test creating retry config."""
        config = RetryConfig(max_attempts=5, base_delay=2.0)
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.retryable_errors == [ConnectionError, TimeoutError, OSError]

    def test_is_retryable(self):
        """Test checking if error is retryable."""
        config = RetryConfig()
        assert config.is_retryable(ConnectionError("Test"))
        assert config.is_retryable(TimeoutError("Test"))
        assert not config.is_retryable(ValueError("Test"))

    def test_get_delay(self):
        """Test calculating delay."""
        config = RetryConfig(base_delay=1.0, exponential_backoff=True, jitter=False)

        # First attempt delay
        delay1 = config.get_delay(1)
        assert delay1 == 1.0

        # Second attempt delay
        delay2 = config.get_delay(2)
        assert delay2 == 2.0

        # Third attempt delay
        delay3 = config.get_delay(3)
        assert delay3 == 4.0


class TestRetryDecorator:
    """Test retry decorator functionality."""

    def test_retry_success(self):
        """Test successful retry."""
        config = RetryConfig(max_attempts=3)
        tracker = ErrorTracker()

        call_count = 0

        @retry_with_config(config, tracker)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 2

    def test_retry_failure(self):
        """Test retry failure after max attempts."""
        config = RetryConfig(max_attempts=2)
        tracker = ErrorTracker()

        call_count = 0

        @retry_with_config(config, tracker)
        def test_func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")

        with pytest.raises(ConnectionError):
            test_func()

        assert call_count == 2
        assert len(tracker.errors) > 0

    def test_non_retryable_error(self):
        """Test non-retryable error fails immediately."""
        config = RetryConfig(max_attempts=3)
        tracker = ErrorTracker()

        call_count = 0

        @retry_with_config(config, tracker)
        def test_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError):
            test_func()

        assert call_count == 1


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    def test_circuit_breaker_creation(self):
        """Test creating circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=30.0)
        assert breaker.failure_threshold == 3
        assert breaker.timeout == 30.0
        assert breaker.failure_count == 0
        assert breaker.state == "closed"

    def test_circuit_breaker_decorator_success(self):
        """Test successful operation with decorator."""
        breaker = CircuitBreaker(failure_threshold=2)

        @breaker
        def test_func():
            return "success"

        # Should succeed
        result = test_func()
        assert result == "success"
        assert breaker.failure_count == 0
        assert breaker.state == "closed"

    def test_circuit_breaker_decorator_failure(self):
        """Test circuit breaker opening after failures."""
        breaker = CircuitBreaker(failure_threshold=2)

        @breaker
        def failing_func():
            raise RuntimeError("Error")

        # First failure
        with pytest.raises(RuntimeError):
            failing_func()

        assert breaker.failure_count == 1
        assert breaker.state == "closed"

        # Second failure - should open circuit
        with pytest.raises(RuntimeError):
            failing_func()

        assert breaker.failure_count == 2
        assert breaker.state == "open"

        # Third call should fail fast
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            failing_func()


class TestErrorRecovery:
    """Test ErrorRecovery functionality."""

    def test_error_recovery_creation(self):
        """Test creating error recovery."""
        tracker = ErrorTracker()
        recovery = ErrorRecovery(tracker)
        assert recovery.error_tracker is tracker

    def test_auto_recover_decorator(self):
        """Test auto recover decorator."""
        tracker = ErrorTracker()
        recovery = ErrorRecovery(tracker)

        call_count = 0

        @recovery.auto_recover
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ImportError("Missing package")
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 2


class TestErrorHandlingSuite:
    """Test error handling suite creation."""

    def test_create_error_handling_suite(self):
        """Test creating complete error handling suite."""
        suite = create_robust_evaluation_context()

        assert "error_tracker" in suite
        assert "network_retry" in suite
        assert "temp_failure_retry" in suite
        assert "circuit_breaker" in suite
        assert "recovery" in suite

        assert isinstance(suite["error_tracker"], ErrorTracker)
        assert isinstance(suite["network_retry"], RetryConfig)
        assert isinstance(suite["temp_failure_retry"], RetryConfig)
        assert isinstance(suite["circuit_breaker"], CircuitBreaker)
        assert isinstance(suite["recovery"], ErrorRecovery)
