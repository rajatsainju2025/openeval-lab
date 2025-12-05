"""Circuit breaker pattern implementation for adapter failure handling.

This module provides a circuit breaker pattern to prevent cascading failures when
adapters fail repeatedly. The circuit breaker monitors adapter health and temporarily
disables failing adapters with exponential backoff.
"""

from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import time
import asyncio

from .logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """States for the circuit breaker."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes needed to close from half-open
    timeout_seconds: float = 60.0  # Time before attempting recovery
    max_timeout_seconds: float = 300.0  # Maximum backoff time
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    half_open_max_calls: int = 3  # Max calls to test in half-open state

    # Error categorization
    count_timeouts: bool = True
    count_rate_limits: bool = False  # Don't open circuit on rate limits
    count_auth_errors: bool = False  # Don't open circuit on auth errors


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker
            config: Configuration for behavior
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()

        self._failure_count = 0
        self._success_count = 0
        self._opened_at: Optional[float] = None
        self._current_timeout = self.config.timeout_seconds
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    def should_allow_request(self) -> bool:
        """Check if a request should be allowed through the circuit.

        Returns:
            True if request should proceed, False if circuit is open
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self._opened_at and (time.time() - self._opened_at) >= self._current_timeout:
                self._transition_to_half_open()
                return True
            return False

        # HALF_OPEN state - allow limited requests
        if self._half_open_calls < self.config.half_open_max_calls:
            return True
        return False

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute (can be sync or async)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpen: If circuit is open and request is rejected
            Exception: Any exception from the wrapped function
        """
        async with self._lock:
            if not self.should_allow_request():
                self.stats.rejected_calls += 1
                logger.warning(
                    f"Circuit breaker '{self.name}' is OPEN, rejecting request "
                    f"(timeout: {self._current_timeout}s)"
                )
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service will be retried in {self._current_timeout:.0f}s"
                )

            if self.state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        self.stats.total_calls += 1

        try:
            # Execute function (handle both sync and async)
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - record it
            await self._on_success()
            return result

        except Exception as e:
            # Failure - record it
            await self._on_failure(e)
            raise

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self.stats.successful_calls += 1
            self.stats.last_success_time = time.time()
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0
            self._failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()

    async def _on_failure(self, error: Exception):
        """Handle failed call.

        Args:
            error: The exception that occurred
        """
        # Determine if this error should count toward opening the circuit
        if not self._should_count_error(error):
            return

        async with self._lock:
            self.stats.failed_calls += 1
            self.stats.last_failure_time = time.time()
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self._failure_count += 1

            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately reopens circuit
                self._transition_to_open()
            elif self.state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()

    def _should_count_error(self, error: Exception) -> bool:
        """Determine if error should count toward circuit breaker thresholds.

        Args:
            error: The exception to evaluate

        Returns:
            True if error should count, False otherwise
        """
        error_str = str(error).lower()

        # Check timeout errors
        if "timeout" in error_str or isinstance(error, TimeoutError):
            return self.config.count_timeouts

        # Check rate limit errors
        if "rate limit" in error_str or "429" in error_str:
            return self.config.count_rate_limits

        # Check auth errors
        if "auth" in error_str or "401" in error_str or "403" in error_str:
            return self.config.count_auth_errors

        # Count most other errors
        return True

    def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        if self.state != CircuitState.OPEN:
            logger.warning(
                f"Circuit breaker '{self.name}' opening after {self._failure_count} failures. "
                f"Timeout: {self._current_timeout}s"
            )
            self.state = CircuitState.OPEN
            self._opened_at = time.time()
            self.stats.state_changes += 1
            self._success_count = 0
            self._half_open_calls = 0

            # Apply exponential backoff
            self._current_timeout = min(
                self._current_timeout * self.config.backoff_multiplier,
                self.config.max_timeout_seconds,
            )

    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state for testing."""
        logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN for testing")
        self.state = CircuitState.HALF_OPEN
        self.stats.state_changes += 1
        self._success_count = 0
        self._half_open_calls = 0

    def _transition_to_closed(self):
        """Transition circuit to CLOSED state (normal operation)."""
        logger.info(
            f"Circuit breaker '{self.name}' closing after {self._success_count} successful test calls"
        )
        self.state = CircuitState.CLOSED
        self.stats.state_changes += 1
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

        # Reset timeout on successful recovery
        self._current_timeout = self.config.timeout_seconds

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        logger.info(f"Circuit breaker '{self.name}' manually reset")
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._current_timeout = self.config.timeout_seconds
        self.stats.state_changes += 1

    def get_state(self) -> Dict[str, Any]:
        """Get current state and statistics.

        Returns:
            Dictionary with state information
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "current_timeout": self._current_timeout,
            "stats": {
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "rejected_calls": self.stats.rejected_calls,
                "success_rate": (
                    self.stats.successful_calls / self.stats.total_calls
                    if self.stats.total_calls > 0
                    else 0.0
                ),
                "consecutive_failures": self.stats.consecutive_failures,
                "consecutive_successes": self.stats.consecutive_successes,
                "state_changes": self.stats.state_changes,
            },
        }


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        """Initialize empty registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one.

        Args:
            name: Circuit breaker identifier
            config: Optional configuration (uses default if not provided)

        Returns:
            Circuit breaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                config=config or self._default_config,
            )
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name.

        Args:
            name: Circuit breaker identifier

        Returns:
            Circuit breaker instance or None if not found
        """
        return self._breakers.get(name)

    def reset_all(self):
        """Reset all circuit breakers to CLOSED state."""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get state information for all circuit breakers.

        Returns:
            Dictionary mapping breaker names to their states
        """
        return {name: breaker.get_state() for name, breaker in self._breakers.items()}

    def health_check(self) -> Dict[str, bool]:
        """Check health of all circuit breakers.

        Returns:
            Dictionary mapping breaker names to health status (True if CLOSED)
        """
        return {
            name: breaker.state == CircuitState.CLOSED for name, breaker in self._breakers.items()
        }


# Global registry instance
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker from global registry.

    Args:
        name: Circuit breaker identifier
        config: Optional configuration

    Returns:
        Circuit breaker instance
    """
    return _global_registry.get_or_create(name, config)


def reset_all_circuit_breakers():
    """Reset all circuit breakers in the global registry."""
    _global_registry.reset_all()


def get_all_circuit_breaker_states() -> Dict[str, Dict[str, Any]]:
    """Get states of all circuit breakers in the global registry.

    Returns:
        Dictionary of breaker states
    """
    return _global_registry.get_all_states()
