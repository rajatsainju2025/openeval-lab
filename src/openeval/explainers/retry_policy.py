"""Retry policies for resilient explanation generation.

Provides configurable retry logic with exponential backoff and jitter.
"""

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from .types import CodeElement, ExplainLevel, ExplanationResult

T = TypeVar("T")


class RetryStrategy(str, Enum):
    """Available retry strategies."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    jitter_factor: float = 0.1
    retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=list)

    def should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger retry.

        Args:
            exception: The exception that occurred.

        Returns:
            True if should retry, False otherwise.
        """
        # Check non-retryable first
        for exc_type in self.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False

        # Check if retryable
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True

        return False


class BackoffCalculator(ABC):
    """Abstract base for backoff calculation."""

    @abstractmethod
    def calculate(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for given attempt.

        Args:
            attempt: Current attempt number (0-indexed).
            config: Retry configuration.

        Returns:
            Delay in seconds.
        """
        pass


class ExponentialBackoff(BackoffCalculator):
    """Exponential backoff with optional jitter."""

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        """Calculate exponential delay."""
        delay = config.base_delay * (2**attempt)
        delay = min(delay, config.max_delay)

        if config.jitter:
            jitter = delay * config.jitter_factor * random.random()
            delay += jitter

        return delay


class LinearBackoff(BackoffCalculator):
    """Linear backoff."""

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        """Calculate linear delay."""
        delay = config.base_delay * (attempt + 1)
        delay = min(delay, config.max_delay)

        if config.jitter:
            jitter = delay * config.jitter_factor * random.random()
            delay += jitter

        return delay


class ConstantBackoff(BackoffCalculator):
    """Constant delay between retries."""

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        """Calculate constant delay."""
        delay = config.base_delay

        if config.jitter:
            jitter = delay * config.jitter_factor * random.random()
            delay += jitter

        return delay


class FibonacciBackoff(BackoffCalculator):
    """Fibonacci sequence backoff."""

    def _fibonacci(self, n: int) -> int:
        """Calculate nth fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        """Calculate fibonacci delay."""
        fib = self._fibonacci(attempt + 2)  # Start from fib(2)
        delay = config.base_delay * fib
        delay = min(delay, config.max_delay)

        if config.jitter:
            jitter = delay * config.jitter_factor * random.random()
            delay += jitter

        return delay


# Strategy registry
_backoff_strategies: Dict[RetryStrategy, BackoffCalculator] = {
    RetryStrategy.EXPONENTIAL: ExponentialBackoff(),
    RetryStrategy.LINEAR: LinearBackoff(),
    RetryStrategy.CONSTANT: ConstantBackoff(),
    RetryStrategy.FIBONACCI: FibonacciBackoff(),
}


@dataclass
class RetryAttempt:
    """Record of a retry attempt."""

    attempt_number: int
    exception: Optional[Exception]
    delay_before: float
    timestamp: float
    success: bool


@dataclass
class RetryResult:
    """Result of a retried operation."""

    success: bool
    result: Optional[Any]
    attempts: List[RetryAttempt]
    total_time: float
    final_exception: Optional[Exception]

    @property
    def attempt_count(self) -> int:
        """Get total number of attempts."""
        return len(self.attempts)


class RetryPolicy:
    """Configurable retry policy for explainers.

    Provides retry logic with configurable backoff strategies.
    """

    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        """Initialize retry policy.

        Args:
            config: Retry configuration (uses defaults if None).
        """
        self.config = config or RetryConfig()

    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> RetryResult:
        """Execute function with retry logic.

        Args:
            func: Function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            RetryResult with outcome and attempt history.
        """
        start_time = time.time()
        attempts: List[RetryAttempt] = []
        backoff = _backoff_strategies[self.config.strategy]

        for attempt in range(self.config.max_attempts):
            delay_before = 0.0 if attempt == 0 else backoff.calculate(attempt - 1, self.config)

            if delay_before > 0:
                time.sleep(delay_before)

            try:
                result = func(*args, **kwargs)
                attempts.append(
                    RetryAttempt(
                        attempt_number=attempt + 1,
                        exception=None,
                        delay_before=delay_before,
                        timestamp=time.time(),
                        success=True,
                    )
                )
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_time=time.time() - start_time,
                    final_exception=None,
                )

            except Exception as e:
                attempts.append(
                    RetryAttempt(
                        attempt_number=attempt + 1,
                        exception=e,
                        delay_before=delay_before,
                        timestamp=time.time(),
                        success=False,
                    )
                )

                # Check if we should retry
                if not self.config.should_retry(e):
                    return RetryResult(
                        success=False,
                        result=None,
                        attempts=attempts,
                        total_time=time.time() - start_time,
                        final_exception=e,
                    )

                # If last attempt, don't retry
                if attempt == self.config.max_attempts - 1:
                    return RetryResult(
                        success=False,
                        result=None,
                        attempts=attempts,
                        total_time=time.time() - start_time,
                        final_exception=e,
                    )

        # Should not reach here, but handle gracefully
        return RetryResult(
            success=False,
            result=None,
            attempts=attempts,
            total_time=time.time() - start_time,
            final_exception=Exception("Max retries exceeded"),
        )


def retry_decorator(
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for adding retry logic to functions.

    Args:
        config: Retry configuration.

    Returns:
        Decorator function.
    """
    policy = RetryPolicy(config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = policy.execute(func, *args, **kwargs)
            if result.success:
                return result.result
            raise result.final_exception or Exception("Retry failed")

        return wrapper

    return decorator


class RetryableExplainer:
    """Wrapper adding retry capability to any explainer."""

    def __init__(
        self,
        explainer: Any,  # CodeExplainer
        config: Optional[RetryConfig] = None,
    ) -> None:
        """Initialize retryable explainer.

        Args:
            explainer: CodeExplainer to wrap.
            config: Retry configuration.
        """
        self.explainer = explainer
        self.policy = RetryPolicy(config)
        self._last_result: Optional[RetryResult] = None

    def explain(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.DETAILED,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExplanationResult:
        """Explain with retry logic.

        Args:
            element: Code element to explain.
            level: Explanation detail level.
            context: Additional context.

        Returns:
            ExplanationResult.

        Raises:
            Exception: If all retries fail.
        """
        result = self.policy.execute(self.explainer.explain, element, level, context)
        self._last_result = result

        if result.success:
            return result.result
        raise result.final_exception or Exception("Explanation failed after retries")

    def get_last_retry_result(self) -> Optional[RetryResult]:
        """Get the result of the last retry operation.

        Returns:
            RetryResult or None if no operations yet.
        """
        return self._last_result


# Preset configurations
RETRY_CONFIGS = {
    "aggressive": RetryConfig(
        max_attempts=5,
        base_delay=0.5,
        max_delay=30.0,
        strategy=RetryStrategy.EXPONENTIAL,
        jitter=True,
    ),
    "conservative": RetryConfig(
        max_attempts=2,
        base_delay=2.0,
        max_delay=10.0,
        strategy=RetryStrategy.LINEAR,
        jitter=False,
    ),
    "patient": RetryConfig(
        max_attempts=10,
        base_delay=1.0,
        max_delay=120.0,
        strategy=RetryStrategy.FIBONACCI,
        jitter=True,
    ),
    "quick": RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        max_delay=1.0,
        strategy=RetryStrategy.CONSTANT,
        jitter=True,
    ),
}


def get_retry_config(preset: str) -> RetryConfig:
    """Get a preset retry configuration.

    Args:
        preset: Name of preset ('aggressive', 'conservative', 'patient', 'quick').

    Returns:
        RetryConfig.

    Raises:
        ValueError: If preset not found.
    """
    if preset not in RETRY_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(RETRY_CONFIGS.keys())}")
    return RETRY_CONFIGS[preset]
