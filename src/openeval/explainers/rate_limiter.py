"""Rate limiting utilities for explainer API calls.

This module provides token bucket and sliding window rate limiters
to control API call rates and prevent overwhelming services.
"""

import asyncio
import time
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Maximum requests per window
    max_requests: int = 100
    # Window size in seconds
    window_seconds: float = 60.0
    # Strategy to use
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    # For token bucket: tokens refilled per second
    refill_rate: Optional[float] = None
    # For token bucket: maximum bucket capacity
    bucket_size: Optional[int] = None
    # Whether to block or raise on limit exceeded
    block_on_limit: bool = False
    # Maximum time to block in seconds
    max_block_time: float = 30.0

    def __post_init__(self) -> None:
        """Set defaults based on strategy."""
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            if self.bucket_size is None:
                self.bucket_size = self.max_requests
            if self.refill_rate is None:
                self.refill_rate = self.max_requests / self.window_seconds


@dataclass
class RateLimitStatus:
    """Current status of rate limiter."""

    allowed: bool
    remaining: int
    limit: int
    reset_at: Optional[datetime] = None
    retry_after: Optional[float] = None

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP rate limit headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
        }
        if self.reset_at:
            headers["X-RateLimit-Reset"] = str(int(self.reset_at.timestamp()))
        if self.retry_after:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    def acquire(self) -> RateLimitStatus:
        """Attempt to acquire a permit.

        Returns:
            RateLimitStatus indicating if request is allowed.
        """
        pass

    @abstractmethod
    def get_status(self) -> RateLimitStatus:
        """Get current rate limit status without consuming a permit."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the rate limiter."""
        pass


class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter.

    Tokens are added at a constant rate up to a maximum bucket size.
    Each request consumes one token.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize token bucket limiter.

        Args:
            config: Rate limit configuration.
        """
        self._config = config
        self._bucket_size = config.bucket_size or config.max_requests
        self._refill_rate = config.refill_rate or (config.max_requests / config.window_seconds)
        self._tokens = float(self._bucket_size)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self._refill_rate
        self._tokens = min(self._bucket_size, self._tokens + tokens_to_add)
        self._last_refill = now

    def acquire(self) -> RateLimitStatus:
        """Attempt to acquire a token."""
        with self._lock:
            self._refill()

            if self._tokens >= 1:
                self._tokens -= 1
                return RateLimitStatus(
                    allowed=True,
                    remaining=int(self._tokens),
                    limit=self._bucket_size,
                )
            else:
                # Calculate time until next token
                retry_after = (1 - self._tokens) / self._refill_rate
                return RateLimitStatus(
                    allowed=False,
                    remaining=0,
                    limit=self._bucket_size,
                    retry_after=retry_after,
                )

    def get_status(self) -> RateLimitStatus:
        """Get current status without consuming a token."""
        with self._lock:
            self._refill()
            return RateLimitStatus(
                allowed=self._tokens >= 1,
                remaining=int(self._tokens),
                limit=self._bucket_size,
            )

    def reset(self) -> None:
        """Reset to full bucket."""
        with self._lock:
            self._tokens = float(self._bucket_size)
            self._last_refill = time.monotonic()


class SlidingWindowLimiter(RateLimiter):
    """Sliding window rate limiter.

    Tracks timestamps of recent requests and limits based on
    requests within the sliding window.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize sliding window limiter.

        Args:
            config: Rate limit configuration.
        """
        self._config = config
        self._window_seconds = config.window_seconds
        self._max_requests = config.max_requests
        self._requests: deque = deque()
        self._lock = threading.Lock()

    def _clean_old_requests(self) -> None:
        """Remove requests outside the current window."""
        cutoff = time.monotonic() - self._window_seconds
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

    def acquire(self) -> RateLimitStatus:
        """Attempt to record a request."""
        with self._lock:
            self._clean_old_requests()
            now = time.monotonic()

            if len(self._requests) < self._max_requests:
                self._requests.append(now)
                return RateLimitStatus(
                    allowed=True,
                    remaining=self._max_requests - len(self._requests),
                    limit=self._max_requests,
                )
            else:
                # Calculate when oldest request expires
                oldest = self._requests[0]
                retry_after = (oldest + self._window_seconds) - now
                return RateLimitStatus(
                    allowed=False,
                    remaining=0,
                    limit=self._max_requests,
                    retry_after=max(0, retry_after),
                )

    def get_status(self) -> RateLimitStatus:
        """Get current status without recording a request."""
        with self._lock:
            self._clean_old_requests()
            remaining = self._max_requests - len(self._requests)
            return RateLimitStatus(
                allowed=remaining > 0,
                remaining=remaining,
                limit=self._max_requests,
            )

    def reset(self) -> None:
        """Clear all recorded requests."""
        with self._lock:
            self._requests.clear()


class FixedWindowLimiter(RateLimiter):
    """Fixed window rate limiter.

    Counts requests within fixed time windows (e.g., per minute).
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize fixed window limiter.

        Args:
            config: Rate limit configuration.
        """
        self._config = config
        self._window_seconds = config.window_seconds
        self._max_requests = config.max_requests
        self._count = 0
        self._window_start = time.monotonic()
        self._lock = threading.Lock()

    def _check_window(self) -> None:
        """Reset count if window has passed."""
        now = time.monotonic()
        if now - self._window_start >= self._window_seconds:
            self._count = 0
            self._window_start = now

    def acquire(self) -> RateLimitStatus:
        """Attempt to record a request."""
        with self._lock:
            self._check_window()

            if self._count < self._max_requests:
                self._count += 1
                reset_at = datetime.fromtimestamp(
                    time.time() + (self._window_seconds - (time.monotonic() - self._window_start))
                )
                return RateLimitStatus(
                    allowed=True,
                    remaining=self._max_requests - self._count,
                    limit=self._max_requests,
                    reset_at=reset_at,
                )
            else:
                retry_after = self._window_seconds - (time.monotonic() - self._window_start)
                return RateLimitStatus(
                    allowed=False,
                    remaining=0,
                    limit=self._max_requests,
                    retry_after=max(0, retry_after),
                )

    def get_status(self) -> RateLimitStatus:
        """Get current status without recording a request."""
        with self._lock:
            self._check_window()
            return RateLimitStatus(
                allowed=self._count < self._max_requests,
                remaining=self._max_requests - self._count,
                limit=self._max_requests,
            )

    def reset(self) -> None:
        """Reset the window."""
        with self._lock:
            self._count = 0
            self._window_start = time.monotonic()


class LeakyBucketLimiter(RateLimiter):
    """Leaky bucket rate limiter.

    Requests are processed at a constant rate (like water leaking from a bucket).
    Excess requests queue up until the bucket overflows.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize leaky bucket limiter.

        Args:
            config: Rate limit configuration.
        """
        self._config = config
        self._bucket_size = config.bucket_size or config.max_requests
        self._leak_rate = config.refill_rate or (config.max_requests / config.window_seconds)
        self._water_level = 0.0
        self._last_leak = time.monotonic()
        self._lock = threading.Lock()

    def _leak(self) -> None:
        """Leak water based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_leak
        leaked = elapsed * self._leak_rate
        self._water_level = max(0, self._water_level - leaked)
        self._last_leak = now

    def acquire(self) -> RateLimitStatus:
        """Attempt to add water (a request) to the bucket."""
        with self._lock:
            self._leak()

            if self._water_level < self._bucket_size:
                self._water_level += 1
                return RateLimitStatus(
                    allowed=True,
                    remaining=int(self._bucket_size - self._water_level),
                    limit=self._bucket_size,
                )
            else:
                # Bucket full, calculate when space becomes available
                retry_after = 1 / self._leak_rate
                return RateLimitStatus(
                    allowed=False,
                    remaining=0,
                    limit=self._bucket_size,
                    retry_after=retry_after,
                )

    def get_status(self) -> RateLimitStatus:
        """Get current status without adding water."""
        with self._lock:
            self._leak()
            space = self._bucket_size - self._water_level
            return RateLimitStatus(
                allowed=space >= 1,
                remaining=int(space),
                limit=self._bucket_size,
            )

    def reset(self) -> None:
        """Empty the bucket."""
        with self._lock:
            self._water_level = 0
            self._last_leak = time.monotonic()


def create_rate_limiter(config: RateLimitConfig) -> RateLimiter:
    """Create a rate limiter based on configuration.

    Args:
        config: Rate limit configuration.

    Returns:
        Appropriate RateLimiter implementation.
    """
    limiters = {
        RateLimitStrategy.TOKEN_BUCKET: TokenBucketLimiter,
        RateLimitStrategy.SLIDING_WINDOW: SlidingWindowLimiter,
        RateLimitStrategy.FIXED_WINDOW: FixedWindowLimiter,
        RateLimitStrategy.LEAKY_BUCKET: LeakyBucketLimiter,
    }
    limiter_class = limiters.get(config.strategy, TokenBucketLimiter)
    return limiter_class(config)


def rate_limit(
    max_requests: int = 100,
    window_seconds: float = 60.0,
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
    block_on_limit: bool = False,
) -> Callable[[F], F]:
    """Decorator for rate limiting function calls.

    Args:
        max_requests: Maximum requests per window.
        window_seconds: Window size in seconds.
        strategy: Rate limiting strategy to use.
        block_on_limit: If True, block until limit resets; if False, raise.

    Returns:
        Decorated function with rate limiting.

    Example:
        @rate_limit(max_requests=10, window_seconds=60)
        def call_api(prompt: str) -> str:
            return api.generate(prompt)
    """
    config = RateLimitConfig(
        max_requests=max_requests,
        window_seconds=window_seconds,
        strategy=strategy,
        block_on_limit=block_on_limit,
    )
    limiter = create_rate_limiter(config)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            status = limiter.acquire()

            if not status.allowed:
                if block_on_limit and status.retry_after:
                    time.sleep(min(status.retry_after, config.max_block_time))
                    status = limiter.acquire()

                if not status.allowed:
                    raise RateLimitExceeded(
                        f"Rate limit exceeded. Retry after {status.retry_after:.1f}s",
                        retry_after=status.retry_after,
                    )

            return func(*args, **kwargs)

        wrapper.get_rate_limit_status = limiter.get_status  # type: ignore
        wrapper.reset_rate_limit = limiter.reset  # type: ignore

        return wrapper  # type: ignore

    return decorator


def async_rate_limit(
    max_requests: int = 100,
    window_seconds: float = 60.0,
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
    block_on_limit: bool = False,
) -> Callable[[F], F]:
    """Decorator for rate limiting async function calls.

    Args:
        max_requests: Maximum requests per window.
        window_seconds: Window size in seconds.
        strategy: Rate limiting strategy to use.
        block_on_limit: If True, await until limit resets; if False, raise.

    Returns:
        Decorated async function with rate limiting.

    Example:
        @async_rate_limit(max_requests=10, window_seconds=60)
        async def call_api_async(prompt: str) -> str:
            return await api.generate_async(prompt)
    """
    config = RateLimitConfig(
        max_requests=max_requests,
        window_seconds=window_seconds,
        strategy=strategy,
        block_on_limit=block_on_limit,
    )
    limiter = create_rate_limiter(config)

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            status = limiter.acquire()

            if not status.allowed:
                if block_on_limit and status.retry_after:
                    await asyncio.sleep(min(status.retry_after, config.max_block_time))
                    status = limiter.acquire()

                if not status.allowed:
                    raise RateLimitExceeded(
                        f"Rate limit exceeded. Retry after {status.retry_after:.1f}s",
                        retry_after=status.retry_after,
                    )

            return await func(*args, **kwargs)

        wrapper.get_rate_limit_status = limiter.get_status  # type: ignore
        wrapper.reset_rate_limit = limiter.reset  # type: ignore

        return wrapper  # type: ignore

    return decorator


@dataclass
class RateLimiterGroup:
    """Manage multiple rate limiters for different resources.

    Useful for API clients that have multiple rate limit tiers
    (e.g., per-endpoint and global limits).
    """

    limiters: Dict[str, RateLimiter] = field(default_factory=dict)

    def add_limiter(self, name: str, config: RateLimitConfig) -> None:
        """Add a rate limiter to the group.

        Args:
            name: Identifier for this limiter.
            config: Rate limit configuration.
        """
        self.limiters[name] = create_rate_limiter(config)

    def acquire_all(self) -> Dict[str, RateLimitStatus]:
        """Attempt to acquire from all limiters.

        All limiters must allow the request, or none are consumed.

        Returns:
            Dict mapping limiter names to their status.
        """
        statuses = {}

        # Check all limiters first
        for name, limiter in self.limiters.items():
            status = limiter.get_status()
            statuses[name] = status
            if not status.allowed:
                return statuses

        # All allowed, now consume from all
        for name, limiter in self.limiters.items():
            statuses[name] = limiter.acquire()

        return statuses

    def get_all_status(self) -> Dict[str, RateLimitStatus]:
        """Get status of all limiters."""
        return {name: limiter.get_status() for name, limiter in self.limiters.items()}

    def reset_all(self) -> None:
        """Reset all limiters."""
        for limiter in self.limiters.values():
            limiter.reset()


class ExplainerRateLimiter:
    """Specialized rate limiter for explainer API calls.

    Provides separate limits for different explanation levels
    and optional global limits.
    """

    def __init__(
        self,
        summary_limit: int = 100,
        detailed_limit: int = 50,
        expert_limit: int = 20,
        window_seconds: float = 60.0,
        global_limit: Optional[int] = None,
    ) -> None:
        """Initialize explainer rate limiter.

        Args:
            summary_limit: Max summary explanations per window.
            detailed_limit: Max detailed explanations per window.
            expert_limit: Max expert explanations per window.
            window_seconds: Window size in seconds.
            global_limit: Optional global limit across all levels.
        """
        self._group = RateLimiterGroup()

        # Per-level limiters
        self._group.add_limiter(
            "summary",
            RateLimitConfig(
                max_requests=summary_limit,
                window_seconds=window_seconds,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
            ),
        )
        self._group.add_limiter(
            "detailed",
            RateLimitConfig(
                max_requests=detailed_limit,
                window_seconds=window_seconds,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
            ),
        )
        self._group.add_limiter(
            "expert",
            RateLimitConfig(
                max_requests=expert_limit,
                window_seconds=window_seconds,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
            ),
        )

        # Optional global limiter
        if global_limit is not None:
            self._group.add_limiter(
                "global",
                RateLimitConfig(
                    max_requests=global_limit,
                    window_seconds=window_seconds,
                    strategy=RateLimitStrategy.TOKEN_BUCKET,
                ),
            )

        self._global_limit = global_limit

    def acquire(self, level: str = "summary") -> RateLimitStatus:
        """Acquire a permit for an explanation request.

        Args:
            level: Explanation level (summary, detailed, expert).

        Returns:
            RateLimitStatus for the request.
        """
        level = level.lower()
        if level not in self._group.limiters:
            level = "summary"

        # Check level-specific limiter
        level_limiter = self._group.limiters[level]
        status = level_limiter.acquire()

        if not status.allowed:
            return status

        # Check global limiter if configured
        if self._global_limit is not None:
            global_status = self._group.limiters["global"].acquire()
            if not global_status.allowed:
                return global_status

        return status

    def get_status(self, level: str = "summary") -> RateLimitStatus:
        """Get status for a specific level."""
        level = level.lower()
        if level not in self._group.limiters:
            level = "summary"
        return self._group.limiters[level].get_status()

    def get_all_status(self) -> Dict[str, RateLimitStatus]:
        """Get status for all levels."""
        return self._group.get_all_status()

    def reset(self) -> None:
        """Reset all limiters."""
        self._group.reset_all()


# Convenience singleton for global rate limiting
_global_explainer_rate_limiter: Optional[ExplainerRateLimiter] = None


def get_explainer_rate_limiter(
    summary_limit: int = 100,
    detailed_limit: int = 50,
    expert_limit: int = 20,
    window_seconds: float = 60.0,
    global_limit: Optional[int] = None,
) -> ExplainerRateLimiter:
    """Get or create the global explainer rate limiter.

    Args:
        summary_limit: Max summary explanations (only used on first call).
        detailed_limit: Max detailed explanations (only used on first call).
        expert_limit: Max expert explanations (only used on first call).
        window_seconds: Window size in seconds (only used on first call).
        global_limit: Optional global limit (only used on first call).

    Returns:
        Global ExplainerRateLimiter instance.
    """
    global _global_explainer_rate_limiter
    if _global_explainer_rate_limiter is None:
        _global_explainer_rate_limiter = ExplainerRateLimiter(
            summary_limit=summary_limit,
            detailed_limit=detailed_limit,
            expert_limit=expert_limit,
            window_seconds=window_seconds,
            global_limit=global_limit,
        )
    return _global_explainer_rate_limiter


def reset_explainer_rate_limiter() -> None:
    """Reset the global explainer rate limiter."""
    global _global_explainer_rate_limiter
    if _global_explainer_rate_limiter is not None:
        _global_explainer_rate_limiter.reset()
    _global_explainer_rate_limiter = None
