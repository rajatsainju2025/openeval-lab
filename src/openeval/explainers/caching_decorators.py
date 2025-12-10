"""Caching decorators for explainer functions.

This module provides simple, reusable decorators for caching
explanation results with configurable TTL and storage backends.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from hashlib import sha256
from typing import Any, Callable, Dict, Optional, TypeVar
import asyncio
import json
import threading

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class CacheEntry:
    """A cached value with metadata."""

    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    hits: int = 0
    key: str = ""

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def access(self) -> Any:
        """Access the cached value and increment hit counter."""
        self.hits += 1
        return self.value


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "total_entries": self.total_entries,
            "hit_rate": self.hit_rate,
        }


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None) -> None:
        """Initialize LRU cache.

        Args:
            maxsize: Maximum number of entries to store.
            ttl: Time-to-live in seconds for entries (None for no expiration).
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.RLock()
        self._stats = CacheStats()

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Create a cache key from function arguments."""
        key_data = {"args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return sha256(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats.hits += 1
            return entry.access()

    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        with self._lock:
            expires_at = None
            if self._ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=self._ttl)

            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = CacheEntry(value=value, expires_at=expires_at, key=key)
            else:
                # Evict if at capacity
                while len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                    self._stats.evictions += 1

                self._cache[key] = CacheEntry(value=value, expires_at=expires_at, key=key)

            self._stats.total_entries = len(self._cache)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._stats.total_entries = 0

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.total_entries = len(self._cache)
                return True
            return False

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


def cache(
    maxsize: int = 128,
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable[[F], F]:
    """Decorator for caching function results with LRU eviction.

    Args:
        maxsize: Maximum number of cached results.
        ttl: Time-to-live in seconds (None for no expiration).
        key_func: Optional function to generate cache keys.

    Returns:
        Decorated function with caching.

    Example:
        @cache(maxsize=100, ttl=300)
        def explain_code(code: str, level: str) -> str:
            # Expensive explanation generation
            return generate_explanation(code, level)
    """
    lru_cache = LRUCache(maxsize=maxsize, ttl=ttl)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_func is not None:
                key = key_func(*args, **kwargs)
            else:
                key = lru_cache._make_key(*args, **kwargs)

            cached = lru_cache.get(key)
            if cached is not None:
                return cached

            result = func(*args, **kwargs)
            lru_cache.set(key, result)
            return result

        # Attach cache control methods
        wrapper.cache_clear = lru_cache.clear  # type: ignore
        wrapper.cache_invalidate = lru_cache.invalidate  # type: ignore
        wrapper.cache_stats = lambda: lru_cache.stats  # type: ignore
        wrapper.cache_info = lambda: lru_cache.stats.to_dict()  # type: ignore

        return wrapper  # type: ignore

    return decorator


def async_cache(
    maxsize: int = 128,
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable[[F], F]:
    """Decorator for caching async function results.

    Args:
        maxsize: Maximum number of cached results.
        ttl: Time-to-live in seconds (None for no expiration).
        key_func: Optional function to generate cache keys.

    Returns:
        Decorated async function with caching.

    Example:
        @async_cache(maxsize=100, ttl=300)
        async def explain_code_async(code: str) -> str:
            return await generate_explanation_async(code)
    """
    lru_cache = LRUCache(maxsize=maxsize, ttl=ttl)
    _lock = asyncio.Lock()

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_func is not None:
                key = key_func(*args, **kwargs)
            else:
                key = lru_cache._make_key(*args, **kwargs)

            cached = lru_cache.get(key)
            if cached is not None:
                return cached

            async with _lock:
                # Double-check after acquiring lock
                cached = lru_cache.get(key)
                if cached is not None:
                    return cached

                result = await func(*args, **kwargs)
                lru_cache.set(key, result)
                return result

        wrapper.cache_clear = lru_cache.clear  # type: ignore
        wrapper.cache_invalidate = lru_cache.invalidate  # type: ignore
        wrapper.cache_stats = lambda: lru_cache.stats  # type: ignore
        wrapper.cache_info = lambda: lru_cache.stats.to_dict()  # type: ignore

        return wrapper  # type: ignore

    return decorator


class CacheNamespace:
    """Namespaced cache for organizing related cached values."""

    def __init__(self, namespace: str, maxsize: int = 128, ttl: Optional[float] = None) -> None:
        """Initialize namespaced cache.

        Args:
            namespace: Namespace identifier for this cache.
            maxsize: Maximum entries per namespace.
            ttl: Default TTL for entries.
        """
        self.namespace = namespace
        self._cache = LRUCache(maxsize=maxsize, ttl=ttl)

    def _namespaced_key(self, key: str) -> str:
        """Create a namespaced key."""
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the namespace."""
        return self._cache.get(self._namespaced_key(key))

    def set(self, key: str, value: Any) -> None:
        """Set a value in the namespace."""
        self._cache.set(self._namespaced_key(key), value)

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific key in the namespace."""
        return self._cache.invalidate(self._namespaced_key(key))

    def clear(self) -> None:
        """Clear all entries in this namespace."""
        self._cache.clear()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.stats


class ExplainerCache:
    """Specialized cache for code explanation results.

    Provides convenient caching specifically for explainer workflows
    with code-aware key generation and namespace support.
    """

    def __init__(
        self,
        maxsize: int = 256,
        ttl: Optional[float] = 3600,  # 1 hour default
    ) -> None:
        """Initialize explainer cache.

        Args:
            maxsize: Maximum cached explanations.
            ttl: TTL in seconds for explanations.
        """
        self._summary_cache = CacheNamespace("summary", maxsize=maxsize, ttl=ttl)
        self._detailed_cache = CacheNamespace("detailed", maxsize=maxsize, ttl=ttl)
        self._expert_cache = CacheNamespace("expert", maxsize=maxsize, ttl=ttl)

    def _code_key(self, code: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a cache key from code and context."""
        key_data = {"code": code, "context": context or {}}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return sha256(key_str.encode()).hexdigest()

    def get_explanation(
        self,
        code: str,
        level: str = "summary",
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Get a cached explanation.

        Args:
            code: The code that was explained.
            level: Explanation level (summary, detailed, expert).
            context: Optional context used for explanation.

        Returns:
            Cached explanation or None.
        """
        key = self._code_key(code, context)
        cache = self._get_cache_for_level(level)
        return cache.get(key)

    def set_explanation(
        self,
        code: str,
        explanation: Any,
        level: str = "summary",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache an explanation.

        Args:
            code: The code that was explained.
            explanation: The explanation to cache.
            level: Explanation level (summary, detailed, expert).
            context: Optional context used for explanation.
        """
        key = self._code_key(code, context)
        cache = self._get_cache_for_level(level)
        cache.set(key, explanation)

    def _get_cache_for_level(self, level: str) -> CacheNamespace:
        """Get the appropriate cache for an explanation level."""
        caches = {
            "summary": self._summary_cache,
            "detailed": self._detailed_cache,
            "expert": self._expert_cache,
        }
        return caches.get(level.lower(), self._summary_cache)

    def invalidate_code(self, code: str) -> None:
        """Invalidate all cached explanations for a piece of code."""
        for level in ["summary", "detailed", "expert"]:
            key = self._code_key(code, None)
            cache = self._get_cache_for_level(level)
            cache.invalidate(key)

    def clear_all(self) -> None:
        """Clear all cached explanations."""
        self._summary_cache.clear()
        self._detailed_cache.clear()
        self._expert_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for all levels."""
        return {
            "summary": self._summary_cache.stats.to_dict(),
            "detailed": self._detailed_cache.stats.to_dict(),
            "expert": self._expert_cache.stats.to_dict(),
        }


def memoize(func: F) -> F:
    """Simple memoization decorator with no eviction.

    Use for functions with limited unique inputs.
    For larger caches, use @cache instead.

    Example:
        @memoize
        def parse_code(code: str) -> AST:
            return ast.parse(code)
    """
    memo: Dict[str, Any] = {}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key_data = {"args": args, "kwargs": kwargs}
        key = json.dumps(key_data, sort_keys=True, default=str)

        if key not in memo:
            memo[key] = func(*args, **kwargs)
        return memo[key]

    wrapper.cache_clear = memo.clear  # type: ignore
    return wrapper  # type: ignore


def conditional_cache(
    condition: Callable[..., bool],
    maxsize: int = 128,
    ttl: Optional[float] = None,
) -> Callable[[F], F]:
    """Cache only when a condition is met.

    Args:
        condition: Function that returns True to enable caching.
        maxsize: Maximum cached results.
        ttl: Time-to-live in seconds.

    Example:
        @conditional_cache(condition=lambda code, level: len(code) > 100)
        def explain(code: str, level: str) -> str:
            # Only cache explanations for code > 100 chars
            return generate_explanation(code, level)
    """
    lru_cache = LRUCache(maxsize=maxsize, ttl=ttl)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            should_cache = condition(*args, **kwargs)

            if should_cache:
                key = lru_cache._make_key(*args, **kwargs)
                cached = lru_cache.get(key)
                if cached is not None:
                    return cached

            result = func(*args, **kwargs)

            if should_cache:
                key = lru_cache._make_key(*args, **kwargs)
                lru_cache.set(key, result)

            return result

        wrapper.cache_clear = lru_cache.clear  # type: ignore
        wrapper.cache_stats = lambda: lru_cache.stats  # type: ignore

        return wrapper  # type: ignore

    return decorator


# Convenience singleton for global explainer caching
_global_explainer_cache: Optional[ExplainerCache] = None


def get_explainer_cache(maxsize: int = 256, ttl: Optional[float] = 3600) -> ExplainerCache:
    """Get or create the global explainer cache.

    Args:
        maxsize: Maximum entries (only used on first call).
        ttl: TTL in seconds (only used on first call).

    Returns:
        Global ExplainerCache instance.
    """
    global _global_explainer_cache
    if _global_explainer_cache is None:
        _global_explainer_cache = ExplainerCache(maxsize=maxsize, ttl=ttl)
    return _global_explainer_cache


def reset_explainer_cache() -> None:
    """Reset the global explainer cache."""
    global _global_explainer_cache
    if _global_explainer_cache is not None:
        _global_explainer_cache.clear_all()
    _global_explainer_cache = None
