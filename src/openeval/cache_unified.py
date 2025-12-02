"""
Unified High-Performance Caching System

Consolidates all caching functionality into an efficient, TTL-aware cache
with memory management, LRU eviction, and async support.
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, Optional, TypeVar, Callable
from functools import wraps
from collections import OrderedDict
import gc

T = TypeVar("T")


class UnifiedCache:
    """High-performance cache with TTL, LRU eviction, and memory management."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = 3600,  # 1 hour default
        cleanup_interval: float = 300,  # 5 minutes
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            self._cleanup_if_needed()

            if key not in self._cache:
                self._misses += 1
                return default

            entry = self._cache[key]

            # Check TTL
            if self._is_expired(entry):
                del self._cache[key]
                self._misses += 1
                return default

            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self._lock:
            now = time.time()
            ttl_value = ttl if ttl is not None else self.default_ttl
            expires_at = now + ttl_value if ttl_value is not None else None

            entry = {"value": value, "created_at": now, "expires_at": expires_at, "access_count": 0}

            self._cache[key] = entry
            self._cache.move_to_end(key)  # Most recent

            # Evict if needed
            self._evict_if_needed()

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self.max_size,
            "memory_usage_mb": self._estimate_memory_usage(),
        }

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if entry is expired."""
        if entry["expires_at"] is None:
            return False
        return time.time() > entry["expires_at"]

    def _cleanup_if_needed(self) -> None:
        """Clean up expired entries if needed."""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        self._cleanup_expired()
        self._last_cleanup = now

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        time.time()
        expired_keys = [key for key, entry in self._cache.items() if self._is_expired(entry)]

        for key in expired_keys:
            del self._cache[key]
            self._evictions += 1

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self._cache) > self.max_size:
            # Remove oldest (first) entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._evictions += 1

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        import sys

        total_size = 0
        sample_size = min(10, len(self._cache))

        if sample_size == 0:
            return 0.0

        # Sample some entries to estimate
        sample_keys = list(self._cache.keys())[:sample_size]
        for key in sample_keys:
            entry = self._cache[key]
            total_size += sys.getsizeof(key)
            total_size += sys.getsizeof(entry)
            total_size += sys.getsizeof(entry["value"])

        # Extrapolate to full cache
        avg_size = total_size / sample_size
        estimated_total = avg_size * len(self._cache)

        return estimated_total / (1024 * 1024)


# Global cache instances for different use cases
_PREDICTION_CACHE = UnifiedCache(max_size=5000, default_ttl=7200)  # 2 hours
_DATASET_CACHE = UnifiedCache(max_size=100, default_ttl=3600)  # 1 hour
_VALIDATION_CACHE = UnifiedCache(max_size=1000, default_ttl=1800)  # 30 minutes
_COMPUTATION_CACHE = UnifiedCache(max_size=2000, default_ttl=3600)  # 1 hour


def cached(
    cache: Optional[UnifiedCache] = None,
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
) -> Callable:
    """Decorator for caching function results."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        nonlocal cache
        if cache is None:
            cache = _COMPUTATION_CACHE

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if cache is None:
                return func(*args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                args_hash = hash((args, tuple(sorted(kwargs.items()))))
                cache_key = f"{func.__name__}:{args_hash}"

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        # Cache is managed internally

        return wrapper

    return decorator


def cache_prediction(ttl: float = 7200) -> Callable:
    """Decorator for caching prediction results."""
    return cached(cache=_PREDICTION_CACHE, ttl=ttl)


def cache_dataset(ttl: float = 3600) -> Callable:
    """Decorator for caching dataset operations."""
    return cached(cache=_DATASET_CACHE, ttl=ttl)


def cache_validation(ttl: float = 1800) -> Callable:
    """Decorator for caching validation results."""
    return cached(cache=_VALIDATION_CACHE, ttl=ttl)


class LRUCache:
    """Simple LRU cache for backward compatibility."""

    def __init__(self, max_size: int = 128):
        self._cache = UnifiedCache(max_size=max_size, default_ttl=None)

    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._cache.set(key, value, ttl=None)

    def __getitem__(self, key: str) -> Any:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


# Memory-aware cache manager
class CacheManager:
    """Global cache manager with memory monitoring."""

    def __init__(self):
        self.caches = {
            "predictions": _PREDICTION_CACHE,
            "datasets": _DATASET_CACHE,
            "validation": _VALIDATION_CACHE,
            "computation": _COMPUTATION_CACHE,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        total_memory = 0
        total_size = 0

        for name, cache in self.caches.items():
            cache_stats = cache.stats()
            stats[name] = cache_stats
            total_memory += cache_stats["memory_usage_mb"]
            total_size += cache_stats["size"]

        stats["total"] = {"memory_usage_mb": total_memory, "total_entries": total_size}

        return stats

    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        gc.collect()  # Force garbage collection

    def cleanup_expired(self) -> None:
        """Clean up expired entries in all caches."""
        for cache in self.caches.values():
            cache._cleanup_expired()

    def optimize_memory(self, target_mb: float = 100) -> None:
        """Optimize memory usage by evicting entries."""
        current_memory = sum(cache.stats()["memory_usage_mb"] for cache in self.caches.values())

        if current_memory <= target_mb:
            return

        # Reduce cache sizes proportionally
        reduction_factor = target_mb / current_memory
        for cache in self.caches.values():
            new_size = int(cache.max_size * reduction_factor)
            cache.max_size = max(10, new_size)  # Minimum size of 10
            cache._evict_if_needed()


# Global cache manager instance
cache_manager = CacheManager()


__all__ = [
    "UnifiedCache",
    "cached",
    "cache_prediction",
    "cache_dataset",
    "cache_validation",
    "LRUCache",
    "CacheManager",
    "cache_manager",
]
