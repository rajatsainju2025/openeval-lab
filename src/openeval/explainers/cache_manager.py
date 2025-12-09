"""Cache management system for code explainers.

Provides pluggable caching strategies for explanation results.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .types import ExplanationResult


class CacheManager(ABC):
    """Abstract base class for caching strategies.

    Enables pluggable cache backends (in-memory, Redis, etc).
    """

    @abstractmethod
    def get(self, key: str) -> Optional[ExplanationResult]:
        """Retrieve cached explanation result.

        Args:
            key: Cache key.

        Returns:
            Cached ExplanationResult or None if not found.
        """
        pass

    @abstractmethod
    def set(self, key: str, value: ExplanationResult, ttl: Optional[int] = None) -> None:
        """Store explanation result in cache.

        Args:
            key: Cache key.
            value: ExplanationResult to cache.
            ttl: Time-to-live in seconds (None = no expiry).
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cached entry.

        Args:
            key: Cache key.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached entries."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key.

        Returns:
            True if key exists.
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with stats like hit_rate, size, etc.
        """
        pass


class InMemoryCacheManager(CacheManager):
    """In-memory cache implementation using dict.

    Suitable for single-process use. Good default.
    """

    def __init__(self) -> None:
        """Initialize in-memory cache."""
        self._cache: Dict[str, ExplanationResult] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[ExplanationResult]:
        """Retrieve from in-memory cache."""
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, key: str, value: ExplanationResult, ttl: Optional[int] = None) -> None:
        """Store in in-memory cache."""
        # Note: ttl not implemented in basic in-memory cache
        # Could be enhanced with expiration tracking
        self._cache[key] = value

    def delete(self, key: str) -> bool:
        """Delete from in-memory cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._cache

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_requests": total,
        }


class NoOpCacheManager(CacheManager):
    """No-op cache implementation.

    Useful for testing or when caching is disabled.
    """

    def get(self, key: str) -> Optional[ExplanationResult]:
        """Always return None."""
        return None

    def set(self, key: str, value: ExplanationResult, ttl: Optional[int] = None) -> None:
        """Do nothing."""
        pass

    def delete(self, key: str) -> bool:
        """Return False."""
        return False

    def clear(self) -> None:
        """Do nothing."""
        pass

    def exists(self, key: str) -> bool:
        """Always return False."""
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Return empty stats."""
        return {
            "size": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0,
            "total_requests": 0,
        }
