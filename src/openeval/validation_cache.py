"""
Validation result caching for repeated spec validation.

Caches validation results keyed by spec hash to avoid re-validation
of identical specs across multiple runs.
"""

import hashlib
import json
import threading
from typing import Any, Dict, Optional, Tuple
import time


class ValidationCache:
    """Thread-safe cache for spec validation results."""

    def __init__(self, ttl: float = 3600.0, max_entries: int = 1000):
        """Initialize validation cache.

        Args:
            ttl: Time-to-live for cache entries in seconds (default: 1 hour)
            max_entries: Maximum number of cached entries
        """
        self.ttl = ttl
        self.max_entries = max_entries
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_spec(spec: Dict[str, Any]) -> str:
        """Compute hash of spec for cache key.

        Args:
            spec: Spec dictionary

        Returns:
            SHA256 hash hex string
        """
        # Sort keys for deterministic hashing
        spec_json = json.dumps(spec, sort_keys=True, default=str)
        return hashlib.sha256(spec_json.encode()).hexdigest()

    def get(self, spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached validation result if available.

        Args:
            spec: Spec dictionary

        Returns:
            Cached validation result or None if not found/expired
        """
        spec_hash = self._hash_spec(spec)

        with self._lock:
            if spec_hash in self._cache:
                result, timestamp = self._cache[spec_hash]
                if time.time() - timestamp < self.ttl:
                    self._hits += 1
                    return result
                else:
                    # Remove expired entry
                    del self._cache[spec_hash]

            self._misses += 1
            return None

    def put(self, spec: Dict[str, Any], validation_result: Dict[str, Any]) -> None:
        """Cache validation result.

        Args:
            spec: Spec dictionary
            validation_result: Validation result to cache
        """
        spec_hash = self._hash_spec(spec)

        with self._lock:
            # Evict oldest entry if cache is full
            if len(self._cache) >= self.max_entries:
                # Find and remove oldest entry
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[spec_hash] = (validation_result, time.time())

    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hit rate and entry count
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "entries": len(self._cache),
                "max_entries": self.max_entries,
            }


# Global validation cache instance
_validation_cache = ValidationCache()


def cache_validation_result(spec: Dict[str, Any], validation_result: Dict[str, Any]) -> None:
    """Cache a spec validation result.

    Args:
        spec: Spec dictionary
        validation_result: Validation result to cache
    """
    _validation_cache.put(spec, validation_result)


def get_cached_validation(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get cached validation result if available.

    Args:
        spec: Spec dictionary

    Returns:
        Cached validation result or None
    """
    return _validation_cache.get(spec)


def clear_validation_cache() -> None:
    """Clear validation cache."""
    _validation_cache.clear()


def get_validation_cache_stats() -> Dict[str, Any]:
    """Get validation cache statistics.

    Returns:
        Dictionary with cache stats
    """
    return _validation_cache.stats()
