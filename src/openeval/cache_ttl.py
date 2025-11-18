"""Cache TTL and LRU limit enforcement.

Adds size limits and TTL support to caches.
"""

import time
from typing import Any, Dict, Optional
from collections import OrderedDict


class TTLCache:
    """Cache with TTL and size limits."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize cache with max size and TTL."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple] = OrderedDict()

    def set(self, key: str, value: Any):
        """Set cache value with TTL."""
        self.cache[key] = (value, time.time())
        if len(self.cache) > self.max_size:
            # Remove oldest item
            self.cache.popitem(last=False)

    def get(self, key: str) -> Optional[Any]:
        """Get cache value, checking TTL."""
        if key not in self.cache:
            return None

        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[key]
            return None

        return value

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()

    def cleanup_expired(self):
        """Remove expired entries."""
        now = time.time()
        expired_keys = [k for k, (_, ts) in self.cache.items() if now - ts > self.ttl_seconds]
        for k in expired_keys:
            del self.cache[k]


__all__ = ["TTLCache"]
