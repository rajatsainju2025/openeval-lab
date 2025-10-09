"""Advanced caching system with LRU eviction and compression."""

from __future__ import annotations

import time
import zlib
import pickle
from typing import Any, Optional
from pathlib import Path
from collections import OrderedDict
import threading


class CacheEntry:
    """A cache entry with metadata for LRU eviction."""

    def __init__(self, key: str, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_access = time.time()

    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def access(self):
        """Mark entry as accessed."""
        self.access_count += 1
        self.last_access = time.time()


class AdvancedCache:
    """Advanced caching system with LRU eviction, compression, and persistence."""

    def __init__(
        self,
        max_size: int = 1000,
        compression_level: int = 6,
        persistence_path: Optional[Path] = None,
        default_ttl: Optional[float] = None,
    ):
        self.max_size = max_size
        self.compression_level = compression_level
        self.persistence_path = persistence_path
        self.default_ttl = default_ttl

        # Thread-safe storage
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Load persisted data if available
        if persistence_path and persistence_path.exists():
            self._load_from_disk()

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get(self, key: str, ttl: Optional[float] = None) -> Optional[Any]:
        """Retrieve a value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_expired:
                    self._cache.pop(key)
                    self.misses += 1
                    return None

                entry.access()
                self._cache.move_to_end(key)  # LRU: move to end
                self.hits += 1
                return self._decompress(entry.value)
            else:
                self.misses += 1
                return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value in cache."""
        with self._lock:
            ttl = ttl or self.default_ttl
            compressed_value = self._compress(value)

            if key in self._cache:
                self._cache[key].value = compressed_value
                self._cache[key].timestamp = time.time()
                self._cache[key].ttl = ttl
                self._cache.move_to_end(key)
            else:
                entry = CacheEntry(key, compressed_value, ttl)
                self._cache[key] = entry
                self._cache.move_to_end(key)

                # Evict if over capacity
                if len(self._cache) > self.max_size:
                    evicted_key, evicted_entry = self._cache.popitem(last=False)
                    self.evictions += 1

            # Persist if configured
            if self.persistence_path:
                self._save_to_disk()

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                if self.persistence_path:
                    self._save_to_disk()
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            if self.persistence_path:
                self._save_to_disk()

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number removed."""
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]
            for key in expired_keys:
                self._cache.pop(key)
            if self.persistence_path and expired_keys:
                self._save_to_disk()
            return len(expired_keys)

    def _compress(self, value: Any) -> bytes:
        """Compress a value using zlib."""
        data = pickle.dumps(value)
        return zlib.compress(data, level=self.compression_level)

    def _decompress(self, compressed: bytes) -> Any:
        """Decompress a value."""
        data = zlib.decompress(compressed)
        return pickle.loads(data)

    def _save_to_disk(self) -> None:
        """Persist cache to disk."""
        if self.persistence_path is None:
            return
        try:
            data = {
                "entries": {
                    key: {
                        "value": entry.value,
                        "timestamp": entry.timestamp,
                        "ttl": entry.ttl,
                        "access_count": entry.access_count,
                        "last_access": entry.last_access,
                    }
                    for key, entry in self._cache.items()
                },
                "stats": {
                    "hits": self.hits,
                    "misses": self.misses,
                    "evictions": self.evictions,
                },
            }
            with open(self.persistence_path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            # Silently fail persistence
            pass

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if self.persistence_path is None or not self.persistence_path.exists():
            return
        try:
            with open(self.persistence_path, "rb") as f:
                data = pickle.load(f)

            for key, entry_data in data.get("entries", {}).items():
                entry = CacheEntry(key=key, value=entry_data["value"], ttl=entry_data.get("ttl"))
                entry.timestamp = entry_data["timestamp"]
                entry.access_count = entry_data.get("access_count", 0)
                entry.last_access = entry_data.get("last_access", entry.timestamp)
                self._cache[key] = entry

            stats = data.get("stats", {})
            self.hits = stats.get("hits", 0)
            self.misses = stats.get("misses", 0)
            self.evictions = stats.get("evictions", 0)

        except Exception:
            # Silently fail loading
            pass


__all__ = ["AdvancedCache"]
