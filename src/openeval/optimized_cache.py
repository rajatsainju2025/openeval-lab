"""
Optimized Cache with LRU and Compression for OpenEval Lab

This module provides an optimized caching system with LRU eviction, advanced compression,
and performance monitoring for improved cache efficiency.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib
import lzma
import bz2
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import OrderedDict as OrderedDictType
import pickle

try:
    import lru_cache

    HAS_LRU_CACHE = True
except ImportError:
    HAS_LRU_CACHE = False

from .logging import get_logger

logger = get_logger(__name__)


class CompressionAlgorithm:
    """Supported compression algorithms."""

    NONE = "none"
    ZLIB = "zlib"
    LZMA = "lzma"
    BZIP2 = "bzip2"
    LZ4 = "lz4"


@dataclass
class CacheEntry:
    """A cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    ttl: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        """Check if the entry is expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) >= self.ttl

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Enhanced cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    sets: int = 0
    compression_savings: int = 0
    total_access_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    @property
    def avg_access_time(self) -> float:
        total_requests = self.hits + self.misses
        return (self.total_access_time / total_requests) if total_requests > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "sets": self.sets,
            "hit_rate": self.hit_rate,
            "compression_savings_mb": self.compression_savings / (1024 * 1024),
            "avg_access_time_ms": self.avg_access_time * 1000,
        }


class LRUCache:
    """
    In-memory LRU cache with compression support.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        compression: str = CompressionAlgorithm.ZLIB,
        compression_threshold: int = 1024,
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression = compression
        self.compression_threshold = compression_threshold
        self._cache: OrderedDictType[str, CacheEntry] = OrderedDictType()
        self._memory_usage = 0
        self.stats = CacheStats()
        self._lock = threading.RLock()

    def _compress_value(self, value: Any) -> Tuple[bytes, bool]:
        """Compress a value if beneficial."""
        if self.compression == CompressionAlgorithm.NONE:
            return pickle.dumps(value), False

        # Serialize first
        data = pickle.dumps(value)

        if len(data) < self.compression_threshold:
            return data, False

        # Try compression
        try:
            if self.compression == CompressionAlgorithm.ZLIB:
                compressed = zlib.compress(data)
            elif self.compression == CompressionAlgorithm.LZMA:
                compressed = lzma.compress(data)
            elif self.compression == CompressionAlgorithm.BZIP2:
                compressed = bz2.compress(data)
            else:
                return data, False

            # Only use compression if it saves space
            if len(compressed) < len(data):
                self.stats.compression_savings += len(data) - len(compressed)
                return compressed, True
            else:
                return data, False

        except Exception:
            return data, False

    def _decompress_value(self, data: bytes, compressed: bool) -> Any:
        """Decompress a value if needed."""
        if not compressed:
            return pickle.loads(data)

        try:
            if self.compression == CompressionAlgorithm.ZLIB:
                decompressed = zlib.decompress(data)
            elif self.compression == CompressionAlgorithm.LZMA:
                decompressed = lzma.decompress(data)
            elif self.compression == CompressionAlgorithm.BZIP2:
                decompressed = bz2.decompress(data)
            else:
                return pickle.loads(data)

            return pickle.loads(decompressed)

        except Exception:
            logger.warning("Failed to decompress cached value, returning raw data")
            return pickle.loads(data)

    def _evict_lru(self) -> None:
        """Evict least recently used items to free memory."""
        while (
            len(self._cache) >= self.max_size or self._memory_usage >= self.max_memory_bytes
        ) and self._cache:
            _, entry = self._cache.popitem(last=False)  # Remove oldest
            self._memory_usage -= entry.size_bytes
            self.stats.evictions += 1

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        start_time = time.time()

        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                self.stats.total_access_time += time.time() - start_time
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                self._memory_usage -= entry.size_bytes
                self.stats.misses += 1
                self.stats.total_access_time += time.time() - start_time
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()

            # Decompress and return
            try:
                value = self._decompress_value(entry.value, entry.compressed)
                self.stats.hits += 1
                self.stats.total_access_time += time.time() - start_time
                return value
            except Exception as e:
                logger.warning(f"Failed to retrieve cached value for key {key}: {e}")
                del self._cache[key]
                self._memory_usage -= entry.size_bytes
                self.stats.misses += 1
                self.stats.total_access_time += time.time() - start_time
                return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache."""
        with self._lock:
            # Compress the value
            compressed_data, is_compressed = self._compress_value(value)
            size_bytes = len(compressed_data)

            # Check if we need to evict
            if key not in self._cache:
                self._evict_lru()

            # Remove old entry if it exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._memory_usage -= old_entry.size_bytes

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=compressed_data,
                created_at=time.time(),
                accessed_at=time.time(),
                size_bytes=size_bytes,
                compressed=is_compressed,
                ttl=ttl,
            )

            self._cache[key] = entry
            self._cache.move_to_end(key)  # Mark as most recently used
            self._memory_usage += size_bytes
            self.stats.sets += 1

    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._memory_usage -= entry.size_bytes
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                **self.stats.to_dict(),
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_mb": self._memory_usage / (1024 * 1024),
                "max_memory_mb": self._memory_usage / (1024 * 1024),
                "compression_enabled": self.compression != CompressionAlgorithm.NONE,
            }


class OptimizedPredictionCache:
    """
    Optimized prediction cache with LRU memory cache and persistent storage.
    """

    def __init__(
        self,
        cache_dir: Path,
        db_name: str = "predictions_optimized.sqlite",
        memory_cache_size: int = 5000,
        memory_cache_mb: int = 200,
        compression: str = CompressionAlgorithm.ZLIB,
        sync_interval: int = 100,  # Sync to disk every N operations
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / db_name
        self.sync_interval = sync_interval

        # Initialize caches
        self.memory_cache = LRUCache(
            max_size=memory_cache_size, max_memory_mb=memory_cache_mb, compression=compression
        )

        # Database connection
        self._conn = sqlite3.connect(self.db_path.as_posix(), check_same_thread=False)
        self._lock = threading.RLock()
        self._operation_count = 0

        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER NOT NULL,
                    compressed INTEGER DEFAULT 0,
                    ttl REAL
                )
            """
            )

            # Create indexes for better performance
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache(accessed_at)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)")

    def close(self) -> None:
        """Close the cache and sync to disk."""
        try:
            self._sync_to_disk()
            self._conn.close()
        except Exception:
            pass

    def _sync_to_disk(self) -> None:
        """Sync memory cache changes to disk."""
        # This is a simplified version - in practice, you'd track dirty entries
        pass

    def _should_sync(self) -> bool:
        """Check if we should sync to disk."""
        self._operation_count += 1
        if self._operation_count >= self.sync_interval:
            self._operation_count = 0
            return True
        return False

    def get(self, key: str, ttl: Optional[float] = None) -> Optional[str]:
        """Get a value from the cache."""
        # Try memory cache first
        result = self.memory_cache.get(key)
        if result is not None:
            return result

        # Try persistent storage
        with self._lock:
            cur = self._conn.execute(
                "SELECT value, created_at, compressed FROM cache WHERE key = ?", (key,)
            )
            row = cur.fetchone()

        if not row:
            return None

        value_blob, created_at, compressed = row

        # Check TTL
        if ttl is not None and (time.time() - created_at) >= ttl:
            # Remove expired entry
            with self._lock:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            return None

        # Deserialize
        try:
            if compressed:
                value_blob = zlib.decompress(value_blob)

            obj = json.loads(value_blob.decode("utf-8"))
            if isinstance(obj, dict) and "output" in obj:
                result = str(obj["output"])
            else:
                result = str(obj)

            # Add to memory cache
            self.memory_cache.set(key, result, ttl)

            # Update access statistics
            with self._lock:
                self._conn.execute(
                    "UPDATE cache SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?",
                    (time.time(), key),
                )

            return result

        except Exception as e:
            logger.debug(f"Failed to deserialize cached value for key {key}: {e}")
            return None

    def set(self, key: str, output: str, ttl: Optional[float] = None) -> None:
        """Set a value in the cache."""
        # Add to memory cache
        self.memory_cache.set(key, output, ttl)

        # Prepare data for persistent storage
        payload = json.dumps({"output": output}).encode("utf-8")
        compressed = 0

        # Compress if beneficial
        if len(payload) > 1024:  # Compress payloads > 1KB
            try:
                compressed_payload = zlib.compress(payload)
                if len(compressed_payload) < len(payload):
                    payload = compressed_payload
                    compressed = 1
            except Exception:
                pass

        # Store in database
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO cache
                (key, value, created_at, accessed_at, size_bytes, compressed, ttl)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (key, payload, time.time(), time.time(), len(payload), compressed, ttl),
            )

        if self._should_sync():
            self._sync_to_disk()

    def clear_expired(self, ttl: float) -> int:
        """Clear expired entries and return number cleared."""
        cutoff = time.time() - ttl
        with self._lock:
            cur = self._conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
            deleted_count = cur.rowcount

        # Also clear from memory cache
        # Note: This is a simplified implementation
        self.memory_cache.clear()

        return deleted_count

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()

        with self._lock:
            cur = self._conn.execute(
                """
                SELECT COUNT(*), SUM(size_bytes), AVG(created_at), AVG(accessed_at)
                FROM cache
            """
            )
            count, total_size, avg_created, avg_accessed = cur.fetchone()

        return {
            "memory_cache": memory_stats,
            "persistent_cache": {
                "entries": count or 0,
                "total_size_bytes": total_size or 0,
                "average_age_seconds": time.time() - (avg_created or time.time()),
                "average_last_access_seconds": time.time() - (avg_accessed or time.time()),
            },
        }

    def optimize(self) -> None:
        """Optimize the cache for better performance."""
        with self._lock:
            # Rebuild indexes
            self._conn.execute("REINDEX")

            # Vacuum database to reclaim space
            self._conn.execute("VACUUM")

            logger.info("Cache optimization completed")


class CacheManager:
    """
    High-level cache manager that coordinates multiple cache instances.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.caches: Dict[str, OptimizedPredictionCache] = {}
        self._lock = threading.RLock()

    def get_cache(self, name: str, **kwargs) -> OptimizedPredictionCache:
        """Get or create a named cache."""
        with self._lock:
            if name not in self.caches:
                cache_path = self.cache_dir / name
                self.caches[name] = OptimizedPredictionCache(cache_path, **kwargs)
            return self.caches[name]

    def close_all(self) -> None:
        """Close all caches."""
        with self._lock:
            for cache in self.caches.values():
                try:
                    cache.close()
                except Exception:
                    pass
            self.caches.clear()

    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        with self._lock:
            stats = {}
            for name, cache in self.caches.items():
                try:
                    stats[name] = cache.get_stats()
                except Exception as e:
                    stats[name] = {"error": str(e)}
            return stats


# Utility functions
def create_optimized_cache(
    cache_dir: Union[str, Path], name: str = "default", **kwargs
) -> OptimizedPredictionCache:
    """Create an optimized cache instance."""
    manager = CacheManager(Path(cache_dir))
    return manager.get_cache(name, **kwargs)


def benchmark_cache_performance(
    cache: OptimizedPredictionCache,
    test_keys: List[str],
    test_values: List[str],
    iterations: int = 1000,
) -> Dict[str, Any]:
    """Benchmark cache performance."""
    import time

    # Warm up
    for key, value in zip(test_keys[:100], test_values[:100]):
        cache.set(key, value)

    # Benchmark sets
    start_time = time.time()
    for _ in range(iterations):
        key, value = test_keys[_ % len(test_keys)], test_values[_ % len(test_values)]
        cache.set(key, value)
    set_time = time.time() - start_time

    # Benchmark gets
    start_time = time.time()
    hits = 0
    for _ in range(iterations):
        key = test_keys[_ % len(test_keys)]
        result = cache.get(key)
        if result is not None:
            hits += 1
    get_time = time.time() - start_time

    return {
        "iterations": iterations,
        "set_time_seconds": set_time,
        "get_time_seconds": get_time,
        "sets_per_second": iterations / set_time,
        "gets_per_second": iterations / get_time,
        "hit_rate": hits / iterations,
        "stats": cache.get_stats(),
    }
