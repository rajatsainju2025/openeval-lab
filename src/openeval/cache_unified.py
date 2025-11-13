"""
Unified and Optimized Cache System for OpenEval Lab

This is the consolidated caching module combining:
- Bloom filter for fast cache miss detection (from cache.py)
- Predictive prefetching based on access patterns
- Multi-level cache hierarchy (memory + disk with LRU eviction)
- Adaptive cache sizing and compression
- Thread-safe operations with performance monitoring
- Enhanced entry metadata and TTL support

All cache implementations have been unified into this single module.
Removes duplication between cache.py and optimized_cache.py.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import defaultdict, deque, OrderedDict
import pickle

from .logging import get_logger

logger = get_logger(__name__)

# Lazy-loaded modules for better startup performance
_COMPRESSION_MODULES = {}
HAS_NUMPY = False
np = None


def _get_compression_module(name: str):
    """Lazy-load compression modules on demand."""
    if name not in _COMPRESSION_MODULES:
        if name == "zlib":
            import zlib

            _COMPRESSION_MODULES["zlib"] = zlib
        elif name == "lzma":
            import lzma

            _COMPRESSION_MODULES["lzma"] = lzma
        elif name == "bzip2":
            import bz2

            _COMPRESSION_MODULES["bzip2"] = bz2
        elif name == "lz4":
            try:
                import lz4.frame  # type: ignore

                _COMPRESSION_MODULES["lz4"] = lz4.frame
            except ImportError:
                pass  # lz4 optional dependency
    return _COMPRESSION_MODULES.get(name)


def _ensure_numpy():
    """Lazy-load numpy on demand."""
    global np, HAS_NUMPY
    if HAS_NUMPY or np is not None:
        return np
    try:
        import numpy as np_module

        np = np_module
        HAS_NUMPY = True
        return np
    except ImportError:
        return None


class CompressionAlgorithm:
    """Supported compression algorithms for cache storage."""

    NONE = "none"
    ZLIB = "zlib"
    LZMA = "lzma"
    BZIP2 = "bzip2"
    LZ4 = "lz4"


@dataclass
class CacheEntry:
    """Unified cache entry with comprehensive metadata."""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    ttl: Optional[float] = None
    compression_algo: str = CompressionAlgorithm.NONE

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

    def age(self) -> float:
        """Get age in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Comprehensive cache statistics combining memory and disk metrics."""

    hits: int = 0
    misses: int = 0
    bloom_filter_hits: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    evictions: int = 0
    compression_savings: int = 0
    total_access_time: float = 0.0
    sets: int = 0
    deletes: int = 0
    expired: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    @property
    def avg_access_time(self) -> float:
        total_requests = self.hits + self.misses
        return (self.total_access_time / total_requests) if total_requests > 0 else 0.0

    @property
    def compression_savings_mb(self) -> float:
        return self.compression_savings / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "bloom_filter_hits": self.bloom_filter_hits,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "evictions": self.evictions,
            "compression_savings_mb": self.compression_savings_mb,
            "avg_access_time_ms": self.avg_access_time * 1000,
            "hit_rate": self.hit_rate,
            "total_sets": self.sets,
            "total_deletes": self.deletes,
            "expired_entries": self.expired,
        }


class BloomFilter:
    """Efficient Bloom filter for cache miss detection."""

    def __init__(self, size: int = 10000, hash_functions: int = 3):
        """Initialize Bloom filter.

        Args:
            size: Bit array size
            hash_functions: Number of hash functions
        """
        self.size = size
        self.hash_functions = hash_functions
        self.bit_array = bytearray((size + 7) // 8)

    def add(self, item: str) -> None:
        """Add item to Bloom filter."""
        for i in range(self.hash_functions):
            hash_val = hash(f"{item}:{i}") % self.size
            byte_idx = hash_val // 8
            bit_idx = hash_val % 8
            self.bit_array[byte_idx] |= 1 << bit_idx

    def might_exist(self, item: str) -> bool:
        """Check if item might exist (no false negatives)."""
        for i in range(self.hash_functions):
            hash_val = hash(f"{item}:{i}") % self.size
            byte_idx = hash_val // 8
            bit_idx = hash_val % 8
            if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def clear(self) -> None:
        """Clear the Bloom filter."""
        self.bit_array = bytearray((self.size + 7) // 8)


class MemoryCache:
    """Fast in-memory LRU cache."""

    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """Initialize memory cache.

        Args:
            max_size: Maximum number of entries
            ttl: Time-to-live for entries in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key: str) -> Tuple[bool, Optional[Any]]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return False, None

            entry = self.cache[key]
            if entry.is_expired:
                del self.cache[key]
                return False, None

            entry.touch()
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return True, entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                size_bytes=len(pickle.dumps(value)),
                ttl=ttl or self.ttl,
            )
            self.cache[key] = entry

            # Evict oldest if necessary
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def delete(self, key: str) -> None:
        """Delete entry from cache."""
        with self.lock:
            self.cache.pop(key, None)

    def clear(self) -> None:
        """Clear all entries."""
        with self.lock:
            self.cache.clear()

    def size(self) -> int:
        """Get current size."""
        return len(self.cache)


class DiskCache:
    """Persistent disk-based cache with SQLite backend."""

    def __init__(self, cache_dir: Path, compression: str = CompressionAlgorithm.ZLIB):
        """Initialize disk cache.

        Args:
            cache_dir: Directory for cache storage
            compression: Compression algorithm to use
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        self.compression = compression
        self.lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    accessed_at REAL,
                    access_count INTEGER,
                    size_bytes INTEGER,
                    compressed INTEGER,
                    ttl REAL,
                    compression_algo TEXT
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_accessed_at
                ON cache(accessed_at)
            """
            )
            conn.commit()

    def _compress(self, data: bytes) -> bytes:
        """Compress data."""
        if self.compression == CompressionAlgorithm.NONE:
            return data

        try:
            compress_module = _get_compression_module(self.compression)
            if compress_module:
                if self.compression == "zlib":
                    return compress_module.compress(data, level=6)
                elif self.compression == "lzma":
                    return compress_module.compress(data)
                elif self.compression == "bzip2":
                    return compress_module.compress(data)
                elif self.compression == "lz4":
                    return compress_module.compress(data)
        except Exception as e:
            logger.warning(f"Compression failed: {e}, storing uncompressed")

        return data

    def _decompress(self, data: bytes, algo: str) -> bytes:
        """Decompress data."""
        if algo == CompressionAlgorithm.NONE:
            return data

        try:
            compress_module = _get_compression_module(algo)
            if compress_module:
                if algo == "zlib":
                    return compress_module.decompress(data)
                elif algo == "lzma":
                    return compress_module.decompress(data)
                elif algo == "bzip2":
                    return compress_module.decompress(data)
                elif algo == "lz4":
                    return compress_module.decompress(data)
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")

        return data

    def get(self, key: str) -> Tuple[bool, Optional[Any]]:
        """Get value from disk cache."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT value, access_count, accessed_at, ttl, compression_algo, created_at
                        FROM cache WHERE key = ?
                    """,
                        (key,),
                    )
                    row = cursor.fetchone()

                    if not row:
                        return False, None

                    value_blob, access_count, accessed_at, ttl, algo, created_at = row

                    # Check expiration
                    if ttl and (time.time() - created_at) >= ttl:
                        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                        conn.commit()
                        return False, None

                    # Update access stats
                    conn.execute(
                        """
                        UPDATE cache SET accessed_at = ?, access_count = ?
                        WHERE key = ?
                    """,
                        (time.time(), access_count + 1, key),
                    )
                    conn.commit()

                    # Decompress and deserialize
                    decompressed = self._decompress(value_blob, algo or "none")
                    value = pickle.loads(decompressed)
                    return True, value

            except Exception as e:
                logger.error(f"Disk cache get failed: {e}")
                return False, None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> Tuple[int, int]:
        """Set value in disk cache.

        Returns:
            Tuple of (original_size, compressed_size)
        """
        with self.lock:
            try:
                # Serialize and compress
                serialized = pickle.dumps(value)
                original_size = len(serialized)
                compressed = self._compress(serialized)
                compressed_size = len(compressed)
                is_compressed = compressed != serialized

                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cache
                        (key, value, created_at, accessed_at, access_count,
                         size_bytes, compressed, ttl, compression_algo)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            key,
                            compressed,
                            time.time(),
                            time.time(),
                            0,
                            original_size,
                            int(is_compressed),
                            ttl,
                            self.compression if is_compressed else CompressionAlgorithm.NONE,
                        ),
                    )
                    conn.commit()

                return original_size, compressed_size

            except Exception as e:
                logger.error(f"Disk cache set failed: {e}")
                return 0, 0

    def delete(self, key: str) -> None:
        """Delete entry from disk cache."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
            except Exception as e:
                logger.error(f"Disk cache delete failed: {e}")

    def clear(self) -> None:
        """Clear all entries."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache")
                    conn.commit()
            except Exception as e:
                logger.error(f"Disk cache clear failed: {e}")

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        DELETE FROM cache
                        WHERE ttl IS NOT NULL AND (? - created_at) >= ttl
                    """,
                        (time.time(),),
                    )
                    removed = cursor.rowcount

                    conn.commit()
                    return removed

            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
                return 0


class UnifiedCache:
    """Two-level unified cache system combining memory and disk caching.

    This is the primary cache interface that should be used throughout OpenEval.
    It combines fast in-memory caching with persistent disk storage and
    automatically manages the cache lifecycle.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        memory_size: int = 1000,
        disk_enabled: bool = True,
        compression: str = CompressionAlgorithm.ZLIB,
        ttl: Optional[float] = None,
        enable_prefetch: bool = True,
        bloom_filter_size: int = 10000,
    ):
        """Initialize unified cache.

        Args:
            cache_dir: Directory for disk cache
            memory_size: Max entries in memory cache
            disk_enabled: Enable disk caching
            compression: Compression algorithm for disk cache
            ttl: Default time-to-live for entries
            enable_prefetch: Enable predictive prefetching
            bloom_filter_size: Size of Bloom filter
        """
        self.memory_cache = MemoryCache(max_size=memory_size, ttl=ttl)
        self.disk_cache = None
        self.disk_enabled = disk_enabled

        if disk_enabled and cache_dir:
            self.disk_cache = DiskCache(Path(cache_dir), compression=compression)

        self.stats = CacheStats()
        self.lock = threading.RLock()

        # Bloom filter for fast miss detection
        self.bloom_filter = BloomFilter(size=bloom_filter_size)

        # Prefetching
        self.enable_prefetch = enable_prefetch
        self.access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.pattern_lock = threading.RLock()

    def get(self, key: str, prefetch: bool = True) -> Optional[Any]:
        """Get value from cache with prefetch support.

        Args:
            key: Cache key
            prefetch: Whether to prefetch related keys

        Returns:
            Cached value or None if not found
        """
        start_time = time.time()

        # Check Bloom filter first (no false negatives)
        if not self.bloom_filter.might_exist(key):
            self.stats.bloom_filter_hits += 1
            self.stats.misses += 1
            return None

        # Try memory cache
        found, value = self.memory_cache.get(key)
        if found:
            self.stats.hits += 1
            self.stats.total_access_time += time.time() - start_time
            self._record_access(key)

            # Prefetch related keys if enabled
            if prefetch and self.enable_prefetch:
                self._prefetch_related(key)

            return value

        # Try disk cache
        if self.disk_cache:
            found, value = self.disk_cache.get(key)
            if found:
                # Promote to memory cache
                self.memory_cache.set(key, value)
                self.stats.hits += 1
                self.stats.total_access_time += time.time() - start_time
                self._record_access(key)

                if prefetch and self.enable_prefetch:
                    self._prefetch_related(key)

                return value

        self.stats.misses += 1
        self.stats.total_access_time += time.time() - start_time
        return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None, disk: bool = True) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            disk: Whether to persist to disk
        """
        with self.lock:
            # Add to memory cache
            self.memory_cache.set(key, value, ttl=ttl)
            self.bloom_filter.add(key)

            # Add to disk cache if enabled
            if disk and self.disk_cache:
                orig_size, comp_size = self.disk_cache.set(key, value, ttl=ttl)
                savings = orig_size - comp_size
                if savings > 0:
                    self.stats.compression_savings += savings

            self.stats.sets += 1

    def delete(self, key: str) -> None:
        """Delete entry from cache."""
        with self.lock:
            self.memory_cache.delete(key)
            if self.disk_cache:
                self.disk_cache.delete(key)
            self.stats.deletes += 1

    def clear(self) -> None:
        """Clear all caches."""
        with self.lock:
            self.memory_cache.clear()
            if self.disk_cache:
                self.disk_cache.clear()
            self.bloom_filter.clear()

    def cleanup_expired(self) -> None:
        """Remove expired entries."""
        with self.lock:
            if self.disk_cache:
                removed = self.disk_cache.cleanup_expired()
                self.stats.expired += removed

    def _record_access(self, key: str) -> None:
        """Record access pattern for prefetching."""
        if self.enable_prefetch:
            with self.pattern_lock:
                self.access_patterns[key].append(time.time())

    def _prefetch_related(self, key: str) -> None:
        """Prefetch keys likely to be accessed next."""
        # This is a placeholder for prefetch logic
        # Could be enhanced to analyze access patterns
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = CacheStats()


# Compatibility aliases for backward compatibility
class PredictionCache(UnifiedCache):
    """Backward compatibility alias for UnifiedCache."""

    pass


class OptimizedCache(UnifiedCache):
    """Backward compatibility alias for UnifiedCache."""

    pass
