"""
Unified Cache System for OpenEval Lab.

High-performance caching with:
- Multi-level cache hierarchy (memory + SQLite)
- Bloom filter for fast miss detection
- Predictive prefetching based on access patterns
- LRU eviction with adaptive sizing
- Compression support (zlib, lzma, bzip2)
- Thread-safe operations
- TTL support and memory limits

This module consolidates:
- cache.py (original - 908 lines)
- cache_unified.py (657 lines)
- cache_ttl.py (51 lines)
- optimized_cache.py (580 lines)
- metrics_cache.py (31 lines)
"""

from __future__ import annotations

import json
import pickle
import re as _re
import sqlite3
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar

__all__ = [
    # Stats and entries
    "CacheStats",
    "CacheEntry",
    # Main cache classes
    "PredictionCache",
    "LRUCache",
    "TTLCache",
    # Helpers
    "BloomFilter",
    "PredictivePrefetcher",
    "AdaptiveCacheSizer",
    "CompressionAlgorithm",
    # Cached functions
    "normalize_text_cached",
    "tokenize_cached",
    "strip_punctuation_cached",
    # Backward compatibility aliases
    "UnifiedCache",
    "AdvancedCache",
    "OptimizedCache",
]

# Lazy-loaded compression modules
_COMPRESSION_MODULES: Dict[str, Any] = {}

T = TypeVar("T")


def _get_compression_module(name: str) -> Any:
    """Lazy-load compression modules on demand."""
    if name not in _COMPRESSION_MODULES:
        try:
            if name == "zlib":
                import zlib

                _COMPRESSION_MODULES["zlib"] = zlib
            elif name == "lzma":
                import lzma

                _COMPRESSION_MODULES["lzma"] = lzma
            elif name == "bzip2":
                import bz2

                _COMPRESSION_MODULES["bzip2"] = bz2
        except ImportError:
            _COMPRESSION_MODULES[name] = None
    return _COMPRESSION_MODULES.get(name)


# =============================================================================
# Compression Algorithms
# =============================================================================


class CompressionAlgorithm:
    """Supported compression algorithms for cache storage."""

    NONE = "none"
    ZLIB = "zlib"
    LZMA = "lzma"
    BZIP2 = "bzip2"


# =============================================================================
# Cache Statistics and Entries
# =============================================================================


@dataclass
class CacheStats:
    """Comprehensive cache statistics.

    Thread-safe counter class using __slots__ for efficiency.
    """

    __slots__ = (
        "hits",
        "misses",
        "bloom_filter_hits",
        "prefetch_hits",
        "prefetch_misses",
        "evictions",
        "compression_savings",
        "total_access_time",
        "sets",
    )

    hits: int
    misses: int
    bloom_filter_hits: int
    prefetch_hits: int
    prefetch_misses: int
    evictions: int
    compression_savings: int
    total_access_time: float
    sets: int

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.bloom_filter_hits = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.evictions = 0
        self.compression_savings = 0
        self.total_access_time = 0.0
        self.sets = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    @property
    def prefetch_hit_rate(self) -> float:
        """Calculate prefetch hit rate."""
        total = self.prefetch_hits + self.prefetch_misses
        return (self.prefetch_hits / total) if total > 0 else 0.0

    @property
    def avg_access_time(self) -> float:
        """Average time per cache access."""
        total = self.hits + self.misses
        return (self.total_access_time / total) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "compression_savings_mb": self.compression_savings / (1024 * 1024),
            "avg_access_time_ms": self.avg_access_time * 1000,
            "prefetch_hit_rate": self.prefetch_hit_rate,
        }


@dataclass
class CacheEntry:
    """Cache entry with metadata for LRU tracking.

    Uses __slots__ for memory efficiency.
    """

    __slots__ = (
        "key",
        "value",
        "created_at",
        "accessed_at",
        "access_count",
        "size_bytes",
        "compressed",
        "ttl",
    )

    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    size_bytes: int
    compressed: bool
    ttl: Optional[float]

    def __init__(
        self,
        key: str,
        value: Any,
        created_at: Optional[float] = None,
        accessed_at: Optional[float] = None,
        access_count: int = 0,
        size_bytes: int = 0,
        compressed: bool = False,
        ttl: Optional[float] = None,
    ) -> None:
        now = time.time()
        self.key = key
        self.value = value
        self.created_at = created_at or now
        self.accessed_at = accessed_at or now
        self.access_count = access_count
        self.size_bytes = size_bytes
        self.compressed = compressed
        self.ttl = ttl

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) >= self.ttl

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


# =============================================================================
# Bloom Filter for Fast Miss Detection
# =============================================================================


class BloomFilter:
    """Memory-efficient Bloom filter for cache miss detection.

    Provides O(1) lookup with no false negatives - if the filter
    says an item doesn't exist, it definitely doesn't.
    """

    __slots__ = ("size", "hash_count", "bit_array", "_lock")

    def __init__(self, size: int = 100000, hash_count: int = 3) -> None:
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bytearray((size + 7) // 8)
        self._lock = threading.Lock()

    def _hash(self, key: str, seed: int) -> int:
        """Generate hash value for key with seed."""
        # Use Python's built-in hash with seed mixing
        return hash(f"{key}:{seed}") % self.size

    def add(self, key: str) -> None:
        """Add a key to the bloom filter."""
        with self._lock:
            for i in range(self.hash_count):
                idx = self._hash(key, i)
                byte_idx, bit_idx = idx // 8, idx % 8
                self.bit_array[byte_idx] |= 1 << bit_idx

    def contains(self, key: str) -> bool:
        """Check if key might be in the set (false positives possible)."""
        with self._lock:
            for i in range(self.hash_count):
                idx = self._hash(key, i)
                byte_idx, bit_idx = idx // 8, idx % 8
                if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                    return False
            return True

    def clear(self) -> None:
        """Clear the bloom filter."""
        with self._lock:
            self.bit_array = bytearray((self.size + 7) // 8)


# =============================================================================
# Predictive Prefetcher
# =============================================================================


class PredictivePrefetcher:
    """Predictive prefetching based on access patterns."""

    __slots__ = ("access_patterns", "reverse_patterns", "max_patterns", "_lock")

    def __init__(self, max_patterns: int = 1000) -> None:
        self.access_patterns: Dict[str, List[str]] = defaultdict(list)
        self.reverse_patterns: Dict[str, List[str]] = defaultdict(list)
        self.max_patterns = max_patterns
        self._lock = threading.Lock()

    def record_access(self, key: str, context_keys: Optional[List[str]] = None) -> None:
        """Record access pattern for predictive prefetching."""
        if not context_keys:
            return

        with self._lock:
            for context_key in context_keys:
                patterns = self.access_patterns[context_key]
                if len(patterns) >= 10:
                    patterns.pop(0)
                patterns.append(key)

                reverse = self.reverse_patterns[key]
                if len(reverse) >= 5:
                    reverse.pop(0)
                reverse.append(context_key)

    def predict_next_accesses(self, current_key: str, max_predictions: int = 3) -> List[str]:
        """Predict next likely accesses based on patterns."""
        with self._lock:
            predictions: List[str] = []

            if current_key in self.reverse_patterns:
                for context in self.reverse_patterns[current_key]:
                    if context in self.access_patterns:
                        pattern = self.access_patterns[context]
                        try:
                            idx = pattern.index(current_key)
                            if idx + 1 < len(pattern):
                                next_key = pattern[idx + 1]
                                if next_key not in predictions:
                                    predictions.append(next_key)
                                    if len(predictions) >= max_predictions:
                                        break
                        except ValueError:
                            continue

            return predictions[:max_predictions]


# =============================================================================
# Adaptive Cache Sizer
# =============================================================================


class AdaptiveCacheSizer:
    """Adaptive cache sizing based on usage patterns."""

    __slots__ = (
        "min_size",
        "max_size",
        "current_size",
        "access_history",
        "_lock",
    )

    def __init__(self, min_size: int = 1000, max_size: int = 100000) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.current_size = min_size
        self.access_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

    def record_access(self, hit: bool) -> None:
        """Record cache access."""
        with self._lock:
            self.access_history.append(hit)

    def get_optimal_size(self) -> int:
        """Calculate optimal cache size based on recent performance."""
        with self._lock:
            if len(self.access_history) < 50:
                return self.current_size

            hit_rate = sum(self.access_history) / len(self.access_history)

            if hit_rate > 0.8:
                new_size = int(self.current_size * 0.9)
            elif hit_rate < 0.3:
                new_size = int(self.current_size * 1.2)
            else:
                new_size = self.current_size

            self.current_size = max(self.min_size, min(self.max_size, new_size))
            return self.current_size


# =============================================================================
# TTL Cache (Simple In-Memory)
# =============================================================================


class TTLCache:
    """Simple in-memory cache with TTL and size limits.

    Thread-safe, uses OrderedDict for LRU ordering.
    """

    __slots__ = ("max_size", "ttl_seconds", "_cache", "_lock")

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()

    def set(self, key: str, value: Any) -> None:
        """Set cache value with TTL."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            self._cache[key] = (value, time.time())
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def get(self, key: str) -> Optional[Any]:
        """Get cache value, checking TTL."""
        with self._lock:
            if key not in self._cache:
                return None

            value, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        with self._lock:
            now = time.time()
            expired = [k for k, (_, ts) in self._cache.items() if now - ts > self.ttl_seconds]
            for k in expired:
                del self._cache[k]
            return len(expired)


# =============================================================================
# LRU Cache (In-Memory with Compression)
# =============================================================================


class LRUCache:
    """In-memory LRU cache with compression support.

    Features:
    - Size and memory limits
    - Optional compression
    - Thread-safe operations
    - Statistics tracking
    """

    __slots__ = (
        "max_size",
        "max_memory_bytes",
        "compression",
        "compression_threshold",
        "_cache",
        "_memory_usage",
        "stats",
        "_lock",
    )

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        compression: str = CompressionAlgorithm.ZLIB,
        compression_threshold: int = 1024,
    ) -> None:
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression = compression
        self.compression_threshold = compression_threshold
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_usage = 0
        self.stats = CacheStats()
        self._lock = threading.RLock()

    def _compress_value(self, value: Any) -> Tuple[bytes, bool]:
        """Compress a value if beneficial."""
        data = pickle.dumps(value)

        if self.compression == CompressionAlgorithm.NONE or len(data) < self.compression_threshold:
            return data, False

        compressor = _get_compression_module(self.compression)
        if not compressor:
            return data, False

        try:
            compressed = compressor.compress(data)
            if len(compressed) < len(data):
                self.stats.compression_savings += len(data) - len(compressed)
                return compressed, True
            return data, False
        except Exception:
            return data, False

    def _decompress_value(self, data: bytes, compressed: bool) -> Any:
        """Decompress a value if needed."""
        if not compressed:
            return pickle.loads(data)

        compressor = _get_compression_module(self.compression)
        if not compressor:
            return pickle.loads(data)

        try:
            return pickle.loads(compressor.decompress(data))
        except Exception:
            return pickle.loads(data)

    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        while (
            len(self._cache) >= self.max_size or self._memory_usage >= self.max_memory_bytes
        ) and self._cache:
            _, entry = self._cache.popitem(last=False)
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
            if entry.is_expired:
                del self._cache[key]
                self._memory_usage -= entry.size_bytes
                self.stats.misses += 1
                self.stats.total_access_time += time.time() - start_time
                return None

            entry.touch()
            self._cache.move_to_end(key)
            self.stats.hits += 1
            self.stats.total_access_time += time.time() - start_time

            return self._decompress_value(entry.value, entry.compressed)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache."""
        with self._lock:
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._memory_usage -= old_entry.size_bytes

            self._evict_lru()

            data, compressed = self._compress_value(value)
            size_bytes = len(data)

            entry = CacheEntry(
                key=key,
                value=data,
                size_bytes=size_bytes,
                compressed=compressed,
                ttl=ttl,
            )

            self._cache[key] = entry
            self._memory_usage += size_bytes
            self.stats.sets += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0


# =============================================================================
# Prediction Cache (SQLite + Memory)
# =============================================================================


class PredictionCache:
    """Advanced SQLite-backed cache with memory layer.

    Features:
    - Bloom filter for fast miss detection
    - Predictive prefetching
    - Adaptive sizing
    - Multi-level caching (memory + disk)
    - Compression
    - Thread-safe operations
    """

    def __init__(
        self,
        cache_dir: Path,
        db_name: str = "predictions.sqlite",
        compress: bool = True,
        memory_cache_size: int = 1000,
        enable_bloom_filter: bool = True,
        enable_prefetching: bool = True,
        enable_adaptive_sizing: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / db_name
        self.compress = compress
        self.memory_cache_size = memory_cache_size

        # Advanced features
        self.bloom_filter = BloomFilter() if enable_bloom_filter else None
        self.prefetcher = PredictivePrefetcher() if enable_prefetching else None
        self.cache_sizer = AdaptiveCacheSizer() if enable_adaptive_sizing else None

        self.cache_stats = CacheStats()

        # SQLite connection
        self._conn = sqlite3.connect(self.path.as_posix(), check_same_thread=False)
        self._lock = threading.RLock()

        # Memory cache
        self._memory_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._access_history: deque = deque(maxlen=100)

        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    compressed INTEGER DEFAULT 0,
                    metadata TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL DEFAULT 0
                )
            """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON kv(created_at)")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute("PRAGMA journal_mode = WAL")

    def _memory_cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get from memory cache."""
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            return self._memory_cache[key]
        return None

    def _memory_cache_set(self, key: str, value: Dict[str, Any]) -> None:
        """Set in memory cache with LRU eviction."""
        if key in self._memory_cache:
            del self._memory_cache[key]
        self._memory_cache[key] = value
        while len(self._memory_cache) > self.memory_cache_size:
            self._memory_cache.popitem(last=False)
            self.cache_stats.evictions += 1

    def get(self, key: str) -> Optional[str]:
        """Get a cached prediction."""
        start_time = time.time()

        # Bloom filter check
        if self.bloom_filter and not self.bloom_filter.contains(key):
            self.cache_stats.bloom_filter_hits += 1
            self.cache_stats.misses += 1
            self.cache_stats.total_access_time += time.time() - start_time
            return None

        # Memory cache check
        mem_entry = self._memory_cache_get(key)
        if mem_entry:
            self.cache_stats.hits += 1
            self.cache_stats.total_access_time += time.time() - start_time
            self._access_history.append(key)
            return mem_entry.get("value")

        # Disk cache check
        with self._lock:
            cur = self._conn.execute("SELECT value, compressed FROM kv WHERE key = ?", (key,))
            row = cur.fetchone()

        if not row:
            self.cache_stats.misses += 1
            self.cache_stats.total_access_time += time.time() - start_time
            if self.cache_sizer:
                self.cache_sizer.record_access(False)
            return None

        payload, compressed = row
        self.cache_stats.hits += 1
        self.cache_stats.total_access_time += time.time() - start_time

        # Decompress if needed
        if compressed:
            zlib_mod = _get_compression_module("zlib")
            if zlib_mod:
                try:
                    payload = zlib_mod.decompress(payload.encode("latin-1")).decode("utf-8")
                except Exception:
                    pass

        data = json.loads(payload)
        output = data.get("output", "")

        # Update memory cache
        self._memory_cache_set(key, {"value": output})
        self._access_history.append(key)

        if self.cache_sizer:
            self.cache_sizer.record_access(True)

        return output

    def set(self, key: str, output: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Cache a prediction."""
        payload = json.dumps({"output": output})
        now = time.time()

        compressed = 0
        if self.compress and len(payload) > 1024:
            zlib_mod = _get_compression_module("zlib")
            if zlib_mod:
                try:
                    compressed_data = zlib_mod.compress(payload.encode("utf-8"))
                    if len(compressed_data) < len(payload):
                        payload = compressed_data.decode("latin-1")
                        compressed = 1
                except Exception:
                    pass

        metadata_str = json.dumps(metadata) if metadata else None

        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO kv
                   (key, value, created_at, compressed, metadata, access_count, last_accessed)
                   VALUES (?, ?, ?, ?, ?, 0, ?)""",
                (key, payload, now, compressed, metadata_str, now),
            )

        self._memory_cache_set(key, {"value": output})

        if self.bloom_filter:
            self.bloom_filter.add(key)

        self.cache_stats.sets += 1

    def get_batch(self, keys: List[str]) -> List[Optional[str]]:
        """Get multiple values from cache."""
        return [self.get(key) for key in keys]

    def set_batch(
        self,
        items: List[Tuple[str, str]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Set multiple values in cache using batch insert."""
        if not items:
            return

        now = time.time()
        batch_data = []

        for idx, (key, output) in enumerate(items):
            payload = json.dumps({"output": output})
            compressed = 0

            if self.compress and len(payload) > 1024:
                zlib_mod = _get_compression_module("zlib")
                if zlib_mod:
                    try:
                        compressed_data = zlib_mod.compress(payload.encode("utf-8"))
                        if len(compressed_data) < len(payload):
                            payload = compressed_data.decode("latin-1")
                            compressed = 1
                    except Exception:
                        pass

            metadata_str = None
            if metadata_list and idx < len(metadata_list):
                metadata_str = json.dumps(metadata_list[idx])

            batch_data.append((key, payload, now, compressed, metadata_str, 0, now))
            self._memory_cache_set(key, {"value": output})
            if self.bloom_filter:
                self.bloom_filter.add(key)

        with self._lock, self._conn:
            self._conn.executemany(
                """INSERT OR REPLACE INTO kv
                   (key, value, created_at, compressed, metadata, access_count, last_accessed)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                batch_data,
            )
            self._conn.commit()

        self.cache_stats.sets += len(items)

    def clear_expired(self, ttl: float) -> int:
        """Clear expired entries. Returns count cleared."""
        cutoff = time.time() - ttl
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM kv WHERE created_at < ?", (cutoff,))
            return cur.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*), SUM(LENGTH(value)) FROM kv")
            count, total_size = cur.fetchone()

        return {
            "entries": count or 0,
            "total_size_bytes": total_size or 0,
            "memory_cache_size": len(self._memory_cache),
            "memory_cache_capacity": self.memory_cache_size,
            "cache_hits": self.cache_stats.hits,
            "cache_misses": self.cache_stats.misses,
            "cache_evictions": self.cache_stats.evictions,
            "hit_rate": self.cache_stats.hit_rate,
            "bloom_filter_enabled": self.bloom_filter is not None,
        }

    def optimize_database(self) -> None:
        """Optimize database performance."""
        with self._lock, self._conn:
            self._conn.execute("VACUUM")
            self._conn.execute("REINDEX")
            self._conn.execute("ANALYZE")


# =============================================================================
# Cached Functions (from metrics_cache.py)
# =============================================================================


@lru_cache(maxsize=10000)
def normalize_text_cached(text: str) -> str:
    """Cache text normalization results."""
    text = text.lower()
    text = _re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=5000)
def tokenize_cached(text: str) -> Tuple[str, ...]:
    """Cache tokenization results (returns tuple for hashability)."""
    return tuple(text.split())


@lru_cache(maxsize=5000)
def strip_punctuation_cached(text: str) -> str:
    """Cache punctuation stripping."""
    return _re.sub(r"[^\w\s]", "", text)


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================


class UnifiedCache(PredictionCache):
    """Backward compatibility alias for PredictionCache."""

    pass


class AdvancedCache(PredictionCache):
    """Backward compatibility alias for PredictionCache."""

    pass


class OptimizedCache(PredictionCache):
    """Backward compatibility alias for PredictionCache."""

    pass
