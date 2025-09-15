from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache
import hashlib


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


class PredictionCache:
    """A lightweight SQLite-backed cache for adapter predictions.

    Keys and values are strings. Values are stored as JSON to allow future expansion.
    Thread-safe for simple get/set usage. Supports compression and metadata.
    """

    def __init__(self, cache_dir: Path, db_name: str = "predictions.sqlite", compress: bool = True, memory_cache_size: int = 1000) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / db_name
        self.compress = compress
        self.memory_cache_size = memory_cache_size
        
        # check_same_thread=False to allow multi-threaded access
        self._conn = sqlite3.connect(self.path.as_posix(), check_same_thread=False)
        self._lock = threading.Lock()
        
        # In-memory LRU cache for frequently accessed items
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_cache_order: list = []
        
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
            # Add new columns if they don't exist (for backward compatibility)
            for column in ["access_count", "last_accessed"]:
                try:
                    self._conn.execute(f"ALTER TABLE kv ADD COLUMN {column} REAL DEFAULT 0")
                except sqlite3.OperationalError:
                    pass  # Column already exists
            
            # Create indexes for better performance
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON kv(created_at)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_access_count ON kv(access_count)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON kv(last_accessed)")
            
            # Optimize database settings
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA cache_size = 10000")  # 10MB cache

    def _get_memory_cache_key(self, key: str) -> str:
        """Generate a memory cache key."""
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _memory_cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get item from memory cache."""
        cache_key = self._get_memory_cache_key(key)
        if cache_key in self._memory_cache:
            # Update access tracking
            entry = self._memory_cache[cache_key]
            entry["last_accessed"] = time.time()
            entry["access_count"] += 1
            return entry
        return None

    def _memory_cache_set(self, key: str, data: Dict[str, Any]) -> None:
        """Set item in memory cache with LRU eviction."""
        cache_key = self._get_memory_cache_key(key)
        
        # Evict if at capacity
        if len(self._memory_cache) >= self.memory_cache_size and cache_key not in self._memory_cache:
            # Find least recently used
            lru_key = min(self._memory_cache.keys(), 
                         key=lambda k: self._memory_cache[k]["last_accessed"])
            del self._memory_cache[lru_key]
        
        self._memory_cache[cache_key] = {
            **data,
            "last_accessed": time.time(),
            "access_count": data.get("access_count", 0) + 1
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def get(self, key: str, *, ttl: Optional[float] = None) -> Optional[str]:
        now = time.time()
        
        # Check memory cache first
        mem_entry = self._memory_cache_get(key)
        if mem_entry:
            # Check TTL for memory cache entry
            if ttl is not None:
                try:
                    ttl_f = float(ttl)
                except Exception:
                    ttl_f = None
                if ttl_f is not None and ttl_f > 0:
                    if (now - mem_entry["created_at"]) >= ttl_f:
                        # Expired, remove from memory cache
                        cache_key = self._get_memory_cache_key(key)
                        self._memory_cache.pop(cache_key, None)
                    else:
                        return mem_entry["value"]
            else:
                return mem_entry["value"]
        
        # Check database
        with self._lock:
            cur = self._conn.execute(
                "SELECT value, created_at, compressed, metadata, access_count FROM kv WHERE key = ?", 
                (key,)
            )
            row = cur.fetchone()
        
        if not row:
            return None
        
        value, created_at, compressed, metadata, access_count = row
        
        # Expiry logic
        if ttl is not None:
            try:
                ttl_f = float(ttl)
            except Exception:
                ttl_f = None
            if ttl_f is not None:
                if ttl_f <= 0:
                    return None
                if (now - float(created_at)) >= ttl_f:
                    return None
        
        # Decompress if needed
        if compressed:
            try:
                value = zlib.decompress(value.encode('latin-1')).decode('utf-8')
            except Exception:
                return None  # Corrupted compressed data
        
        try:
            # value might be JSON string containing {"output": ...}
            obj = json.loads(value)
            if isinstance(obj, dict) and "output" in obj:
                final_value = str(obj["output"])
            else:
                final_value = str(value)
        except Exception:
            final_value = str(value)
        
        # Update access statistics in database
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE kv SET access_count = access_count + 1, last_accessed = ? WHERE key = ?",
                (now, key)
            )
        
        # Cache in memory for future use
        self._memory_cache_set(key, {
            "value": final_value,
            "created_at": created_at,
            "compressed": compressed,
            "metadata": metadata,
            "access_count": access_count + 1
        })
        
        return final_value

    def set(self, key: str, output: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = json.dumps({"output": output})
        now = time.time()
        
        # Compress if enabled and payload is large enough
        compressed = 0
        if self.compress and len(payload) > 1024:  # Compress payloads > 1KB
            try:
                compressed_payload = zlib.compress(payload.encode('utf-8'))
                if len(compressed_payload) < len(payload):  # Only use if smaller
                    payload = compressed_payload.decode('latin-1')
                    compressed = 1
            except Exception:
                pass  # Fall back to uncompressed
        
        metadata_str = json.dumps(metadata) if metadata else None
        
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv(key, value, created_at, compressed, metadata, access_count, last_accessed) VALUES(?,?,?,?,?,?,?)",
                (key, payload, now, compressed, metadata_str, 0, now),
            )
        
        # Update memory cache
        self._memory_cache_set(key, {
            "value": output,
            "created_at": now,
            "compressed": compressed,
            "metadata": metadata,
            "access_count": 0
        })

    def clear_expired(self, ttl: float) -> int:
        """Clear expired entries and return number cleared."""
        cutoff = time.time() - ttl
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM kv WHERE created_at < ?", (cutoff,))
            return cur.rowcount

    def optimize_database(self) -> None:
        """Optimize database performance."""
        with self._lock, self._conn:
            self._conn.execute("VACUUM")
            self._conn.execute("REINDEX")
            self._conn.execute("ANALYZE")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            cur = self._conn.execute("""
                SELECT COUNT(*), SUM(LENGTH(value)), AVG(created_at), 
                       AVG(access_count), MAX(last_accessed)
                FROM kv
            """)
            count, total_size, avg_age, avg_access, max_access = cur.fetchone()
        
        now = time.time()
        return {
            "entries": count or 0,
            "total_size_bytes": total_size or 0,
            "average_age_seconds": (now - (avg_age or now)) if avg_age else 0,
            "average_access_count": avg_access or 0,
            "memory_cache_size": len(self._memory_cache),
            "memory_cache_capacity": self.memory_cache_size,
            "last_access_age_seconds": (now - (max_access or now)) if max_access else 0
        }
