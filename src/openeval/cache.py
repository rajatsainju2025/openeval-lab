from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


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

    def __init__(self, cache_dir: Path, db_name: str = "predictions.sqlite", compress: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / db_name
        self.compress = compress
        # check_same_thread=False to allow multi-threaded access
        self._conn = sqlite3.connect(self.path.as_posix(), check_same_thread=False)
        self._lock = threading.Lock()
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    compressed INTEGER DEFAULT 0,
                    metadata TEXT
                )
                """
            )
            # Add metadata column if it doesn't exist (for backward compatibility)
            try:
                self._conn.execute("ALTER TABLE kv ADD COLUMN metadata TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def get(self, key: str, *, ttl: Optional[float] = None) -> Optional[str]:
        now = time.time()
        with self._lock:
            cur = self._conn.execute("SELECT value, created_at, compressed, metadata FROM kv WHERE key = ?", (key,))
            row = cur.fetchone()
        if not row:
            return None
        value, created_at, compressed, metadata = row
        # Expiry logic: ttl=None -> no expiry; ttl<=0 -> always expired; else compare age >= ttl
        if ttl is not None:
            try:
                ttl_f = float(ttl)
            except Exception:
                ttl_f = None  # treat as no expiry if invalid
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
                return str(obj["output"])
        except Exception:
            pass
        return str(value)

    def set(self, key: str, output: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = json.dumps({"output": output})
        
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
                "INSERT OR REPLACE INTO kv(key, value, created_at, compressed, metadata) VALUES(?,?,?,?,?)",
                (key, payload, time.time(), compressed, metadata_str),
            )

    def clear_expired(self, ttl: float) -> int:
        """Clear expired entries and return number cleared."""
        cutoff = time.time() - ttl
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM kv WHERE created_at < ?", (cutoff,))
            return cur.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*), SUM(LENGTH(value)), AVG(created_at) FROM kv")
            count, total_size, avg_age = cur.fetchone()
        
        return {
            "entries": count or 0,
            "total_size_bytes": total_size or 0,
            "average_age_seconds": time.time() - (avg_age or time.time())
        }
