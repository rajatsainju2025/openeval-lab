from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    Thread-safe for simple get/set usage.
    """

    def __init__(self, cache_dir: Path, db_name: str = "predictions.sqlite") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / db_name
        # check_same_thread=False to allow multi-threaded access
        self._conn = sqlite3.connect(self.path.as_posix(), check_same_thread=False)
        self._lock = threading.Lock()
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def get(self, key: str, *, ttl: Optional[float] = None) -> Optional[str]:
        now = time.time()
        with self._lock:
            cur = self._conn.execute("SELECT value, created_at FROM kv WHERE key = ?", (key,))
            row = cur.fetchone()
        if not row:
            return None
        value, created_at = row
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
        try:
            # value might be JSON string containing {"output": ...}
            obj = json.loads(value)
            if isinstance(obj, dict) and "output" in obj:
                return str(obj["output"])
        except Exception:
            pass
        return str(value)

    def set(self, key: str, output: str) -> None:
        payload = json.dumps({"output": output})
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv(key, value, created_at) VALUES(?,?,?)",
                (key, payload, time.time()),
            )
