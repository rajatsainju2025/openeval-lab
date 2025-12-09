"""Redis cache backend for distributed caching.

Extends cache system with distributed Redis support.
"""

from typing import Any, Dict, Optional

from .cache_manager import CacheManager
from .types import ExplanationResult


class RedisCacheManager(CacheManager):
    """Redis-backed cache for distributed scenarios.

    Requires redis package: pip install redis
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: Optional[int] = None,
    ) -> None:
        """Initialize Redis cache manager.

        Args:
            host: Redis server host.
            port: Redis server port.
            db: Redis database number.
            password: Redis password if required.
            default_ttl: Default TTL in seconds.
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self._redis = None
        self._hits = 0
        self._misses = 0

        try:
            import redis

            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
            )
            # Test connection
            self._redis.ping()
        except ImportError:
            raise ImportError(
                "redis package required for RedisCacheManager. " "Install with: pip install redis"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Redis: {e}")

    def get(self, key: str) -> Optional[ExplanationResult]:
        """Retrieve from Redis cache."""
        try:

            value = self._redis.get(key)
            if value:
                self._hits += 1
                # Note: In production, properly deserialize ExplanationResult
                return value  # Simplified
            self._misses += 1
            return None
        except Exception:
            self._misses += 1
            return None

    def set(self, key: str, value: ExplanationResult, ttl: Optional[int] = None) -> None:
        """Store in Redis cache."""
        try:

            ttl = ttl or self.default_ttl
            # Simplified: in production, properly serialize ExplanationResult
            self._redis.setex(key, ttl or 3600, str(value))
        except Exception:
            pass  # Silently fail on cache errors

    def delete(self, key: str) -> bool:
        """Delete from Redis cache."""
        try:
            return bool(self._redis.delete(key))
        except Exception:
            return False

    def clear(self) -> None:
        """Clear all cached entries in this DB."""
        try:
            self._redis.flushdb()
            self._hits = 0
            self._misses = 0
        except Exception:
            pass

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return bool(self._redis.exists(key))
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0

        try:
            info = self._redis.info()
            redis_memory = info.get("used_memory_human", "unknown")
        except Exception:
            redis_memory = "unknown"

        return {
            "size": self._redis.dbsize() if self._redis else 0,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_requests": total,
            "backend": "redis",
            "host": self.host,
            "port": self.port,
            "redis_memory": redis_memory,
        }

    def get_connection_info(self) -> Dict[str, Any]:
        """Get Redis connection info.

        Returns:
            Dictionary with connection details.
        """
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "connected": bool(self._redis),
            "default_ttl": self.default_ttl,
        }


class MemcachedCacheManager(CacheManager):
    """Memcached-backed cache for distributed scenarios.

    Requires pymemcache package: pip install pymemcache
    """

    def __init__(
        self,
        servers: list = None,
        default_ttl: Optional[int] = None,
    ) -> None:
        """Initialize Memcached cache manager.

        Args:
            servers: List of (host, port) tuples.
            default_ttl: Default TTL in seconds.
        """
        self.servers = servers or [("localhost", 11211)]
        self.default_ttl = default_ttl
        self._memcached = None
        self._hits = 0
        self._misses = 0

        try:
            from pymemcache.hash_client import HashClient

            self._memcached = HashClient(self.servers)
        except ImportError:
            raise ImportError(
                "pymemcache package required for MemcachedCacheManager. "
                "Install with: pip install pymemcache"
            )

    def get(self, key: str) -> Optional[ExplanationResult]:
        """Retrieve from Memcached."""
        try:
            value = self._memcached.get(key)
            if value:
                self._hits += 1
                return value
            self._misses += 1
            return None
        except Exception:
            self._misses += 1
            return None

    def set(self, key: str, value: ExplanationResult, ttl: Optional[int] = None) -> None:
        """Store in Memcached."""
        try:
            ttl = ttl or self.default_ttl or 3600
            self._memcached.set(key, str(value), expire=ttl)
        except Exception:
            pass

    def delete(self, key: str) -> bool:
        """Delete from Memcached."""
        try:
            self._memcached.delete(key)
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear all cached entries."""
        try:
            self._memcached.flush_all()
            self._hits = 0
            self._misses = 0
        except Exception:
            pass

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_requests": total,
            "backend": "memcached",
            "servers": self.servers,
        }
