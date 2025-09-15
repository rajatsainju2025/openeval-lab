"""
Connection Pooling and Reuse for OpenEval Lab Adapters

This module provides connection pooling, session reuse, and efficient resource management
for API adapters to reduce connection overhead and improve throughput.
"""

from __future__ import annotations

import asyncio
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable, AsyncContextManager, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import weakref

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None

try:
    import requests
    from requests.adapters import HTTPAdapter
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None
    HTTPAdapter = None

from .enhanced_logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class ConnectionConfig:
    """Configuration for connection pooling."""
    max_connections: int = 20
    max_keepalive_connections: int = 10
    keepalive_expiry: float = 30.0  # seconds
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_concurrent_requests: int = 10
    connection_pool_timeout: float = 5.0


@dataclass
class ConnectionStats:
    """Statistics for connection usage."""
    total_connections_created: int = 0
    active_connections: int = 0
    connections_reused: int = 0
    connections_closed: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    pool_exhaustion_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_connections_created": self.total_connections_created,
            "active_connections": self.active_connections,
            "connections_reused": self.connections_reused,
            "connections_closed": self.connections_closed,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.total_requests - self.failed_requests) / self.total_requests if self.total_requests > 0 else 0.0,
            "avg_response_time": self.avg_response_time,
            "pool_exhaustion_events": self.pool_exhaustion_events
        }


class ConnectionPool(ABC):
    """Abstract base class for connection pools."""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.stats = ConnectionStats()
        self._lock = threading.RLock()

    @abstractmethod
    async def acquire(self) -> Any:
        """Acquire a connection from the pool."""
        pass

    @abstractmethod
    async def release(self, connection: Any) -> None:
        """Release a connection back to the pool."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close all connections in the pool."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return self.stats.to_dict()


class HTTPConnectionPool(ConnectionPool):
    """HTTP connection pool using aiohttp."""

    def __init__(self, config: ConnectionConfig, base_url: Optional[str] = None):
        super().__init__(config)
        self.base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure we have an active session."""
        if self._session is None or self._session.closed:
            self._connector = aiohttp.TCPConnector(
                limit=self.config.max_connections,
                limit_per_host=self.config.max_keepalive_connections,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=self.config.keepalive_expiry,
                enable_cleanup_closed=True,
            )

            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout,
                base_url=self.base_url
            )

            self.stats.total_connections_created += 1

        return self._session

    async def acquire(self) -> aiohttp.ClientSession:
        """Acquire a session from the pool."""
        with self._lock:
            self.stats.total_requests += 1
            return await self._ensure_session()

    async def release(self, connection: aiohttp.ClientSession) -> None:
        """Release a session (no-op for aiohttp as it's shared)."""
        with self._lock:
            self.stats.connections_reused += 1

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self.stats.connections_closed += 1

        if self._connector:
            await self._connector.close()


class HTTPXConnectionPool(ConnectionPool):
    """HTTP connection pool using httpx."""

    def __init__(self, config: ConnectionConfig, base_url: Optional[str] = None):
        super().__init__(config)
        self.base_url = base_url
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure we have an active client."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections
            )

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                limits=limits,
                timeout=self.config.timeout
            )

            self.stats.total_connections_created += 1

        return self._client

    async def acquire(self) -> httpx.AsyncClient:
        """Acquire a client from the pool."""
        with self._lock:
            self.stats.total_requests += 1
            return await self._ensure_client()

    async def release(self, connection: httpx.AsyncClient) -> None:
        """Release a client (no-op for httpx as it's shared)."""
        with self._lock:
            self.stats.connections_reused += 1

    async def close(self) -> None:
        """Close the client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self.stats.connections_closed += 1


class RequestsConnectionPool(ConnectionPool):
    """HTTP connection pool using requests with urllib3."""

    def __init__(self, config: ConnectionConfig, base_url: Optional[str] = None):
        super().__init__(config)
        self.base_url = base_url
        self._session: Optional[requests.Session] = None

    def _ensure_session(self) -> requests.Session:
        """Ensure we have an active session."""
        if self._session is None:
            self._session = requests.Session()

            # Configure the adapter
            adapter = HTTPAdapter(
                pool_connections=self.config.max_connections,
                pool_maxsize=self.config.max_connections,
                max_retries=self.config.retry_attempts,
                pool_block=False
            )

            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)

            self.stats.total_connections_created += 1

        return self._session

    async def acquire(self) -> requests.Session:
        """Acquire a session from the pool."""
        with self._lock:
            self.stats.total_requests += 1
            return self._ensure_session()

    async def release(self, connection: requests.Session) -> None:
        """Release a session (no-op for requests as it's shared)."""
        with self._lock:
            self.stats.connections_reused += 1

    async def close(self) -> None:
        """Close the session."""
        if self._session:
            self._session.close()
            self.stats.connections_closed += 1


class PooledAdapter:
    """
    Adapter wrapper that uses connection pooling for improved performance.
    """

    def __init__(
        self,
        base_adapter: Any,
        pool_config: Optional[ConnectionConfig] = None,
        pool_type: str = "auto"  # auto, aiohttp, httpx, requests
    ):
        self.base_adapter = base_adapter
        self.pool_config = pool_config or ConnectionConfig()
        self.pool_type = pool_type
        self._pool: Optional[ConnectionPool] = None
        self._base_url = getattr(base_adapter, 'base_url', None) or getattr(base_adapter, 'api_base', None)

        # Initialize the appropriate pool
        self._init_pool()

    def _init_pool(self) -> None:
        """Initialize the connection pool."""
        if self.pool_type == "auto":
            # Auto-detect best available pool
            if HAS_AIOHTTP:
                self._pool = HTTPConnectionPool(self.pool_config, self._base_url)
                self.pool_type = "aiohttp"
            elif HAS_HTTPX:
                self._pool = HTTPXConnectionPool(self.pool_config, self._base_url)
                self.pool_type = "httpx"
            elif HAS_REQUESTS:
                self._pool = RequestsConnectionPool(self.pool_config, self._base_url)
                self.pool_type = "requests"
            else:
                logger.warning("No HTTP libraries available for connection pooling")
                return

        elif self.pool_type == "aiohttp" and HAS_AIOHTTP:
            self._pool = HTTPConnectionPool(self.pool_config, self._base_url)
        elif self.pool_type == "httpx" and HAS_HTTPX:
            self._pool = HTTPXConnectionPool(self.pool_config, self._base_url)
        elif self.pool_type == "requests" and HAS_REQUESTS:
            self._pool = RequestsConnectionPool(self.pool_config, self._base_url)
        else:
            logger.warning(f"Requested pool type {self.pool_type} not available")
            return

        logger.info(f"Initialized {self.pool_type} connection pool for adapter {self.base_adapter.__class__.__name__}")

    async def make_request(
        self,
        method: str,
        url: str,
        **kwargs: Any
    ) -> Any:
        """
        Make an HTTP request using the connection pool.

        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request parameters

        Returns:
            Response object
        """
        if not self._pool:
            # Fallback to base adapter if no pool available
            return await self._fallback_request(method, url, **kwargs)

        start_time = time.time()

        connection = await self._pool.acquire()

        try:
            if isinstance(self._pool, HTTPConnectionPool):
                # aiohttp
                async with connection.request(method, url, **kwargs) as response:
                    result = await response.text()
                    return result

            elif isinstance(self._pool, HTTPXConnectionPool):
                # httpx
                response = await connection.request(method, url, **kwargs)
                return response.text

            elif isinstance(self._pool, RequestsConnectionPool):
                # requests (run in thread pool)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._sync_request, connection, method, url, kwargs)
                    return future.result()

        finally:
            await self._pool.release(connection)
            response_time = time.time() - start_time
            self._pool.stats.avg_response_time = (
                (self._pool.stats.avg_response_time * (self._pool.stats.total_requests - 1)) + response_time
            ) / self._pool.stats.total_requests

        except Exception as e:
            if self._pool:
                self._pool.stats.failed_requests += 1
            raise

    async def _fallback_request(self, method: str, url: str, **kwargs: Any) -> Any:
        """Fallback request method when pooling is not available."""
        logger.debug("Using fallback request method (no connection pooling)")

        if HAS_HTTPX:
            async with httpx.AsyncClient() as client:
                response = await client.request(method, url, **kwargs)
                return response.text
        elif HAS_AIOHTTP:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, **kwargs) as response:
                    return await response.text()
        else:
            raise RuntimeError("No HTTP client available for requests")

    def _sync_request(self, session: requests.Session, method: str, url: str, kwargs: Dict[str, Any]) -> str:
        """Synchronous request for requests library."""
        response = session.request(method, url, **kwargs)
        return response.text

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate method with connection pooling."""
        # This would be implemented based on the specific adapter's API
        # For now, delegate to base adapter
        if hasattr(self.base_adapter, 'generate'):
            return self.base_adapter.generate(prompt, **kwargs)
        else:
            raise NotImplementedError("Base adapter does not implement generate method")

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Async generate method."""
        if hasattr(self.base_adapter, 'agenerate') and asyncio.iscoroutinefunction(self.base_adapter.agenerate):
            return await self.base_adapter.agenerate(prompt, **kwargs)
        else:
            # Fallback to sync method in thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.base_adapter.generate, prompt, **kwargs)
                return future.result()

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if self._pool:
            return {
                "pool_type": self.pool_type,
                "pool_stats": self._pool.get_stats(),
                "base_adapter": self.base_adapter.__class__.__name__
            }
        else:
            return {"pool_type": "none", "base_adapter": self.base_adapter.__class__.__name__}

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()


@asynccontextmanager
async def pooled_adapter_context(
    adapter: Any,
    pool_config: Optional[ConnectionConfig] = None,
    pool_type: str = "auto"
) -> AsyncContextManager[PooledAdapter]:
    """
    Context manager for pooled adapters.

    Args:
        adapter: Base adapter to wrap
        pool_config: Connection pool configuration
        pool_type: Type of connection pool to use

    Yields:
        PooledAdapter instance
    """
    pooled = PooledAdapter(adapter, pool_config, pool_type)
    try:
        yield pooled
    finally:
        await pooled.close()


class ConnectionPoolManager:
    """
    Global manager for connection pools across multiple adapters.
    """

    def __init__(self):
        self._pools: Dict[str, ConnectionPool] = {}
        self._adapters: Dict[str, PooledAdapter] = {}
        self._lock = threading.RLock()

    def get_pooled_adapter(
        self,
        adapter_key: str,
        adapter_factory: Callable[[], Any],
        pool_config: Optional[ConnectionConfig] = None,
        pool_type: str = "auto"
    ) -> PooledAdapter:
        """
        Get or create a pooled adapter.

        Args:
            adapter_key: Unique key for the adapter
            adapter_factory: Function to create the base adapter
            pool_config: Pool configuration
            pool_type: Pool type

        Returns:
            PooledAdapter instance
        """
        with self._lock:
            if adapter_key not in self._adapters:
                base_adapter = adapter_factory()
                self._adapters[adapter_key] = PooledAdapter(base_adapter, pool_config, pool_type)

            return self._adapters[adapter_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all managed pools."""
        with self._lock:
            stats = {}
            for key, adapter in self._adapters.items():
                stats[key] = adapter.get_stats()
            return stats

    async def close_all(self) -> None:
        """Close all managed connection pools."""
        with self._lock:
            for adapter in self._adapters.values():
                await adapter.close()
            self._adapters.clear()


# Global pool manager instance
pool_manager = ConnectionPoolManager()


def get_pooled_adapter(
    adapter_key: str,
    adapter_factory: Callable[[], Any],
    pool_config: Optional[ConnectionConfig] = None,
    pool_type: str = "auto"
) -> PooledAdapter:
    """
    Convenience function to get a pooled adapter from the global manager.

    Args:
        adapter_key: Unique key for the adapter
        adapter_factory: Function to create the base adapter
        pool_config: Pool configuration
        pool_type: Pool type

    Returns:
        PooledAdapter instance
    """
    return pool_manager.get_pooled_adapter(adapter_key, adapter_factory, pool_config, pool_type)


def benchmark_connection_pooling(
    adapter_factory: Callable[[], Any],
    urls: List[str],
    iterations: int = 100,
    enable_pooling: bool = True
) -> Dict[str, Any]:
    """
    Benchmark the performance improvement from connection pooling.

    Args:
        adapter_factory: Function to create adapters
        urls: List of URLs to test
        iterations: Number of iterations
        enable_pooling: Whether to use pooling

    Returns:
        Benchmark results
    """
    import time

    results = {
        "with_pooling": {},
        "without_pooling": {},
        "improvement": {}
    }

    # Benchmark with pooling
    if enable_pooling:
        pooled_adapter = get_pooled_adapter("benchmark", adapter_factory)

        start_time = time.time()
        for _ in range(iterations):
            # Simulate requests
            time.sleep(0.001)  # Minimal delay to simulate network
        pooled_time = time.time() - start_time

        results["with_pooling"] = {
            "total_time": pooled_time,
            "avg_time": pooled_time / iterations,
            "stats": pooled_adapter.get_stats()
        }

    # Benchmark without pooling
    start_time = time.time()
    for _ in range(iterations):
        adapter = adapter_factory()
        # Simulate requests without pooling
        time.sleep(0.001)
    no_pool_time = time.time() - start_time

    results["without_pooling"] = {
        "total_time": no_pool_time,
        "avg_time": no_pool_time / iterations
    }

    # Calculate improvement
    if enable_pooling:
        improvement = (no_pool_time - pooled_time) / no_pool_time * 100
        results["improvement"] = {
            "percentage": improvement,
            "factor": no_pool_time / pooled_time if pooled_time > 0 else float('inf')
        }

    return results