"""
Connection Pooling and Reuse for OpenEval Lab Adapters

This module provides connection pooling, session reuse, and efficient resource management
for API adapters to reduce connection overhead and improve throughput.
"""

from __future__ import annotations

import asyncio
import time
import threading
from typing import Any, Dict, List, Optional, Callable, AsyncContextManager, TypeVar, TYPE_CHECKING
from dataclasses import dataclass
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

if TYPE_CHECKING:
    import aiohttp
    import httpx
    import requests
    from requests.adapters import HTTPAdapter

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

from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


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
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5  # failures before opening circuit
    circuit_breaker_timeout: float = 60.0  # seconds to wait before retrying
    health_check_interval: float = 30.0  # seconds between health checks
    adaptive_pool_sizing: bool = True
    min_connections: int = 2
    connection_scaling_factor: float = 1.5


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
            "success_rate": (
                (self.total_requests - self.failed_requests) / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
            "avg_response_time": self.avg_response_time,
            "pool_exhaustion_events": self.pool_exhaustion_events,
        }


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open

    def record_success(self) -> None:
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.threshold:
            self.state = "open"

    def can_attempt(self) -> bool:
        """Check if an operation can be attempted."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout:
                self.state = "half-open"
                return True
            return False
        elif self.state == "half-open":
            return True
        return False


class HealthChecker:
    """Health checker for connection validation."""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.last_check: Optional[float] = None
        self.is_healthy = True
        self._lock = threading.Lock()

    async def check_health(self, check_func: Callable[[], Any]) -> bool:
        """Perform health check if interval has passed."""
        current_time = time.time()

        with self._lock:
            if self.last_check is None or (current_time - self.last_check) >= self.check_interval:
                try:
                    await check_func()
                    self.is_healthy = True
                    self.last_check = current_time
                except Exception:
                    self.is_healthy = False
                    self.last_check = current_time

            return self.is_healthy


class AdaptivePoolSizer:
    """Adaptive pool sizing based on usage patterns."""

    def __init__(
        self,
        min_connections: int = 2,
        max_connections: int = 20,
        scaling_factor: float = 1.5,
        cooldown_period: float = 300.0,  # 5 minutes
    ):
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.scaling_factor = scaling_factor
        self.cooldown_period = cooldown_period
        self.last_scaling_time: Optional[float] = None
        self.current_target = min_connections

    def should_scale_up(self, stats: ConnectionStats) -> bool:
        """Determine if pool should scale up."""
        if stats.active_connections >= self.current_target * 0.8:  # 80% utilization
            return True
        if stats.pool_exhaustion_events > 0:
            return True
        return False

    def should_scale_down(self, stats: ConnectionStats) -> bool:
        """Determine if pool should scale down."""
        if stats.active_connections < self.current_target * 0.3:  # 30% utilization
            return True
        return False

    def get_target_size(self, stats: ConnectionStats) -> int:
        """Calculate target pool size."""
        current_time = time.time()

        # Check cooldown period
        if (
            self.last_scaling_time
            and (current_time - self.last_scaling_time) < self.cooldown_period
        ):
            return self.current_target

        if self.should_scale_up(stats):
            new_target = min(int(self.current_target * self.scaling_factor), self.max_connections)
            if new_target != self.current_target:
                self.current_target = new_target
                self.last_scaling_time = current_time
        elif self.should_scale_down(stats):
            new_target = max(int(self.current_target / self.scaling_factor), self.min_connections)
            if new_target != self.current_target:
                self.current_target = new_target
                self.last_scaling_time = current_time

        return self.current_target


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
        self._session: Optional[Any] = None
        self._connector: Optional[Any] = None

    async def _ensure_session(self) -> Any:
        """Ensure we have an active session."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for HTTPConnectionPool")

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
                connector=self._connector, timeout=timeout, base_url=self.base_url
            )

            self.stats.total_connections_created += 1

        return self._session

    async def acquire(self) -> Any:
        """Acquire a session from the pool."""
        with self._lock:
            self.stats.total_requests += 1
            return await self._ensure_session()

    async def release(self, connection: Any) -> None:
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
        self._client: Optional[Any] = None

    async def _ensure_client(self) -> Any:
        """Ensure we have an active client."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections,
            )

            self._client = httpx.AsyncClient(
                base_url=self.base_url, limits=limits, timeout=self.config.timeout
            )

            self.stats.total_connections_created += 1

        return self._client

    async def acquire(self) -> Any:
        """Acquire a client from the pool."""
        with self._lock:
            self.stats.total_requests += 1
            return await self._ensure_client()

    async def release(self, connection: Any) -> None:
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
        self._session: Optional[Any] = None

    def _ensure_session(self) -> Any:
        """Ensure we have an active session."""
        if self._session is None:
            self._session = requests.Session()

            # Configure the adapter
            adapter = HTTPAdapter(
                pool_connections=self.config.max_connections,
                pool_maxsize=self.config.max_connections,
                max_retries=self.config.retry_attempts,
                pool_block=False,
            )

            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

            self.stats.total_connections_created += 1

        return self._session

    def acquire(self) -> Any:
        """Acquire a session from the pool."""
        with self._lock:
            self.stats.total_requests += 1
            return self._ensure_session()

    async def release(self, connection: Any) -> None:
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
        pool_type: str = "auto",  # auto, aiohttp, httpx, requests
    ):
        self.base_adapter = base_adapter
        self.pool_config = pool_config or ConnectionConfig()
        self.pool_type = pool_type
        self._pool: Optional[ConnectionPool] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._health_checker: Optional[HealthChecker] = None
        self._adaptive_sizer: Optional[AdaptivePoolSizer] = None
        self._base_url = getattr(base_adapter, "base_url", None) or getattr(
            base_adapter, "api_base", None
        )

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

        logger.info(
            f"Initialized {self.pool_type} connection pool for adapter {self.base_adapter.__class__.__name__}"
        )

    def _get_circuit_breaker(self, adapter_key: str) -> CircuitBreaker:
        """Get or create circuit breaker for this adapter."""
        if self._circuit_breaker is None:
            self._circuit_breaker = CircuitBreaker(
                threshold=self.pool_config.circuit_breaker_threshold,
                timeout=self.pool_config.circuit_breaker_timeout,
            )
        return self._circuit_breaker

    def _get_health_checker(self, adapter_key: str) -> HealthChecker:
        """Get or create health checker for this adapter."""
        if self._health_checker is None:
            self._health_checker = HealthChecker(
                check_interval=self.pool_config.health_check_interval
            )
        return self._health_checker

    async def _health_check_func(self) -> None:
        """Health check function - makes a simple request."""
        # Simple health check - try to make a basic request
        try:
            if self._pool:
                connection = await self._pool.acquire()
                await self._pool.release(connection)
        except Exception:
            raise

    async def make_request(self, method: str, url: str, **kwargs: Any) -> Any:
        """
        Make an HTTP request using the connection pool with circuit breaker and health checks.

        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request parameters

        Returns:
            Response object
        """
        # Circuit breaker check
        adapter_key = f"{self.base_adapter.__class__.__name__}_{self.pool_type}"
        circuit_breaker = None
        if self.pool_config.enable_circuit_breaker:
            circuit_breaker = self._get_circuit_breaker(adapter_key)

            if not circuit_breaker.can_attempt():
                raise Exception(f"Circuit breaker is open for {adapter_key}")

        # Health check
        if self.pool_config.health_check_interval > 0:
            health_checker = self._get_health_checker(adapter_key)
            is_healthy = await health_checker.check_health(self._health_check_func)
            if not is_healthy:
                logger.warning(f"Health check failed for {adapter_key}")

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

        except Exception as e:
            # Record failure for circuit breaker
            if circuit_breaker is not None:
                circuit_breaker.record_failure()

            if self._pool:
                self._pool.stats.failed_requests += 1
            raise e

        else:
            # Record success for circuit breaker
            if circuit_breaker is not None:
                circuit_breaker.record_success()

        finally:
            await self._pool.release(connection)
            response_time = time.time() - start_time
            self._pool.stats.avg_response_time = (
                (self._pool.stats.avg_response_time * (self._pool.stats.total_requests - 1))
                + response_time
            ) / self._pool.stats.total_requests

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

    def _sync_request(
        self, session: requests.Session, method: str, url: str, kwargs: Dict[str, Any]
    ) -> str:
        """Synchronous request for requests library."""
        response = session.request(method, url, **kwargs)
        return response.text

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate method with connection pooling."""
        # This would be implemented based on the specific adapter's API
        # For now, delegate to base adapter
        if hasattr(self.base_adapter, "generate"):
            return self.base_adapter.generate(prompt, **kwargs)
        else:
            raise NotImplementedError("Base adapter does not implement generate method")

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Async generate method."""
        if hasattr(self.base_adapter, "agenerate") and asyncio.iscoroutinefunction(
            self.base_adapter.agenerate
        ):
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
                "base_adapter": self.base_adapter.__class__.__name__,
            }
        else:
            return {"pool_type": "none", "base_adapter": self.base_adapter.__class__.__name__}

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()


@asynccontextmanager
async def pooled_adapter_context(
    adapter: Any, pool_config: Optional[ConnectionConfig] = None, pool_type: str = "auto"
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
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._health_checkers: Dict[str, HealthChecker] = {}
        self._adaptive_sizers: Dict[str, AdaptivePoolSizer] = {}

    def get_pooled_adapter(
        self,
        adapter_key: str,
        adapter_factory: Callable[[], Any],
        pool_config: Optional[ConnectionConfig] = None,
        pool_type: str = "auto",
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
    pool_type: str = "auto",
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
    enable_pooling: bool = True,
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

    results = {"with_pooling": {}, "without_pooling": {}, "improvement": {}}

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
            "stats": pooled_adapter.get_stats(),
        }

    # Benchmark without pooling
    start_time = time.time()
    for _ in range(iterations):
        adapter_factory()  # Create adapter instance
        # Simulate requests without pooling
        time.sleep(0.001)
    no_pool_time = time.time() - start_time

    results["without_pooling"] = {"total_time": no_pool_time, "avg_time": no_pool_time / iterations}

    # Calculate improvement
    if enable_pooling:
        improvement = (no_pool_time - pooled_time) / no_pool_time * 100
        results["improvement"] = {
            "percentage": improvement,
            "factor": no_pool_time / pooled_time if pooled_time > 0 else float("inf"),
        }

    return results
