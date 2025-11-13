"""
Unified Connection Pooling System for OpenEval Lab

Consolidates connection management across HTTP, database, and other
protocol adapters. This replaces scattered connection pooling logic
with a unified, efficient implementation.

Features:
- Protocol-agnostic connection pooling
- Configurable pool sizing and timeouts
- Health checking and automatic reconnection
- Connection reuse metrics
- Thread-safe operations
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Any, Callable, TypeVar, Generic
from collections import deque
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PoolMetrics:
    """Metrics for connection pool performance."""

    connections_created: int = 0
    connections_reused: int = 0
    connections_closed: int = 0
    peak_utilization: float = 0.0
    average_wait_time_ms: float = 0.0
    failed_connections: int = 0
    timeouts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "connections_created": self.connections_created,
            "connections_reused": self.connections_reused,
            "connections_closed": self.connections_closed,
            "peak_utilization": self.peak_utilization,
            "average_wait_time_ms": self.average_wait_time_ms,
            "failed_connections": self.failed_connections,
            "timeouts": self.timeouts,
        }


class ConnectionPoolItem(Generic[T]):
    """Wrapper for a pooled connection."""

    def __init__(self, connection: T):
        """Initialize pool item.

        Args:
            connection: The connection object
        """
        self.connection = connection
        self.created_at = time.time()
        self.last_used_at = time.time()
        self.use_count = 0
        self.is_healthy = True

    def mark_used(self) -> None:
        """Mark connection as used."""
        self.last_used_at = time.time()
        self.use_count += 1

    def age(self) -> float:
        """Get age in seconds."""
        return time.time() - self.created_at

    def idle_time(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used_at


class ConnectionPool(Generic[T]):
    """Generic connection pool for any connection type.

    Provides connection reuse, health checking, and lifecycle management.
    """

    def __init__(
        self,
        create_connection: Callable[[], T],
        validate_connection: Callable[[T], bool],
        destroy_connection: Callable[[T], None],
        min_size: int = 5,
        max_size: int = 20,
        max_idle_time: float = 300.0,
        max_lifetime: float = 3600.0,
        timeout: float = 30.0,
    ):
        """Initialize connection pool.

        Args:
            create_connection: Function to create new connections
            validate_connection: Function to validate connection health
            destroy_connection: Function to destroy connections
            min_size: Minimum pool size
            max_size: Maximum pool size
            max_idle_time: Maximum idle time before recycling (seconds)
            max_lifetime: Maximum lifetime before recycling (seconds)
            timeout: Timeout for acquiring connection (seconds)
        """
        self.create_connection = create_connection
        self.validate_connection = validate_connection
        self.destroy_connection = destroy_connection
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.max_lifetime = max_lifetime
        self.timeout = timeout

        self.available: deque[ConnectionPoolItem[T]] = deque()
        self.in_use: set[ConnectionPoolItem[T]] = set()
        self.lock = threading.RLock()
        self.metrics = PoolMetrics()

        # Initialize minimum connections
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Initialize pool with minimum connections."""
        for _ in range(self.min_size):
            try:
                conn = self.create_connection()
                item = ConnectionPoolItem(conn)
                self.available.append(item)
                self.metrics.connections_created += 1
            except Exception as e:
                logger.error(f"Failed to create connection: {e}")
                self.metrics.failed_connections += 1

    def acquire(self, timeout: Optional[float] = None) -> T:
        """Acquire a connection from the pool.

        Args:
            timeout: Timeout in seconds

        Returns:
            A connection

        Raises:
            TimeoutError: If timeout exceeded
        """
        timeout = timeout or self.timeout
        start_time = time.time()

        with self.lock:
            # Try to get an available connection
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self.metrics.timeouts += 1
                    raise TimeoutError(f"Failed to acquire connection within {timeout}s")

                # Reuse available connection if valid
                while self.available:
                    item = self.available.popleft()

                    # Check if connection is still valid
                    if (
                        item.is_healthy
                        and item.idle_time() < self.max_idle_time
                        and item.age() < self.max_lifetime
                        and self.validate_connection(item.connection)
                    ):
                        item.mark_used()
                        self.in_use.add(item)
                        self.metrics.connections_reused += 1
                        return item.connection

                    # Connection is invalid, remove it
                    try:
                        self.destroy_connection(item.connection)
                        self.metrics.connections_closed += 1
                    except Exception as e:
                        logger.warning(f"Error destroying connection: {e}")

                # Create new connection if below max
                if len(self.in_use) + len(self.available) < self.max_size:
                    try:
                        conn = self.create_connection()
                        item = ConnectionPoolItem(conn)
                        item.mark_used()
                        self.in_use.add(item)
                        self.metrics.connections_created += 1

                        # Update peak utilization
                        utilization = len(self.in_use) / self.max_size
                        self.metrics.peak_utilization = max(
                            self.metrics.peak_utilization, utilization
                        )

                        return conn
                    except Exception as e:
                        logger.error(f"Failed to create connection: {e}")
                        self.metrics.failed_connections += 1

                # Wait briefly before trying again
                time.sleep(0.01)

    def release(self, connection: T) -> None:
        """Release a connection back to the pool.

        Args:
            connection: The connection to release
        """
        with self.lock:
            # Find the connection item
            item = None
            for conn_item in list(self.in_use):
                if conn_item.connection == connection:
                    item = conn_item
                    break

            if not item:
                logger.warning("Attempted to release unknown connection")
                return

            self.in_use.remove(item)
            self.available.append(item)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self.lock:
            # Close in-use connections
            for item in self.in_use:
                try:
                    self.destroy_connection(item.connection)
                    self.metrics.connections_closed += 1
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")

            # Close available connections
            while self.available:
                item = self.available.popleft()
                try:
                    self.destroy_connection(item.connection)
                    self.metrics.connections_closed += 1
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")

            self.in_use.clear()
            self.available.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get pool status.

        Returns:
            Status dictionary
        """
        with self.lock:
            return {
                "available": len(self.available),
                "in_use": len(self.in_use),
                "max_size": self.max_size,
                "total": len(self.in_use) + len(self.available),
                "utilization": len(self.in_use) / self.max_size,
                "metrics": self.metrics.to_dict(),
            }

    def health_check(self) -> None:
        """Perform health check on all connections."""
        with self.lock:
            # Check in-use connections
            for item in list(self.in_use):
                try:
                    if not self.validate_connection(item.connection):
                        item.is_healthy = False
                except Exception as e:
                    logger.warning(f"Health check failed: {e}")
                    item.is_healthy = False

            # Check available connections
            for item in list(self.available):
                try:
                    if not self.validate_connection(item.connection):
                        item.is_healthy = False
                except Exception as e:
                    logger.warning(f"Health check failed: {e}")
                    item.is_healthy = False

    def __enter__(self) -> "ConnectionPool":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close_all()


class PoolContextManager(Generic[T]):
    """Context manager for acquiring connections from a pool."""

    def __init__(self, pool: ConnectionPool[T], timeout: Optional[float] = None) -> None:
        """Initialize context manager.

        Args:
            pool: The connection pool
            timeout: Acquisition timeout
        """
        self.pool = pool
        self.timeout = timeout
        self.connection: Optional[T] = None

    def __enter__(self) -> Optional[T]:
        """Acquire connection on enter."""
        self.connection = self.pool.acquire(self.timeout)
        return self.connection

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release connection on exit."""
        if self.connection:
            self.pool.release(self.connection)
