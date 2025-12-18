"""Resource manager module for managing compute and memory resources.

This module provides tools for tracking, allocating, and managing resources
during explanation generation, including memory management, CPU limits,
and resource pooling.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Generator


class ResourceType(Enum):
    """Types of resources that can be managed."""

    MEMORY = auto()
    CPU = auto()
    GPU = auto()
    DISK = auto()
    NETWORK = auto()
    THREAD = auto()
    PROCESS = auto()
    TOKEN = auto()


class ResourceState(Enum):
    """State of a resource."""

    AVAILABLE = auto()
    ALLOCATED = auto()
    RESERVED = auto()
    EXHAUSTED = auto()
    ERROR = auto()


class AllocationStrategy(Enum):
    """Strategy for resource allocation."""

    FIRST_FIT = auto()
    BEST_FIT = auto()
    WORST_FIT = auto()
    ROUND_ROBIN = auto()
    PRIORITY = auto()


@dataclass
class ResourceUsage:
    """Current usage statistics for a resource type."""

    resource_type: ResourceType
    total: float
    used: float
    available: float
    peak: float = 0.0
    unit: str = "bytes"
    timestamp: float = field(default_factory=time.time)

    @property
    def usage_percentage(self) -> float:
        """Get usage as percentage."""
        return (self.used / self.total * 100) if self.total > 0 else 0.0

    @property
    def is_critical(self) -> bool:
        """Check if usage is at critical level."""
        return self.usage_percentage > 90

    @property
    def is_high(self) -> bool:
        """Check if usage is high."""
        return self.usage_percentage > 75


@dataclass
class ResourceAllocation:
    """A specific resource allocation."""

    allocation_id: str
    resource_type: ResourceType
    amount: float
    allocated_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    owner: str | None = None
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if allocation has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get age of allocation in seconds."""
        return time.time() - self.allocated_at


@dataclass
class ResourceQuota:
    """Quota limits for a resource."""

    resource_type: ResourceType
    soft_limit: float
    hard_limit: float
    burst_limit: float | None = None
    unit: str = "bytes"

    def is_within_soft_limit(self, current: float) -> bool:
        """Check if current usage is within soft limit."""
        return current <= self.soft_limit

    def is_within_hard_limit(self, current: float) -> bool:
        """Check if current usage is within hard limit."""
        return current <= self.hard_limit

    def can_burst(self, current: float) -> bool:
        """Check if burst is allowed."""
        if self.burst_limit is None:
            return False
        return current <= self.burst_limit


@dataclass
class ResourcePoolConfig:
    """Configuration for a resource pool."""

    pool_name: str
    resource_type: ResourceType
    initial_size: float
    max_size: float
    growth_factor: float = 1.5
    shrink_threshold: float = 0.25
    allocation_strategy: AllocationStrategy = AllocationStrategy.FIRST_FIT


class ResourceTracker(ABC):
    """Abstract base class for resource trackers."""

    @abstractmethod
    def get_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        pass

    @abstractmethod
    def track(self) -> None:
        """Start tracking resource."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop tracking resource."""
        pass


class MemoryTracker(ResourceTracker):
    """Tracker for memory resources."""

    def __init__(self) -> None:
        """Initialize memory tracker."""
        self._peak_usage: float = 0
        self._tracking = False
        self._lock = threading.Lock()

    def get_usage(self) -> ResourceUsage:
        """Get current memory usage."""
        import sys

        # Get process memory info
        try:
            import resource

            mem_info = resource.getrusage(resource.RUSAGE_SELF)
            used = mem_info.ru_maxrss * 1024  # Convert to bytes on macOS
        except ImportError:
            # Fallback to sys.getsizeof for basic measurement
            used = sys.getsizeof([]) * 1000000  # Rough estimate

        # Estimate total available memory (platform dependent)
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError):
            total = 8 * 1024 * 1024 * 1024  # Default 8GB

        with self._lock:
            if used > self._peak_usage:
                self._peak_usage = used

        return ResourceUsage(
            resource_type=ResourceType.MEMORY,
            total=float(total),
            used=float(used),
            available=float(total - used),
            peak=self._peak_usage,
            unit="bytes",
        )

    def track(self) -> None:
        """Start tracking memory."""
        self._tracking = True
        gc.collect()

    def stop(self) -> None:
        """Stop tracking memory."""
        self._tracking = False


class CPUTracker(ResourceTracker):
    """Tracker for CPU resources."""

    def __init__(self) -> None:
        """Initialize CPU tracker."""
        self._start_time: float = 0
        self._tracking = False
        self._cpu_count = os.cpu_count() or 1

    def get_usage(self) -> ResourceUsage:
        """Get current CPU usage."""
        try:
            load = os.getloadavg()[0]
        except (AttributeError, OSError):
            load = 0.5  # Default estimate

        total = self._cpu_count * 100
        used = load / self._cpu_count * 100

        return ResourceUsage(
            resource_type=ResourceType.CPU,
            total=float(total),
            used=float(used),
            available=float(total - used),
            unit="percent",
        )

    def track(self) -> None:
        """Start tracking CPU."""
        self._tracking = True
        self._start_time = time.time()

    def stop(self) -> None:
        """Stop tracking CPU."""
        self._tracking = False


class ThreadPoolTracker(ResourceTracker):
    """Tracker for thread pool resources."""

    def __init__(self, max_threads: int = 10) -> None:
        """Initialize thread pool tracker."""
        self._max_threads = max_threads
        self._active_threads = 0
        self._lock = threading.Lock()

    def get_usage(self) -> ResourceUsage:
        """Get current thread pool usage."""
        with self._lock:
            return ResourceUsage(
                resource_type=ResourceType.THREAD,
                total=float(self._max_threads),
                used=float(self._active_threads),
                available=float(self._max_threads - self._active_threads),
                unit="threads",
            )

    def track(self) -> None:
        """Start tracking threads."""
        pass

    def stop(self) -> None:
        """Stop tracking threads."""
        pass

    def acquire(self) -> bool:
        """Acquire a thread from pool."""
        with self._lock:
            if self._active_threads < self._max_threads:
                self._active_threads += 1
                return True
            return False

    def release(self) -> None:
        """Release a thread back to pool."""
        with self._lock:
            if self._active_threads > 0:
                self._active_threads -= 1


class ResourcePool:
    """Pool of resources for allocation."""

    def __init__(self, config: ResourcePoolConfig) -> None:
        """Initialize resource pool."""
        self._config = config
        self._allocations: dict[str, ResourceAllocation] = {}
        self._current_size = config.initial_size
        self._lock = threading.RLock()
        self._allocation_counter = 0

    @property
    def available(self) -> float:
        """Get available resources in pool."""
        with self._lock:
            allocated = sum(a.amount for a in self._allocations.values())
            return self._current_size - allocated

    @property
    def utilization(self) -> float:
        """Get pool utilization percentage."""
        with self._lock:
            allocated = sum(a.amount for a in self._allocations.values())
            return (allocated / self._current_size * 100) if self._current_size > 0 else 0

    def allocate(
        self,
        amount: float,
        owner: str | None = None,
        priority: int = 0,
        ttl_seconds: float | None = None,
    ) -> ResourceAllocation | None:
        """Allocate resources from pool."""
        with self._lock:
            # Clean up expired allocations
            self._cleanup_expired()

            # Check if we have enough resources
            if amount > self.available:
                # Try to grow the pool
                if not self._try_grow(amount - self.available):
                    return None

            self._allocation_counter += 1
            allocation_id = f"{self._config.pool_name}-{self._allocation_counter}"

            allocation = ResourceAllocation(
                allocation_id=allocation_id,
                resource_type=self._config.resource_type,
                amount=amount,
                owner=owner,
                priority=priority,
                expires_at=time.time() + ttl_seconds if ttl_seconds else None,
            )

            self._allocations[allocation_id] = allocation
            return allocation

    def release(self, allocation_id: str) -> bool:
        """Release an allocation back to pool."""
        with self._lock:
            if allocation_id in self._allocations:
                del self._allocations[allocation_id]
                self._try_shrink()
                return True
            return False

    def _try_grow(self, needed: float) -> bool:
        """Try to grow the pool."""
        new_size = min(
            self._current_size * self._config.growth_factor,
            self._config.max_size,
        )

        if new_size >= self._current_size + needed:
            self._current_size = new_size
            return True
        return False

    def _try_shrink(self) -> None:
        """Try to shrink the pool if utilization is low."""
        if self.utilization < self._config.shrink_threshold * 100:
            allocated = sum(a.amount for a in self._allocations.values())
            min_size = max(allocated * 2, self._config.initial_size)
            self._current_size = max(min_size, self._current_size / self._config.growth_factor)

    def _cleanup_expired(self) -> None:
        """Clean up expired allocations."""
        expired = [aid for aid, alloc in self._allocations.items() if alloc.is_expired]
        for aid in expired:
            del self._allocations[aid]


class ResourceLimiter:
    """Rate limiter for resource access."""

    def __init__(
        self,
        max_rate: float,
        time_window: float = 1.0,
        burst_size: int = 1,
    ) -> None:
        """Initialize resource limiter."""
        self._max_rate = max_rate
        self._time_window = time_window
        self._burst_size = burst_size
        self._tokens = float(burst_size)
        self._last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0, block: bool = True) -> bool:
        """Acquire tokens from limiter."""
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            if not block:
                return False

        # Block until tokens are available
        while True:
            time.sleep(0.1)
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        new_tokens = elapsed * self._max_rate / self._time_window
        self._tokens = min(self._tokens + new_tokens, float(self._burst_size))


class ResourceManager:
    """Main resource manager for the explainers system."""

    def __init__(self) -> None:
        """Initialize resource manager."""
        self._trackers: dict[ResourceType, ResourceTracker] = {}
        self._pools: dict[str, ResourcePool] = {}
        self._quotas: dict[ResourceType, ResourceQuota] = {}
        self._limiters: dict[str, ResourceLimiter] = {}
        self._lock = threading.RLock()

        # Register default trackers
        self.register_tracker(ResourceType.MEMORY, MemoryTracker())
        self.register_tracker(ResourceType.CPU, CPUTracker())
        self.register_tracker(ResourceType.THREAD, ThreadPoolTracker())

    def register_tracker(self, resource_type: ResourceType, tracker: ResourceTracker) -> None:
        """Register a resource tracker."""
        with self._lock:
            self._trackers[resource_type] = tracker

    def register_pool(self, config: ResourcePoolConfig) -> ResourcePool:
        """Create and register a resource pool."""
        with self._lock:
            pool = ResourcePool(config)
            self._pools[config.pool_name] = pool
            return pool

    def register_quota(self, quota: ResourceQuota) -> None:
        """Register a resource quota."""
        with self._lock:
            self._quotas[quota.resource_type] = quota

    def register_limiter(
        self,
        name: str,
        max_rate: float,
        time_window: float = 1.0,
        burst_size: int = 1,
    ) -> ResourceLimiter:
        """Create and register a rate limiter."""
        with self._lock:
            limiter = ResourceLimiter(max_rate, time_window, burst_size)
            self._limiters[name] = limiter
            return limiter

    def get_usage(self, resource_type: ResourceType) -> ResourceUsage | None:
        """Get current usage for a resource type."""
        tracker = self._trackers.get(resource_type)
        if tracker:
            return tracker.get_usage()
        return None

    def get_all_usage(self) -> dict[ResourceType, ResourceUsage]:
        """Get usage for all tracked resources."""
        return {rt: tracker.get_usage() for rt, tracker in self._trackers.items()}

    def get_pool(self, pool_name: str) -> ResourcePool | None:
        """Get a resource pool by name."""
        return self._pools.get(pool_name)

    def get_limiter(self, name: str) -> ResourceLimiter | None:
        """Get a rate limiter by name."""
        return self._limiters.get(name)

    def check_quota(self, resource_type: ResourceType) -> bool:
        """Check if current usage is within quota."""
        quota = self._quotas.get(resource_type)
        if not quota:
            return True

        usage = self.get_usage(resource_type)
        if not usage:
            return True

        return quota.is_within_hard_limit(usage.used)

    def allocate_from_pool(
        self,
        pool_name: str,
        amount: float,
        owner: str | None = None,
        ttl_seconds: float | None = None,
    ) -> ResourceAllocation | None:
        """Allocate resources from a named pool."""
        pool = self._pools.get(pool_name)
        if pool:
            return pool.allocate(amount, owner, ttl_seconds=ttl_seconds)
        return None

    def release_to_pool(self, pool_name: str, allocation_id: str) -> bool:
        """Release an allocation back to a pool."""
        pool = self._pools.get(pool_name)
        if pool:
            return pool.release(allocation_id)
        return False

    def acquire_rate_limit(
        self, limiter_name: str, tokens: float = 1.0, block: bool = True
    ) -> bool:
        """Acquire tokens from a rate limiter."""
        limiter = self._limiters.get(limiter_name)
        if limiter:
            return limiter.acquire(tokens, block)
        return True  # No limiter = always allowed

    @contextmanager
    def scoped_allocation(
        self,
        pool_name: str,
        amount: float,
        owner: str | None = None,
    ) -> Generator[ResourceAllocation | None, None, None]:
        """Context manager for scoped resource allocation."""
        allocation = self.allocate_from_pool(pool_name, amount, owner)
        try:
            yield allocation
        finally:
            if allocation:
                self.release_to_pool(pool_name, allocation.allocation_id)

    @contextmanager
    def track_resource(
        self, resource_type: ResourceType
    ) -> Generator[ResourceTracker | None, None, None]:
        """Context manager for tracking a resource."""
        tracker = self._trackers.get(resource_type)
        if tracker:
            tracker.track()
            try:
                yield tracker
            finally:
                tracker.stop()
        else:
            yield None

    def force_cleanup(self) -> None:
        """Force cleanup of resources."""
        gc.collect()

    def get_status(self) -> dict[str, Any]:
        """Get overall resource manager status."""
        return {
            "trackers": list(self._trackers.keys()),
            "pools": {
                name: {"utilization": pool.utilization, "available": pool.available}
                for name, pool in self._pools.items()
            },
            "quotas": list(self._quotas.keys()),
            "limiters": list(self._limiters.keys()),
            "usage": {rt.name: usage.__dict__ for rt, usage in self.get_all_usage().items()},
        }


# Global instance
_resource_manager: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    """Get or create global resource manager."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def reset_resource_manager() -> None:
    """Reset global resource manager."""
    global _resource_manager
    _resource_manager = None


# Convenience functions
def get_memory_usage() -> ResourceUsage | None:
    """Get current memory usage."""
    return get_resource_manager().get_usage(ResourceType.MEMORY)


def get_cpu_usage() -> ResourceUsage | None:
    """Get current CPU usage."""
    return get_resource_manager().get_usage(ResourceType.CPU)


def create_resource_pool(
    pool_name: str,
    resource_type: ResourceType,
    initial_size: float,
    max_size: float,
    **kwargs: Any,
) -> ResourcePool:
    """Create a resource pool."""
    config = ResourcePoolConfig(
        pool_name=pool_name,
        resource_type=resource_type,
        initial_size=initial_size,
        max_size=max_size,
        **kwargs,
    )
    return get_resource_manager().register_pool(config)


def create_resource_limiter(
    name: str,
    max_rate: float,
    time_window: float = 1.0,
    burst_size: int = 1,
) -> ResourceLimiter:
    """Create a rate limiter."""
    return get_resource_manager().register_limiter(name, max_rate, time_window, burst_size)


def set_quota(
    resource_type: ResourceType,
    soft_limit: float,
    hard_limit: float,
    burst_limit: float | None = None,
) -> None:
    """Set a resource quota."""
    quota = ResourceQuota(
        resource_type=resource_type,
        soft_limit=soft_limit,
        hard_limit=hard_limit,
        burst_limit=burst_limit,
    )
    get_resource_manager().register_quota(quota)


def check_resources() -> dict[str, Any]:
    """Check all resource statuses."""
    return get_resource_manager().get_status()


@contextmanager
def allocate_resources(
    pool_name: str,
    amount: float,
    owner: str | None = None,
) -> Generator[ResourceAllocation | None, None, None]:
    """Context manager for temporary resource allocation."""
    with get_resource_manager().scoped_allocation(pool_name, amount, owner) as alloc:
        yield alloc


@contextmanager
def track_memory() -> Generator[ResourceTracker | None, None, None]:
    """Context manager for tracking memory usage."""
    with get_resource_manager().track_resource(ResourceType.MEMORY) as tracker:
        yield tracker


def force_gc() -> None:
    """Force garbage collection."""
    get_resource_manager().force_cleanup()
