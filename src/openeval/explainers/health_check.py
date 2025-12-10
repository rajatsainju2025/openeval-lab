"""Health check system for explainer components.

Provides health monitoring and status reporting for explainer services.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List

from .base import CodeExplainer
from .types import CodeElement, CodeElementType, ExplainLevel


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "components": [c.to_dict() for c in self.components],
            "summary": {
                "total": len(self.components),
                "healthy": sum(1 for c in self.components if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in self.components if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for c in self.components if c.status == HealthStatus.UNHEALTHY),
            },
        }


# Health check function type
HealthCheckFunc = Callable[[], ComponentHealth]


class HealthChecker:
    """Health checker for explainer system components.

    Monitors and reports health status of various components.
    """

    def __init__(self, version: str = "1.0.0") -> None:
        """Initialize health checker.

        Args:
            version: System version string.
        """
        self.version = version
        self._checks: Dict[str, HealthCheckFunc] = {}
        self._last_results: Dict[str, ComponentHealth] = {}

    def register(self, name: str, check_func: HealthCheckFunc) -> "HealthChecker":
        """Register a health check.

        Args:
            name: Component name.
            check_func: Function that returns ComponentHealth.

        Returns:
            Self for method chaining.
        """
        self._checks[name] = check_func
        return self

    def unregister(self, name: str) -> "HealthChecker":
        """Unregister a health check.

        Args:
            name: Component name to remove.

        Returns:
            Self for method chaining.
        """
        self._checks.pop(name, None)
        self._last_results.pop(name, None)
        return self

    def check_component(self, name: str) -> ComponentHealth:
        """Check health of a single component.

        Args:
            name: Component name.

        Returns:
            ComponentHealth for the component.

        Raises:
            KeyError: If component not registered.
        """
        if name not in self._checks:
            raise KeyError(f"Component '{name}' not registered")

        import time

        start = time.time()
        try:
            health = self._checks[name]()
            health.latency_ms = (time.time() - start) * 1000
            health.last_check = datetime.utcnow()
        except Exception as e:
            health = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                latency_ms=(time.time() - start) * 1000,
            )

        self._last_results[name] = health
        return health

    def check_all(self) -> SystemHealth:
        """Check health of all registered components.

        Returns:
            SystemHealth with all component statuses.
        """
        components = []
        for name in self._checks:
            components.append(self.check_component(name))

        # Determine overall status
        if all(c.status == HealthStatus.HEALTHY for c in components):
            overall = HealthStatus.HEALTHY
        elif any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in components):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNKNOWN

        return SystemHealth(
            status=overall,
            components=components,
            version=self.version,
        )

    def get_last_results(self) -> Dict[str, ComponentHealth]:
        """Get cached results from last health check.

        Returns:
            Dictionary of component names to their last health status.
        """
        return dict(self._last_results)

    def is_healthy(self) -> bool:
        """Quick check if system is healthy.

        Returns:
            True if all components are healthy.
        """
        health = self.check_all()
        return health.status == HealthStatus.HEALTHY


def create_explainer_health_check(
    explainer: CodeExplainer,
    name: str = "explainer",
) -> HealthCheckFunc:
    """Create a health check function for an explainer.

    Args:
        explainer: CodeExplainer to check.
        name: Component name.

    Returns:
        Health check function.
    """

    def check() -> ComponentHealth:
        import time

        start = time.time()
        try:
            # Create minimal test element
            test_element = CodeElement(
                type=CodeElementType.FUNCTION,
                name="health_check_test",
                source_code="def test(): pass",
                line_start=1,
                line_end=1,
            )

            # Try to explain
            result = explainer.explain(test_element, ExplainLevel.SUMMARY)

            latency = (time.time() - start) * 1000

            # Check result quality
            if result and result.explanation and len(result.explanation) > 0:
                if latency < 5000:
                    status = HealthStatus.HEALTHY
                    message = "Explainer responding normally"
                else:
                    status = HealthStatus.DEGRADED
                    message = f"High latency: {latency:.0f}ms"
            else:
                status = HealthStatus.DEGRADED
                message = "Returned empty or invalid result"

            return ComponentHealth(
                name=name,
                status=status,
                message=message,
                latency_ms=latency,
                metadata={"confidence": result.confidence if result else 0},
            )

        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}",
                latency_ms=(time.time() - start) * 1000,
            )

    return check


def create_cache_health_check(
    cache_manager: Any,  # CacheManager
    name: str = "cache",
) -> HealthCheckFunc:
    """Create a health check function for a cache manager.

    Args:
        cache_manager: CacheManager to check.
        name: Component name.

    Returns:
        Health check function.
    """

    def check() -> ComponentHealth:
        import time

        start = time.time()
        try:
            # Try basic operations
            test_key = "__health_check_test__"

            # Check get/exists (should work even if key doesn't exist)
            cache_manager.exists(test_key)

            stats = cache_manager.get_stats()
            latency = (time.time() - start) * 1000

            # Analyze stats
            hit_rate = stats.get("hit_rate", 0)
            size = stats.get("size", 0)

            if latency < 100:
                status = HealthStatus.HEALTHY
                message = f"Cache operational (size={size}, hit_rate={hit_rate:.1f}%)"
            else:
                status = HealthStatus.DEGRADED
                message = f"Cache slow ({latency:.0f}ms)"

            return ComponentHealth(
                name=name,
                status=status,
                message=message,
                latency_ms=latency,
                metadata=stats,
            )

        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}",
                latency_ms=(time.time() - start) * 1000,
            )

    return check


def create_memory_health_check(
    warn_threshold_mb: float = 500,
    critical_threshold_mb: float = 1000,
    name: str = "memory",
) -> HealthCheckFunc:
    """Create a health check for memory usage.

    Args:
        warn_threshold_mb: Memory usage to trigger warning.
        critical_threshold_mb: Memory usage to trigger critical.
        name: Component name.

    Returns:
        Health check function.
    """

    def check() -> ComponentHealth:
        import os

        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb < warn_threshold_mb:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_mb:.1f}MB"
            elif memory_mb < critical_threshold_mb:
                status = HealthStatus.DEGRADED
                message = f"Memory usage elevated: {memory_mb:.1f}MB"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {memory_mb:.1f}MB"

            return ComponentHealth(
                name=name,
                status=status,
                message=message,
                metadata={
                    "memory_mb": memory_mb,
                    "warn_threshold": warn_threshold_mb,
                    "critical_threshold": critical_threshold_mb,
                },
            )

        except ImportError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="psutil not installed",
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Error: {str(e)}",
            )

    return check


# Global health checker instance
_global_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance.

    Returns:
        HealthChecker singleton.
    """
    return _global_health_checker
