"""
Resource monitoring and warnings for memory and CPU usage.

Provides real-time monitoring of system resources with proactive warnings
when approaching resource limits.
"""

import threading
import time
from typing import Any, Dict, Optional

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None  # type: ignore


class ResourceMonitor:
    """Monitor system resources and issue warnings."""

    def __init__(
        self,
        memory_threshold_percent: float = 85.0,
        cpu_threshold_percent: float = 90.0,
        check_interval: float = 5.0,
    ):
        """Initialize resource monitor.

        Args:
            memory_threshold_percent: Memory threshold for warnings (0-100)
            cpu_threshold_percent: CPU threshold for warnings (0-100)
            check_interval: Check interval in seconds
        """
        self.memory_threshold = memory_threshold_percent
        self.cpu_threshold = cpu_threshold_percent
        self.check_interval = check_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._warnings: Dict[str, Dict[str, Any]] = {}
        self._stats = {
            "peak_memory_mb": 0.0,
            "peak_cpu_percent": 0.0,
            "memory_warnings": 0,
            "cpu_warnings": 0,
        }

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(daemon=True, target=self._monitor_loop)
        self._thread.start()

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self._check_resources()
                time.sleep(self.check_interval)
            except Exception:
                # Continue monitoring even if check fails
                pass

    def _check_resources(self) -> None:
        """Check current resource usage."""
        if not HAS_PSUTIL or psutil is None:
            return

        # Check memory
        try:
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent

            with self._lock:
                self._stats["peak_memory_mb"] = max(
                    self._stats["peak_memory_mb"], memory_info.used / (1024 * 1024)
                )

                if memory_percent > self.memory_threshold:
                    self._stats["memory_warnings"] += 1
                    self._warnings["memory"] = {
                        "percent": memory_percent,
                        "used_gb": memory_info.used / (1024**3),
                        "available_gb": memory_info.available / (1024**3),
                        "timestamp": time.time(),
                    }
        except Exception:
            pass

        # Check CPU
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)

            with self._lock:
                self._stats["peak_cpu_percent"] = max(self._stats["peak_cpu_percent"], cpu_percent)

                if cpu_percent > self.cpu_threshold:
                    self._stats["cpu_warnings"] += 1
                    self._warnings["cpu"] = {
                        "percent": cpu_percent,
                        "timestamp": time.time(),
                    }
        except Exception:
            pass

    def get_memory_info(self) -> Optional[Dict[str, Any]]:
        """Get current memory information.

        Returns:
            Dictionary with memory stats or None if psutil unavailable
        """
        if not HAS_PSUTIL or psutil is None:
            return None

        try:
            memory_info = psutil.virtual_memory()
            return {
                "percent": memory_info.percent,
                "used_mb": memory_info.used / (1024**2),
                "available_mb": memory_info.available / (1024**2),
                "total_mb": memory_info.total / (1024**2),
            }
        except Exception:
            return None

    def get_cpu_info(self) -> Optional[Dict[str, Any]]:
        """Get current CPU information.

        Returns:
            Dictionary with CPU stats or None if psutil unavailable
        """
        if not HAS_PSUTIL or psutil is None:
            return None

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            return {
                "percent": cpu_percent,
                "cpu_count": cpu_count,
            }
        except Exception:
            return None

    def get_warnings(self) -> Dict[str, Any]:
        """Get current resource warnings.

        Returns:
            Dictionary of warnings
        """
        with self._lock:
            return self._warnings.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics.

        Returns:
            Dictionary with stats
        """
        with self._lock:
            return self._stats.copy()

    def warn_if_high_memory(self, percentage: float = 80.0) -> Optional[str]:
        """Check if memory usage is high and return warning message.

        Args:
            percentage: Memory threshold percentage

        Returns:
            Warning message or None
        """
        memory_info = self.get_memory_info()
        if memory_info and memory_info["percent"] > percentage:
            return (
                f"⚠️  High memory usage: {memory_info['percent']:.1f}% "
                f"({memory_info['used_mb']:.0f}MB / {memory_info['total_mb']:.0f}MB)"
            )
        return None

    def warn_if_high_cpu(self, percentage: float = 85.0) -> Optional[str]:
        """Check if CPU usage is high and return warning message.

        Args:
            percentage: CPU threshold percentage

        Returns:
            Warning message or None
        """
        cpu_info = self.get_cpu_info()
        if cpu_info and cpu_info["percent"] > percentage:
            return (
                f"⚠️  High CPU usage: {cpu_info['percent']:.1f}% " f"({cpu_info['cpu_count']} cores)"
            )
        return None


# Global resource monitor instance
_monitor = ResourceMonitor()


def start_monitoring() -> None:
    """Start global resource monitoring."""
    _monitor.start()


def stop_monitoring() -> None:
    """Stop global resource monitoring."""
    _monitor.stop()


def get_memory_info() -> Optional[Dict]:
    """Get current memory information."""
    return _monitor.get_memory_info()


def get_cpu_info() -> Optional[Dict]:
    """Get current CPU information."""
    return _monitor.get_cpu_info()


def get_resource_warnings() -> Dict:
    """Get current resource warnings."""
    return _monitor.get_warnings()
