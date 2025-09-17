"""
Intelligent Resource Management for OpenEval Lab

This module provides intelligent resource management with dynamic scaling,
load balancing, and adaptive resource allocation based on system conditions.

Features:
- Dynamic CPU/GPU resource allocation
- Memory usage monitoring and optimization
- Load balancing across multiple processes
- Adaptive scaling based on workload patterns
- Resource contention detection and resolution
- Predictive resource allocation using ML
"""

from __future__ import annotations

import time
import threading
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import statistics
import psutil
from contextlib import contextmanager

try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    GPUtil = None
    HAS_GPU = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from .enhanced_logging import get_logger

logger = get_logger(__name__)


@dataclass
class ResourceMetrics:
    """Real-time resource metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    gpu_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_available_gb": self.memory_available_gb,
            "disk_usage_percent": self.disk_usage_percent,
            "network_io": self.network_io,
            "gpu_metrics": self.gpu_metrics,
            "timestamp": self.timestamp
        }


@dataclass
class ResourceLimits:
    """Resource limits and thresholds."""
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 85.0
    max_gpu_memory_percent: float = 90.0
    min_memory_available_gb: float = 1.0
    adaptive_scaling: bool = True
    scaling_cooldown_seconds: int = 60


class IntelligentResourceManager:
    """
    Intelligent resource manager with dynamic scaling and load balancing.
    """

    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self.metrics_history: List[ResourceMetrics] = []
        self._resource_lock = threading.Lock()
        self._scaling_cooldown_until = 0

        # Adaptive scaling parameters
        self.cpu_scaling_factor = 1.0
        self.memory_scaling_factor = 1.0
        self.thread_pool_size = min(mp.cpu_count(), 8)
        self.process_pool_size = max(1, mp.cpu_count() // 2)

        # Performance tracking
        self.task_completion_times: List[float] = []
        self.resource_efficiency_scores: List[float] = []

    def start_monitoring(self) -> None:
        """Start resource monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitoring_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")

    def _monitor_resources(self) -> None:
        """Monitor system resources continuously."""
        while not self._stop_monitoring.is_set():
            try:
                metrics = self._collect_resource_metrics()
                with self._resource_lock:
                    self.metrics_history.append(metrics)

                    # Keep only recent history (last 100 measurements)
                    if len(self.metrics_history) > 100:
                        self.metrics_history = self.metrics_history[-100:]

                # Adaptive scaling
                if self.limits.adaptive_scaling:
                    self._adaptive_scaling(metrics)

                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(5.0)  # Back off on errors

    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        metrics = ResourceMetrics()

        try:
            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_used_gb = memory.used / (1024 ** 3)
            metrics.memory_available_gb = memory.available / (1024 ** 3)

            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.disk_usage_percent = disk.percent

            # Network I/O (simplified)
            network = psutil.net_io_counters()
            if network:
                metrics.network_io = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }

            # GPU metrics
            if HAS_GPU and GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Primary GPU
                        metrics.gpu_metrics = {
                            "gpu_percent": gpu.load * 100,
                            "gpu_memory_percent": gpu.memoryUtil * 100,
                            "gpu_memory_used_mb": gpu.memoryUsed,
                            "gpu_memory_total_mb": gpu.memoryTotal,
                            "gpu_temperature": gpu.temperature
                        }
                except Exception as e:
                    logger.debug(f"GPU metrics collection failed: {e}")

        except Exception as e:
            logger.error(f"Failed to collect resource metrics: {e}")

        return metrics

    def _adaptive_scaling(self, metrics: ResourceMetrics) -> None:
        """Perform adaptive scaling based on current metrics."""
        current_time = time.time()

        # Check cooldown
        if current_time < self._scaling_cooldown_until:
            return

        scaling_needed = False

        # CPU scaling
        if metrics.cpu_percent > self.limits.max_cpu_percent:
            self.cpu_scaling_factor = max(0.5, self.cpu_scaling_factor * 0.9)
            scaling_needed = True
            logger.info(f"Reducing CPU scaling factor to {self.cpu_scaling_factor:.2f}")
        elif metrics.cpu_percent < 50.0 and self.cpu_scaling_factor < 1.0:
            self.cpu_scaling_factor = min(1.0, self.cpu_scaling_factor * 1.1)
            scaling_needed = True
            logger.info(f"Increasing CPU scaling factor to {self.cpu_scaling_factor:.2f}")

        # Memory scaling
        if metrics.memory_percent > self.limits.max_memory_percent:
            self.memory_scaling_factor = max(0.5, self.memory_scaling_factor * 0.9)
            scaling_needed = True
            logger.info(f"Reducing memory scaling factor to {self.memory_scaling_factor:.2f}")
        elif metrics.memory_available_gb > self.limits.min_memory_available_gb * 2:
            self.memory_scaling_factor = min(1.0, self.memory_scaling_factor * 1.05)
            scaling_needed = True
            logger.info(f"Increasing memory scaling factor to {self.memory_scaling_factor:.2f}")

        if scaling_needed:
            self._scaling_cooldown_until = current_time + self.limits.scaling_cooldown_seconds

    def get_optimal_thread_count(self, task_type: str = "general") -> int:
        """Get optimal thread count based on current resource usage."""
        with self._resource_lock:
            if not self.metrics_history:
                return min(mp.cpu_count(), 4)

            latest = self.metrics_history[-1]

            # Base calculation on CPU availability
            available_cpu_percent = 100.0 - latest.cpu_percent
            base_threads = max(1, int(available_cpu_percent / 25.0))  # 25% CPU per thread

            # Apply scaling factor
            optimal_threads = int(base_threads * self.cpu_scaling_factor)

            # Task-specific adjustments
            if task_type == "cpu_intensive":
                optimal_threads = max(1, optimal_threads // 2)
            elif task_type == "io_intensive":
                optimal_threads = min(mp.cpu_count(), optimal_threads * 2)

            return min(mp.cpu_count(), max(1, optimal_threads))

    def get_optimal_batch_size(self, item_size_estimate: int = 1000) -> int:
        """Get optimal batch size based on available memory."""
        with self._resource_lock:
            if not self.metrics_history:
                return 1000

            latest = self.metrics_history[-1]

            # Estimate available memory for processing (leave 20% buffer)
            available_memory_gb = latest.memory_available_gb * 0.8

            # Estimate items that can fit in memory (rough heuristic)
            max_items = int((available_memory_gb * 1024 * 1024 * 1024) / item_size_estimate)

            # Apply memory scaling factor
            optimal_batch = int(max_items * self.memory_scaling_factor)

            return max(10, min(10000, optimal_batch))

    def should_throttle(self) -> Tuple[bool, str]:
        """Check if processing should be throttled."""
        with self._resource_lock:
            if not self.metrics_history:
                return False, ""

            latest = self.metrics_history[-1]

            if latest.cpu_percent > self.limits.max_cpu_percent:
                return True, f"CPU usage too high: {latest.cpu_percent:.1f}%"

            if latest.memory_percent > self.limits.max_memory_percent:
                return True, f"Memory usage too high: {latest.memory_percent:.1f}%"

            if latest.memory_available_gb < self.limits.min_memory_available_gb:
                return True, f"Available memory too low: {latest.memory_available_gb:.2f}GB"

            return False, ""

    @contextmanager
    def resource_aware_execution(self, task_name: str = "unnamed_task"):
        """Context manager for resource-aware task execution."""
        start_time = time.time()
        start_metrics = self._collect_resource_metrics()

        try:
            yield
        finally:
            end_time = time.time()
            end_metrics = self._collect_resource_metrics()

            execution_time = end_time - start_time

            # Track performance
            with self._resource_lock:
                self.task_completion_times.append(execution_time)

                # Calculate resource efficiency score
                cpu_efficiency = 1.0 - (end_metrics.cpu_percent - start_metrics.cpu_percent) / 100.0
                memory_efficiency = 1.0 - (end_metrics.memory_percent - start_metrics.memory_percent) / 100.0
                efficiency_score = (cpu_efficiency + memory_efficiency) / 2.0

                self.resource_efficiency_scores.append(efficiency_score)

                # Keep history bounded
                if len(self.task_completion_times) > 50:
                    self.task_completion_times = self.task_completion_times[-50:]
                if len(self.resource_efficiency_scores) > 50:
                    self.resource_efficiency_scores = self.resource_efficiency_scores[-50:]

            logger.debug(f"Task '{task_name}' completed in {execution_time:.3f}s with efficiency {efficiency_score:.3f}")

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource usage summary."""
        with self._resource_lock:
            if not self.metrics_history:
                return {"error": "No metrics available"}

            latest = self.metrics_history[-1]

            summary = {
                "current": latest.to_dict(),
                "scaling_factors": {
                    "cpu": self.cpu_scaling_factor,
                    "memory": self.memory_scaling_factor
                },
                "optimal_settings": {
                    "thread_count": self.get_optimal_thread_count(),
                    "batch_size": self.get_optimal_batch_size()
                }
            }

            # Performance statistics
            if self.task_completion_times:
                summary["performance"] = {
                    "avg_task_time": statistics.mean(self.task_completion_times),
                    "median_task_time": statistics.median(self.task_completion_times),
                    "task_count": len(self.task_completion_times)
                }

            if self.resource_efficiency_scores:
                summary["efficiency"] = {
                    "avg_efficiency": statistics.mean(self.resource_efficiency_scores),
                    "efficiency_trend": "improving" if len(self.resource_efficiency_scores) > 5 and
                                      self.resource_efficiency_scores[-1] > self.resource_efficiency_scores[-5] else "stable"
                }

            return summary

    def predict_resource_needs(self, task_complexity: float) -> Dict[str, Any]:
        """Predict resource needs based on task complexity using historical data."""
        with self._resource_lock:
            if len(self.task_completion_times) < 5:
                return {"error": "Insufficient historical data"}

            # Simple linear prediction based on historical performance
            if HAS_NUMPY and np is not None:
                complexities = np.arange(len(self.task_completion_times))
                times = np.array(self.task_completion_times)

                # Fit linear model
                coeffs = np.polyfit(complexities, times, 1)
                predicted_time = coeffs[0] * task_complexity + coeffs[1]

                # Estimate resource usage based on predicted time
                predicted_cpu_percent = min(100.0, 20.0 + predicted_time * 10.0)
                predicted_memory_gb = 0.5 + predicted_time * 0.1

                return {
                    "predicted_time": float(predicted_time),
                    "predicted_cpu_percent": predicted_cpu_percent,
                    "predicted_memory_gb": predicted_memory_gb,
                    "recommended_threads": max(1, int(4.0 / (1.0 + predicted_time))),
                    "confidence": "medium" if len(self.task_completion_times) > 10 else "low"
                }
            else:
                # Fallback prediction
                avg_time = statistics.mean(self.task_completion_times)
                return {
                    "predicted_time": avg_time * task_complexity,
                    "predicted_cpu_percent": 50.0,
                    "predicted_memory_gb": 1.0,
                    "recommended_threads": 2,
                    "confidence": "low"
                }


# Global resource manager instance
_resource_manager: Optional[IntelligentResourceManager] = None


def get_resource_manager() -> IntelligentResourceManager:
    """Get the global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = IntelligentResourceManager()
        _resource_manager.start_monitoring()
    return _resource_manager


def resource_aware_task(func: Callable) -> Callable:
    """Decorator for resource-aware task execution."""
    def wrapper(*args, **kwargs):
        manager = get_resource_manager()
        with manager.resource_aware_execution(func.__name__):
            return func(*args, **kwargs)
    return wrapper


# Utility functions for easy integration
def get_optimal_concurrency() -> int:
    """Get optimal concurrency level based on current resources."""
    return get_resource_manager().get_optimal_thread_count()


def get_optimal_batch_size(item_size: int = 1000) -> int:
    """Get optimal batch size based on available memory."""
    return get_resource_manager().get_optimal_batch_size(item_size)


def should_throttle_processing() -> Tuple[bool, str]:
    """Check if processing should be throttled due to resource constraints."""
    return get_resource_manager().should_throttle()