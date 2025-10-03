"""
Machine Learning Optimization Module for OpenEval

This module integrates machine learning techniques to optimize evaluation strategies,
predict performance bottlenecks, and adapt evaluation parameters dynamically.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque
import statistics
import random

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    optim = None
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceData:
    """Performance data point for ML training."""

    timestamp: float
    task_type: str
    input_size: int
    parameters: Dict[str, Any]
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success: bool
    error_type: Optional[str] = None


@dataclass
class OptimizationModel:
    """Machine learning model for optimization predictions."""

    model_type: str
    features: List[str]
    target: str
    trained: bool = False
    accuracy: float = 0.0
    last_trained: Optional[float] = None


class PerformancePredictor:
    """Predicts performance metrics using historical data."""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.performance_history: deque = deque(maxlen=max_history)
        self.task_models: Dict[str, OptimizationModel] = {}
        self._lock = threading.Lock()

        # Simple statistical models as fallback
        self.task_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def record_performance(self, data: PerformanceData) -> None:
        """Record a performance data point."""
        with self._lock:
            self.performance_history.append(data)

            # Update statistical models
            stats = self.task_stats[data.task_type]
            stats["execution_time"].append(data.execution_time)
            stats["memory_usage"].append(data.memory_usage)
            stats["cpu_usage"].append(data.cpu_usage)
            stats["success_rate"].append(1.0 if data.success else 0.0)

            # Keep only recent data
            for key in stats:
                if len(stats[key]) > 1000:
                    stats[key] = stats[key][-500:]

    def predict_execution_time(
        self, task_type: str, input_size: int, parameters: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Predict execution time with confidence interval."""
        with self._lock:
            if task_type in self.task_stats and self.task_stats[task_type]["execution_time"]:
                times = self.task_stats[task_type]["execution_time"]
                mean_time = statistics.mean(times)
                std_time = statistics.stdev(times) if len(times) > 1 else mean_time * 0.1

                # Adjust for input size (simple linear scaling)
                size_factor = input_size / 1000.0  # Normalize to baseline
                predicted_time = mean_time * size_factor

                return predicted_time, std_time
            else:
                # Default prediction
                return 1.0, 0.5

    def predict_resource_usage(self, task_type: str) -> Dict[str, float]:
        """Predict resource usage for a task type."""
        with self._lock:
            if task_type in self.task_stats:
                stats = self.task_stats[task_type]
                return {
                    "cpu_percent": statistics.mean(stats.get("cpu_usage", [50.0])),
                    "memory_mb": statistics.mean(stats.get("memory_usage", [100.0])),
                    "success_rate": statistics.mean(stats.get("success_rate", [0.9])),
                }
            else:
                return {"cpu_percent": 50.0, "memory_mb": 100.0, "success_rate": 0.9}

    def get_optimal_parameters(self, task_type: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal parameters for a task type given constraints."""
        # Simple rule-based optimization as baseline
        optimal_params = {}

        if task_type == "evaluation":
            # Optimize batch size based on available memory
            memory_mb = constraints.get("memory_mb", 1000)
            optimal_params["batch_size"] = min(32, max(1, int(memory_mb / 50)))

            # Optimize concurrency based on CPU cores
            cpu_count = constraints.get("cpu_count", 4)
            optimal_params["max_concurrent"] = min(8, max(1, cpu_count))

        elif task_type == "metric_computation":
            # Optimize for vectorization
            optimal_params["use_simd"] = True
            optimal_params["use_gpu"] = constraints.get("gpu_available", False)

        elif task_type == "data_processing":
            # Optimize I/O operations
            optimal_params["buffer_size"] = (
                constraints.get("memory_mb", 1000) * 1024 * 1024 // 10
            )  # 10% of memory
            optimal_params["num_workers"] = constraints.get("cpu_count", 4)

        return optimal_params


class AdaptiveOptimizer:
    """Adaptive optimizer that learns from performance data."""

    def __init__(self, predictor: PerformancePredictor):
        self.predictor = predictor
        self.learning_rate = 0.1
        self.exploration_rate = 0.1
        self.parameter_ranges = {
            "batch_size": (1, 128),
            "max_concurrent": (1, 16),
            "buffer_size": (1024, 1048576),  # 1KB to 1MB
            "timeout": (10, 300),
        }

    def optimize_parameters(
        self,
        task_type: str,
        current_params: Dict[str, Any],
        performance_data: List[PerformanceData],
    ) -> Dict[str, Any]:
        """Optimize parameters using reinforcement learning approach."""
        optimized_params = current_params.copy()

        # Exploration: randomly adjust some parameters
        if random.random() < self.exploration_rate:
            for param_name, (min_val, max_val) in self.parameter_ranges.items():
                if param_name in optimized_params:
                    # Add some noise to current value
                    current_val = optimized_params[param_name]
                    noise = random.uniform(-0.2, 0.2) * (max_val - min_val)
                    new_val = max(min_val, min(max_val, current_val + noise))
                    optimized_params[param_name] = (
                        int(new_val) if isinstance(current_val, int) else new_val
                    )

        # Exploitation: use performance data to guide optimization
        if performance_data:
            # Find best performing parameter combinations
            successful_runs = [d for d in performance_data if d.success]

            if successful_runs:
                # Use parameters from best performing runs
                best_run = min(successful_runs, key=lambda x: x.execution_time)
                for param_name in self.parameter_ranges.keys():
                    if param_name in best_run.parameters:
                        # Gradually move towards best parameters
                        current_val = optimized_params.get(
                            param_name, best_run.parameters[param_name]
                        )
                        best_val = best_run.parameters[param_name]
                        optimized_params[param_name] = current_val + self.learning_rate * (
                            best_val - current_val
                        )

        return optimized_params

    def evaluate_parameter_set(self, task_type: str, parameters: Dict[str, Any]) -> float:
        """Evaluate how good a parameter set is."""
        # Predict performance
        predicted_time, time_uncertainty = self.predictor.predict_execution_time(
            task_type, parameters.get("input_size", 1000), parameters
        )

        resource_usage = self.predictor.predict_resource_usage(task_type)

        # Calculate score (lower is better)
        time_score = predicted_time + time_uncertainty
        resource_score = (
            resource_usage["cpu_percent"] / 100.0 + resource_usage["memory_mb"] / 1000.0
        )
        success_score = (1.0 - resource_usage["success_rate"]) * 10  # Penalty for low success rate

        return time_score + resource_score + success_score


class BottleneckDetector:
    """Detects performance bottlenecks using ML techniques."""

    def __init__(self):
        self.bottleneck_patterns = {
            "cpu_bound": ["high_cpu", "low_memory", "fast_io"],
            "memory_bound": ["low_cpu", "high_memory", "normal_io"],
            "io_bound": ["low_cpu", "low_memory", "slow_io"],
            "network_bound": ["normal_cpu", "normal_memory", "slow_network"],
        }

    def detect_bottleneck(self, metrics: Dict[str, float]) -> str:
        """Detect the primary bottleneck from system metrics."""
        cpu_percent = metrics.get("cpu_percent", 50)
        memory_percent = metrics.get("memory_percent", 50)
        io_wait = metrics.get("io_wait", 10)
        network_latency = metrics.get("network_latency", 50)

        # Simple rule-based classification
        if cpu_percent > 80:
            return "cpu_bound"
        elif memory_percent > 85:
            return "memory_bound"
        elif io_wait > 50:
            return "io_bound"
        elif network_latency > 200:
            return "network_bound"
        else:
            return "balanced"

    def suggest_optimization(
        self, bottleneck_type: str, current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest optimizations based on detected bottleneck."""
        suggestions = {}

        if bottleneck_type == "cpu_bound":
            suggestions.update(
                {
                    "increase_parallelism": True,
                    "use_vectorization": True,
                    "reduce_batch_size": True,
                    "consider_gpu_acceleration": True,
                }
            )
        elif bottleneck_type == "memory_bound":
            suggestions.update(
                {
                    "reduce_batch_size": True,
                    "enable_compression": True,
                    "use_streaming": True,
                    "increase_swap_space": True,
                }
            )
        elif bottleneck_type == "io_bound":
            suggestions.update(
                {
                    "increase_buffer_size": True,
                    "use_async_io": True,
                    "enable_caching": True,
                    "consider_ssd": True,
                }
            )
        elif bottleneck_type == "network_bound":
            suggestions.update(
                {
                    "reduce_data_transfer": True,
                    "use_compression": True,
                    "batch_requests": True,
                    "consider_cdn": True,
                }
            )

        return suggestions


class MLOptimizationEngine:
    """Main ML optimization engine coordinating all components."""

    def __init__(self):
        self.predictor = PerformancePredictor()
        self.optimizer = AdaptiveOptimizer(self.predictor)
        self.detector = BottleneckDetector()
        self.optimization_history: List[Dict[str, Any]] = []
        self._monitoring_thread: Optional[threading.Thread] = None
        self._running = False

    def start_monitoring(self) -> None:
        """Start the monitoring and optimization loop."""
        if self._running:
            return

        self._running = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("ML optimization monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the monitoring and optimization loop."""
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("ML optimization monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring and optimization loop."""
        while self._running:
            try:
                # Analyze recent performance
                self._analyze_performance()

                # Update optimization models
                self._update_models()

                # Apply optimizations
                self._apply_optimizations()

                time.sleep(60)  # Run every minute

            except Exception as e:
                logger.error(f"ML optimization monitoring error: {e}")
                time.sleep(30)

    def _analyze_performance(self) -> None:
        """Analyze recent performance data."""
        # This would analyze recent performance data and detect trends
        # For now, it's a placeholder for more sophisticated analysis
        pass

    def _update_models(self) -> None:
        """Update ML models with new data."""
        # This would retrain models periodically
        # For now, it's a placeholder
        pass

    def _apply_optimizations(self) -> None:
        """Apply learned optimizations."""
        # This would apply optimizations based on learned patterns
        # For now, it's a placeholder
        pass

    def optimize_evaluation(
        self, task_type: str, input_data: Any, current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize evaluation parameters for given task and data."""
        # Get performance predictions
        predicted_time, _ = self.predictor.predict_execution_time(
            task_type, len(str(input_data)) if input_data else 1000, current_config
        )

        # Detect potential bottlenecks
        resource_usage = self.predictor.predict_resource_usage(task_type)
        bottleneck = self.detector.detect_bottleneck(
            {
                "cpu_percent": resource_usage["cpu_percent"],
                "memory_percent": resource_usage["memory_mb"] / 10,  # Convert to percentage
                "io_wait": 10,  # Placeholder
                "network_latency": 50,  # Placeholder
            }
        )

        # Get optimization suggestions
        suggestions = self.detector.suggest_optimization(bottleneck, current_config)

        # Get optimal parameters
        optimal_params = self.predictor.get_optimal_parameters(
            task_type,
            {
                "memory_mb": resource_usage["memory_mb"],
                "cpu_count": 4,  # Placeholder
                "gpu_available": False,  # Placeholder
            },
        )

        # Combine all optimizations
        optimized_config = current_config.copy()
        optimized_config.update(optimal_params)

        # Apply bottleneck-specific optimizations
        if suggestions.get("reduce_batch_size") and "batch_size" in optimized_config:
            optimized_config["batch_size"] = max(1, optimized_config["batch_size"] // 2)
        if suggestions.get("increase_parallelism") and "max_concurrent" in optimized_config:
            optimized_config["max_concurrent"] = min(16, optimized_config["max_concurrent"] * 2)

        # Record optimization
        self.optimization_history.append(
            {
                "timestamp": time.time(),
                "task_type": task_type,
                "original_config": current_config,
                "optimized_config": optimized_config,
                "predicted_time": predicted_time,
                "detected_bottleneck": bottleneck,
                "suggestions": suggestions,
            }
        )

        return optimized_config

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about optimization performance."""
        if not self.optimization_history:
            return {"status": "no_data"}

        recent_optimizations = self.optimization_history[-100:]  # Last 100 optimizations

        task_types = {}
        for opt in recent_optimizations:
            task_type = opt["task_type"]
            if task_type not in task_types:
                task_types[task_type] = []
            task_types[task_type].append(opt)

        stats = {
            "total_optimizations": len(self.optimization_history),
            "recent_optimizations": len(recent_optimizations),
            "task_type_breakdown": {k: len(v) for k, v in task_types.items()},
            "bottleneck_distribution": defaultdict(int),
        }

        for opt in recent_optimizations:
            stats["bottleneck_distribution"][opt["detected_bottleneck"]] += 1

        return dict(stats)
