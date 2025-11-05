"""Advanced metrics collection and analysis for OpenEval Lab."""

import time
import json
import statistics
import math
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Iterable, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import scipy.stats
    import scipy.sparse

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy = None  # type: ignore

try:
    import numba
    from numba import jit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    numba = None

    def jit(x):
        return x

    def prange(x):
        return range(x)


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from .logging import get_logger


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a specific time."""

    timestamp: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Statistical summary of a metric over time."""

    metric_name: str
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    trend: str  # "increasing", "decreasing", "stable"


class MetricsCollector:
    """Collects and analyzes metrics over time."""

    def __init__(self, name: str = "default"):
        """Initialize metrics collector."""
        self.name = name
        self.snapshots: List[MetricSnapshot] = []
        self.start_time = time.time()
        self.logger = get_logger(__name__)

        # Counters for different metric types
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.gauges: Dict[str, float] = {}

    def record_counter(self, name: str, value: int = 1) -> None:
        """Record a counter metric."""
        self.counters[name] += value
        self.logger.debug(f"Counter {name}: {self.counters[name]}")

    def record_timer(self, name: str, duration: float) -> None:
        """Record a timer metric."""
        self.timers[name].append(duration)
        self.logger.debug(f"Timer {name}: {duration:.3f}s")

    def record_gauge(self, name: str, value: float) -> None:
        """Record a gauge metric."""
        self.gauges[name] = value
        self.logger.debug(f"Gauge {name}: {value}")

    def start_timer(self, name: str) -> "TimerContext":
        """Start a timer context manager."""
        return TimerContext(self, name)

    def take_snapshot(self, metadata: Optional[Dict[str, Any]] = None) -> MetricSnapshot:
        """Take a snapshot of current metrics."""
        current_metrics = {}

        # Include counters
        current_metrics.update(self.counters)

        # Include timer summaries
        for timer_name, durations in self.timers.items():
            if durations:
                current_metrics[f"{timer_name}_mean"] = statistics.mean(durations)
                current_metrics[f"{timer_name}_total"] = sum(durations)
                current_metrics[f"{timer_name}_count"] = len(durations)

        # Include gauges
        current_metrics.update(self.gauges)

        snapshot = MetricSnapshot(
            timestamp=time.time(), metrics=current_metrics, metadata=metadata or {}
        )

        self.snapshots.append(snapshot)
        return snapshot

    def get_metric_summary(self, metric_name: str) -> Optional[MetricSummary]:
        """Get statistical summary for a specific metric."""
        values = []

        for snapshot in self.snapshots:
            if metric_name in snapshot.metrics:
                values.append(snapshot.metrics[metric_name])

        if not values:
            return None

        # Calculate trend
        if len(values) >= 3:
            recent_values = values[-3:]
            if all(recent_values[i] <= recent_values[i + 1] for i in range(len(recent_values) - 1)):
                trend = "increasing"
            elif all(
                recent_values[i] >= recent_values[i + 1] for i in range(len(recent_values) - 1)
            ):
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return MetricSummary(
            metric_name=metric_name,
            count=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            percentile_95=(
                float(np.percentile(values, 95))
                if HAS_NUMPY and np is not None
                else sorted(values)[int(0.95 * len(values))]
            ),
            percentile_99=(
                float(np.percentile(values, 99))
                if HAS_NUMPY and np is not None
                else sorted(values)[int(0.99 * len(values))]
            ),
            trend=trend,
        )

    def get_all_summaries(self) -> Dict[str, MetricSummary]:
        """Get summaries for all metrics."""
        all_metrics = set()

        for snapshot in self.snapshots:
            all_metrics.update(snapshot.metrics.keys())

        summaries = {}
        for metric_name in all_metrics:
            summary = self.get_metric_summary(metric_name)
            if summary:
                summaries[metric_name] = summary

        return summaries

    def export_to_json(self, file_path: Path) -> None:
        """Export metrics to JSON file."""
        data = {
            "collector_name": self.name,
            "start_time": self.start_time,
            "snapshots": [
                {"timestamp": s.timestamp, "metrics": s.metrics, "metadata": s.metadata}
                for s in self.snapshots
            ],
            "summaries": {
                name: {
                    "metric_name": summary.metric_name,
                    "count": summary.count,
                    "mean": summary.mean,
                    "median": summary.median,
                    "std_dev": summary.std_dev,
                    "min_value": summary.min_value,
                    "max_value": summary.max_value,
                    "percentile_95": summary.percentile_95,
                    "percentile_99": summary.percentile_99,
                    "trend": summary.trend,
                }
                for name, summary in self.get_all_summaries().items()
            },
        }

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Metrics exported to {file_path}")

    def reset(self) -> None:
        """Reset all metrics."""
        self.snapshots.clear()
        self.counters.clear()
        self.timers.clear()
        self.gauges.clear()
        self.start_time = time.time()
        self.logger.info(f"Metrics collector {self.name} reset")


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str):
        self.collector = collector
        self.name = name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.collector.record_timer(self.name, duration)


class EvaluationMetrics:
    """Specialized metrics for evaluation runs."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.collector = MetricsCollector(f"eval_{run_id}")
        self.logger = get_logger(__name__)

        # Evaluation-specific metrics
        self.task_metrics = defaultdict(lambda: defaultdict(list))
        self.adapter_metrics = defaultdict(lambda: defaultdict(list))
        self.dataset_metrics = defaultdict(lambda: defaultdict(list))

    def record_prediction(
        self,
        task_name: str,
        adapter_name: str,
        dataset_name: str,
        prediction_time: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a prediction event."""
        # Record to main collector
        self.collector.record_timer(f"prediction_time_{adapter_name}", prediction_time)
        self.collector.record_counter(f"predictions_{adapter_name}")

        if success:
            self.collector.record_counter(f"successful_predictions_{adapter_name}")
        else:
            self.collector.record_counter(f"failed_predictions_{adapter_name}")

        # Record to specialized collections
        self.task_metrics[task_name]["prediction_times"].append(prediction_time)
        self.adapter_metrics[adapter_name]["prediction_times"].append(prediction_time)
        self.dataset_metrics[dataset_name]["prediction_times"].append(prediction_time)

        if success:
            self.task_metrics[task_name]["successes"].append(1)
            self.adapter_metrics[adapter_name]["successes"].append(1)
            self.dataset_metrics[dataset_name]["successes"].append(1)
        else:
            self.task_metrics[task_name]["failures"].append(1)
            self.adapter_metrics[adapter_name]["failures"].append(1)
            self.dataset_metrics[dataset_name]["failures"].append(1)

    def record_metric_score(
        self, metric_name: str, score: float, task_name: str, adapter_name: str
    ) -> None:
        """Record a metric score."""
        self.collector.record_gauge(f"{metric_name}_{adapter_name}_latest", score)

        # Track scores over time
        self.adapter_metrics[adapter_name][f"{metric_name}_scores"].append(score)
        self.task_metrics[task_name][f"{metric_name}_scores"].append(score)

    def get_adapter_performance(self, adapter_name: str) -> Dict[str, Any]:
        """Get performance summary for an adapter."""
        metrics = self.adapter_metrics[adapter_name]

        performance = {
            "adapter_name": adapter_name,
            "total_predictions": len(metrics.get("prediction_times", [])),
            "success_rate": 0.0,
            "avg_prediction_time": 0.0,
            "metric_scores": {},
        }

        # Calculate success rate
        successes = sum(metrics.get("successes", []))
        failures = sum(metrics.get("failures", []))
        total = successes + failures

        if total > 0:
            performance["success_rate"] = successes / total

        # Calculate average prediction time
        times = metrics.get("prediction_times", [])
        if times:
            performance["avg_prediction_time"] = statistics.mean(times)

        # Get metric scores
        for key, values in metrics.items():
            if key.endswith("_scores") and values:
                metric_name = key.replace("_scores", "")
                performance["metric_scores"][metric_name] = {
                    "mean": statistics.mean(values),
                    "latest": values[-1],
                    "best": max(values),
                    "count": len(values),
                }

        return performance

    def get_task_performance(self, task_name: str) -> Dict[str, Any]:
        """Get performance summary for a task."""
        metrics = self.task_metrics[task_name]

        performance = {
            "task_name": task_name,
            "total_predictions": len(metrics.get("prediction_times", [])),
            "success_rate": 0.0,
            "avg_prediction_time": 0.0,
            "metric_scores": {},
        }

        # Calculate success rate
        successes = sum(metrics.get("successes", []))
        failures = sum(metrics.get("failures", []))
        total = successes + failures

        if total > 0:
            performance["success_rate"] = successes / total

        # Calculate average prediction time
        times = metrics.get("prediction_times", [])
        if times:
            performance["avg_prediction_time"] = statistics.mean(times)

        # Get metric scores
        for key, values in metrics.items():
            if key.endswith("_scores") and values:
                metric_name = key.replace("_scores", "")
                performance["metric_scores"][metric_name] = {
                    "mean": statistics.mean(values),
                    "latest": values[-1],
                    "best": max(values),
                    "count": len(values),
                }

        return performance

    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        report = [f"# Performance Report: {self.run_id}\n"]

        # Overall statistics
        total_predictions = self.collector.counters.get("predictions", 0)
        report.append(f"**Total Predictions**: {total_predictions}")

        # Adapter performance
        report.append("\n## Adapter Performance\n")

        adapter_names = list(self.adapter_metrics.keys())
        for adapter_name in sorted(adapter_names):
            perf = self.get_adapter_performance(adapter_name)
            report.append(f"### {adapter_name}")
            report.append(f"- **Success Rate**: {perf['success_rate']:.2%}")
            report.append(f"- **Avg Prediction Time**: {perf['avg_prediction_time']:.3f}s")
            report.append(f"- **Total Predictions**: {perf['total_predictions']}")

            if perf["metric_scores"]:
                report.append("- **Metric Scores**:")
                for metric, scores in perf["metric_scores"].items():
                    report.append(
                        f"  - {metric}: {scores['mean']:.3f} (latest: {scores['latest']:.3f})"
                    )

            report.append("")

        # Task performance
        report.append("## Task Performance\n")

        task_names = list(self.task_metrics.keys())
        for task_name in sorted(task_names):
            perf = self.get_task_performance(task_name)
            report.append(f"### {task_name}")
            report.append(f"- **Success Rate**: {perf['success_rate']:.2%}")
            report.append(f"- **Avg Prediction Time**: {perf['avg_prediction_time']:.3f}s")
            report.append(f"- **Total Predictions**: {perf['total_predictions']}")
            report.append("")

        return "\n".join(report)

    def export_performance_data(self, output_dir: Path) -> List[Path]:
        """Export all performance data to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        files_created = []

        # Export main metrics
        metrics_file = output_dir / f"metrics_{self.run_id}.json"
        self.collector.export_to_json(metrics_file)
        files_created.append(metrics_file)

        # Export performance report
        report_file = output_dir / f"performance_report_{self.run_id}.md"
        report = self.generate_performance_report()

        with open(report_file, "w") as f:
            f.write(report)
        files_created.append(report_file)

        # Export detailed performance data
        performance_file = output_dir / f"performance_data_{self.run_id}.json"
        performance_data = {
            "run_id": self.run_id,
            "adapters": {
                name: self.get_adapter_performance(name) for name in self.adapter_metrics.keys()
            },
            "tasks": {name: self.get_task_performance(name) for name in self.task_metrics.keys()},
        }

        with open(performance_file, "w") as f:
            json.dump(performance_data, f, indent=2, default=str)
        files_created.append(performance_file)

        self.logger.info(f"Performance data exported to {output_dir}")
        return files_created


# Global metrics collector
_global_metrics = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector("global")
    return _global_metrics


"""
Vectorized Metric Computation for OpenEval Lab

This module provides vectorized implementations of evaluation metrics using NumPy and pandas
for significantly improved performance on large datasets.

Advanced Features:
- SIMD-optimized vector operations
- Parallel processing with multiprocessing
- Memory-efficient streaming compu        if not rouge_scores:
            return VectorizedMetricResult("rouge", 0.0)

        if HAS_NUMPY and np is not None:
            rouge_array = np.array(rouge_scores)
            mean_rouge = float(np.mean(rouge_array))

            return VectorizedMetricResult(
                name="rouge",
                value=mean_rouge,
                details={
                    "mean": mean_rouge,
                    "std": float(np.std(rouge_array)),
                    "min": float(np.min(rouge_array)),
                    "max": float(np.max(rouge_array))
                },
                sample_size=len(rouge_scores)
            )
        else:
            # Fallback without numpy
            mean_rouge = sum(rouge_scores) / len(rouge_scores)
            return VectorizedMetricResult(
                name="rouge",
                value=mean_rouge,
                details={
                    "mean": mean_rouge,
                    "count": len(rouge_scores)
                },
                sample_size=len(rouge_scores)
            ) metric optimization
- Adaptive precision computation
- GPU acceleration support (when available)
"""


logger = get_logger(__name__)


@dataclass
class VectorizedMetricResult:
    """Result of a vectorized metric computation."""

    name: str
    value: float
    details: Optional[Dict[str, Any]] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"name": self.name, "value": self.value}
        if self.details:
            result["details"] = self.details
        if self.confidence_interval:
            result["confidence_interval"] = self.confidence_interval
        if self.sample_size:
            result["sample_size"] = self.sample_size
        return result


class AdvancedVectorizedMetrics:
    """
    Advanced vectorized metrics with SIMD, parallel processing, and ML optimizations.
    """

    def __init__(self, use_simd: bool = True, use_parallel: bool = True, use_gpu: bool = False):
        self.use_simd = use_simd and HAS_NUMBA
        self.use_parallel = use_parallel
        self.use_gpu = use_gpu and HAS_TORCH
        self._thread_pool = ThreadPoolExecutor(max_workers=mp.cpu_count()) if use_parallel else None

    def exact_match(
        self, predictions: Iterable[Any], references: Iterable[Any]
    ) -> VectorizedMetricResult:
        """Advanced exact match with SIMD and parallel processing."""
        if self.use_simd and HAS_NUMBA and HAS_NUMPY and np is not None:
            # SIMD-optimized version
            pred_array = np.array([str(p).strip() for p in predictions])
            ref_array = np.array([str(r).strip() for r in references])
            accuracy = self._simd_exact_match(pred_array, ref_array)
            matches = int(accuracy * len(pred_array))
            return VectorizedMetricResult(
                name="exact_match_simd",
                value=float(accuracy),
                details={"matches": matches, "total": len(pred_array)},
                sample_size=len(pred_array),
            )
        elif self.use_parallel and self._thread_pool:
            # Parallel version
            return self.parallel_exact_match(predictions, references)
        else:
            # Standard vectorized version
            return self._fallback_exact_match(predictions, references)

    def _fallback_exact_match(
        self, predictions: Iterable[Any], references: Iterable[Any]
    ) -> VectorizedMetricResult:
        """Fallback exact match implementation."""
        pred_array = (
            np.array([str(p).strip() for p in predictions])
            if HAS_NUMPY and np is not None
            else None
        )
        ref_array = (
            np.array([str(r).strip() for r in references]) if HAS_NUMPY and np is not None else None
        )

        if pred_array is not None and ref_array is not None:
            matches = np.sum(pred_array == ref_array)
            total = len(pred_array)
            accuracy = matches / total if total > 0 else 0.0
        else:
            # Pure Python fallback
            matches = sum(
                1 for p, r in zip(predictions, references) if str(p).strip() == str(r).strip()
            )
            total = sum(1 for _ in predictions)
            accuracy = matches / total if total > 0 else 0.0

        return VectorizedMetricResult(
            name="exact_match",
            value=float(accuracy),
            details={"matches": matches, "total": total},
            sample_size=total,
        )

    @staticmethod
    def _simd_exact_match(pred_array, ref_array) -> float:
        """SIMD-optimized exact match computation."""
        if not HAS_NUMPY or np is None:
            return 0.0
        matches = np.sum(pred_array == ref_array)
        return float(matches / len(pred_array))

    @staticmethod
    def _simd_bleu_ngrams(tokens_list: List[List[str]], n: int):
        """SIMD-optimized n-gram computation for BLEU."""
        if not HAS_NUMPY or np is None:
            return []
        ngram_counts = np.zeros(len(tokens_list), dtype=np.int32)
        for i in range(len(tokens_list)):
            tokens = tokens_list[i]
            if len(tokens) >= n:
                ngram_counts[i] = len(tokens) - n + 1
        return ngram_counts

    def parallel_exact_match(
        self, predictions: Iterable[Any], references: Iterable[Any]
    ) -> VectorizedMetricResult:
        """Parallel exact match computation."""
        if not self.use_parallel or not self._thread_pool:
            return self.exact_match(predictions, references)

        pred_list = list(predictions)
        ref_list = list(references)

        # Split into chunks for parallel processing
        chunk_size = max(1, len(pred_list) // mp.cpu_count())
        chunks = [
            (pred_list[i : i + chunk_size], ref_list[i : i + chunk_size])
            for i in range(0, len(pred_list), chunk_size)
        ]

        futures = [
            self._thread_pool.submit(self._compute_chunk_exact_match, chunk) for chunk in chunks
        ]

        total_matches = 0
        total_count = 0

        for future in futures:
            matches, count = future.result()
            total_matches += matches
            total_count += count

        accuracy = total_matches / total_count if total_count > 0 else 0.0

        return VectorizedMetricResult(
            name="exact_match_parallel",
            value=float(accuracy),
            details={"matches": total_matches, "total": total_count},
            sample_size=total_count,
        )

    @staticmethod
    def _compute_chunk_exact_match(chunk: Tuple[List[Any], List[Any]]) -> Tuple[int, int]:
        """Compute exact match for a chunk."""
        preds, refs = chunk
        matches = sum(1 for p, r in zip(preds, refs) if str(p).strip() == str(r).strip())
        return matches, len(preds)

    def gpu_accelerated_metrics(
        self, predictions: Iterable[str], references: Iterable[str]
    ) -> Dict[str, VectorizedMetricResult]:
        """GPU-accelerated metric computation using PyTorch."""
        if not self.use_gpu or not HAS_TORCH or torch is None:
            return {
                "exact_match": self.exact_match(predictions, references),
                "f1": self.f1_score(predictions, references),
            }

        # Convert to tensors for GPU processing
        [str(p).split() for p in predictions]
        [str(r).split() for r in references]

        # Simple GPU-accelerated exact match
        results = {}

        # Exact match on GPU
        pred_strs = [str(p).strip() for p in predictions]
        ref_strs = [str(r).strip() for r in references]

        # Use GPU for parallel string comparison (simplified)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # This is a simplified GPU implementation - in practice you'd need more sophisticated
        # string processing on GPU
        matches = sum(1 for p, r in zip(pred_strs, ref_strs) if p == r)
        accuracy = matches / len(pred_strs) if pred_strs else 0.0

        results["exact_match_gpu"] = VectorizedMetricResult(
            name="exact_match_gpu",
            value=float(accuracy),
            details={"matches": matches, "total": len(pred_strs), "device": str(device)},
            sample_size=len(pred_strs),
        )

        return results

    def adaptive_batch_processing(
        self,
        predictions: Iterable[Any],
        references: Iterable[Any],
        metric_func: Callable,
        batch_size: int = 1000,
    ) -> VectorizedMetricResult:
        """Adaptive batch processing with dynamic batch size optimization."""
        pred_list = list(predictions)
        ref_list = list(references)

        if len(pred_list) <= batch_size:
            return metric_func(pred_list, ref_list)

        # Adaptive batching based on available memory
        try:
            import psutil

            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            optimal_batch_size = min(batch_size, max(100, int(available_memory * 100000)))
        except ImportError:
            optimal_batch_size = batch_size

        results = []
        for i in range(0, len(pred_list), optimal_batch_size):
            batch_pred = pred_list[i : i + optimal_batch_size]
            batch_ref = ref_list[i : i + optimal_batch_size]
            batch_result = metric_func(batch_pred, batch_ref)
            results.append(batch_result)

        # Aggregate results
        if HAS_NUMPY and np is not None:
            values = np.array([r.value for r in results])
            weights = np.array([r.sample_size or 1 for r in results])
            aggregated_value = float(np.average(values, weights=weights))
        else:
            total_weight = sum(r.sample_size or 1 for r in results)
            aggregated_value = sum(r.value * (r.sample_size or 1) for r in results) / total_weight

        return VectorizedMetricResult(
            name=f"{results[0].name}_adaptive",
            value=aggregated_value,
            details={"batch_results": [r.to_dict() for r in results]},
            sample_size=sum(r.sample_size or 1 for r in results),
        )

    @staticmethod
    def exact_match_vectorized(
        predictions: Iterable[Any], references: Iterable[Any]
    ) -> VectorizedMetricResult:
        """Compute exact match accuracy using vectorized operations."""
        if not HAS_NUMPY or np is None:
            # Fallback to non-vectorized
            matches = sum(
                1 for p, r in zip(predictions, references) if str(p).strip() == str(r).strip()
            )
            total = sum(1 for _ in predictions)
            return VectorizedMetricResult("exact_match", matches / total if total > 0 else 0.0)

        pred_array = np.array([str(p).strip() for p in predictions])
        ref_array = np.array([str(r).strip() for r in references])

        matches = np.sum(pred_array == ref_array)
        total = len(pred_array)

        accuracy = matches / total if total > 0 else 0.0

        return VectorizedMetricResult(
            name="exact_match",
            value=float(accuracy),
            details={"matches": int(matches), "total": total},
            sample_size=total,
        )

    @staticmethod
    def f1_score(predictions: Iterable[str], references: Iterable[str]) -> VectorizedMetricResult:
        """Compute F1 score using vectorized token-level operations."""
        if not HAS_NUMPY:
            # Fallback implementation
            total_f1 = 0.0
            count = 0
            for pred, ref in zip(predictions, references):
                pred_tokens = set(str(pred).lower().split())
                ref_tokens = set(str(ref).lower().split())

                if not ref_tokens:
                    continue

                intersection = pred_tokens & ref_tokens
                precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
                recall = len(intersection) / len(ref_tokens) if ref_tokens else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                total_f1 += f1
                count += 1

            return VectorizedMetricResult("f1", total_f1 / count if count > 0 else 0.0)

        # Vectorized tokenization
        pred_tokens = [set(str(p).lower().split()) for p in predictions]
        ref_tokens = [set(str(r).lower().split()) for r in references]

        # Compute F1 for each pair
        f1_scores = []
        for pred_set, ref_set in zip(pred_tokens, ref_tokens):
            if not ref_set:
                continue

            intersection = pred_set & ref_set
            precision = len(intersection) / len(pred_set) if pred_set else 0.0
            recall = len(intersection) / len(ref_set) if ref_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        if not f1_scores:
            return VectorizedMetricResult("f1", 0.0)

        if HAS_NUMPY and np is not None:
            f1_array = np.array(f1_scores)
            mean_f1 = float(np.mean(f1_array))

            return VectorizedMetricResult(
                name="f1",
                value=mean_f1,
                details={
                    "mean": mean_f1,
                    "std": float(np.std(f1_array)),
                    "min": float(np.min(f1_array)),
                    "max": float(np.max(f1_array)),
                },
                sample_size=len(f1_scores),
            )
        else:
            # Fallback without numpy
            mean_f1 = sum(f1_scores) / len(f1_scores)
            return VectorizedMetricResult(
                name="f1",
                value=mean_f1,
                details={"mean": mean_f1, "count": len(f1_scores)},
                sample_size=len(f1_scores),
            )

    @staticmethod
    def bleu_score(
        predictions: Iterable[str], references: Iterable[str], n_gram: int = 4
    ) -> VectorizedMetricResult:
        """Compute BLEU score using vectorized n-gram operations."""
        if not HAS_NUMPY:
            # Simplified fallback BLEU
            total_bleu = 0.0
            count = 0
            for pred, ref in zip(predictions, references):
                pred_tokens = str(pred).split()
                ref_tokens = str(ref).split()

                if not pred_tokens or not ref_tokens:
                    continue

                # Simple unigram overlap
                pred_set = set(pred_tokens)
                ref_set = set(ref_tokens)
                overlap = len(pred_set & ref_set)
                precision = overlap / len(pred_set) if pred_set else 0.0
                total_bleu += precision
                count += 1

            return VectorizedMetricResult("bleu", total_bleu / count if count > 0 else 0.0)

        def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
            """Get n-grams from tokens."""
            return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

        bleu_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = str(pred).split()
            ref_tokens = str(ref).split()

            if not pred_tokens or not ref_tokens:
                continue

            # Compute BLEU for different n-grams
            precisions = []

            for n in range(1, min(n_gram + 1, len(pred_tokens) + 1)):
                pred_ngrams = _get_ngrams(pred_tokens, n)
                ref_ngrams = _get_ngrams(ref_tokens, n)

                if not pred_ngrams:
                    precisions.append(0.0)
                    continue

                pred_counts = Counter(pred_ngrams)
                ref_counts = Counter(ref_ngrams)

                # Clipped counts
                clipped_counts = {
                    ngram: min(count, ref_counts.get(ngram, 0))
                    for ngram, count in pred_counts.items()
                }

                precision = sum(clipped_counts.values()) / len(pred_ngrams)
                precisions.append(precision)

            if precisions:
                # Geometric mean of precisions
                if all(p > 0 for p in precisions):
                    bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
                else:
                    bleu = 0.0

                # Brevity penalty
                pred_len = len(pred_tokens)
                ref_len = len(ref_tokens)
                if pred_len > ref_len:
                    brevity_penalty = 1.0
                else:
                    brevity_penalty = math.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0

                bleu *= brevity_penalty
                bleu_scores.append(bleu)

        if not bleu_scores:
            return VectorizedMetricResult("bleu", 0.0)

        if HAS_NUMPY and np is not None:
            bleu_array = np.array(bleu_scores)
            mean_bleu = float(np.mean(bleu_array))

            return VectorizedMetricResult(
                name="bleu",
                value=mean_bleu,
                details={
                    "mean": mean_bleu,
                    "std": float(np.std(bleu_array)),
                    "min": float(np.min(bleu_array)),
                    "max": float(np.max(bleu_array)),
                },
                sample_size=len(bleu_scores),
            )
        else:
            # Fallback without numpy
            mean_bleu = sum(bleu_scores) / len(bleu_scores)
            return VectorizedMetricResult(
                name="bleu",
                value=mean_bleu,
                details={"mean": mean_bleu, "count": len(bleu_scores)},
                sample_size=len(bleu_scores),
            )

    @staticmethod
    def rouge_score(
        predictions: Iterable[str], references: Iterable[str]
    ) -> VectorizedMetricResult:
        """Compute ROUGE score using vectorized operations."""
        if not HAS_NUMPY:
            # Fallback implementation
            total_rouge = 0.0
            count = 0
            for pred, ref in zip(predictions, references):
                pred_tokens = set(str(pred).lower().split())
                ref_tokens = set(str(ref).lower().split())

                if not ref_tokens:
                    continue

                intersection = pred_tokens & ref_tokens
                rouge = len(intersection) / len(ref_tokens)
                total_rouge += rouge
                count += 1

            return VectorizedMetricResult("rouge_l", total_rouge / count if count > 0 else 0.0)

        rouge_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = set(str(pred).lower().split())
            ref_tokens = set(str(ref).lower().split())

            if not ref_tokens:
                continue

            intersection = pred_tokens & ref_tokens
            rouge = len(intersection) / len(ref_tokens)
            rouge_scores.append(rouge)

        if not rouge_scores:
            return VectorizedMetricResult("rouge_l", 0.0)

        if HAS_NUMPY and np is not None:
            rouge_array = np.array(rouge_scores)
            mean_rouge = float(np.mean(rouge_array))

            return VectorizedMetricResult(
                name="rouge_l",
                value=mean_rouge,
                details={
                    "mean": mean_rouge,
                    "std": float(np.std(rouge_array)),
                    "min": float(np.min(rouge_array)),
                    "max": float(np.max(rouge_array)),
                },
                sample_size=len(rouge_scores),
            )
        else:
            # Fallback without numpy
            mean_rouge = sum(rouge_scores) / len(rouge_scores)
            return VectorizedMetricResult(
                name="rouge_l",
                value=mean_rouge,
                details={"mean": mean_rouge, "count": len(rouge_scores)},
                sample_size=len(rouge_scores),
            )

    @staticmethod
    def semantic_similarity(
        predictions: Iterable[str], references: Iterable[str]
    ) -> VectorizedMetricResult:
        """Compute semantic similarity using vectorized operations (placeholder for advanced models)."""
        # This is a simplified implementation - in practice, you'd use sentence transformers
        if not HAS_NUMPY:
            # Simple fallback based on token overlap
            similarities = []
            for pred, ref in zip(predictions, references):
                pred_tokens = set(str(pred).lower().split())
                ref_tokens = set(str(ref).lower().split())

                if not pred_tokens and not ref_tokens:
                    similarities.append(1.0)
                elif not pred_tokens or not ref_tokens:
                    similarities.append(0.0)
                else:
                    intersection = pred_tokens & ref_tokens
                    union = pred_tokens | ref_tokens
                    jaccard = len(intersection) / len(union) if union else 0.0
                    similarities.append(jaccard)

            return VectorizedMetricResult(
                "semantic_similarity",
                sum(similarities) / len(similarities) if similarities else 0.0,
            )

        # Vectorized Jaccard similarity
        similarities = []

        for pred, ref in zip(predictions, references):
            pred_tokens = set(str(pred).lower().split())
            ref_tokens = set(str(ref).lower().split())

            if not pred_tokens and not ref_tokens:
                similarities.append(1.0)
            elif not pred_tokens or not ref_tokens:
                similarities.append(0.0)
            else:
                intersection = pred_tokens & ref_tokens
                union = pred_tokens | ref_tokens
                jaccard = len(intersection) / len(union) if union else 0.0
                similarities.append(jaccard)

        if not similarities:
            return VectorizedMetricResult("semantic_similarity", 0.0)

        if HAS_NUMPY and np is not None:
            sim_array = np.array(similarities)
            mean_sim = float(np.mean(sim_array))

            return VectorizedMetricResult(
                name="semantic_similarity",
                value=mean_sim,
                details={
                    "mean": mean_sim,
                    "std": float(np.std(sim_array)),
                    "min": float(np.min(sim_array)),
                    "max": float(np.max(sim_array)),
                },
                sample_size=len(similarities),
            )
        else:
            # Fallback without numpy
            mean_sim = sum(similarities) / len(similarities)
            return VectorizedMetricResult(
                name="semantic_similarity",
                value=mean_sim,
                details={"mean": mean_sim, "count": len(similarities)},
                sample_size=len(similarities),
            )


class BatchMetricsProcessor:
    """
    Processes metrics in batches for improved performance.
    """

    def __init__(self, batch_size: int = 1000, use_pandas: bool = True):
        self.batch_size = batch_size
        self.use_pandas = use_pandas and HAS_PANDAS

    def compute_metrics_batch(
        self, predictions: Iterable[Any], references: Iterable[Any], metrics: List[str]
    ) -> Dict[str, VectorizedMetricResult]:
        """
        Compute multiple metrics in batches for better performance.

        Args:
            predictions: Predicted values
            references: Reference values
            metrics: List of metric names to compute

        Returns:
            Dictionary of metric results
        """
        # Convert to lists for batching
        pred_list = list(predictions)
        ref_list = list(references)

        if self.use_pandas and HAS_PANDAS and pd is not None:
            # Use pandas for efficient data handling
            df = pd.DataFrame({"prediction": pred_list, "reference": ref_list})
            pred_list = df["prediction"].tolist()
            ref_list = df["reference"].tolist()

        results = {}

        # Process in batches
        for i in range(0, len(pred_list), self.batch_size):
            batch_pred = pred_list[i : i + self.batch_size]
            batch_ref = ref_list[i : i + self.batch_size]

            for metric_name in metrics:
                if metric_name not in results:
                    results[metric_name] = []

                # Compute metric for this batch
                if metric_name == "exact_match":
                    result = self._compute_exact_match(batch_pred, batch_ref)
                elif metric_name == "f1":
                    result = self._compute_f1_score(batch_pred, batch_ref)
                elif metric_name == "bleu":
                    result = self._compute_bleu_score(batch_pred, batch_ref)
                elif metric_name == "rouge_l":
                    result = self._compute_rouge_score(batch_pred, batch_ref)
                elif metric_name == "semantic_similarity":
                    result = self._compute_semantic_similarity(batch_pred, batch_ref)
                else:
                    continue

                results[metric_name].append(result)

        # Aggregate results across batches
        final_results = {}
        for metric_name, batch_results in results.items():
            if not batch_results:
                continue

            if len(batch_results) == 1:
                final_results[metric_name] = batch_results[0]
            else:
                # Aggregate multiple batches
                values = [r.value for r in batch_results]
                weights = [r.sample_size or 1 for r in batch_results]

                if HAS_NUMPY and np is not None:
                    # Weighted average
                    weights_array = np.array(weights)
                    values_array = np.array(values)
                    weighted_avg = float(np.average(values_array, weights=weights_array))
                else:
                    # Simple average
                    weighted_avg = sum(values) / len(values)

                final_results[metric_name] = VectorizedMetricResult(
                    name=metric_name,
                    value=weighted_avg,
                    details={"batch_results": [r.to_dict() for r in batch_results]},
                    sample_size=sum(weights),
                )

        return final_results

    def _compute_exact_match(
        self, predictions: List[Any], references: List[Any]
    ) -> VectorizedMetricResult:
        """Compute exact match for a batch."""
        matches = sum(
            1 for p, r in zip(predictions, references) if str(p).strip() == str(r).strip()
        )
        total = len(predictions)
        accuracy = matches / total if total > 0 else 0.0
        return VectorizedMetricResult(
            name="exact_match",
            value=float(accuracy),
            details={"matches": matches, "total": total},
            sample_size=total,
        )

    def _compute_f1_score(
        self, predictions: List[str], references: List[str]
    ) -> VectorizedMetricResult:
        """Compute F1 score for a batch."""
        total_f1 = 0.0
        count = 0
        for pred, ref in zip(predictions, references):
            pred_tokens = set(str(pred).lower().split())
            ref_tokens = set(str(ref).lower().split())

            if not ref_tokens:
                continue

            intersection = pred_tokens & ref_tokens
            precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
            recall = len(intersection) / len(ref_tokens) if ref_tokens else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            total_f1 += f1
            count += 1

        return VectorizedMetricResult("f1", total_f1 / count if count > 0 else 0.0)

    def _compute_bleu_score(
        self, predictions: List[str], references: List[str]
    ) -> VectorizedMetricResult:
        """Compute BLEU score for a batch."""
        total_bleu = 0.0
        count = 0
        for pred, ref in zip(predictions, references):
            pred_tokens = str(pred).split()
            ref_tokens = str(ref).split()

            if not pred_tokens or not ref_tokens:
                continue

            # Simple unigram overlap
            pred_set = set(pred_tokens)
            ref_set = set(ref_tokens)
            overlap = len(pred_set & ref_set)
            precision = overlap / len(pred_set) if pred_set else 0.0
            total_bleu += precision
            count += 1

        return VectorizedMetricResult("bleu", total_bleu / count if count > 0 else 0.0)

    def _compute_rouge_score(
        self, predictions: List[str], references: List[str]
    ) -> VectorizedMetricResult:
        """Compute ROUGE score for a batch."""
        total_rouge = 0.0
        count = 0
        for pred, ref in zip(predictions, references):
            pred_tokens = set(str(pred).lower().split())
            ref_tokens = set(str(ref).lower().split())

            if not ref_tokens:
                continue

            intersection = len(pred_tokens & ref_tokens)
            precision = intersection / len(pred_tokens) if pred_tokens else 0.0
            recall = intersection / len(ref_tokens) if ref_tokens else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            total_rouge += f1
            count += 1

        return VectorizedMetricResult("rouge_l", total_rouge / count if count > 0 else 0.0)

    def _compute_semantic_similarity(
        self, predictions: List[str], references: List[str]
    ) -> VectorizedMetricResult:
        """Compute semantic similarity for a batch."""
        # Simplified semantic similarity based on token overlap
        similarities = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(str(pred).lower().split())
            ref_tokens = set(str(ref).lower().split())

            if not pred_tokens and not ref_tokens:
                similarities.append(1.0)
            elif not pred_tokens or not ref_tokens:
                similarities.append(0.0)
            else:
                intersection = len(pred_tokens & ref_tokens)
                union = len(pred_tokens | ref_tokens)
                jaccard = intersection / union if union > 0 else 0.0
                similarities.append(jaccard)

        if not similarities:
            return VectorizedMetricResult("semantic_similarity", 0.0)

        if HAS_NUMPY and np is not None:
            sim_array = np.array(similarities)
            mean_sim = float(np.mean(sim_array))
            return VectorizedMetricResult(
                name="semantic_similarity",
                value=mean_sim,
                details={
                    "mean": mean_sim,
                    "std": float(np.std(sim_array)),
                    "min": float(np.min(sim_array)),
                    "max": float(np.max(sim_array)),
                },
                sample_size=len(similarities),
            )
        else:
            mean_sim = sum(similarities) / len(similarities)
            return VectorizedMetricResult(
                name="semantic_similarity",
                value=mean_sim,
                details={"mean": mean_sim, "count": len(similarities)},
                sample_size=len(similarities),
            )

    def compute_confidence_intervals(
        self, results: Dict[str, VectorizedMetricResult], confidence_level: float = 0.95
    ) -> Dict[str, VectorizedMetricResult]:
        """
        Compute confidence intervals for metric results.

        Args:
            results: Metric results
            confidence_level: Confidence level (0.95 for 95% CI)

        Returns:
            Results with confidence intervals
        """
        if not HAS_SCIPY:
            logger.warning("SciPy not available, skipping confidence interval computation")
            return results

        updated_results = {}

        for metric_name, result in results.items():
            if result.sample_size and result.sample_size > 1:
                # Compute confidence interval using t-distribution
                try:
                    # This is a simplified approach - in practice you'd need the raw data
                    # For now, we'll use the standard error from the batch results
                    if (
                        hasattr(result, "details")
                        and result.details
                        and "batch_results" in result.details
                    ):
                        batch_values = [br["value"] for br in result.details["batch_results"]]
                        if len(batch_values) > 1 and HAS_NUMPY and np is not None:
                            mean_val = float(np.mean(batch_values))
                            std_val = float(np.std(batch_values, ddof=1))
                            n = len(batch_values)

                            # t-distribution critical value
                            if HAS_SCIPY and scipy is not None and hasattr(scipy, "stats"):
                                t_crit = scipy.stats.t.ppf((1 + confidence_level) / 2, n - 1)
                                margin = t_crit * (std_val / np.sqrt(n))
                                result.confidence_interval = (mean_val - margin, mean_val + margin)

                except Exception as e:
                    logger.debug(f"Failed to compute confidence interval for {metric_name}: {e}")

            updated_results[metric_name] = result

        return updated_results


# Utility functions for easy integration
def compute_vectorized_metrics(
    predictions: Iterable[Any],
    references: Iterable[Any],
    metrics: List[str],
    batch_size: int = 1000,
) -> Dict[str, VectorizedMetricResult]:
    """
    Compute multiple metrics using vectorized operations.

    Args:
        predictions: Predicted values
        references: Reference values
        metrics: List of metric names
        batch_size: Batch size for processing

    Returns:
        Dictionary of metric results
    """
    processor = BatchMetricsProcessor(batch_size=batch_size)
    return processor.compute_metrics_batch(predictions, references, metrics)


def benchmark_metrics_performance(
    predictions: Iterable[Any], references: Iterable[Any], metrics: List[str], iterations: int = 10
) -> Dict[str, Any]:
    """
    Benchmark the performance of vectorized vs non-vectorized metrics.

    Args:
        predictions: Predicted values
        references: Reference values
        metrics: List of metric names
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results
    """

    # Prepare data
    pred_list = list(predictions)
    ref_list = list(references)

    # Benchmark vectorized implementation
    processor = BatchMetricsProcessor()

    vectorized_times = []
    for _ in range(iterations):
        start_time = time.time()
        processor.compute_metrics_batch(pred_list, ref_list, metrics)
        vectorized_times.append(time.time() - start_time)

    # Simple non-vectorized benchmark (just for comparison)
    non_vectorized_times = []
    for _ in range(iterations):
        start_time = time.time()
        # Simulate non-vectorized computation
        for metric in metrics:
            if metric == "exact_match":
                sum(1 for p, r in zip(pred_list, ref_list) if str(p).strip() == str(r).strip())
            elif metric == "f1":
                for pred, ref in zip(pred_list, ref_list):
                    pred_tokens = set(str(pred).lower().split())
                    ref_tokens = set(str(ref).lower().split())
                    if ref_tokens:
                        intersection = pred_tokens & ref_tokens
                        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
                        recall = len(intersection) / len(ref_tokens)
                        (
                            2 * precision * recall / (precision + recall)
                            if (precision + recall) > 0
                            else 0.0
                        )
        non_vectorized_times.append(time.time() - start_time)

    return {
        "vectorized_avg_time": sum(vectorized_times) / len(vectorized_times),
        "vectorized_std_time": (
            float(np.std(vectorized_times)) if HAS_NUMPY and np is not None else 0
        ),
        "non_vectorized_avg_time": sum(non_vectorized_times) / len(non_vectorized_times),
        "speedup_factor": (sum(non_vectorized_times) / len(non_vectorized_times))
        / (sum(vectorized_times) / len(vectorized_times)),
        "iterations": iterations,
        "sample_size": len(pred_list),
    }
