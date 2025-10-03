"""Advanced metrics collection and analysis for OpenEval Lab."""

import time
import json
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import numpy as np

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
        self.logger = get_logger()

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
            percentile_95=float(np.percentile(values, 95)),
            percentile_99=float(np.percentile(values, 99)),
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
        self.logger = get_logger()

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
