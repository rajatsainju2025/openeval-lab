"""Metrics aggregation for combining results across multiple evaluation runs.

This module provides utilities for aggregating metrics from multiple evaluation
runs, supporting weighted averaging, confidence intervals, and statistical
significance testing.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import statistics


@dataclass
class MetricRun:
    """Represents a single evaluation run with metrics."""

    run_id: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    timestamp: Optional[str] = None


@dataclass
class AggregatedMetric:
    """Aggregated statistics for a single metric across runs."""

    metric_name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    confidence_interval: Tuple[float, float]
    num_runs: int
    values: List[float] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Result of comparing metrics between two runs or groups."""

    metric_name: str
    baseline_mean: float
    comparison_mean: float
    difference: float
    percent_change: float
    p_value: Optional[float] = None
    is_significant: bool = False
    effect_size: Optional[float] = None


class MetricsAggregator:
    """Aggregate metrics across multiple evaluation runs."""

    def __init__(
        self,
        confidence_level: float = 0.95,
        min_runs_for_ci: int = 3,
        significance_threshold: float = 0.05,
    ):
        """Initialize metrics aggregator.

        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
            min_runs_for_ci: Minimum runs needed to compute confidence intervals
            significance_threshold: P-value threshold for statistical significance
        """
        self.confidence_level = confidence_level
        self.min_runs_for_ci = min_runs_for_ci
        self.significance_threshold = significance_threshold
        self.runs: List[MetricRun] = []

    def add_run(
        self,
        run_id: str,
        metrics: Dict[str, float],
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Add a single evaluation run.

        Args:
            run_id: Unique identifier for the run
            metrics: Dictionary of metric name to value
            weight: Weight for weighted averaging (default: 1.0)
            metadata: Optional metadata about the run
            timestamp: Optional timestamp string
        """
        self.runs.append(
            MetricRun(
                run_id=run_id,
                metrics=metrics,
                weight=weight,
                metadata=metadata or {},
                timestamp=timestamp,
            )
        )

    def load_from_files(self, result_files: List[Union[str, Path]]) -> None:
        """Load evaluation results from JSON files.

        Args:
            result_files: List of paths to result JSON files
        """
        for file_path in result_files:
            file_path = Path(file_path)
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract metrics from various result formats
            if "metrics" in data:
                metrics = data["metrics"]
            elif "results" in data and isinstance(data["results"], dict):
                metrics = data["results"]
            else:
                # Assume the whole file is metrics
                metrics = {k: v for k, v in data.items() if isinstance(v, (int, float))}

            run_id = data.get("run_id", file_path.stem)
            metadata = {k: v for k, v in data.items() if k not in ["metrics", "results"]}

            self.add_run(run_id=run_id, metrics=metrics, metadata=metadata)

    def aggregate(
        self, metric_names: Optional[List[str]] = None, weighted: bool = False
    ) -> Dict[str, AggregatedMetric]:
        """Aggregate metrics across all runs.

        Args:
            metric_names: List of metric names to aggregate (None = all metrics)
            weighted: Whether to use weighted averaging

        Returns:
            Dictionary mapping metric names to aggregated statistics
        """
        if not self.runs:
            return {}

        # Collect all metric names if not specified
        if metric_names is None:
            all_names = set()
            for run in self.runs:
                all_names.update(run.metrics.keys())
            metric_names = sorted(all_names)

        aggregated = {}
        for metric_name in metric_names:
            values = []
            weights = []

            for run in self.runs:
                if metric_name in run.metrics:
                    values.append(run.metrics[metric_name])
                    weights.append(run.weight)

            if not values:
                continue

            # Compute aggregated statistics
            if weighted and sum(weights) > 0:
                mean_val = sum(v * w for v, w in zip(values, weights)) / sum(weights)
            else:
                mean_val = statistics.mean(values)

            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            median_val = statistics.median(values)
            min_val = min(values)
            max_val = max(values)

            # Compute confidence interval
            ci = self._compute_confidence_interval(values)

            aggregated[metric_name] = AggregatedMetric(
                metric_name=metric_name,
                mean=mean_val,
                std=std_val,
                min=min_val,
                max=max_val,
                median=median_val,
                confidence_interval=ci,
                num_runs=len(values),
                values=values,
            )

        return aggregated

    def compare_groups(
        self, baseline_runs: List[str], comparison_runs: List[str]
    ) -> Dict[str, ComparisonResult]:
        """Compare metrics between two groups of runs.

        Args:
            baseline_runs: List of run IDs for baseline group
            comparison_runs: List of run IDs for comparison group

        Returns:
            Dictionary mapping metric names to comparison results
        """
        # Separate runs into groups
        baseline_group = [r for r in self.runs if r.run_id in baseline_runs]
        comparison_group = [r for r in self.runs if r.run_id in comparison_runs]

        if not baseline_group or not comparison_group:
            return {}

        # Find common metrics
        baseline_metrics = set(baseline_group[0].metrics.keys())
        comparison_metrics = set(comparison_group[0].metrics.keys())
        common_metrics = baseline_metrics & comparison_metrics

        results = {}
        for metric_name in common_metrics:
            baseline_values = [r.metrics[metric_name] for r in baseline_group]
            comparison_values = [r.metrics[metric_name] for r in comparison_group]

            baseline_mean = statistics.mean(baseline_values)
            comparison_mean = statistics.mean(comparison_values)
            diff = comparison_mean - baseline_mean
            pct_change = (diff / baseline_mean * 100) if baseline_mean != 0 else 0

            # Compute statistical significance
            p_value = None
            effect_size = None
            is_significant = False

            if len(baseline_values) >= 2 and len(comparison_values) >= 2:
                p_value = self._t_test(baseline_values, comparison_values)
                effect_size = self._cohens_d(baseline_values, comparison_values)
                is_significant = p_value < self.significance_threshold if p_value else False

            results[metric_name] = ComparisonResult(
                metric_name=metric_name,
                baseline_mean=baseline_mean,
                comparison_mean=comparison_mean,
                difference=diff,
                percent_change=pct_change,
                p_value=p_value,
                is_significant=is_significant,
                effect_size=effect_size,
            )

        return results

    def detect_regressions(
        self, baseline_runs: List[str], comparison_runs: List[str], threshold: float = 0.05
    ) -> Dict[str, ComparisonResult]:
        """Detect significant performance regressions.

        Args:
            baseline_runs: List of run IDs for baseline group
            comparison_runs: List of run IDs for comparison group
            threshold: Threshold for detecting regressions (fraction)

        Returns:
            Dictionary of metrics with detected regressions
        """
        comparisons = self.compare_groups(baseline_runs, comparison_runs)

        regressions = {}
        for metric_name, result in comparisons.items():
            # Check if it's a regression (negative change with significance)
            if result.difference < 0 and abs(result.difference) > threshold:
                if result.is_significant or result.p_value is None:
                    regressions[metric_name] = result

        return regressions

    def _compute_confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Compute confidence interval for values using t-distribution.

        Args:
            values: List of metric values

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(values) < self.min_runs_for_ci:
            mean_val = statistics.mean(values)
            return (mean_val, mean_val)

        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        n = len(values)

        # Use t-distribution critical value (approximation for 95% CI)
        # For n >= 30, use 1.96; otherwise use conservative 2.0
        t_critical = 1.96 if n >= 30 else 2.0

        margin = t_critical * std_val / math.sqrt(n)

        return (mean_val - margin, mean_val + margin)

    def _t_test(self, group1: List[float], group2: List[float]) -> float:
        """Perform Welch's t-test (unequal variances).

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            P-value for the test
        """
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 1.0

        mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
        var1 = statistics.variance(group1)
        var2 = statistics.variance(group2)

        # Welch's t-statistic
        t_stat = (mean1 - mean2) / math.sqrt(var1 / n1 + var2 / n2)

        # Degrees of freedom (Welch-Satterthwaite equation)
        df = (var1 / n1 + var2 / n2) ** 2 / (
            (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        )

        # Simplified p-value approximation (two-tailed)
        # For production use, consider scipy.stats.t.sf
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        return max(0.0, min(1.0, p_value))

    def _t_cdf(self, t: float, df: float) -> float:
        """Approximate t-distribution CDF."""
        # Very simple approximation - for better accuracy, use scipy
        x = df / (df + t * t)
        return 1 - 0.5 * x ** (df / 2)

    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Compute Cohen's d effect size.

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0

        mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
        var1 = statistics.variance(group1)
        var2 = statistics.variance(group2)

        # Pooled standard deviation
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    def export_summary(self, output_path: Union[str, Path]) -> None:
        """Export aggregated summary to JSON file.

        Args:
            output_path: Path to save the summary
        """
        aggregated = self.aggregate()

        summary = {
            "num_runs": len(self.runs),
            "run_ids": [r.run_id for r in self.runs],
            "metrics": {},
        }

        for metric_name, agg in aggregated.items():
            summary["metrics"][metric_name] = {
                "mean": agg.mean,
                "std": agg.std,
                "min": agg.min,
                "max": agg.max,
                "median": agg.median,
                "confidence_interval": list(agg.confidence_interval),
                "num_runs": agg.num_runs,
            }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)


def aggregate_from_files(
    result_files: List[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
    weighted: bool = False,
) -> Dict[str, AggregatedMetric]:
    """Convenience function to aggregate metrics from result files.

    Args:
        result_files: List of paths to result JSON files
        output_path: Optional path to save aggregated summary
        weighted: Whether to use weighted averaging

    Returns:
        Dictionary of aggregated metrics
    """
    aggregator = MetricsAggregator()
    aggregator.load_from_files(result_files)
    aggregated = aggregator.aggregate(weighted=weighted)

    if output_path:
        aggregator.export_summary(output_path)

    return aggregated
