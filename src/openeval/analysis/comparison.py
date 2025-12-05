"""Result comparison utilities for analyzing evaluation results.

This module provides tools for comparing evaluation results across different runs,
detecting regressions, and performing statistical tests for performance changes.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import statistics
from datetime import datetime

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricComparison:
    """Comparison of a single metric between two evaluations."""

    metric_name: str
    baseline_value: float
    comparison_value: float
    absolute_diff: float
    relative_diff: float  # Percentage change
    is_improvement: bool
    is_regression: bool
    is_significant: Optional[bool] = None
    p_value: Optional[float] = None


@dataclass
class ResultComparison:
    """Complete comparison between two evaluation results."""

    baseline_name: str
    comparison_name: str
    metric_comparisons: List[MetricComparison]
    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ResultComparer:
    """Compare evaluation results and detect changes."""

    def __init__(
        self,
        regression_threshold: float = 0.01,  # 1% drop is a regression
        improvement_threshold: float = 0.01,  # 1% gain is an improvement
        significance_level: float = 0.05,
    ):
        """Initialize result comparer.

        Args:
            regression_threshold: Threshold for detecting regressions (fraction)
            improvement_threshold: Threshold for detecting improvements (fraction)
            significance_level: P-value threshold for statistical significance
        """
        self.regression_threshold = regression_threshold
        self.improvement_threshold = improvement_threshold
        self.significance_level = significance_level

    def compare(
        self,
        baseline: Dict[str, float],
        comparison: Dict[str, float],
        baseline_name: str = "baseline",
        comparison_name: str = "comparison",
    ) -> ResultComparison:
        """Compare two sets of metrics.

        Args:
            baseline: Baseline metrics dictionary
            comparison: Comparison metrics dictionary
            baseline_name: Name for baseline results
            comparison_name: Name for comparison results

        Returns:
            ResultComparison object with detailed comparison
        """
        metric_comparisons = []

        # Find all metrics present in either result
        all_metrics = set(baseline.keys()) | set(comparison.keys())

        for metric_name in sorted(all_metrics):
            base_val = baseline.get(metric_name, 0.0)
            comp_val = comparison.get(metric_name, 0.0)

            # Skip if both are zero
            if base_val == 0 and comp_val == 0:
                continue

            abs_diff = comp_val - base_val
            rel_diff = (abs_diff / base_val) if base_val != 0 else 0.0

            # Determine if change is improvement or regression
            # (assuming higher is better - can be inverted for specific metrics)
            is_improvement = rel_diff > self.improvement_threshold
            is_regression = rel_diff < -self.regression_threshold

            metric_comp = MetricComparison(
                metric_name=metric_name,
                baseline_value=base_val,
                comparison_value=comp_val,
                absolute_diff=abs_diff,
                relative_diff=rel_diff,
                is_improvement=is_improvement,
                is_regression=is_regression,
            )
            metric_comparisons.append(metric_comp)

        # Generate summary
        summary = self._generate_summary(metric_comparisons)

        return ResultComparison(
            baseline_name=baseline_name,
            comparison_name=comparison_name,
            metric_comparisons=metric_comparisons,
            summary=summary,
        )

    def compare_from_files(
        self,
        baseline_path: Union[str, Path],
        comparison_path: Union[str, Path],
    ) -> ResultComparison:
        """Compare results from JSON files.

        Args:
            baseline_path: Path to baseline results JSON
            comparison_path: Path to comparison results JSON

        Returns:
            ResultComparison object
        """
        baseline_path = Path(baseline_path)
        comparison_path = Path(comparison_path)

        # Load results
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)

        with open(comparison_path, "r") as f:
            comparison_data = json.load(f)

        # Extract metrics
        baseline_metrics = self._extract_metrics(baseline_data)
        comparison_metrics = self._extract_metrics(comparison_data)

        return self.compare(
            baseline=baseline_metrics,
            comparison=comparison_metrics,
            baseline_name=baseline_path.stem,
            comparison_name=comparison_path.stem,
        )

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics from result data.

        Args:
            data: Result data dictionary

        Returns:
            Dictionary of metric name to value
        """
        # Try different possible locations for metrics
        if "metrics" in data:
            metrics = data["metrics"]
        elif "results" in data:
            metrics = data["results"]
        else:
            # Assume top level contains metrics
            metrics = {k: v for k, v in data.items() if isinstance(v, (int, float))}

        # Flatten nested metrics
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        flat_metrics[f"{key}.{subkey}"] = float(subvalue)
            elif isinstance(value, (int, float)):
                flat_metrics[key] = float(value)

        return flat_metrics

    def _generate_summary(self, comparisons: List[MetricComparison]) -> Dict[str, Any]:
        """Generate summary statistics from comparisons.

        Args:
            comparisons: List of metric comparisons

        Returns:
            Summary dictionary
        """
        if not comparisons:
            return {}

        improvements = [c for c in comparisons if c.is_improvement]
        regressions = [c for c in comparisons if c.is_regression]
        unchanged = [c for c in comparisons if not c.is_improvement and not c.is_regression]

        avg_change = statistics.mean(c.relative_diff for c in comparisons)

        return {
            "total_metrics": len(comparisons),
            "improvements": len(improvements),
            "regressions": len(regressions),
            "unchanged": len(unchanged),
            "avg_relative_change": avg_change,
            "max_improvement": (max((c.relative_diff for c in improvements), default=0.0)),
            "max_regression": (min((c.relative_diff for c in regressions), default=0.0)),
            "improved_metrics": [c.metric_name for c in improvements],
            "regressed_metrics": [c.metric_name for c in regressions],
        }

    def detect_regressions(self, comparison: ResultComparison) -> List[MetricComparison]:
        """Extract regressions from comparison.

        Args:
            comparison: Result comparison

        Returns:
            List of metrics with detected regressions
        """
        return [c for c in comparison.metric_comparisons if c.is_regression]

    def detect_improvements(self, comparison: ResultComparison) -> List[MetricComparison]:
        """Extract improvements from comparison.

        Args:
            comparison: Result comparison

        Returns:
            List of metrics with detected improvements
        """
        return [c for c in comparison.metric_comparisons if c.is_improvement]


def side_by_side_table(comparison: ResultComparison) -> str:
    """Generate side-by-side comparison table as formatted string.

    Args:
        comparison: Result comparison

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 100)
    lines.append("Comparison: {comparison.baseline_name} vs {comparison.comparison_name}")
    lines.append("=" * 100)
    lines.append("")

    # Header
    lines.append(f"{'Metric':<30} {'Baseline':>12} {'Comparison':>12} {'Change':>12} {'%':>8}")
    lines.append("-" * 100)

    # Sort by absolute relative diff descending
    sorted_comparisons = sorted(
        comparison.metric_comparisons,
        key=lambda x: abs(x.relative_diff),
        reverse=True,
    )

    for comp in sorted_comparisons:
        # Format change indicator
        if comp.is_regression:
            indicator = "↓"
        elif comp.is_improvement:
            indicator = "↑"
        else:
            indicator = "="

        # Format percentage
        pct_str = f"{comp.relative_diff * 100:+.2f}%"

        lines.append(
            f"{comp.metric_name:<30} "
            f"{comp.baseline_value:>12.4f} "
            f"{comp.comparison_value:>12.4f} "
            f"{comp.absolute_diff:>11.4f} {indicator} "
            f"{pct_str:>8}"
        )

    # Summary
    lines.append("-" * 100)
    lines.append("")
    lines.append("Summary:")
    lines.append(f"  Total metrics: {comparison.summary['total_metrics']}")
    lines.append(f"  Improvements:  {comparison.summary['improvements']}")
    lines.append(f"  Regressions:   {comparison.summary['regressions']}")
    lines.append(f"  Unchanged:     {comparison.summary['unchanged']}")
    lines.append(f"  Avg change:    {comparison.summary['avg_relative_change'] * 100:+.2f}%")

    if comparison.summary["regressed_metrics"]:
        lines.append("")
        lines.append("Regressed metrics:")
        for metric in comparison.summary["regressed_metrics"]:
            lines.append(f"  - {metric}")

    if comparison.summary["improved_metrics"]:
        lines.append("")
        lines.append("Improved metrics:")
        for metric in comparison.summary["improved_metrics"]:
            lines.append(f"  - {metric}")

    lines.append("=" * 100)

    return "\n".join(lines)


def export_comparison(comparison: ResultComparison, output_path: Union[str, Path]) -> None:
    """Export comparison results to JSON file.

    Args:
        comparison: Result comparison
        output_path: Path to save comparison
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict for JSON serialization
    data = {
        "baseline_name": comparison.baseline_name,
        "comparison_name": comparison.comparison_name,
        "timestamp": comparison.timestamp,
        "summary": comparison.summary,
        "metrics": [
            {
                "metric_name": c.metric_name,
                "baseline_value": c.baseline_value,
                "comparison_value": c.comparison_value,
                "absolute_diff": c.absolute_diff,
                "relative_diff": c.relative_diff,
                "is_improvement": c.is_improvement,
                "is_regression": c.is_regression,
            }
            for c in comparison.metric_comparisons
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Exported comparison to {output_path}")


def diff_results(
    baseline_path: Union[str, Path],
    comparison_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    print_table: bool = True,
) -> ResultComparison:
    """Convenience function to diff two result files.

    Args:
        baseline_path: Path to baseline results
        comparison_path: Path to comparison results
        output_path: Optional path to save comparison JSON
        print_table: Whether to print comparison table

    Returns:
        ResultComparison object
    """
    comparer = ResultComparer()
    comparison = comparer.compare_from_files(baseline_path, comparison_path)

    if print_table:
        print(side_by_side_table(comparison))

    if output_path:
        export_comparison(comparison, output_path)

    return comparison
