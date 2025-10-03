"""
Results Analysis and Comparison Framework for OpenEval Lab

This module provides comprehensive analysis and comparison of evaluation results,
including statistical significance testing, performance trends, and detailed reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import statistics
from collections import defaultdict
import itertools

try:
    import numpy as np
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    np = None
    stats = None

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .enhanced_logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Represents a single evaluation result."""

    model_name: str
    task: str
    dataset: str
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    run_id: Optional[str] = None

    @property
    def primary_metric(self) -> Optional[float]:
        """Get the primary evaluation metric."""
        # Try common primary metrics in order
        primary_candidates = ["accuracy", "f1", "bleu", "rouge_l", "exact_match", "pass_rate"]
        for metric in primary_candidates:
            if metric in self.metrics:
                return self.metrics[metric]
        # Return first available metric
        return next(iter(self.metrics.values())) if self.metrics else None


@dataclass
class StatisticalTestResult:
    """Result of statistical significance test."""

    test_name: str
    p_value: float
    statistic: float
    significant: bool
    alpha: float = 0.05
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

    def summary(self) -> str:
        """Get a summary of the test result."""
        sig_symbol = "✅" if self.significant else "❌"
        return (
            f"{sig_symbol} {self.test_name}: p={self.p_value:.4f}, significant={self.significant}"
        )


@dataclass
class PerformanceComparison:
    """Comparison between two evaluation results."""

    result_a: EvaluationResult
    result_b: EvaluationResult
    metric_name: str
    difference: float
    relative_difference: float
    statistical_test: Optional[StatisticalTestResult] = None

    def summary(self) -> str:
        """Get a summary of the comparison."""
        better = "A" if self.difference > 0 else "B"
        sig_note = ""
        if self.statistical_test:
            sig_note = (
                f" ({'significant' if self.statistical_test.significant else 'not significant'})"
            )

        return f"{better} better by {abs(self.relative_difference):.1f}% on {self.metric_name}{sig_note}"


@dataclass
class AnalysisReport:
    """Comprehensive analysis report."""

    title: str
    results: List[EvaluationResult] = field(default_factory=list)
    comparisons: List[PerformanceComparison] = field(default_factory=list)
    statistical_tests: List[StatisticalTestResult] = field(default_factory=list)
    rankings: Dict[str, List[str]] = field(default_factory=dict)
    trends: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


class ResultsAnalyzer:
    """
    Comprehensive analyzer for evaluation results.
    """

    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_results(
        self, results_paths: Optional[List[Union[str, Path]]] = None, pattern: str = "*.json"
    ) -> List[EvaluationResult]:
        """
        Load evaluation results from files.

        Args:
            results_paths: Specific result files to load (if None, scan results_dir)
            pattern: File pattern to match

        Returns:
            List of EvaluationResult objects
        """
        results = []

        if results_paths:
            files = [Path(p) for p in results_paths]
        else:
            files = list(self.results_dir.glob(pattern))

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                result = self._parse_result_data(data, file_path.stem)
                if result:
                    results.append(result)

            except Exception as e:
                logger.warning(f"Failed to load result from {file_path}: {e}")

        logger.info(f"Loaded {len(results)} evaluation results")
        return results

    def _parse_result_data(self, data: Dict[str, Any], filename: str) -> Optional[EvaluationResult]:
        """Parse evaluation result data."""
        try:
            # Extract basic information
            model_name = data.get("model", {}).get("name", filename)
            task = data.get("task", "unknown")
            dataset = data.get("dataset", {}).get("path", "unknown")

            # Extract metrics
            metrics = {}
            if "results" in data and "metrics" in data["results"]:
                metrics = data["results"]["metrics"]

            # Extract config
            config = data.get("config", {})

            # Extract run ID and timestamp
            run_id = data.get("run_id", filename)
            timestamp_str = data.get("timestamp")
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()

            return EvaluationResult(
                model_name=model_name,
                task=task,
                dataset=dataset,
                metrics=metrics,
                config=config,
                timestamp=timestamp,
                run_id=run_id,
            )

        except Exception as e:
            logger.error(f"Failed to parse result data: {e}")
            return None

    def compare_results(
        self,
        results: List[EvaluationResult],
        metric_name: Optional[str] = None,
        group_by: str = "model",
    ) -> AnalysisReport:
        """
        Compare evaluation results.

        Args:
            results: List of evaluation results to compare
            metric_name: Specific metric to compare (if None, use primary metric)
            group_by: How to group results ('model', 'task', 'dataset')

        Returns:
            AnalysisReport with comparisons and analysis
        """
        report = AnalysisReport(title=f"Results Comparison ({group_by} grouping)", results=results)

        if not results:
            return report

        # Determine metric to use
        if not metric_name:
            metric_name = self._find_common_metric(results)
            if not metric_name:
                logger.warning("No common metric found for comparison")
                return report

        # Group results
        grouped_results = self._group_results(results, group_by)

        # Generate rankings
        report.rankings = self._generate_rankings(grouped_results, metric_name)

        # Generate pairwise comparisons
        report.comparisons = self._generate_comparisons(grouped_results, metric_name)

        # Perform statistical tests
        report.statistical_tests = self._perform_statistical_tests(grouped_results, metric_name)

        # Analyze trends
        report.trends = self._analyze_trends(results, metric_name)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _find_common_metric(self, results: List[EvaluationResult]) -> Optional[str]:
        """Find a metric that exists in all results."""
        if not results:
            return None

        # Get all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())

        # Find metrics present in all results
        common_metrics = set()
        for metric in all_metrics:
            if all(metric in result.metrics for result in results):
                common_metrics.add(metric)

        # Return primary metric if available, otherwise first common metric
        primary_candidates = ["accuracy", "f1", "bleu", "rouge_l", "exact_match", "pass_rate"]
        for candidate in primary_candidates:
            if candidate in common_metrics:
                return candidate

        return next(iter(common_metrics)) if common_metrics else None

    def _group_results(
        self, results: List[EvaluationResult], group_by: str
    ) -> Dict[str, List[EvaluationResult]]:
        """Group results by specified attribute."""
        grouped = defaultdict(list)

        for result in results:
            if group_by == "model":
                key = result.model_name
            elif group_by == "task":
                key = result.task
            elif group_by == "dataset":
                key = result.dataset
            else:
                key = result.model_name  # Default to model

            grouped[key].append(result)

        return dict(grouped)

    def _generate_rankings(
        self, grouped_results: Dict[str, List[EvaluationResult]], metric_name: str
    ) -> Dict[str, List[str]]:
        """Generate rankings for each group."""
        rankings = {}

        for group_name, group_results in grouped_results.items():
            # Calculate average metric value for each result
            scored_results = []
            for result in group_results:
                score = result.metrics.get(metric_name)
                if score is not None:
                    scored_results.append((result.model_name, score))

            # Sort by score (descending)
            scored_results.sort(key=lambda x: x[1], reverse=True)
            rankings[group_name] = [name for name, _ in scored_results]

        return rankings

    def _generate_comparisons(
        self, grouped_results: Dict[str, List[EvaluationResult]], metric_name: str
    ) -> List[PerformanceComparison]:
        """Generate pairwise comparisons."""
        comparisons = []

        # Get all unique model names
        all_models = set()
        for group_results in grouped_results.values():
            for result in group_results:
                all_models.add(result.model_name)

        # Generate pairwise comparisons for each group
        for group_name, group_results in grouped_results.items():
            if len(group_results) < 2:
                continue

            # Create model to result mapping
            model_results = {}
            for result in group_results:
                if result.model_name not in model_results:
                    model_results[result.model_name] = result

            # Generate all pairwise comparisons
            model_names = list(model_results.keys())
            for i, j in itertools.combinations(range(len(model_names)), 2):
                model_a, model_b = model_names[i], model_names[j]
                result_a, result_b = model_results[model_a], model_results[model_b]

                score_a = result_a.metrics.get(metric_name)
                score_b = result_b.metrics.get(metric_name)

                if score_a is not None and score_b is not None:
                    difference = score_a - score_b
                    relative_difference = (difference / score_b) * 100 if score_b != 0 else 0

                    comparison = PerformanceComparison(
                        result_a=result_a,
                        result_b=result_b,
                        metric_name=metric_name,
                        difference=difference,
                        relative_difference=relative_difference,
                    )
                    comparisons.append(comparison)

        return comparisons

    def _perform_statistical_tests(
        self, grouped_results: Dict[str, List[EvaluationResult]], metric_name: str
    ) -> List[StatisticalTestResult]:
        """Perform statistical significance tests."""
        if not HAS_SCIPY:
            logger.warning("SciPy not available, skipping statistical tests")
            return []

        tests = []

        # Perform t-tests between top performers in each group
        for group_name, group_results in grouped_results.items():
            if len(group_results) < 2:
                continue

            # Get scores for all models
            model_scores = defaultdict(list)
            for result in group_results:
                score = result.metrics.get(metric_name)
                if score is not None:
                    model_scores[result.model_name].append(score)

            # Perform pairwise t-tests between models with multiple runs
            model_names = list(model_scores.keys())
            for i, j in itertools.combinations(range(len(model_names)), 2):
                model_a, model_b = model_names[i], model_names[j]
                scores_a = model_scores[model_a]
                scores_b = model_scores[model_b]

                if len(scores_a) >= 2 and len(scores_b) >= 2:
                    try:
                        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
                        significant = p_value < 0.05

                        # Calculate effect size (Cohen's d)
                        mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
                        std_a, std_b = np.std(scores_a, ddof=1), np.std(scores_b, ddof=1)
                        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
                        effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

                        test_result = StatisticalTestResult(
                            test_name=f"t-test: {model_a} vs {model_b} ({group_name})",
                            p_value=p_value,
                            statistic=t_stat,
                            significant=significant,
                            effect_size=effect_size,
                        )
                        tests.append(test_result)

                    except Exception as e:
                        logger.warning(f"Failed to perform t-test: {e}")

        return tests

    def _analyze_trends(self, results: List[EvaluationResult], metric_name: str) -> Dict[str, Any]:
        """Analyze performance trends."""
        trends = {}

        # Group by timestamp
        time_series = defaultdict(list)
        for result in results:
            score = result.metrics.get(metric_name)
            if score is not None:
                time_series[result.timestamp.date()].append(score)

        # Calculate daily averages
        daily_averages = {}
        for date, scores in time_series.items():
            daily_averages[date] = statistics.mean(scores)

        trends["daily_averages"] = dict(sorted(daily_averages.items()))

        # Calculate overall trend
        if len(daily_averages) > 1:
            dates = list(daily_averages.keys())
            scores = [daily_averages[date] for date in dates]

            # Simple linear trend
            if len(scores) > 1:
                x = list(range(len(scores)))
                slope = statistics.linear_regression(x, scores).slope
                trends["overall_trend"] = "improving" if slope > 0 else "declining"
                trends["trend_slope"] = slope

        return trends

    def _generate_recommendations(self, report: AnalysisReport) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Check for statistical significance
        significant_findings = [test for test in report.statistical_tests if test.significant]
        if significant_findings:
            recommendations.append(
                f"Found {len(significant_findings)} statistically significant differences. "
                "Consider focusing on the top-performing configurations."
            )

        # Check rankings consistency
        if len(report.rankings) > 1:
            # Check if rankings are consistent across groups
            all_rankings = list(report.rankings.values())
            if len(set(tuple(r) for r in all_rankings)) == 1:
                recommendations.append("Model rankings are consistent across all groups.")
            else:
                recommendations.append(
                    "Model rankings vary across groups - consider task-specific optimization."
                )

        # Check trends
        if "overall_trend" in report.trends:
            trend = report.trends["overall_trend"]
            if trend == "improving":
                recommendations.append(
                    "Performance is trending upward - continue current optimization approach."
                )
            elif trend == "declining":
                recommendations.append(
                    "Performance is trending downward - review recent changes or configurations."
                )

        # General recommendations
        if not report.statistical_tests:
            recommendations.append(
                "Consider running multiple evaluation runs for statistical significance testing."
            )

        return recommendations

    def generate_report(
        self,
        report: AnalysisReport,
        output_format: str = "html",
        include_visualizations: bool = True,
    ) -> Path:
        """
        Generate a comprehensive analysis report.

        Args:
            report: Analysis report to generate
            output_format: Output format ('html', 'json', 'markdown')
            include_visualizations: Whether to include visualizations

        Returns:
            Path to generated report
        """
        timestamp = int(datetime.now().timestamp())

        if output_format == "html":
            content = self._generate_html_report(report, include_visualizations)
            file_path = self.results_dir / f"analysis_report_{timestamp}.html"

        elif output_format == "json":
            content = self._generate_json_report(report)
            file_path = self.results_dir / f"analysis_report_{timestamp}.json"

        elif output_format == "markdown":
            content = self._generate_markdown_report(report)
            file_path = self.results_dir / f"analysis_report_{timestamp}.md"

        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Generated analysis report: {file_path}")
        return file_path

    def _generate_html_report(self, report: AnalysisReport, include_visualizations: bool) -> str:
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OpenEval Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 3px; }}
        .ranking {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .comparison {{ background: #d1ecf1; padding: 10px; border-radius: 5px; margin: 5px 0; }}
        .recommendation {{ background: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 OpenEval Analysis Report</h1>
        <h2>{report.title}</h2>
        <p>Generated on {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>📊 Summary</h2>
        <div class="metric">Total Results: {len(report.results)}</div>
        <div class="metric">Comparisons: {len(report.comparisons)}</div>
        <div class="metric">Statistical Tests: {len(report.statistical_tests)}</div>
    </div>
"""

        if report.rankings:
            html += """
    <div class="section">
        <h2>🏆 Rankings</h2>
"""
            for group_name, ranking in report.rankings.items():
                html += f"""
        <div class="ranking">
            <h3>{group_name}</h3>
            <ol>
"""
                for i, model in enumerate(ranking, 1):
                    html += f"                <li>{model}</li>\n"
                html += "            </ol>\n        </div>\n"

        if report.comparisons:
            html += """
    <div class="section">
        <h2>⚖️ Key Comparisons</h2>
"""
            # Show top 10 comparisons
            for comparison in report.comparisons[:10]:
                html += f"""
        <div class="comparison">
            {comparison.summary()}
        </div>"""

        if report.statistical_tests:
            html += """
    <div class="section">
        <h2>📈 Statistical Tests</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>P-Value</th>
                <th>Significant</th>
                <th>Effect Size</th>
            </tr>
"""
            for test in report.statistical_tests:
                html += f"""
            <tr>
                <td>{test.test_name}</td>
                <td>{test.p_value:.4f}</td>
                <td>{'✅' if test.significant else '❌'}</td>
                <td>{test.effect_size:.3f} if test.effect_size else 'N/A'</td>
            </tr>"""
            html += """
        </table>
    </div>
"""

        if report.recommendations:
            html += """
    <div class="section">
        <h2>💡 Recommendations</h2>
"""
            for rec in report.recommendations:
                html += f"""
        <div class="recommendation">
            {rec}
        </div>"""

        html += """
</body>
</html>"""

        return html

    def _generate_json_report(self, report: AnalysisReport) -> str:
        """Generate JSON report."""
        data = {
            "title": report.title,
            "generated_at": report.generated_at.isoformat(),
            "summary": {
                "total_results": len(report.results),
                "comparisons": len(report.comparisons),
                "statistical_tests": len(report.statistical_tests),
            },
            "rankings": report.rankings,
            "comparisons": [
                {
                    "model_a": c.result_a.model_name,
                    "model_b": c.result_b.model_name,
                    "metric": c.metric_name,
                    "difference": c.difference,
                    "relative_difference": c.relative_difference,
                    "significant": c.statistical_test.significant if c.statistical_test else None,
                }
                for c in report.comparisons
            ],
            "statistical_tests": [
                {
                    "test_name": t.test_name,
                    "p_value": t.p_value,
                    "statistic": t.statistic,
                    "significant": t.significant,
                    "effect_size": t.effect_size,
                }
                for t in report.statistical_tests
            ],
            "trends": report.trends,
            "recommendations": report.recommendations,
        }

        return json.dumps(data, indent=2, ensure_ascii=False)

    def _generate_markdown_report(self, report: AnalysisReport) -> str:
        """Generate Markdown report."""
        md = f"""# OpenEval Analysis Report

## {report.title}

Generated on {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- Total Results: {len(report.results)}
- Comparisons: {len(report.comparisons)}
- Statistical Tests: {len(report.statistical_tests)}

"""

        if report.rankings:
            md += "## Rankings\n\n"
            for group_name, ranking in report.rankings.items():
                md += f"### {group_name}\n\n"
                for i, model in enumerate(ranking, 1):
                    md += f"{i}. {model}\n"
                md += "\n"

        if report.comparisons:
            md += "## Key Comparisons\n\n"
            for comparison in report.comparisons[:10]:
                md += f"- {comparison.summary()}\n"
            md += "\n"

        if report.statistical_tests:
            md += "## Statistical Tests\n\n"
            md += "| Test | P-Value | Significant | Effect Size |\n"
            md += "|------|---------|-------------|-------------|\n"
            for test in report.statistical_tests:
                effect_size = f"{test.effect_size:.3f}" if test.effect_size else "N/A"
                md += f"| {test.test_name} | {test.p_value:.4f} | {'✅' if test.significant else '❌'} | {effect_size} |\n"
            md += "\n"

        if report.recommendations:
            md += "## Recommendations\n\n"
            for rec in report.recommendations:
                md += f"- {rec}\n"
            md += "\n"

        return md


def create_results_analyzer(results_dir: Optional[Path] = None) -> ResultsAnalyzer:
    """Create a results analyzer instance."""
    return ResultsAnalyzer(results_dir)


def quick_comparison(
    results_files: List[Union[str, Path]],
    metric_name: Optional[str] = None,
    output_format: str = "html",
) -> Path:
    """
    Perform a quick comparison of evaluation results.

    Args:
        results_files: List of result files to compare
        metric_name: Metric to compare (if None, auto-detect)
        output_format: Output format for the report

    Returns:
        Path to generated comparison report
    """
    analyzer = ResultsAnalyzer()
    results = analyzer.load_results(results_files)
    report = analyzer.compare_results(results, metric_name)
    return analyzer.generate_report(report, output_format)
