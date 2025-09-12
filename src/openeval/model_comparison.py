"""
Model Comparison Dashboard for OpenEval Lab

This module provides a comprehensive dashboard for comparing multiple models
across different evaluation tasks, metrics, and datasets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import statistics

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .enhanced_logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """Represents evaluation results for a single model."""
    model_name: str
    task: str
    dataset: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def primary_metric(self) -> Optional[float]:
        """Get the primary evaluation metric."""
        # Try common primary metrics in order
        primary_candidates = ['accuracy', 'f1', 'bleu', 'rouge_l', 'exact_match']
        for metric in primary_candidates:
            if metric in self.metrics:
                return self.metrics[metric]
        # Return first available metric
        return next(iter(self.metrics.values())) if self.metrics else None


@dataclass
class ComparisonReport:
    """Comprehensive model comparison report."""
    models: List[ModelResult] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    rankings: Dict[str, List[str]] = field(default_factory=dict)
    visualizations: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: ModelResult) -> None:
        """Add a model result to the comparison."""
        self.models.append(result)

    def generate_summary(self) -> None:
        """Generate summary statistics."""
        if not self.models:
            return

        # Group by task and dataset
        task_groups = {}
        dataset_groups = {}

        for result in self.models:
            task_key = result.task
            dataset_key = result.dataset

            if task_key not in task_groups:
                task_groups[task_key] = []
            task_groups[task_key].append(result)

            if dataset_key not in dataset_groups:
                dataset_groups[dataset_key] = []
            dataset_groups[dataset_key].append(result)

        # Calculate summary statistics
        self.summary_stats = {
            'total_models': len(self.models),
            'unique_tasks': len(task_groups),
            'unique_datasets': len(dataset_groups),
            'task_breakdown': {task: len(results) for task, results in task_groups.items()},
            'dataset_breakdown': {dataset: len(results) for dataset, results in dataset_groups.items()}
        }

    def generate_rankings(self) -> None:
        """Generate model rankings by different criteria."""
        if not self.models:
            return

        rankings = {}

        # Group by task for task-specific rankings
        task_groups = {}
        for result in self.models:
            if result.task not in task_groups:
                task_groups[result.task] = []
            task_groups[result.task].append(result)

        # Overall ranking (across all tasks)
        all_primary_scores = [(r.model_name, r.primary_metric or 0) for r in self.models]
        rankings['overall'] = [name for name, _ in sorted(all_primary_scores, key=lambda x: x[1], reverse=True)]

        # Task-specific rankings
        for task, results in task_groups.items():
            task_scores = [(r.model_name, r.primary_metric or 0) for r in results]
            rankings[f'task_{task}'] = [name for name, _ in sorted(task_scores, key=lambda x: x[1], reverse=True)]

        self.rankings = rankings


class ModelComparisonDashboard:
    """
    Interactive dashboard for model comparison and analysis.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("model_comparison_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports: Dict[str, ComparisonReport] = {}

    def load_evaluation_results(
        self,
        results_dir: Union[str, Path],
        pattern: str = "*.json"
    ) -> str:
        """
        Load evaluation results from directory.

        Args:
            results_dir: Directory containing evaluation result files
            pattern: File pattern to match

        Returns:
            Report ID
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_path}")

        report = ComparisonReport()
        report_id = f"report_{int(datetime.now().timestamp())}"

        # Load result files
        for result_file in results_path.glob(pattern):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Parse result data
                model_result = self._parse_result_data(data, result_file.stem)
                if model_result:
                    report.add_result(model_result)

            except Exception as e:
                logger.warning(f"Failed to load {result_file}: {e}")

        # Generate analysis
        report.generate_summary()
        report.generate_rankings()

        self.reports[report_id] = report
        logger.info(f"Loaded {len(report.models)} model results into report {report_id}")

        return report_id

    def _parse_result_data(self, data: Dict[str, Any], filename: str) -> Optional[ModelResult]:
        """Parse evaluation result data into ModelResult."""
        try:
            # Extract model information
            model_name = data.get('model', {}).get('name', filename)
            task = data.get('task', 'unknown')
            dataset = data.get('dataset', {}).get('path', 'unknown')

            # Extract metrics
            metrics = {}
            if 'results' in data and 'metrics' in data['results']:
                metrics = data['results']['metrics']

            # Extract metadata
            metadata = {
                'config': data.get('config', {}),
                'timestamp': data.get('timestamp'),
                'version': data.get('version')
            }

            return ModelResult(
                model_name=model_name,
                task=task,
                dataset=dataset,
                metrics=metrics,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Failed to parse result data: {e}")
            return None

    def generate_comparison_report(
        self,
        report_id: str,
        include_visualizations: bool = True
    ) -> Path:
        """
        Generate comprehensive comparison report.

        Args:
            report_id: Report identifier
            include_visualizations: Whether to include visualizations

        Returns:
            Path to generated report
        """
        if report_id not in self.reports:
            raise ValueError(f"Report not found: {report_id}")

        report = self.reports[report_id]

        # Generate visualizations if requested
        if include_visualizations and HAS_PLOTLY:
            report.visualizations = self._generate_visualizations(report)

        # Create report content
        report_content = self._create_report_content(report)

        # Save report
        report_file = self.output_dir / f"model_comparison_{report_id}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"Generated comparison report: {report_file}")
        return report_file

    def _generate_visualizations(self, report: ComparisonReport) -> Dict[str, Any]:
        """Generate visualizations for the report."""
        visualizations = {}

        if not HAS_PLOTLY or not report.models:
            return visualizations

        # Model performance comparison
        visualizations['performance_chart'] = self._create_performance_chart(report)

        # Metric distribution
        visualizations['metric_distribution'] = self._create_metric_distribution_chart(report)

        # Ranking visualization
        visualizations['ranking_chart'] = self._create_ranking_chart(report)

        return visualizations

    def _create_performance_chart(self, report: ComparisonReport) -> str:
        """Create performance comparison chart."""
        models = [r.model_name for r in report.models]
        scores = [r.primary_metric or 0 for r in report.models]

        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=scores,
                text=[f"{s:.3f}" for s in scores],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Primary Metric Score",
            showlegend=False
        )

        return fig.to_html(full_html=False)

    def _create_metric_distribution_chart(self, report: ComparisonReport) -> str:
        """Create metric distribution chart."""
        if not HAS_PANDAS:
            return "<p>Pandas required for metric distribution chart</p>"

        # Collect all metrics
        all_metrics = {}
        for result in report.models:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append((result.model_name, value))

        if not all_metrics:
            return "<p>No metrics available for distribution chart</p>"

        # Create subplot for each metric
        metrics_list = list(all_metrics.keys())
        n_metrics = len(metrics_list)

        if n_metrics == 0:
            return "<p>No metrics to display</p>"

        fig = make_subplots(
            rows=(n_metrics + 2) // 3,  # 3 columns
            cols=min(3, n_metrics),
            subplot_titles=metrics_list
        )

        for i, metric in enumerate(metrics_list):
            data = all_metrics[metric]
            models = [d[0] for d in data]
            values = [d[1] for d in data]

            row = (i // 3) + 1
            col = (i % 3) + 1

            fig.add_trace(
                go.Bar(x=models, y=values, name=metric),
                row=row, col=col
            )

        fig.update_layout(
            title="Metric Distribution Across Models",
            showlegend=False
        )

        return fig.to_html(full_html=False)

    def _create_ranking_chart(self, report: ComparisonReport) -> str:
        """Create ranking visualization."""
        if 'overall' not in report.rankings:
            return "<p>No ranking data available</p>"

        ranking = report.rankings['overall']
        positions = list(range(1, len(ranking) + 1))

        fig = go.Figure(data=[
            go.Scatter(
                x=positions,
                y=ranking,
                mode='markers+text',
                text=ranking,
                textposition="top center",
                marker=dict(size=20, color=positions, colorscale='Viridis')
            )
        ])

        fig.update_layout(
            title="Model Ranking (Overall Performance)",
            xaxis_title="Rank Position",
            yaxis_title="Model",
            showlegend=False
        )

        return fig.to_html(full_html=False)

    def _create_report_content(self, report: ComparisonReport) -> str:
        """Create HTML report content."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 3px; }}
        .ranking {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Model Comparison Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>📊 Summary Statistics</h2>
        <div class="metric">Total Models: {report.summary_stats.get('total_models', 0)}</div>
        <div class="metric">Unique Tasks: {report.summary_stats.get('unique_tasks', 0)}</div>
        <div class="metric">Unique Datasets: {report.summary_stats.get('unique_datasets', 0)}</div>
    </div>

    <div class="section">
        <h2>🏆 Model Rankings</h2>
"""

        if report.rankings:
            for ranking_type, ranking in report.rankings.items():
                html += f"""
        <div class="ranking">
            <h3>{ranking_type.replace('_', ' ').title()}</h3>
            <ol>
"""
                for i, model in enumerate(ranking, 1):
                    html += f"                <li>{model}</li>\n"
                html += "            </ol>\n        </div>\n"

        html += """
    </div>

    <div class="section">
        <h2>📈 Detailed Results</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Task</th>
                <th>Dataset</th>
                <th>Primary Metric</th>
                <th>All Metrics</th>
            </tr>
"""

        for result in report.models:
            primary = result.primary_metric or "N/A"
            metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in result.metrics.items()])
            html += f"""
            <tr>
                <td>{result.model_name}</td>
                <td>{result.task}</td>
                <td>{result.dataset}</td>
                <td>{primary}</td>
                <td>{metrics_str}</td>
            </tr>"""

        html += """
        </table>
    </div>
"""

        # Add visualizations
        if report.visualizations:
            html += """
    <div class="section">
        <h2>📊 Visualizations</h2>
"""
            for viz_name, viz_html in report.visualizations.items():
                html += f"""
        <div class="section">
            <h3>{viz_name.replace('_', ' ').title()}</h3>
            {viz_html}
        </div>
"""

        html += """
</body>
</html>"""

        return html

    def export_comparison_data(
        self,
        report_id: str,
        format: str = "json"
    ) -> Path:
        """
        Export comparison data in various formats.

        Args:
            report_id: Report identifier
            format: Export format (json, csv, excel)

        Returns:
            Path to exported file
        """
        if report_id not in self.reports:
            raise ValueError(f"Report not found: {report_id}")

        report = self.reports[report_id]

        if format == "json":
            data = {
                "summary": report.summary_stats,
                "rankings": report.rankings,
                "models": [
                    {
                        "model_name": r.model_name,
                        "task": r.task,
                        "dataset": r.dataset,
                        "metrics": r.metrics,
                        "metadata": r.metadata,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in report.models
                ]
            }

            export_file = self.output_dir / f"comparison_data_{report_id}.json"
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format in ["csv", "excel"] and HAS_PANDAS:
            # Convert to DataFrame
            rows = []
            for result in report.models:
                row = {
                    "model_name": result.model_name,
                    "task": result.task,
                    "dataset": result.dataset,
                    "timestamp": result.timestamp.isoformat()
                }
                # Convert metrics to strings for CSV/Excel compatibility
                for metric_name, metric_value in result.metrics.items():
                    row[metric_name] = str(metric_value)
                rows.append(row)

            df = pd.DataFrame(rows)

            if format == "csv":
                export_file = self.output_dir / f"comparison_data_{report_id}.csv"
                df.to_csv(export_file, index=False)
            else:  # excel
                export_file = self.output_dir / f"comparison_data_{report_id}.xlsx"
                df.to_excel(export_file, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported comparison data to: {export_file}")
        return export_file

    def get_report_summary(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a comparison report."""
        if report_id not in self.reports:
            return None

        report = self.reports[report_id]
        return {
            "report_id": report_id,
            "total_models": len(report.models),
            "summary_stats": report.summary_stats,
            "rankings": report.rankings
        }


def create_model_comparison_dashboard(output_dir: Optional[Path] = None) -> ModelComparisonDashboard:
    """Create a model comparison dashboard instance."""
    return ModelComparisonDashboard(output_dir)


def compare_models_from_directory(
    results_dir: Union[str, Path],
    output_dir: Optional[Path] = None,
    include_visualizations: bool = True
) -> Path:
    """
    Convenience function to compare models from a results directory.

    Args:
        results_dir: Directory containing evaluation results
        output_dir: Output directory for reports
        include_visualizations: Whether to include visualizations

    Returns:
        Path to generated comparison report
    """
    dashboard = ModelComparisonDashboard(output_dir)
    report_id = dashboard.load_evaluation_results(results_dir)
    return dashboard.generate_comparison_report(report_id, include_visualizations)