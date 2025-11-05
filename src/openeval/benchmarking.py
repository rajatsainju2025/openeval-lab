"""Model comparison and benchmarking suite for OpenEval Lab."""

import json
import time
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    stats = None
    HAS_SCIPY = False

from .core import Task, Dataset, Adapter, Metric
from .logging import get_logger


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    adapter_name: str
    task_name: str
    dataset_name: str
    metric_scores: Dict[str, float]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing multiple adapters."""

    benchmark_name: str
    adapters: List[str]
    results: Dict[str, Dict[str, Any]]  # adapter_name -> metrics
    rankings: Dict[str, List[str]]  # metric_name -> ranked adapter list
    statistical_significance: Dict[str, Dict[str, float]]  # metric -> adapter pairs -> p-value
    execution_summary: Dict[str, Any]


class BenchmarkSuite:
    """Comprehensive benchmarking suite for model comparison."""

    def __init__(self, name: str = "default_benchmark"):
        """Initialize benchmark suite."""
        self.name = name
        self.tasks: List[Task] = []
        self.datasets: List[Dataset] = []
        self.metrics: List[Metric] = []
        self.adapters: List[Adapter] = []
        self.logger = get_logger()

        # Configuration
        self.timeout_seconds = 300
        self.retry_attempts = 3
        self.sample_size: Optional[int] = None  # None = use full dataset

    def add_task(self, task: Task) -> "BenchmarkSuite":
        """Add a task to the benchmark suite."""
        self.tasks.append(task)
        self.logger.info(f"Added task: {task.__class__.__name__}")
        return self

    def add_dataset(self, dataset: Dataset) -> "BenchmarkSuite":
        """Add a dataset to the benchmark suite."""
        self.datasets.append(dataset)
        self.logger.info(f"Added dataset: {dataset.__class__.__name__}")
        return self

    def add_metric(self, metric: Metric) -> "BenchmarkSuite":
        """Add a metric to the benchmark suite."""
        self.metrics.append(metric)
        self.logger.info(f"Added metric: {metric.__class__.__name__}")
        return self

    def add_adapter(self, adapter: Adapter) -> "BenchmarkSuite":
        """Add an adapter to the benchmark suite."""
        self.adapters.append(adapter)
        self.logger.info(f"Added adapter: {adapter.name}")
        return self

    def run_single_benchmark(
        self, adapter: Adapter, task: Task, dataset: Dataset, metrics: List[Metric]
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        start_time = time.time()

        try:
            self.logger.info(f"Running benchmark: {adapter.name} on {task.__class__.__name__}")

            # Prepare dataset samples using streaming iterator
            import itertools

            if self.sample_size:
                samples = list(itertools.islice(dataset, self.sample_size))
            else:
                samples = list(dataset)

            # Build prompts for all samples
            prompts = []
            references = []
            cache_keys = []

            for sample in samples:
                try:
                    prompt = task.build_prompt(sample)
                    prompts.append(prompt)
                    references.append(sample.reference)
                    # Generate cache key based on prompt
                    cache_keys.append(f"{adapter.name}:{hash(prompt)}")
                except Exception as e:
                    self.logger.warning(f"Prompt building failed for sample {sample.id}: {str(e)}")
                    prompts.append("")
                    references.append(sample.reference)
                    cache_keys.append(f"{adapter.name}:error:{sample.id}")

            # Use AsyncEvaluationEngine for parallel predictions
            try:
                import asyncio
                from .async_evaluation_engine import AsyncEvaluationEngine, AsyncTaskConfig

                config = AsyncTaskConfig(
                    max_concurrent_requests=min(
                        10, len(prompts)
                    ),  # Limit concurrency for benchmarks
                    request_timeout=30.0,
                    enable_progress_tracking=False,  # Disable progress for benchmarks
                )

                # Create event loop for async execution
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Run predictions in parallel
                engine = AsyncEvaluationEngine(config)
                results = loop.run_until_complete(
                    engine.evaluate_batch_optimized(adapter, prompts, cache_keys)
                )

                # Extract predictions from results
                predictions = [result.prediction for result in results]

            except Exception as e:
                self.logger.warning(f"Async evaluation failed, falling back to sync: {str(e)}")
                # Fallback to synchronous processing
                predictions = []
                for prompt in prompts:
                    try:
                        prediction = adapter.generate(prompt)
                        predictions.append(prediction)
                    except Exception as e:
                        self.logger.warning(f"Prediction failed: {str(e)}")
                        predictions.append("")

            # Compute metrics
            metric_scores = {}
            for metric in metrics:
                try:
                    scores = metric.compute(predictions, references)
                    if isinstance(scores, dict):
                        metric_scores.update(scores)
                    elif isinstance(scores, (int, float)):
                        metric_scores[metric.name] = float(scores)
                    else:
                        # Try to extract numeric value
                        try:
                            metric_scores[metric.name] = float(str(scores))
                        except Exception:
                            metric_scores[metric.name] = 0.0

                except Exception as e:
                    self.logger.warning(f"Metric computation failed for {metric.name}: {str(e)}")
                    metric_scores[metric.name] = 0.0

            execution_time = time.time() - start_time

            return BenchmarkResult(
                adapter_name=adapter.name,
                task_name=task.__class__.__name__,
                dataset_name=dataset.__class__.__name__,
                metric_scores=metric_scores,
                execution_time=execution_time,
                success=True,
                metadata={
                    "sample_count": len(samples),
                    "prediction_count": len([p for p in predictions if p]),
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Benchmark failed: {str(e)}")

            return BenchmarkResult(
                adapter_name=adapter.name,
                task_name=task.__class__.__name__,
                dataset_name=dataset.__class__.__name__,
                metric_scores={},
                execution_time=execution_time,
                success=False,
                error_message=str(e),
            )

    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run the complete benchmark suite."""
        self.logger.info(f"Starting full benchmark suite: {self.name}")

        results = []
        total_combinations = len(self.adapters) * len(self.tasks) * len(self.datasets)
        completed = 0

        for adapter in self.adapters:
            for task in self.tasks:
                for dataset in self.datasets:
                    completed += 1
                    self.logger.info(
                        f"Running benchmark {completed}/{total_combinations}: "
                        f"{adapter.name} + {task.__class__.__name__} + {dataset.__class__.__name__}"
                    )

                    result = self.run_single_benchmark(adapter, task, dataset, self.metrics)
                    results.append(result)

                    if result.success:
                        self.logger.info(f"Benchmark completed in {result.execution_time:.2f}s")
                    else:
                        self.logger.warning(f"Benchmark failed: {result.error_message}")

        self.logger.info(f"Completed full benchmark suite with {len(results)} results")
        return results

    def compare_adapters(self, results: List[BenchmarkResult]) -> ComparisonResult:
        """Compare adapter performance across all benchmarks."""
        self.logger.info("Comparing adapter performance")

        # Group results by adapter
        adapter_results = defaultdict(list)
        for result in results:
            if result.success:
                adapter_results[result.adapter_name].append(result)

        # Calculate aggregate metrics for each adapter
        adapter_metrics = {}
        all_metric_names = set()

        for adapter_name, adapter_result_list in adapter_results.items():
            metrics_by_name = defaultdict(list)

            for result in adapter_result_list:
                for metric_name, score in result.metric_scores.items():
                    metrics_by_name[metric_name].append(score)
                    all_metric_names.add(metric_name)

            # Calculate statistics for each metric
            adapter_metrics[adapter_name] = {}
            for metric_name, scores in metrics_by_name.items():
                if scores:
                    adapter_metrics[adapter_name][metric_name] = {
                        "mean": statistics.mean(scores),
                        "median": statistics.median(scores),
                        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                        "min": min(scores),
                        "max": max(scores),
                        "count": len(scores),
                    }

        # Calculate rankings for each metric
        rankings = {}
        # Pre-compute available metrics per adapter for O(1) lookups
        adapter_metric_sets = {
            adapter_name: set(metrics.keys()) for adapter_name, metrics in adapter_metrics.items()
        }

        for metric_name in all_metric_names:
            adapter_scores = []

            for adapter_name, metrics in adapter_metrics.items():
                # Use pre-computed set for O(1) membership check
                if metric_name in adapter_metric_sets[adapter_name]:
                    adapter_scores.append((adapter_name, metrics[metric_name]["mean"]))

            # Sort by mean score (descending)
            adapter_scores.sort(key=lambda x: x[1], reverse=True)
            rankings[metric_name] = [adapter_name for adapter_name, _ in adapter_scores]

        # Calculate execution time statistics
        execution_times = defaultdict(list)
        success_rates = defaultdict(lambda: {"success": 0, "total": 0})

        for result in results:
            execution_times[result.adapter_name].append(result.execution_time)
            success_rates[result.adapter_name]["total"] += 1
            if result.success:
                success_rates[result.adapter_name]["success"] += 1

        execution_summary = {}
        for adapter_name in adapter_results.keys():
            times = execution_times[adapter_name]
            rates = success_rates[adapter_name]

            execution_summary[adapter_name] = {
                "avg_execution_time": statistics.mean(times) if times else 0.0,
                "total_execution_time": sum(times),
                "success_rate": rates["success"] / rates["total"] if rates["total"] > 0 else 0.0,
                "total_runs": rates["total"],
            }

        # Calculate statistical significance
        statistical_significance = {}
        if HAS_SCIPY and stats is not None:
            for metric_name in all_metric_names:
                statistical_significance[metric_name] = {}
                # Get scores for all adapters for this metric
                for adapter_name, metrics in adapter_metrics.items():
                    if metric_name in metrics:
                        # We need raw scores, but we only have aggregates. For now, use means
                        # In a real implementation, we'd store raw scores
                        statistical_significance[metric_name][adapter_name] = 1.0  # placeholder
        else:
            statistical_significance = {metric_name: {} for metric_name in all_metric_names}

        return ComparisonResult(
            benchmark_name=self.name,
            adapters=list(adapter_results.keys()),
            results=adapter_metrics,
            rankings=rankings,
            statistical_significance=statistical_significance,
            execution_summary=execution_summary,
        )

    def generate_comparison_report(self, comparison: ComparisonResult) -> str:
        """Generate a comprehensive comparison report."""
        report = [f"# Benchmark Comparison Report: {comparison.benchmark_name}\n"]

        # Executive summary
        report.append("## Executive Summary\n")
        report.append(f"**Adapters Compared**: {len(comparison.adapters)}")
        report.append(f"**Metrics Evaluated**: {len(comparison.rankings)}")
        report.append("")

        # Overall rankings
        report.append("## Overall Rankings\n")

        for metric_name, ranking in comparison.rankings.items():
            report.append(f"### {metric_name}")
            for i, adapter_name in enumerate(ranking, 1):
                score = comparison.results[adapter_name].get(metric_name, {}).get("mean", 0.0)
                report.append(f"{i}. **{adapter_name}**: {score:.4f}")
            report.append("")

        # Detailed performance
        report.append("## Detailed Performance\n")

        for adapter_name in sorted(comparison.adapters):
            report.append(f"### {adapter_name}")

            # Execution summary
            exec_summary = comparison.execution_summary.get(adapter_name, {})
            report.append(f"- **Success Rate**: {exec_summary.get('success_rate', 0.0):.2%}")
            report.append(
                f"- **Avg Execution Time**: {exec_summary.get('avg_execution_time', 0.0):.2f}s"
            )
            report.append(f"- **Total Runs**: {exec_summary.get('total_runs', 0)}")

            # Metric scores
            adapter_metrics = comparison.results.get(adapter_name, {})
            if adapter_metrics:
                report.append("- **Metric Scores**:")
                for metric_name, stats in adapter_metrics.items():
                    mean_score = stats.get("mean", 0.0)
                    std_score = stats.get("std", 0.0)
                    report.append(f"  - {metric_name}: {mean_score:.4f} (±{std_score:.4f})")

            report.append("")

        # Performance matrix
        report.append("## Performance Matrix\n")

        if comparison.rankings:
            # Create a table
            metric_names = list(comparison.rankings.keys())
            report.append("| Adapter | " + " | ".join(metric_names) + " |")
            report.append("|---------|" + "|".join(["---------"] * len(metric_names)) + "|")

            for adapter_name in sorted(comparison.adapters):
                row = [adapter_name]
                for metric_name in metric_names:
                    score = (
                        comparison.results.get(adapter_name, {})
                        .get(metric_name, {})
                        .get("mean", 0.0)
                    )
                    row.append(f"{score:.4f}")
                report.append("| " + " | ".join(row) + " |")

            report.append("")

        return "\n".join(report)

    def save_results(
        self, results: List[BenchmarkResult], comparison: ComparisonResult, output_dir: Path
    ) -> List[Path]:
        """Save benchmark results and comparison to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        files_created = []

        # Save raw results
        results_file = output_dir / f"benchmark_results_{self.name}.json"
        results_data = []

        for result in results:
            results_data.append(
                {
                    "adapter_name": result.adapter_name,
                    "task_name": result.task_name,
                    "dataset_name": result.dataset_name,
                    "metric_scores": result.metric_scores,
                    "execution_time": result.execution_time,
                    "success": result.success,
                    "error_message": result.error_message,
                    "metadata": result.metadata,
                }
            )

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        files_created.append(results_file)

        # Save comparison results
        comparison_file = output_dir / f"comparison_{self.name}.json"
        comparison_data = {
            "benchmark_name": comparison.benchmark_name,
            "adapters": comparison.adapters,
            "results": comparison.results,
            "rankings": comparison.rankings,
            "execution_summary": comparison.execution_summary,
        }

        with open(comparison_file, "w") as f:
            json.dump(comparison_data, f, indent=2)
        files_created.append(comparison_file)

        # Save comparison report
        report_file = output_dir / f"comparison_report_{self.name}.md"
        report = self.generate_comparison_report(comparison)

        with open(report_file, "w") as f:
            f.write(report)
        files_created.append(report_file)

        self.logger.info(f"Benchmark results saved to {output_dir}")
        return files_created


class StandardBenchmarks:
    """Collection of standard benchmark configurations."""

    @staticmethod
    def create_qa_benchmark() -> BenchmarkSuite:
        """Create a standard QA benchmark suite."""
        # Import here to avoid circular dependencies
        try:
            from .tasks.qa import QATask
            from .metrics.accuracy import ExactMatch

            suite = BenchmarkSuite("qa_benchmark")
            suite.add_task(QATask())
            suite.add_metric(ExactMatch())

        except ImportError:
            # Fallback to basic suite
            suite = BenchmarkSuite("qa_benchmark")

        return suite

    @staticmethod
    def create_code_benchmark() -> BenchmarkSuite:
        """Create a standard code generation benchmark suite."""
        # Import here to avoid circular dependencies
        try:
            from .metrics.code_execution import CodeExecutionMetric

            suite = BenchmarkSuite("code_benchmark")
            # Skip task instantiation for now - would need concrete implementation
            suite.add_metric(CodeExecutionMetric())

        except ImportError:
            # Fallback to basic suite
            suite = BenchmarkSuite("code_benchmark")

        return suite

    @staticmethod
    def create_comprehensive_benchmark() -> BenchmarkSuite:
        """Create a comprehensive benchmark covering multiple domains."""
        suite = BenchmarkSuite("comprehensive_benchmark")

        # This would include multiple tasks, datasets, and metrics
        # Implementation depends on available components

        return suite
