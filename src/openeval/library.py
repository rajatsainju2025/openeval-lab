"""
Curated library of standard evaluation tasks and benchmarks.

This module provides pre-configured tasks, datasets, and metrics
for common evaluation scenarios, making it easy to run standard
benchmarks without manual configuration.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from .core import Task, Dataset, Adapter, Metric
from .spec import EvalSpec


class TaskLibrary:
    """Central registry for curated evaluation tasks."""

    def __init__(self):
        self._tasks = {}
        self._categories = {}
        self._initialize_tasks()

    def _initialize_tasks(self):
        """Initialize the curated task library."""

        # Question Answering tasks
        self.register_task(
            "qa_basic",
            category="question_answering",
            description="Basic question answering with exact match",
            spec={
                "task": "openeval.tasks.qa:QATask",
                "dataset": "openeval.datasets.jsonl:JSONLDataset",
                "adapter": "openeval.adapters.echo:EchoAdapter",
                "dataset_kwargs": {"path": "examples/qa_toy.jsonl"},
                "metrics": [
                    {"name": "openeval.metrics.exact_match:ExactMatch"},
                    {"name": "openeval.metrics.token_f1:TokenF1"},
                ],
            },
        )

        self.register_task(
            "qa_judge",
            category="question_answering",
            description="Question answering with LLM judge evaluation",
            spec={
                "task": "openeval.tasks.qa:QATask",
                "dataset": "openeval.datasets.jsonl:JSONLDataset",
                "adapter": "openeval.adapters.echo:EchoAdapter",
                "dataset_kwargs": {"path": "examples/qa_toy.jsonl"},
                "metrics": [
                    {
                        "name": "openeval.metrics.llm_judge:LLMJudge",
                        "kwargs": {
                            "judge_prompt": "Rate the answer quality on a scale of 1-5",
                            "criteria": ["accuracy", "completeness", "clarity"],
                        },
                    }
                ],
            },
        )

        # Multiple Choice tasks
        self.register_task(
            "mcqa_standard",
            category="multiple_choice",
            description="Multiple choice QA with text generation",
            spec={
                "task": "openeval.tasks.mcqa:MCQATask",
                "dataset": "openeval.datasets.mcqa:MCQADataset",
                "adapter": "openeval.adapters.echo:EchoAdapter",
                "dataset_kwargs": {"path": "examples/mcqa_toy.jsonl"},
                "metrics": [{"name": "openeval.metrics.exact_match:ExactMatch"}],
            },
        )

        self.register_task(
            "mcqa_loglik",
            category="multiple_choice",
            description="Multiple choice QA with log-likelihood evaluation",
            spec={
                "task": "openeval.tasks.mcqa:MCQATask",
                "dataset": "openeval.datasets.mcqa:MCQADataset",
                "adapter": "openeval.loglikelihood:MockLogLikelihoodAdapter",
                "dataset_kwargs": {"path": "examples/mcqa_toy.jsonl"},
                "metrics": [{"name": "openeval.metrics.loglik_accuracy:LogLikelihoodAccuracy"}],
            },
        )

        # Code Generation tasks
        self.register_task(
            "code_humaneval",
            category="code_generation",
            description="HumanEval-style code generation with execution",
            spec={
                "task": "openeval.tasks.code:HumanEvalTask",
                "dataset": "openeval.datasets.code:HumanEvalDataset",
                "adapter": "openeval.adapters.echo:EchoAdapter",
                "dataset_kwargs": {"path": "examples/code_toy.jsonl"},
                "metrics": [
                    {
                        "name": "openeval.metrics.code_execution:HumanEvalMetric",
                        "kwargs": {"timeout": 3.0, "k_values": [1, 10, 100]},
                    }
                ],
            },
        )

        self.register_task(
            "code_basic",
            category="code_generation",
            description="Basic code generation without execution",
            spec={
                "task": "openeval.tasks.code:CodeGenerationTask",
                "dataset": "openeval.datasets.code:CodeDataset",
                "adapter": "openeval.adapters.echo:EchoAdapter",
                "dataset_kwargs": {"path": "examples/code_toy.jsonl"},
                "metrics": [
                    {"name": "openeval.metrics.exact_match:ExactMatch"},
                    {"name": "openeval.metrics.token_f1:TokenF1"},
                ],
            },
        )

        # Summarization tasks
        self.register_task(
            "summarization_rouge",
            category="summarization",
            description="Text summarization with ROUGE metrics",
            spec={
                "task": "openeval.tasks.summarization:SummarizationTask",
                "dataset": "openeval.datasets.jsonl:JSONLDataset",
                "adapter": "openeval.adapters.echo:EchoAdapter",
                "dataset_kwargs": {"path": "examples/sum_toy.jsonl"},
                "metrics": [{"name": "openeval.metrics.rouge:ROUGEL"}],
            },
        )

        # Benchmark suites
        self.register_task(
            "comprehensive_nlp",
            category="benchmark_suite",
            description="Comprehensive NLP evaluation across multiple tasks",
            spec={
                "tasks": [
                    {"task_id": "qa_basic", "weight": 0.3},
                    {"task_id": "mcqa_standard", "weight": 0.3},
                    {"task_id": "summarization_rouge", "weight": 0.4},
                ],
                "aggregate_metrics": ["accuracy", "f1_score", "rouge_l"],
            },
        )

    def register_task(
        self,
        task_id: str,
        category: str,
        description: str,
        spec: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ):
        """Register a new task in the library."""
        self._tasks[task_id] = {
            "id": task_id,
            "category": category,
            "description": description,
            "spec": spec,
            "tags": tags or [],
        }

        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(task_id)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_task_spec(self, task_id: str) -> Optional[EvalSpec]:
        """Get a task spec as EvalSpec object."""
        task = self.get_task(task_id)
        if task:
            return EvalSpec(**task["spec"])
        return None

    def list_tasks(
        self, category: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List available tasks, optionally filtered by category or tags."""
        tasks = list(self._tasks.values())

        if category:
            tasks = [t for t in tasks if t["category"] == category]

        if tags:
            tasks = [t for t in tasks if any(tag in t.get("tags", []) for tag in tags)]

        return tasks

    def list_categories(self) -> List[str]:
        """List available task categories."""
        return list(self._categories.keys())

    def get_category_tasks(self, category: str) -> List[str]:
        """Get task IDs in a specific category."""
        return self._categories.get(category, [])

    def search_tasks(self, query: str) -> List[Dict[str, Any]]:
        """Search tasks by description or ID."""
        query = query.lower()
        results = []

        for task in self._tasks.values():
            if (
                query in task["id"].lower()
                or query in task["description"].lower()
                or query in task["category"].lower()
            ):
                results.append(task)

        return results

    def export_task(self, task_id: str, output_path: str):
        """Export a task spec to a JSON file."""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        with open(output_path, "w") as f:
            json.dump(task["spec"], f, indent=2)

    def import_task(
        self,
        task_id: str,
        spec_path: str,
        category: str,
        description: str,
        tags: Optional[List[str]] = None,
    ):
        """Import a task from a JSON spec file."""
        with open(spec_path, "r") as f:
            spec = json.load(f)

        self.register_task(task_id, category, description, spec, tags)


# Global task library instance
task_library = TaskLibrary()


def get_task_library() -> TaskLibrary:
    """Get the global task library instance."""
    return task_library


def list_available_tasks() -> List[str]:
    """Get list of all available task IDs."""
    return list(task_library._tasks.keys())


def get_task_info(task_id: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific task."""
    return task_library.get_task(task_id)


def run_standard_benchmark(benchmark_name: str, adapter: Adapter, **kwargs) -> Dict[str, Any]:
    """
    Run a standard benchmark by name.

    Args:
        benchmark_name: Name of the benchmark task
        adapter: Adapter to use for evaluation
        **kwargs: Additional arguments for evaluation

    Returns:
        Evaluation results
    """
    # For now, return spec for manual evaluation
    # This will be fully implemented when core run_evaluation is available
    spec = task_library.get_task_spec(benchmark_name)
    if not spec:
        raise ValueError(f"Benchmark {benchmark_name} not found")

    return {
        "benchmark_name": benchmark_name,
        "spec": spec.model_dump() if spec else None,
        "adapter_info": f"{adapter.__class__.__module__}:{adapter.__class__.__name__}",
        "status": "spec_ready",
        "note": "Use this spec with the main evaluation pipeline",
    }


class BenchmarkSuite:
    """Collection of related benchmarks that can be run together."""

    def __init__(self, name: str, task_ids: List[str], weights: Optional[List[float]] = None):
        self.name = name
        self.task_ids = task_ids
        self.weights = weights or [1.0] * len(task_ids)

        if len(self.weights) != len(self.task_ids):
            raise ValueError("Number of weights must match number of tasks")

    def run(self, adapter: Adapter, **kwargs) -> Dict[str, Any]:
        """Run all tasks in the suite."""
        results = {}
        total_score = 0.0

        for task_id, weight in zip(self.task_ids, self.weights):
            print(f"Running {task_id}...")
            # For now, just get task spec - full integration pending
            task_spec = task_library.get_task_spec(task_id)
            if task_spec:
                results[task_id] = {"spec": task_spec.model_dump(), "status": "spec_ready"}
                # Mock score for testing
                total_score += 0.8 * weight  # Mock 80% score

        # Calculate weighted average
        weighted_score = total_score / sum(self.weights)

        return {
            "suite_name": self.name,
            "weighted_score": weighted_score,
            "task_results": results,
            "task_weights": dict(zip(self.task_ids, self.weights)),
            "status": "specs_ready",
            "note": "Use specs with main evaluation pipeline",
        }

    def _extract_primary_score(self, metrics: Dict[str, Any]) -> float:
        """Extract primary score from metrics."""
        # Priority order for selecting primary metric
        priority_metrics = ["accuracy", "f1_score", "rouge_l", "pass@1", "humaneval_score"]

        for metric_name in priority_metrics:
            if metric_name in metrics:
                score = metrics[metric_name]
                return float(score) if isinstance(score, (int, float)) else 0.0

        # Fallback: use first numeric metric
        for value in metrics.values():
            if isinstance(value, (int, float)):
                return float(value)

        return 0.0


# Pre-defined benchmark suites
NLP_BASICS_SUITE = BenchmarkSuite(
    "nlp_basics", ["qa_basic", "mcqa_standard", "summarization_rouge"], [0.4, 0.3, 0.3]
)

CODE_SUITE = BenchmarkSuite("code_generation", ["code_humaneval", "code_basic"], [0.8, 0.2])

COMPREHENSIVE_SUITE = BenchmarkSuite(
    "comprehensive",
    ["qa_judge", "mcqa_loglik", "code_humaneval", "summarization_rouge"],
    [0.25, 0.25, 0.25, 0.25],
)
