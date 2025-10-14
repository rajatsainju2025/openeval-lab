from __future__ import annotations

from typing import Dict, Type, Optional, TypeVar, Union
import functools

from openeval.core import Task, Dataset, Adapter, Metric

"""
Registry module for OpenEval Lab components.

This module provides a centralized registry system for tasks, datasets, adapters, and metrics.
It uses lazy loading to avoid heavy imports at startup time. Users can reference registered
components using short names in their evaluation specs.

Examples:
    Using registered components in a spec:
    ```python
    spec = {
        "task": "qa",              # Uses QATask
        "dataset": "jsonl",        # Uses JSONLinesDataset
        "adapter": "echo",         # Uses EchoAdapter
        "metrics": ["exact_match"] # Uses ExactMatchMetric
    }
    ```

Note:
    All registry entries use string paths to enable lazy loading. The actual classes
    are only imported when needed, improving startup performance.
"""

# Type variables for improved type safety
T = TypeVar("T")
TaskType = TypeVar("TaskType", bound=Task)
DatasetType = TypeVar("DatasetType", bound=Dataset)
AdapterType = TypeVar("AdapterType", bound=Adapter)
MetricType = TypeVar("MetricType", bound=Metric)

# Registry of evaluation tasks and their import paths
TASKS: Dict[str, str] = {
    "qa": "openeval.tasks.qa.QATask",  # Question-answering task
    "summarization": "openeval.tasks.summarization.SummarizationTask",  # Text summarization
    "tool_use": "openeval.tasks.tooluse.ToolUseTask",  # Tool-use evaluation
}

# Human-readable descriptions of available tasks
TASK_DESCRIPTIONS: Dict[str, str] = {
    "qa": "Basic question answering with exact-match and F1 metrics.",
    "summarization": "Summarization task supporting ROUGE/BERTScore.",
    "tool_use": "Agent tool-use task with function-call evaluation.",
}

# Registry of dataset loaders and their import paths
DATASETS: Dict[str, str] = {
    "jsonl": "openeval.datasets.jsonl.JSONLinesDataset",  # JSON Lines format
    "csv": "openeval.datasets.csv.CSVDataset",  # CSV format
    "hf": "openeval.datasets.hf.HFDataset",  # HuggingFace datasets
    "inline": "openeval.datasets.inline.InlineDataset",  # Inline data
}

# Human-readable descriptions of available dataset formats
DATASET_DESCRIPTIONS: Dict[str, str] = {
    "jsonl": "JSON Lines file with one sample per line.",
    "csv": "CSV file with configurable field mappings.",
    "hf": "Hugging Face datasets loader.",
    "inline": "Inline list of examples embedded in the spec.",
}

ADAPTERS: Dict[str, str] = {
    "echo": "openeval.adapters.echo.EchoAdapter",
    "openai-chat": "openeval.adapters.openai.chat_completions.OpenAIChatAdapter",
    "anthropic": "openeval.adapters.anthropic_adapter.AnthropicAdapter",
    "huggingface": "openeval.adapters.huggingface_adapter.HuggingFaceAdapter",
    "local-api": "openeval.adapters.local_api_adapter.LocalAPIAdapter",
    "multimodal": "openeval.adapters.multimodal_adapter.MultimodalAdapter",
    "vllm": "openeval.adapters.vllm_adapter.VLLMAdapter",
}

ADAPTER_DESCRIPTIONS: Dict[str, str] = {
    "echo": "Deterministic adapter that echos the prompt (for testing).",
    "openai-chat": "OpenAI Chat Completions API adapter.",
    "anthropic": "Anthropic Claude API adapter.",
    "huggingface": "Hugging Face Transformers model adapter.",
    "local-api": "Local model server API adapter.",
    "multimodal": "Multimodal model adapter for vision-language tasks.",
    "vllm": "vLLM inference server adapter for high-throughput evaluation.",
}

METRICS: Dict[str, str] = {
    "exact_match": "openeval.metrics.accuracy.ExactMatch",
    "token_f1": "openeval.metrics.accuracy.TokenF1",
    "f1_score": "openeval.metrics.f1_score.F1Score",
    "sacrebleu": "openeval.metrics.bleu.SacreBLEU",
    "bertscore": "openeval.metrics.bertscore.BERTScore",
    "rouge_l": "openeval.metrics.rouge.ROUGEL",
    "llm_judge": "openeval.metrics.judge.LLMJudge",
    "tool_execution": "openeval.metrics.tool_execution.ToolExecutionMetric",
    "char_edit": "openeval.metrics.edit_distance.CharEditDistance",
    "calibration_error": "openeval.metrics.calibration.CalibrationError",
    "loglik_accuracy": "openeval.metrics.loglik_accuracy.LogLikelihoodAccuracy",
    "code_execution": "openeval.metrics.code_execution.CodeExecutionMetric",
}

METRIC_DESCRIPTIONS: Dict[str, str] = {
    "exact_match": "Exact string match between prediction and reference.",
    "token_f1": "Whitespace-tokenized F1 score.",
    "f1_score": "Token-level F1 score with precision and recall balancing.",
    "sacrebleu": "SacreBLEU machine translation metric.",
    "bertscore": "Semantic similarity via BERTScore.",
    "rouge_l": "ROUGE-L summarization metric.",
    "llm_judge": "LLM-as-a-judge with configurable rubric.",
    "tool_execution": "Validate agent tool invocations and outputs.",
    "char_edit": "Character edit distance and similarity.",
    "calibration_error": "Expected Calibration Error for confidence assessment.",
    "loglik_accuracy": "Log-likelihood based accuracy metric.",
    "code_execution": "Code execution and correctness validation.",
}


def _get_map(kind: str) -> Dict[str, str]:
    """
    Get the appropriate registry map for a given component kind.

    Args:
        kind: The type of component ("task", "dataset", "adapter", or "metric")

    Returns:
        A dictionary mapping short names to import paths

    Raises:
        KeyError: If the kind is not recognized
    """
    # Use a single mapping to avoid multiple comparisons on hot paths.
    _REGISTRY_MAP: Dict[str, Dict[str, str]] = {
        "task": TASKS,
        "dataset": DATASETS,
        "adapter": ADAPTERS,
        "metric": METRICS,
    }

    try:
        return _REGISTRY_MAP[kind]
    except KeyError as exc:
        raise KeyError(f"Unknown registry kind: {kind}") from exc


def lookup(kind: str, name: str) -> Optional[str]:
    """
    Look up the import path for a registered component.

    Args:
        kind: The type of component ("task", "dataset", "adapter", or "metric")
        name: The short name of the component (e.g., "qa", "jsonl", "openai-chat")

    Returns:
        The dotted import path if the component is registered, None otherwise

    Example:
        >>> lookup("task", "qa")
        'openeval.tasks.qa.QATask'
        >>> lookup("dataset", "unknown")
        None
    """
    return _get_map(kind).get(name)


def _get_desc_map(kind: str) -> Dict[str, str]:
    """
    Get the appropriate description map for a given component kind.

    Args:
        kind: The type of component ("task", "dataset", "adapter", or "metric")

    Returns:
        A dictionary mapping short names to human-readable descriptions

    Raises:
        KeyError: If the kind is not recognized
    """
    if kind == "task":
        return TASK_DESCRIPTIONS
    if kind == "dataset":
        return DATASET_DESCRIPTIONS
    if kind == "adapter":
        return ADAPTER_DESCRIPTIONS
    if kind == "metric":
        return METRIC_DESCRIPTIONS
    raise KeyError(f"Unknown registry kind: {kind}")


def info(kind: str, name: str) -> Optional[Dict[str, str]]:
    """
    Get metadata about a registered component.

    Args:
        kind: The type of component ("task", "dataset", "adapter", or "metric")
        name: The short name of the component

    Returns:
        A dictionary containing metadata (name, path, description) if registered,
        None otherwise

    Example:
        >>> info("task", "qa")
        {
            'name': 'qa',
            'path': 'openeval.tasks.qa.QATask',
            'description': 'Basic question answering with exact-match and F1 metrics.'
        }
    """
    path = lookup(kind, name)
    if path is None:
        return None
    desc = _get_desc_map(kind).get(name, "")
    return {"name": name, "path": path, "description": desc}


def load_component(
    kind: str, name: str
) -> Optional[Union[Type[Task], Type[Dataset], Type[Adapter], Type[Metric]]]:
    """
    Dynamically load and return a component class from the registry.

    This function uses importlib to dynamically import and load component classes
    based on their registry entries. It performs lazy loading to avoid importing
    heavy dependencies until they are actually needed.

    Args:
        kind: The type of component ("task", "dataset", "adapter", or "metric")
        name: The short name of the component to load

    Returns:
        The loaded component class if found and successfully imported,
        None otherwise. Return type will be one of:
        - Type[TaskType] for tasks
        - Type[DatasetType] for datasets
        - Type[AdapterType] for adapters
        - Type[MetricType] for metrics

    Example:
        >>> QATask = load_component("task", "qa")
        >>> task = QATask(config)  # Instantiate the loaded class

    Note:
        This function catches and logs import errors, making it safe to use
        for components with optional dependencies. If a component fails to load,
        the error will be logged and None will be returned.
    """
    return _load_component_cached(kind, name)


@functools.lru_cache(maxsize=128)
def _load_component_cached(
    kind: str, name: str
) -> Optional[Union[Type[Task], Type[Dataset], Type[Adapter], Type[Metric]]]:
    """
    Cached version of load_component to avoid repeated dynamic imports.

    This function is cached to prevent repeated imports of the same components,
    which can be expensive for large libraries or when called frequently.
    """
    import importlib

    path = lookup(kind, name)
    if not path:
        return None

    try:
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        import logging

        logging.error(f"Failed to load {kind} '{name}' from {path}: {e}")
        return None
    return {"name": name, "path": path, "description": _get_desc_map(kind).get(name, "")}


def list_items(kind: str) -> Dict[str, Dict[str, str]]:
    """List all items for a kind with metadata keyed by short name."""
    mp = _get_map(kind)
    dm = _get_desc_map(kind)
    return {k: {"name": k, "path": v, "description": dm.get(k, "")} for k, v in mp.items()}
