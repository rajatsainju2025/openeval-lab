from __future__ import annotations

from typing import Any, Dict, Type

# Central registry for tasks, datasets, adapters, and metrics.
# Users can reference these by short names in specs (e.g., "qa", "jsonl", "echo", "exact_match").

# Import lazily-resolved classes to avoid heavy imports at module import time.

TASKS: Dict[str, str] = {
    "qa": "openeval.tasks.qa.QATask",
    "summarization": "openeval.tasks.summarization.SummarizationTask",
}

DATASETS: Dict[str, str] = {
    "jsonl": "openeval.datasets.jsonl.JSONLinesDataset",
    "csv": "openeval.datasets.csv.CSVDataset",
    "hf": "openeval.datasets.hf.HFDataset",
}

ADAPTERS: Dict[str, str] = {
    "echo": "openeval.adapters.echo.EchoAdapter",
    "openai-chat": "openeval.adapters.openai.chat_completions.OpenAIChatAdapter",
}

METRICS: Dict[str, str] = {
    "exact_match": "openeval.metrics.accuracy.ExactMatch",
    "token_f1": "openeval.metrics.accuracy.TokenF1",
    "sacrebleu": "openeval.metrics.bleu.SacreBLEU",
    "bertscore": "openeval.metrics.bertscore.BERTScore",
    "rouge_l": "openeval.metrics.rouge.ROUGEL",
}


def _get_map(kind: str) -> Dict[str, str]:
    if kind == "task":
        return TASKS
    if kind == "dataset":
        return DATASETS
    if kind == "adapter":
        return ADAPTERS
    if kind == "metric":
        return METRICS
    raise KeyError(f"Unknown registry kind: {kind}")


def lookup(kind: str, name: str) -> str | None:
    """Return the dotted path for a registered short name if present."""
    return _get_map(kind).get(name)
