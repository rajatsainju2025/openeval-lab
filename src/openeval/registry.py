from __future__ import annotations

from typing import Any, Dict, Type, Optional

# Central registry for tasks, datasets, adapters, and metrics.
# Users can reference these by short names in specs (e.g., "qa", "jsonl", "echo", "exact_match").

# Import lazily-resolved classes to avoid heavy imports at module import time.

TASKS: Dict[str, str] = {
    "qa": "openeval.tasks.qa.QATask",
    "summarization": "openeval.tasks.summarization.SummarizationTask",
    "tool_use": "openeval.tasks.tooluse.ToolUseTask",
}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "qa": "Basic question answering with exact-match and F1 metrics.",
    "summarization": "Summarization task supporting ROUGE/BERTScore.",
    "tool_use": "Agent tool-use task with function-call evaluation.",
}

DATASETS: Dict[str, str] = {
    "jsonl": "openeval.datasets.jsonl.JSONLinesDataset",
    "csv": "openeval.datasets.csv.CSVDataset",
    "hf": "openeval.datasets.hf.HFDataset",
    "inline": "openeval.datasets.inline.InlineDataset",
}

DATASET_DESCRIPTIONS: Dict[str, str] = {
    "jsonl": "JSON Lines file with one sample per line.",
    "csv": "CSV file with configurable field mappings.",
    "hf": "Hugging Face datasets loader.",
    "inline": "Inline list of examples embedded in the spec.",
}

ADAPTERS: Dict[str, str] = {
    "echo": "openeval.adapters.echo.EchoAdapter",
    "openai-chat": "openeval.adapters.openai.chat_completions.OpenAIChatAdapter",
}

ADAPTER_DESCRIPTIONS: Dict[str, str] = {
    "echo": "Deterministic adapter that echos the prompt (for testing).",
    "openai-chat": "OpenAI Chat Completions API adapter.",
}

METRICS: Dict[str, str] = {
    "exact_match": "openeval.metrics.accuracy.ExactMatch",
    "token_f1": "openeval.metrics.accuracy.TokenF1",
    "sacrebleu": "openeval.metrics.bleu.SacreBLEU",
    "bertscore": "openeval.metrics.bertscore.BERTScore",
    "rouge_l": "openeval.metrics.rouge.ROUGEL",
    "llm_judge": "openeval.metrics.judge.LLMJudge",
    "tool_execution": "openeval.metrics.tool_execution.ToolExecutionMetric",
    "char_edit": "openeval.metrics.edit_distance.CharEditDistance",
}

METRIC_DESCRIPTIONS: Dict[str, str] = {
    "exact_match": "Exact string match between prediction and reference.",
    "token_f1": "Whitespace-tokenized F1 score.",
    "sacrebleu": "SacreBLEU machine translation metric.",
    "bertscore": "Semantic similarity via BERTScore.",
    "rouge_l": "ROUGE-L summarization metric.",
    "llm_judge": "LLM-as-a-judge with configurable rubric.",
    "tool_execution": "Validate agent tool invocations and outputs.",
    "char_edit": "Character edit distance and similarity.",
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


def _get_desc_map(kind: str) -> Dict[str, str]:
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
    """Return metadata about a registry item: {name, path, description}."""
    path = lookup(kind, name)
    if path is None:
        return None
    desc = _get_desc_map(kind).get(name, "")
    return {"name": name, "path": path, "description": desc}


def list_items(kind: str) -> Dict[str, Dict[str, str]]:
    """List all items for a kind with metadata keyed by short name."""
    mp = _get_map(kind)
    dm = _get_desc_map(kind)
    return {k: {"name": k, "path": v, "description": dm.get(k, "")} for k, v in mp.items()}
