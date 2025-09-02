from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from .core import Adapter, Dataset, Metric, Task
from .datasets.inline import InlineDataset
from .registry import lookup

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


class MetricSpec(BaseModel):
    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class EvalSpec(BaseModel):
    task: str
    dataset: Any
    adapter: str

    task_kwargs: dict[str, Any] = Field(default_factory=dict)
    dataset_kwargs: dict[str, Any] = Field(default_factory=dict)
    adapter_kwargs: dict[str, Any] = Field(default_factory=dict)

    metrics: List[MetricSpec] = Field(default_factory=list)
    output: str = "results.json"
    # Optional agent block for agentic tasks
    agent: Optional[Dict[str, Any]] = None

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()


def _load_dotted(path: str):
    # support both module.Class and module:Class (colon)
    dotted = path.replace(":", ".")
    mod_name, cls_name = dotted.rsplit(".", 1)
    mod = import_module(mod_name)
    return getattr(mod, cls_name)


def _resolve_or_load(kind: str, value: str):
    # allow short names via registry (e.g., 'qa', 'jsonl', 'echo', 'exact_match')
    dotted = lookup(kind, value) or value
    return _load_dotted(dotted)


# Simple import hook for tests and external callers
def import_class(path: str):  # pragma: no cover - thin wrapper, patched in tests
    """Import a class from a dotted or colon path.

    Provided for compatibility with tests that patch `openeval.spec.import_class`.
    Uses the same resolution rules as internal helpers.
    """
    # Try direct dotted import first
    try:
        return _load_dotted(path)
    except Exception:
        # As a fallback, attempt to resolve via registry without kind hint
        # Common kinds to try in order
        for kind in ("task", "dataset", "adapter", "metric"):
            try:
                return _resolve_or_load(kind, path)
            except Exception:
                continue
        # Re-raise original style error if nothing matched
        raise


def _read_spec_file(p: Path) -> dict[str, Any]:
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise SystemExit("PyYAML not installed; install openeval-lab to parse YAML specs.")
        return yaml.safe_load(p.read_text())
    return json.loads(p.read_text())


def load_spec(path: Path | str) -> Tuple[Task, Dataset, Adapter, List[Metric], str]:
    p = Path(path)
    data = _read_spec_file(p)
    # Normalize metrics to dicts if provided as strings
    metrics_raw = data.get("metrics")
    if isinstance(metrics_raw, list) and metrics_raw and isinstance(metrics_raw[0], str):
        data["metrics"] = [{"name": m} for m in metrics_raw]

    # If agent block present, map to task_kwargs for ToolUseTask
    agent_block = data.get("agent")
    if agent_block and data.get("task") in (
        "openeval.tasks.tooluse.ToolUseTask",
        "openeval.tasks.tooluse:ToolUseTask",
        "tool_use",
    ):
        tk = data.get("task_kwargs") or {}
        if "agent_type" not in tk:
            tk["agent_type"] = agent_block.get("type")
        if "tools" not in tk and isinstance(agent_block.get("tools"), list):
            tk["tools"] = agent_block.get("tools")
        data["task_kwargs"] = tk

    try:
        spec = EvalSpec(**data)
    except ValidationError as e:
        raise SystemExit(f"Invalid spec: {e}")

    # Route through import_class so tests can patch it and short names work
    task_cls = import_class(spec.task)
    # Dataset can be: short/dotted string or inline object
    if isinstance(spec.dataset, str):
        dataset_cls = import_class(spec.dataset)
        dataset: Dataset = dataset_cls(**spec.dataset_kwargs)
    elif isinstance(spec.dataset, dict) and (spec.dataset.get("type") == "inline"):
        dataset = InlineDataset(name=spec.dataset.get("name", "inline"), examples=spec.dataset.get("examples", []))
    else:
        # Unsupported dataset type provided
        raise SystemExit(
            "Invalid dataset value in spec: expected short/dotted string or inline object."
        )
    adapter_cls = import_class(spec.adapter)

    task: Task = task_cls(**spec.task_kwargs)
    adapter: Adapter = adapter_cls(**spec.adapter_kwargs)

    metrics: list[Metric] = []
    for m in spec.metrics:
        m_cls = import_class(m.name)
        metrics.append(m_cls(**m.kwargs))

    # Validate agent tooling for ToolUseTask
    if task.__class__.__module__ == "openeval.tasks.tooluse" and task.__class__.__name__ == "ToolUseTask":
        # If spec provided agent block, we already mapped it into task_kwargs
        if not getattr(task, "_agent_type", None):
            raise SystemExit("ToolUseTask requires 'agent.type' in spec (mapped to task_kwargs.agent_type)")
        if not getattr(task, "_tool_types", None):
            raise SystemExit("ToolUseTask requires 'agent.tools' list in spec (mapped to task_kwargs.tools)")

    return task, dataset, adapter, metrics, spec.output
