from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, List, Tuple

from pydantic import BaseModel, Field, ValidationError

from .core import Adapter, Dataset, Metric, Task
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
    dataset: str
    adapter: str

    task_kwargs: dict[str, Any] = Field(default_factory=dict)
    dataset_kwargs: dict[str, Any] = Field(default_factory=dict)
    adapter_kwargs: dict[str, Any] = Field(default_factory=dict)

    metrics: List[MetricSpec] = Field(default_factory=list)
    output: str = "results.json"

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


def _read_spec_file(p: Path) -> dict[str, Any]:
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise SystemExit("PyYAML not installed; install openeval-lab to parse YAML specs.")
        return yaml.safe_load(p.read_text())
    return json.loads(p.read_text())


def load_spec(path: Path | str) -> Tuple[Task, Dataset, Adapter, List[Metric], str]:
    p = Path(path)
    data = _read_spec_file(p)
    try:
        spec = EvalSpec(**data)
    except ValidationError as e:
        raise SystemExit(f"Invalid spec: {e}")

    task_cls = _resolve_or_load("task", spec.task)
    dataset_cls = _resolve_or_load("dataset", spec.dataset)
    adapter_cls = _resolve_or_load("adapter", spec.adapter)

    task: Task = task_cls(**spec.task_kwargs)
    dataset: Dataset = dataset_cls(**spec.dataset_kwargs)
    adapter: Adapter = adapter_cls(**spec.adapter_kwargs)

    metrics: list[Metric] = []
    for m in spec.metrics:
        m_cls = _resolve_or_load("metric", m.name)
        metrics.append(m_cls(**m.kwargs))

    return task, dataset, adapter, metrics, spec.output
