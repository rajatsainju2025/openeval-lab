from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, List, Tuple

from pydantic import BaseModel, Field, ValidationError

from .core import Adapter, Dataset, Metric, Task


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


def _load_dotted(path: str):
    # support both module.Class and module:Class (colon)
    dotted = path.replace(":", ".")
    mod_name, cls_name = dotted.rsplit(".", 1)
    mod = import_module(mod_name)
    return getattr(mod, cls_name)


def load_spec(path: Path | str) -> Tuple[Task, Dataset, Adapter, List[Metric], str]:
    p = Path(path)
    data = json.loads(p.read_text())
    try:
        spec = EvalSpec(**data)
    except ValidationError as e:
        raise SystemExit(f"Invalid spec: {e}")

    task_cls = _load_dotted(spec.task)
    dataset_cls = _load_dotted(spec.dataset)
    adapter_cls = _load_dotted(spec.adapter)

    task: Task = task_cls(**spec.task_kwargs)
    dataset: Dataset = dataset_cls(**spec.dataset_kwargs)
    adapter: Adapter = adapter_cls(**spec.adapter_kwargs)

    metrics: list[Metric] = []
    for m in spec.metrics:
        m_cls = _load_dotted(m.name)
        metrics.append(m_cls(**m.kwargs))

    return task, dataset, adapter, metrics, spec.output
