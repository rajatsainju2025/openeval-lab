from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .core import Adapter, Dataset, Metric, Task

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def version():
    from importlib.metadata import version as _v

    print({"openeval": _v("openeval-lab")})


@app.command()
def run(spec: Path = typer.Argument(..., help="Path to JSON spec")):
    """Run an evaluation from a JSON spec file."""
    with open(spec, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Simple dynamic import based on dotted paths in spec
    def load(dotted: str):
        mod_name, cls_name = dotted.rsplit(".", 1)
        mod = __import__(mod_name, fromlist=[cls_name])
        return getattr(mod, cls_name)

    task_cls = load(cfg["task"])  # e.g., "openeval.tasks.qa:QATask" -> use dot notation
    dataset_cls = load(cfg["dataset"])  # e.g., "openeval.datasets.jsonl:JSONLinesDataset"
    adapter_cls = load(cfg["adapter"])  # e.g., "openeval.adapters.echo:EchoAdapter"

    task: Task = task_cls(**cfg.get("task_kwargs", {}))
    dataset: Dataset = dataset_cls(**cfg.get("dataset_kwargs", {}))
    adapter: Adapter = adapter_cls(**cfg.get("adapter_kwargs", {}))

    metrics: list[Metric] = []
    for m in cfg.get("metrics", []):
        metric_cls = load(m["name"])  # dotted path
        metrics.append(metric_cls(**m.get("kwargs", {})))

    result = task.evaluate(adapter, dataset, metrics)
    out = cfg.get("output", "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print({"saved": out})


if __name__ == "__main__":
    app()
