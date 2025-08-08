from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .core import Adapter, Dataset, Metric, Task
from .spec import EvalSpec, load_spec

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def version():
    from importlib.metadata import version as _v

    print({"openeval": _v("openeval-lab")})


@app.command()
def schema(out: Optional[Path] = typer.Option(None, "--out", help="Write JSON schema to file")):
    """Print the JSON schema for experiment specs."""
    sch = EvalSpec.model_json_schema()
    payload = json.dumps(sch, indent=2)
    if out:
        out.write_text(payload)
        print({"saved": str(out)})
    else:
        print(payload)


@app.command()
def init(
    out: Path = typer.Argument(..., help="Path to write a starter spec (json or yaml)"),
    fmt: str = typer.Option("json", help="Format: json|yaml"),
):
    """Generate a starter spec file."""
    ex = {
        "task": "openeval.tasks.qa.QATask",
        "dataset": "openeval.datasets.jsonl.JSONLinesDataset",
        "adapter": "openeval.adapters.echo.EchoAdapter",
        "dataset_kwargs": {"path": "examples/qa_toy.jsonl"},
        "metrics": [{"name": "openeval.metrics.accuracy.ExactMatch"}],
        "output": "results.json",
    }
    if fmt.lower() == "yaml" or out.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise typer.Exit(code=2)
        out.write_text(yaml.safe_dump(ex, sort_keys=False))
    else:
        out.write_text(json.dumps(ex, indent=2))
    print({"saved": str(out)})


@app.command()
def run(
    spec: Path = typer.Argument(..., help="Path to JSON/YAML spec"),
    seed: Optional[int] = typer.Option(0, help="Deterministic seed"),
):
    """Run an evaluation from a spec file."""
    try:
        task, dataset, adapter, metrics, out = load_spec(spec)
    except SystemExit as e:
        raise typer.Exit(code=2) from e

    result = task.evaluate(adapter, dataset, metrics, seed=seed)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print({"saved": out})


if __name__ == "__main__":
    app()
