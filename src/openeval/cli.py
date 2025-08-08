from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .core import Adapter, Dataset, Metric, Task
from .spec import load_spec

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def version():
    from importlib.metadata import version as _v

    print({"openeval": _v("openeval-lab")})


@app.command()
def run(spec: Path = typer.Argument(..., help="Path to JSON spec")):
    """Run an evaluation from a JSON spec file."""
    try:
        task, dataset, adapter, metrics, out = load_spec(spec)
    except SystemExit as e:
        raise typer.Exit(code=2) from e

    result = task.evaluate(adapter, dataset, metrics)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print({"saved": out})


if __name__ == "__main__":
    app()
