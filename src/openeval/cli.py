from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .spec import EvalSpec, load_spec
from .utils import hash_file

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
        except Exception:  # pragma: no cover
            raise typer.Exit(code=2)
        out.write_text(yaml.safe_dump(ex, sort_keys=False))
    else:
        out.write_text(json.dumps(ex, indent=2))
    print({"saved": str(out)})


@app.command()
def run(
    spec: Path = typer.Argument(..., help="Path to JSON/YAML spec"),
    seed: Optional[int] = typer.Option(0, help="Deterministic seed"),
    records: bool = typer.Option(False, "--records", help="Include per-example records in output"),
    artifacts: Optional[Path] = typer.Option(None, "--artifacts", help="Dir to write results"),
    timestamped: bool = typer.Option(
        True, help="When writing to --artifacts, save as runs/<timestamp>.json"
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Optional label for this run"),
    concurrency: int = typer.Option(1, help="Max concurrent requests (adapters may ignore)"),
    max_retries: int = typer.Option(0, help="Max retries per request on failure"),
    request_timeout: Optional[float] = typer.Option(None, help="Timeout per request (seconds)"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Prediction cache directory"),
    cache_mode: str = typer.Option("off", "--cache", help="Cache mode: off|read|write|rw"),
    cache_ttl: Optional[float] = typer.Option(None, "--cache-ttl", help="Cache TTL seconds (optional)"),
):
    """Run an evaluation from a spec file."""
    try:
        task, dataset, adapter, metrics, out = load_spec(spec)
    except SystemExit as e:
        raise typer.Exit(code=2) from e

    # attach runtime adapter knobs when available
    _set_opts = getattr(adapter, "set_runtime_options", None)
    if callable(_set_opts):
        try:
            _set_opts(concurrency=concurrency, max_retries=max_retries, request_timeout=request_timeout)
        except Exception:
            pass

    # Pass cache options into task via special attributes on adapter (simple plumbing)
    if cache_dir is not None:
        setattr(adapter, "_cache_dir", str(cache_dir))
    setattr(adapter, "_cache_mode", cache_mode)
    if cache_ttl is not None:
        setattr(adapter, "_cache_ttl", float(cache_ttl))

    result = task.evaluate(
        adapter,
        dataset,
        metrics,
        seed=seed,
        collect_records=records,
        concurrency=concurrency,
        max_retries=max_retries,
        request_timeout=request_timeout,
    )

    # enrich with spec metadata and optional run name
    result["spec_path"] = str(spec)
    try:
        result["spec_hash_sha256"] = hash_file(spec)
    except Exception:
        pass
    if run_name:
        result["run_name"] = run_name

    out_path = Path(out)
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
        if timestamped:
            import datetime as _dt

            ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            out_path = artifacts / f"{ts}.json"
        else:
            out_path = artifacts / out_path.name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print({"saved": str(out_path)})


runs_app = typer.Typer(help="Manage and aggregate runs")
app.add_typer(runs_app, name="runs")


@runs_app.command("collect")
def runs_collect(
    dir: Path = typer.Option(Path("runs"), "--dir", help="Directory containing run .json files"),
    out: Path = typer.Option(
        Path("runs/index.json"), "--out", help="Where to save the aggregated index"
    ),
):
    """Aggregate run JSON files into an index for the leaderboard."""
    dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in sorted(dir.glob("*.json")):
        # Skip the output file itself and any obvious aggregate files
        if p.resolve() == out.resolve() or p.name.lower().startswith("index"):
            continue
        try:
            data = json.loads(p.read_text())
            # Heuristic: only include single-run payloads with metrics and task
            if not isinstance(data, dict) or "metrics" not in data or "task" not in data:
                continue
            data["_file"] = p.name
            entries.append(data)
        except Exception:
            continue
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": entries}, indent=2))
    print({"saved": str(out), "count": len(entries)})


@app.command("lock")
def lock(
    from_run: Path = typer.Option(..., "--from", help="Path to a run JSON to lock"),
    out: Path = typer.Option(Path("openeval-lock.json"), "--out", help="Lockfile path"),
):
    """Create a reproducibility lockfile from a run JSON."""
    try:
        payload = json.loads(Path(from_run).read_text())
    except Exception as e:
        raise typer.Exit(code=2) from e

    lock = {
        "task": payload.get("task"),
        "adapter": payload.get("adapter"),
        "dataset": payload.get("dataset"),
        "size": payload.get("size"),
        "seed": payload.get("seed"),
        "dataset_path": payload.get("dataset_path"),
        "dataset_hash_sha256": payload.get("dataset_hash_sha256"),
        "spec_path": payload.get("spec_path"),
        "spec_hash_sha256": payload.get("spec_hash_sha256"),
        "manifest": payload.get("manifest", {}),
        "metrics_present": list((payload.get("metrics") or {}).keys()),
    }
    out.write_text(json.dumps(lock, indent=2))
    print({"saved": str(out)})


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev only)"),
):
    """Launch the dashboard server."""
    try:
        import uvicorn  # type: ignore
    except Exception as e:  # pragma: no cover
        print({"error": f"uvicorn not available: {e}"})
        raise typer.Exit(code=2)
    uvicorn.run("openeval.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
