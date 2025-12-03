"""
OpenEval Lab CLI - Command Line Interface for LLM Evaluation Framework

This module provides the main CLI interface for OpenEval Lab, offering commands
for evaluation, validation, monitoring, and management of LLM evaluation tasks.

Features:
- Declarative evaluation specifications (YAML/JSON)
- Plugin architecture for tasks, datasets, adapters, and metrics
- Concurrent execution with performance monitoring
- Rich logging with JSON support
- Comprehensive error handling and user feedback

Usage:
    openeval run spec <spec> --concurrency 4
    openeval validate <spec>
    openeval web --reload

Author: OpenEval Lab Team
Version: 0.1.0
"""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console

from .commands import base_app, eval_app, run_app
from .commands.base import registry_list, registry_info, tutorial, docs, version, doctor
from .commands.evaluation import validate_spec, validate_results, compare, write_out
from ..results_schema import RESULTS_JSON_SCHEMA
from ..spec import EvalSpec
from . import cli_help
from .. import version_utils

# Create the main app
app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="OpenEval Lab - Enterprise-grade LLM Evaluation Framework\n\n"
    "Evaluate language models, multimodal models, and AI agents with "
    "reproducible, extensible, and production-ready workflows.\n\n"
    "Quick Start:\n"
    "  openeval run examples/qa_spec.json\n"
    "  openeval tutorial\n"
    "  openeval doctor\n\n"
    "For detailed help on any command, use: openeval <command> --help",
)

console = Console()

# Add command groups
app.add_typer(base_app, name="base")
app.add_typer(eval_app, name="eval")
app.add_typer(run_app, name="run")

# For backward compatibility, also add commands directly to main app
# Import and add key commands from submodules

app.command()(registry_list)
app.command()(registry_info)
app.command()(tutorial)
app.command()(docs)
app.command()(version)
app.command()(doctor)

app.command("validate")(validate_spec)
app.command("validate-results")(validate_results)
app.command("compare")(compare)
app.command("write_out")(write_out)


# Enhanced help commands
@app.command()
def examples(
    category: Optional[str] = typer.Argument(
        None, help="Category: quickstart, evaluation, validation, analysis, advanced"
    )
):
    """Show practical CLI command examples."""
    cli_help.show_command_examples(category)


@app.command()
def spec_guide():
    """Show comprehensive guide for creating specification files."""
    cli_help.show_spec_guide()


@app.command()
def workflow():
    """Show typical evaluation workflow steps."""
    cli_help.show_workflow_guide()


@app.command()
def troubleshoot():
    """Show common troubleshooting tips and solutions."""
    cli_help.show_troubleshooting()


@app.command("registry-help")
def registry_help():
    """Show help for using the component registry."""
    cli_help.show_registry_help()


@app.command("perf-tips")
def perf_tips():
    """Show performance optimization tips."""
    cli_help.show_performance_tips()


@app.command("startup-check")
def startup_check():
    """Check CLI startup performance and report timing breakdown.

    Useful for diagnosing slow startup times and identifying
    which imports or initializations are taking longest.
    """
    import time

    console.print("[bold]OpenEval CLI Startup Performance Check[/bold]\n")

    timings = []

    # Time core imports
    start = time.perf_counter()
    from ..core import Task, Dataset, Adapter, Metric  # noqa: F401

    timings.append(("Core abstractions import", time.perf_counter() - start))

    # Time spec import
    start = time.perf_counter()
    from ..spec import EvalSpec  # noqa: F401

    timings.append(("EvalSpec import", time.perf_counter() - start))

    # Time validation import
    start = time.perf_counter()
    from ..validation import SpecValidator  # noqa: F401

    timings.append(("Validation import", time.perf_counter() - start))

    # Time cache import
    start = time.perf_counter()
    from ..cache import PredictionCache  # noqa: F401

    timings.append(("Cache import", time.perf_counter() - start))

    # Time optional heavy imports
    start = time.perf_counter()
    try:
        import numpy  # noqa: F401

        timings.append(("NumPy import", time.perf_counter() - start))
    except ImportError:
        timings.append(("NumPy import (not available)", 0.0))

    start = time.perf_counter()
    try:
        import pandas  # noqa: F401

        timings.append(("Pandas import", time.perf_counter() - start))
    except ImportError:
        timings.append(("Pandas import (not available)", 0.0))

    # Display results
    total = sum(t for _, t in timings)

    console.print("[cyan]Component Timing Breakdown:[/cyan]")
    for name, duration in sorted(timings, key=lambda x: -x[1]):
        bar_len = int(duration / total * 30) if total > 0 else 0
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pct = (duration / total * 100) if total > 0 else 0
        console.print(f"  {name:35} {bar} {duration*1000:6.1f}ms ({pct:4.1f}%)")

    console.print(f"\n[bold]Total measured time:[/bold] {total*1000:.1f}ms")

    # Performance rating
    if total < 0.2:
        rating = "[green]Excellent[/green] ✓"
    elif total < 0.5:
        rating = "[yellow]Good[/yellow]"
    elif total < 1.0:
        rating = "[orange1]Fair[/orange1] - Consider lazy loading"
    else:
        rating = "[red]Slow[/red] - Needs optimization"

    console.print(f"[bold]Performance rating:[/bold] {rating}")


@app.command()
def advanced():
    """Show advanced features and use cases."""
    cli_help.show_advanced_features()


# Version management commands
@app.command("version-info")
def version_info():
    """Show detailed version and git information."""
    version_utils.show_version_info()


@app.command()
def release(
    part: str = typer.Option("patch", help="Version part to bump: major, minor, or patch"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would happen without making changes"
    ),
):
    """Prepare a new release with version bump and changelog generation."""
    version_utils.create_release(part, dry_run)


# Legacy commands that were in the original CLI
# These will be moved to appropriate modules over time


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


@app.command()
def schema(
    out: Optional[typer.FileText] = typer.Option(None, "--out", help="Write JSON schema to file")
):
    """Print the JSON schema for experiment specs."""
    sch = EvalSpec.model_json_schema()
    if out:
        # Ensure valid JSON is written to file
        json.dump(sch, out, indent=2)
    else:
        # Pretty-print JSON to console
        try:
            console.print_json(data=sch)
        except Exception:
            console.print(sch)


@app.command("results-schema")
def results_schema(
    out: Optional[typer.FileText] = typer.Option(
        None, "--out", help="Write results JSON schema to file"
    )
):
    """Print the JSON schema for OpenEval results payloads."""
    if out:
        json.dump(RESULTS_JSON_SCHEMA, out, indent=2)
    else:
        try:
            console.print_json(data=RESULTS_JSON_SCHEMA)
        except Exception:
            console.print(RESULTS_JSON_SCHEMA)


@app.command()
def init(
    out: typer.FileText = typer.Argument(..., help="Path to write a starter spec (json or yaml)"),
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
    if fmt.lower() == "yaml" or str(out).endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except Exception:  # pragma: no cover
            raise typer.Exit(code=2)
        import yaml

        yaml.safe_dump(ex, out, sort_keys=False)
    else:
        json.dump(ex, out, indent=2)
    console.print(f"Created starter spec: {out}")


@app.command()
def lock(
    from_run: typer.FileText = typer.Option(..., "--from", help="Path to a run JSON to lock"),
    out: typer.FileText = typer.Option(..., "--out", help="Lockfile path"),
):
    """Create a reproducibility lockfile from a run JSON."""
    try:
        data = json.load(from_run)
    except Exception as e:
        raise typer.Exit(code=2) from e

    lock = {
        "task": data.get("task"),
        "adapter": data.get("adapter"),
        "dataset": data.get("dataset"),
        "size": data.get("size"),
        "seed": data.get("seed"),
        "dataset_path": data.get("dataset_path"),
        "dataset_hash_sha256": data.get("dataset_hash_sha256"),
        "spec_path": data.get("spec_path"),
        "spec_hash_sha256": data.get("spec_hash_sha256"),
        "manifest": data.get("manifest", {}),
        "metrics_present": list((data.get("metrics") or {}).keys()),
    }
    json.dump(lock, out, indent=2)
    console.print(f"Created lockfile: {out}")
