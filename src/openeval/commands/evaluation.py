"""
Evaluation management commands for OpenEval Lab CLI.

Provides commands for validating specs, comparing results,
and managing evaluation outputs.
"""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..results_schema import validate_results_payload
from ..spec import EvalSpec

console = Console()


def validate_spec(
    spec_path: typer.FileText = typer.Argument(..., help="Path to evaluation spec file")
):
    """Validate an evaluation specification file."""
    try:
        data = json.load(spec_path)
        spec = EvalSpec.model_validate(data)
        console.print("[green]✓ Specification is valid[/green]")
        console.print(f"Task: {spec.task}")
        console.print(f"Dataset: {spec.dataset}")
        console.print(f"Adapter: {spec.adapter}")
        console.print(f"Metrics: {len(spec.metrics) if spec.metrics else 0}")

        # Try to resolve the components to ensure they exist
        try:
            from ..spec import import_class

            import_class(spec.task)
            if isinstance(spec.dataset, str):
                import_class(spec.dataset)
            import_class(spec.adapter)
            for metric in spec.metrics:
                import_class(metric.name)
            console.print("[green]✓ All referenced components found[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Warning: Could not resolve components: {e}[/yellow]")
            console.print("[yellow]Hint: Check that all class paths are correct[/yellow]")

    except json.JSONDecodeError as e:
        console.print(f"[red]✗ Invalid JSON format: {e}[/red]")
        console.print(
            "[yellow]Hint: Check for syntax errors like missing commas or quotes[/yellow]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Specification invalid: {e}[/red]")
        console.print("[yellow]Hint: Use 'openeval schema' to see the expected format[/yellow]")
        raise typer.Exit(1)


def validate_results(
    results_path: typer.FileText = typer.Argument(..., help="Path to results JSON file")
):
    """Validate a results file against the schema."""
    try:
        data = json.load(results_path)

        # Use proper schema validation
        is_valid, errors = validate_results_payload(data)

        if is_valid:
            console.print("[green]✓ Results file is valid[/green]")
        else:
            console.print("[red]✗ Results file validation failed:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            raise typer.Exit(1)

        if "metrics" in data:
            console.print(f"Metrics found: {list(data['metrics'].keys())}")
        if "size" in data:
            console.print(f"Dataset size: {data['size']}")

    except Exception as e:
        console.print(f"[red]✗ Results validation failed: {e}[/red]")
        raise typer.Exit(1)


def compare(
    result1: typer.FileText = typer.Argument(..., help="First results file"),
    result2: typer.FileText = typer.Argument(..., help="Second results file"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m", help="Focus on specific metric"),
):
    """Compare two evaluation result files."""
    try:
        data1 = json.load(result1)
        data2 = json.load(result2)

        console.print("[bold blue]Results Comparison[/bold blue]\n")

        # Basic info comparison
        table = Table()
        table.add_column("Property")
        table.add_column("File 1")
        table.add_column("File 2")
        table.add_column("Difference")

        # Dataset sizes
        size1 = data1.get("size", "Unknown")
        size2 = data2.get("size", "Unknown")
        diff = "N/A" if size1 == "Unknown" or size2 == "Unknown" else str(int(size2) - int(size1))
        table.add_row("Dataset Size", str(size1), str(size2), diff)

        console.print(table)
        console.print()

        # Metrics comparison
        metrics1 = data1.get("metrics", {})
        metrics2 = data2.get("metrics", {})

        if metrics1 and metrics2:
            console.print("[bold green]Metrics Comparison:[/bold green]")

            all_metrics = set(metrics1.keys()) | set(metrics2.keys())
            if metric:
                all_metrics = {metric} if metric in all_metrics else set()

            for m in sorted(all_metrics):
                val1 = metrics1.get(m, {})
                val2 = metrics2.get(m, {})

                console.print(f"\n[bold]{m}:[/bold]")

                # Compare accuracy/main score
                if isinstance(val1, dict) and isinstance(val2, dict):
                    for key in sorted(set(val1.keys()) | set(val2.keys())):
                        v1 = val1.get(key, "N/A")
                        v2 = val2.get(key, "N/A")

                        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                            diff_val = v2 - v1
                            diff_str = f"{diff_val:+.4f}" if diff_val != 0 else "0.0000"
                            console.print(f"  {key}: {v1:.4f} → {v2:.4f} ({diff_str})")
                        else:
                            console.print(f"  {key}: {v1} → {v2}")

    except Exception as e:
        console.print(f"[red]Error comparing results: {e}[/red]")
        raise typer.Exit(1)


def write_out(
    results_path: typer.FileText = typer.Argument(..., help="Results JSON file"),
    output_path: str = typer.Argument(..., help="Output markdown file path"),
    template: Optional[str] = typer.Option(None, "--template", help="Template file path"),
):
    """Generate markdown report from evaluation results."""
    try:
        data = json.load(results_path)

        # Default template
        if template is None:
            template_content = """# Evaluation Results

## Overview
- **Task**: {task}
- **Dataset**: {dataset}
- **Size**: {size} examples
- **Date**: {date}

## Results
{metrics_table}

## Details
- **Adapter**: {adapter}
- **Seed**: {seed}
- **Runtime**: {runtime}s
"""
        else:
            with open(template) as f:
                template_content = f.read()

        # Extract data
        task = data.get("task", "Unknown")
        dataset = data.get("dataset", "Unknown")
        size = data.get("size", "Unknown")
        date = data.get("date", "Unknown")
        adapter = data.get("adapter", "Unknown")
        seed = data.get("seed", "Unknown")
        runtime = data.get("runtime", "Unknown")

        # Format metrics
        metrics_table = ""
        if "metrics" in data:
            metrics_table = "| Metric | Score |\n|--------|-------|\n"
            for metric, values in data["metrics"].items():
                if isinstance(values, dict):
                    for key, val in values.items():
                        if isinstance(val, (int, float)):
                            metrics_table += f"| {metric}.{key} | {val:.4f} |\n"
                        else:
                            metrics_table += f"| {metric}.{key} | {val} |\n"
                else:
                    metrics_table += f"| {metric} | {values} |\n"

        # Fill template
        output = template_content.format(
            task=task,
            dataset=dataset,
            size=size,
            date=date,
            adapter=adapter,
            seed=seed,
            runtime=runtime,
            metrics_table=metrics_table,
        )

        # Write output
        with open(output_path, "w") as f:
            f.write(output)

        console.print(f"[green]✓ Report written to {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")
        raise typer.Exit(1)
