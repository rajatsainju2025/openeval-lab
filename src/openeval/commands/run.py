"""
Run commands for OpenEval Lab CLI.

Provides the main evaluation execution functionality.
"""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console

from ..spec import EvalSpec

console = Console()


def run(
    spec_path: str = typer.Argument(..., help="Path to evaluation spec file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Override output path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
):
    """Run an evaluation from a specification file."""
    try:
        # Load and validate spec
        with open(spec_path) as f:
            spec_data = json.load(f)

        spec = EvalSpec.model_validate(spec_data)

        # Override output if specified
        if output:
            spec.output = output

        console.print(f"[blue]Running evaluation:[/blue] {spec_path}")
        console.print(f"[dim]Task: {spec.task}[/dim]")
        console.print(f"[dim]Dataset: {spec.dataset}[/dim]")
        console.print(f"[dim]Adapter: {spec.adapter}[/dim]")

        if verbose:
            console.print(f"[dim]Output: {spec.output}[/dim]")
            if spec.metrics:
                console.print(f"[dim]Metrics: {len(spec.metrics)}[/dim]")

        # Run evaluation
        console.print("\n[green]Starting evaluation...[/green]")

        try:
            # TODO: Implement actual evaluation execution
            # For now, create a placeholder result
            results = {
                "spec_path": spec_path,
                "task": spec.task,
                "dataset": spec.dataset,
                "adapter": spec.adapter,
                "size": 0,
                "metrics": {},
                "status": "placeholder - evaluation not yet implemented",
            }

            # Write results to output file
            if spec.output:
                with open(spec.output, "w") as f:
                    json.dump(results, f, indent=2)

                console.print(f"[yellow]⚠ Placeholder results written to: {spec.output}[/yellow]")
                console.print("[yellow]Actual evaluation execution not yet implemented[/yellow]")
            else:
                console.print("[yellow]No output file specified[/yellow]")

        except Exception as e:
            console.print(f"[red]✗ Evaluation failed: {e}[/red]")
            if debug:
                import traceback

                traceback.print_exc()
            raise typer.Exit(1)

    except FileNotFoundError:
        console.print(f"[red]✗ Spec file not found: {spec_path}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]✗ Invalid JSON in spec file: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        if debug:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)
