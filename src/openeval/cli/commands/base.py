"""
Base system commands for OpenEval Lab CLI.

Provides core functionality like registry management, documentation,
version information, and system health checks.
"""

from __future__ import annotations


import typer
from rich.console import Console

from .. import registry
from ..utils import get_project_root

console = Console()


def registry_list():
    """List all available registry components."""
    console.print("[bold blue]OpenEval Registry Components[/bold blue]\n")

    # Tasks
    console.print("[bold green]Tasks:[/bold green]")
    for name, desc in registry.TASK_DESCRIPTIONS.items():
        console.print(f"  • {name}: {desc}")

    console.print()

    # Datasets
    console.print("[bold green]Datasets:[/bold green]")
    for name, desc in registry.DATASET_DESCRIPTIONS.items():
        console.print(f"  • {name}: {desc}")

    console.print()

    # Adapters
    console.print("[bold green]Adapters:[/bold green]")
    for name, desc in registry.ADAPTER_DESCRIPTIONS.items():
        console.print(f"  • {name}: {desc}")

    console.print()

    # Metrics
    console.print("[bold green]Metrics:[/bold green]")
    for name, desc in registry.METRIC_DESCRIPTIONS.items():
        console.print(f"  • {name}: {desc}")


def registry_info(
    kind: str = typer.Argument(..., help="Component type: task, dataset, adapter, metric"),
    name: str = typer.Argument(..., help="Component name"),
):
    """Get detailed information about a registry component."""
    try:
        component_class = registry.load_component(kind, name)
        if not component_class:
            console.print(f"[red]Component '{kind}/{name}' not found[/red]")
            raise typer.Exit(1)

        console.print(f"[bold blue]{kind.title()}: {name}[/bold blue]")
        console.print(f"Class: {component_class.__name__}")
        console.print(f"Module: {component_class.__module__}")

        if hasattr(component_class, "__doc__") and component_class.__doc__:
            console.print(f"Description: {component_class.__doc__.strip()}")

    except Exception as e:
        console.print(f"[red]Error loading component: {e}[/red]")
        raise typer.Exit(1)


def tutorial():
    """Show tutorial and getting started information."""
    console.print(
        """
[bold blue]OpenEval Lab Tutorial[/bold blue]

[bold green]1. Create an evaluation spec:[/bold green]
   openeval init my_spec.json

[bold green]2. Run an evaluation:[/bold green]
   openeval run my_spec.json

[bold green]3. View results:[/bold green]
   Results will be saved to the output file specified in your spec.

[bold green]4. List available components:[/bold green]
   openeval registry-list

[bold green]5. Get help on any command:[/bold green]
   openeval [command] --help

For more information, visit the documentation or run 'openeval docs'.
    """
    )


def docs():
    """Open documentation in the browser."""
    console.print("[blue]Documentation is available at:[/blue]")
    console.print("https://github.com/your-org/openeval-lab/docs")
    console.print("\n[dim]Local docs not yet implemented[/dim]")


def version():
    """Show version information."""
    try:
        from importlib.metadata import version as get_version

        ver = get_version("openeval")
        console.print(f"OpenEval Lab version: {ver}")
    except Exception:
        console.print("OpenEval Lab version: [development]")


def doctor():
    """Run system health checks."""
    console.print("[bold blue]OpenEval Lab System Health Check[/bold blue]\n")

    issues = []

    # Check Python version
    import sys

    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    else:
        console.print(
            f"[green]✓[/green] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

    # Check project structure
    try:
        root = get_project_root()
        console.print(f"[green]✓[/green] Project root: {root}")
    except Exception as e:
        issues.append(f"Project structure issue: {e}")

    # Check registry
    try:
        registry.lookup("task", "qa")
        console.print("[green]✓[/green] Registry operational")
    except Exception as e:
        issues.append(f"Registry issue: {e}")

    # Check dependencies
    deps = ["typer", "rich", "pydantic"]
    for dep in deps:
        try:
            __import__(dep)
            console.print(f"[green]✓[/green] {dep} available")
        except ImportError:
            issues.append(f"Missing dependency: {dep}")

    if issues:
        console.print(f"\n[red]Issues found ({len(issues)}):[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
        raise typer.Exit(1)
    else:
        console.print("\n[green]All systems operational![/green]")
