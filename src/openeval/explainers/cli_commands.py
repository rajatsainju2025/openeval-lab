"""Enhanced CLI commands for explainer configuration and monitoring.

Provides production-ready CLI for managing explanation pipelines.
"""

import json
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    pass


class ExplainerCLI:
    """Enhanced CLI commands for explainers."""

    def __init__(self) -> None:
        """Initialize CLI."""
        self.console = Console()
        self.app = None

    def setup_commands(self) -> None:
        """Setup all CLI commands."""
        # This would integrate with the existing CLI module
        pass

    def list_explainers(self) -> None:
        """List available explainer types."""
        from .factory import get_explainer_factory

        factory = get_explainer_factory()
        types = factory.list_types()

        table = Table(title="Available Explainer Types")
        table.add_column("Type", style="cyan")
        table.add_column("Class", style="green")

        for explainer_type in types:
            info = factory.get_type_info(explainer_type)
            table.add_row(explainer_type, info["class"])

        self.console.print(table)

    def list_prompt_templates(self) -> None:
        """List available prompt templates."""
        from .prompt_templates import PromptTemplateManager

        manager = PromptTemplateManager()
        templates = manager.list_templates()

        table = Table(title="Available Prompt Templates")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Default", style="yellow")

        for name, info in templates.items():
            is_default = "✓" if info["is_default"] else ""
            table.add_row(name, info["description"], is_default)

        self.console.print(table)

    def list_quality_metrics(self) -> None:
        """List available quality metrics."""
        from .metrics_plugin import get_metrics_registry

        registry = get_metrics_registry()
        metrics = registry.list_metrics()

        table = Table(title="Available Quality Metrics")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")

        for name, info in metrics.items():
            table.add_row(name, info["description"])

        self.console.print(table)

    def list_middleware(self) -> None:
        """List available middleware."""
        from .middleware import (
            LoggingMiddleware,
            ValidationMiddleware,
            EnrichmentMiddleware,
            CachingMiddleware,
        )

        middleware_classes = [
            LoggingMiddleware,
            ValidationMiddleware,
            EnrichmentMiddleware,
            CachingMiddleware,
        ]

        table = Table(title="Available Middleware")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")

        for mw_class in middleware_classes:
            instance = mw_class() if mw_class != CachingMiddleware else mw_class()
            table.add_row(instance.get_name(), mw_class.__name__)

        self.console.print(table)

    def show_config(self, config_file: Optional[str] = None) -> None:
        """Show current or loaded configuration."""
        if config_file:
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    self.console.print_json(data=config)
            except FileNotFoundError:
                self.console.print(f"[red]Config file not found: {config_file}[/red]")
        else:
            self.console.print("[yellow]No config file specified. Use --config <file>[/yellow]")

    def validate_config(self, config_file: str) -> None:
        """Validate configuration file."""
        from .factory import get_explainer_factory

        try:
            with open(config_file) as f:
                config = json.load(f)

            factory = get_explainer_factory()
            # Try to create explainer from config
            explainer = factory.create_from_dict(config)

            self.console.print(
                f"[green]✓ Configuration valid[/green]\n"
                f"Type: {config.get('type')}\n"
                f"Explainer class: {explainer.__class__.__name__}"
            )
        except FileNotFoundError:
            self.console.print(f"[red]Config file not found: {config_file}[/red]")
        except Exception as e:
            self.console.print(f"[red]Configuration invalid: {e}[/red]")

    def show_cache_stats(self) -> None:
        """Show cache statistics."""
        self.console.print("[yellow]Cache stats would be shown here with active explainer[/yellow]")

    def show_system_info(self) -> None:
        """Show system and environment info."""
        import sys

        table = Table(title="System Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Python Version", sys.version.split()[0])
        table.add_row("Platform", sys.platform)

        # Try to get package version
        try:
            import openeval

            table.add_row("OpenEval Version", getattr(openeval, "__version__", "unknown"))
        except ImportError:
            table.add_row("OpenEval Version", "not installed")

        self.console.print(table)


# Global CLI instance
_global_cli = ExplainerCLI()


def get_explainer_cli() -> ExplainerCLI:
    """Get the global CLI instance.

    Returns:
        ExplainerCLI instance.
    """
    return _global_cli
