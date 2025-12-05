"""Progress tracking and live status displays for CLI using rich.

This module provides rich progress bars, live status displays, and real-time
metrics visualization for evaluation runs.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TaskID,
)
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationStats:
    """Statistics for ongoing evaluation run."""

    total_examples: int = 0
    completed_examples: int = 0
    successful: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.time)
    current_throughput: float = 0.0
    avg_latency: float = 0.0
    error_rate: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


class EvaluationProgressTracker:
    """Track and display evaluation progress with rich UI."""

    def __init__(
        self,
        total_examples: int,
        show_throughput: bool = True,
        show_errors: bool = True,
        show_metrics: bool = True,
        refresh_per_second: int = 10,
    ):
        """Initialize progress tracker.

        Args:
            total_examples: Total number of examples to evaluate
            show_throughput: Whether to show throughput stats
            show_errors: Whether to show error rates
            show_metrics: Whether to show live metrics
            refresh_per_second: Refresh rate for live display
        """
        self.total_examples = total_examples
        self.show_throughput = show_throughput
        self.show_errors = show_errors
        self.show_metrics = show_metrics
        self.refresh_per_second = refresh_per_second

        self.console = Console()
        self.stats = EvaluationStats(total_examples=total_examples)

        # Create progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=refresh_per_second,
        )

        self.task_id: Optional[TaskID] = None
        self.live: Optional[Live] = None

    def start(self, description: str = "Evaluating") -> None:
        """Start progress tracking.

        Args:
            description: Description to display
        """
        self.task_id = self.progress.add_task(description, total=self.total_examples)
        self.stats.start_time = time.time()

        # Create live display layout
        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=3),
            Layout(name="stats", size=10),
        )

        layout["progress"].update(self.progress)
        layout["stats"].update(self._create_stats_panel())

        self.live = Live(
            layout,
            console=self.console,
            refresh_per_second=self.refresh_per_second,
        )
        self.live.start()

    def update(
        self,
        advance: int = 1,
        success: bool = True,
        latency: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update progress with new completion.

        Args:
            advance: Number of examples completed
            success: Whether the evaluation succeeded
            latency: Latency in seconds for this example
            metrics: Updated metrics dictionary
        """
        if self.task_id is None:
            return

        # Update progress bar
        self.progress.update(self.task_id, advance=advance)

        # Update stats
        self.stats.completed_examples += advance
        if success:
            self.stats.successful += advance
        else:
            self.stats.failed += advance

        # Update latency
        if latency is not None:
            # Simple exponential moving average
            alpha = 0.1
            if self.stats.avg_latency == 0:
                self.stats.avg_latency = latency
            else:
                self.stats.avg_latency = alpha * latency + (1 - alpha) * self.stats.avg_latency

        # Update throughput
        elapsed = time.time() - self.stats.start_time
        if elapsed > 0:
            self.stats.current_throughput = self.stats.completed_examples / elapsed

        # Update error rate
        if self.stats.completed_examples > 0:
            self.stats.error_rate = self.stats.failed / self.stats.completed_examples

        # Update metrics
        if metrics:
            self.stats.metrics.update(metrics)

        # Refresh live display
        if self.live and self.live.is_started:
            layout = self.live.renderable
            if isinstance(layout, Layout):
                layout["stats"].update(self._create_stats_panel())

    def stop(self) -> None:
        """Stop progress tracking and display final summary."""
        if self.live and self.live.is_started:
            self.live.stop()

        # Display final summary
        self._display_summary()

    def _create_stats_panel(self) -> Panel:
        """Create stats panel for live display."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="white")

        # Add completion stats
        table.add_row(
            "Progress:",
            f"{self.stats.completed_examples}/{self.stats.total_examples} "
            f"({self.stats.completed_examples / self.stats.total_examples * 100:.1f}%)",
        )

        table.add_row(
            "Success:",
            f"[green]{self.stats.successful}[/green] / " f"[red]{self.stats.failed}[/red]",
        )

        # Add throughput stats
        if self.show_throughput:
            table.add_row(
                "Throughput:",
                f"{self.stats.current_throughput:.2f} examples/sec",
            )

            if self.stats.avg_latency > 0:
                table.add_row(
                    "Avg Latency:",
                    f"{self.stats.avg_latency:.3f}s",
                )

        # Add error rate
        if self.show_errors and self.stats.completed_examples > 0:
            error_pct = self.stats.error_rate * 100
            error_color = "red" if error_pct > 5 else "yellow" if error_pct > 1 else "green"
            table.add_row(
                "Error Rate:",
                f"[{error_color}]{error_pct:.2f}%[/{error_color}]",
            )

        # Add live metrics
        if self.show_metrics and self.stats.metrics:
            table.add_row("", "")  # Spacer
            table.add_row("[bold]Live Metrics:", "")
            for metric_name, metric_value in sorted(self.stats.metrics.items())[:5]:
                table.add_row(
                    f"  {metric_name}:",
                    f"{metric_value:.4f}",
                )

        return Panel(
            table,
            title="[bold]Evaluation Statistics[/bold]",
            border_style="blue",
        )

    def _display_summary(self) -> None:
        """Display final evaluation summary."""
        elapsed = time.time() - self.stats.start_time

        summary = Table(title="[bold green]Evaluation Complete[/bold green]")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="white")

        summary.add_row("Total Examples", str(self.stats.total_examples))
        summary.add_row("Completed", str(self.stats.completed_examples))
        summary.add_row(
            "Success Rate",
            (
                f"{self.stats.successful / self.stats.completed_examples * 100:.2f}%"
                if self.stats.completed_examples > 0
                else "N/A"
            ),
        )
        summary.add_row(
            "Error Rate",
            f"{self.stats.error_rate * 100:.2f}%",
        )
        summary.add_row("Total Time", f"{elapsed:.2f}s")
        summary.add_row(
            "Avg Throughput",
            f"{self.stats.current_throughput:.2f} examples/sec",
        )
        summary.add_row("Avg Latency", f"{self.stats.avg_latency:.3f}s")

        self.console.print()
        self.console.print(summary)

        # Display final metrics
        if self.stats.metrics:
            metrics_table = Table(title="[bold]Final Metrics[/bold]")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="white")

            for metric_name, metric_value in sorted(self.stats.metrics.items()):
                metrics_table.add_row(metric_name, f"{metric_value:.4f}")

            self.console.print()
            self.console.print(metrics_table)


class BatchProgressTracker:
    """Simple progress tracker for batch operations."""

    def __init__(self, total_batches: int, description: str = "Processing"):
        """Initialize batch progress tracker.

        Args:
            total_batches: Total number of batches
            description: Description to display
        """
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self.task_id = self.progress.add_task(description, total=total_batches)

    def __enter__(self):
        """Context manager entry."""
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.progress.stop()

    def update(self, advance: int = 1):
        """Update progress."""
        self.progress.update(self.task_id, advance=advance)


def create_progress_tracker(
    total: int,
    description: str = "Processing",
    simple: bool = False,
) -> Any:
    """Create appropriate progress tracker.

    Args:
        total: Total number of items
        description: Description to display
        simple: Whether to use simple progress bar

    Returns:
        Progress tracker instance
    """
    if simple:
        return BatchProgressTracker(total, description)
    else:
        tracker = EvaluationProgressTracker(total)
        tracker.start(description)
        return tracker


def display_results_table(results: List[Dict[str, Any]], title: str = "Results") -> None:
    """Display results in a formatted table.

    Args:
        results: List of result dictionaries
        title: Table title
    """
    if not results:
        Console().print("[yellow]No results to display[/yellow]")
        return

    # Get all keys from first result
    keys = list(results[0].keys())

    table = Table(title=title)
    for key in keys:
        table.add_column(key, style="cyan")

    for result in results:
        row = [str(result.get(key, "")) for key in keys]
        table.add_row(*row)

    Console().print(table)


def display_comparison_table(
    baseline: Dict[str, float],
    comparison: Dict[str, float],
    title: str = "Metric Comparison",
) -> None:
    """Display side-by-side metric comparison.

    Args:
        baseline: Baseline metrics
        comparison: Comparison metrics
        title: Table title
    """
    console = Console()
    table = Table(title=title)

    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", style="white")
    table.add_column("Comparison", style="white")
    table.add_column("Change", style="white")

    all_metrics = set(baseline.keys()) | set(comparison.keys())

    for metric in sorted(all_metrics):
        base_val = baseline.get(metric, 0.0)
        comp_val = comparison.get(metric, 0.0)
        diff = comp_val - base_val
        pct_change = (diff / base_val * 100) if base_val != 0 else 0

        # Color code the change
        if diff > 0:
            change_str = f"[green]+{diff:.4f} (+{pct_change:.2f}%)[/green]"
        elif diff < 0:
            change_str = f"[red]{diff:.4f} ({pct_change:.2f}%)[/red]"
        else:
            change_str = "[yellow]No change[/yellow]"

        table.add_row(
            metric,
            f"{base_val:.4f}",
            f"{comp_val:.4f}",
            change_str,
        )

    console.print(table)


def show_error_summary(errors: List[Dict[str, Any]], max_display: int = 10) -> None:
    """Display error summary.

    Args:
        errors: List of error dictionaries
        max_display: Maximum number of errors to display
    """
    console = Console()

    if not errors:
        console.print("[green]✓ No errors![/green]")
        return

    console.print(f"[red]✗ Found {len(errors)} error(s)[/red]\n")

    table = Table(title="Error Summary")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Message", style="red")

    for i, error in enumerate(errors[:max_display]):
        table.add_row(
            error.get("id", str(i)),
            error.get("type", "Unknown"),
            error.get("message", "")[:80],  # Truncate long messages
        )

    console.print(table)

    if len(errors) > max_display:
        console.print(f"\n[yellow]... and {len(errors) - max_display} more errors[/yellow]")
