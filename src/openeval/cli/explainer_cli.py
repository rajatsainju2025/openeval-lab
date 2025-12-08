"""CLI commands for code explainer functionality.

Provides 'openeval explain' command group with options for analysis and explanation.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openeval.explainers import (
    ExplainLevel,
)
from openeval.explainers.ast_analyzer import PythonASTAnalyzer
from openeval.explainers.complexity_metrics import PythonComplexityAnalyzer
from openeval.explainers.evaluation_metrics import (
    ExplanationCoverageMeasure,
    ExplanationQualityEvaluator,
)
from openeval.explainers.formatter import CodeFormatter, OutputFormat
from openeval.explainers.llm_explainer import LLMCodeExplainer
from openeval.explainers.semantic_analyzer import PythonSemanticAnalyzer

app = typer.Typer(help="Code explainer commands")
console = Console()


@app.command()
def explain_file(
    filepath: str = typer.Argument(..., help="Path to Python file to explain"),
    level: str = typer.Option(
        "detailed",
        "--level",
        "-l",
        help="Explanation level: summary, detailed, expert",
    ),
    format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text, markdown, ansi, html"
    ),
    model: str = typer.Option("gpt-4", "--model", "-m", help="LLM model to use for explanations"),
    analyze_only: bool = typer.Option(
        False,
        "--analyze",
        "-a",
        help="Only analyze code, don't explain",
    ),
    show_metrics: bool = typer.Option(True, "--metrics", "-M", help="Show complexity metrics"),
) -> None:
    """Explain a Python file.

    Analyzes the code and generates natural language explanations
    for functions, classes, and modules.
    """
    try:
        # Read file
        path = Path(filepath)
        if not path.exists():
            console.print(f"[red]Error: File not found: {filepath}[/red]")
            raise typer.Exit(1)

        if not path.suffix == ".py":
            console.print("[yellow]Warning: File does not have .py extension[/yellow]")

        code = path.read_text()

        # Parse explanation level
        try:
            explain_level = ExplainLevel(level)
        except ValueError:
            console.print("[red]Invalid level. Must be: summary, detailed, expert[/red]")
            raise typer.Exit(1)

        # Parse output format
        try:
            output_format = OutputFormat(format)
        except ValueError:
            console.print("[red]Invalid format. Must be: text, markdown, ansi, html[/red]")
            raise typer.Exit(1)

        # Analyze code
        console.print("[blue]Analyzing code...[/blue]")
        analyzer = PythonASTAnalyzer()
        analysis = analyzer.analyze(code)

        # Show analysis results
        console.print(
            Panel.fit(
                f"[bold green]Analysis Results[/bold green]\n"
                f"Functions: {analysis.metadata['total_elements']}\n"
                f"Imports: {len(analysis.imports)}",
                border_style="green",
            )
        )

        if analyze_only:
            # Just show structure
            _display_analysis(analysis, output_format)
            return

        # Get complexity metrics
        if show_metrics:
            console.print("[blue]Calculating complexity metrics...[/blue]")
            complexity_analyzer = PythonComplexityAnalyzer()
            metrics = complexity_analyzer.calculate(code)
            _display_metrics(metrics)

        # Generate explanations
        console.print(f"[blue]Generating explanations ({level})...[/blue]")
        explainer = LLMCodeExplainer(model=model)
        formatter = CodeFormatter()

        # Explain each extracted element
        for i, element in enumerate(analysis.elements, 1):
            console.print(f"\n[cyan]Element {i}/{len(analysis.elements)}[/cyan]")

            # Generate explanation
            result = explainer.explain(element, explain_level)

            # Format and display
            formatted = formatter.format_explanation_result(
                result, output_format, include_code=True
            )
            console.print(formatted)

            # Evaluate explanation
            evaluator = ExplanationQualityEvaluator()
            quality_score = evaluator.get_overall_score(result.explanation, element.source_code)
            quality_rating = evaluator.rate_quality(result.explanation, element.source_code)

            console.print(f"[yellow]Quality: {quality_rating} ({quality_score:.1%})[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    filepath: str = typer.Argument(..., help="Path to Python file"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed analysis"),
) -> None:
    """Analyze Python code structure and complexity.

    Shows functions, classes, dependencies, and metrics.
    """
    try:
        path = Path(filepath)
        if not path.exists():
            console.print(f"[red]Error: File not found: {filepath}[/red]")
            raise typer.Exit(1)

        code = path.read_text()

        # Run analyzers
        ast_analyzer = PythonASTAnalyzer()
        semantic_analyzer = PythonSemanticAnalyzer()
        complexity_analyzer = PythonComplexityAnalyzer()

        analysis = ast_analyzer.analyze(code)
        semantic_analysis = semantic_analyzer.analyze(code)
        metrics = complexity_analyzer.calculate(code)

        # Display results
        console.print(
            Panel.fit(
                f"[bold cyan]{path.name}[/bold cyan]",
                border_style="cyan",
            )
        )

        # Show code elements
        console.print("\n[bold]Code Elements:[/bold]")
        for elem in analysis.elements:
            console.print(f"  • {elem.type.value}: {elem.name}")

        # Show dependencies
        if analysis.dependencies:
            console.print("\n[bold]Dependencies:[/bold]")
            for dep in analysis.dependencies[:10]:
                console.print(f"  • {dep}")

        # Show metrics
        console.print("\n[bold]Metrics:[/bold]")
        _display_metrics(metrics)

        # Show semantic analysis if detailed
        if detailed and semantic_analysis.metadata:
            console.print("\n[bold]Variable Analysis:[/bold]")
            var_info = semantic_analysis.metadata.get("variable_analysis", {})
            console.print(f"  Total variables: {var_info.get('total_variables', 0)}")
            console.print(f"  Unused variables: {len(var_info.get('unused_variables', []))}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    explanation: str = typer.Argument(..., help="Explanation text"),
    code: str = typer.Option("", "--code", "-c", help="Code snippet being explained"),
) -> None:
    """Evaluate explanation quality.

    Scores explanation on clarity, completeness, relevance, etc.
    """
    try:
        evaluator = ExplanationQualityEvaluator()

        # Get scores
        scores = evaluator.evaluate(explanation, code)
        overall = evaluator.get_overall_score(explanation, code)
        rating = evaluator.rate_quality(explanation, code)

        # Display results
        console.print(
            Panel.fit(
                f"[bold green]Explanation Quality: {rating}[/bold green]\n"
                f"Overall Score: {overall:.1%}",
                border_style="green",
            )
        )

        # Show individual scores
        console.print("\n[bold]Detailed Scores:[/bold]")
        for metric, score in scores.items():
            bar_length = int(score * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            console.print(f"  {metric:15} {bar} {score:.1%}")

        # Show coverage
        coverage = ExplanationCoverageMeasure.get_coverage(explanation)
        console.print("\n[bold]Topic Coverage:[/bold]")
        for topic, covered in coverage.items():
            status = "[green]✓[/green]" if covered else "[red]✗[/red]"
            console.print(f"  {status} {topic}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _display_analysis(analysis, output_format: OutputFormat) -> None:
    """Display analysis results."""
    console.print("\n[bold]Code Elements:[/bold]")
    for elem in analysis.elements:
        console.print(f"  • {elem.type.value}: {elem.name}")

    if analysis.imports:
        console.print("\n[bold]Imports:[/bold]")
        for imp in analysis.imports[:5]:
            console.print(f"  • {imp}")


def _display_metrics(metrics) -> None:
    """Display complexity metrics in a table."""
    table = Table(title="Code Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    metrics_list = [
        ("Cyclomatic Complexity", f"{metrics.cyclomatic_complexity:.1f}"),
        ("Lines of Code", str(metrics.lines_of_code)),
        ("Comment Ratio", f"{metrics.comment_ratio:.1%}"),
        ("Max Nesting Depth", str(metrics.nesting_depth)),
        ("Functions", str(metrics.function_count)),
        ("Classes", str(metrics.class_count)),
        ("Avg Function Length", f"{metrics.average_function_length:.1f}"),
    ]

    for metric, value in metrics_list:
        table.add_row(metric, value)

    console.print(table)
