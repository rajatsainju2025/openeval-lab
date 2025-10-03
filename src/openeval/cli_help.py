"""Enhanced CLI help and documentation utilities."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def show_command_examples():
    """Show practical examples for common CLI commands."""

    examples = [
        {
            "title": "Basic Evaluation",
            "description": "Run a simple evaluation with echo adapter",
            "command": "openeval run examples/qa_spec.json --records --artifacts runs",
            "notes": "Uses offline echo adapter for testing",
        },
        {
            "title": "OpenAI Evaluation",
            "description": "Run evaluation with OpenAI GPT-4",
            "command": "openeval run my_spec.json --concurrency 5 --max-retries 2",
            "notes": "Requires OPENAI_API_KEY environment variable",
        },
        {
            "title": "Cached Evaluation",
            "description": "Use prediction caching to avoid re-runs",
            "command": "openeval run spec.json --cache-dir ./cache --cache rw",
            "notes": "Cache mode: off|read|write|rw",
        },
        {
            "title": "Bootstrap Comparison",
            "description": "Compare two model runs statistically",
            "command": "openeval compare runs/run1.json runs/run2.json --bootstrap 1000",
            "notes": "Outputs confidence intervals and p-values",
        },
        {
            "title": "Robustness Testing",
            "description": "Test model robustness with input noise",
            "command": "openeval run spec.json --robustness-noise 0.05 --records",
            "notes": "Applies character-level noise to inputs",
        },
        {
            "title": "Calibration Analysis",
            "description": "Analyze model confidence calibration",
            "command": "openeval run spec.json --calibration --records",
            "notes": "Requires adapter with logprobs support",
        },
        {
            "title": "Dataset Validation",
            "description": "Validate dataset quality before evaluation",
            "command": "openeval validate-dataset data.jsonl --output report.json",
            "notes": "Checks format, duplicates, encoding issues",
        },
        {
            "title": "Interactive Mode",
            "description": "Step through examples manually",
            "command": "openeval run spec.json --interactive",
            "notes": "Preview prompts and control execution",
        },
    ]

    console.print("\n[bold cyan]OpenEval CLI Examples[/bold cyan]\n")

    for example in examples:
        panel_content = f"""[bold]{example['description']}[/bold]

[yellow]Command:[/yellow]
[green]{example['command']}[/green]

[dim]{example['notes']}[/dim]"""

        console.print(
            Panel(panel_content, title=example["title"], border_style="blue", padding=(1, 2))
        )
        console.print()


def show_spec_guide():
    """Show guide for creating specification files."""

    spec_md = """
# OpenEval Specification Guide

## Basic Structure

```yaml
task: "openeval.tasks.qa.QuestionAnswering"
dataset: "openeval.datasets.jsonl.JSONLinesDataset"
adapter: "openeval.adapters.openai.OpenAIAdapter"
metrics:
  - name: "openeval.metrics.accuracy.ExactMatch"
dataset_kwargs:
  path: "data/qa_dataset.jsonl"
adapter_kwargs:
  model: "gpt-4"
  temperature: 0.0
output: "results.json"
```

## Field Descriptions

- **task**: Python path to task class (handles prompting)
- **dataset**: Python path to dataset class (loads examples)
- **adapter**: Python path to model adapter class (API wrapper)
- **metrics**: List of evaluation metrics to compute
- **dataset_kwargs**: Arguments passed to dataset constructor
- **adapter_kwargs**: Arguments passed to adapter constructor
- **output**: Where to save results

## Common Tasks

- `openeval.tasks.qa.QuestionAnswering` - Q&A with context
- `openeval.tasks.summarization.Summarization` - Text summarization
- `openeval.tasks.classification.Classification` - Text classification

## Common Datasets

- `openeval.datasets.jsonl.JSONLinesDataset` - JSONL file format
- `openeval.datasets.csv.CSVDataset` - CSV file format
- `openeval.datasets.hf.HuggingFaceDataset` - HF datasets

## Common Adapters

- `openeval.adapters.openai.OpenAIAdapter` - OpenAI models
- `openeval.adapters.echo.EchoAdapter` - Testing/debugging
- `openeval.adapters.hf.HuggingFaceAdapter` - Local HF models

## Common Metrics

- `openeval.metrics.accuracy.ExactMatch` - Exact string match
- `openeval.metrics.rouge.RougeL` - ROUGE-L score
- `openeval.metrics.bleu.SacreBLEU` - BLEU score
"""

    console.print(Markdown(spec_md))


def show_workflow_guide():
    """Show typical evaluation workflow."""

    workflow_steps = [
        (
            "1. Prepare Data",
            "Create or validate your dataset",
            "openeval validate-dataset data.jsonl",
        ),
        (
            "2. Create Spec",
            "Write specification file",
            "# Edit spec.yaml with your task/dataset/adapter",
        ),
        ("3. Validate Spec", "Check spec is valid", "openeval validate spec.yaml"),
        (
            "4. Test Run",
            "Small test with echo adapter",
            "openeval run spec.yaml --adapter echo --records",
        ),
        ("5. Full Run", "Complete evaluation", "openeval run spec.yaml --records --artifacts runs"),
        (
            "6. Compare",
            "Compare with baseline",
            "openeval compare runs/baseline.json runs/new.json",
        ),
        ("7. Dashboard", "View results", "openeval web --reload"),
    ]

    console.print("\n[bold cyan]Typical OpenEval Workflow[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Step", style="dim", width=12)
    table.add_column("Description", width=30)
    table.add_column("Command", style="green", width=50)

    for step, description, command in workflow_steps:
        table.add_row(step, description, command)

    console.print(table)
    console.print()


def show_troubleshooting():
    """Show common troubleshooting tips."""

    issues = [
        {
            "problem": "ImportError: No module named 'openai'",
            "solution": "Install OpenAI package: pip install openai",
            "category": "Dependencies",
        },
        {
            "problem": "Spec validation fails",
            "solution": "Check field names and Python import paths. Use openeval validate spec.yaml",
            "category": "Configuration",
        },
        {
            "problem": "Empty results or all errors",
            "solution": "Check API keys, network connectivity, and adapter configuration",
            "category": "Runtime",
        },
        {
            "problem": "Slow evaluation",
            "solution": "Use --concurrency flag, enable caching with --cache-dir",
            "category": "Performance",
        },
        {
            "problem": "Out of memory errors",
            "solution": "Reduce batch size, use streaming datasets, or run on smaller subset",
            "category": "Resources",
        },
        {
            "problem": "Inconsistent results",
            "solution": "Set --seed for reproducibility, check for non-deterministic adapters",
            "category": "Reproducibility",
        },
    ]

    console.print("\n[bold cyan]Troubleshooting Guide[/bold cyan]\n")

    categories = {}
    for issue in issues:
        cat = issue["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(issue)

    for category, category_issues in categories.items():
        console.print(f"\n[bold yellow]{category} Issues[/bold yellow]")
        for issue in category_issues:
            console.print(f"\n[red]Problem:[/red] {issue['problem']}")
            console.print(f"[green]Solution:[/green] {issue['solution']}")


def show_registry_help():
    """Show help for using the component registry."""

    registry_md = """
# OpenEval Component Registry

The registry provides short names for common components, making specs more readable.

## Usage

Instead of full Python paths:
```yaml
task: "openeval.tasks.qa.QuestionAnswering"
```

Use short names:
```yaml
task: "qa"
```

## Commands

- `openeval registry-list task` - List available tasks
- `openeval registry-list adapter` - List available adapters
- `openeval registry-list metric` - List available metrics
- `openeval registry-info metric rouge_l` - Get details about a component

## Benefits

- Shorter, more readable specs
- Auto-completion in CLI
- Validation with helpful suggestions
- Easier to share and document
"""

    console.print(Markdown(registry_md))


def show_performance_tips():
    """Show performance optimization tips."""

    tips = [
        ("Use Concurrency", "Set --concurrency N for parallel requests", "Faster evaluation"),
        ("Enable Caching", "Use --cache-dir with --cache rw", "Avoid re-computation"),
        ("Batch Processing", "Configure adapter batch_size", "Efficient API usage"),
        ("Async Adapters", "Use --async flag when available", "Better throughput"),
        ("Streaming", "Use streaming datasets for large data", "Lower memory usage"),
        ("Subset Testing", "Test on small subset first", "Quick iteration"),
        ("Results Schema", "Use validate-results for quality checks", "Catch issues early"),
        ("Monitoring", "Use --artifacts to track runs", "Progress tracking"),
    ]

    console.print("\n[bold cyan]Performance Optimization Tips[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Optimization", style="cyan", width=20)
    table.add_column("How To", width=40)
    table.add_column("Benefit", style="green", width=25)

    for tip, how, benefit in tips:
        table.add_row(tip, how, benefit)

    console.print(table)
    console.print()


def show_advanced_features():
    """Show advanced features and use cases."""

    features_md = """
# Advanced OpenEval Features

## Statistical Analysis

- Bootstrap confidence intervals with `--bootstrap` flag
- Paired significance testing with `openeval compare`
- Robustness analysis with `--robustness-noise`
- Calibration assessment with `--calibration`

## Caching System

```bash
# Write cache on first run
openeval run spec.json --cache-dir ./cache --cache write

# Read from cache on subsequent runs
openeval run spec.json --cache-dir ./cache --cache read

# Read-write mode (read existing, write new)
openeval run spec.json --cache-dir ./cache --cache rw
```

## Interactive Mode

Step through examples manually:
```bash
openeval run spec.json --interactive
```

## Experiment Tracking

- Automatic timestamping with `--artifacts`
- Run comparison with `openeval compare`
- Reproducibility with `openeval lock`

## Quality Assurance

- Dataset validation before runs
- Results schema validation
- Comprehensive error reporting
"""

    console.print(Markdown(features_md))
