"""
Run commands for OpenEval Lab CLI.

Provides the main evaluation execution functionality.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..spec import load_spec
from ..async_evaluation_engine import AsyncEvaluationEngine, AsyncTaskConfig
from ..cache import PredictionCache
from . import run_app

console = Console()


def run(
    spec_path: str = typer.Argument(..., help="Path to evaluation spec file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Override output path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug mode"),
    cache: Optional[str] = typer.Option(None, "--cache", help="Cache directory for predictions"),
    max_concurrent: int = typer.Option(10, "--max-concurrent", help="Maximum concurrent requests"),
    request_timeout: Optional[float] = typer.Option(
        30.0, "--request-timeout", help="Request timeout in seconds"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
):
    """Run an evaluation from a specification file."""
    try:
        # Load components from spec
        task, dataset, adapter, metrics, default_output = load_spec(spec_path)

        # Override output if specified
        output_path = output or default_output

        console.print(f"[blue]Running evaluation:[/blue] {spec_path}")
        console.print(f"[dim]Task: {task.__class__.__name__}[/dim]")
        console.print(f"[dim]Dataset: {dataset.__class__.__name__}[/dim]")
        console.print(f"[dim]Adapter: {adapter.__class__.__name__}[/dim]")
        console.print(f"[dim]Metrics: {len(metrics)}[/dim]")

        if verbose:
            console.print(f"[dim]Output: {output_path}[/dim]")
            console.print(f"[dim]Dataset size: {len(dataset)}[/dim]")

        # Set up evaluation engine
        config = AsyncTaskConfig(
            max_concurrent_requests=max_concurrent,
            request_timeout=request_timeout,
            enable_progress_tracking=True,
        )
        engine = AsyncEvaluationEngine(config)

        # Set up caching if enabled
        if not no_cache:
            cache_dir = cache or ".cache"
            prediction_cache = PredictionCache(cache_dir=Path(cache_dir))
            engine.set_cache(prediction_cache)
            if verbose:
                console.print(f"[dim]Cache: {cache_dir}[/dim]")

        # Run evaluation
        console.print("\n[green]Starting evaluation...[/green]")

        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not verbose,
        ) as progress:
            task_progress = progress.add_task("Starting evaluation...", total=None)

            # Use streaming evaluation to reduce memory footprint
            # Process samples in batches instead of loading all into memory
            batch_size = min(max_concurrent, 50) if max_concurrent > 0 else 50

            prompts = []
            references = []
            predictions = []
            latencies = []
            sample_count = 0

            import asyncio

            async def process_batch(batch_prompts, batch_refs):
                """Process a batch of prompts asynchronously."""
                results = await engine.evaluate_batch(
                    adapter=adapter,
                    prompts=batch_prompts,
                )
                return results, batch_refs

            # Process dataset in batches using generator pattern
            progress.update(task_progress, description="Processing dataset in batches...")

            batch_prompts = []
            batch_refs = []

            for sample in dataset:  # Generator - no memory overhead
                sample_count += 1
                prompt = task.build_prompt_with_template(sample)
                batch_prompts.append(prompt)
                batch_refs.append(sample.reference)

                # Process batch when full
                if len(batch_prompts) >= batch_size:
                    progress.update(
                        task_progress, description=f"Processing samples 1-{sample_count}..."
                    )

                    batch_results, batch_references = asyncio.run(
                        process_batch(batch_prompts, batch_refs)
                    )

                    # Collect results
                    prompts.extend(batch_prompts)
                    references.extend(batch_references)
                    for result in batch_results:
                        predictions.append(result.prediction)
                        latencies.append(result.latency)

                    # Clear batch for next iteration
                    batch_prompts = []
                    batch_refs = []

            # Process final partial batch
            if batch_prompts:
                progress.update(task_progress, description="Processing final batch...")
                batch_results, batch_references = asyncio.run(
                    process_batch(batch_prompts, batch_refs)
                )

                prompts.extend(batch_prompts)
                references.extend(batch_references)
                for result in batch_results:
                    predictions.append(result.prediction)
                    latencies.append(result.latency)

            progress.update(
                task_progress, description=f"Computing metrics for {sample_count} samples..."
            )

            # Compute metrics
            metrics_results = {}
            for metric in metrics:
                try:
                    score = metric.compute(predictions, references)
                    metrics_results[metric.name] = score
                except Exception as e:
                    if debug:
                        console.print(f"[red]Error computing {metric.name}: {e}[/red]")
                    metrics_results[metric.name] = {"error": str(e)}

        # Prepare results
        end_time = time.time()
        runtime = end_time - start_time

        results = {
            "spec_path": spec_path,
            "task": task.__class__.__module__ + "." + task.__class__.__name__,
            "dataset": dataset.__class__.__module__ + "." + dataset.__class__.__name__,
            "adapter": adapter.__class__.__module__ + "." + adapter.__class__.__name__,
            "size": sample_count,
            "metrics": metrics_results,
            "runtime": runtime,
            "cache_stats": engine.get_stats() if hasattr(engine, "get_stats") else {},
            "status": "completed",
        }

        # Write results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"[green]✓ Evaluation completed in {runtime:.2f}s[/green]")
        console.print(f"[green]✓ Results written to: {output_path}[/green]")

        if verbose:
            console.print(f"[dim]Average latency: {sum(latencies)/len(latencies):.3f}s[/dim]")
            console.print(f"[dim]Cache hits: {results['cache_stats'].get('cache_hits', 0)}[/dim]")

    except FileNotFoundError:
        console.print(f"[red]✗ Spec file not found: {spec_path}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]✗ Invalid JSON in spec file: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Evaluation failed: {e}[/red]")
        if debug:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)


# Register command with run_app
run_app.command("spec")(run)
