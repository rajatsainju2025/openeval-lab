from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys
import subprocess
import time
from datetime import datetime

import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .spec import EvalSpec, load_spec
from .utils import hash_file
from .data_quality import DataQualityAssessor
from .experiment_tracking import experiment_tracker
from .optimization import performance_monitor

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def version():
    """Show OpenEval version information."""
    try:
        from importlib.metadata import version as _v
        version_info = _v("openeval-lab")
        
        console.print(f"OpenEval Lab version: {version_info}", style="blue")
        console.print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        # Show additional package versions
        packages = ['typer', 'pydantic', 'rich']
        for pkg in packages:
            try:
                pkg_version = _v(pkg)
                console.print(f"{pkg}: {pkg_version}")
            except Exception:
                console.print(f"{pkg}: not found", style="red")
                
    except Exception:
        console.print("OpenEval Lab version: unknown", style="yellow")
        console.print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


@app.command()
def quality(
    spec: Path = typer.Argument(..., help="Path to JSON/YAML spec"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output path for quality report"),
    sample_limit: Optional[int] = typer.Option(1000, "--sample-limit", help="Max samples to assess"),
):
    """Assess dataset quality and generate recommendations."""
    try:
        console.print("🔍 Loading dataset for quality assessment...", style="blue")
        task, dataset, adapter, metrics, out = load_spec(spec)
        
        # Initialize quality assessor
        assessor = DataQualityAssessor()
        
        # Perform assessment
        with console.status("[bold green]Analyzing dataset quality..."):
            quality_report = assessor.assess_dataset(dataset, sample_limit)
        
        # Display summary
        console.print(f"\n📊 Quality Assessment Results for: {quality_report.dataset_name}")
        console.print(f"Samples analyzed: {quality_report.sample_count:,}")
        console.print(f"Overall quality score: {quality_report.overall_score:.2f}/1.00")
        
        # Color-coded status
        if quality_report.overall_score >= 0.8:
            status_text = "🟢 GOOD - Dataset meets quality standards"
            status_style = "green"
        elif quality_report.overall_score >= 0.6:
            status_text = "🟡 FAIR - Dataset has some quality issues"
            status_style = "yellow"
        else:
            status_text = "🔴 POOR - Dataset requires significant improvement"
            status_style = "red"
        
        console.print(f"Status: {status_text}", style=status_style)
        
        # Show metrics table
        table = Table(title="Quality Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Description")
        
        for metric in quality_report.metrics:
            status_icon = "✅" if metric.passed else "❌" if metric.passed is False else "ℹ️"
            score_text = f"{metric.value:.3f}"
            if metric.threshold is not None:
                score_text += f" / {metric.threshold:.3f}"
            
            table.add_row(
                metric.name,
                score_text,
                status_icon,
                metric.description
            )
        
        console.print(table)
        
        # Show issues and recommendations
        if quality_report.issues:
            console.print("\n⚠️ Issues Found:", style="red bold")
            for issue in quality_report.issues:
                console.print(f"  • {issue}", style="red")
        
        if quality_report.recommendations:
            console.print("\n💡 Recommendations:", style="blue bold")
            for rec in quality_report.recommendations:
                console.print(f"  • {rec}", style="blue")
        
        # Save detailed report if requested
        if output:
            report_path = assessor.save_report(quality_report, output)
            console.print(f"\n📄 Detailed report saved to: {report_path}", style="green")
        
    except Exception as e:
        console.print(f"❌ Quality assessment failed: {e}", style="red")
        raise typer.Exit(code=1)


@app.command()
def experiment(
    action: str = typer.Argument(..., help="Action: create|list|compare|export"),
    name: Optional[str] = typer.Option(None, "--name", help="Experiment name"),
    description: Optional[str] = typer.Option(None, "--description", help="Experiment description"),
    tags: Optional[List[str]] = typer.Option(None, "--tag", help="Experiment tags"),
    experiment_ids: Optional[List[str]] = typer.Option(None, "--id", help="Experiment IDs"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file for export"),
    limit: Optional[int] = typer.Option(10, "--limit", help="Limit number of experiments listed"),
):
    """Manage experiment tracking and comparison."""
    try:
        if action == "create":
            if not name:
                console.print("❌ Experiment name is required for creation", style="red")
                raise typer.Exit(code=1)
            
            exp_id = experiment_tracker.create_experiment(
                name=name,
                description=description or "",
                tags=tags or []
            )
            console.print(f"✅ Created experiment: {exp_id}", style="green")
            
        elif action == "list":
            experiments = experiment_tracker.list_experiments(limit=limit)
            
            if not experiments:
                console.print("No experiments found", style="yellow")
                return
            
            table = Table(title="Experiments")
            table.add_column("ID", style="cyan")
            table.add_column("Name")
            table.add_column("Status")
            table.add_column("Primary Score", justify="right")
            table.add_column("Runtime", justify="right")
            table.add_column("Created")
            
            for exp in experiments:
                created_date = exp.created_at[:10]  # YYYY-MM-DD
                runtime_text = f"{exp.metrics.runtime_seconds:.1f}s" if exp.metrics.runtime_seconds > 0 else "N/A"
                score_text = f"{exp.metrics.primary_score:.3f}" if exp.metrics.primary_score > 0 else "N/A"
                
                table.add_row(
                    exp.experiment_id[:12] + "...",
                    exp.name,
                    exp.status,
                    score_text,
                    runtime_text,
                    created_date
                )
            
            console.print(table)
            
        elif action == "compare":
            if not experiment_ids or len(experiment_ids) < 2:
                console.print("❌ At least 2 experiment IDs required for comparison", style="red")
                raise typer.Exit(code=1)
            
            comparison = experiment_tracker.compare_experiments(experiment_ids)
            
            if "error" in comparison:
                console.print(f"❌ Comparison failed: {comparison['error']}", style="red")
                raise typer.Exit(code=1)
            
            console.print("📊 Experiment Comparison", style="blue bold")
            
            # Experiments table
            table = Table(title="Experiment Details")
            table.add_column("ID")
            table.add_column("Name")
            table.add_column("Primary Score", justify="right")
            table.add_column("Runtime", justify="right")
            table.add_column("Throughput", justify="right")
            
            for exp in comparison["experiments"]:
                table.add_row(
                    exp["id"][:12] + "...",
                    exp["name"],
                    f"{exp['primary_score']:.3f}",
                    f"{exp['runtime_seconds']:.1f}s",
                    f"{exp['throughput']:.2f}/s"
                )
            
            console.print(table)
            
            # Best performers
            console.print(f"\n🏆 Best Primary Score: {comparison['best_primary_score']:.3f}")
            console.print(f"⚡ Best Runtime: {comparison['best_runtime']:.1f}s")
            console.print(f"🚀 Best Throughput: {comparison['best_throughput']:.2f}/s")
            
        elif action == "export":
            if not output:
                console.print("❌ Output file required for export", style="red")
                raise typer.Exit(code=1)
            
            experiment_tracker.export_experiments(output, experiment_ids)
            console.print(f"✅ Experiments exported to: {output}", style="green")
            
        else:
            console.print(f"❌ Unknown action: {action}", style="red")
            console.print("Available actions: create, list, compare, export")
            raise typer.Exit(code=1)
            
    except Exception as e:
        console.print(f"❌ Experiment command failed: {e}", style="red")
        raise typer.Exit(code=1)


@app.command()
def monitor(
    duration: Optional[int] = typer.Option(60, "--duration", help="Monitoring duration in seconds"),
    interval: Optional[float] = typer.Option(1.0, "--interval", help="Sampling interval in seconds"),
    output: Optional[Path] = typer.Option(None, "--output", help="Save performance data to file"),
):
    """Monitor system performance and resources."""
    try:
        console.print("🔍 Starting performance monitoring...", style="blue")
        
        # Configure monitor
        monitor = performance_monitor
        if interval is not None:
            monitor.sample_interval = interval
        
        # Start monitoring
        monitor.start_monitoring()
        
        start_time = time.time()
        duration = duration or 60  # Default to 60 seconds
        
        try:
            while time.time() - start_time < duration:
                # Get current summary
                summary = monitor.get_performance_summary()
                
                if "error" not in summary:
                    current = summary["current"]
                    
                    # Clear screen and show live stats
                    console.clear()
                    console.print("📊 System Performance Monitor", style="blue bold")
                    console.print(f"Duration: {time.time() - start_time:.1f}s / {duration}s")
                    console.print(f"Memory: {current['memory_used_mb']:.1f} MB")
                    console.print(f"Threads: {current['thread_count']}")
                    
                    if "averages" in summary:
                        avg = summary["averages"]
                        console.print(f"Avg Memory: {avg['memory_used_mb']:.1f} MB")
                    
                    if "peaks" in summary:
                        peaks = summary["peaks"]
                        console.print(f"Peak Memory: {peaks['max_memory_used_mb']:.1f} MB")
                
                time.sleep(2)  # Update every 2 seconds
                
        finally:
            monitor.stop_monitoring()
        
        # Final summary
        final_summary = monitor.get_performance_summary()
        console.print("\n📋 Final Performance Summary", style="green bold")
        console.print(json.dumps(final_summary, indent=2))
        
        # Save data if requested
        if output:
            with open(output, 'w') as f:
                json.dump({
                    "duration": duration,
                    "interval": interval,
                    "summary": final_summary,
                    "metrics": [
                        {
                            "name": m.name,
                            "value": m.value,
                            "unit": m.unit,
                            "timestamp": m.timestamp,
                            "metadata": m.metadata
                        }
                        for m in monitor.metrics
                    ]
                }, f, indent=2)
            console.print(f"📄 Performance data saved to: {output}", style="green")
        
    except KeyboardInterrupt:
        console.print("\n⏹️ Monitoring stopped by user", style="yellow")
    except Exception as e:
        console.print(f"❌ Monitoring failed: {e}", style="red")
        raise typer.Exit(code=1)


@app.command()
def diagnose(
    spec: Optional[Path] = typer.Option(None, "--spec", help="Spec file to diagnose"),
    check_deps: bool = typer.Option(True, "--check-deps", help="Check dependencies"),
    check_config: bool = typer.Option(True, "--check-config", help="Check configuration"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
):
    """Diagnose system and configuration issues."""
    try:
        console.print("🔧 Running OpenEval diagnostics...", style="blue")
        
        issues = []
        warnings = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            issues.append(f"Python {python_version.major}.{python_version.minor} is not supported. Requires Python 3.8+")
        else:
            console.print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check dependencies
        if check_deps:
            console.print("🔍 Checking dependencies...")
            
            required_deps = ['typer', 'pydantic', 'rich']
            optional_deps = ['openai', 'anthropic', 'transformers', 'torch']
            
            for dep in required_deps:
                try:
                    __import__(dep)
                    console.print(f"✅ {dep} - installed")
                except ImportError:
                    issues.append(f"Required dependency '{dep}' not found")
            
            for dep in optional_deps:
                try:
                    __import__(dep)
                    console.print(f"✅ {dep} - installed")
                except ImportError:
                    if verbose:
                        warnings.append(f"Optional dependency '{dep}' not found")
        
        # Check spec file if provided
        if spec:
            console.print(f"🔍 Checking spec file: {spec}")
            
            if not spec.exists():
                issues.append(f"Spec file not found: {spec}")
            else:
                try:
                    task, dataset, adapter, metrics, out = load_spec(spec)
                    console.print("✅ Spec file loaded successfully")
                    
                    # Try to initialize components
                    try:
                        list(dataset)  # Try to load dataset
                        console.print("✅ Dataset can be loaded")
                    except Exception as e:
                        issues.append(f"Dataset loading failed: {e}")
                    
                    try:
                        adapter.generate("test prompt")  # Test adapter
                        console.print("✅ Adapter is functional")
                    except Exception as e:
                        warnings.append(f"Adapter test failed (may be expected): {e}")
                        
                except Exception as e:
                    issues.append(f"Spec file validation failed: {e}")
        
        # Check environment variables
        if check_config:
            console.print("🔍 Checking environment configuration...")
            
            import os
            env_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN']
            
            for var in env_vars:
                if os.getenv(var):
                    console.print(f"✅ {var} - configured")
                else:
                    if verbose:
                        warnings.append(f"Environment variable '{var}' not set")
        
        # Show summary
        console.print("\n📋 Diagnostic Summary", style="blue bold")
        
        if not issues and not warnings:
            console.print("🎉 All checks passed! System is ready.", style="green")
        else:
            if issues:
                console.print(f"\n❌ Issues found ({len(issues)}):", style="red")
                for issue in issues:
                    console.print(f"  • {issue}", style="red")
            
            if warnings:
                console.print(f"\n⚠️ Warnings ({len(warnings)}):", style="yellow")
                for warning in warnings:
                    console.print(f"  • {warning}", style="yellow")
            
            if issues:
                console.print("\n🔧 Please fix the issues above before running evaluations.", style="red")
                raise typer.Exit(code=1)
    
    except Exception as e:
        console.print(f"❌ Diagnostic failed: {e}", style="red")
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    spec: Path = typer.Argument(..., help="Path to JSON/YAML spec"),
    iterations: int = typer.Option(3, "--iterations", help="Number of benchmark iterations"),
    warmup: int = typer.Option(1, "--warmup", help="Number of warmup iterations"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file for benchmark results"),
):
    """Benchmark evaluation performance."""
    try:
        console.print("🏃 Starting benchmark...", style="blue")
        
        # Load spec
        task, dataset, adapter, metrics, out = load_spec(spec)
        
        # Warmup iterations
        if warmup > 0:
            console.print(f"🔥 Running {warmup} warmup iterations...")
            for i in range(warmup):
                with console.status(f"Warmup {i+1}/{warmup}"):
                    try:
                        # Try to run task evaluation with correct parameter order
                        task.evaluate(adapter, dataset, metrics)
                    except (AttributeError, TypeError):
                        # Fallback for different task interface
                        pass
        
        # Benchmark iterations
        console.print(f"⏱️ Running {iterations} benchmark iterations...")
        
        times = []
        memory_usage = []
        
        for i in range(iterations):
            start_time = time.time()
            start_memory = performance_monitor._get_memory_usage()
            
            with console.status(f"Iteration {i+1}/{iterations}"):
                try:
                    result = task.evaluate(adapter, dataset, metrics)
                except (AttributeError, TypeError):
                    # Fallback: just measure dataset loading
                    result = list(dataset)
            
            end_time = time.time()
            end_memory = performance_monitor._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            times.append(duration)
            memory_usage.append(memory_delta)
            
            console.print(f"Iteration {i+1}: {duration:.2f}s, Memory: +{memory_delta:.1f}MB")
        
        # Calculate statistics
        import statistics
        
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        
        avg_memory = statistics.mean(memory_usage)
        
        # Get dataset size for throughput calculation
        dataset_size = len(list(dataset))
        throughput = dataset_size / avg_time
        
        # Display results
        console.print("\n📊 Benchmark Results", style="green bold")
        console.print(f"Iterations: {iterations}")
        console.print(f"Dataset size: {dataset_size} samples")
        console.print(f"Average time: {avg_time:.3f}s (±{std_time:.3f}s)")
        console.print(f"Min time: {min_time:.3f}s")
        console.print(f"Max time: {max_time:.3f}s")
        console.print(f"Throughput: {throughput:.2f} samples/second")
        console.print(f"Average memory delta: {avg_memory:.1f}MB")
        
        # Save results if requested
        if output:
            benchmark_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "spec_file": str(spec),
                "iterations": iterations,
                "warmup": warmup,
                "dataset_size": dataset_size,
                "times": times,
                "memory_usage": memory_usage,
                "statistics": {
                    "avg_time": avg_time,
                    "std_time": std_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "throughput": throughput,
                    "avg_memory": avg_memory
                }
            }
            
            with open(output, 'w') as f:
                json.dump(benchmark_data, f, indent=2)
            
            console.print(f"📄 Benchmark results saved to: {output}", style="green")
    
    except Exception as e:
        console.print(f"❌ Benchmark failed: {e}", style="red")
        raise typer.Exit(code=1)


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
    interactive: bool = typer.Option(False, "--interactive", help="Step through examples interactively"),
    records: bool = typer.Option(False, "--records", help="Include per-example records in output"),
    artifacts: Optional[Path] = typer.Option(None, "--artifacts", help="Dir to write results"),
    timestamped: bool = typer.Option(
        True, help="When writing to --artifacts, save as runs/<timestamp>.json"
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Optional label for this run"),
    concurrency: int = typer.Option(1, help="Max concurrent requests (adapters may ignore)"),
    max_retries: int = typer.Option(0, help="Max retries per request on failure"),
    request_timeout: Optional[float] = typer.Option(None, help="Timeout per request (seconds)"),
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir", help="Prediction cache directory"
    ),
    cache_mode: str = typer.Option("off", "--cache", help="Cache mode: off|read|write|rw"),
    cache_ttl: Optional[float] = typer.Option(
        None, "--cache-ttl", help="Cache TTL seconds (optional)"
    ),
    cache_key_mode: str = typer.Option("strict", "--cache-key", help="Cache key mode: strict|compat"),
    traces: bool = typer.Option(False, "--traces", help="Include agent step traces in records when supported"),
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
            _set_opts(
                concurrency=concurrency, max_retries=max_retries, request_timeout=request_timeout
            )
        except Exception:
            pass

    # Pass cache options into task via special attributes on adapter (simple plumbing)
    if cache_dir is not None:
        setattr(adapter, "_cache_dir", str(cache_dir))
    setattr(adapter, "_cache_mode", cache_mode)
    if cache_ttl is not None:
        setattr(adapter, "_cache_ttl", float(cache_ttl))
    setattr(adapter, "_cache_key_mode", cache_key_mode)

    if not interactive:
        # Hint tasks that can emit traces
        try:
            setattr(task, "_collect_traces", bool(traces))
        except Exception:
            pass
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
    else:
        # Interactive loop: preview prompts and control flow
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
        import platform, sys, time

        examples = list(iter(dataset))
        predictions = []
        references = []
        per_latency = []
        recs = []
        t0 = time.perf_counter()
        for idx, ex in enumerate(examples):
            prompt = task.build_prompt_with_template(ex)
            console.print(f"\n[bold]Example {idx+1}/{len(examples)}[/bold] id={ex.id}", style="cyan")
            console.print(f"Input: {str(ex.input)[:200]}" )
            console.print("Show full prompt? [y/N], skip [s], quit [q]", style="muted")
            ans = input("Action: ").strip().lower()
            if ans == "q":
                break
            if ans == "y":
                console.print("\n--- Prompt ---\n" + prompt + "\n---------------")
            if ans == "s":
                continue
            s = time.perf_counter()
            out = adapter.generate(prompt)
            e = time.perf_counter()
            pred = task.postprocess(out)
            predictions.append(pred)
            references.append(ex.reference)
            per_latency.append(e - s)
            if records:
                recs.append({
                    "id": ex.id,
                    "input": ex.input,
                    "reference": ex.reference,
                    "prompt": prompt,
                    "prediction": pred,
                    "latency_ms": (e - s) * 1000.0,
                })
        total_duration = time.perf_counter() - t0

        # Compute metrics
        results = {}
        for m in metrics:
            try:
                results[m.name] = m.compute(predictions, references)
            except Exception as err:
                results[m.name] = {"error": str(err)}

        # Manifest basics
        def _maybe_ver(pkg: str):
            try:
                return _pkg_version(pkg)
            except Exception:
                return None

        latencies = [x for x in per_latency if x > 0]
        result = {
            "task": getattr(task, "name", task.__class__.__name__),
            "dataset": getattr(dataset, "name", dataset.__class__.__name__),
            "size": len(predictions),
            "metrics": results,
            "adapter": getattr(adapter, "name", adapter.__class__.__name__),
            "seed": seed,
            "timing": {
                "avg_latency_ms": (sum(latencies) / len(latencies) * 1000.0) if latencies else 0.0,
                "total_seconds": total_duration,
                "throughput_eps": (len(predictions) / total_duration) if total_duration > 0 else 0.0,
            },
            "manifest": {
                "openeval_version": _maybe_ver("openeval-lab"),
                "python": {"version": sys.version.split()[0], "executable": sys.executable},
                "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
            },
        }
        if records:
            result["records"] = recs

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


@app.command()
def validate(spec: Path = typer.Argument(..., help="Path to JSON/YAML spec to validate")):
    """Validate a spec file (schema + importability)."""
    try:
        # Attempt full load to validate dotted imports and kwargs
        load_spec(spec)
        print({"valid": True, "spec": str(spec)})
    except SystemExit as e:
        print({"valid": False, "spec": str(spec), "error": str(e)})
        raise typer.Exit(code=1)


@app.command("write_out")
def write_out(
    spec: Path = typer.Argument(..., help="Path to JSON/YAML spec"),
    out: Optional[Path] = typer.Option(None, "--out", help="Write prompts to this file (JSONL)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max examples to render"),
    preview: int = typer.Option(
        5, "--preview", help="How many prompts to print when not writing to a file"
    ),
):
    """Render task prompts for the dataset in a spec (for debugging)."""
    try:
        task, dataset, adapter, metrics, _ = load_spec(spec)
    except SystemExit as e:
        raise typer.Exit(code=2) from e

    # Iterate and build prompts
    rows = []
    count = 0
    for ex in dataset:
        prompt = task.build_prompt_with_template(ex)
        rows.append(
            {
                "id": ex.id,
                "input": ex.input,
                "reference": ex.reference,
                "prompt": prompt,
            }
        )
        count += 1
        if limit is not None and count >= limit:
            break

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print({"saved": str(out), "count": len(rows)})
    else:
        k = min(preview, len(rows))
        print({"preview_count": k, "total": len(rows)})
        for r in rows[:k]:
            print(r)


@app.command()
def library(
    action: str = typer.Argument(..., help="Action: list|info|export|categories|sync|get"),
    task_id: Optional[str] = typer.Argument(None, help="Task ID for info/export actions"),
    category: Optional[str] = typer.Option(None, "--category", help="Filter by category"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file for export"),
):
    """Interact with the curated task library."""
    from .library import get_task_library

    lib = get_task_library()

    if action == "list":
        tasks = lib.list_tasks(category=category)
        if not tasks:
            print("No tasks found")
            return

        print(f"Found {len(tasks)} tasks:")
        for task in tasks:
            print(f"  {task['id']} ({task['category']}): {task['description']}")

    elif action == "info":
        if not task_id:
            print("Task ID required for info action")
            raise typer.Exit(1)

        task = lib.get_task(task_id)
        if not task:
            print(f"Task {task_id} not found")
            raise typer.Exit(1)

        print(json.dumps(task, indent=2))

    elif action == "export":
        if not task_id:
            print("Task ID required for export action")
            raise typer.Exit(1)

        if not output:
            output = Path(f"{task_id}_spec.json")

        try:
            lib.export_task(task_id, str(output))
            print(f"Exported {task_id} to {output}")
        except ValueError as e:
            print(str(e))
            raise typer.Exit(1)

    elif action == "categories":
        categories = lib.list_categories()
        print("Available categories:")
        for cat in categories:
            tasks = lib.get_category_tasks(cat)
            print(f"  {cat} ({len(tasks)} tasks)")

    elif action == "sync":
        # Placeholder: in future, fetch from remote registry; for now, just report success
        print("Synchronized with registry (local placeholder)")
    elif action == "get":
        if not task_id:
            print("Task ID required for get action")
            raise typer.Exit(1)
        task = lib.get_task(task_id)
        if not task:
            print(f"Task {task_id} not found")
            raise typer.Exit(1)
        out = Path(f"{task_id}_spec.json")
        lib.export_task(task_id, str(out))
        print(f"Saved to {out}")
    else:
        print(f"Unknown action: {action}")
        print("Available actions: list, info, export, categories, sync, get")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
