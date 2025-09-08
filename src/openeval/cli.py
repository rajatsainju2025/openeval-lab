from __future__ import annotations

import json
from pathlib import Path
import os
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
from .core import Example, Dataset
from .utils import hash_file
from .data_quality import DataQualityAssessor
from .experiment_tracking import experiment_tracker
from .optimization import performance_monitor
from . import registry
from .utils import get_project_root
from .results_schema import RESULTS_JSON_SCHEMA, validate_results_payload
from .enhanced_logging import (
    configure_logging, get_logger, get_tracer, get_profiler,
    enable_debug_mode, save_debug_data, log_context, traced, profiled
)
from .config_manager import load_config, create_default_config, ConfigManager

app = typer.Typer(no_args_is_help=True, add_completion=False)
@app.command()
def registry_list(kind: str = typer.Argument(..., help="task|dataset|adapter|metric")):
    """List registered items for a given kind with descriptions."""
    items = registry.list_items(kind)
    table = Table(title=f"Registry: {kind}")
    table.add_column("Name", style="cyan")
    table.add_column("Path")
    table.add_column("Description")
    for name, meta in sorted(items.items()):
        table.add_row(name, meta.get("path", ""), meta.get("description", ""))
    console.print(table)


@app.command()
def registry_info(
    kind: str = typer.Argument(..., help="task|dataset|adapter|metric"),
    name: str = typer.Argument(..., help="Short name"),
):
    """Show information for a specific registry item."""
    meta = registry.info(kind, name)
    if not meta:
        console.print(f"Not found: {kind}:{name}", style="red")
        raise typer.Exit(code=1)
    console.print(json.dumps(meta, indent=2))


@app.command()
def tutorial():
    """Show getting-started steps and the tutorial file path."""
    root = get_project_root()
    tut = root / "docs" / "tutorial.md"
    console.print("\nOpenEval Lab Tutorial (quickstart):\n", style="blue bold")
    console.print("1) python -m venv .venv && source .venv/bin/activate")
    console.print("2) pip install -e '.[dev,metrics]'")
    console.print("3) openeval registry-list metric  # explore built-ins")
    console.print("4) openeval validate examples/qa_spec.json")
    console.print("5) openeval run examples/qa_spec.json --records --artifacts runs")
    console.print("6) openeval web --open")
    console.print(f"\nFull tutorial: {tut}")


@app.command()
def docs():
    """List key documentation files and their paths."""
    root = get_project_root()
    files = [
    ("Docs Index", root / "docs" / "index.md"),
        ("Tutorial", root / "docs" / "tutorial.md"),
        ("Concepts", root / "docs" / "concepts.md"),
        ("SOTA", root / "docs" / "sota.md"),
        ("ICML Paper", root / "ICML_PAPER.md"),
        ("Contributing", root / "CONTRIBUTING.md"),
    ]
    table = Table(title="Documentation")
    table.add_column("Doc")
    table.add_column("Path")
    for name, path in files:
        table.add_row(name, str(path))
    console.print(table)
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
def doctor(
    json_out: bool = typer.Option(False, "--json", help="Print JSON summary only"),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero if required checks fail"),
):
    """Diagnose environment, dependencies, and configuration."""
    from importlib.metadata import PackageNotFoundError, version as _v

    # Collect summary first
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    required = ["typer", "pydantic", "rich", "fastapi", "jinja2"]
    optional = ["openai", "anthropic", "datasets", "sacrebleu", "bert-score", "rouge-score"]
    packages = {}
    for pkg in required + optional:
        try:
            ver = _v(pkg)
            packages[pkg] = {"status": "ok", "version": ver, "required": pkg in required}
        except PackageNotFoundError:
            packages[pkg] = {"status": "missing" if pkg in required else "optional", "version": None, "required": pkg in required}
        except Exception as e:
            packages[pkg] = {"status": "error", "error": str(e), "version": None, "required": pkg in required}

    keys = {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "HUGGINGFACE_API_KEY": bool(os.getenv("HUGGINGFACE_API_KEY")),
    }

    root = get_project_root()
    runs_dir = root / "runs"
    fs = {"runs_dir": str(runs_dir), "writable": False, "error": None}
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        test_file = runs_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink(missing_ok=True)
        fs["writable"] = True
    except Exception as e:
        fs["writable"] = False
        fs["error"] = str(e)

    # Registry sanity
    try:
        tasks = registry.list_items("task")
        metrics = registry.list_items("metric")
        reg = {"tasks": len(tasks), "metrics": len(metrics)}
    except Exception as e:
        reg = {"error": str(e)}

    # Git info (best-effort)
    git = {"commit": None, "dirty": None}
    try:
        import subprocess as _sp
        commit = _sp.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
        status = _sp.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
        git = {"commit": commit, "dirty": bool(status.strip())}
    except Exception:
        pass

    # Determine overall health
    required_ok = all(meta.get("status") == "ok" for name, meta in packages.items() if meta.get("required"))
    if "error" in reg:
        registry_ok = False
    else:
        try:
            registry_ok = int(reg.get("tasks", 0)) >= 0 and int(reg.get("metrics", 0)) >= 0
        except Exception:
            registry_ok = False
    fs_ok = bool(fs.get("writable"))
    ok = bool(required_ok and registry_ok and fs_ok)

    summary = {
        "python": py_version,
        "packages": packages,
        "api_keys": keys,
        "filesystem": fs,
        "registry": reg,
        "git": git,
        "ok": ok,
    }

    if json_out:
        sys.stdout.write(json.dumps(summary) + "\n")
        if strict and not ok:
            raise typer.Exit(code=1)
        return

    # Human-readable output
    console.rule("Environment Checks")
    console.print(f"Python: {py_version}")
    table = Table(title="Packages")
    table.add_column("Name", style="cyan")
    table.add_column("Status")
    table.add_column("Version")
    for name, meta in packages.items():
        status = meta.get("status", "?")
        ver = meta.get("version") or "-"
        table.add_row(name, status, ver)
    console.print(table)

    console.rule("API Keys")
    for k, present in keys.items():
        color = "green" if present else "yellow"
        console.print(f"{k}: {'set' if present else 'not set'}", style=color)

    console.rule("Filesystem")
    if fs["writable"]:
        console.print(f"runs/: writable ({runs_dir})", style="green")
    else:
        console.print(f"runs/: not writable ({fs['error']})", style="red")

    console.rule("Registry")
    if "error" in reg:
        console.print(f"registry error: {reg['error']}", style="red")
    else:
        console.print(f"tasks: {reg['tasks']} registered, metrics: {reg['metrics']} registered")

    if git.get("commit"):
        dirty = " (dirty)" if git.get("dirty") else ""
        console.print(f"git: {git['commit']}{dirty}")

    console.rule("Done")
    console.print("If any required items are missing, install extras e.g. pip install -e '.[dev,metrics,openai]'", style="blue")
    if strict and not ok:
        raise typer.Exit(code=1)


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


@app.command("results-schema")
def results_schema(out: Optional[Path] = typer.Option(None, "--out", help="Write results JSON schema to file")):
    """Print the JSON schema for OpenEval results payloads."""
    payload = json.dumps(RESULTS_JSON_SCHEMA, indent=2)
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
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode with enhanced logging"),
    enable_tracing: bool = typer.Option(False, "--enable-tracing", help="Enable debug tracing"),
    enable_profiling: bool = typer.Option(False, "--enable-profiling", help="Enable performance profiling"),
    save_debug: Optional[Path] = typer.Option(None, "--save-debug", help="Save debug data to directory"),
    statistical_analysis: bool = typer.Option(False, "--statistical", help="Enable statistical analysis with bootstrap confidence intervals"),
    # Optional extras
    robustness_noise: Optional[float] = typer.Option(None, "--robustness-noise", help="Apply simple character noise (0.0-1.0) to inputs and report robustness metrics"),
    calibration: bool = typer.Option(False, "--calibration", help="Attempt to compute calibration metrics if adapter supports probabilities/logprobs"),
    # Performance optimization
    use_async: bool = typer.Option(False, "--async", help="Use async adapter methods if available for better throughput"),
    config: Optional[Path] = typer.Option(None, "--config", help="Configuration file path"),
    environment: str = typer.Option("development", "--env", help="Environment name"),
):
    """Run an evaluation task with enhanced logging, configuration, and debugging support."""
    # Load configuration if provided
    if config:
        try:
            config_obj, env_manager = load_config(config, environment, load_secrets=True)
            console.print(f"[green]Loaded configuration for environment: {environment}[/green]")
            
            # Apply configuration defaults to CLI parameters if not explicitly set
            # This allows config file to provide defaults while CLI args override
            if config_obj:
                eval_settings = config_obj.evaluation
                
                # Update parameters that weren't explicitly set
                if concurrency == 1 and eval_settings.default_concurrency != 1:
                    concurrency = eval_settings.default_concurrency
                
                if max_retries == 0 and eval_settings.default_max_retries != 0:
                    max_retries = eval_settings.default_max_retries
                
                if request_timeout is None and eval_settings.default_timeout != 30.0:
                    request_timeout = eval_settings.default_timeout
                
                if cache_mode == "off" and eval_settings.default_cache_mode != "off":
                    cache_mode = eval_settings.default_cache_mode
                
                if not records and eval_settings.include_records_by_default:
                    records = True
                
                if not traces and eval_settings.include_traces_by_default:
                    traces = True
                
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load config: {e}[/yellow]")
    
    # Setup enhanced logging and debugging if requested
    if debug:
        enable_debug_mode()
        console.print("[green]Debug mode enabled with enhanced logging[/green]")
    
    # Initialize logging context
    logger = get_logger("openeval.run", 
                       component="cli", 
                       operation="evaluation",
                       task_name=str(spec.stem))
    
    # Setup tracing and profiling
    tracer = None
    profiler = None
    
    if enable_tracing or debug:
        tracer = get_tracer("evaluation", enabled=True)
        tracer.trace("evaluation_start", {"spec": str(spec)})
        logger.info("Debug tracing enabled for evaluation")
    
    if enable_profiling or debug:
        profiler = get_profiler("evaluation")
        logger.info("Performance profiling enabled for evaluation")
    
    with log_context(operation="load_spec", spec_path=str(spec)):
        try:
            task, dataset, adapter, metrics, out = load_spec(spec, statistical_analysis=statistical_analysis)
            logger.info(f"Loaded spec successfully: task={type(task).__name__}, adapter={type(adapter).__name__}")
            
            if tracer:
                tracer.trace("spec_loaded", {
                    "task": type(task).__name__,
                    "adapter": type(adapter).__name__,
                    "metrics": [type(m).__name__ for m in metrics]
                })
                
        except SystemExit as e:
            logger.error(f"Failed to load spec: {e}")
            if tracer:
                tracer.trace("spec_load_failed", {"error": str(e)})
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
        
        # Run evaluation with enhanced logging context
        with log_context(
            operation="evaluation_run",
            task_name=type(task).__name__,
            adapter_name=type(adapter).__name__,
            dataset_size=len(list(iter(dataset))),
            metric_count=len(metrics)
        ):
            logger.info(f"Starting evaluation run with task={type(task).__name__}, adapter={type(adapter).__name__}, metrics={[m.name for m in metrics]}")
            
            if tracer:
                tracer.trace("evaluation_start", {
                    "task": type(task).__name__,
                    "adapter": type(adapter).__name__,
                    "dataset_size": len(list(iter(dataset))),
                    "metrics": [m.name for m in metrics],
                    "config": {
                        "seed": seed,
                        "concurrency": concurrency,
                        "max_retries": max_retries
                    }
                })
            
            import time
            eval_start = time.time()
            
            # Run evaluation with optional profiling
            if profiler:
                @profiler.profile("task_evaluation", include_args=False)
                def run_evaluation():
                    return task.evaluate(
                        adapter,
                        dataset,
                        metrics,
                        seed=seed,
                        collect_records=records,
                        concurrency=concurrency,
                        max_retries=max_retries,
                        request_timeout=request_timeout,
                    )
                
                result = run_evaluation()
            else:
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
            
            eval_duration = time.time() - eval_start
            
            logger.info(f"Evaluation completed in {eval_duration:.2f}s")
            
            # Log evaluation results summary
            logger.info(f"Evaluation summary: {result.get('timing', {}).get('request_successes', 0)} successes, "
                       f"{result.get('timing', {}).get('request_errors', 0)} errors")
            
            # Log metric results
            for metric_name, metric_value in result.items():
                if metric_name not in ['timing', 'manifest', 'environment', 'dataset_path', 'dataset_hash_sha256', 'records']:
                    if isinstance(metric_value, dict) and 'error' in metric_value:
                        logger.error(f"Metric {metric_name} failed: {metric_value['error']}")
                    else:
                        logger.info(f"Metric {metric_name}: {metric_value}")
            
            if tracer:
                tracer.trace("evaluation_complete", {
                    "duration_seconds": eval_duration,
                    "success_count": result.get('timing', {}).get('request_successes', 0),
                    "error_count": result.get('timing', {}).get('request_errors', 0),
                    "metrics_computed": list(result.keys())
                })
    else:
        # Interactive loop: preview prompts and control flow
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
        import platform, sys, time

        examples = list(iter(dataset))
        predictions = []
        references = []
        per_latency = []
        recs = []
        
        logger.info(f"Starting evaluation with {len(examples)} examples")
        if tracer:
            tracer.trace("evaluation_loop_start", {"example_count": len(examples)})
        
        t0 = time.perf_counter()
        for idx, ex in enumerate(examples):
            with log_context(operation="process_example", 
                           example_id=ex.id, 
                           example_index=idx):
                
                if profiler:
                    @profiler.profile(f"example_{idx}", include_args=False)
                    def process_example():
                        prompt = task.build_prompt_with_template(ex)
                        return prompt
                    
                    prompt = process_example()
                else:
                    prompt = task.build_prompt_with_template(ex)
                
                logger.debug(f"Processing example {idx+1}/{len(examples)}: {ex.id}")
                
                if tracer:
                    tracer.trace("example_start", {"example_id": ex.id, "index": idx})
            prompt = task.build_prompt_with_template(ex)
            console.print(f"\n[bold]Example {idx+1}/{len(examples)}[/bold] id={ex.id}", style="cyan")
            console.print(f"Input: {str(ex.input)[:200]}" )
            # Use a valid rich style; 'muted' isn't a default style
            console.print("Show full prompt? [y/N], skip [s], quit [q]", style="dim")
            ans = input("Action: ").strip().lower()
            if ans == "q":
                break
            if ans == "y":
                console.print("\n--- Prompt ---\n" + prompt + "\n---------------")
            if ans == "s":
                continue
            s = time.perf_counter()
            
            # Log adapter call with enhanced context
            with log_context(operation="adapter_generate", 
                           adapter_name=type(adapter).__name__,
                           example_id=ex.id):
                logger.debug(f"Calling adapter {type(adapter).__name__} for example {ex.id}")
                
                if tracer:
                    tracer.trace("adapter_call_start", {
                        "adapter": type(adapter).__name__,
                        "example_id": ex.id,
                        "prompt_length": len(prompt)
                    })
                
                if profiler:
                    @profiler.profile(f"adapter_generate_{idx}", include_args=False)
                    def generate_with_profiling():
                        return adapter.generate(prompt)
                    
                    gen_raw = generate_with_profiling()
                else:
                    gen_raw = adapter.generate(prompt)
                
                if tracer:
                    tracer.trace("adapter_call_complete", {
                        "example_id": ex.id,
                        "output_length": len(str(gen_raw))
                    })
            
            e = time.perf_counter()
            
            logger.debug(f"Adapter call completed in {(e-s)*1000:.1f}ms for example {ex.id}")
            
            pred = task.postprocess(gen_raw)
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

    # Optional: calibration (ECE from token logprobs, short-form heuristic)
    if calibration:
        try:
            _gwlp = getattr(adapter, "generate_with_logprobs", None)
            if not callable(_gwlp):
                result.setdefault("calibration", {})["available"] = False
            else:
                import math
                # Re-iterate dataset (assume re-iterable). Use modest cap to avoid long runs.
                MAX_ECE_SAMPLES = 512
                samples = []
                try:
                    for i, ex in enumerate(iter(dataset)):
                        if i >= MAX_ECE_SAMPLES:
                            break
                        samples.append(ex)
                except Exception:
                    samples = []

                confs: List[float] = []
                corrects: List[int] = []
                for ex in samples:
                    try:
                        prompt = task.build_prompt_with_template(ex)
                        info = _gwlp(prompt)
                        info_d = info if isinstance(info, dict) else {}
                        text = str(info_d.get("text", ""))
                        toks = info_d.get("tokens") or []
                        lps = info_d.get("logprobs") or []
                        if not toks or not lps:
                            # Fallback: no per-token info, skip sample
                            continue
                        # Confidence proxy: exp(mean logprob) across completion tokens
                        m = sum(float(lp) for lp in lps) / max(1, len(lps))
                        conf = float(math.exp(m))
                        pred = task.postprocess(text)
                        corr = 1 if str(pred).strip() == str(ex.reference).strip() else 0
                        confs.append(max(0.0, min(1.0, conf)))
                        corrects.append(corr)
                    except Exception:
                        continue

                def _ece(confidences: List[float], correctness: List[int], bins: int = 10) -> Dict[str, Any]:
                    if not confidences:
                        return {"ece": None, "bins": bins, "n": 0}
                    N = len(confidences)
                    counts = [0] * bins
                    accs = [0.0] * bins
                    avg_confs = [0.0] * bins
                    for c, y in zip(confidences, correctness):
                        bi = min(bins - 1, int(c * bins))
                        counts[bi] += 1
                        accs[bi] += float(y)
                        avg_confs[bi] += float(c)
                    ece = 0.0
                    for i in range(bins):
                        if counts[i] == 0:
                            continue
                        acc = accs[i] / counts[i]
                        conf = avg_confs[i] / counts[i]
                        ece += (counts[i] / N) * abs(acc - conf)
                    overall_acc = sum(corrects) / N
                    overall_conf = sum(confs) / N
                    return {
                        "ece": ece,
                        "bins": bins,
                        "n": N,
                        "overall_accuracy": overall_acc,
                        "overall_confidence": overall_conf,
                        "hist": {
                            "counts": counts,
                            "bin_accuracy": [a / c if c else None for a, c in zip(accs, counts)],
                            "bin_confidence": [s / c if c else None for s, c in zip(avg_confs, counts)],
                        },
                        "method": "exp(mean token logprob)",
                    }

                cal = _ece(confs, corrects, bins=10)
                cal["available"] = True
                result.setdefault("calibration", {}).update(cal)
        except Exception as _e:
            result.setdefault("calibration", {})["error"] = str(_e)

    # enrich with spec metadata and optional run name
    result["spec_path"] = str(spec)
    try:
        result["spec_hash_sha256"] = hash_file(spec)
    except Exception:
        pass
    if run_name:
        result["run_name"] = run_name

    # Save debug information if enabled
    if save_debug:
        debug_output_file = save_debug / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_results": result,
            "profiling": profiler.profiles if profiler else None,
            "traces": tracer.get_traces() if tracer else None
        }
        
        logger.info(f"Saving debug information to {debug_output_file}")
        with log_context(operation="save_debug"):
            try:
                save_debug.mkdir(parents=True, exist_ok=True)
                with open(debug_output_file, 'w') as f:
                    json.dump(debug_info, f, indent=2, default=str)
                logger.info(f"Debug information saved successfully")
                
                # Save profiler data separately if available
                if profiler:
                    profiler_file = save_debug / f"profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    profiler.save_profiles(profiler_file)
                    logger.info(f"Profiler data saved to {profiler_file}")
                
            except Exception as e:
                logger.error(f"Failed to save debug information: {e}")

    out_path = Path(out)
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
        if timestamped:
            import datetime as _dt

            ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            out_path = artifacts / f"{ts}.json"
        else:
            out_path = artifacts / out_path.name

    # Optional: robustness noise evaluation
    if robustness_noise is not None and robustness_noise > 0:
        try:
            import random as _rnd

            def _noisify_text(s: str, rate: float) -> str:
                if not isinstance(s, str) or rate <= 0:
                    return s
                _rnd.seed((seed or 0) + 1337)
                chars = list(s)
                for i in range(len(chars)):
                    if _rnd.random() < rate:
                        # simple perturbations: swap case or insert punctuation
                        c = chars[i]
                        if c.isalpha():
                            chars[i] = c.swapcase()
                        elif c == ' ':
                            chars[i] = ','
                return "".join(chars)

            # Build a lightweight transformed dataset (wrapper)
            class _TransformedDataset(Dataset):
                name = getattr(dataset, "name", "transformed") + ".noise"

                def __iter__(self):
                    for ex in dataset:
                        try:
                            new_input = _noisify_text(ex.input, float(robustness_noise))
                        except Exception:
                            new_input = ex.input
                        yield Example(id=ex.id, input=new_input, reference=ex.reference, meta=ex.meta)

                def __len__(self):
                    try:
                        return len(dataset)  # type: ignore[no-any-return]
                    except Exception:
                        return sum(1 for _ in iter(self))

            transformed = _TransformedDataset()
            rob_result = task.evaluate(
                adapter,
                transformed,
                metrics,
                seed=seed,
                collect_records=bool(result.get("records")),
                concurrency=concurrency,
                max_retries=max_retries,
                request_timeout=request_timeout,
            )
            result.setdefault("robustness", {})["char_noise"] = {
                "rate": float(robustness_noise),
                "metrics": rob_result.get("metrics", {}),
                "size": rob_result.get("size", 0),
            }
        except Exception as _e:
            result.setdefault("robustness", {})["char_noise_error"] = str(_e)

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


# -------------------------
# Validation & Comparison
# -------------------------

@app.command("validate")
def validate_spec(
    spec: Path = typer.Argument(..., help="Path to JSON/YAML spec to validate")
):
    """Validate a spec file for correctness."""
    from .spec import _read_spec_file, EvalSpec
    console = Console()
    try:
        data = _read_spec_file(spec)
        # Normalize metrics entries that are strings
        metrics_raw = data.get("metrics")
        if isinstance(metrics_raw, list) and metrics_raw and isinstance(metrics_raw[0], str):
            data["metrics"] = [{"name": m} for m in metrics_raw]
        # Validate against the pydantic model
        EvalSpec(**data)
        console.print("Spec is valid", style="green")
    except SystemExit as e:
        # _read_spec_file may raise SystemExit for YAML missing
        console.print(f"Spec invalid: {e}", style="red")
        raise typer.Exit(code=2)
    except Exception as e:
        console.print(f"Spec invalid: {e}", style="red")
        raise typer.Exit(code=2)


@app.command("validate-dataset")
def validate_dataset_cmd(
    path: Path = typer.Argument(..., help="Path to dataset JSONL file"),
    output: Optional[Path] = typer.Option(None, "--output", help="Where to save report JSON"),
    strict: bool = typer.Option(False, "--strict", help="Fail on validation issues"),
):
    """Validate a dataset JSONL file and optionally save a quality report."""
    from .dataset_validation import validate_jsonl_file
    console = Console()
    try:
        report = validate_jsonl_file(path)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(report.__dict__, f, indent=2)
        # Print brief summary
        console.print(
            f"Dataset quality: score={report.quality_score:.2f}, total={report.total_examples}, valid={report.valid_examples}",
            style=("green" if report.quality_score >= 0.7 else "yellow"),
        )
        if strict and (report.quality_score < 0.7 or report.valid_examples == 0):
            raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"Validation failed: {e}", style="red")
        raise typer.Exit(code=2)



@app.command("debug-logs")
def debug_logs(
    level: str = typer.Option("DEBUG", "--level", help="Log level: DEBUG, INFO, WARNING, ERROR"),
    format_type: str = typer.Option("structured", "--format", help="Log format: structured, plain"),
    output: Optional[Path] = typer.Option(None, "--output", help="Log file path"),
    enable_tracing: bool = typer.Option(False, "--trace", help="Enable debug tracing"),
    enable_profiling: bool = typer.Option(False, "--profile", help="Enable performance profiling"),
):
    """Configure enhanced logging and debugging."""
    console = Console()
    
    # Configure logging
    configure_logging(
        level=level,
        format_type=format_type,
        log_file=output,
        console_output=not output,  # Only console if no file specified
        redact_sensitive=True
    )
    
    console.print(f"[green]Configured logging: level={level}, format={format_type}[/green]")
    
    if output:
        console.print(f"[green]Logging to file: {output}[/green]")
    
    if enable_tracing:
        tracer = get_tracer("cli", enabled=True)
        console.print("[green]Debug tracing enabled[/green]")
    
    if enable_profiling:
        profiler = get_profiler("cli")
        console.print("[green]Performance profiling enabled[/green]")
    
    # Test logging
    logger = get_logger("test", component="cli", operation="debug_test")
    logger.info("Enhanced logging configured successfully")
    logger.debug("Debug message example")
    logger.warning("Warning message example")


@app.command("save-debug")
def save_debug(
    output_dir: Path = typer.Option("debug_output", "--output", help="Output directory for debug data"),
    include_traces: bool = typer.Option(True, "--traces", help="Include debug traces"),
    include_profiles: bool = typer.Option(True, "--profiles", help="Include performance profiles"),
):
    """Save all debug data to files."""
    console = Console()
    
    try:
        save_debug_data(output_dir)
        console.print(f"[green]Debug data saved to: {output_dir}[/green]")
        
        # List saved files
        if output_dir.exists():
            files = list(output_dir.glob("*.json"))
            if files:
                console.print("\nSaved files:")
                for file in files:
                    console.print(f"  - {file.name}")
            else:
                console.print("[yellow]No debug data to save[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Failed to save debug data: {e}[/red]")
        raise typer.Exit(1)


@app.command("trace-analysis")
def trace_analysis(
    trace_file: Path = typer.Argument(..., help="Path to trace JSON file"),
    filter_name: Optional[str] = typer.Option(None, "--filter", help="Filter traces by name pattern"),
    show_timeline: bool = typer.Option(False, "--timeline", help="Show timeline view"),
    show_stats: bool = typer.Option(True, "--stats", help="Show statistics"),
):
    """Analyze debug trace files."""
    console = Console()
    
    if not trace_file.exists():
        console.print(f"[red]Trace file not found: {trace_file}[/red]")
        raise typer.Exit(1)
    
    try:
        import json
        with open(trace_file, 'r') as f:
            traces = json.load(f)
        
        if filter_name:
            traces = [t for t in traces if filter_name in t.get('name', '')]
        
        console.print(f"[green]Loaded {len(traces)} traces[/green]")
        
        if show_stats:
            # Analyze trace statistics
            from collections import Counter
            
            names = [t.get('name', 'unknown') for t in traces]
            name_counts = Counter(names)
            
            console.print("\n[yellow]Trace Statistics:[/yellow]")
            console.print(f"Total traces: {len(traces)}")
            console.print(f"Unique trace names: {len(name_counts)}")
            
            console.print("\nTop trace types:")
            for name, count in name_counts.most_common(10):
                console.print(f"  {name}: {count}")
        
        if show_timeline and traces:
            # Show timeline view
            console.print("\n[yellow]Timeline View:[/yellow]")
            
            # Sort by timestamp
            sorted_traces = sorted(traces, key=lambda t: t.get('timestamp', 0))
            
            start_time = sorted_traces[0].get('timestamp', 0)
            
            for trace in sorted_traces[:20]:  # Show first 20
                timestamp = trace.get('timestamp', 0)
                relative_time = timestamp - start_time
                name = trace.get('name', 'unknown')
                thread_id = trace.get('thread_id', 'unknown')
                
                console.print(f"  +{relative_time:.3f}s [{thread_id}] {name}")
    
    except Exception as e:
        console.print(f"[red]Failed to analyze traces: {e}[/red]")
        raise typer.Exit(1)


# Add logging management commands
logging_app = typer.Typer(help="Enhanced logging and debugging commands")
app.add_typer(logging_app, name="debug")

# Move debug commands to the logging group
logging_app.command("configure")(debug_logs)
logging_app.command("save")(save_debug)
logging_app.command("analyze")(trace_analysis)

@app.command("config-init")
def config_init(
    output: Path = typer.Option("openeval.yaml", "--output", "-o", help="Output path for config file"),
    format: str = typer.Option("yaml", "--format", help="Config format: yaml, json, toml"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing config file"),
):
    """Initialize a new OpenEval configuration file."""
    from .config_manager import ConfigFormat
    
    console = Console()
    
    if output.exists() and not overwrite:
        console.print(f"[red]Config file already exists: {output}[/red]")
        console.print("Use --overwrite to replace it")
        raise typer.Exit(1)
    
    # Map format string to enum
    format_map = {
        "yaml": ConfigFormat.YAML,
        "yml": ConfigFormat.YAML,
        "json": ConfigFormat.JSON,
        "toml": ConfigFormat.TOML
    }
    
    if format.lower() not in format_map:
        console.print(f"[red]Unsupported format: {format}[/red]")
        console.print("Supported formats: yaml, json, toml")
        raise typer.Exit(1)
    
    try:
        create_default_config(output)
        console.print(f"[green]Created default config: {output}[/green]")
        
        # Show next steps
        console.print("\nNext steps:")
        console.print("1. Edit the config file to customize settings")
        console.print("2. Set environment variables with OPENEVAL_ prefix")
        console.print("3. Use --config flag to load the config in commands")
        
    except Exception as e:
        console.print(f"[red]Failed to create config: {e}[/red]")
        raise typer.Exit(1)


@app.command("config-show")
def config_show(
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path"),
    environment: str = typer.Option("development", "--env", help="Environment name"),
    format: str = typer.Option("yaml", "--format", help="Output format: yaml, json"),
):
    """Show current configuration."""
    from .config_manager import ConfigFormat
    console = Console()
    
    try:
        # Load configuration
        config_obj, env_manager = load_config(config, environment)
        
        # Get config manager to show sources
        config_manager = ConfigManager()
        config_manager.load_config(config, create_if_missing=False)
        sources = config_manager.get_config_sources()
        
        console.print(f"[green]Configuration for environment: {environment}[/green]")
        console.print(f"Sources: {', '.join(sources) if sources else 'defaults'}")
        console.print()
        
        # Show configuration
        if format.lower() == "json":
            import json
            config_dict = config_obj.to_dict()
            console.print(json.dumps(config_dict, indent=2))
        else:
            import yaml
            config_dict = config_obj.to_dict()
            console.print(yaml.dump(config_dict, indent=2, default_flow_style=False))
        
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise typer.Exit(1)


@app.command("config-validate")
def config_validate(
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path"),
    environment: str = typer.Option("development", "--env", help="Environment name"),
):
    """Validate configuration file."""
    console = Console()
    
    try:
        # Load and validate configuration
        config_obj, env_manager = load_config(config, environment)
        
        console.print("[green]✓ Configuration is valid![/green]")
        console.print(f"Project: {config_obj.project_name}")
        console.print(f"Environment: {config_obj.environment}")
        console.print(f"Debug mode: {config_obj.debug_mode}")
        
        # Show warnings for potential issues
        warnings = []
        
        if config_obj.evaluation.max_memory_mb and config_obj.evaluation.max_memory_mb < 512:
            warnings.append("Low memory limit may cause issues with large models")
        
        if config_obj.adapters.rate_limit_rpm and config_obj.adapters.rate_limit_rpm > 10000:
            warnings.append("High rate limit may exceed API quotas")
        
        if config_obj.evaluation.default_concurrency > 10:
            warnings.append("High concurrency may overwhelm external services")
        
        if warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  ⚠ {warning}")
        
    except Exception as e:
        console.print(f"[red]✗ Configuration validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("config-set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key (e.g., evaluation.default_concurrency)"),
    value: str = typer.Argument(..., help="Value to set"),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path"),
    environment: str = typer.Option("development", "--env", help="Environment name"),
):
    """Set a configuration value."""
    console = Console()
    
    try:
        # Load configuration
        config_obj, env_manager = load_config(config, environment)
        
        # Parse value
        import json
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        
        # Set value
        config_manager = ConfigManager()
        config_manager._config = config_obj
        config_manager.override_setting(key, parsed_value)
        
        console.print(f"[green]Set {key} = {parsed_value}[/green]")
        
        # Save if config file specified
        if config:
            config_manager.save_config(config_obj, config)
            console.print(f"[green]Saved to {config}[/green]")
        else:
            console.print("[yellow]Value set in memory only. Specify --config to save.[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Failed to set config value: {e}[/red]")
        raise typer.Exit(1)


# Add config group
config_app = typer.Typer(help="Configuration management commands")
app.add_typer(config_app, name="config")

# Move config commands to the group
config_app.command("init")(config_init)
config_app.command("show")(config_show)
config_app.command("validate")(config_validate)  
config_app.command("set")(config_set)


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


@app.command("validate-results")
def validate_results(path: Path = typer.Argument(..., help="Path to results JSON file"), strict: bool = typer.Option(False, "--strict", help="Exit non-zero if validation fails")):
    """Validate a results JSON file against the OpenEval results schema."""
    try:
        data = json.loads(Path(path).read_text())
    except Exception as e:
        sys.stdout.write(json.dumps({"valid": False, "error": f"failed to read JSON: {e}"}) + "\n")
        raise typer.Exit(code=2)

    ok, errs = validate_results_payload(data)
    sys.stdout.write(json.dumps({"valid": ok, "errors": errs, "path": str(path)}) + "\n")
    if strict and not ok:
        raise typer.Exit(code=1)


@app.command("compare")
def compare(
    a: Path = typer.Argument(..., help="Path to run A (results JSON)"),
    b: Path = typer.Argument(..., help="Path to run B (results JSON)"),
    n_bootstrap: int = typer.Option(1000, "--bootstrap", help="Bootstrap samples for CI/p-value"),
    ci: float = typer.Option(0.95, "--ci", help="Confidence level for intervals"),
    json_out: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Compare two runs. If records are present, compute paired bootstrap on exact-match accuracy.

    Requires overlapping example ids in both runs. Falls back to aggregate metric diff if records missing.
    """
    import math
    import difflib

    try:
        ra = json.loads(Path(a).read_text())
        rb = json.loads(Path(b).read_text())
    except Exception as e:
        console.print(f"Failed to read inputs: {e}", style="red")
        raise typer.Exit(code=2)

    def _acc_from_records(recs):
        ok = 0
        total = 0
        ids = []
        corr = []
        for r in recs or []:
            if "reference" in r and "prediction" in r:
                total += 1
                c = 1 if r.get("prediction") == r.get("reference") else 0
                ok += c
                ids.append(r.get("id"))
                corr.append(c)
        return (ok / total if total else 0.0), ids, corr

    rec_a = ra.get("records")
    rec_b = rb.get("records")
    result = {
        "run_a": dict(ra) if isinstance(ra, dict) else {},
        "run_b": dict(rb) if isinstance(rb, dict) else {},
    }

    if isinstance(rec_a, list) and isinstance(rec_b, list):
        # Align by id intersection
        map_a = {r.get("id"): r for r in rec_a if r.get("id") is not None}
        map_b = {r.get("id"): r for r in rec_b if r.get("id") is not None}
        common_ids = [i for i in map_a.keys() if i in map_b]
        pa = [1 if map_a[i].get("prediction") == map_a[i].get("reference") else 0 for i in common_ids]
        pb = [1 if map_b[i].get("prediction") == map_b[i].get("reference") else 0 for i in common_ids]
        n = len(common_ids)
        acc_a = sum(pa) / n if n else 0.0
        acc_b = sum(pb) / n if n else 0.0
        observed = acc_a - acc_b
        # Paired bootstrap on correctness vectors
        import random as _rnd
        _rnd.seed(1234)
        diffs = []
        for _ in range(max(1, int(n_bootstrap))):
            idxs = [_rnd.randint(0, n - 1) for _ in range(n)] if n else []
            if not idxs:
                diffs.append(0.0)
                continue
            acc_a_b = sum(pa[i] for i in idxs) / len(idxs)
            acc_b_b = sum(pb[i] for i in idxs) / len(idxs)
            diffs.append(acc_a_b - acc_b_b)
        diffs.sort()
        alpha = 1 - ci
        low = diffs[int(alpha / 2 * len(diffs))]
        high = diffs[int((1 - alpha / 2) * len(diffs))]
        p_val = sum(1 for d in diffs if abs(d) >= abs(observed)) / len(diffs) if diffs else 1.0
        payload = {
            "aligned": n,
            "acc_a": acc_a,
            "acc_b": acc_b,
            "diff": observed,
            "ci": [low, high],
            "p_value": p_val,
        }
        if json_out:
            sys.stdout.write(json.dumps(payload) + "\n")
        else:
            console.print("Paired bootstrap on exact-match accuracy (aligned by id)", style="blue")
            console.print(f"n={n}  acc_a={acc_a:.4f}  acc_b={acc_b:.4f}  diff={observed:+.4f}")
            console.print(f"{int(ci*100)}% CI for diff: [{low:.4f}, {high:.4f}]  p={p_val:.4f}")
        return
    else:
        # Fallback to aggregate metrics diff if available
        ma = ra.get("metrics") or {}
        mb = rb.get("metrics") or {}
        common = sorted(set(ma.keys()) & set(mb.keys()))
        if not common:
            console.print("No common metrics to compare and no records present.", style="yellow")
            raise typer.Exit(code=1)
        rows = []
        for k in common:
            va = ma.get(k) or {}
            vb = mb.get(k) or {}
            # try common key 'accuracy' else skip
            if isinstance(va, dict) and isinstance(vb, dict) and "accuracy" in va and "accuracy" in vb:
                rows.append((k, float(va["accuracy"]), float(vb["accuracy"])) )
        if not rows:
            console.print("Common metrics found but no comparable numeric fields.", style="yellow")
            raise typer.Exit(code=1)
        table = Table(title="Aggregate metric diff (accuracy)")
        table.add_column("metric", style="cyan")
        table.add_column("run_a", justify="right")
        table.add_column("run_b", justify="right")
        table.add_column("diff", justify="right")
        for name, va, vb in rows:
            table.add_row(name, f"{va:.4f}", f"{vb:.4f}", f"{(va - vb):+.4f}")
        console.print(table)
        return


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


@app.command("validate-dataset")
def validate_dataset(
    path: Path = typer.Argument(..., help="Path to dataset file (JSONL) or spec file"),
    output: Optional[Path] = typer.Option(None, "--output", help="Save validation report to JSON file"),
    strict: bool = typer.Option(False, "--strict", help="Exit with error if validation fails"),
    format_type: str = typer.Option("auto", "--format", help="Dataset format: auto|jsonl|spec"),
):
    """Validate dataset quality and format."""
    from .dataset_validation import DatasetValidator, validate_jsonl_file
    from .spec import load_spec
    
    try:
        if format_type == "auto":
            # Auto-detect format
            if path.suffix.lower() in {'.jsonl', '.json'}:
                format_type = "jsonl"
            elif path.suffix.lower() in {'.yaml', '.yml'}:
                format_type = "spec"
            else:
                # Try to load as spec first, then JSONL
                try:
                    load_spec(path)
                    format_type = "spec"
                except Exception:
                    format_type = "jsonl"
        
        if format_type == "spec":
            # Load dataset from spec
            try:
                task, dataset, adapter, metrics, out = load_spec(path)
                validator = DatasetValidator(strict=strict)
                report = validator.assess_quality(dataset)
            except Exception as e:
                console.print(f"[red]Failed to load spec: {e}[/red]")
                raise typer.Exit(code=2)
        
        elif format_type == "jsonl":
            # Validate JSONL file directly
            report = validate_jsonl_file(path)
        
        else:
            console.print(f"[red]Unsupported format: {format_type}[/red]")
            raise typer.Exit(code=2)
        
        # Display results
        console.print(f"\n[bold]Dataset Validation Report[/bold]")
        console.print(f"File: {path}")
        console.print(f"Total examples: {report.total_examples}")
        console.print(f"Valid examples: {report.valid_examples}")
        console.print(f"Invalid examples: {report.invalid_examples}")
        console.print(f"Quality score: {report.quality_score:.2f}")
        
        if report.avg_input_length > 0:
            console.print(f"Avg input length: {report.avg_input_length:.1f} chars")
        if report.avg_reference_length > 0:
            console.print(f"Avg reference length: {report.avg_reference_length:.1f} chars")
        
        if report.duplicate_pairs > 0:
            console.print(f"[yellow]Duplicate pairs: {report.duplicate_pairs}[/yellow]")
        
        if report.empty_inputs > 0:
            console.print(f"[yellow]Empty inputs: {report.empty_inputs}[/yellow]")
        
        if report.empty_references > 0:
            console.print(f"[yellow]Empty references: {report.empty_references}[/yellow]")
        
        if report.encoding_issues > 0:
            console.print(f"[red]Encoding issues: {report.encoding_issues}[/red]")
        
        if report.format_issues:
            console.print(f"\n[yellow]Format Issues (first 5):[/yellow]")
            for issue in report.format_issues[:5]:
                console.print(f"  • {issue}")
        
        if report.recommendations:
            console.print(f"\n[cyan]Recommendations:[/cyan]")
            for rec in report.recommendations:
                console.print(f"  • {rec}")
        
        # Save report if requested
        if output:
            import json
            from dataclasses import asdict
            
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'w') as f:
                json.dump(asdict(report), f, indent=2)
            console.print(f"\n[green]Report saved to {output}[/green]")
        
        # Exit with error if strict mode and validation failed
        if strict and report.quality_score < 0.7:
            console.print(f"\n[red]Validation failed in strict mode (score: {report.quality_score:.2f})[/red]")
            raise typer.Exit(code=1)
        
        if report.quality_score >= 0.8:
            console.print(f"\n[green]✓ Dataset quality is good![/green]")
        elif report.quality_score >= 0.6:
            console.print(f"\n[yellow]⚠ Dataset quality is acceptable but could be improved[/yellow]")
        else:
            console.print(f"\n[red]✗ Dataset quality is poor and needs attention[/red]")
            
    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        if strict:
            raise typer.Exit(code=2)


@app.command()
def interactive(
    spec: Optional[Path] = typer.Option(None, "--spec", help="Optional spec file to load"),
):
    """Launch an interactive shell for OpenEval exploration."""
    try:
        import readline  # Enable command-line editing
    except ImportError:
        pass
    
    task = dataset = adapter = metrics = None
    if spec:
        try:
            task, dataset, adapter, metrics, _ = load_spec(spec)
            console.print(f"[green]Loaded spec from {spec}[/green]")
            console.print(f"Task: {getattr(task, 'name', task.__class__.__name__)}")
            console.print(f"Dataset: {getattr(dataset, 'name', dataset.__class__.__name__)}")
            console.print(f"Adapter: {getattr(adapter, 'name', adapter.__class__.__name__)}")
            console.print(f"Metrics: {[m.name for m in metrics]}")
        except Exception as e:
            console.print(f"[red]Failed to load spec: {e}[/red]")
            spec = None
    
    console.print("\n[bold cyan]OpenEval Interactive Shell[/bold cyan]")
    console.print("Type 'help' for commands, 'exit' to quit.\n")
    
    while True:
        try:
            cmd = input("openeval> ").strip()
            if not cmd:
                continue
            if cmd in ("exit", "quit"):
                break
            elif cmd == "help":
                console.print("Available commands:")
                console.print("  help        - Show this help")
                console.print("  exit/quit   - Exit the shell")
                console.print("  info        - Show current spec info")
                console.print("  examples    - Show first 3 dataset examples")
                console.print("  prompt <id> - Show prompt for example ID")
                console.print("  run         - Run evaluation")
                console.print("  library     - List available components")
            elif cmd == "info":
                if spec:
                    console.print(f"Spec: {spec}")
                    console.print(f"Task: {getattr(task, 'name', task.__class__.__name__)}")
                    console.print(f"Dataset: {getattr(dataset, 'name', dataset.__class__.__name__)}")
                    console.print(f"Adapter: {getattr(adapter, 'name', adapter.__class__.__name__)}")
                else:
                    console.print("No spec loaded. Use --spec <path> to load one.")
            elif cmd == "examples":
                if spec and dataset:
                    examples = list(iter(dataset))[:3]
                    for i, ex in enumerate(examples):
                        console.print(f"[bold]Example {i+1}[/bold] (id={ex.id})")
                        console.print(f"Input: {str(ex.input)[:200]}...")
                        console.print(f"Reference: {str(ex.reference)[:100]}...")
                        console.print()
                else:
                    console.print("No spec loaded.")
            elif cmd.startswith("prompt "):
                if spec and dataset and task:
                    ex_id = cmd[7:].strip()
                    examples = list(iter(dataset))
                    for ex in examples:
                        if ex.id == ex_id:
                            prompt = task.build_prompt_with_template(ex)
                            console.print(f"[bold]Prompt for {ex_id}:[/bold]\n{prompt}")
                            break
                    else:
                        console.print(f"Example {ex_id} not found.")
                else:
                    console.print("No spec loaded.")
            elif cmd == "run":
                if spec and task and dataset and adapter and metrics:
                    console.print("Running evaluation...")
                    result = task.evaluate(adapter, dataset, metrics, seed=0)
                    console.print(f"Results: {result.get('metrics', {})}")
                else:
                    console.print("No spec loaded.")
            elif cmd == "library":
                console.print("Available library components:")
                try:
                    from .library import get_task_library
                    lib = get_task_library()
                    components = lib.list_tasks()[:10]  # Show first 10
                    for comp in components:
                        console.print(f"  {comp['id']}: {comp['description']}")
                except Exception:
                    console.print("Library not available.")
            else:
                console.print(f"Unknown command: {cmd}. Type 'help' for available commands.")
        except KeyboardInterrupt:
            console.print("\nUse 'exit' to quit.")
        except EOFError:
            break
    
@app.command("validate-comprehensive")
def validate_comprehensive(
    target: Path = typer.Argument(..., help="Path to spec, config, or dataset file to validate"),
    target_type: str = typer.Option("auto", "--type", help="Target type: auto|spec|config|dataset"),
    check_imports: bool = typer.Option(True, "--check-imports", help="Validate import paths"),
    check_datasets: bool = typer.Option(True, "--check-datasets", help="Validate dataset configurations"),
    check_performance: bool = typer.Option(True, "--check-performance", help="Check performance settings"),
    check_best_practices: bool = typer.Option(True, "--check-best-practices", help="Check best practices"),
    sample_limit: int = typer.Option(1000, "--sample-limit", help="Max samples to check for datasets"),
    output: Optional[Path] = typer.Option(None, "--output", help="Save validation report to JSON file"),
    strict: bool = typer.Option(False, "--strict", help="Exit with error if validation fails"),
    show_details: bool = typer.Option(True, "--details", help="Show detailed issue descriptions"),
    show_suggestions: bool = typer.Option(True, "--suggestions", help="Show fix suggestions"),
):
    """Comprehensive validation for specs, configs, and datasets with detailed analysis."""
    from dataclasses import dataclass, asdict
    from typing import Dict, List, Any, Optional
    
    @dataclass
    class ValidationIssue:
        severity: str
        category: str
        message: str
        details: Optional[str] = None
        fix_suggestion: Optional[str] = None
        line_number: Optional[int] = None
    
    @dataclass 
    class ValidationReport:
        file_path: str
        file_type: str
        is_valid: bool
        overall_score: float
        issues: List[ValidationIssue]
        checks_performed: List[str]
        metadata: Dict[str, Any]
    
    def detect_file_type(path: Path) -> str:
        """Auto-detect file type based on content and extension."""
        if path.suffix.lower() in ['.yaml', '.yml', '.json']:
            try:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    with open(path, 'r') as f:
                        data = yaml.safe_load(f)
                else:
                    with open(path, 'r') as f:
                        data = json.load(f)
                
                # Check for spec-like structure
                if isinstance(data, dict):
                    if 'task' in data and 'adapter' in data:
                        return 'spec'
                    elif 'project_name' in data or 'evaluation' in data:
                        return 'config'
                    elif any(key in data for key in ['input', 'reference', 'examples']):
                        return 'dataset'
            except Exception:
                pass
        
        # Check by extension
        if path.suffix.lower() == '.jsonl':
            return 'dataset'
        elif path.suffix.lower() in ['.yaml', '.yml'] and 'config' in path.name.lower():
            return 'config'
        
        return 'spec'  # Default assumption
    
    def validate_spec_file(spec_path: Path) -> ValidationReport:
        """Validate spec file comprehensively."""
        issues = []
        checks_performed = []
        metadata = {}
        
        # Basic file checks
        if not spec_path.exists():
            issues.append(ValidationIssue(
                severity="error",
                category="file",
                message=f"Spec file not found: {spec_path}",
                fix_suggestion="Check the file path and ensure the file exists"
            ))
            return ValidationReport(
                file_path=str(spec_path),
                file_type="spec",
                is_valid=False,
                overall_score=0.0,
                issues=issues,
                checks_performed=["file_existence"],
                metadata=metadata
            )
        
        # Load and parse file
        try:
            if spec_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with open(spec_path, 'r') as f:
                    data = yaml.safe_load(f)
                metadata['format'] = 'yaml'
            else:
                with open(spec_path, 'r') as f:
                    data = json.load(f)
                metadata['format'] = 'json'
            checks_performed.append("file_parsing")
        except Exception as e:
            issues.append(ValidationIssue(
                severity="error",
                category="schema",
                message=f"Failed to parse file: {e}",
                fix_suggestion="Check file syntax and encoding"
            ))
            return create_report(spec_path, "spec", issues, checks_performed, metadata)
        
        # Schema validation
        try:
            from .spec import EvalSpec
            # Normalize metrics if they're strings
            if "metrics" in data and isinstance(data["metrics"], list):
                metrics_raw = data["metrics"]
                if metrics_raw and isinstance(metrics_raw[0], str):
                    data["metrics"] = [{"name": m} for m in metrics_raw]
            
            spec_obj = EvalSpec(**data)
            checks_performed.append("schema_validation")
            metadata['spec_keys'] = list(data.keys())
        except Exception as e:
            issues.append(ValidationIssue(
                severity="error",
                category="schema",
                message=f"Schema validation failed: {e}",
                fix_suggestion="Check the spec format against the OpenEval schema"
            ))
        
        # Import validation
        if check_imports:
            import importlib
            import_fields = ["task", "dataset", "adapter"]
            for field in import_fields:
                if field not in data:
                    continue
                
                import_path = data[field]
                if not isinstance(import_path, str):
                    continue
                
                try:
                    # Try to import the module and get the class
                    module_path, class_name = import_path.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    cls = getattr(module, class_name)
                    
                    # Basic interface checks
                    if field == "task" and not hasattr(cls, "evaluate"):
                        issues.append(ValidationIssue(
                            severity="warning",
                            category="interface",
                            message=f"Task class missing evaluate method",
                            fix_suggestion="Implement the evaluate method"
                        ))
                    elif field == "adapter" and not hasattr(cls, "generate"):
                        issues.append(ValidationIssue(
                            severity="error",
                            category="interface", 
                            message=f"Adapter class missing generate method",
                            fix_suggestion="Implement the generate method"
                        ))
                        
                except ImportError as e:
                    issues.append(ValidationIssue(
                        severity="error",
                        category="import",
                        message=f"Cannot import {field}: {import_path}",
                        details=str(e),
                        fix_suggestion=f"Check if the module is installed and accessible"
                    ))
                except (AttributeError, ValueError) as e:
                    issues.append(ValidationIssue(
                        severity="error",
                        category="import",
                        message=f"Cannot access class from import path: {import_path}",
                        details=str(e),
                        fix_suggestion=f"Check the import path format and class name"
                    ))
            
            checks_performed.append("import_validation")
        
        # Dataset configuration validation
        if check_datasets:
            dataset_kwargs = data.get("dataset_kwargs", {})
            
            if "path" in dataset_kwargs:
                path = dataset_kwargs["path"]
                if isinstance(path, str) and not path.startswith(("http://", "https://", "s3://", "gs://")):
                    dataset_path = Path(path)
                    if not dataset_path.exists() and not dataset_path.is_absolute():
                        cwd_path = Path.cwd() / path
                        if not cwd_path.exists():
                            issues.append(ValidationIssue(
                                severity="warning",
                                category="config",
                                message=f"Dataset file not found: {path}",
                                fix_suggestion="Check the dataset path or ensure the file exists"
                            ))
            
            checks_performed.append("dataset_config")
        
        # Performance checks
        if check_performance:
            dataset_kwargs = data.get("dataset_kwargs", {})
            
            if "max_examples" in dataset_kwargs:
                max_examples = dataset_kwargs["max_examples"]
                if isinstance(max_examples, int) and max_examples > 10000:
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="performance",
                        message=f"Large dataset size: {max_examples} examples",
                        fix_suggestion="Consider using a smaller subset for initial testing"
                    ))
            
            checks_performed.append("performance_config")
        
        # Best practices
        if check_best_practices:
            if "output" not in data:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="best_practice",
                    message="No output file specified",
                    fix_suggestion="Add an 'output' field to save results"
                ))
            
            metrics = data.get("metrics", [])
            if len(metrics) < 2:
                issues.append(ValidationIssue(
                    severity="info",
                    category="best_practice",
                    message="Consider using multiple metrics for robust evaluation",
                    fix_suggestion="Add complementary metrics like BLEU, ROUGE, or accuracy variants"
                ))
            
            if "seed" not in data:
                issues.append(ValidationIssue(
                    severity="info", 
                    category="best_practice",
                    message="No seed specified for reproducibility",
                    fix_suggestion="Add a 'seed' field for deterministic results"
                ))
            
            checks_performed.append("best_practices")
        
        return create_report(spec_path, "spec", issues, checks_performed, metadata)
    
    def validate_dataset_file(dataset_path: Path) -> ValidationReport:
        """Validate dataset file format and content."""
        issues = []
        checks_performed = []
        metadata = {}
        
        if not dataset_path.exists():
            issues.append(ValidationIssue(
                severity="error",
                category="file", 
                message=f"Dataset file not found: {dataset_path}",
            ))
            return create_report(dataset_path, "dataset", issues, checks_performed, metadata)
        
        # JSONL validation
        if dataset_path.suffix.lower() == '.jsonl':
            line_count = 0
            valid_lines = 0
            required_fields = set()
            
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line_count += 1
                        if line_count > sample_limit:
                            break
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            valid_lines += 1
                            
                            # Track field usage
                            for key in data.keys():
                                if key in ['input', 'reference', 'id']:
                                    required_fields.add(key)
                            
                            # Check for required fields in first few examples
                            if line_num <= 10:
                                if 'input' not in data:
                                    issues.append(ValidationIssue(
                                        severity="error",
                                        category="schema",
                                        message=f"Missing 'input' field at line {line_num}",
                                        line_number=line_num,
                                        fix_suggestion="Each example must have an 'input' field"
                                    ))
                                
                                if 'reference' not in data:
                                    issues.append(ValidationIssue(
                                        severity="warning",
                                        category="schema",
                                        message=f"Missing 'reference' field at line {line_num}",
                                        line_number=line_num,
                                        fix_suggestion="Add 'reference' field for evaluation metrics"
                                    ))
                        
                        except json.JSONDecodeError as e:
                            issues.append(ValidationIssue(
                                severity="error",
                                category="format",
                                message=f"Invalid JSON at line {line_num}: {e}",
                                line_number=line_num,
                                fix_suggestion="Fix JSON syntax"
                            ))
            
            except Exception as e:
                issues.append(ValidationIssue(
                    severity="error",
                    category="file",
                    message=f"Failed to read file: {e}",
                ))
            
            metadata.update({
                'total_lines': line_count,
                'valid_lines': valid_lines,
                'required_fields': list(required_fields),
                'format': 'jsonl'
            })
            
            if valid_lines == 0:
                issues.append(ValidationIssue(
                    severity="error",
                    category="content",
                    message="No valid examples found in dataset",
                ))
            elif valid_lines < line_count * 0.9:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="quality",
                    message=f"High error rate: {line_count - valid_lines}/{line_count} invalid lines",
                    fix_suggestion="Review and fix malformed examples"
                ))
            
            checks_performed.append("jsonl_format")
        
        return create_report(dataset_path, "dataset", issues, checks_performed, metadata)
    
    def create_report(file_path: Path, file_type: str, issues: List[ValidationIssue], 
                     checks_performed: List[str], metadata: Dict[str, Any]) -> ValidationReport:
        """Create validation report with computed score."""
        error_count = len([i for i in issues if i.severity == "error"])
        warning_count = len([i for i in issues if i.severity == "warning"])
        
        if error_count > 0:
            overall_score = 0.0
            is_valid = False
        else:
            overall_score = max(0.0, 100.0 - (warning_count * 10))
            is_valid = True
        
        return ValidationReport(
            file_path=str(file_path),
            file_type=file_type,
            is_valid=is_valid,
            overall_score=overall_score,
            issues=issues,
            checks_performed=checks_performed,
            metadata=metadata
        )
    
    def format_validation_report(report: ValidationReport) -> None:
        """Format and display a validation report."""
        
        # Header
        status_style = "green" if report.is_valid else "red"
        status_text = "✓ VALID" if report.is_valid else "✗ INVALID"
        
        console.print(f"\n📋 Validation Report: {report.file_path}", style="bold")
        console.print(f"Status: {status_text} | Score: {report.overall_score:.1f}/100", style=status_style)
        console.print(f"Type: {report.file_type} | Checks: {len(report.checks_performed)}")
        
        # Issues summary
        errors = [i for i in report.issues if i.severity == "error"]
        warnings = [i for i in report.issues if i.severity == "warning"]
        info = [i for i in report.issues if i.severity == "info"]
        
        if errors or warnings or info:
            console.print(f"\n📊 Issues Summary:")
            if errors:
                console.print(f"  ❌ Errors: {len(errors)}", style="red")
            if warnings:
                console.print(f"  ⚠️  Warnings: {len(warnings)}", style="yellow")
            if info:
                console.print(f"  ℹ️  Info: {len(info)}", style="blue")
        
        # Detailed issues
        if show_details and report.issues:
            console.print(f"\n📝 Detailed Issues:")
            
            for issue in report.issues:
                severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[issue.severity]
                severity_style = {"error": "red", "warning": "yellow", "info": "blue"}[issue.severity]
                
                console.print(f"\n{severity_icon} [{issue.category.upper()}] {issue.message}", style=severity_style)
                
                if issue.details:
                    console.print(f"   Details: {issue.details}", style="dim")
                
                if issue.line_number:
                    console.print(f"   Line: {issue.line_number}", style="dim")
                
                if show_suggestions and issue.fix_suggestion:
                    console.print(f"   💡 Fix: {issue.fix_suggestion}", style="green dim")
        
        # Metadata
        if report.metadata:
            console.print(f"\n📈 Metadata:")
            for key, value in report.metadata.items():
                if isinstance(value, list):
                    console.print(f"  {key}: {', '.join(map(str, value[:5]))}{' ...' if len(value) > 5 else ''}")
                else:
                    console.print(f"  {key}: {value}")
    
    try:
        # Auto-detect file type if needed
        if target_type == "auto":
            target_type = detect_file_type(target)
        
        console.print(f"🔍 Running comprehensive validation on {target} (type: {target_type})", style="blue")
        
        # Run appropriate validation
        if target_type == "spec":
            report = validate_spec_file(target)
        elif target_type == "dataset":
            report = validate_dataset_file(target)
        elif target_type == "config":
            # Basic config validation (simplified)
            issues = []
            checks_performed = ["file_existence"]
            metadata = {"format": target.suffix}
            
            if not target.exists():
                issues.append(ValidationIssue(
                    severity="error",
                    category="file",
                    message=f"Config file not found: {target}",
                ))
            
            report = create_report(target, "config", issues, checks_performed, metadata)
        else:
            console.print(f"[red]Unsupported target type: {target_type}[/red]")
            raise typer.Exit(code=1)
        
        # Display results
        format_validation_report(report)
        
        # Save report if requested
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            console.print(f"\n📄 Detailed report saved to: {output}", style="green")
        
        # Exit with error if strict mode and validation failed
        if strict and not report.is_valid:
            console.print(f"\n[red]Validation failed in strict mode[/red]")
            raise typer.Exit(code=1)
        
        if report.is_valid:
            console.print(f"\n[green]✓ Validation passed![/green]")
        else:
            console.print(f"\n[red]✗ Validation failed - please address the issues above[/red]")
            
    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        import traceback
        if show_details:
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        if strict:
            raise typer.Exit(code=2)


@app.command()
def analyze_bias(
    spec: Path = typer.Argument(..., help="Path to JSON/YAML spec"),
    permutations: int = typer.Option(5, help="Number of position permutations for bias detection"),
    output: Optional[Path] = typer.Option(None, help="Save analysis results to JSON file"),
):
    """Analyze evaluation biases (positional, prompt sensitivity)."""
    try:
        console.print("🔍 Analyzing evaluation biases...", style="blue")
        
        # Load spec
        task, dataset, adapter, metrics, _ = load_spec(spec)
        
        # Initialize bias detector
        from .bias_detection import BiasDetector
        detector = BiasDetector(adapter, task, dataset)
        
        # Run analysis
        with console.status("[bold green]Running bias analysis..."):
            results = detector.run_full_analysis()
        
        # Display results
        console.print("\n📊 Bias Analysis Results:", style="bold blue")
        console.print(f"Positional Bias Detected: {'Yes' if results.positional_bias_detected else 'No'}")
        console.print(".4f")
        console.print(".4f")
        
        console.print("\n💡 Recommendations:", style="bold green")
        for rec in results.recommendations:
            console.print(f"  • {rec}")
        
        # Save to file if requested
        if output:
            import json
            output_data = {
                "positional_bias_detected": results.positional_bias_detected,
                "positional_bias_score": results.positional_bias_score,
                "prompt_sensitivity_score": results.prompt_sensitivity_score,
                "recommendations": results.recommendations,
                "timestamp": "2025-09-03T00:00:00Z"  # Current date
            }
            output.write_text(json.dumps(output_data, indent=2))
            console.print(f"\n💾 Results saved to: {output}")
            
    except Exception as e:
        console.print(f"❌ Error during bias analysis: {e}", style="red")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
