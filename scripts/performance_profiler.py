#!/usr/bin/env python3
"""
Performance Profiling Tool for OpenEval Lab

This script provides comprehensive perform                   report_lines.append(f"Total memory: {total_size / 1024 / 1024:.2f} MB")
                report_lines.append("Top memory consumers:")            report_lines.append(f"### Snapshot {i+1}: {label}")
                report_lines.append(f"Total memory: {total_size / 1024 / 1024:.2f} MB")
                report_lines.append("Top memory consumers:")ce profiling including:
- Memory usage analysis
- CPU profiling
- Function call timing
- Bottleneck identification
- Performance recommendations
"""

import cProfile
import io
import pstats
import sys
import time
import tracemalloc
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    HAS_PSUTIL = False


class PerformanceProfiler:
    """Comprehensive performance profiler for OpenEval Lab."""

    def __init__(self):
        self.snapshots: List[Tuple[str, Any, float]] = []
        self.start_time: Optional[float] = None
        self.profiles: List[pstats.Stats] = []

    def start_profiling(self) -> None:
        """Start all profiling mechanisms."""
        # Start memory tracing
        tracemalloc.start()
        self.snapshots = []

        # Start CPU profiling
        self.profiler = cProfile.Profile()
        self.profiler.enable()

        # Record start time
        self.start_time = time.time()

        print("🔍 Performance profiling started...")

    def take_snapshot(self, label: str = "") -> None:
        """Take a memory snapshot."""
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append((label, snapshot, time.time()))
            print(f"📸 Memory snapshot taken: {label}")

    def stop_profiling(self) -> None:
        """Stop all profiling and generate reports."""
        # Stop CPU profiling
        self.profiler.disable()

        # Stop memory tracing
        if tracemalloc.is_tracing():
            final_snapshot = tracemalloc.take_snapshot()
            self.snapshots.append(("final", final_snapshot, time.time()))
            tracemalloc.stop()

        end_time = time.time()
        duration = end_time - (self.start_time or end_time)

        print(f"⏱️  Performance profiling completed in {duration:.2f} seconds")
        self.generate_report(duration)

    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a specific function."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            start_memory = self._get_memory_usage()

            result = func(*args, **kwargs)

            end_time = time.time()
            end_memory = self._get_memory_usage()

            print(f"⚡ Function {func.__name__}:")
            print(f"  Duration: {end_time - start_time:.4f} seconds")
            print(f"  Memory delta: {end_memory - start_memory:.2f} MB")
            return result
        return wrapper

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if HAS_PSUTIL and psutil is not None:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        return 0.0

    def generate_report(self, total_duration: float) -> None:
        """Generate comprehensive performance report."""
        report_lines = ["# Performance Profiling Report\n"]

        report_lines.append(f"Total Duration: {total_duration:.2f} seconds\n")

        # Memory analysis
        if self.snapshots:
            report_lines.append("## Memory Analysis\n")
            for i, (label, snapshot, timestamp) in enumerate(self.snapshots):
                stats = snapshot.statistics('lineno')
                total_size = sum(stat.size for stat in stats)
                report_lines.append(f"### Snapshot {i+1}: {label}")
                report_lines.append(".2f"                report_lines.append("Top memory consumers:"
                for stat in stats[:10]:
                    report_lines.append(f"  - {stat.traceback.format()[-1]}: {stat.size / 1024:.1f} KB")
                report_lines.append("")

        # CPU profiling
        report_lines.append("## CPU Profiling\n")
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        report_lines.append("```")
        report_lines.append(s.getvalue())
        report_lines.append("```")

        # System resources
        if HAS_PSUTIL:
            report_lines.append("## System Resources\n")
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            report_lines.append(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            report_lines.append(f"CPU Usage: {cpu_percent:.1f}%")            report_lines.append(f"CPU Usage: {cpu_percent}%")

        # Recommendations
        report_lines.append("## Recommendations\n")
        report_lines.extend(self._generate_recommendations())

        # Save report
        report_path = Path("performance_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"📄 Performance report saved to {report_path}")

    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        # Analyze memory usage
        if self.snapshots:
            initial_snapshot = self.snapshots[0][1]
            final_snapshot = self.snapshots[-1][1]
            initial_stats = initial_snapshot.statistics('lineno')
            final_stats = final_snapshot.statistics('lineno')

            initial_memory = sum(stat.size for stat in initial_stats)
            final_memory = sum(stat.size for stat in final_stats)
            memory_growth = final_memory - initial_memory

            if memory_growth > 50 * 1024 * 1024:  # 50MB
                recommendations.append("- Consider implementing memory-efficient data processing")
                recommendations.append("- Use generators instead of lists for large datasets")
                recommendations.append("- Implement proper cleanup of large objects")

        # General recommendations
        recommendations.extend([
            "- Profile individual functions with @profiler.profile_function decorator",
            "- Use asyncio for I/O-bound operations",
            "- Consider caching expensive computations",
            "- Optimize database queries and file I/O operations",
            "- Use multiprocessing for CPU-intensive tasks"
        ])

        return recommendations


def profile_script(func: Callable) -> Callable:
    """Decorator to profile an entire script."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        profiler = PerformanceProfiler()
        profiler.start_profiling()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.stop_profiling()
    return wrapper


def main():
    """Main entry point for performance profiling."""
    if len(sys.argv) < 2:
        print("Usage: python performance_profiler.py <script_to_profile.py> [args...]")
        sys.exit(1)

    script_path = sys.argv[1]
    script_args = sys.argv[2:]

    # Import and run the script with profiling
    profiler = PerformanceProfiler()
    profiler.start_profiling()

    try:
        # Execute the script
        exec(open(script_path).read())
    except Exception as e:
        print(f"Error profiling script: {e}")
    finally:
        profiler.stop_profiling()


if __name__ == "__main__":
    main()
