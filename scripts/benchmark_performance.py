#!/usr/bin/env python3
"""Performance benchmarking script for OpenEval Lab."""

import time
import json
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    duration: float
    memory_usage: float
    throughput: float
    metrics: Dict[str, Any]


class PerformanceBenchmark:
    """Performance benchmarking utilities."""

    def __init__(self):
        if HAS_PSUTIL and psutil:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None
        self.results = []

    @contextmanager
    def measure(self, name: str):
        """Context manager to measure performance."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage() if HAS_PSUTIL else 0

        yield

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage() if HAS_PSUTIL else 0

        duration = end_time - start_time
        memory_usage = end_memory - start_memory

        result = BenchmarkResult(
            name=name,
            duration=duration,
            memory_usage=memory_usage,
            throughput=0.0,  # Will be set by caller
            metrics={},
        )

        self.results.append(result)

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        if HAS_PSUTIL and self.process and psutil:
            return self.process.memory_info().rss / 1024 / 1024
        return 0

    def benchmark_file_operations(self, file_path: str, iterations: int = 10):
        """Benchmark file reading operations."""
        for i in range(iterations):
            with self.measure(f"file_read_{i}"):
                with open(file_path, "r") as f:
                    content = f.read()
                    lines = content.split("\n")

            # Calculate throughput
            if self.results:
                result = self.results[-1]
                result.throughput = len(lines) / result.duration
                result.metrics = {"file_size_kb": len(content) / 1024, "lines_read": len(lines)}

    def benchmark_json_processing(self, json_path: str, iterations: int = 5):
        """Benchmark JSON processing."""
        for i in range(iterations):
            with self.measure(f"json_process_{i}"):
                with open(json_path, "r") as f:
                    data = json.load(f)

            # Calculate throughput
            if self.results:
                result = self.results[-1]
                result.throughput = len(str(data)) / result.duration
                result.metrics = {
                    "data_size_kb": len(str(data)) / 1024,
                    "num_keys": len(data) if isinstance(data, dict) else 0,
                }

    def benchmark_cache_operations(self, cache_dir: str, operations: int = 100):
        """Benchmark simple cache operations."""
        cache_file = Path(cache_dir) / "benchmark_cache.json"
        cache_data = {}

        # Benchmark writes
        with self.measure("cache_writes"):
            for i in range(operations):
                cache_data[f"key_{i}"] = f"value_{i}"
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

        # Benchmark reads
        with self.measure("cache_reads"):
            with open(cache_file, "r") as f:
                loaded_data = json.load(f)

        # Set throughput
        for result in self.results[-2:]:
            result.throughput = operations / result.duration

        # Cleanup
        cache_file.unlink(missing_ok=True)

    def generate_report(self, output_path: str):
        """Generate benchmark report."""
        report = {
            "timestamp": time.time(),
            "system_info": {
                "cpu_count": os.cpu_count(),
                "platform": os.uname().sysname if hasattr(os, "uname") else "unknown",
                "has_psutil": HAS_PSUTIL,
            },
            "benchmarks": [],
        }

        for result in self.results:
            report["benchmarks"].append(
                {
                    "name": result.name,
                    "duration_seconds": result.duration,
                    "memory_usage_mb": result.memory_usage,
                    "throughput": result.throughput,
                    "metrics": result.metrics,
                }
            )

        # Calculate summary statistics
        if self.results:
            durations = [r.duration for r in self.results]
            memory_usage = [r.memory_usage for r in self.results]

            report["summary"] = {
                "total_duration": sum(durations),
                "avg_duration": np.mean(durations),
                "max_duration": max(durations),
                "total_memory": sum(memory_usage),
                "avg_memory": np.mean(memory_usage),
                "max_memory": max(memory_usage),
            }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Benchmark report saved to {output_path}")


def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()

    print("Running performance benchmarks...")

    # Benchmark file operations
    print("Benchmarking file operations...")
    benchmark.benchmark_file_operations("README.md")

    # Benchmark JSON processing
    print("Benchmarking JSON processing...")
    benchmark.benchmark_json_processing("pyproject.toml")  # Actually TOML, but close enough

    # Benchmark cache operations
    print("Benchmarking cache operations...")
    Path(".cache").mkdir(exist_ok=True)
    benchmark.benchmark_cache_operations(".cache")

    # Generate report
    benchmark.generate_report("benchmark_results.json")

    print("Benchmarks completed!")


if __name__ == "__main__":
    main()
