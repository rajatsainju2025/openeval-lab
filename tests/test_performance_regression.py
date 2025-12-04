"""Performance regression tests using pytest-benchmark.

Tests critical path performance to detect regressions over time.
Run with: pytest tests/test_performance_regression.py --benchmark-only

Also includes legacy time-based tests for backward compatibility.
"""

import pytest
import time
import json
import tempfile
from pathlib import Path

from openeval.core import Example
from openeval.cache import PredictionCache, CacheConfig
from openeval.datasets.jsonl import JSONLinesDataset
from openeval.metrics.accuracy import ExactMatch


# Pytest-benchmark fixtures and tests
@pytest.fixture
def sample_examples():
    """Generate sample examples for benchmarking."""
    return [
        Example(input=f"Question {i}", reference=f"Answer {i}", metadata={"id": i})
        for i in range(100)
    ]


@pytest.fixture
def temp_cache_dir():
    """Temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_jsonl_file():
    """Temporary JSONL dataset file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(100):
            f.write(json.dumps({"input": f"Question {i}", "reference": f"Answer {i}"}) + "\n")
        f.flush()
        path = Path(f.name)

    yield path
    path.unlink()


# Pytest-benchmark tests
class TestCacheBenchmarks:
    """Benchmark cache operations with pytest-benchmark."""

    def test_cache_write_benchmark(self, benchmark, temp_cache_dir, sample_examples):
        """Benchmark cache write throughput."""
        config = CacheConfig(
            cache_dir=temp_cache_dir,
            max_size_mb=100,
            compression_enabled=False,
        )
        cache = PredictionCache(config=config)

        def write_predictions():
            for ex in sample_examples:
                cache.set(ex.input, "prediction", adapter_name="test")

        benchmark(write_predictions)

    def test_cache_read_benchmark(self, benchmark, temp_cache_dir, sample_examples):
        """Benchmark cache read throughput."""
        config = CacheConfig(
            cache_dir=temp_cache_dir,
            max_size_mb=100,
            compression_enabled=False,
        )
        cache = PredictionCache(config=config)

        # Pre-populate cache
        for ex in sample_examples:
            cache.set(ex.input, "prediction", adapter_name="test")

        def read_predictions():
            for ex in sample_examples:
                cache.get(ex.input, adapter_name="test")

        benchmark(read_predictions)


class TestDatasetBenchmarks:
    """Benchmark dataset operations."""

    def test_jsonl_iteration_benchmark(self, benchmark, temp_jsonl_file):
        """Benchmark JSONL dataset iteration."""
        dataset = JSONLinesDataset(temp_jsonl_file)

        def iterate_dataset():
            count = 0
            for _ in dataset:
                count += 1
            return count

        result = benchmark(iterate_dataset)
        assert result == 100


class TestMetricBenchmarks:
    """Benchmark metric computation."""

    def test_exact_match_benchmark(self, benchmark):
        """Benchmark ExactMatch metric computation."""
        metric = ExactMatch()
        predictions = [f"answer_{i}" for i in range(1000)]
        references = [f"answer_{i}" for i in range(1000)]

        def compute_metric():
            return metric.compute(predictions, references)

        result = benchmark(compute_metric)
        assert result["accuracy"] == 1.0


# Legacy time-based tests (backward compatible)


def test_cache_performance():
    """Test cache operations don't regress."""
    from src.openeval.cache import Cache

    cache = Cache()
    start = time.time()

    for i in range(10000):
        cache.set(f"key_{i}", f"value_{i}")

    elapsed = time.time() - start
    # Should complete 10k operations in < 5 seconds
    assert elapsed < 5.0, f"Cache set too slow: {elapsed}s"


def test_validation_performance():
    """Test validation doesn't regress."""
    from src.openeval.validation_unified import validate_config

    start = time.time()

    for i in range(1000):
        validate_config({"test": "config"})

    elapsed = time.time() - start
    # Should validate 1k configs in < 2 seconds
    assert elapsed < 2.0, f"Validation too slow: {elapsed}s"


def test_string_normalization_performance():
    """Test string normalization doesn't regress."""
    from src.openeval.metrics_cache import normalize_text_cached

    start = time.time()

    for i in range(10000):
        normalize_text_cached(f"Text with spaces   {i}")

    elapsed = time.time() - start
    # Should normalize 10k texts in < 1 second (due to caching)
    assert elapsed < 1.0, f"Normalization too slow: {elapsed}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
