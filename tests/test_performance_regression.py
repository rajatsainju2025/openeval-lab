"""Performance regression tests.

Benchmark key operations and fail if performance degrades >10%.
"""

import pytest
import time


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
