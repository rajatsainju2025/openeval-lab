"""Tests for batch cache operations."""

import pytest
from openeval.cache import PredictionCache


@pytest.fixture
def temp_cache(tmp_path):
    """Create a temporary cache for testing."""
    cache = PredictionCache(cache_dir=tmp_path / ".cache_test", compress=False)
    yield cache
    # Cleanup would happen here if needed


def test_cache_batch_get_and_set(temp_cache):
    """Test batch get and set operations."""
    # Set batch of items
    items = [
        ("key1", "value1"),
        ("key2", "value2"),
        ("key3", "value3"),
    ]
    temp_cache.set_batch(items)

    # Get batch
    results = temp_cache.get_batch(["key1", "key2", "key3"])

    assert results == ["value1", "value2", "value3"]


def test_cache_batch_get_with_missing_keys(temp_cache):
    """Test batch get with some missing keys."""
    items = [("key1", "value1"), ("key2", "value2")]
    temp_cache.set_batch(items)

    results = temp_cache.get_batch(["key1", "key2", "missing_key"])

    assert results[0] == "value1"
    assert results[1] == "value2"
    assert results[2] is None


def test_cache_batch_set_overwrites(temp_cache):
    """Test that batch set overwrites existing keys."""
    # First set
    temp_cache.set("key1", "old_value")

    # Batch set with new value
    temp_cache.set_batch([("key1", "new_value")])

    # Verify overwrite
    assert temp_cache.get("key1") == "new_value"


def test_cache_batch_empty(temp_cache):
    """Test batch operations with empty lists."""
    # Empty batch set should not fail
    temp_cache.set_batch([])

    # Empty batch get should return empty list
    results = temp_cache.get_batch([])
    assert results == []


def test_cache_batch_large_values(temp_cache):
    """Test batch operations with large values."""
    large_value = "x" * 10000
    items = [
        ("large1", large_value),
        ("large2", large_value),
        ("large3", large_value),
    ]
    temp_cache.set_batch(items)

    results = temp_cache.get_batch(["large1", "large2", "large3"])

    assert all(v == large_value for v in results)


def test_cache_batch_with_metadata(temp_cache):
    """Test batch set with metadata."""
    items = [
        ("key1", "value1"),
        ("key2", "value2"),
    ]
    metadata = [
        {"source": "test1"},
        {"source": "test2"},
    ]
    temp_cache.set_batch(items, metadata_list=metadata)

    # Verify items were set
    results = temp_cache.get_batch(["key1", "key2"])
    assert results == ["value1", "value2"]
