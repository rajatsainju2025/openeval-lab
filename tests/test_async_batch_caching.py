"""Tests for async batch caching optimization."""

import asyncio
import pytest
from unittest.mock import Mock
from openeval.async_evaluation_engine import (
    AsyncEvaluationEngine,
    AsyncTaskConfig,
    AsyncEvaluationResult,
    AsyncAdapterWrapper,
)
from openeval.cache import PredictionCache


@pytest.mark.asyncio
async def test_cached_generate_batch_with_hits():
    """Test batch caching with cache hits."""
    config = AsyncTaskConfig(max_concurrent_requests=5)
    engine = AsyncEvaluationEngine(config)

    # Mock cache
    mock_cache = Mock(spec=PredictionCache)
    mock_cache.get = Mock(side_effect=lambda k: "cached_result" if k == "key_0" else None)
    mock_cache.set = Mock()
    engine.set_cache(mock_cache)

    # Mock adapter
    mock_adapter = Mock()
    async_adapter = AsyncAdapterWrapper(mock_adapter)

    # Mock agenerate to return a unique result
    async def mock_agenerate(prompt, **kwargs):
        return f"generated_{prompt}"

    async_adapter.agenerate = mock_agenerate

    prompts = ["prompt_0", "prompt_1", "prompt_2"]
    cache_keys = ["key_0", "key_1", "key_2"]

    results = await engine._cached_generate_batch(async_adapter, prompts, cache_keys)

    # Check results order and values
    assert len(results) == 3
    assert results[0][0] == "cached_result"  # Cache hit
    assert results[0][1] is True  # marked as cached
    assert results[1][0] == "generated_prompt_1"  # Cache miss
    assert results[1][1] is False  # not marked as cached
    assert results[2][0] == "generated_prompt_2"
    assert results[2][1] is False

    # Verify cache stats
    assert engine.cache_stats.hits == 1
    assert engine.cache_stats.misses == 2


@pytest.mark.asyncio
async def test_cached_generate_batch_order_preservation():
    """Test that batch caching preserves original order."""
    config = AsyncTaskConfig(max_concurrent_requests=3)
    engine = AsyncEvaluationEngine(config)

    # Mock cache with all misses
    mock_cache = Mock(spec=PredictionCache)
    mock_cache.get = Mock(return_value=None)
    mock_cache.set = Mock()
    engine.set_cache(mock_cache)

    # Mock adapter
    mock_adapter = Mock()
    async_adapter = AsyncAdapterWrapper(mock_adapter)

    results_dict = {"p0": "gen_0", "p1": "gen_1", "p2": "gen_2", "p3": "gen_3"}

    async def mock_agenerate(prompt, **kwargs):
        await asyncio.sleep(0.01 * int(prompt[-1]))  # Staggered delays
        return results_dict[prompt]

    async_adapter.agenerate = mock_agenerate

    prompts = ["p0", "p1", "p2", "p3"]
    cache_keys = ["ck0", "ck1", "ck2", "ck3"]

    results = await engine._cached_generate_batch(async_adapter, prompts, cache_keys)

    # Verify order is preserved despite async completion
    assert results[0][0] == "gen_0"
    assert results[1][0] == "gen_1"
    assert results[2][0] == "gen_2"
    assert results[3][0] == "gen_3"


@pytest.mark.asyncio
async def test_evaluate_batch_optimized():
    """Test evaluate_batch_optimized method."""
    config = AsyncTaskConfig(max_concurrent_requests=5)
    engine = AsyncEvaluationEngine(config)

    # Mock cache
    mock_cache = Mock(spec=PredictionCache)
    mock_cache.get = Mock(return_value=None)
    mock_cache.set = Mock()
    engine.set_cache(mock_cache)

    # Create a real adapter with async generate
    class TestAdapter:
        async def agenerate(self, prompt, **kwargs):
            return f"output_for_{prompt}"

    adapter = TestAdapter()

    prompts = ["p1", "p2", "p3"]
    cache_keys = ["ck1", "ck2", "ck3"]
    priorities = [1, 2, 3]

    results = await engine.evaluate_batch_optimized(
        adapter, prompts, cache_keys=cache_keys, priorities=priorities
    )

    assert len(results) == 3
    assert all(isinstance(r, AsyncEvaluationResult) for r in results)
    assert results[0].index == 0
    assert results[1].index == 1
    assert results[2].index == 2
    assert results[0].prediction == "output_for_p1"
    assert results[1].prediction == "output_for_p2"
    assert results[2].prediction == "output_for_p3"
    assert results[0].priority == 1
    assert results[1].priority == 2
    assert results[2].priority == 3


@pytest.mark.asyncio
async def test_evaluate_batch_optimized_error_handling():
    """Test error handling in evaluate_batch_optimized."""
    config = AsyncTaskConfig(max_concurrent_requests=5)
    engine = AsyncEvaluationEngine(config)

    # Create an adapter that raises errors
    class FailingAdapter:
        async def agenerate(self, prompt, **kwargs):
            raise RuntimeError("Adapter error")

    adapter = FailingAdapter()

    prompts = ["p1", "p2"]
    cache_keys = ["ck1", "ck2"]

    results = await engine.evaluate_batch_optimized(adapter, prompts, cache_keys=cache_keys)

    # Should return error results for all prompts
    assert len(results) == 2
    assert all(r.error is not None for r in results)


@pytest.mark.asyncio
async def test_cached_generate_batch_without_cache():
    """Test batch generation when cache is None."""
    config = AsyncTaskConfig(max_concurrent_requests=3)
    engine = AsyncEvaluationEngine(config)

    # Don't set cache
    assert engine.cache is None

    # Mock adapter
    mock_adapter = Mock()
    async_adapter = AsyncAdapterWrapper(mock_adapter)

    async def mock_agenerate(prompt, **kwargs):
        return f"result_{prompt}"

    async_adapter.agenerate = mock_agenerate

    prompts = ["p1", "p2"]
    cache_keys = ["ck1", "ck2"]

    results = await engine._cached_generate_batch(async_adapter, prompts, cache_keys)

    assert len(results) == 2
    assert results[0][0] == "result_p1"
    assert results[1][0] == "result_p2"
    assert results[0][1] is False  # not cached
    assert results[1][1] is False


@pytest.mark.asyncio
async def test_cached_generate_batch_semaphore_limiting():
    """Test that batch generation respects semaphore limits."""
    config = AsyncTaskConfig(max_concurrent_requests=2)
    engine = AsyncEvaluationEngine(config)

    # Track concurrent acquisitions
    concurrent_count = 0
    max_concurrent = 0

    # Mock adapter
    mock_adapter = Mock()
    async_adapter = AsyncAdapterWrapper(mock_adapter)

    async def mock_agenerate(prompt, **kwargs):
        nonlocal concurrent_count, max_concurrent
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.01)
        concurrent_count -= 1
        return f"result_{prompt}"

    async_adapter.agenerate = mock_agenerate

    prompts = ["p1", "p2", "p3", "p4"]
    cache_keys = ["ck1", "ck2", "ck3", "ck4"]

    results = await engine._cached_generate_batch(async_adapter, prompts, cache_keys)

    # Should not exceed semaphore limit
    assert max_concurrent <= 2
    assert len(results) == 4
