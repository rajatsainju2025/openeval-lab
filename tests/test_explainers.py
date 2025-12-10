"""Comprehensive tests for explainer modules.

Tests cover cache manager, factory, chain, async, metrics, middleware, and templates.
"""

from typing import Any, Dict, List, Optional

import pytest

from openeval.explainers import (
    AsyncExplainer,
    ChainStrategy,
    ClarityMetric,
    CompletenessMetric,
    ConcisennessMetric,
    ExplainerChain,
    ExplainerFactory,
    InMemoryCacheManager,
    LoggingMiddleware,
    MetricsRegistry,
    MiddlewareChain,
    NoOpCacheManager,
    PromptTemplateManager,
    ValidationMiddleware,
    get_explainer_factory,
    get_metrics_registry,
)
from openeval.explainers.base import CodeExplainer
from openeval.explainers.types import (
    CodeElement,
    CodeElementType,
    ExplainLevel,
    ExplanationResult,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_element() -> CodeElement:
    """Create a sample code element for testing."""
    return CodeElement(
        name="test_function",
        type=CodeElementType.FUNCTION,
        source_code="def test_function(x: int) -> int:\n    return x * 2",
        line_start=1,
        line_end=2,
    )


@pytest.fixture
def sample_result(sample_element: CodeElement) -> ExplanationResult:
    """Create a sample explanation result."""
    return ExplanationResult(
        element=sample_element,
        explanation="This function doubles the input parameter.",
        level=ExplainLevel.DETAILED,
        confidence=0.95,
        model_used="test-model",
    )


class MockExplainer(CodeExplainer):
    """Mock explainer for testing."""

    def __init__(
        self,
        explanation: str = "Mock explanation",
        confidence: float = 0.9,
        should_fail: bool = False,
    ):
        self.explanation = explanation
        self.confidence = confidence
        self.should_fail = should_fail
        self.explain_called = 0
        self.batch_explain_called = 0

    def explain(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.DETAILED,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExplanationResult:
        self.explain_called += 1
        if self.should_fail:
            raise RuntimeError("Mock explainer failed")
        return ExplanationResult(
            element=element,
            explanation=self.explanation,
            level=level,
            confidence=self.confidence,
            model_used="mock-model",
            analysis_metadata={"level": level.value},
        )

    def batch_explain(
        self,
        elements: List[CodeElement],
        level: ExplainLevel = ExplainLevel.DETAILED,
    ) -> List[ExplanationResult]:
        self.batch_explain_called += 1
        return [self.explain(e, level) for e in elements]


# ============================================================================
# Cache Manager Tests
# ============================================================================


def _make_result(element: CodeElement, explanation: str = "Test") -> ExplanationResult:
    """Helper to create ExplanationResult for cache tests."""
    return ExplanationResult(
        element=element,
        explanation=explanation,
        level=ExplainLevel.DETAILED,
        confidence=0.9,
    )


class TestInMemoryCacheManager:
    """Tests for InMemoryCacheManager."""

    @pytest.fixture
    def element(self) -> CodeElement:
        """Create element for cache tests."""
        return CodeElement(
            name="test",
            type=CodeElementType.FUNCTION,
            source_code="def f(): pass",
            line_start=1,
            line_end=1,
        )

    def test_get_set(self, element):
        """Test basic get/set operations."""
        cache = InMemoryCacheManager()
        result = _make_result(element, "value1")
        cache.set("key1", result)
        assert cache.get("key1") == result

    def test_get_missing_key(self):
        """Test get with missing key."""
        cache = InMemoryCacheManager()
        assert cache.get("nonexistent") is None

    def test_delete(self, element):
        """Test delete operation."""
        cache = InMemoryCacheManager()
        result = _make_result(element)
        cache.set("key1", result)
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self, element):
        """Test clear operation."""
        cache = InMemoryCacheManager()
        result1 = _make_result(element, "value1")
        result2 = _make_result(element, "value2")
        cache.set("key1", result1)
        cache.set("key2", result2)
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_exists(self, element):
        """Test exists check."""
        cache = InMemoryCacheManager()
        result = _make_result(element)
        cache.set("key1", result)
        assert cache.exists("key1") is True
        assert cache.exists("nonexistent") is False

    def test_stats(self, element):
        """Test cache statistics."""
        cache = InMemoryCacheManager()
        result = _make_result(element)
        cache.set("key1", result)
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1


class TestNoOpCacheManager:
    """Tests for NoOpCacheManager."""

    @pytest.fixture
    def element(self) -> CodeElement:
        """Create element for cache tests."""
        return CodeElement(
            name="test",
            type=CodeElementType.FUNCTION,
            source_code="def f(): pass",
            line_start=1,
            line_end=1,
        )

    def test_always_returns_none(self, element):
        """Test that get always returns None."""
        cache = NoOpCacheManager()
        result = _make_result(element)
        cache.set("key", result)
        assert cache.get("key") is None

    def test_exists_always_false(self, element):
        """Test that exists always returns False."""
        cache = NoOpCacheManager()
        result = _make_result(element)
        cache.set("key", result)
        assert cache.exists("key") is False


# ============================================================================
# Factory Tests
# ============================================================================


class TestExplainerFactory:
    """Tests for ExplainerFactory."""

    def test_create_llm_explainer(self):
        """Test creating LLM explainer."""
        factory = ExplainerFactory()
        explainer = factory.create_llm(
            adapter_name="openai",
            model="gpt-4",
            cache_enabled=False,
        )
        assert explainer.adapter_name == "openai"
        assert explainer.model == "gpt-4"

    def test_register_custom_explainer(self):
        """Test registering custom explainer."""
        factory = ExplainerFactory()
        factory.register("mock", MockExplainer)
        explainer = factory.create("mock")
        assert isinstance(explainer, MockExplainer)

    def test_register_invalid_type(self):
        """Test that registering non-explainer raises TypeError."""
        factory = ExplainerFactory()
        with pytest.raises(TypeError):
            factory.register("invalid", str)

    def test_create_unknown_type(self):
        """Test creating unknown explainer type raises ValueError."""
        factory = ExplainerFactory()
        with pytest.raises(ValueError):
            factory.create("unknown_type")

    def test_singleton_factory(self):
        """Test singleton factory access."""
        factory1 = get_explainer_factory()
        factory2 = get_explainer_factory()
        assert factory1 is factory2

    def test_create_chain(self):
        """Test creating explainer chain."""
        factory = ExplainerFactory()
        factory.register("mock", MockExplainer)
        chain = factory.create_chain(
            explainer_configs=[{"type": "mock"}],
            strategy=ChainStrategy.FIRST_SUCCESS,
        )
        assert isinstance(chain, ExplainerChain)


# ============================================================================
# Chain Tests
# ============================================================================


class TestExplainerChain:
    """Tests for ExplainerChain."""

    def test_first_success_strategy(self, sample_element):
        """Test first success strategy."""
        explainer1 = MockExplainer(explanation="First")
        explainer2 = MockExplainer(explanation="Second")

        chain = ExplainerChain(
            explainers=[explainer1, explainer2],
            strategy=ChainStrategy.FIRST_SUCCESS,
        )
        result = chain.explain(sample_element)
        assert result.explanation == "First"
        assert explainer1.explain_called == 1
        assert explainer2.explain_called == 0

    def test_fallback_on_failure(self, sample_element):
        """Test fallback when first explainer fails."""
        explainer1 = MockExplainer(should_fail=True)
        explainer2 = MockExplainer(explanation="Fallback")

        chain = ExplainerChain(
            explainers=[explainer1, explainer2],
            strategy=ChainStrategy.FIRST_SUCCESS,
            continue_on_error=True,
        )
        result = chain.explain(sample_element)
        assert result.explanation == "Fallback"

    def test_aggregate_strategy(self, sample_element):
        """Test aggregate strategy combines results."""
        explainer1 = MockExplainer(explanation="Part 1")
        explainer2 = MockExplainer(explanation="Part 2")

        chain = ExplainerChain(
            explainers=[explainer1, explainer2],
            strategy=ChainStrategy.AGGREGATE,
        )
        result = chain.explain(sample_element)
        assert "Part 1" in result.explanation
        assert "Part 2" in result.explanation

    def test_empty_chain_raises(self, sample_element):
        """Test that empty chain raises RuntimeError."""
        chain = ExplainerChain()
        with pytest.raises(RuntimeError):
            chain.explain(sample_element)

    def test_add_explainer_fluent(self):
        """Test fluent add_explainer API."""
        chain = ExplainerChain()
        result = chain.add_explainer(MockExplainer()).add_explainer(MockExplainer())
        assert result is chain
        assert len(chain.explainers) == 2

    def test_batch_explain(self, sample_element):
        """Test batch explain on chain."""
        chain = ExplainerChain([MockExplainer()])
        elements = [sample_element, sample_element]
        results = chain.batch_explain(elements)
        assert len(results) == 2


# ============================================================================
# Async Explainer Tests
# ============================================================================


class TestAsyncExplainer:
    """Tests for AsyncExplainer."""

    @pytest.mark.asyncio
    async def test_explain_async(self, sample_element):
        """Test async explain."""
        mock = MockExplainer()
        async_explainer = AsyncExplainer(mock)
        result = await async_explainer.explain_async(sample_element)
        assert result.explanation == "Mock explanation"

    @pytest.mark.asyncio
    async def test_batch_explain_async(self, sample_element):
        """Test batch async explain."""
        mock = MockExplainer()
        async_explainer = AsyncExplainer(mock)
        results = await async_explainer.batch_explain_async([sample_element, sample_element])
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_streaming(self, sample_element):
        """Test streaming explanation."""
        mock = MockExplainer(explanation="Hello World")
        async_explainer = AsyncExplainer(mock)
        chunks = []
        async for chunk in async_explainer.explain_streaming(sample_element, chunk_size=5):
            chunks.append(chunk)
        assert "".join(chunks) == "Hello World"


# ============================================================================
# Quality Metrics Tests
# ============================================================================


class TestQualityMetrics:
    """Tests for quality metrics."""

    def test_clarity_metric(self):
        """Test clarity metric evaluation."""
        metric = ClarityMetric()
        score = metric.evaluate(
            "This function takes a number. It doubles the value. Then returns the result.",
            "def f(x): return x * 2",
        )
        assert 0 <= score <= 1
        assert metric.get_name() == "Clarity"

    def test_completeness_metric(self):
        """Test completeness metric evaluation."""
        metric = CompletenessMetric()
        score = metric.evaluate(
            "This function takes parameters and returns a result. The algorithm is efficient.",
            "def f(x): return x * 2",
        )
        assert 0 <= score <= 1
        assert metric.get_name() == "Completeness"

    def test_conciseness_metric(self):
        """Test conciseness metric evaluation."""
        metric = ConcisennessMetric()
        score = metric.evaluate(
            "Doubles input.",
            "def f(x): return x * 2",
        )
        assert 0 <= score <= 1
        assert metric.get_name() == "Conciseness"

    def test_metrics_registry(self):
        """Test metrics registry."""
        registry = MetricsRegistry()
        # Builtins already registered, just test evaluate_all
        scores = registry.evaluate_all(
            "This is an explanation.",
            "def f(): pass",
        )
        assert "clarity" in scores
        assert "completeness" in scores

    def test_singleton_registry(self):
        """Test singleton registry."""
        reg1 = get_metrics_registry()
        reg2 = get_metrics_registry()
        assert reg1 is reg2


# ============================================================================
# Middleware Tests
# ============================================================================


class TestMiddleware:
    """Tests for middleware system."""

    def test_logging_middleware(self, sample_element):
        """Test logging middleware."""
        logs = []
        middleware = LoggingMiddleware(log_callback=logs.append)

        element, level, context = middleware.process_request(
            sample_element, ExplainLevel.DETAILED, {}
        )
        assert element is sample_element
        assert len(logs) == 1
        assert "ExplainerRequest" in logs[0]

    def test_validation_middleware(self, sample_result):
        """Test validation middleware."""
        middleware = ValidationMiddleware(min_length=10, max_length=1000)

        # Process request (no-op)
        middleware.process_request(sample_result.element, ExplainLevel.DETAILED, {})

        # Process response
        result = middleware.process_response(sample_result)
        assert result is sample_result

    def test_middleware_chain(self, sample_element, sample_result):
        """Test middleware chain."""
        logs = []

        chain = MiddlewareChain()
        chain.add(LoggingMiddleware(log_callback=logs.append))
        chain.add(ValidationMiddleware())

        # Process request through chain
        element, level, context = chain.process_request(sample_element, ExplainLevel.DETAILED, {})
        assert element is sample_element

        # Process response through chain
        result = chain.process_response(sample_result)
        assert result is sample_result
        assert len(logs) >= 1


# ============================================================================
# Prompt Templates Tests
# ============================================================================


class TestPromptTemplates:
    """Tests for prompt template system."""

    def test_template_manager_registration(self):
        """Test template manager registration."""
        manager = PromptTemplateManager()
        templates = manager.list_templates()
        assert "direct" in templates
        assert "chain_of_thought" in templates

    def test_get_template(self):
        """Test getting a template."""
        manager = PromptTemplateManager()
        template = manager.get("direct")
        assert template is not None

    def test_build_prompt(self, sample_element):
        """Test building prompt from template."""
        manager = PromptTemplateManager()
        prompt = manager.build_prompt(
            sample_element,
            ExplainLevel.DETAILED,
            {},
            template_name="direct",
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestExplainerIntegration:
    """Integration tests combining multiple components."""

    def test_factory_chain_middleware(self, sample_element):
        """Test factory creating chain with middleware."""
        factory = ExplainerFactory()
        factory.register("mock", MockExplainer)

        # Create chain
        chain = ExplainerChain(
            explainers=[MockExplainer(), MockExplainer()],
            strategy=ChainStrategy.FIRST_SUCCESS,
        )

        # Add middleware
        logs = []
        middleware_chain = MiddlewareChain()
        middleware_chain.add(LoggingMiddleware(log_callback=logs.append))

        # Process through middleware then explain
        element, level, context = middleware_chain.process_request(
            sample_element, ExplainLevel.DETAILED, {}
        )
        result = chain.explain(element, level, context)
        result = middleware_chain.process_response(result)

        assert result is not None
        assert len(logs) >= 1

    def test_cache_with_explainer(self, sample_element):
        """Test cache manager with explainer."""
        cache = InMemoryCacheManager()
        key = f"test:{sample_element.name}"

        # Cache miss
        assert cache.get(key) is None

        # Generate and cache
        explainer = MockExplainer()
        result = explainer.explain(sample_element)
        cache.set(key, result)  # Store full result object

        # Cache hit
        cached = cache.get(key)
        assert cached is not None
        assert cached.explanation == result.explanation

    def test_metrics_on_result(self, sample_result):
        """Test running metrics on explanation result."""
        registry = MetricsRegistry()  # Uses built-in metrics

        scores = registry.evaluate_all(
            sample_result.explanation,
            sample_result.element.source_code,
        )
        assert len(scores) >= 2
        assert all(0 <= s <= 1 for s in scores.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
