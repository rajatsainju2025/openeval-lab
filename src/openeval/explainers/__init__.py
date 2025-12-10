"""Package initialization for code explainers.

Provides main API for code explanation functionality.
"""

from .async_explainer import AsyncExplainer
from .base import (
    CodeAnalyzer,
    CodeExplainer,
    ComplexityAnalyzer,
    ExplainerRegistry,
    ExplanationEvaluator,
    ExplanationFormatter,
    get_global_registry,
)
from .batch_processor import (
    BatchItem,
    BatchProcessor,
    BatchResult,
    BatchStatus,
    ConcurrencyLimiter,
    ExplainerBatchProcessor,
    run_batch_async,
)
from .cache_backends import MemcachedCacheManager, RedisCacheManager
from .cache_manager import CacheManager, InMemoryCacheManager, NoOpCacheManager
from .chain import ChainStrategy, ExplainerChain
from .cli_commands import ExplainerCLI, get_explainer_cli
from .diff_explainer import (
    ChangeSignificance,
    DiffLine,
    DiffSection,
    DiffTracker,
    DiffType,
    ExplanationDiff,
    compare_explanation_texts,
    compare_explanations,
    get_diff_tracker,
)
from .event_hooks import (
    EventEmitter,
    EventSubscriber,
    ExplainerEvent,
    ExplainerEventType,
    get_event_emitter,
    get_event_subscriber,
)
from .factory import ExplainerFactory, get_explainer_factory
from .health_check import (
    ComponentHealth,
    HealthChecker,
    HealthStatus,
    SystemHealth,
    create_cache_health_check,
    create_explainer_health_check,
    create_memory_health_check,
    get_health_checker,
)
from .metrics_plugin import (
    ClarityMetric,
    CompletenessMetric,
    ConcisennessMetric,
    MetricsRegistry,
    QualityMetric,
    get_metrics_registry,
)
from .middleware import (
    CachingMiddleware,
    EnrichmentMiddleware,
    ExplainerMiddleware,
    LoggingMiddleware,
    MiddlewareChain,
    ValidationMiddleware,
)
from .model_selector import ModelSelector
from .prompt_templates import (
    ChainOfThoughtPromptTemplate,
    DirectPromptTemplate,
    ExpertPromptTemplate,
    PromptStyle,
    PromptTemplate,
    PromptTemplateManager,
    SocraticPromptTemplate,
)
from .retry_policy import (
    RetryableExplainer,
    RetryConfig,
    RetryPolicy,
    RetryResult,
    RetryStrategy,
    get_retry_config,
    retry_decorator,
)
from .types import (
    AnalysisResult,
    CodeElement,
    CodeElementType,
    ComplexityMetrics,
    ExplainLevel,
    ExplanationResult,
)
from .versioning import (
    ExplanationVersion,
    VersionedExplainer,
    VersionTracker,
    get_version_tracker,
)
from .caching_decorators import (
    CacheEntry,
    CacheNamespace,
    CacheStats,
    ExplainerCache,
    LRUCache,
    async_cache,
    cache,
    conditional_cache,
    get_explainer_cache,
    memoize,
    reset_explainer_cache,
)

__all__ = [
    # Async support
    "AsyncExplainer",
    # Batch processing
    "BatchItem",
    "BatchProcessor",
    "BatchResult",
    "BatchStatus",
    "ConcurrencyLimiter",
    "ExplainerBatchProcessor",
    "run_batch_async",
    # Base classes
    "CodeAnalyzer",
    "CodeExplainer",
    "ComplexityAnalyzer",
    "ExplanationFormatter",
    "ExplanationEvaluator",
    "ExplainerRegistry",
    # Cache management
    "CacheManager",
    "InMemoryCacheManager",
    "NoOpCacheManager",
    "RedisCacheManager",
    "MemcachedCacheManager",
    # Diff system
    "ChangeSignificance",
    "DiffLine",
    "DiffSection",
    "DiffTracker",
    "DiffType",
    "ExplanationDiff",
    "compare_explanations",
    "compare_explanation_texts",
    "get_diff_tracker",
    # Explainer chaining
    "RedisCacheManager",
    "MemcachedCacheManager",
    # Explainer chaining
    "ExplainerChain",
    "ChainStrategy",
    # Event system
    "ExplainerEvent",
    "ExplainerEventType",
    "EventEmitter",
    "EventSubscriber",
    "get_event_emitter",
    "get_event_subscriber",
    # Factory
    "ExplainerFactory",
    "get_explainer_factory",
    # Health check
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    "create_explainer_health_check",
    "create_cache_health_check",
    "create_memory_health_check",
    "get_health_checker",
    # CLI
    "ExplainerCLI",
    "get_explainer_cli",
    # Quality metrics
    "QualityMetric",
    "ClarityMetric",
    "CompletenessMetric",
    "ConcisennessMetric",
    "MetricsRegistry",
    "get_metrics_registry",
    # Middleware
    "ExplainerMiddleware",
    "LoggingMiddleware",
    "ValidationMiddleware",
    "EnrichmentMiddleware",
    "CachingMiddleware",
    "MiddlewareChain",
    # Model selection
    "ModelSelector",
    # Prompt templates
    "PromptStyle",
    "PromptTemplate",
    "PromptTemplateManager",
    "DirectPromptTemplate",
    "ChainOfThoughtPromptTemplate",
    "SocraticPromptTemplate",
    "ExpertPromptTemplate",
    # Retry policy
    "RetryConfig",
    "RetryPolicy",
    "RetryResult",
    "RetryStrategy",
    "RetryableExplainer",
    "retry_decorator",
    "get_retry_config",
    # Versioning
    "ExplanationVersion",
    "VersionTracker",
    "VersionedExplainer",
    "get_version_tracker",
    # Caching decorators
    "CacheEntry",
    "CacheNamespace",
    "CacheStats",
    "ExplainerCache",
    "LRUCache",
    "async_cache",
    "cache",
    "conditional_cache",
    "get_explainer_cache",
    "memoize",
    "reset_explainer_cache",
    # Type definitions
    "CodeElement",
    "CodeElementType",
    "ExplainLevel",
    "ExplanationResult",
    "AnalysisResult",
    "ComplexityMetrics",
    # Registry
    "get_global_registry",
]

__doc__ = """OpenEval Code Explainers

Modular, extensible system for generating and evaluating code explanations.

Quick Start:
    from openeval.explainers import CodeExplainer, get_global_registry

    # Get a registered explainer
    registry = get_global_registry()
    explainer_class = registry.get_explainer('llm')
    explainer = explainer_class()

    # Or use directly if available
    from openeval.explainers.ast_analyzer import PythonASTAnalyzer
    analyzer = PythonASTAnalyzer()
    analysis = analyzer.analyze(code_string)

Core Interfaces:
    - CodeAnalyzer: Parse and analyze code structure
    - CodeExplainer: Generate explanations for code elements
    - ComplexityAnalyzer: Compute code metrics
    - ExplanationFormatter: Format explanations for display
    - ExplanationEvaluator: Score explanation quality

Data Types:
    - CodeElement: Represents a code element (function, class, etc)
    - ExplanationResult: Result of explaining a code element
    - AnalysisResult: Result of analyzing code structure
    - ComplexityMetrics: Computed code metrics

See: docs/explainers.md for architecture and examples
"""
