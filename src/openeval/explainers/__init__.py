"""Package initialization for code explainers.

Provides main API for code explanation functionality.
"""

from .base import (
    CodeAnalyzer,
    CodeExplainer,
    ComplexityAnalyzer,
    ExplainerRegistry,
    ExplanationEvaluator,
    ExplanationFormatter,
    get_global_registry,
)
from .cache_manager import CacheManager, InMemoryCacheManager, NoOpCacheManager
from .chain import ChainStrategy, ExplainerChain
from .prompt_templates import (
    ChainOfThoughtPromptTemplate,
    DirectPromptTemplate,
    ExpertPromptTemplate,
    PromptStyle,
    PromptTemplate,
    PromptTemplateManager,
    SocraticPromptTemplate,
)
from .types import (
    AnalysisResult,
    CodeElement,
    CodeElementType,
    ComplexityMetrics,
    ExplainLevel,
    ExplanationResult,
)

__all__ = [
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
    # Explainer chaining
    "ExplainerChain",
    "ChainStrategy",
    # Prompt templates
    "PromptStyle",
    "PromptTemplate",
    "PromptTemplateManager",
    "DirectPromptTemplate",
    "ChainOfThoughtPromptTemplate",
    "SocraticPromptTemplate",
    "ExpertPromptTemplate",
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
