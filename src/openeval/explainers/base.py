"""Abstract base classes for code explainer system.

This module provides the core interfaces that all explainers must implement.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .types import (
    AnalysisResult,
    CodeElement,
    ComplexityMetrics,
    ExplainLevel,
    ExplanationResult,
)

if TYPE_CHECKING:
    from .cache_manager import CacheManager


class CodeAnalyzer(ABC):
    """Abstract base class for code analysis.

    Subclasses should implement language-specific analysis strategies.
    """

    @abstractmethod
    def analyze(self, code: str) -> AnalysisResult:
        """Analyze code and extract structural information.

        Args:
            code: Source code to analyze.

        Returns:
            AnalysisResult containing extracted elements and metadata.

        Raises:
            ValueError: If code cannot be parsed.
            SyntaxError: If code has syntax errors.
        """
        pass

    @abstractmethod
    def extract_elements(self, code: str) -> List[CodeElement]:
        """Extract individual code elements (functions, classes, etc).

        Args:
            code: Source code to extract from.

        Returns:
            List of CodeElement objects.
        """
        pass

    @abstractmethod
    def get_dependencies(self, code: str) -> List[str]:
        """Extract code dependencies and imports.

        Args:
            code: Source code to analyze.

        Returns:
            List of dependency names.
        """
        pass


class CodeExplainer(ABC):
    """Abstract base class for code explanation.

    Subclasses implement different explanation strategies (LLM-based, rule-based, etc).
    Supports pluggable cache managers for flexible caching strategies.
    """

    @abstractmethod
    def explain(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.DETAILED,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExplanationResult:
        """Generate explanation for a code element.

        Args:
            element: Code element to explain.
            level: Detail level for explanation (summary, detailed, expert).
            context: Additional context (e.g., surrounding code, documentation).

        Returns:
            ExplanationResult with explanation text and metadata.

        Raises:
            ValueError: If element cannot be explained.
            TimeoutError: If explanation generation times out.
        """
        pass

    @abstractmethod
    def batch_explain(
        self,
        elements: List[CodeElement],
        level: ExplainLevel = ExplainLevel.DETAILED,
    ) -> List[ExplanationResult]:
        """Generate explanations for multiple code elements.

        Args:
            elements: List of code elements to explain.
            level: Detail level for all explanations.

        Returns:
            List of ExplanationResult objects in same order as input.
        """
        pass

    def reset_cache(self) -> None:
        """Reset any internal caches. Optional override."""
        pass

    def set_cache_manager(self, cache_manager: "CacheManager") -> None:
        """Set the cache manager for this explainer.

        Args:
            cache_manager: CacheManager instance to use.
        """
        pass

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics. Optional override.

        Returns:
            Dictionary with cache statistics.
        """
        return {}


class ComplexityAnalyzer(ABC):
    """Abstract base class for complexity analysis."""

    @abstractmethod
    def calculate(self, code: str) -> ComplexityMetrics:
        """Calculate code complexity metrics.

        Args:
            code: Source code to analyze.

        Returns:
            ComplexityMetrics object with computed values.
        """
        pass

    @abstractmethod
    def calculate_cyclomatic_complexity(self, code: str) -> float:
        """Calculate cyclomatic complexity.

        Args:
            code: Source code to analyze.

        Returns:
            Cyclomatic complexity score.
        """
        pass


class ExplanationFormatter(ABC):
    """Abstract base class for formatting explanations."""

    @abstractmethod
    def format(
        self,
        result: ExplanationResult,
        include_metadata: bool = False,
    ) -> str:
        """Format explanation for display.

        Args:
            result: ExplanationResult to format.
            include_metadata: Whether to include analysis metadata.

        Returns:
            Formatted string representation.
        """
        pass

    @abstractmethod
    def format_multiple(
        self,
        results: List[ExplanationResult],
        include_metadata: bool = False,
    ) -> str:
        """Format multiple explanations.

        Args:
            results: List of ExplanationResult objects.
            include_metadata: Whether to include metadata for each.

        Returns:
            Formatted string with all explanations.
        """
        pass


class ExplanationEvaluator(ABC):
    """Abstract base class for evaluating explanation quality."""

    @abstractmethod
    def evaluate(self, explanation: str, code: str) -> Dict[str, float]:
        """Evaluate quality of an explanation.

        Args:
            explanation: The explanation text.
            code: The original code being explained.

        Returns:
            Dictionary of metric names to scores (0.0 to 1.0).
        """
        pass

    @abstractmethod
    def batch_evaluate(
        self,
        explanations: List[str],
        codes: List[str],
    ) -> List[Dict[str, float]]:
        """Evaluate multiple explanations.

        Args:
            explanations: List of explanation texts.
            codes: List of corresponding code snippets.

        Returns:
            List of evaluation metric dictionaries.
        """
        pass


class ExplainerRegistry:
    """Registry for managing explainers and analyzers."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._explainers: Dict[str, type] = {}
        self._analyzers: Dict[str, type] = {}
        self._formatters: Dict[str, type] = {}

    def register_explainer(self, name: str, explainer_class: type) -> None:
        """Register a code explainer class.

        Args:
            name: Unique name for the explainer.
            explainer_class: Class implementing CodeExplainer.
        """
        if not issubclass(explainer_class, CodeExplainer):
            raise TypeError(f"{explainer_class} must inherit from CodeExplainer")
        self._explainers[name] = explainer_class

    def register_analyzer(self, name: str, analyzer_class: type) -> None:
        """Register a code analyzer class.

        Args:
            name: Unique name for the analyzer.
            analyzer_class: Class implementing CodeAnalyzer.
        """
        if not issubclass(analyzer_class, CodeAnalyzer):
            raise TypeError(f"{analyzer_class} must inherit from CodeAnalyzer")
        self._analyzers[name] = analyzer_class

    def register_formatter(self, name: str, formatter_class: type) -> None:
        """Register a formatter class.

        Args:
            name: Unique name for the formatter.
            formatter_class: Class implementing ExplanationFormatter.
        """
        if not issubclass(formatter_class, ExplanationFormatter):
            raise TypeError(f"{formatter_class} must inherit from ExplanationFormatter")
        self._formatters[name] = formatter_class

    def get_explainer(self, name: str) -> type:
        """Get registered explainer class by name.

        Args:
            name: Explainer name.

        Returns:
            Explainer class.

        Raises:
            KeyError: If explainer not found.
        """
        if name not in self._explainers:
            raise KeyError(
                f"Explainer '{name}' not found. Available: {list(self._explainers.keys())}"
            )
        return self._explainers[name]

    def get_analyzer(self, name: str) -> type:
        """Get registered analyzer class by name.

        Args:
            name: Analyzer name.

        Returns:
            Analyzer class.

        Raises:
            KeyError: If analyzer not found.
        """
        if name not in self._analyzers:
            raise KeyError(
                f"Analyzer '{name}' not found. Available: {list(self._analyzers.keys())}"
            )
        return self._analyzers[name]

    def get_formatter(self, name: str) -> type:
        """Get registered formatter class by name.

        Args:
            name: Formatter name.

        Returns:
            Formatter class.

        Raises:
            KeyError: If formatter not found.
        """
        if name not in self._formatters:
            raise KeyError(
                f"Formatter '{name}' not found. Available: {list(self._formatters.keys())}"
            )
        return self._formatters[name]

    def list_explainers(self) -> List[str]:
        """List all registered explainer names."""
        return list(self._explainers.keys())

    def list_analyzers(self) -> List[str]:
        """List all registered analyzer names."""
        return list(self._analyzers.keys())

    def list_formatters(self) -> List[str]:
        """List all registered formatter names."""
        return list(self._formatters.keys())


# Global registry instance
_global_registry = ExplainerRegistry()


def get_global_registry() -> ExplainerRegistry:
    """Get the global explainer registry.

    Returns:
        Global ExplainerRegistry instance.
    """
    return _global_registry
