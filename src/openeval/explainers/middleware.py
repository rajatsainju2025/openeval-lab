"""Middleware system for explainer pipelines.

Enables pre- and post-processing of explanations through composable middleware.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from .types import CodeElement, ExplainLevel, ExplanationResult


class ExplainerMiddleware(ABC):
    """Abstract base for explainer middleware."""

    @abstractmethod
    def process_request(
        self, element: CodeElement, level: ExplainLevel, context: Optional[Dict]
    ) -> tuple:
        """Pre-process explanation request.

        Args:
            element: Code element to explain.
            level: Explanation level.
            context: Additional context.

        Returns:
            Tuple of (element, level, context) potentially modified.
        """
        pass

    @abstractmethod
    def process_response(self, result: ExplanationResult) -> ExplanationResult:
        """Post-process explanation result.

        Args:
            result: Generated explanation result.

        Returns:
            Modified ExplanationResult.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get middleware name."""
        pass


class LoggingMiddleware(ExplainerMiddleware):
    """Logs explanation requests and responses."""

    def __init__(self, log_callback: Optional[Callable] = None) -> None:
        """Initialize logging middleware.

        Args:
            log_callback: Optional custom logging function.
        """
        self.log_callback = log_callback or print

    def process_request(
        self, element: CodeElement, level: ExplainLevel, context: Optional[Dict]
    ) -> tuple:
        """Log request details."""
        self.log_callback(
            f"[ExplainerRequest] element={element.name}, "
            f"type={element.type.value}, level={level.value}"
        )
        return element, level, context

    def process_response(self, result: ExplanationResult) -> ExplanationResult:
        """Log response details."""
        self.log_callback(
            f"[ExplainerResponse] element={result.element.name}, "
            f"confidence={result.confidence:.2f}, "
            f"explanation_length={len(result.explanation)}"
        )
        return result

    def get_name(self) -> str:
        """Get middleware name."""
        return "Logging"


class ValidationMiddleware(ExplainerMiddleware):
    """Validates explanations meet quality criteria."""

    def __init__(self, min_length: int = 50, max_length: int = 5000) -> None:
        """Initialize validation middleware.

        Args:
            min_length: Minimum explanation length.
            max_length: Maximum explanation length.
        """
        self.min_length = min_length
        self.max_length = max_length

    def process_request(
        self, element: CodeElement, level: ExplainLevel, context: Optional[Dict]
    ) -> tuple:
        """No pre-processing."""
        return element, level, context

    def process_response(self, result: ExplanationResult) -> ExplanationResult:
        """Validate explanation."""
        exp_len = len(result.explanation)

        if exp_len < self.min_length:
            result.analysis_metadata["validation_warning"] = (
                f"Explanation too short ({exp_len} < {self.min_length})"
            )

        if exp_len > self.max_length:
            result.analysis_metadata["validation_warning"] = (
                f"Explanation too long ({exp_len} > {self.max_length})"
            )

        return result

    def get_name(self) -> str:
        """Get middleware name."""
        return "Validation"


class EnrichmentMiddleware(ExplainerMiddleware):
    """Enriches explanations with additional context."""

    def process_request(
        self, element: CodeElement, level: ExplainLevel, context: Optional[Dict]
    ) -> tuple:
        """No pre-processing."""
        return element, level, context

    def process_response(self, result: ExplanationResult) -> ExplanationResult:
        """Add enrichment metadata."""
        result.analysis_metadata["enriched"] = True
        result.analysis_metadata["character_count"] = len(result.explanation)
        result.analysis_metadata["word_count"] = len(result.explanation.split())
        result.analysis_metadata["sentence_count"] = (
            result.explanation.count(".")
            + result.explanation.count("!")
            + result.explanation.count("?")
        )

        return result

    def get_name(self) -> str:
        """Get middleware name."""
        return "Enrichment"


class CachingMiddleware(ExplainerMiddleware):
    """Adds caching layer to explanations."""

    def __init__(self) -> None:
        """Initialize caching middleware."""
        self._cache: Dict[str, ExplanationResult] = {}

    def process_request(
        self, element: CodeElement, level: ExplainLevel, context: Optional[Dict]
    ) -> tuple:
        """No pre-processing."""
        return element, level, context

    def process_response(self, result: ExplanationResult) -> ExplanationResult:
        """Cache the result."""
        cache_key = f"{result.element.name}_{result.level.value}"
        self._cache[cache_key] = result
        return result

    def get_name(self) -> str:
        """Get middleware name."""
        return "Caching"

    def get_cached(self, element: CodeElement, level: ExplainLevel) -> Optional[ExplanationResult]:
        """Get cached explanation if available."""
        cache_key = f"{element.name}_{level.value}"
        return self._cache.get(cache_key)


class MiddlewareChain:
    """Chain of middleware for processing explanation pipelines."""

    def __init__(self, middleware: Optional[List[ExplainerMiddleware]] = None) -> None:
        """Initialize middleware chain.

        Args:
            middleware: List of ExplainerMiddleware to apply in order.
        """
        self.middleware = middleware or []

    def add(self, middleware: ExplainerMiddleware) -> "MiddlewareChain":
        """Add middleware to chain.

        Args:
            middleware: ExplainerMiddleware to add.

        Returns:
            Self for method chaining.
        """
        self.middleware.append(middleware)
        return self

    def process_request(
        self, element: CodeElement, level: ExplainLevel, context: Optional[Dict]
    ) -> tuple:
        """Process request through all middleware.

        Args:
            element: Code element.
            level: Explanation level.
            context: Additional context.

        Returns:
            Tuple of (element, level, context) after all middleware.
        """
        for mw in self.middleware:
            element, level, context = mw.process_request(element, level, context)

        return element, level, context

    def process_response(self, result: ExplanationResult) -> ExplanationResult:
        """Process response through all middleware.

        Args:
            result: Explanation result.

        Returns:
            ExplanationResult after all middleware.
        """
        for mw in self.middleware:
            result = mw.process_response(result)

        return result

    def get_middleware_info(self) -> List[Dict[str, str]]:
        """Get information about middleware in chain.

        Returns:
            List of dicts with middleware info.
        """
        return [{"name": mw.get_name(), "type": mw.__class__.__name__} for mw in self.middleware]
