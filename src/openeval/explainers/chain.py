"""Explainer chaining system for composition and fallback.

Implements chain-of-responsibility pattern for composable explainers.
"""

from enum import Enum
from typing import List, Optional

from .base import CodeExplainer
from .types import CodeElement, ExplainLevel, ExplanationResult


class ChainStrategy(str, Enum):
    """Strategy for combining results from multiple explainers."""

    FIRST_SUCCESS = "first_success"  # Return first non-error result
    FALLBACK = "fallback"  # Same as first_success
    AGGREGATE = "aggregate"  # Combine all successful results
    VOTING = "voting"  # Weight-based voting (not yet implemented)


class ExplainerChain(CodeExplainer):
    """Chain multiple explainers for flexibility and robustness.

    Implements chain-of-responsibility pattern for:
    - Fallback behavior (use secondary explainer if primary fails)
    - Aggregate results from multiple strategies
    - Graceful degradation in case of errors
    """

    def __init__(
        self,
        explainers: Optional[List[CodeExplainer]] = None,
        strategy: ChainStrategy = ChainStrategy.FIRST_SUCCESS,
        continue_on_error: bool = True,
    ) -> None:
        """Initialize explainer chain.

        Args:
            explainers: List of explainers to chain (in order).
            strategy: How to combine results from multiple explainers.
            continue_on_error: Whether to try next explainer if one fails.
        """
        self.explainers = explainers or []
        self.strategy = strategy
        self.continue_on_error = continue_on_error

    def add_explainer(self, explainer: CodeExplainer) -> "ExplainerChain":
        """Add an explainer to the chain.

        Args:
            explainer: CodeExplainer to add.

        Returns:
            Self for method chaining.
        """
        self.explainers.append(explainer)
        return self

    def add_explainers(self, explainers: List[CodeExplainer]) -> "ExplainerChain":
        """Add multiple explainers to the chain.

        Args:
            explainers: List of CodeExplainers to add.

        Returns:
            Self for method chaining.
        """
        self.explainers.extend(explainers)
        return self

    def explain(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.DETAILED,
        context=None,
    ) -> ExplanationResult:
        """Explain using chained explainers based on strategy.

        Args:
            element: Code element to explain.
            level: Explanation detail level.
            context: Additional context.

        Returns:
            ExplanationResult from first successful explainer or best attempt.

        Raises:
            RuntimeError: If all explainers fail and continue_on_error is False.
        """
        if not self.explainers:
            raise RuntimeError("No explainers in chain")

        if self.strategy in (ChainStrategy.FIRST_SUCCESS, ChainStrategy.FALLBACK):
            return self._explain_first_success(element, level, context)
        elif self.strategy == ChainStrategy.AGGREGATE:
            return self._explain_aggregate(element, level, context)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _explain_first_success(
        self, element: CodeElement, level: ExplainLevel, context
    ) -> ExplanationResult:
        """Use first successful explainer (fallback strategy).

        Args:
            element: Code element to explain.
            level: Explanation detail level.
            context: Additional context.

        Returns:
            ExplanationResult from first successful explainer.

        Raises:
            RuntimeError: If all explainers fail and continue_on_error is False.
        """
        last_error = None

        for explainer in self.explainers:
            try:
                result = explainer.explain(element, level, context)
                return result
            except Exception as e:
                last_error = e
                if not self.continue_on_error:
                    raise
                # Continue to next explainer
                continue

        # All explainers failed
        if last_error:
            raise RuntimeError(
                f"All explainers in chain failed. Last error: {last_error}"
            ) from last_error
        else:
            raise RuntimeError("All explainers in chain failed")

    def _explain_aggregate(
        self, element: CodeElement, level: ExplainLevel, context
    ) -> ExplanationResult:
        """Aggregate results from all explainers.

        Args:
            element: Code element to explain.
            level: Explanation detail level.
            context: Additional context.

        Returns:
            ExplanationResult combining explanations from all explainers.
        """
        results = []
        errors = []

        for explainer in self.explainers:
            try:
                result = explainer.explain(element, level, context)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
                if not self.continue_on_error:
                    raise

        if not results:
            raise RuntimeError(f"All explainers failed in aggregate mode: {errors}")

        # Aggregate explanations
        combined_explanation = self._combine_explanations(results, errors)

        # Use first result as template, combine explanations
        base_result = results[0]
        return ExplanationResult(
            element=base_result.element,
            explanation=combined_explanation,
            level=base_result.level,
            confidence=self._calculate_aggregate_confidence(results),
            analysis_metadata={
                "chain_size": len(self.explainers),
                "successful_explainers": len(results),
                "failed_explainers": len(errors),
                "strategy": self.strategy.value,
                "explainer_types": [e.__class__.__name__ for e in self.explainers],
            },
        )

    def batch_explain(
        self,
        elements: List[CodeElement],
        level: ExplainLevel = ExplainLevel.DETAILED,
    ) -> List[ExplanationResult]:
        """Explain multiple elements using chain.

        Args:
            elements: List of code elements to explain.
            level: Explanation detail level.

        Returns:
            List of ExplanationResult objects.
        """
        results = []
        for element in elements:
            try:
                result = self.explain(element, level)
                results.append(result)
            except Exception as e:
                results.append(
                    ExplanationResult(
                        element=element,
                        explanation=f"Error in chain explanation: {e}",
                        level=level,
                        confidence=0.0,
                        analysis_metadata={"error": str(e)},
                    )
                )
        return results

    def reset_cache(self) -> None:
        """Reset cache for all explainers in chain."""
        for explainer in self.explainers:
            try:
                explainer.reset_cache()
            except Exception:
                # Ignore errors from individual explainers
                pass

    def get_chain_info(self) -> dict:
        """Get information about the chain configuration.

        Returns:
            Dictionary with chain metadata.
        """
        return {
            "size": len(self.explainers),
            "strategy": self.strategy.value,
            "continue_on_error": self.continue_on_error,
            "explainers": [
                {
                    "type": e.__class__.__name__,
                    "module": e.__class__.__module__,
                }
                for e in self.explainers
            ],
        }

    @staticmethod
    def _combine_explanations(results: List[ExplanationResult], errors: list) -> str:
        """Combine multiple explanations into one.

        Args:
            results: Successful explanation results.
            errors: Error messages from failed explainers.

        Returns:
            Combined explanation text.
        """
        parts = []

        # Add explanations from all successful explainers
        for i, result in enumerate(results, 1):
            explainer_type = result.analysis_metadata.get("model", result.__class__.__name__)
            parts.append(f"## Explanation {i} (via {explainer_type}):\n\n{result.explanation}\n")

        # Add error notes if any
        if errors:
            parts.append("\n## Notes on Failed Attempts:\n")
            for error in errors:
                parts.append(f"- {error}\n")

        return "".join(parts)

    @staticmethod
    def _calculate_aggregate_confidence(results: List[ExplanationResult]) -> float:
        """Calculate overall confidence from multiple results.

        Args:
            results: ExplanationResult objects.

        Returns:
            Average confidence score.
        """
        if not results:
            return 0.0
        total = sum(r.confidence for r in results)
        return total / len(results)
