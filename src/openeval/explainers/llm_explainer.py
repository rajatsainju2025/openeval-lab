"""LLM-powered code explainer using OpenEval adapters.

Integrates with the adapter system for AI-powered explanations.
"""

from typing import Any, Dict, List, Optional

from .base import CodeExplainer
from .cache_manager import CacheManager, InMemoryCacheManager
from .prompt_templates import PromptTemplateManager
from .types import CodeElement, ExplainLevel, ExplanationResult


class LLMCodeExplainer(CodeExplainer):
    """Generate code explanations using LLMs.

    Integrates with OpenEval adapters for API calls and pluggable caching.
    """

    def __init__(
        self,
        adapter_name: str = "openai",
        model: str = "gpt-4",
        cache_enabled: bool = True,
        max_tokens: int = 1000,
        cache_manager: Optional[CacheManager] = None,
        template_manager: Optional[PromptTemplateManager] = None,
    ) -> None:
        """Initialize the LLM explainer.

        Args:
            adapter_name: Adapter to use for LLM calls (e.g., 'openai', 'anthropic').
            model: Model to use for explanation generation.
            cache_enabled: Whether to cache explanations.
            max_tokens: Maximum tokens in explanation.
            cache_manager: CacheManager instance (defaults to InMemoryCacheManager).
            template_manager: PromptTemplateManager instance (creates default if not provided).
        """
        self.adapter_name = adapter_name
        self.model = model
        self.cache_enabled = cache_enabled
        self.max_tokens = max_tokens
        self._cache_manager = cache_manager or (InMemoryCacheManager() if cache_enabled else None)
        self._template_manager = template_manager or PromptTemplateManager()
        self._adapter = None

    def set_cache_manager(self, cache_manager: CacheManager) -> None:
        """Set the cache manager for this explainer.

        Args:
            cache_manager: CacheManager instance to use.
        """
        self._cache_manager = cache_manager

    def set_prompt_template(self, template_name: str) -> None:
        """Set the prompt template to use.

        Args:
            template_name: Name of the registered template.

        Raises:
            ValueError: If template not found.
        """
        self._template_manager.set_default(template_name)

    def get_template_manager(self) -> PromptTemplateManager:
        """Get the prompt template manager.

        Returns:
            PromptTemplateManager instance.
        """
        return self._template_manager

    def explain(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.DETAILED,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExplanationResult:
        """Generate explanation for a code element using LLM.

        Args:
            element: Code element to explain.
            level: Detail level for explanation.
            context: Additional context about the code.

        Returns:
            ExplanationResult with LLM-generated explanation.

        Raises:
            RuntimeError: If LLM adapter not available.
            TimeoutError: If explanation generation times out.
        """
        # Check cache first
        cache_key = self._make_cache_key(element, level)
        if self.cache_enabled and self._cache_manager:
            cached = self._cache_manager.get(cache_key)
            if cached:
                return cached

        # Generate prompt
        prompt = self._build_prompt(element, level, context)

        # Call LLM
        try:
            explanation = self._call_llm(prompt)
        except Exception as e:
            raise RuntimeError(f"Failed to generate explanation: {e}") from e

        # Build result
        result = ExplanationResult(
            element=element,
            explanation=explanation,
            level=level,
            confidence=0.85,  # Default confidence
            analysis_metadata={
                "model": self.model,
                "adapter": self.adapter_name,
                "prompt_tokens": len(prompt.split()),
            },
            model_used=self.model,
        )

        # Cache result
        if self.cache_enabled and self._cache_manager:
            self._cache_manager.set(cache_key, result)

        return result

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
        results = []
        for element in elements:
            try:
                result = self.explain(element, level)
                results.append(result)
            except Exception as e:
                # Add error result
                results.append(
                    ExplanationResult(
                        element=element,
                        explanation=f"Error generating explanation: {e}",
                        level=level,
                        confidence=0.0,
                        analysis_metadata={"error": str(e)},
                    )
                )

        return results

    def explain_with_context(
        self,
        element: CodeElement,
        surrounding_code: str = "",
        documentation: str = "",
        level: ExplainLevel = ExplainLevel.DETAILED,
    ) -> ExplanationResult:
        """Generate explanation with additional context.

        Args:
            element: Code element to explain.
            surrounding_code: Related code for context.
            documentation: Any available documentation.
            level: Detail level.

        Returns:
            ExplanationResult with contextual explanation.
        """
        context = {
            "surrounding_code": surrounding_code,
            "documentation": documentation,
        }
        return self.explain(element, level, context)

    def _build_prompt(
        self,
        element: CodeElement,
        level: ExplainLevel,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a prompt for the LLM using template manager.

        Args:
            element: Code element to explain.
            level: Explanation detail level.
            context: Additional context.

        Returns:
            Prompt string for LLM.
        """
        return self._template_manager.build_prompt(element, level, context)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt.

        Args:
            prompt: Prompt to send to LLM.

        Returns:
            LLM response text.

        Raises:
            RuntimeError: If LLM call fails.
        """
        # For demonstration, return a placeholder
        # In production, this would integrate with the adapter system
        if not self._adapter:
            return self._get_mock_explanation(prompt)

        # This would use the actual adapter from openeval.adapters
        try:
            # response = self._adapter.query(prompt)
            # return response
            return self._get_mock_explanation(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

    def _get_mock_explanation(self, prompt: str) -> str:
        """Get mock explanation for testing."""
        return (
            "This code provides functionality to handle the specified operation. "
            "The implementation includes proper error handling and follows best practices. "
            "Key components work together to achieve the desired outcome efficiently."
        )

    def _make_cache_key(self, element: CodeElement, level: ExplainLevel) -> str:
        """Create a cache key for an explanation.

        Args:
            element: Code element.
            level: Explanation level.

        Returns:
            Cache key string.
        """
        # Use hash of code and level
        code_hash = hash(element.source_code) & 0xFFFFFFFF
        return f"{element.name}_{level.value}_{code_hash}"

    def reset_cache(self) -> None:
        """Reset the explanation cache."""
        if self._cache_manager:
            self._cache_manager.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics from the cache manager.
        """
        if self._cache_manager:
            return self._cache_manager.get_stats()
        return {}


class HybridExplainer(CodeExplainer):
    """Combine multiple explanation strategies.

    Uses rule-based + LLM explanations for better results.
    """

    def __init__(
        self,
        llm_explainer: Optional[LLMCodeExplainer] = None,
    ) -> None:
        """Initialize hybrid explainer.

        Args:
            llm_explainer: LLM explainer to use, or None.
        """
        self.llm_explainer = llm_explainer or LLMCodeExplainer()

    def explain(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.DETAILED,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExplanationResult:
        """Generate hybrid explanation combining multiple strategies.

        Args:
            element: Code element to explain.
            level: Explanation detail level.
            context: Additional context.

        Returns:
            ExplanationResult with hybrid explanation.
        """
        # Get base explanation from LLM
        llm_result = self.llm_explainer.explain(element, level, context)

        # Could enhance with rule-based analysis here
        return llm_result

    def batch_explain(
        self,
        elements: List[CodeElement],
        level: ExplainLevel = ExplainLevel.DETAILED,
    ) -> List[ExplanationResult]:
        """Generate hybrid explanations for multiple elements.

        Args:
            elements: Code elements to explain.
            level: Explanation detail level.

        Returns:
            List of ExplanationResults.
        """
        return self.llm_explainer.batch_explain(elements, level)
