"""Async/streaming support for explanations.

Enables non-blocking and streaming explanation generation.
"""

import asyncio
from typing import AsyncGenerator, List

from .base import CodeExplainer
from .types import CodeElement, ExplainLevel, ExplanationResult


class AsyncExplainer:
    """Wrapper adding async capabilities to CodeExplainer."""

    def __init__(self, explainer: CodeExplainer) -> None:
        """Initialize async wrapper.

        Args:
            explainer: CodeExplainer to wrap.
        """
        self.explainer = explainer

    async def explain_async(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.DETAILED,
        context=None,
    ) -> ExplanationResult:
        """Explain asynchronously.

        Args:
            element: Code element to explain.
            level: Explanation detail level.
            context: Additional context.

        Returns:
            ExplanationResult.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.explainer.explain, element, level, context)

    async def batch_explain_async(
        self,
        elements: List[CodeElement],
        level: ExplainLevel = ExplainLevel.DETAILED,
    ) -> List[ExplanationResult]:
        """Explain multiple elements asynchronously.

        Args:
            elements: List of code elements.
            level: Explanation detail level.

        Returns:
            List of ExplanationResults.
        """
        tasks = [self.explain_async(element, level) for element in elements]
        return await asyncio.gather(*tasks)

    async def explain_streaming(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.DETAILED,
        context=None,
        chunk_size: int = 100,
    ) -> AsyncGenerator[str, None]:
        """Stream explanation in chunks.

        Args:
            element: Code element to explain.
            level: Explanation detail level.
            context: Additional context.
            chunk_size: Characters per chunk.

        Yields:
            Explanation text chunks.
        """
        result = await self.explain_async(element, level, context)
        explanation = result.explanation

        # Yield explanation in chunks
        for i in range(0, len(explanation), chunk_size):
            yield explanation[i : i + chunk_size]
            # Simulate streaming delay
            await asyncio.sleep(0.01)
