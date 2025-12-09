"""Adaptive model selection based on code complexity.

Intelligently selects models based on code characteristics.
"""

from typing import Optional

from .complexity_metrics import PythonComplexityAnalyzer
from .types import CodeElement


class ModelSelector:
    """Selects appropriate model based on code complexity."""

    # Model complexity mappings (lower cost to higher quality)
    MODELS_BY_TIER = {
        "basic": [
            "gpt-3.5-turbo",
            "claude-instant",
        ],
        "standard": [
            "gpt-4",
            "claude-opus",
        ],
        "advanced": [
            "gpt-4-turbo",
            "claude-opus-100k",
        ],
    }

    # Complexity thresholds (higher = more complex)
    COMPLEXITY_THRESHOLDS = {
        "low": 5,  # Very simple code
        "medium": 15,  # Moderate complexity
        "high": 30,  # Complex code
        "very_high": 50,  # Very complex
    }

    def __init__(self) -> None:
        """Initialize model selector."""
        self.analyzer = PythonComplexityAnalyzer()

    def select_model(
        self,
        element: CodeElement,
        preferred_tier: Optional[str] = None,
    ) -> str:
        """Select appropriate model based on code complexity.

        Args:
            element: Code element to analyze.
            preferred_tier: Preferred tier (basic, standard, advanced).

        Returns:
            Recommended model name.
        """
        # If tier specified, use it
        if preferred_tier and preferred_tier in self.MODELS_BY_TIER:
            models = self.MODELS_BY_TIER[preferred_tier]
            return models[0]  # Return first model in tier

        # Analyze complexity
        try:
            metrics = self.analyzer.calculate(element.source_code)
            complexity = metrics.cyclomatic_complexity
        except Exception:
            # Default to standard on error
            return "gpt-4"

        # Select tier based on complexity
        if complexity < self.COMPLEXITY_THRESHOLDS["low"]:
            tier = "basic"
        elif complexity < self.COMPLEXITY_THRESHOLDS["medium"]:
            tier = "basic"
        elif complexity < self.COMPLEXITY_THRESHOLDS["high"]:
            tier = "standard"
        elif complexity < self.COMPLEXITY_THRESHOLDS["very_high"]:
            tier = "advanced"
        else:
            tier = "advanced"

        return self.MODELS_BY_TIER[tier][0]

    def estimate_cost(self, element: CodeElement, model: Optional[str] = None) -> dict:
        """Estimate cost for explaining code with a model.

        Args:
            element: Code element.
            model: Model name (auto-selected if not provided).

        Returns:
            Dictionary with cost estimates.
        """
        if not model:
            model = self.select_model(element)

        # Simple estimation based on code size
        code_tokens = len(element.source_code.split()) * 1.3  # Rough estimate
        explanation_tokens = code_tokens * 2  # Assume 2x tokens for explanation

        # Rough pricing (USD per 1K tokens)
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "claude-instant": {"input": 0.0008, "output": 0.0024},
            "claude-opus": {"input": 0.015, "output": 0.075},
            "claude-opus-100k": {"input": 0.0015, "output": 0.0075},
        }

        model_pricing = pricing.get(model, pricing["gpt-4"])

        input_cost = (code_tokens / 1000) * model_pricing["input"]
        output_cost = (explanation_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "model": model,
            "input_tokens": code_tokens,
            "output_tokens": explanation_tokens,
            "total_tokens": code_tokens + explanation_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

    def get_cost_breakdown(self, element: CodeElement) -> dict:
        """Get cost breakdown for all tiers.

        Args:
            element: Code element.

        Returns:
            Dictionary with costs for each tier.
        """
        breakdown = {}
        for tier in ["basic", "standard", "advanced"]:
            model = self.MODELS_BY_TIER[tier][0]
            cost_info = self.estimate_cost(element, model)
            breakdown[tier] = cost_info

        return breakdown

    def get_complexity_analysis(self, element: CodeElement) -> dict:
        """Get detailed complexity analysis.

        Args:
            element: Code element.

        Returns:
            Dictionary with complexity metrics.
        """
        try:
            metrics = self.analyzer.calculate(element.source_code)
            return {
                "cyclomatic_complexity": metrics.cyclomatic_complexity,
                "lines_of_code": metrics.lines_of_code,
                "nesting_depth": metrics.nesting_depth,
                "comment_ratio": metrics.comment_ratio,
            }
        except Exception as e:
            return {"error": str(e)}
