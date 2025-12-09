"""Pluggable quality metrics system for explanations.

Enables custom quality metrics through plugin architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class QualityMetric(ABC):
    """Abstract base class for explanation quality metrics."""

    @abstractmethod
    def evaluate(self, explanation: str, code: str) -> float:
        """Evaluate quality metric (0-1).

        Args:
            explanation: Generated explanation text.
            code: Original code snippet.

        Returns:
            Score between 0 and 1.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get metric description."""
        pass


class ClarityMetric(QualityMetric):
    """Measures clarity of explanation."""

    def evaluate(self, explanation: str, code: str) -> float:
        """Evaluate clarity score."""
        # Simple heuristic: longer explanations with good structure
        words = len(explanation.split())
        sentences = explanation.count(".") + explanation.count("!") + explanation.count("?")

        if words == 0:
            return 0.0
        if sentences == 0:
            return 0.1

        avg_words_per_sentence = words / sentences
        # Good range: 10-20 words per sentence
        if 10 <= avg_words_per_sentence <= 20:
            return 1.0
        elif 5 <= avg_words_per_sentence <= 30:
            return 0.8
        else:
            return 0.5

    def get_name(self) -> str:
        """Get metric name."""
        return "Clarity"

    def get_description(self) -> str:
        """Get metric description."""
        return "Measures sentence structure and readability"


class CompletenessMetric(QualityMetric):
    """Measures completeness of explanation."""

    def evaluate(self, explanation: str, code: str) -> float:
        """Evaluate completeness score."""
        # Check for key sections
        keywords = [
            "function",
            "purpose",
            "returns",
            "parameters",
            "algorithm",
            "efficiency",
        ]
        matched = sum(1 for kw in keywords if kw.lower() in explanation.lower())

        return min(1.0, matched / len(keywords))

    def get_name(self) -> str:
        """Get metric name."""
        return "Completeness"

    def get_description(self) -> str:
        """Get metric description."""
        return "Checks for coverage of key topics"


class ConcisennessMetric(QualityMetric):
    """Measures conciseness without sacrificing clarity."""

    def evaluate(self, explanation: str, code: str) -> float:
        """Evaluate conciseness score."""
        code_words = len(code.split())
        explanation_words = len(explanation.split())

        if code_words == 0:
            return 0.5

        ratio = explanation_words / code_words
        # Good ratio: 2-5x the code length
        if 2 <= ratio <= 5:
            return 1.0
        elif 1 <= ratio <= 10:
            return 0.8
        else:
            return 0.5

    def get_name(self) -> str:
        """Get metric name."""
        return "Conciseness"

    def get_description(self) -> str:
        """Get metric description."""
        return "Measures efficiency of explanation relative to code"


class MetricsRegistry:
    """Registry for quality metrics with plugin support."""

    def __init__(self) -> None:
        """Initialize metrics registry."""
        self._metrics: Dict[str, QualityMetric] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in metrics."""
        self.register("clarity", ClarityMetric())
        self.register("completeness", CompletenessMetric())
        self.register("conciseness", ConcisennessMetric())

    def register(self, name: str, metric: QualityMetric) -> None:
        """Register a quality metric.

        Args:
            name: Unique metric name.
            metric: QualityMetric implementation.
        """
        self._metrics[name.lower()] = metric

    def get(self, name: str) -> Optional[QualityMetric]:
        """Get a metric by name.

        Args:
            name: Metric name.

        Returns:
            QualityMetric or None if not found.
        """
        return self._metrics.get(name.lower())

    def evaluate_all(self, explanation: str, code: str) -> Dict[str, float]:
        """Evaluate all registered metrics.

        Args:
            explanation: Generated explanation.
            code: Original code.

        Returns:
            Dictionary of metric names to scores.
        """
        results = {}
        for name, metric in self._metrics.items():
            try:
                results[name] = metric.evaluate(explanation, code)
            except Exception:
                # Log but continue with other metrics
                results[name] = -1.0  # Error indicator

        return results

    def evaluate(
        self, explanation: str, code: str, metric_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate specific metrics.

        Args:
            explanation: Generated explanation.
            code: Original code.
            metric_names: List of metrics to evaluate (all if None).

        Returns:
            Dictionary of metric names to scores.
        """
        if not metric_names:
            return self.evaluate_all(explanation, code)

        results = {}
        for name in metric_names:
            metric = self.get(name)
            if metric:
                results[name] = metric.evaluate(explanation, code)

        return results

    def get_overall_score(self, explanation: str, code: str) -> float:
        """Get overall quality score.

        Args:
            explanation: Generated explanation.
            code: Original code.

        Returns:
            Average score across all metrics.
        """
        scores = self.evaluate_all(explanation, code)
        valid_scores = [s for s in scores.values() if s >= 0]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    def list_metrics(self) -> Dict[str, Dict[str, str]]:
        """List all available metrics.

        Returns:
            Dictionary with metric names and info.
        """
        return {
            name: {
                "name": metric.get_name(),
                "description": metric.get_description(),
            }
            for name, metric in self._metrics.items()
        }

    def rate_quality(self, explanation: str, code: str) -> str:
        """Rate explanation quality as text.

        Args:
            explanation: Generated explanation.
            code: Original code.

        Returns:
            Quality rating (Poor/Fair/Good/Very Good/Excellent).
        """
        score = self.get_overall_score(explanation, code)
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.75:
            return "Very Good"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"


# Global registry instance
_global_metrics_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry instance.

    Returns:
        MetricsRegistry singleton.
    """
    return _global_metrics_registry
