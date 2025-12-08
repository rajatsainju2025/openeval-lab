"""Metrics for evaluating code explanation quality.

Scores explanations on clarity, correctness, and relevance.
"""

import re
from typing import Dict, List

from .base import ExplanationEvaluator


class ExplanationQualityEvaluator(ExplanationEvaluator):
    """Evaluate quality of code explanations."""

    def evaluate(self, explanation: str, code: str) -> Dict[str, float]:
        """Evaluate explanation quality.

        Args:
            explanation: The explanation text.
            code: The corresponding code being explained.

        Returns:
            Dictionary of metric names to scores (0.0 to 1.0).
        """
        scores = {
            "clarity": self._score_clarity(explanation),
            "completeness": self._score_completeness(explanation, code),
            "relevance": self._score_relevance(explanation, code),
            "conciseness": self._score_conciseness(explanation),
            "accuracy": self._score_accuracy(explanation, code),
        }

        return scores

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
        results = []
        for explanation, code in zip(explanations, codes):
            results.append(self.evaluate(explanation, code))

        return results

    def _score_clarity(self, explanation: str) -> float:
        """Score explanation clarity (0-1).

        Measures readability and understandability.

        Args:
            explanation: Explanation text.

        Returns:
            Clarity score.
        """
        clarity = 0.5

        # Check for clear structure
        if any(
            marker in explanation for marker in ["1.", "2.", "3.", "-", "•", "The code", "This"]
        ):
            clarity += 0.2

        # Check for simple language (avoid overly long words)
        words = explanation.split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        if avg_word_length < 6:
            clarity += 0.15
        elif avg_word_length > 8:
            clarity -= 0.1

        # Check for code references
        if "`" in explanation or "(" in explanation or ")" in explanation:
            clarity += 0.1

        return min(1.0, max(0.0, clarity))

    def _score_completeness(self, explanation: str, code: str) -> float:
        """Score explanation completeness (0-1).

        Measures whether key concepts are covered.

        Args:
            explanation: Explanation text.
            code: Code being explained.

        Returns:
            Completeness score.
        """
        completeness = 0.5

        # Check for coverage of key topics
        topics = {
            "what": ["does", "performs", "executes", "returns"],
            "how": ["by", "using", "with", "through", "algorithm"],
            "why": ["reason", "purpose", "benefit", "efficient"],
            "input/output": ["input", "output", "parameter", "argument", "return"],
        }

        covered_topics = 0
        for topic, keywords in topics.items():
            if any(kw in explanation.lower() for kw in keywords):
                covered_topics += 1

        completeness += (covered_topics / len(topics)) * 0.4

        # Check if explanation is substantive (not too short)
        sentences = len([s for s in explanation.split(".") if s.strip()])
        if sentences >= 3:
            completeness += 0.1

        return min(1.0, max(0.0, completeness))

    def _score_relevance(self, explanation: str, code: str) -> float:
        """Score explanation relevance (0-1).

        Measures alignment between explanation and code.

        Args:
            explanation: Explanation text.
            code: Code being explained.

        Returns:
            Relevance score.
        """
        relevance = 0.5

        # Extract common terms from code
        code_terms = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", code.lower()))

        # Check if explanation mentions code elements
        explanation_lower = explanation.lower()
        matched_terms = sum(1 for term in code_terms if term in explanation_lower)

        if code_terms:
            term_match_ratio = matched_terms / len(code_terms)
            relevance += term_match_ratio * 0.4

        # Check for direct code references
        if "def " in explanation or "class " in explanation:
            relevance += 0.1

        return min(1.0, max(0.0, relevance))

    def _score_conciseness(self, explanation: str) -> float:
        """Score explanation conciseness (0-1).

        Penalizes excessive verbosity.

        Args:
            explanation: Explanation text.

        Returns:
            Conciseness score.
        """
        word_count = len(explanation.split())

        # Ideal range: 50-300 words
        if 50 <= word_count <= 300:
            return 1.0
        elif 30 <= word_count < 50:
            return 0.8
        elif 300 < word_count <= 500:
            return 0.8
        elif 500 < word_count <= 800:
            return 0.6
        else:
            return 0.4

    def _score_accuracy(self, explanation: str, code: str) -> float:
        """Score explanation accuracy (0-1).

        Heuristic check for common mistakes.

        Args:
            explanation: Explanation text.
            code: Code being explained.

        Returns:
            Accuracy score (based on heuristics).
        """
        accuracy = 0.8  # Assume correct unless we detect issues

        # Check for contradictions with code structure
        if "does not" in explanation.lower() and "def " in code:
            accuracy -= 0.2

        # Check for nonsensical phrases
        bad_phrases = [
            "jibberish",
            "unclear",
            "undefined",
            "broken",
            "error",
        ]
        if any(phrase in explanation.lower() for phrase in bad_phrases):
            accuracy -= 0.1

        # Python-specific accuracy checks
        if "return" in code and "return" not in explanation.lower():
            accuracy -= 0.05

        return min(1.0, max(0.0, accuracy))

    def get_overall_score(self, explanation: str, code: str) -> float:
        """Get overall explanation quality score.

        Args:
            explanation: Explanation text.
            code: Code being explained.

        Returns:
            Overall score 0-1.
        """
        scores = self.evaluate(explanation, code)
        return sum(scores.values()) / len(scores) if scores else 0.0

    def rate_quality(self, explanation: str, code: str) -> str:
        """Rate explanation quality as string.

        Args:
            explanation: Explanation text.
            code: Code being explained.

        Returns:
            Quality rating: "Poor", "Fair", "Good", or "Excellent".
        """
        score = self.get_overall_score(explanation, code)

        if score >= 0.85:
            return "Excellent"
        elif score >= 0.70:
            return "Good"
        elif score >= 0.55:
            return "Fair"
        else:
            return "Poor"


class CodeClarityMetric:
    """Metric for code clarity/readability."""

    @staticmethod
    def score(code: str) -> float:
        """Score code clarity.

        Args:
            code: Code to score.

        Returns:
            Clarity score 0-1.
        """
        clarity = 0.5

        # Check for descriptive variable names
        long_names = len([w for w in re.findall(r"\b[a-z_][a-z0-9_]*\b", code) if len(w) > 3])
        short_names = len([w for w in re.findall(r"\b[a-z_][a-z0-9_]*\b", code) if len(w) <= 2])

        if long_names > short_names:
            clarity += 0.2

        # Check for comments
        comment_lines = len([line for line in code.split("\n") if "#" in line])
        if comment_lines > 0:
            clarity += min(0.2, comment_lines * 0.05)

        # Penalize deeply nested code
        max_indent = max(
            (len(line) - len(line.lstrip())) // 4 for line in code.split("\n") if line.strip()
        )
        if max_indent > 3:
            clarity -= 0.1

        return min(1.0, max(0.0, clarity))


class ExplanationCoverageMeasure:
    """Measure what topics are covered in an explanation."""

    TOPICS = {
        "purpose": [
            "purpose",
            "intended to",
            "designed to",
            "used to",
            "goal",
            "objective",
        ],
        "algorithm": ["algorithm", "approach", "method", "process", "step"],
        "inputs": [
            "input",
            "parameter",
            "argument",
            "accepts",
            "takes",
            "receives",
        ],
        "outputs": ["output", "returns", "result", "yields", "produces"],
        "complexity": [
            "complexity",
            "efficient",
            "performance",
            "time",
            "space",
            "fast",
        ],
        "edge_cases": [
            "edge case",
            "corner case",
            "special case",
            "handles",
            "error",
            "exception",
        ],
    }

    @staticmethod
    def get_coverage(explanation: str) -> Dict[str, bool]:
        """Get topic coverage in explanation.

        Args:
            explanation: Explanation text.

        Returns:
            Dictionary mapping topics to coverage boolean.
        """
        explanation_lower = explanation.lower()
        coverage = {}

        for topic, keywords in ExplanationCoverageMeasure.TOPICS.items():
            coverage[topic] = any(kw in explanation_lower for kw in keywords)

        return coverage

    @staticmethod
    def coverage_score(explanation: str) -> float:
        """Get coverage score 0-1.

        Args:
            explanation: Explanation text.

        Returns:
            Coverage score.
        """
        coverage = ExplanationCoverageMeasure.get_coverage(explanation)
        covered = sum(1 for v in coverage.values() if v)
        return covered / len(coverage) if coverage else 0.0
