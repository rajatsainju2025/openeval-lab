"""Quality analyzer for evaluating explanation quality.

This module provides comprehensive quality analysis for code explanations,
including readability, completeness, accuracy, and coherence metrics.
"""

from __future__ import annotations

import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


from .types import CodeElement, ExplainLevel, ExplanationResult


class QualityDimension(Enum):
    """Dimensions of explanation quality."""

    CLARITY = auto()  # How clear and understandable
    COMPLETENESS = auto()  # How thorough
    CONCISENESS = auto()  # Not too verbose
    ACCURACY = auto()  # Factually correct
    COHERENCE = auto()  # Logical flow
    RELEVANCE = auto()  # On-topic
    TECHNICAL_DEPTH = auto()  # Appropriate detail level
    ACTIONABILITY = auto()  # Provides useful guidance


class QualityLevel(Enum):
    """Quality level assessment."""

    EXCELLENT = auto()
    GOOD = auto()
    ADEQUATE = auto()
    POOR = auto()
    UNACCEPTABLE = auto()


@dataclass
class QualityScore:
    """Score for a quality dimension."""

    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    level: QualityLevel
    feedback: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_score(
        cls,
        dimension: QualityDimension,
        score: float,
        feedback: str = "",
        details: dict[str, Any] | None = None,
    ) -> "QualityScore":
        """Create a QualityScore from a numeric score."""
        if score >= 0.9:
            level = QualityLevel.EXCELLENT
        elif score >= 0.7:
            level = QualityLevel.GOOD
        elif score >= 0.5:
            level = QualityLevel.ADEQUATE
        elif score >= 0.3:
            level = QualityLevel.POOR
        else:
            level = QualityLevel.UNACCEPTABLE

        return cls(
            dimension=dimension,
            score=score,
            level=level,
            feedback=feedback,
            details=details or {},
        )


@dataclass
class QualityReport:
    """Complete quality analysis report."""

    element: CodeElement
    explanation_level: ExplainLevel
    scores: list[QualityScore] = field(default_factory=list)
    overall_score: float = 0.0
    overall_level: QualityLevel = QualityLevel.ADEQUATE
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_score(self, dimension: QualityDimension) -> QualityScore | None:
        """Get score for a specific dimension."""
        for score in self.scores:
            if score.dimension == dimension:
                return score
        return None

    def compute_overall(self, weights: dict[QualityDimension, float] | None = None) -> None:
        """Compute overall score from dimension scores."""
        if not self.scores:
            return

        if weights:
            total_weight = sum(weights.get(s.dimension, 1.0) for s in self.scores)
            weighted_sum = sum(s.score * weights.get(s.dimension, 1.0) for s in self.scores)
            self.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            self.overall_score = statistics.mean(s.score for s in self.scores)

        # Determine overall level
        if self.overall_score >= 0.9:
            self.overall_level = QualityLevel.EXCELLENT
        elif self.overall_score >= 0.7:
            self.overall_level = QualityLevel.GOOD
        elif self.overall_score >= 0.5:
            self.overall_level = QualityLevel.ADEQUATE
        elif self.overall_score >= 0.3:
            self.overall_level = QualityLevel.POOR
        else:
            self.overall_level = QualityLevel.UNACCEPTABLE


class QualityMetric(ABC):
    """Abstract base class for quality metrics."""

    @property
    @abstractmethod
    def dimension(self) -> QualityDimension:
        """The dimension this metric measures."""
        ...

    @abstractmethod
    def evaluate(
        self,
        result: ExplanationResult,
    ) -> QualityScore:
        """Evaluate the quality metric."""
        ...


class ClarityMetric(QualityMetric):
    """Measures how clear and understandable an explanation is."""

    @property
    def dimension(self) -> QualityDimension:
        return QualityDimension.CLARITY

    def evaluate(self, result: ExplanationResult) -> QualityScore:
        """Evaluate clarity based on various indicators."""
        text = result.explanation
        score = 0.0
        details: dict[str, Any] = {}
        feedback_parts = []

        # Check sentence structure
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        details["sentence_count"] = len(sentences)

        if sentences:
            avg_sentence_length = statistics.mean(len(s.split()) for s in sentences)
            details["avg_sentence_length"] = avg_sentence_length

            # Optimal sentence length is 15-20 words
            if 10 <= avg_sentence_length <= 25:
                score += 0.3
            elif 5 <= avg_sentence_length <= 35:
                score += 0.2
            else:
                feedback_parts.append("Sentence length could be improved")
                score += 0.1

        # Check for jargon without explanation
        complex_terms = re.findall(
            r"\b(paradigm|abstraction|polymorphism|encapsulation|inheritance|recursion)\b",
            text.lower(),
        )
        if complex_terms:
            # Check if terms are explained
            has_explanations = any(
                f"({term}" in text.lower() or f"{term} is" in text.lower() for term in complex_terms
            )
            if has_explanations:
                score += 0.2
            else:
                feedback_parts.append("Technical terms could use brief explanations")
                score += 0.1
        else:
            score += 0.2

        # Check for structure (headers, bullets, etc.)
        has_structure = bool(re.search(r"^[-*•]|\n[-*•]|^\d+\.|^#+\s", text, re.MULTILINE))
        details["has_structure"] = has_structure
        if has_structure:
            score += 0.2
        elif len(text) > 500:
            feedback_parts.append("Long explanations benefit from bullet points or sections")
            score += 0.1
        else:
            score += 0.15

        # Check for code examples
        has_examples = "```" in text or "`" in text
        details["has_examples"] = has_examples
        if has_examples:
            score += 0.2
        else:
            feedback_parts.append("Code examples can improve clarity")
            score += 0.1

        # Normalize score
        score = min(1.0, score)

        feedback = "; ".join(feedback_parts) if feedback_parts else "Good clarity"

        return QualityScore.from_score(
            dimension=self.dimension,
            score=score,
            feedback=feedback,
            details=details,
        )


class CompletenessMetric(QualityMetric):
    """Measures how thorough an explanation is."""

    @property
    def dimension(self) -> QualityDimension:
        return QualityDimension.COMPLETENESS

    def evaluate(self, result: ExplanationResult) -> QualityScore:
        """Evaluate completeness based on coverage indicators."""
        text = result.explanation
        element = result.element
        score = 0.0
        details: dict[str, Any] = {}
        feedback_parts = []

        # Check if element name is mentioned
        if element.name.lower() in text.lower():
            score += 0.15
        else:
            feedback_parts.append(f"Should reference '{element.name}' directly")

        # Check for key components based on element type
        expected_sections = self._get_expected_sections(element.type.name)
        covered = 0
        for section in expected_sections:
            if section.lower() in text.lower():
                covered += 1

        if expected_sections:
            section_coverage = covered / len(expected_sections)
            score += section_coverage * 0.35
            details["section_coverage"] = section_coverage
            details["expected_sections"] = expected_sections
            details["covered_sections"] = covered
            if section_coverage < 0.5:
                feedback_parts.append(f"Consider covering: {', '.join(expected_sections[:3])}")

        # Check explanation length relative to code length
        code_lines = len(element.source_code.split("\n")) if element.source_code else 1
        explanation_words = len(text.split())
        words_per_line = explanation_words / max(1, code_lines)
        details["words_per_code_line"] = words_per_line

        # Target: 5-15 words of explanation per line of code
        if 5 <= words_per_line <= 20:
            score += 0.25
        elif 3 <= words_per_line <= 30:
            score += 0.15
        else:
            feedback_parts.append("Explanation length may not match code complexity")
            score += 0.05

        # Check for docstring mention if present
        if element.docstring:
            if "docstring" in text.lower() or "documentation" in text.lower():
                score += 0.15
            else:
                feedback_parts.append("Consider mentioning the existing docstring")
                score += 0.05
        else:
            score += 0.1

        # Normalize score
        score = min(1.0, score)

        feedback = "; ".join(feedback_parts) if feedback_parts else "Comprehensive coverage"

        return QualityScore.from_score(
            dimension=self.dimension,
            score=score,
            feedback=feedback,
            details=details,
        )

    def _get_expected_sections(self, element_type: str) -> list[str]:
        """Get expected sections based on element type."""
        sections_map = {
            "FUNCTION": ["parameter", "return", "purpose", "example"],
            "CLASS": ["attribute", "method", "purpose", "inheritance", "usage"],
            "MODULE": ["import", "export", "purpose", "structure"],
            "BLOCK": ["purpose", "flow", "variable"],
            "EXPRESSION": ["evaluate", "type", "result"],
            "CONTROL_FLOW": ["condition", "branch", "loop", "iteration"],
        }
        return sections_map.get(element_type, ["purpose"])


class ConcisenessMetric(QualityMetric):
    """Measures if explanation is appropriately concise."""

    @property
    def dimension(self) -> QualityDimension:
        return QualityDimension.CONCISENESS

    def evaluate(self, result: ExplanationResult) -> QualityScore:
        """Evaluate conciseness."""
        text = result.explanation
        score = 0.0
        details: dict[str, Any] = {}
        feedback_parts = []

        # Check for repetition
        words = text.lower().split()
        unique_words = set(words)
        if words:
            uniqueness_ratio = len(unique_words) / len(words)
            details["uniqueness_ratio"] = uniqueness_ratio

            if uniqueness_ratio > 0.5:
                score += 0.3
            elif uniqueness_ratio > 0.3:
                score += 0.2
                feedback_parts.append("Some repetition detected")
            else:
                score += 0.1
                feedback_parts.append("Significant repetition in explanation")

        # Check for filler phrases
        filler_patterns = [
            r"\b(basically|essentially|simply put|in other words|as you can see)\b",
            r"\b(it is worth noting|it should be noted|importantly)\b",
            r"\b(as mentioned|as stated|as discussed)\b",
        ]
        filler_count = sum(len(re.findall(p, text.lower())) for p in filler_patterns)
        details["filler_count"] = filler_count

        if filler_count == 0:
            score += 0.25
        elif filler_count <= 2:
            score += 0.15
            feedback_parts.append("Some filler phrases could be removed")
        else:
            score += 0.05
            feedback_parts.append("Too many filler phrases")

        # Check explanation density
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            # Count meaningful words vs total
            content_words = re.findall(r"\b\w{4,}\b", text.lower())
            density = len(content_words) / len(words) if words else 0
            details["content_density"] = density

            if density > 0.5:
                score += 0.25
            elif density > 0.3:
                score += 0.15
            else:
                feedback_parts.append("Explanation could be more information-dense")
                score += 0.1

        # Check for appropriate length based on explanation level
        level_length_map = {
            ExplainLevel.SUMMARY: (50, 200),
            ExplainLevel.DETAILED: (100, 500),
            ExplainLevel.EXPERT: (200, 1000),
        }
        min_words, max_words = level_length_map.get(result.level, (100, 500))
        word_count = len(words)
        details["word_count"] = word_count

        if min_words <= word_count <= max_words:
            score += 0.2
        elif word_count < min_words:
            feedback_parts.append("Explanation may be too brief for this level")
            score += 0.1
        else:
            feedback_parts.append("Explanation may be too verbose for this level")
            score += 0.1

        # Normalize
        score = min(1.0, score)

        feedback = "; ".join(feedback_parts) if feedback_parts else "Good conciseness"

        return QualityScore.from_score(
            dimension=self.dimension,
            score=score,
            feedback=feedback,
            details=details,
        )


class CoherenceMetric(QualityMetric):
    """Measures logical flow and coherence."""

    @property
    def dimension(self) -> QualityDimension:
        return QualityDimension.COHERENCE

    def evaluate(self, result: ExplanationResult) -> QualityScore:
        """Evaluate coherence and logical flow."""
        text = result.explanation
        score = 0.0
        details: dict[str, Any] = {}
        feedback_parts = []

        # Check for transition words
        transitions = [
            r"\b(first|second|third|finally|then|next|after|before)\b",
            r"\b(however|therefore|thus|consequently|because|since)\b",
            r"\b(additionally|moreover|furthermore|also|in addition)\b",
            r"\b(for example|for instance|such as|specifically)\b",
        ]
        transition_count = sum(len(re.findall(p, text.lower(), re.IGNORECASE)) for p in transitions)
        details["transition_count"] = transition_count

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        details["sentence_count"] = sentence_count

        if sentence_count > 0:
            transitions_per_sentence = transition_count / sentence_count
            if transitions_per_sentence >= 0.3:
                score += 0.3
            elif transitions_per_sentence >= 0.15:
                score += 0.2
            else:
                feedback_parts.append("More transition words would improve flow")
                score += 0.1

        # Check for logical structure indicators
        structure_patterns = [
            r"^(first|1\.|step 1)",
            r"(in summary|to summarize|in conclusion)",
            r"(the main|the key|the purpose)",
        ]
        has_structure = any(re.search(p, text.lower(), re.MULTILINE) for p in structure_patterns)
        details["has_logical_structure"] = has_structure

        if has_structure:
            score += 0.25
        elif sentence_count > 5:
            feedback_parts.append("Consider adding structural elements for longer explanations")
            score += 0.1
        else:
            score += 0.15

        # Check for topic consistency (keywords from first sentence appear later)
        if sentences:
            first_sentence_words = set(re.findall(r"\b\w{4,}\b", sentences[0].lower()))
            if len(sentences) > 1:
                rest_text = " ".join(sentences[1:]).lower()
                rest_words = set(re.findall(r"\b\w{4,}\b", rest_text))
                overlap = len(first_sentence_words & rest_words) / max(1, len(first_sentence_words))
                details["topic_consistency"] = overlap

                if overlap > 0.3:
                    score += 0.25
                elif overlap > 0.15:
                    score += 0.15
                else:
                    feedback_parts.append("Topic may drift from introduction")
                    score += 0.1
            else:
                score += 0.2

        # Normalize
        score = min(1.0, score)

        feedback = "; ".join(feedback_parts) if feedback_parts else "Good coherence"

        return QualityScore.from_score(
            dimension=self.dimension,
            score=score,
            feedback=feedback,
            details=details,
        )


class RelevanceMetric(QualityMetric):
    """Measures how relevant the explanation is to the code."""

    @property
    def dimension(self) -> QualityDimension:
        return QualityDimension.RELEVANCE

    def evaluate(self, result: ExplanationResult) -> QualityScore:
        """Evaluate relevance to the code element."""
        text = result.explanation
        element = result.element
        score = 0.0
        details: dict[str, Any] = {}
        feedback_parts = []

        # Extract identifiers from code
        code = element.source_code or ""
        code_identifiers = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code))
        code_identifiers = {
            i
            for i in code_identifiers
            if len(i) > 2
            and i
            not in {
                "def",
                "class",
                "return",
                "if",
                "else",
                "for",
                "while",
                "in",
                "and",
                "or",
                "not",
                "True",
                "False",
                "None",
                "self",
                "cls",
            }
        }
        details["code_identifiers"] = len(code_identifiers)

        # Check how many code identifiers are mentioned
        text_lower = text.lower()
        mentioned = sum(1 for i in code_identifiers if i.lower() in text_lower)
        if code_identifiers:
            mention_ratio = mentioned / len(code_identifiers)
            details["identifier_mention_ratio"] = mention_ratio

            if mention_ratio > 0.5:
                score += 0.35
            elif mention_ratio > 0.25:
                score += 0.25
            else:
                feedback_parts.append("Consider referencing more code identifiers")
                score += 0.15
        else:
            score += 0.25

        # Check if element name is prominently featured
        if element.name.lower() in text_lower[:200]:
            score += 0.2
        elif element.name.lower() in text_lower:
            score += 0.15
        else:
            feedback_parts.append(f"'{element.name}' should be mentioned early")
            score += 0.05

        # Check for off-topic content
        off_topic_phrases = [
            "i don't know",
            "i'm not sure",
            "as an ai",
            "i cannot",
            "disclaimer",
        ]
        has_off_topic = any(phrase in text_lower for phrase in off_topic_phrases)
        details["has_off_topic"] = has_off_topic

        if not has_off_topic:
            score += 0.25
        else:
            feedback_parts.append("Contains off-topic or meta content")
            score += 0.05

        # Normalize
        score = min(1.0, score)

        feedback = "; ".join(feedback_parts) if feedback_parts else "Highly relevant"

        return QualityScore.from_score(
            dimension=self.dimension,
            score=score,
            feedback=feedback,
            details=details,
        )


class QualityAnalyzer:
    """Main analyzer for explanation quality."""

    def __init__(
        self,
        metrics: list[QualityMetric] | None = None,
        weights: dict[QualityDimension, float] | None = None,
    ):
        """Initialize analyzer with metrics."""
        self.metrics = metrics or [
            ClarityMetric(),
            CompletenessMetric(),
            ConcisenessMetric(),
            CoherenceMetric(),
            RelevanceMetric(),
        ]
        self.weights = weights

    def add_metric(self, metric: QualityMetric) -> "QualityAnalyzer":
        """Add a custom metric."""
        self.metrics.append(metric)
        return self

    def set_weights(self, weights: dict[QualityDimension, float]) -> "QualityAnalyzer":
        """Set dimension weights for overall score."""
        self.weights = weights
        return self

    def analyze(self, result: ExplanationResult) -> QualityReport:
        """Analyze quality of an explanation result."""
        scores = [metric.evaluate(result) for metric in self.metrics]

        report = QualityReport(
            element=result.element,
            explanation_level=result.level,
            scores=scores,
        )

        report.compute_overall(self.weights)

        # Generate summary and recommendations
        report.summary = self._generate_summary(report)
        report.recommendations = self._generate_recommendations(report)

        return report

    def analyze_batch(self, results: list[ExplanationResult]) -> list[QualityReport]:
        """Analyze multiple results."""
        return [self.analyze(result) for result in results]

    def _generate_summary(self, report: QualityReport) -> str:
        """Generate a summary of the quality analysis."""
        level_descriptions = {
            QualityLevel.EXCELLENT: "excellent quality",
            QualityLevel.GOOD: "good quality",
            QualityLevel.ADEQUATE: "adequate quality",
            QualityLevel.POOR: "needs improvement",
            QualityLevel.UNACCEPTABLE: "significant issues",
        }

        best_score = max(report.scores, key=lambda s: s.score)
        worst_score = min(report.scores, key=lambda s: s.score)

        return (
            f"Overall {level_descriptions[report.overall_level]} "
            f"(score: {report.overall_score:.2f}). "
            f"Strongest: {best_score.dimension.name.lower()} ({best_score.score:.2f}). "
            f"Weakest: {worst_score.dimension.name.lower()} ({worst_score.score:.2f})."
        )

    def _generate_recommendations(self, report: QualityReport) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Sort scores by lowest first
        sorted_scores = sorted(report.scores, key=lambda s: s.score)

        for score in sorted_scores[:3]:  # Top 3 areas for improvement
            if score.score < 0.7 and score.feedback:
                recommendations.append(f"{score.dimension.name}: {score.feedback}")

        return recommendations


class QualityThreshold:
    """Define quality thresholds for acceptance."""

    def __init__(
        self,
        minimum_overall: float = 0.5,
        minimum_per_dimension: dict[QualityDimension, float] | None = None,
        required_level: QualityLevel = QualityLevel.ADEQUATE,
    ):
        """Initialize thresholds."""
        self.minimum_overall = minimum_overall
        self.minimum_per_dimension = minimum_per_dimension or {}
        self.required_level = required_level

    def check(self, report: QualityReport) -> tuple[bool, list[str]]:
        """Check if report meets thresholds.

        Returns (passes, list of failures)
        """
        failures = []

        # Check overall score
        if report.overall_score < self.minimum_overall:
            failures.append(
                f"Overall score {report.overall_score:.2f} below minimum {self.minimum_overall}"
            )

        # Check overall level
        if report.overall_level.value > self.required_level.value:
            failures.append(
                f"Quality level {report.overall_level.name} below required {self.required_level.name}"
            )

        # Check per-dimension minimums
        for dimension, minimum in self.minimum_per_dimension.items():
            score = report.get_score(dimension)
            if score and score.score < minimum:
                failures.append(f"{dimension.name} score {score.score:.2f} below minimum {minimum}")

        return len(failures) == 0, failures


class QualityGate:
    """Gate for enforcing quality standards."""

    def __init__(
        self,
        analyzer: QualityAnalyzer,
        threshold: QualityThreshold,
    ):
        """Initialize quality gate."""
        self.analyzer = analyzer
        self.threshold = threshold

    def evaluate(self, result: ExplanationResult) -> tuple[bool, QualityReport, list[str]]:
        """Evaluate if result passes quality gate."""
        report = self.analyzer.analyze(result)
        passes, failures = self.threshold.check(report)
        return passes, report, failures


@dataclass
class BatchQualityStats:
    """Statistics for batch quality analysis."""

    count: int = 0
    mean_overall: float = 0.0
    median_overall: float = 0.0
    std_overall: float = 0.0
    min_overall: float = 0.0
    max_overall: float = 0.0
    by_dimension: dict[QualityDimension, float] = field(default_factory=dict)
    pass_rate: float = 0.0
    level_distribution: dict[QualityLevel, int] = field(default_factory=dict)


class BatchQualityAnalyzer:
    """Analyze quality across multiple explanations."""

    def __init__(
        self,
        analyzer: QualityAnalyzer | None = None,
        threshold: QualityThreshold | None = None,
    ):
        """Initialize batch analyzer."""
        self.analyzer = analyzer or QualityAnalyzer()
        self.threshold = threshold or QualityThreshold()

    def analyze_batch(
        self, results: list[ExplanationResult]
    ) -> tuple[list[QualityReport], BatchQualityStats]:
        """Analyze a batch of results."""
        reports = self.analyzer.analyze_batch(results)
        stats = self._compute_stats(reports)
        return reports, stats

    def _compute_stats(self, reports: list[QualityReport]) -> BatchQualityStats:
        """Compute statistics for batch."""
        stats = BatchQualityStats()

        if not reports:
            return stats

        stats.count = len(reports)
        overall_scores = [r.overall_score for r in reports]

        stats.mean_overall = statistics.mean(overall_scores)
        stats.median_overall = statistics.median(overall_scores)
        stats.min_overall = min(overall_scores)
        stats.max_overall = max(overall_scores)

        if len(overall_scores) > 1:
            stats.std_overall = statistics.stdev(overall_scores)

        # Compute pass rate
        passes = sum(1 for r in reports if self.threshold.check(r)[0])
        stats.pass_rate = passes / len(reports)

        # Level distribution
        for report in reports:
            stats.level_distribution[report.overall_level] = (
                stats.level_distribution.get(report.overall_level, 0) + 1
            )

        # By dimension
        for dimension in QualityDimension:
            dim_scores = []
            for report in reports:
                score = report.get_score(dimension)
                if score:
                    dim_scores.append(score.score)
            if dim_scores:
                stats.by_dimension[dimension] = statistics.mean(dim_scores)

        return stats


# Convenience functions
def analyze_quality(result: ExplanationResult) -> QualityReport:
    """Analyze quality of a single explanation."""
    analyzer = QualityAnalyzer()
    return analyzer.analyze(result)


def check_quality(
    result: ExplanationResult,
    minimum_score: float = 0.5,
) -> tuple[bool, QualityReport]:
    """Check if explanation meets minimum quality."""
    report = analyze_quality(result)
    return report.overall_score >= minimum_score, report


def create_analyzer(
    dimensions: list[QualityDimension] | None = None,
) -> QualityAnalyzer:
    """Create an analyzer with specific dimensions."""
    metric_map = {
        QualityDimension.CLARITY: ClarityMetric,
        QualityDimension.COMPLETENESS: CompletenessMetric,
        QualityDimension.CONCISENESS: ConcisenessMetric,
        QualityDimension.COHERENCE: CoherenceMetric,
        QualityDimension.RELEVANCE: RelevanceMetric,
    }

    if dimensions:
        metrics = [metric_map[d]() for d in dimensions if d in metric_map]
    else:
        metrics = None

    return QualityAnalyzer(metrics=metrics)


def create_quality_gate(
    minimum_overall: float = 0.5,
    required_level: QualityLevel = QualityLevel.ADEQUATE,
) -> QualityGate:
    """Create a quality gate with specified thresholds."""
    analyzer = QualityAnalyzer()
    threshold = QualityThreshold(
        minimum_overall=minimum_overall,
        required_level=required_level,
    )
    return QualityGate(analyzer, threshold)


__all__ = [
    # Enums
    "QualityDimension",
    "QualityLevel",
    # Data classes
    "QualityScore",
    "QualityReport",
    "BatchQualityStats",
    # Metrics
    "QualityMetric",
    "ClarityMetric",
    "CompletenessMetric",
    "ConcisenessMetric",
    "CoherenceMetric",
    "RelevanceMetric",
    # Analyzers
    "QualityAnalyzer",
    "BatchQualityAnalyzer",
    # Thresholds and gates
    "QualityThreshold",
    "QualityGate",
    # Functions
    "analyze_quality",
    "check_quality",
    "create_analyzer",
    "create_quality_gate",
]
