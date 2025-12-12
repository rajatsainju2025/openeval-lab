"""Deep comparison system for code explanations.

This module provides tools for comparing explanations across versions, models,
and configurations to understand differences and improvements.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import ExplanationResult


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class ComparisonType(str, Enum):
    """Types of comparisons."""

    SEMANTIC = "semantic"  # Meaning-based comparison
    STRUCTURAL = "structural"  # Structure-based comparison
    LEXICAL = "lexical"  # Word-level comparison
    TECHNICAL = "technical"  # Technical accuracy comparison
    COVERAGE = "coverage"  # Topic coverage comparison


class DifferenceType(str, Enum):
    """Types of differences found."""

    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"
    REORDERING = "reordering"
    STYLE_CHANGE = "style_change"
    DETAIL_CHANGE = "detail_change"


class SignificanceLevel(str, Enum):
    """Significance of differences."""

    CRITICAL = "critical"  # Major semantic difference
    MAJOR = "major"  # Significant change
    MINOR = "minor"  # Small change
    TRIVIAL = "trivial"  # Negligible change


class ComparisonMetric(str, Enum):
    """Metrics for comparison."""

    SIMILARITY_SCORE = "similarity_score"
    COVERAGE_OVERLAP = "coverage_overlap"
    TECHNICAL_AGREEMENT = "technical_agreement"
    STRUCTURAL_MATCH = "structural_match"
    LENGTH_RATIO = "length_ratio"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Difference:
    """A single difference between explanations."""

    id: str
    type: DifferenceType
    significance: SignificanceLevel
    location: str  # Where in the explanation
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    context: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "significance": self.significance.value,
            "location": self.location,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "context": self.context,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class ComparisonScores:
    """Scores from a comparison."""

    overall_similarity: float = 0.0
    semantic_similarity: float = 0.0
    structural_similarity: float = 0.0
    lexical_similarity: float = 0.0
    coverage_overlap: float = 0.0
    technical_agreement: float = 0.0
    custom_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def combined_score(self) -> float:
        """Get weighted combined score."""
        weights = {
            "semantic": 0.3,
            "structural": 0.2,
            "lexical": 0.2,
            "coverage": 0.15,
            "technical": 0.15,
        }
        return (
            self.semantic_similarity * weights["semantic"]
            + self.structural_similarity * weights["structural"]
            + self.lexical_similarity * weights["lexical"]
            + self.coverage_overlap * weights["coverage"]
            + self.technical_agreement * weights["technical"]
        )


@dataclass
class TopicAnalysis:
    """Analysis of topics covered."""

    shared_topics: Set[str] = field(default_factory=set)
    only_in_first: Set[str] = field(default_factory=set)
    only_in_second: Set[str] = field(default_factory=set)
    topic_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing two explanations."""

    id: str
    first_explanation_id: str
    second_explanation_id: str
    comparison_type: ComparisonType
    scores: ComparisonScores
    differences: List[Difference] = field(default_factory=list)
    topic_analysis: Optional[TopicAnalysis] = None
    summary: str = ""
    recommendation: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical_differences(self) -> bool:
        """Check if there are critical differences."""
        return any(d.significance == SignificanceLevel.CRITICAL for d in self.differences)

    @property
    def difference_count(self) -> Dict[str, int]:
        """Count differences by type."""
        counts: Dict[str, int] = {}
        for diff in self.differences:
            key = diff.type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "first_explanation_id": self.first_explanation_id,
            "second_explanation_id": self.second_explanation_id,
            "comparison_type": self.comparison_type.value,
            "scores": {
                "overall_similarity": self.scores.overall_similarity,
                "semantic_similarity": self.scores.semantic_similarity,
                "structural_similarity": self.scores.structural_similarity,
                "lexical_similarity": self.scores.lexical_similarity,
                "coverage_overlap": self.scores.coverage_overlap,
                "technical_agreement": self.scores.technical_agreement,
                "combined_score": self.scores.combined_score,
            },
            "differences": [d.to_dict() for d in self.differences],
            "difference_count": self.difference_count,
            "has_critical_differences": self.has_critical_differences,
            "summary": self.summary,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class BatchComparisonResult:
    """Result of batch comparison."""

    comparisons: List[ComparisonResult] = field(default_factory=list)
    average_similarity: float = 0.0
    agreement_rate: float = 0.0
    total_differences: int = 0
    critical_differences: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ComparatorConfig:
    """Configuration for the comparator."""

    comparison_types: List[ComparisonType] = field(
        default_factory=lambda: [
            ComparisonType.LEXICAL,
            ComparisonType.STRUCTURAL,
            ComparisonType.COVERAGE,
        ]
    )
    similarity_threshold: float = 0.8  # Below this, consider different
    critical_threshold: float = 0.5  # Below this, mark as critical
    extract_topics: bool = True
    include_context: bool = True
    max_differences: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Comparison Strategies
# =============================================================================


class ComparisonStrategy(ABC):
    """Abstract base class for comparison strategies."""

    @property
    @abstractmethod
    def comparison_type(self) -> ComparisonType:
        """Get the comparison type."""
        pass

    @abstractmethod
    def compare(
        self, first: str, second: str, config: ComparatorConfig
    ) -> Tuple[float, List[Difference]]:
        """Compare two explanation texts.

        Args:
            first: First explanation text.
            second: Second explanation text.
            config: Comparison configuration.

        Returns:
            Tuple of (similarity_score, list of differences).
        """
        pass


class LexicalComparisonStrategy(ComparisonStrategy):
    """Lexical (word-level) comparison strategy."""

    @property
    def comparison_type(self) -> ComparisonType:
        return ComparisonType.LEXICAL

    def compare(
        self, first: str, second: str, config: ComparatorConfig
    ) -> Tuple[float, List[Difference]]:
        """Compare texts lexically."""
        # Use sequence matcher for similarity
        matcher = SequenceMatcher(None, first, second)
        similarity = matcher.ratio()

        differences = []
        diff_id = 0

        # Find differences
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            diff_id += 1
            old_text = first[i1:i2] if tag in ("delete", "replace") else None
            new_text = second[j1:j2] if tag in ("insert", "replace") else None

            # Determine difference type
            if tag == "delete":
                diff_type = DifferenceType.DELETION
            elif tag == "insert":
                diff_type = DifferenceType.ADDITION
            else:
                diff_type = DifferenceType.MODIFICATION

            # Determine significance
            length = max(len(old_text or ""), len(new_text or ""))
            if length > 100:
                significance = SignificanceLevel.MAJOR
            elif length > 30:
                significance = SignificanceLevel.MINOR
            else:
                significance = SignificanceLevel.TRIVIAL

            diff = Difference(
                id=f"lex_{diff_id}",
                type=diff_type,
                significance=significance,
                location=f"chars {i1}-{i2}" if old_text else f"chars {j1}-{j2}",
                old_value=old_text[:100] if old_text and len(old_text) > 100 else old_text,
                new_value=new_text[:100] if new_text and len(new_text) > 100 else new_text,
            )
            differences.append(diff)

            if len(differences) >= config.max_differences:
                break

        return similarity, differences


class StructuralComparisonStrategy(ComparisonStrategy):
    """Structural comparison strategy based on sections/paragraphs."""

    @property
    def comparison_type(self) -> ComparisonType:
        return ComparisonType.STRUCTURAL

    def compare(
        self, first: str, second: str, config: ComparatorConfig
    ) -> Tuple[float, List[Difference]]:
        """Compare texts structurally."""
        # Split into sections
        first_sections = self._extract_sections(first)
        second_sections = self._extract_sections(second)

        differences = []
        matched = 0
        total = max(len(first_sections), len(second_sections))

        if total == 0:
            return 1.0, []

        # Match sections
        used_second = set()
        for i, first_sec in enumerate(first_sections):
            best_match = -1
            best_score = 0.0

            for j, second_sec in enumerate(second_sections):
                if j in used_second:
                    continue
                score = SequenceMatcher(None, first_sec, second_sec).ratio()
                if score > best_score:
                    best_score = score
                    best_match = j

            if best_match >= 0 and best_score >= 0.5:
                used_second.add(best_match)
                matched += best_score

                if best_score < 0.95:
                    # Section was modified
                    differences.append(
                        Difference(
                            id=f"struct_{len(differences)+1}",
                            type=DifferenceType.MODIFICATION,
                            significance=self._score_to_significance(best_score),
                            location=f"section {i+1}",
                            old_value=first_sec[:200] + ("..." if len(first_sec) > 200 else ""),
                            new_value=second_sections[best_match][:200]
                            + ("..." if len(second_sections[best_match]) > 200 else ""),
                        )
                    )
            else:
                # Section was deleted
                differences.append(
                    Difference(
                        id=f"struct_{len(differences)+1}",
                        type=DifferenceType.DELETION,
                        significance=SignificanceLevel.MAJOR,
                        location=f"section {i+1}",
                        old_value=first_sec[:200] + ("..." if len(first_sec) > 200 else ""),
                    )
                )

        # Find added sections
        for j in range(len(second_sections)):
            if j not in used_second:
                differences.append(
                    Difference(
                        id=f"struct_{len(differences)+1}",
                        type=DifferenceType.ADDITION,
                        significance=SignificanceLevel.MINOR,
                        location=f"new section {j+1}",
                        new_value=second_sections[j][:200]
                        + ("..." if len(second_sections[j]) > 200 else ""),
                    )
                )

        similarity = matched / total if total > 0 else 1.0
        return similarity, differences[: config.max_differences]

    def _extract_sections(self, text: str) -> List[str]:
        """Extract sections from text."""
        # Split by double newlines or headers
        sections = re.split(r"\n\s*\n|\n(?=[#*-]\s)", text)
        return [s.strip() for s in sections if s.strip()]

    def _score_to_significance(self, score: float) -> SignificanceLevel:
        """Convert similarity score to significance level."""
        if score < 0.5:
            return SignificanceLevel.CRITICAL
        elif score < 0.7:
            return SignificanceLevel.MAJOR
        elif score < 0.9:
            return SignificanceLevel.MINOR
        return SignificanceLevel.TRIVIAL


class CoverageComparisonStrategy(ComparisonStrategy):
    """Topic/concept coverage comparison strategy."""

    @property
    def comparison_type(self) -> ComparisonType:
        return ComparisonType.COVERAGE

    def compare(
        self, first: str, second: str, config: ComparatorConfig
    ) -> Tuple[float, List[Difference]]:
        """Compare topic coverage."""
        first_topics = self._extract_topics(first)
        second_topics = self._extract_topics(second)

        shared = first_topics & second_topics
        only_first = first_topics - second_topics
        only_second = second_topics - first_topics
        all_topics = first_topics | second_topics

        # Calculate Jaccard similarity
        similarity = len(shared) / len(all_topics) if all_topics else 1.0

        differences = []

        # Missing topics
        for topic in only_first:
            differences.append(
                Difference(
                    id=f"cov_{len(differences)+1}",
                    type=DifferenceType.DELETION,
                    significance=SignificanceLevel.MINOR,
                    location="topic coverage",
                    old_value=topic,
                    description=f"Topic '{topic}' not in second explanation",
                )
            )

        # Added topics
        for topic in only_second:
            differences.append(
                Difference(
                    id=f"cov_{len(differences)+1}",
                    type=DifferenceType.ADDITION,
                    significance=SignificanceLevel.MINOR,
                    location="topic coverage",
                    new_value=topic,
                    description=f"Topic '{topic}' only in second explanation",
                )
            )

        return similarity, differences[: config.max_differences]

    def _extract_topics(self, text: str) -> Set[str]:
        """Extract key topics from text."""
        # Technical keywords and patterns
        keywords = set()

        # Code-related terms
        code_terms = re.findall(
            r"\b(?:function|method|class|variable|parameter|return|loop|condition|exception|import|module)\b",
            text.lower(),
        )
        keywords.update(code_terms)

        # Technical terms (camelCase, snake_case)
        technical = re.findall(r"\b[a-z]+(?:_[a-z]+)+\b|\b[a-z]+(?:[A-Z][a-z]*)+\b", text)
        keywords.update(t.lower() for t in technical)

        # Quoted terms
        quoted = re.findall(r'["\'](\w+)["\']', text)
        keywords.update(q.lower() for q in quoted)

        # Backtick code
        backtick = re.findall(r"`(\w+)`", text)
        keywords.update(b.lower() for b in backtick)

        return keywords


class TechnicalComparisonStrategy(ComparisonStrategy):
    """Technical accuracy comparison strategy."""

    @property
    def comparison_type(self) -> ComparisonType:
        return ComparisonType.TECHNICAL

    def compare(
        self, first: str, second: str, config: ComparatorConfig
    ) -> Tuple[float, List[Difference]]:
        """Compare technical accuracy."""
        differences = []

        # Extract technical assertions
        first_assertions = self._extract_assertions(first)
        second_assertions = self._extract_assertions(second)

        # Check for contradictions
        contradictions = self._find_contradictions(first_assertions, second_assertions)

        for i, contradiction in enumerate(contradictions):
            differences.append(
                Difference(
                    id=f"tech_{i+1}",
                    type=DifferenceType.MODIFICATION,
                    significance=SignificanceLevel.CRITICAL,
                    location="technical claim",
                    old_value=contradiction["first"],
                    new_value=contradiction["second"],
                    description="Potential contradiction in technical claims",
                )
            )

        # Calculate agreement score
        if first_assertions or second_assertions:
            all_assertions = len(first_assertions) + len(second_assertions)
            contradiction_count = len(contradictions) * 2
            similarity = max(0.0, 1.0 - (contradiction_count / all_assertions))
        else:
            similarity = 1.0

        return similarity, differences[: config.max_differences]

    def _extract_assertions(self, text: str) -> List[str]:
        """Extract technical assertions from text."""
        assertions = []

        # Look for declarative statements about behavior
        patterns = [
            r"(?:returns?|produces?|generates?|creates?|outputs?)\s+[^.]+",
            r"(?:takes?|accepts?|receives?|expects?)\s+[^.]+",
            r"(?:is|are|was|were)\s+(?:a|an|the)?\s*[^.]+",
            r"(?:will|would|should|must|can|cannot)\s+[^.]+",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            assertions.extend(matches)

        return assertions

    def _find_contradictions(self, first: List[str], second: List[str]) -> List[Dict[str, str]]:
        """Find potential contradictions between assertion sets."""
        contradictions = []

        negation_pairs = [
            (r"\bnot\b", ""),
            (r"\bnever\b", r"\balways\b"),
            (r"\bno\b", r"\bsome\b"),
            (r"\bcannot\b", r"\bcan\b"),
        ]

        for first_assertion in first:
            for second_assertion in second:
                # Check if assertions are about the same thing
                first_lower = first_assertion.lower()
                second_lower = second_assertion.lower()

                # Simple similarity check
                common_words = set(first_lower.split()) & set(second_lower.split())
                if len(common_words) < 2:
                    continue

                # Check for negation patterns
                for neg1, neg2 in negation_pairs:
                    first_has_neg = bool(re.search(neg1, first_lower))
                    second_has_neg = bool(re.search(neg1, second_lower))

                    if first_has_neg != second_has_neg:
                        contradictions.append(
                            {"first": first_assertion, "second": second_assertion}
                        )
                        break

        return contradictions


# =============================================================================
# Main Comparator
# =============================================================================


class ExplanationComparator:
    """Compare explanations using multiple strategies."""

    def __init__(self, config: Optional[ComparatorConfig] = None):
        """Initialize comparator.

        Args:
            config: Optional comparator configuration.
        """
        self.config = config or ComparatorConfig()
        self._strategies: Dict[ComparisonType, ComparisonStrategy] = {
            ComparisonType.LEXICAL: LexicalComparisonStrategy(),
            ComparisonType.STRUCTURAL: StructuralComparisonStrategy(),
            ComparisonType.COVERAGE: CoverageComparisonStrategy(),
            ComparisonType.TECHNICAL: TechnicalComparisonStrategy(),
        }

    def compare(
        self,
        first: ExplanationResult,
        second: ExplanationResult,
        comparison_types: Optional[List[ComparisonType]] = None,
    ) -> ComparisonResult:
        """Compare two explanation results.

        Args:
            first: First explanation.
            second: Second explanation.
            comparison_types: Optional specific comparison types to use.

        Returns:
            ComparisonResult with detailed analysis.
        """
        types_to_use = comparison_types or self.config.comparison_types

        all_differences = []
        scores = ComparisonScores()

        # Run each comparison strategy
        for comp_type in types_to_use:
            if comp_type in self._strategies:
                strategy = self._strategies[comp_type]
                similarity, differences = strategy.compare(
                    first.explanation, second.explanation, self.config
                )

                # Store scores
                if comp_type == ComparisonType.LEXICAL:
                    scores.lexical_similarity = similarity
                elif comp_type == ComparisonType.STRUCTURAL:
                    scores.structural_similarity = similarity
                elif comp_type == ComparisonType.COVERAGE:
                    scores.coverage_overlap = similarity
                elif comp_type == ComparisonType.TECHNICAL:
                    scores.technical_agreement = similarity

                all_differences.extend(differences)

        # Calculate overall similarity
        scores.overall_similarity = scores.combined_score

        # Extract topics if configured
        topic_analysis = None
        if self.config.extract_topics:
            topic_analysis = self._analyze_topics(first.explanation, second.explanation)

        # Generate summary and recommendation
        summary = self._generate_summary(scores, all_differences)
        recommendation = self._generate_recommendation(scores, all_differences)

        return ComparisonResult(
            id=self._generate_comparison_id(first, second),
            first_explanation_id=self._get_explanation_id(first),
            second_explanation_id=self._get_explanation_id(second),
            comparison_type=types_to_use[0] if len(types_to_use) == 1 else ComparisonType.SEMANTIC,
            scores=scores,
            differences=all_differences[: self.config.max_differences],
            topic_analysis=topic_analysis,
            summary=summary,
            recommendation=recommendation,
            metadata={
                "first_element": first.element.name,
                "second_element": second.element.name,
                "first_model": first.model_used,
                "second_model": second.model_used,
            },
        )

    def compare_texts(
        self,
        first_text: str,
        second_text: str,
        comparison_types: Optional[List[ComparisonType]] = None,
    ) -> ComparisonResult:
        """Compare two explanation texts directly.

        Args:
            first_text: First explanation text.
            second_text: Second explanation text.
            comparison_types: Optional specific comparison types to use.

        Returns:
            ComparisonResult with detailed analysis.
        """
        types_to_use = comparison_types or self.config.comparison_types

        all_differences = []
        scores = ComparisonScores()

        # Run each comparison strategy
        for comp_type in types_to_use:
            if comp_type in self._strategies:
                strategy = self._strategies[comp_type]
                similarity, differences = strategy.compare(first_text, second_text, self.config)

                if comp_type == ComparisonType.LEXICAL:
                    scores.lexical_similarity = similarity
                elif comp_type == ComparisonType.STRUCTURAL:
                    scores.structural_similarity = similarity
                elif comp_type == ComparisonType.COVERAGE:
                    scores.coverage_overlap = similarity
                elif comp_type == ComparisonType.TECHNICAL:
                    scores.technical_agreement = similarity

                all_differences.extend(differences)

        scores.overall_similarity = scores.combined_score

        topic_analysis = None
        if self.config.extract_topics:
            topic_analysis = self._analyze_topics(first_text, second_text)

        summary = self._generate_summary(scores, all_differences)
        recommendation = self._generate_recommendation(scores, all_differences)

        comp_id = hashlib.sha256(f"{first_text[:50]}:{second_text[:50]}".encode()).hexdigest()[:16]

        return ComparisonResult(
            id=comp_id,
            first_explanation_id="text_1",
            second_explanation_id="text_2",
            comparison_type=types_to_use[0] if len(types_to_use) == 1 else ComparisonType.SEMANTIC,
            scores=scores,
            differences=all_differences[: self.config.max_differences],
            topic_analysis=topic_analysis,
            summary=summary,
            recommendation=recommendation,
        )

    def compare_batch(
        self,
        pairs: List[Tuple[ExplanationResult, ExplanationResult]],
    ) -> BatchComparisonResult:
        """Compare multiple pairs of explanations.

        Args:
            pairs: List of explanation pairs to compare.

        Returns:
            BatchComparisonResult with aggregated results.
        """
        comparisons = []
        total_differences = 0
        critical_count = 0
        similarity_sum = 0.0

        for first, second in pairs:
            result = self.compare(first, second)
            comparisons.append(result)
            total_differences += len(result.differences)
            critical_count += sum(
                1 for d in result.differences if d.significance == SignificanceLevel.CRITICAL
            )
            similarity_sum += result.scores.overall_similarity

        avg_similarity = similarity_sum / len(pairs) if pairs else 0.0
        agreement_rate = (
            sum(
                1
                for c in comparisons
                if c.scores.overall_similarity >= self.config.similarity_threshold
            )
            / len(pairs)
            if pairs
            else 0.0
        )

        return BatchComparisonResult(
            comparisons=comparisons,
            average_similarity=avg_similarity,
            agreement_rate=agreement_rate,
            total_differences=total_differences,
            critical_differences=critical_count,
        )

    def add_strategy(self, comparison_type: ComparisonType, strategy: ComparisonStrategy) -> None:
        """Add or replace a comparison strategy.

        Args:
            comparison_type: Type of comparison.
            strategy: Strategy implementation.
        """
        self._strategies[comparison_type] = strategy

    def _analyze_topics(self, first: str, second: str) -> TopicAnalysis:
        """Analyze topics covered in both explanations."""
        strategy = CoverageComparisonStrategy()
        first_topics = strategy._extract_topics(first)
        second_topics = strategy._extract_topics(second)

        return TopicAnalysis(
            shared_topics=first_topics & second_topics,
            only_in_first=first_topics - second_topics,
            only_in_second=second_topics - first_topics,
        )

    def _generate_summary(self, scores: ComparisonScores, differences: List[Difference]) -> str:
        """Generate a summary of the comparison."""
        if scores.overall_similarity >= 0.95:
            base = "Explanations are nearly identical"
        elif scores.overall_similarity >= 0.8:
            base = "Explanations are similar with minor differences"
        elif scores.overall_similarity >= 0.6:
            base = "Explanations show moderate differences"
        else:
            base = "Explanations differ significantly"

        critical = sum(1 for d in differences if d.significance == SignificanceLevel.CRITICAL)
        if critical > 0:
            base += f" ({critical} critical difference{'s' if critical > 1 else ''})"

        return base

    def _generate_recommendation(
        self, scores: ComparisonScores, differences: List[Difference]
    ) -> str:
        """Generate a recommendation based on comparison."""
        if scores.overall_similarity >= 0.9:
            return "Both explanations are acceptable; prefer the more detailed one."

        critical = [d for d in differences if d.significance == SignificanceLevel.CRITICAL]
        if critical:
            return "Review critical differences for accuracy before using either explanation."

        if scores.technical_agreement < 0.7:
            return "Verify technical claims as explanations show disagreement."

        if scores.coverage_overlap < 0.7:
            return "Consider merging explanations to improve topic coverage."

        return "Both explanations are adequate; choose based on target audience."

    def _generate_comparison_id(self, first: ExplanationResult, second: ExplanationResult) -> str:
        """Generate a unique comparison ID."""
        data = f"{first.element.name}:{second.element.name}:{first.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _get_explanation_id(self, explanation: ExplanationResult) -> str:
        """Get or generate explanation ID."""
        if hasattr(explanation, "id"):
            return explanation.id
        data = f"{explanation.element.name}:{explanation.explanation[:50]}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# =============================================================================
# Model Comparison
# =============================================================================


class ModelComparator:
    """Compare explanations from different models."""

    def __init__(self, comparator: Optional[ExplanationComparator] = None):
        """Initialize model comparator.

        Args:
            comparator: Optional base comparator to use.
        """
        self.comparator = comparator or ExplanationComparator()
        self._model_results: Dict[str, List[ExplanationResult]] = {}

    def add_result(self, model: str, result: ExplanationResult) -> None:
        """Add a result for a model.

        Args:
            model: Model identifier.
            result: Explanation result.
        """
        if model not in self._model_results:
            self._model_results[model] = []
        self._model_results[model].append(result)

    def compare_models(self, model_a: str, model_b: str) -> List[ComparisonResult]:
        """Compare all results between two models.

        Args:
            model_a: First model identifier.
            model_b: Second model identifier.

        Returns:
            List of comparison results.
        """
        results_a = self._model_results.get(model_a, [])
        results_b = self._model_results.get(model_b, [])

        comparisons = []

        # Match by element name
        for result_a in results_a:
            for result_b in results_b:
                if result_a.element.name == result_b.element.name:
                    comparison = self.comparator.compare(result_a, result_b)
                    comparison.metadata["model_a"] = model_a
                    comparison.metadata["model_b"] = model_b
                    comparisons.append(comparison)
                    break

        return comparisons

    def get_model_agreement(self, model_a: str, model_b: str) -> float:
        """Get agreement rate between two models.

        Args:
            model_a: First model identifier.
            model_b: Second model identifier.

        Returns:
            Agreement rate (0.0 to 1.0).
        """
        comparisons = self.compare_models(model_a, model_b)
        if not comparisons:
            return 0.0

        high_agreement = sum(1 for c in comparisons if c.scores.overall_similarity >= 0.8)
        return high_agreement / len(comparisons)

    def rank_models(self) -> List[Dict[str, Any]]:
        """Rank models by consistency.

        Returns:
            List of models sorted by consistency score.
        """
        models = list(self._model_results.keys())
        scores: Dict[str, List[float]] = {m: [] for m in models}

        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                agreement = self.get_model_agreement(model_a, model_b)
                scores[model_a].append(agreement)
                scores[model_b].append(agreement)

        rankings = []
        for model in models:
            model_scores = scores[model]
            avg_score = sum(model_scores) / len(model_scores) if model_scores else 0.0
            rankings.append(
                {
                    "model": model,
                    "consistency_score": avg_score,
                    "comparison_count": len(model_scores),
                    "result_count": len(self._model_results[model]),
                }
            )

        rankings.sort(key=lambda x: x["consistency_score"], reverse=True)
        return rankings


# =============================================================================
# Version Comparison
# =============================================================================


class VersionComparator:
    """Compare explanations across versions."""

    def __init__(self, comparator: Optional[ExplanationComparator] = None):
        """Initialize version comparator.

        Args:
            comparator: Optional base comparator to use.
        """
        self.comparator = comparator or ExplanationComparator()
        self._versions: Dict[str, Dict[str, ExplanationResult]] = {}

    def add_version(self, version: str, element_name: str, result: ExplanationResult) -> None:
        """Add an explanation for a specific version.

        Args:
            version: Version identifier.
            element_name: Name of the code element.
            result: Explanation result.
        """
        if version not in self._versions:
            self._versions[version] = {}
        self._versions[version][element_name] = result

    def compare_versions(
        self, version_a: str, version_b: str, element_name: str
    ) -> Optional[ComparisonResult]:
        """Compare explanations for an element across versions.

        Args:
            version_a: First version.
            version_b: Second version.
            element_name: Element to compare.

        Returns:
            ComparisonResult or None if element not in both versions.
        """
        if version_a not in self._versions or version_b not in self._versions:
            return None

        result_a = self._versions[version_a].get(element_name)
        result_b = self._versions[version_b].get(element_name)

        if not result_a or not result_b:
            return None

        comparison = self.comparator.compare(result_a, result_b)
        comparison.metadata["version_a"] = version_a
        comparison.metadata["version_b"] = version_b
        return comparison

    def get_version_changes(self, version_a: str, version_b: str) -> List[ComparisonResult]:
        """Get all changes between two versions.

        Args:
            version_a: First version.
            version_b: Second version.

        Returns:
            List of comparison results.
        """
        if version_a not in self._versions or version_b not in self._versions:
            return []

        # Get common elements
        elements_a = set(self._versions[version_a].keys())
        elements_b = set(self._versions[version_b].keys())
        common = elements_a & elements_b

        comparisons = []
        for element in common:
            comparison = self.compare_versions(version_a, version_b, element)
            if comparison:
                comparisons.append(comparison)

        return comparisons


# =============================================================================
# Global Instance Management
# =============================================================================


_global_comparator: Optional[ExplanationComparator] = None


def get_comparator() -> ExplanationComparator:
    """Get the global comparator instance."""
    global _global_comparator
    if _global_comparator is None:
        _global_comparator = ExplanationComparator()
    return _global_comparator


def reset_comparator() -> None:
    """Reset the global comparator."""
    global _global_comparator
    _global_comparator = None


def compare_explanations(first: ExplanationResult, second: ExplanationResult) -> ComparisonResult:
    """Convenience function to compare two explanations."""
    return get_comparator().compare(first, second)


def compare_texts(first: str, second: str) -> ComparisonResult:
    """Convenience function to compare two explanation texts."""
    return get_comparator().compare_texts(first, second)


def create_comparator(config: Optional[ComparatorConfig] = None) -> ExplanationComparator:
    """Create a new comparator with optional config."""
    return ExplanationComparator(config=config)
