"""Explanation quality optimizer for improving and refining explanations.

This module provides tools for optimizing explanations to improve clarity,
reduce token usage, enhance accuracy, and adapt to target audiences.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .types import ExplanationResult


# =============================================================================
# Enums and Type Definitions
# =============================================================================


class OptimizationType(str, Enum):
    """Types of optimization strategies."""

    CLARITY = "clarity"  # Improve readability
    CONCISENESS = "conciseness"  # Reduce length
    TECHNICAL_ACCURACY = "technical_accuracy"  # Improve precision
    COMPLETENESS = "completeness"  # Add missing info
    AUDIENCE_ADAPTATION = "audience_adaptation"  # Adapt to audience
    TOKEN_REDUCTION = "token_reduction"  # Minimize tokens
    STRUCTURE = "structure"  # Improve organization
    TERMINOLOGY = "terminology"  # Standardize terms


class AudienceLevel(str, Enum):
    """Target audience expertise levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class OptimizationPriority(str, Enum):
    """Priority levels for optimization."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class OptimizationSuggestion:
    """A suggestion for improving an explanation."""

    id: str
    type: OptimizationType
    priority: OptimizationPriority
    description: str
    original_text: Optional[str] = None
    suggested_text: Optional[str] = None
    location: Optional[str] = None
    reason: str = ""
    estimated_improvement: float = 0.0  # 0-1 scale
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "description": self.description,
            "original_text": self.original_text,
            "suggested_text": self.suggested_text,
            "location": self.location,
            "reason": self.reason,
            "estimated_improvement": self.estimated_improvement,
            "metadata": self.metadata,
        }


@dataclass
class OptimizationResult:
    """Result of optimizing an explanation."""

    original_text: str
    optimized_text: str
    suggestions_applied: List[OptimizationSuggestion] = field(default_factory=list)
    suggestions_skipped: List[OptimizationSuggestion] = field(default_factory=list)
    original_length: int = 0
    optimized_length: int = 0
    token_reduction: float = 0.0
    clarity_improvement: float = 0.0
    quality_score_before: float = 0.0
    quality_score_after: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length_reduction(self) -> float:
        """Calculate length reduction percentage."""
        if self.original_length == 0:
            return 0.0
        return 1 - (self.optimized_length / self.original_length)

    @property
    def overall_improvement(self) -> float:
        """Calculate overall improvement score."""
        return (
            self.clarity_improvement * 0.4
            + self.token_reduction * 0.3
            + (self.quality_score_after - self.quality_score_before) * 0.3
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_length": self.original_length,
            "optimized_length": self.optimized_length,
            "length_reduction": self.length_reduction,
            "token_reduction": self.token_reduction,
            "clarity_improvement": self.clarity_improvement,
            "quality_score_before": self.quality_score_before,
            "quality_score_after": self.quality_score_after,
            "overall_improvement": self.overall_improvement,
            "suggestions_applied": len(self.suggestions_applied),
            "suggestions_skipped": len(self.suggestions_skipped),
            "timestamp": self.timestamp,
        }


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""

    enabled_optimizations: Set[OptimizationType] = field(
        default_factory=lambda: {
            OptimizationType.CLARITY,
            OptimizationType.CONCISENESS,
            OptimizationType.STRUCTURE,
        }
    )
    target_audience: AudienceLevel = AudienceLevel.INTERMEDIATE
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    target_token_count: Optional[int] = None
    preserve_technical_terms: bool = True
    preserve_code_references: bool = True
    auto_apply_suggestions: bool = True
    min_suggestion_confidence: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Optimization Strategies
# =============================================================================


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""

    @property
    @abstractmethod
    def optimization_type(self) -> OptimizationType:
        """Get the optimization type."""
        pass

    @abstractmethod
    def analyze(self, text: str, config: OptimizerConfig) -> List[OptimizationSuggestion]:
        """Analyze text and generate suggestions.

        Args:
            text: Explanation text to analyze.
            config: Optimizer configuration.

        Returns:
            List of optimization suggestions.
        """
        pass

    @abstractmethod
    def apply(self, text: str, suggestion: OptimizationSuggestion) -> str:
        """Apply a suggestion to text.

        Args:
            text: Original text.
            suggestion: Suggestion to apply.

        Returns:
            Modified text.
        """
        pass


class ClarityOptimizer(OptimizationStrategy):
    """Optimizes for clarity and readability."""

    @property
    def optimization_type(self) -> OptimizationType:
        return OptimizationType.CLARITY

    def analyze(self, text: str, config: OptimizerConfig) -> List[OptimizationSuggestion]:
        """Analyze for clarity issues."""
        suggestions = []

        # Check for overly long sentences
        sentences = re.split(r"[.!?]+", text)
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            if len(words) > 35:
                suggestions.append(
                    OptimizationSuggestion(
                        id=f"clarity_{i}_long_sentence",
                        type=OptimizationType.CLARITY,
                        priority=OptimizationPriority.MEDIUM,
                        description="Consider breaking up long sentence",
                        original_text=sentence.strip()[:100] + "...",
                        location=f"sentence {i+1}",
                        reason=f"Sentence has {len(words)} words. Consider splitting for readability.",
                        estimated_improvement=0.3,
                    )
                )

        # Check for passive voice
        passive_patterns = [
            r"\b(is|are|was|were|been|being)\s+\w+ed\b",
            r"\b(has|have|had)\s+been\s+\w+ed\b",
        ]
        for pattern in passive_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                suggestions.append(
                    OptimizationSuggestion(
                        id=f"clarity_passive_{match.start()}",
                        type=OptimizationType.CLARITY,
                        priority=OptimizationPriority.LOW,
                        description="Consider using active voice",
                        original_text=match.group(),
                        location=f"position {match.start()}",
                        reason="Active voice is often clearer and more direct.",
                        estimated_improvement=0.1,
                    )
                )

        # Check for complex words
        complex_words = {
            "utilize": "use",
            "implement": "add/create",
            "facilitate": "help",
            "commence": "start",
            "terminate": "end",
            "subsequently": "then",
            "consequently": "so",
            "aforementioned": "mentioned above",
        }

        for complex_word, simple in complex_words.items():
            if complex_word in text.lower():
                suggestions.append(
                    OptimizationSuggestion(
                        id=f"clarity_word_{complex_word}",
                        type=OptimizationType.CLARITY,
                        priority=OptimizationPriority.LOW,
                        description=f"Consider simpler alternative for '{complex_word}'",
                        original_text=complex_word,
                        suggested_text=simple,
                        reason="Simpler words improve readability.",
                        estimated_improvement=0.1,
                    )
                )

        return suggestions

    def apply(self, text: str, suggestion: OptimizationSuggestion) -> str:
        """Apply clarity suggestion."""
        if suggestion.original_text and suggestion.suggested_text:
            return text.replace(suggestion.original_text, suggestion.suggested_text)
        return text


class ConcisenessOptimizer(OptimizationStrategy):
    """Optimizes for conciseness and brevity."""

    @property
    def optimization_type(self) -> OptimizationType:
        return OptimizationType.CONCISENESS

    def analyze(self, text: str, config: OptimizerConfig) -> List[OptimizationSuggestion]:
        """Analyze for verbosity."""
        suggestions = []

        # Redundant phrases
        redundant_phrases = {
            "in order to": "to",
            "due to the fact that": "because",
            "at this point in time": "now",
            "in the event that": "if",
            "for the purpose of": "to",
            "with regard to": "about",
            "in spite of the fact that": "although",
            "has the ability to": "can",
            "is able to": "can",
            "make a decision": "decide",
            "give consideration to": "consider",
            "is dependent on": "depends on",
            "in close proximity to": "near",
            "a large number of": "many",
            "a small number of": "few",
            "at the present time": "now",
            "in the near future": "soon",
        }

        for verbose, concise in redundant_phrases.items():
            if verbose.lower() in text.lower():
                suggestions.append(
                    OptimizationSuggestion(
                        id=f"concise_{verbose.replace(' ', '_')}",
                        type=OptimizationType.CONCISENESS,
                        priority=OptimizationPriority.MEDIUM,
                        description=f"Replace '{verbose}' with '{concise}'",
                        original_text=verbose,
                        suggested_text=concise,
                        reason="Shorter phrase conveys the same meaning.",
                        estimated_improvement=0.2,
                    )
                )

        # Filler words
        filler_patterns = [
            (r"\bvery\s+", ""),
            (r"\breally\s+", ""),
            (r"\bjust\s+", ""),
            (r"\bsimply\s+", ""),
            (r"\bbasically\s+", ""),
            (r"\bactually\s+", ""),
            (r"\bquite\s+", ""),
        ]

        for pattern, replacement in filler_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                suggestions.append(
                    OptimizationSuggestion(
                        id=f"concise_filler_{pattern[2:6]}",
                        type=OptimizationType.CONCISENESS,
                        priority=OptimizationPriority.LOW,
                        description=f"Consider removing filler word: {pattern[2:-3]}",
                        original_text=pattern[2:-3],
                        suggested_text="",
                        reason="Filler words add length without adding meaning.",
                        estimated_improvement=0.1,
                    )
                )

        return suggestions

    def apply(self, text: str, suggestion: OptimizationSuggestion) -> str:
        """Apply conciseness suggestion."""
        if suggestion.original_text is not None:
            pattern = re.compile(re.escape(suggestion.original_text), re.IGNORECASE)
            return pattern.sub(suggestion.suggested_text or "", text)
        return text


class StructureOptimizer(OptimizationStrategy):
    """Optimizes explanation structure and organization."""

    @property
    def optimization_type(self) -> OptimizationType:
        return OptimizationType.STRUCTURE

    def analyze(self, text: str, config: OptimizerConfig) -> List[OptimizationSuggestion]:
        """Analyze structure issues."""
        suggestions = []

        # Check for missing structure
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # Long explanation without headers
        if len(text) > 500 and not re.search(r"^#+\s", text, re.MULTILINE):
            suggestions.append(
                OptimizationSuggestion(
                    id="structure_no_headers",
                    type=OptimizationType.STRUCTURE,
                    priority=OptimizationPriority.MEDIUM,
                    description="Consider adding section headers",
                    reason="Headers improve navigation in longer explanations.",
                    estimated_improvement=0.3,
                )
            )

        # Long paragraph
        for i, para in enumerate(paragraphs):
            if len(para) > 400:
                suggestions.append(
                    OptimizationSuggestion(
                        id=f"structure_long_para_{i}",
                        type=OptimizationType.STRUCTURE,
                        priority=OptimizationPriority.MEDIUM,
                        description="Consider breaking up long paragraph",
                        original_text=para[:100] + "...",
                        location=f"paragraph {i+1}",
                        reason="Shorter paragraphs are easier to read.",
                        estimated_improvement=0.2,
                    )
                )

        # Check for bullet point opportunities
        if len(paragraphs) >= 3 and not re.search(r"^[-*•]\s", text, re.MULTILINE):
            # Check if paragraphs follow a list-like pattern
            list_indicators = ["first", "second", "third", "also", "additionally", "finally"]
            indicator_count = sum(
                1 for p in paragraphs for ind in list_indicators if ind in p.lower()
            )
            if indicator_count >= 2:
                suggestions.append(
                    OptimizationSuggestion(
                        id="structure_use_bullets",
                        type=OptimizationType.STRUCTURE,
                        priority=OptimizationPriority.MEDIUM,
                        description="Consider using bullet points for list-like content",
                        reason="Bullet points make lists easier to scan.",
                        estimated_improvement=0.3,
                    )
                )

        return suggestions

    def apply(self, text: str, suggestion: OptimizationSuggestion) -> str:
        """Apply structure suggestion."""
        # Structure changes are typically manual
        return text


class TerminologyOptimizer(OptimizationStrategy):
    """Optimizes technical terminology consistency."""

    @property
    def optimization_type(self) -> OptimizationType:
        return OptimizationType.TERMINOLOGY

    def analyze(self, text: str, config: OptimizerConfig) -> List[OptimizationSuggestion]:
        """Analyze terminology usage."""
        suggestions = []

        # Common term variations
        term_normalizations = {
            "func": "function",
            "arg": "argument",
            "param": "parameter",
            "var": "variable",
            "impl": "implementation",
            "init": "initialization",
            "config": "configuration",
            "auth": "authentication",
            "db": "database",
        }

        # Check for inconsistent terminology
        for abbrev, full in term_normalizations.items():
            abbrev_pattern = rf"\b{abbrev}\b"
            full_pattern = rf"\b{full}\b"

            has_abbrev = bool(re.search(abbrev_pattern, text, re.IGNORECASE))
            has_full = bool(re.search(full_pattern, text, re.IGNORECASE))

            if has_abbrev and has_full:
                suggestions.append(
                    OptimizationSuggestion(
                        id=f"term_inconsistent_{abbrev}",
                        type=OptimizationType.TERMINOLOGY,
                        priority=OptimizationPriority.LOW,
                        description=f"Inconsistent use of '{abbrev}' and '{full}'",
                        original_text=abbrev,
                        suggested_text=full,
                        reason="Consistent terminology improves clarity.",
                        estimated_improvement=0.1,
                    )
                )

        return suggestions

    def apply(self, text: str, suggestion: OptimizationSuggestion) -> str:
        """Apply terminology suggestion."""
        if suggestion.original_text and suggestion.suggested_text:
            pattern = re.compile(rf"\b{re.escape(suggestion.original_text)}\b", re.IGNORECASE)
            return pattern.sub(suggestion.suggested_text, text)
        return text


class AudienceOptimizer(OptimizationStrategy):
    """Adapts explanation to target audience."""

    @property
    def optimization_type(self) -> OptimizationType:
        return OptimizationType.AUDIENCE_ADAPTATION

    def analyze(self, text: str, config: OptimizerConfig) -> List[OptimizationSuggestion]:
        """Analyze for audience appropriateness."""
        suggestions = []

        # Technical jargon for beginners
        advanced_terms = {
            "polymorphism",
            "encapsulation",
            "abstraction",
            "inheritance",
            "recursion",
            "memoization",
            "idempotent",
            "immutable",
            "deterministic",
            "asynchronous",
            "callback",
            "closure",
            "decorator",
            "metaclass",
            "generator",
            "coroutine",
        }

        if config.target_audience == AudienceLevel.BEGINNER:
            for term in advanced_terms:
                if term.lower() in text.lower():
                    # Check if term is explained
                    explain_patterns = [
                        rf"{term}\s*(?:is|means|refers to)",
                        rf"(?:called|known as|termed)\s+{term}",
                    ]
                    is_explained = any(re.search(p, text, re.IGNORECASE) for p in explain_patterns)

                    if not is_explained:
                        suggestions.append(
                            OptimizationSuggestion(
                                id=f"audience_explain_{term}",
                                type=OptimizationType.AUDIENCE_ADAPTATION,
                                priority=OptimizationPriority.MEDIUM,
                                description=f"Consider explaining '{term}' for beginner audience",
                                original_text=term,
                                reason=f"'{term}' may be unfamiliar to beginners.",
                                estimated_improvement=0.3,
                            )
                        )

        # Oversimplification for experts
        elif config.target_audience == AudienceLevel.EXPERT:
            oversimple_phrases = [
                "in simple terms",
                "basically",
                "in other words",
                "to put it simply",
                "for example",
            ]

            phrase_count = sum(1 for p in oversimple_phrases if p in text.lower())
            if phrase_count >= 2:
                suggestions.append(
                    OptimizationSuggestion(
                        id="audience_too_simple",
                        type=OptimizationType.AUDIENCE_ADAPTATION,
                        priority=OptimizationPriority.LOW,
                        description="Explanation may be oversimplified for expert audience",
                        reason="Experts may prefer more technical depth.",
                        estimated_improvement=0.2,
                    )
                )

        return suggestions

    def apply(self, text: str, suggestion: OptimizationSuggestion) -> str:
        """Apply audience suggestion."""
        # Audience changes are typically manual
        return text


class TokenReductionOptimizer(OptimizationStrategy):
    """Optimizes for token count reduction."""

    @property
    def optimization_type(self) -> OptimizationType:
        return OptimizationType.TOKEN_REDUCTION

    def analyze(self, text: str, config: OptimizerConfig) -> List[OptimizationSuggestion]:
        """Analyze for token reduction opportunities."""
        suggestions = []

        # Estimate token count (rough approximation)
        estimated_tokens = len(text.split()) * 1.3  # Approximate tokens

        if config.target_token_count and estimated_tokens > config.target_token_count:
            reduction_needed = 1 - (config.target_token_count / estimated_tokens)
            suggestions.append(
                OptimizationSuggestion(
                    id="token_over_limit",
                    type=OptimizationType.TOKEN_REDUCTION,
                    priority=OptimizationPriority.HIGH,
                    description=f"Reduce content by ~{reduction_needed*100:.0f}% to meet token target",
                    reason=f"Current ~{estimated_tokens:.0f} tokens, target: {config.target_token_count}",
                    estimated_improvement=0.4,
                    metadata={"estimated_tokens": estimated_tokens},
                )
            )

        # Identify verbose sections
        sentences = re.split(r"[.!?]+", text)
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            if len(words) > 25:
                suggestions.append(
                    OptimizationSuggestion(
                        id=f"token_verbose_sent_{i}",
                        type=OptimizationType.TOKEN_REDUCTION,
                        priority=OptimizationPriority.MEDIUM,
                        description="Consider condensing verbose sentence",
                        original_text=sentence.strip()[:80] + "...",
                        location=f"sentence {i+1}",
                        reason=f"Sentence uses ~{len(words)} words and could be more concise.",
                        estimated_improvement=0.2,
                    )
                )

        return suggestions

    def apply(self, text: str, suggestion: OptimizationSuggestion) -> str:
        """Apply token reduction suggestion."""
        # Token reduction typically requires manual editing
        return text


# =============================================================================
# Main Optimizer
# =============================================================================


class ExplanationOptimizer:
    """Optimizes explanations using multiple strategies."""

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """Initialize optimizer.

        Args:
            config: Optional optimizer configuration.
        """
        self.config = config or OptimizerConfig()
        self._strategies: Dict[OptimizationType, OptimizationStrategy] = {
            OptimizationType.CLARITY: ClarityOptimizer(),
            OptimizationType.CONCISENESS: ConcisenessOptimizer(),
            OptimizationType.STRUCTURE: StructureOptimizer(),
            OptimizationType.TERMINOLOGY: TerminologyOptimizer(),
            OptimizationType.AUDIENCE_ADAPTATION: AudienceOptimizer(),
            OptimizationType.TOKEN_REDUCTION: TokenReductionOptimizer(),
        }

    def analyze(self, explanation: ExplanationResult) -> List[OptimizationSuggestion]:
        """Analyze an explanation and generate suggestions.

        Args:
            explanation: Explanation to analyze.

        Returns:
            List of optimization suggestions.
        """
        return self.analyze_text(explanation.explanation)

    def analyze_text(self, text: str) -> List[OptimizationSuggestion]:
        """Analyze text and generate suggestions.

        Args:
            text: Explanation text to analyze.

        Returns:
            List of optimization suggestions.
        """
        all_suggestions = []

        for opt_type in self.config.enabled_optimizations:
            if opt_type in self._strategies:
                strategy = self._strategies[opt_type]
                suggestions = strategy.analyze(text, self.config)
                all_suggestions.extend(suggestions)

        # Sort by priority
        priority_order = {
            OptimizationPriority.CRITICAL: 0,
            OptimizationPriority.HIGH: 1,
            OptimizationPriority.MEDIUM: 2,
            OptimizationPriority.LOW: 3,
        }
        all_suggestions.sort(key=lambda s: priority_order.get(s.priority, 4))

        return all_suggestions

    def optimize(
        self,
        explanation: ExplanationResult,
        suggestions: Optional[List[OptimizationSuggestion]] = None,
    ) -> OptimizationResult:
        """Optimize an explanation.

        Args:
            explanation: Explanation to optimize.
            suggestions: Optional pre-computed suggestions.

        Returns:
            OptimizationResult with improved text.
        """
        return self.optimize_text(explanation.explanation, suggestions=suggestions)

    def optimize_text(
        self,
        text: str,
        suggestions: Optional[List[OptimizationSuggestion]] = None,
    ) -> OptimizationResult:
        """Optimize explanation text.

        Args:
            text: Explanation text to optimize.
            suggestions: Optional pre-computed suggestions.

        Returns:
            OptimizationResult with improved text.
        """
        if suggestions is None:
            suggestions = self.analyze_text(text)

        original_text = text
        optimized_text = text
        applied = []
        skipped = []

        if self.config.auto_apply_suggestions:
            for suggestion in suggestions:
                if suggestion.estimated_improvement >= self.config.min_suggestion_confidence:
                    opt_type = suggestion.type
                    if opt_type in self._strategies:
                        strategy = self._strategies[opt_type]
                        new_text = strategy.apply(optimized_text, suggestion)
                        if new_text != optimized_text:
                            optimized_text = new_text
                            applied.append(suggestion)
                        else:
                            skipped.append(suggestion)
                    else:
                        skipped.append(suggestion)
                else:
                    skipped.append(suggestion)
        else:
            skipped = suggestions

        # Calculate metrics
        original_length = len(original_text)
        optimized_length = len(optimized_text)
        token_reduction = 1 - (len(optimized_text.split()) / max(1, len(original_text.split())))

        # Estimate clarity improvement
        clarity_improvement = self._estimate_clarity_improvement(original_text, optimized_text)

        return OptimizationResult(
            original_text=original_text,
            optimized_text=optimized_text,
            suggestions_applied=applied,
            suggestions_skipped=skipped,
            original_length=original_length,
            optimized_length=optimized_length,
            token_reduction=max(0, token_reduction),
            clarity_improvement=clarity_improvement,
        )

    def add_strategy(self, opt_type: OptimizationType, strategy: OptimizationStrategy) -> None:
        """Add or replace an optimization strategy.

        Args:
            opt_type: Type of optimization.
            strategy: Strategy implementation.
        """
        self._strategies[opt_type] = strategy

    def _estimate_clarity_improvement(self, original: str, optimized: str) -> float:
        """Estimate clarity improvement."""
        # Simple heuristics
        original_avg_sentence = self._avg_sentence_length(original)
        optimized_avg_sentence = self._avg_sentence_length(optimized)

        # Shorter sentences generally improve clarity
        sentence_improvement = 0.0
        if original_avg_sentence > 20 and optimized_avg_sentence < original_avg_sentence:
            sentence_improvement = min(
                0.3, (original_avg_sentence - optimized_avg_sentence) / original_avg_sentence
            )

        # Fewer complex words is better
        original_complex = self._count_complex_words(original)
        optimized_complex = self._count_complex_words(optimized)

        complexity_improvement = 0.0
        if original_complex > 0 and optimized_complex < original_complex:
            complexity_improvement = min(
                0.3, (original_complex - optimized_complex) / max(1, original_complex)
            )

        return min(1.0, sentence_improvement + complexity_improvement)

    def _avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        return sum(len(s.split()) for s in sentences) / len(sentences)

    def _count_complex_words(self, text: str) -> int:
        """Count complex words (3+ syllables)."""
        words = text.lower().split()
        return sum(1 for w in words if self._syllable_count(w) >= 3)

    def _syllable_count(self, word: str) -> int:
        """Estimate syllable count."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e") and count > 1:
            count -= 1

        return max(1, count)


# =============================================================================
# Batch Optimizer
# =============================================================================


class BatchOptimizer:
    """Optimizes multiple explanations."""

    def __init__(self, optimizer: Optional[ExplanationOptimizer] = None):
        """Initialize batch optimizer.

        Args:
            optimizer: Base optimizer to use.
        """
        self.optimizer = optimizer or ExplanationOptimizer()

    def optimize_batch(self, explanations: List[ExplanationResult]) -> List[OptimizationResult]:
        """Optimize multiple explanations.

        Args:
            explanations: List of explanations to optimize.

        Returns:
            List of optimization results.
        """
        return [self.optimizer.optimize(e) for e in explanations]

    def get_batch_summary(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """Get summary statistics for batch optimization.

        Args:
            results: List of optimization results.

        Returns:
            Summary statistics.
        """
        if not results:
            return {}

        total_original = sum(r.original_length for r in results)
        total_optimized = sum(r.optimized_length for r in results)
        total_suggestions = sum(
            len(r.suggestions_applied) + len(r.suggestions_skipped) for r in results
        )
        total_applied = sum(len(r.suggestions_applied) for r in results)

        return {
            "total_explanations": len(results),
            "total_suggestions": total_suggestions,
            "suggestions_applied": total_applied,
            "total_length_reduction": (
                1 - (total_optimized / total_original) if total_original > 0 else 0
            ),
            "avg_clarity_improvement": sum(r.clarity_improvement for r in results) / len(results),
            "avg_token_reduction": sum(r.token_reduction for r in results) / len(results),
        }


# =============================================================================
# Global Instance Management
# =============================================================================


_global_optimizer: Optional[ExplanationOptimizer] = None


def get_optimizer() -> ExplanationOptimizer:
    """Get the global optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = ExplanationOptimizer()
    return _global_optimizer


def reset_optimizer() -> None:
    """Reset the global optimizer."""
    global _global_optimizer
    _global_optimizer = None


def create_optimizer(config: Optional[OptimizerConfig] = None) -> ExplanationOptimizer:
    """Create a new optimizer with optional config."""
    return ExplanationOptimizer(config=config)


def optimize_explanation(explanation: ExplanationResult) -> OptimizationResult:
    """Convenience function to optimize an explanation."""
    return get_optimizer().optimize(explanation)


def analyze_explanation(
    explanation: ExplanationResult,
) -> List[OptimizationSuggestion]:
    """Convenience function to analyze an explanation."""
    return get_optimizer().analyze(explanation)


def optimize_text(text: str) -> OptimizationResult:
    """Convenience function to optimize explanation text."""
    return get_optimizer().optimize_text(text)
