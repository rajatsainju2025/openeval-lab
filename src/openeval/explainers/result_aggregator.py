"""Result aggregator for combining and analyzing multiple explanation results.

This module provides utilities for aggregating explanation results from multiple
explainers, computing statistics, and generating comprehensive reports.
"""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
)

from .types import CodeElementType, ExplainLevel, ExplanationResult


class AggregationStrategy(Enum):
    """Strategy for aggregating multiple results."""

    FIRST = auto()  # Take the first result
    LAST = auto()  # Take the last result
    BEST = auto()  # Take the best by confidence
    WORST = auto()  # Take the worst by confidence
    MAJORITY = auto()  # Take the majority consensus
    WEIGHTED = auto()  # Weighted combination
    MERGE = auto()  # Merge all results
    ENSEMBLE = auto()  # Ensemble combination


@dataclass
class AggregationStats:
    """Statistics about aggregated results."""

    count: int = 0
    mean_confidence: float = 0.0
    median_confidence: float = 0.0
    std_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    confidence_scores: list[float] = field(default_factory=list)

    # Timing stats
    total_time_ms: float = 0.0
    mean_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0

    # Coverage stats
    elements_covered: int = 0
    elements_total: int = 0
    coverage_ratio: float = 0.0

    # Quality stats
    agreement_ratio: float = 0.0
    diversity_score: float = 0.0

    def update(self, confidences: list[float], times: list[float] | None = None) -> None:
        """Update statistics with new data."""
        if confidences:
            self.count = len(confidences)
            self.confidence_scores = confidences
            self.mean_confidence = statistics.mean(confidences)
            self.median_confidence = statistics.median(confidences)
            self.min_confidence = min(confidences)
            self.max_confidence = max(confidences)
            if len(confidences) > 1:
                self.std_confidence = statistics.stdev(confidences)

        if times:
            self.total_time_ms = sum(times)
            self.mean_time_ms = statistics.mean(times) if times else 0.0
            self.min_time_ms = min(times) if times else 0.0
            self.max_time_ms = max(times) if times else 0.0


@dataclass
class AggregatedResult:
    """Result of aggregating multiple explanation results."""

    primary_result: ExplanationResult | None = None
    all_results: list[ExplanationResult] = field(default_factory=list)
    stats: AggregationStats = field(default_factory=AggregationStats)
    strategy_used: AggregationStrategy = AggregationStrategy.BEST
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def explanation(self) -> str:
        """Get the primary explanation text."""
        return self.primary_result.explanation if self.primary_result else ""

    @property
    def confidence(self) -> float:
        """Get the primary confidence score."""
        return self.primary_result.confidence if self.primary_result else 0.0

    @property
    def count(self) -> int:
        """Get the number of aggregated results."""
        return len(self.all_results)


T = TypeVar("T")


class ResultAggregator(ABC, Generic[T]):
    """Abstract base class for result aggregators."""

    @abstractmethod
    def aggregate(self, results: list[T]) -> T | None:
        """Aggregate multiple results into a single result."""
        ...

    @abstractmethod
    def compute_stats(self, results: list[T]) -> AggregationStats:
        """Compute statistics for a list of results."""
        ...


class ExplanationAggregator(ResultAggregator[ExplanationResult]):
    """Aggregator for explanation results."""

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.BEST,
        weights: dict[str, float] | None = None,
        merge_separator: str = "\n\n---\n\n",
    ):
        """Initialize aggregator.

        Args:
            strategy: Aggregation strategy to use
            weights: Weights for weighted aggregation (explainer name -> weight)
            merge_separator: Separator for merged explanations
        """
        self.strategy = strategy
        self.weights = weights or {}
        self.merge_separator = merge_separator
        self._custom_scorers: list[Callable[[ExplanationResult], float]] = []

    def add_scorer(self, scorer: Callable[[ExplanationResult], float]) -> None:
        """Add a custom scorer for result ranking."""
        self._custom_scorers.append(scorer)

    def aggregate(self, results: list[ExplanationResult]) -> ExplanationResult | None:
        """Aggregate multiple explanation results."""
        if not results:
            return None

        if len(results) == 1:
            return results[0]

        strategy_handlers = {
            AggregationStrategy.FIRST: self._aggregate_first,
            AggregationStrategy.LAST: self._aggregate_last,
            AggregationStrategy.BEST: self._aggregate_best,
            AggregationStrategy.WORST: self._aggregate_worst,
            AggregationStrategy.MAJORITY: self._aggregate_majority,
            AggregationStrategy.WEIGHTED: self._aggregate_weighted,
            AggregationStrategy.MERGE: self._aggregate_merge,
            AggregationStrategy.ENSEMBLE: self._aggregate_ensemble,
        }

        handler = strategy_handlers.get(self.strategy, self._aggregate_best)
        return handler(results)

    def compute_stats(self, results: list[ExplanationResult]) -> AggregationStats:
        """Compute statistics for explanation results."""
        stats = AggregationStats()
        if not results:
            return stats

        confidences = [r.confidence for r in results]
        times = []
        for r in results:
            if r.analysis_metadata and "duration_ms" in r.analysis_metadata:
                times.append(r.analysis_metadata["duration_ms"])

        stats.update(confidences, times)

        # Compute agreement ratio (how similar are the results)
        if len(results) > 1:
            stats.agreement_ratio = self._compute_agreement(results)
            stats.diversity_score = self._compute_diversity(results)

        return stats

    def aggregate_with_stats(self, results: list[ExplanationResult]) -> AggregatedResult:
        """Aggregate results and compute statistics."""
        return AggregatedResult(
            primary_result=self.aggregate(results),
            all_results=results,
            stats=self.compute_stats(results),
            strategy_used=self.strategy,
            metadata={"weights": self.weights},
        )

    def _aggregate_first(self, results: list[ExplanationResult]) -> ExplanationResult:
        """Return the first result."""
        return results[0]

    def _aggregate_last(self, results: list[ExplanationResult]) -> ExplanationResult:
        """Return the last result."""
        return results[-1]

    def _aggregate_best(self, results: list[ExplanationResult]) -> ExplanationResult:
        """Return the result with highest confidence."""
        return max(results, key=lambda r: self._score_result(r))

    def _aggregate_worst(self, results: list[ExplanationResult]) -> ExplanationResult:
        """Return the result with lowest confidence."""
        return min(results, key=lambda r: self._score_result(r))

    def _aggregate_majority(self, results: list[ExplanationResult]) -> ExplanationResult:
        """Return the most common result (by explanation similarity)."""
        # Group by similar explanations
        groups: dict[str, list[ExplanationResult]] = defaultdict(list)
        for r in results:
            # Use first 100 chars as key for grouping
            key = r.explanation[:100].lower().strip()
            groups[key].append(r)

        # Find largest group
        largest_group = max(groups.values(), key=len)
        # Return highest confidence from largest group
        return max(largest_group, key=lambda r: r.confidence)

    def _aggregate_weighted(self, results: list[ExplanationResult]) -> ExplanationResult:
        """Return weighted combination of results."""
        if not self.weights:
            return self._aggregate_best(results)

        # Score each result with weights
        def weighted_score(r: ExplanationResult) -> float:
            model = r.model_used or "default"
            weight = self.weights.get(model, 1.0)
            return r.confidence * weight

        return max(results, key=weighted_score)

    def _aggregate_merge(self, results: list[ExplanationResult]) -> ExplanationResult:
        """Merge all explanations into one."""
        if not results:
            return results[0]

        merged_explanation = self.merge_separator.join(
            f"[{r.model_used or 'Unknown'}]\n{r.explanation}" for r in results
        )

        # Use average confidence
        avg_confidence = statistics.mean(r.confidence for r in results)

        return ExplanationResult(
            element=results[0].element,
            explanation=merged_explanation,
            level=results[0].level,
            confidence=avg_confidence,
            analysis_metadata={
                "merged_from": len(results),
                "source_models": [r.model_used for r in results],
            },
            model_used="merged",
        )

    def _aggregate_ensemble(self, results: list[ExplanationResult]) -> ExplanationResult:
        """Ensemble aggregation combining multiple strategies."""
        # Get candidates from different strategies
        best = self._aggregate_best(results)
        majority = self._aggregate_majority(results)

        # If they agree, return with boosted confidence
        if best.explanation[:100] == majority.explanation[:100]:
            return ExplanationResult(
                element=best.element,
                explanation=best.explanation,
                level=best.level,
                confidence=min(1.0, best.confidence * 1.1),  # Boost confidence
                analysis_metadata={
                    "ensemble_agreement": True,
                    **best.analysis_metadata,
                },
                model_used=best.model_used,
            )

        # Otherwise return the best with ensemble metadata
        return ExplanationResult(
            element=best.element,
            explanation=best.explanation,
            level=best.level,
            confidence=best.confidence,
            analysis_metadata={
                "ensemble_agreement": False,
                "alternative_explanation": majority.explanation[:200],
                **best.analysis_metadata,
            },
            model_used=best.model_used,
        )

    def _score_result(self, result: ExplanationResult) -> float:
        """Score a result using base confidence and custom scorers."""
        score = result.confidence
        for scorer in self._custom_scorers:
            score += scorer(result)
        return score

    def _compute_agreement(self, results: list[ExplanationResult]) -> float:
        """Compute agreement ratio between results."""
        if len(results) < 2:
            return 1.0

        # Simple word overlap metric
        word_sets = [set(r.explanation.lower().split()) for r in results]
        if not word_sets:
            return 0.0

        # Compute pairwise Jaccard similarity
        similarities = []
        for i, s1 in enumerate(word_sets):
            for s2 in word_sets[i + 1 :]:
                if s1 or s2:
                    intersection = len(s1 & s2)
                    union = len(s1 | s2)
                    similarities.append(intersection / union if union > 0 else 0)

        return statistics.mean(similarities) if similarities else 0.0

    def _compute_diversity(self, results: list[ExplanationResult]) -> float:
        """Compute diversity score (inverse of agreement)."""
        return 1.0 - self._compute_agreement(results)


class ResultCollector:
    """Collect and organize results for aggregation."""

    def __init__(self):
        """Initialize collector."""
        self._results: dict[str, list[ExplanationResult]] = defaultdict(list)
        self._by_element: dict[str, list[ExplanationResult]] = defaultdict(list)
        self._by_level: dict[ExplainLevel, list[ExplanationResult]] = defaultdict(list)
        self._by_model: dict[str, list[ExplanationResult]] = defaultdict(list)
        self._timestamps: list[datetime] = []

    def add(self, result: ExplanationResult, source: str = "default") -> None:
        """Add a result to the collector."""
        self._results[source].append(result)
        self._by_element[result.element.name].append(result)
        self._by_level[result.level].append(result)
        if result.model_used:
            self._by_model[result.model_used].append(result)
        if result.timestamp:
            self._timestamps.append(
                datetime.fromisoformat(result.timestamp)
                if isinstance(result.timestamp, str)
                else datetime.utcnow()
            )

    def add_batch(self, results: list[ExplanationResult], source: str = "default") -> None:
        """Add multiple results."""
        for result in results:
            self.add(result, source)

    def get_all(self) -> list[ExplanationResult]:
        """Get all collected results."""
        return [r for results in self._results.values() for r in results]

    def get_by_source(self, source: str) -> list[ExplanationResult]:
        """Get results from a specific source."""
        return self._results.get(source, [])

    def get_by_element(self, element_name: str) -> list[ExplanationResult]:
        """Get results for a specific element."""
        return self._by_element.get(element_name, [])

    def get_by_level(self, level: ExplainLevel) -> list[ExplanationResult]:
        """Get results at a specific explanation level."""
        return self._by_level.get(level, [])

    def get_by_model(self, model: str) -> list[ExplanationResult]:
        """Get results from a specific model."""
        return self._by_model.get(model, [])

    def get_sources(self) -> list[str]:
        """Get all sources."""
        return list(self._results.keys())

    def get_models(self) -> list[str]:
        """Get all models used."""
        return list(self._by_model.keys())

    def get_elements(self) -> list[str]:
        """Get all element names."""
        return list(self._by_element.keys())

    def count(self) -> int:
        """Get total result count."""
        return sum(len(results) for results in self._results.values())

    def clear(self) -> None:
        """Clear all collected results."""
        self._results.clear()
        self._by_element.clear()
        self._by_level.clear()
        self._by_model.clear()
        self._timestamps.clear()


@dataclass
class GroupedStats:
    """Statistics grouped by a key."""

    key: str
    count: int
    stats: AggregationStats
    results: list[ExplanationResult]


class StatsReporter:
    """Generate statistics reports from aggregated results."""

    def __init__(self, aggregator: ExplanationAggregator | None = None):
        """Initialize reporter."""
        self.aggregator = aggregator or ExplanationAggregator()

    def report_by_element(self, collector: ResultCollector) -> dict[str, GroupedStats]:
        """Generate stats grouped by code element."""
        report = {}
        for element_name in collector.get_elements():
            results = collector.get_by_element(element_name)
            stats = self.aggregator.compute_stats(results)
            report[element_name] = GroupedStats(
                key=element_name,
                count=len(results),
                stats=stats,
                results=results,
            )
        return report

    def report_by_model(self, collector: ResultCollector) -> dict[str, GroupedStats]:
        """Generate stats grouped by model."""
        report = {}
        for model in collector.get_models():
            results = collector.get_by_model(model)
            stats = self.aggregator.compute_stats(results)
            report[model] = GroupedStats(
                key=model,
                count=len(results),
                stats=stats,
                results=results,
            )
        return report

    def report_by_level(self, collector: ResultCollector) -> dict[ExplainLevel, GroupedStats]:
        """Generate stats grouped by explanation level."""
        report = {}
        for level in ExplainLevel:
            results = collector.get_by_level(level)
            if results:
                stats = self.aggregator.compute_stats(results)
                report[level] = GroupedStats(
                    key=level.name,
                    count=len(results),
                    stats=stats,
                    results=results,
                )
        return report

    def summary_report(self, collector: ResultCollector) -> dict[str, Any]:
        """Generate a summary report."""
        all_results = collector.get_all()
        stats = self.aggregator.compute_stats(all_results)

        return {
            "total_results": len(all_results),
            "sources": collector.get_sources(),
            "models": collector.get_models(),
            "elements": collector.get_elements(),
            "stats": {
                "mean_confidence": stats.mean_confidence,
                "median_confidence": stats.median_confidence,
                "std_confidence": stats.std_confidence,
                "min_confidence": stats.min_confidence,
                "max_confidence": stats.max_confidence,
                "agreement_ratio": stats.agreement_ratio,
                "diversity_score": stats.diversity_score,
            },
            "by_model": {
                model: len(collector.get_by_model(model)) for model in collector.get_models()
            },
            "by_level": {level.name: len(collector.get_by_level(level)) for level in ExplainLevel},
        }


class BulkAggregator:
    """Aggregate results in bulk with parallel processing support."""

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.BEST,
        batch_size: int = 100,
    ):
        """Initialize bulk aggregator."""
        self.strategy = strategy
        self.batch_size = batch_size
        self.aggregator = ExplanationAggregator(strategy=strategy)

    def aggregate_by_element(self, results: list[ExplanationResult]) -> dict[str, AggregatedResult]:
        """Aggregate results grouped by element."""
        collector = ResultCollector()
        collector.add_batch(results)

        aggregated = {}
        for element_name in collector.get_elements():
            element_results = collector.get_by_element(element_name)
            aggregated[element_name] = self.aggregator.aggregate_with_stats(element_results)
        return aggregated

    def aggregate_by_type(
        self, results: list[ExplanationResult]
    ) -> dict[CodeElementType, AggregatedResult]:
        """Aggregate results grouped by element type."""
        by_type: dict[CodeElementType, list[ExplanationResult]] = defaultdict(list)
        for r in results:
            by_type[r.element.type].append(r)

        return {
            element_type: self.aggregator.aggregate_with_stats(type_results)
            for element_type, type_results in by_type.items()
        }

    def aggregate_all(self, results: list[ExplanationResult]) -> AggregatedResult:
        """Aggregate all results into a single result."""
        return self.aggregator.aggregate_with_stats(results)


@dataclass
class ComparisonResult:
    """Result of comparing multiple aggregators."""

    aggregator_name: str
    result: AggregatedResult
    rank: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class AggregatorComparison:
    """Compare results from multiple aggregation strategies."""

    def __init__(self):
        """Initialize comparison."""
        self._aggregators: dict[str, ExplanationAggregator] = {}

    def add_aggregator(
        self, name: str, aggregator: ExplanationAggregator
    ) -> "AggregatorComparison":
        """Add an aggregator for comparison."""
        self._aggregators[name] = aggregator
        return self

    def add_strategy(self, name: str, strategy: AggregationStrategy) -> "AggregatorComparison":
        """Add a strategy for comparison."""
        self._aggregators[name] = ExplanationAggregator(strategy=strategy)
        return self

    def compare(self, results: list[ExplanationResult]) -> list[ComparisonResult]:
        """Compare all aggregators on the given results."""
        comparisons = []

        for name, aggregator in self._aggregators.items():
            aggregated = aggregator.aggregate_with_stats(results)

            # Score based on confidence and agreement
            score = aggregated.stats.mean_confidence * 0.6 + aggregated.stats.agreement_ratio * 0.4

            comparisons.append(
                ComparisonResult(
                    aggregator_name=name,
                    result=aggregated,
                    rank=0,  # Will be set after sorting
                    score=score,
                    metadata={"strategy": aggregator.strategy.name},
                )
            )

        # Sort by score and assign ranks
        comparisons.sort(key=lambda c: c.score, reverse=True)
        for i, comparison in enumerate(comparisons):
            comparison.rank = i + 1

        return comparisons

    def best_aggregator(self, results: list[ExplanationResult]) -> tuple[str, AggregatedResult]:
        """Find the best aggregator for the given results."""
        comparisons = self.compare(results)
        if not comparisons:
            return "", AggregatedResult()
        best = comparisons[0]
        return best.aggregator_name, best.result


# Convenience functions
def aggregate_results(
    results: list[ExplanationResult],
    strategy: AggregationStrategy = AggregationStrategy.BEST,
) -> AggregatedResult:
    """Aggregate explanation results with the given strategy."""
    aggregator = ExplanationAggregator(strategy=strategy)
    return aggregator.aggregate_with_stats(results)


def compute_result_stats(results: list[ExplanationResult]) -> AggregationStats:
    """Compute statistics for explanation results."""
    aggregator = ExplanationAggregator()
    return aggregator.compute_stats(results)


def merge_explanations(
    results: list[ExplanationResult],
    separator: str = "\n\n---\n\n",
) -> ExplanationResult | None:
    """Merge multiple explanations into one."""
    aggregator = ExplanationAggregator(
        strategy=AggregationStrategy.MERGE,
        merge_separator=separator,
    )
    return aggregator.aggregate(results)


def collect_results() -> ResultCollector:
    """Create a new result collector."""
    return ResultCollector()


def create_reporter(
    strategy: AggregationStrategy = AggregationStrategy.BEST,
) -> StatsReporter:
    """Create a stats reporter."""
    return StatsReporter(ExplanationAggregator(strategy=strategy))


# Singleton for default collector
_default_collector: ResultCollector | None = None


def get_default_collector() -> ResultCollector:
    """Get or create the default result collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = ResultCollector()
    return _default_collector


def reset_default_collector() -> None:
    """Reset the default collector."""
    global _default_collector
    if _default_collector is not None:
        _default_collector.clear()
    _default_collector = None


__all__ = [
    # Strategies
    "AggregationStrategy",
    # Stats
    "AggregationStats",
    "AggregatedResult",
    "GroupedStats",
    # Aggregators
    "ResultAggregator",
    "ExplanationAggregator",
    "BulkAggregator",
    # Collection
    "ResultCollector",
    # Reporting
    "StatsReporter",
    # Comparison
    "ComparisonResult",
    "AggregatorComparison",
    # Functions
    "aggregate_results",
    "compute_result_stats",
    "merge_explanations",
    "collect_results",
    "create_reporter",
    "get_default_collector",
    "reset_default_collector",
]
