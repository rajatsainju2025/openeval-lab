"""A/B testing framework for comparing explainer configurations.

This module provides utilities for running controlled experiments comparing
different explainer configurations, models, or strategies.
"""

from __future__ import annotations

import hashlib
import random
import statistics
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
)

from .types import CodeElement, ExplainLevel, ExplanationResult


class ExperimentStatus(Enum):
    """Status of an A/B experiment."""

    DRAFT = auto()  # Not started
    RUNNING = auto()  # Currently active
    PAUSED = auto()  # Temporarily stopped
    COMPLETED = auto()  # Finished
    CANCELLED = auto()  # Cancelled


class VariantType(Enum):
    """Type of variant in an experiment."""

    CONTROL = auto()  # Control group (baseline)
    TREATMENT = auto()  # Treatment group (new approach)


class AssignmentStrategy(Enum):
    """Strategy for assigning subjects to variants."""

    RANDOM = auto()  # Random assignment
    HASH = auto()  # Hash-based deterministic assignment
    ROUND_ROBIN = auto()  # Alternating assignment
    WEIGHTED = auto()  # Weighted random assignment
    STICKY = auto()  # Sticky assignment based on subject ID


@dataclass
class Variant:
    """A variant in an A/B experiment."""

    name: str
    type: VariantType
    config: dict[str, Any] = field(default_factory=dict)
    weight: float = 0.5  # Traffic allocation weight
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""

    name: str
    description: str = ""
    variants: list[Variant] = field(default_factory=list)
    assignment_strategy: AssignmentStrategy = AssignmentStrategy.RANDOM
    sample_size: int = 100  # Target sample size per variant
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.05
    start_time: datetime | None = None
    end_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure we have at least control and treatment variants."""
        if not self.variants:
            self.variants = [
                Variant(name="control", type=VariantType.CONTROL),
                Variant(name="treatment", type=VariantType.TREATMENT),
            ]


@dataclass
class Assignment:
    """Assignment of a subject to a variant."""

    subject_id: str
    variant: Variant
    experiment_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of running an A/B experiment."""

    experiment_id: str
    variant: Variant
    element: CodeElement
    result: ExplanationResult
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantStats:
    """Statistics for a variant."""

    variant_name: str
    sample_size: int = 0
    mean_confidence: float = 0.0
    std_confidence: float = 0.0
    mean_duration_ms: float = 0.0
    std_duration_ms: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    results: list[ExperimentResult] = field(default_factory=list)

    def update(self, results: list[ExperimentResult]) -> None:
        """Update statistics from results."""
        self.results = results
        self.sample_size = len(results)

        if not results:
            return

        confidences = [r.result.confidence for r in results]
        durations = [r.duration_ms for r in results]

        self.mean_confidence = statistics.mean(confidences)
        self.mean_duration_ms = statistics.mean(durations)

        if len(results) > 1:
            self.std_confidence = statistics.stdev(confidences)
            self.std_duration_ms = statistics.stdev(durations)


@dataclass
class ExperimentSummary:
    """Summary of an A/B experiment."""

    experiment_id: str
    name: str
    status: ExperimentStatus
    variant_stats: dict[str, VariantStats] = field(default_factory=dict)
    winner: str | None = None
    confidence: float = 0.0
    effect_size: float = 0.0
    is_significant: bool = False
    recommendation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


T = TypeVar("T")


class VariantAssigner(ABC):
    """Abstract base for variant assignment strategies."""

    @abstractmethod
    def assign(self, subject_id: str, variants: list[Variant], experiment_id: str) -> Assignment:
        """Assign a subject to a variant."""
        ...


class RandomAssigner(VariantAssigner):
    """Random variant assignment."""

    def __init__(self, seed: int | None = None):
        """Initialize with optional seed."""
        self._random = random.Random(seed)

    def assign(self, subject_id: str, variants: list[Variant], experiment_id: str) -> Assignment:
        """Randomly assign to a variant based on weights."""
        total_weight = sum(v.weight for v in variants)
        r = self._random.random() * total_weight
        cumulative = 0.0

        for variant in variants:
            cumulative += variant.weight
            if r <= cumulative:
                return Assignment(
                    subject_id=subject_id,
                    variant=variant,
                    experiment_id=experiment_id,
                )

        return Assignment(
            subject_id=subject_id,
            variant=variants[-1],
            experiment_id=experiment_id,
        )


class HashAssigner(VariantAssigner):
    """Hash-based deterministic variant assignment."""

    def assign(self, subject_id: str, variants: list[Variant], experiment_id: str) -> Assignment:
        """Assign based on hash of subject_id + experiment_id."""
        key = f"{subject_id}:{experiment_id}"
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        total_weight = sum(v.weight for v in variants)
        position = (hash_value % 1000) / 1000 * total_weight
        cumulative = 0.0

        for variant in variants:
            cumulative += variant.weight
            if position <= cumulative:
                return Assignment(
                    subject_id=subject_id,
                    variant=variant,
                    experiment_id=experiment_id,
                )

        return Assignment(
            subject_id=subject_id,
            variant=variants[-1],
            experiment_id=experiment_id,
        )


class RoundRobinAssigner(VariantAssigner):
    """Round-robin variant assignment."""

    def __init__(self):
        """Initialize counter."""
        self._counter = 0

    def assign(self, subject_id: str, variants: list[Variant], experiment_id: str) -> Assignment:
        """Assign in round-robin order."""
        variant = variants[self._counter % len(variants)]
        self._counter += 1
        return Assignment(
            subject_id=subject_id,
            variant=variant,
            experiment_id=experiment_id,
        )


class StickyAssigner(VariantAssigner):
    """Sticky assignment - same subject always gets same variant."""

    def __init__(self):
        """Initialize assignment cache."""
        self._assignments: dict[str, Assignment] = {}
        self._hash_assigner = HashAssigner()

    def assign(self, subject_id: str, variants: list[Variant], experiment_id: str) -> Assignment:
        """Return cached assignment or create new one."""
        key = f"{subject_id}:{experiment_id}"
        if key not in self._assignments:
            self._assignments[key] = self._hash_assigner.assign(subject_id, variants, experiment_id)
        return self._assignments[key]


class Experiment:
    """An A/B experiment for comparing explainer configurations."""

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment."""
        self.id = str(uuid.uuid4())
        self.config = config
        self.status = ExperimentStatus.DRAFT
        self._results: list[ExperimentResult] = []
        self._assigner = self._create_assigner()
        self._assignments: dict[str, Assignment] = {}
        self.created_at = datetime.utcnow()

    def _create_assigner(self) -> VariantAssigner:
        """Create assigner based on strategy."""
        assigners = {
            AssignmentStrategy.RANDOM: RandomAssigner,
            AssignmentStrategy.HASH: HashAssigner,
            AssignmentStrategy.ROUND_ROBIN: RoundRobinAssigner,
            AssignmentStrategy.WEIGHTED: RandomAssigner,
            AssignmentStrategy.STICKY: StickyAssigner,
        }
        assigner_class = assigners.get(self.config.assignment_strategy, RandomAssigner)
        return assigner_class()

    def start(self) -> "Experiment":
        """Start the experiment."""
        if self.status == ExperimentStatus.DRAFT:
            self.status = ExperimentStatus.RUNNING
            self.config.start_time = datetime.utcnow()
        return self

    def pause(self) -> "Experiment":
        """Pause the experiment."""
        if self.status == ExperimentStatus.RUNNING:
            self.status = ExperimentStatus.PAUSED
        return self

    def resume(self) -> "Experiment":
        """Resume a paused experiment."""
        if self.status == ExperimentStatus.PAUSED:
            self.status = ExperimentStatus.RUNNING
        return self

    def complete(self) -> "Experiment":
        """Mark the experiment as complete."""
        if self.status in (ExperimentStatus.RUNNING, ExperimentStatus.PAUSED):
            self.status = ExperimentStatus.COMPLETED
            self.config.end_time = datetime.utcnow()
        return self

    def cancel(self) -> "Experiment":
        """Cancel the experiment."""
        self.status = ExperimentStatus.CANCELLED
        self.config.end_time = datetime.utcnow()
        return self

    def assign(self, subject_id: str) -> Assignment:
        """Assign a subject to a variant."""
        if subject_id in self._assignments:
            return self._assignments[subject_id]

        assignment = self._assigner.assign(subject_id, self.config.variants, self.id)
        self._assignments[subject_id] = assignment
        return assignment

    def record_result(self, result: ExperimentResult) -> None:
        """Record an experiment result."""
        self._results.append(result)

    def get_results(self, variant_name: str | None = None) -> list[ExperimentResult]:
        """Get results, optionally filtered by variant."""
        if variant_name is None:
            return self._results
        return [r for r in self._results if r.variant.name == variant_name]

    def get_variant_stats(self, variant_name: str) -> VariantStats:
        """Get statistics for a variant."""
        stats = VariantStats(variant_name=variant_name)
        results = self.get_results(variant_name)
        stats.update(results)
        return stats

    def get_summary(self) -> ExperimentSummary:
        """Get experiment summary with statistical analysis."""
        variant_stats = {v.name: self.get_variant_stats(v.name) for v in self.config.variants}

        # Find control and treatment
        control_stats = None
        treatment_stats = None
        for variant in self.config.variants:
            stats = variant_stats[variant.name]
            if variant.type == VariantType.CONTROL:
                control_stats = stats
            elif variant.type == VariantType.TREATMENT:
                treatment_stats = stats

        # Calculate significance
        winner = None
        confidence = 0.0
        effect_size = 0.0
        is_significant = False
        recommendation = "Not enough data for analysis"

        if control_stats and treatment_stats:
            if control_stats.sample_size > 0 and treatment_stats.sample_size > 0:
                effect_size = treatment_stats.mean_confidence - control_stats.mean_confidence

                # Simple significance calculation (would use proper stats in production)
                if control_stats.sample_size >= 30 and treatment_stats.sample_size >= 30:
                    # Simplified z-test approximation
                    pooled_std = (control_stats.std_confidence + treatment_stats.std_confidence) / 2
                    if pooled_std > 0:
                        z_score = abs(effect_size) / pooled_std
                        # Approximate p-value (simplified)
                        if z_score > 1.96:
                            is_significant = True
                            confidence = min(0.99, 0.5 + z_score * 0.1)

                if effect_size > self.config.minimum_effect_size and is_significant:
                    winner = "treatment"
                    recommendation = "Treatment variant shows significant improvement"
                elif effect_size < -self.config.minimum_effect_size and is_significant:
                    winner = "control"
                    recommendation = "Control variant performs better"
                elif is_significant:
                    recommendation = "Statistically significant but effect size below threshold"
                else:
                    recommendation = "No statistically significant difference detected"

        return ExperimentSummary(
            experiment_id=self.id,
            name=self.config.name,
            status=self.status,
            variant_stats=variant_stats,
            winner=winner,
            confidence=confidence,
            effect_size=effect_size,
            is_significant=is_significant,
            recommendation=recommendation,
        )

    def is_complete(self) -> bool:
        """Check if experiment has enough data."""
        for variant in self.config.variants:
            stats = self.get_variant_stats(variant.name)
            if stats.sample_size < self.config.sample_size:
                return False
        return True


class ExperimentRunner(Generic[T]):
    """Runner for executing A/B experiments."""

    def __init__(
        self,
        experiment: Experiment,
        control_fn: Callable[[CodeElement, ExplainLevel], ExplanationResult],
        treatment_fn: Callable[[CodeElement, ExplainLevel], ExplanationResult],
    ):
        """Initialize runner.

        Args:
            experiment: The experiment to run
            control_fn: Function to generate control results
            treatment_fn: Function to generate treatment results
        """
        self.experiment = experiment
        self._control_fn = control_fn
        self._treatment_fn = treatment_fn
        self._variant_fns: dict[str, Callable] = {}

        # Map variant names to functions
        for variant in experiment.config.variants:
            if variant.type == VariantType.CONTROL:
                self._variant_fns[variant.name] = control_fn
            elif variant.type == VariantType.TREATMENT:
                self._variant_fns[variant.name] = treatment_fn

    def add_variant_fn(
        self,
        variant_name: str,
        fn: Callable[[CodeElement, ExplainLevel], ExplanationResult],
    ) -> "ExperimentRunner":
        """Add a function for a specific variant."""
        self._variant_fns[variant_name] = fn
        return self

    def run_single(
        self,
        element: CodeElement,
        level: ExplainLevel = ExplainLevel.DETAILED,
        subject_id: str | None = None,
    ) -> ExperimentResult:
        """Run experiment for a single element."""
        # Generate subject ID if not provided
        if subject_id is None:
            subject_id = f"{element.name}:{element.line_start}"

        # Get assignment
        assignment = self.experiment.assign(subject_id)
        variant = assignment.variant

        # Get appropriate function
        fn = self._variant_fns.get(variant.name)
        if fn is None:
            raise ValueError(f"No function registered for variant: {variant.name}")

        # Execute and time
        start = datetime.utcnow()
        result = fn(element, level)
        end = datetime.utcnow()
        duration_ms = (end - start).total_seconds() * 1000

        # Create experiment result
        exp_result = ExperimentResult(
            experiment_id=self.experiment.id,
            variant=variant,
            element=element,
            result=result,
            metrics={"confidence": result.confidence},
            duration_ms=duration_ms,
        )

        # Record result
        self.experiment.record_result(exp_result)

        return exp_result

    def run_batch(
        self,
        elements: list[CodeElement],
        level: ExplainLevel = ExplainLevel.DETAILED,
    ) -> list[ExperimentResult]:
        """Run experiment for multiple elements."""
        return [self.run_single(element, level) for element in elements]


class ABTestingManager:
    """Manager for multiple A/B experiments."""

    def __init__(self):
        """Initialize manager."""
        self._experiments: dict[str, Experiment] = {}
        self._active_experiments: dict[str, Experiment] = {}

    def create_experiment(
        self,
        name: str,
        description: str = "",
        control_config: dict[str, Any] | None = None,
        treatment_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Experiment:
        """Create a new experiment."""
        variants = [
            Variant(
                name="control",
                type=VariantType.CONTROL,
                config=control_config or {},
            ),
            Variant(
                name="treatment",
                type=VariantType.TREATMENT,
                config=treatment_config or {},
            ),
        ]

        config = ExperimentConfig(
            name=name,
            description=description,
            variants=variants,
            **kwargs,
        )

        experiment = Experiment(config)
        self._experiments[experiment.id] = experiment
        return experiment

    def create_multivariate_experiment(
        self,
        name: str,
        variants: list[dict[str, Any]],
        description: str = "",
        **kwargs: Any,
    ) -> Experiment:
        """Create a multivariate experiment with multiple variants."""
        experiment_variants = []
        for i, variant_config in enumerate(variants):
            variant_type = VariantType.CONTROL if i == 0 else VariantType.TREATMENT
            experiment_variants.append(
                Variant(
                    name=variant_config.get("name", f"variant_{i}"),
                    type=variant_type,
                    config=variant_config.get("config", {}),
                    weight=variant_config.get("weight", 1.0 / len(variants)),
                    description=variant_config.get("description", ""),
                )
            )

        config = ExperimentConfig(
            name=name,
            description=description,
            variants=experiment_variants,
            **kwargs,
        )

        experiment = Experiment(config)
        self._experiments[experiment.id] = experiment
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)

    def get_experiment_by_name(self, name: str) -> Experiment | None:
        """Get an experiment by name."""
        for experiment in self._experiments.values():
            if experiment.config.name == name:
                return experiment
        return None

    def list_experiments(self, status: ExperimentStatus | None = None) -> list[Experiment]:
        """List all experiments, optionally filtered by status."""
        if status is None:
            return list(self._experiments.values())
        return [exp for exp in self._experiments.values() if exp.status == status]

    def start_experiment(self, experiment_id: str) -> Experiment | None:
        """Start an experiment."""
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment.start()
            self._active_experiments[experiment_id] = experiment
        return experiment

    def stop_experiment(self, experiment_id: str) -> Experiment | None:
        """Stop an experiment."""
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment.complete()
            self._active_experiments.pop(experiment_id, None)
        return experiment

    def get_active_experiments(self) -> list[Experiment]:
        """Get all currently running experiments."""
        return list(self._active_experiments.values())

    def get_all_summaries(self) -> list[ExperimentSummary]:
        """Get summaries for all experiments."""
        return [exp.get_summary() for exp in self._experiments.values()]


class FeatureFlag:
    """Feature flag for controlling experiment exposure."""

    def __init__(
        self,
        name: str,
        default_enabled: bool = False,
        experiment: Experiment | None = None,
    ):
        """Initialize feature flag."""
        self.name = name
        self.default_enabled = default_enabled
        self.experiment = experiment
        self._overrides: dict[str, bool] = {}

    def is_enabled(self, subject_id: str | None = None) -> bool:
        """Check if feature is enabled for subject."""
        # Check overrides first
        if subject_id and subject_id in self._overrides:
            return self._overrides[subject_id]

        # Check experiment assignment
        if self.experiment and subject_id:
            assignment = self.experiment.assign(subject_id)
            return assignment.variant.type == VariantType.TREATMENT

        return self.default_enabled

    def override(self, subject_id: str, enabled: bool) -> None:
        """Override feature state for a subject."""
        self._overrides[subject_id] = enabled

    def clear_override(self, subject_id: str) -> None:
        """Clear override for a subject."""
        self._overrides.pop(subject_id, None)


class ExperimentReporter:
    """Generate reports for experiments."""

    def __init__(self):
        """Initialize reporter."""
        pass

    def generate_report(self, experiment: Experiment) -> dict[str, Any]:
        """Generate a detailed report for an experiment."""
        summary = experiment.get_summary()

        return {
            "experiment_id": summary.experiment_id,
            "name": summary.name,
            "status": summary.status.name,
            "variants": {
                name: {
                    "sample_size": stats.sample_size,
                    "mean_confidence": stats.mean_confidence,
                    "std_confidence": stats.std_confidence,
                    "mean_duration_ms": stats.mean_duration_ms,
                }
                for name, stats in summary.variant_stats.items()
            },
            "analysis": {
                "winner": summary.winner,
                "confidence": summary.confidence,
                "effect_size": summary.effect_size,
                "is_significant": summary.is_significant,
                "recommendation": summary.recommendation,
            },
            "created_at": experiment.created_at.isoformat(),
            "start_time": (
                experiment.config.start_time.isoformat() if experiment.config.start_time else None
            ),
            "end_time": (
                experiment.config.end_time.isoformat() if experiment.config.end_time else None
            ),
        }

    def compare_experiments(self, experiments: list[Experiment]) -> dict[str, Any]:
        """Compare multiple experiments."""
        reports = [self.generate_report(exp) for exp in experiments]

        # Find best performing treatment
        best_effect = -float("inf")
        best_experiment = None
        for report in reports:
            if report["analysis"]["is_significant"]:
                effect = report["analysis"]["effect_size"]
                if effect > best_effect:
                    best_effect = effect
                    best_experiment = report["name"]

        return {
            "experiments": reports,
            "best_experiment": best_experiment,
            "best_effect_size": best_effect if best_experiment else None,
        }


# Convenience functions
def create_experiment(
    name: str,
    control_config: dict[str, Any] | None = None,
    treatment_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Experiment:
    """Create a new A/B experiment."""
    config = ExperimentConfig(
        name=name,
        variants=[
            Variant(
                name="control",
                type=VariantType.CONTROL,
                config=control_config or {},
            ),
            Variant(
                name="treatment",
                type=VariantType.TREATMENT,
                config=treatment_config or {},
            ),
        ],
        **kwargs,
    )
    return Experiment(config)


def run_ab_test(
    control_fn: Callable[[CodeElement, ExplainLevel], ExplanationResult],
    treatment_fn: Callable[[CodeElement, ExplainLevel], ExplanationResult],
    elements: list[CodeElement],
    name: str = "ab_test",
    level: ExplainLevel = ExplainLevel.DETAILED,
) -> ExperimentSummary:
    """Run a simple A/B test and return summary."""
    experiment = create_experiment(name)
    experiment.start()

    runner = ExperimentRunner(experiment, control_fn, treatment_fn)
    runner.run_batch(elements, level)

    experiment.complete()
    return experiment.get_summary()


# Singleton manager
_default_manager: ABTestingManager | None = None


def get_ab_testing_manager() -> ABTestingManager:
    """Get or create the default A/B testing manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ABTestingManager()
    return _default_manager


def reset_ab_testing_manager() -> None:
    """Reset the default manager."""
    global _default_manager
    _default_manager = None


__all__ = [
    # Enums
    "ExperimentStatus",
    "VariantType",
    "AssignmentStrategy",
    # Data classes
    "Variant",
    "ExperimentConfig",
    "Assignment",
    "ExperimentResult",
    "VariantStats",
    "ExperimentSummary",
    # Assigners
    "VariantAssigner",
    "RandomAssigner",
    "HashAssigner",
    "RoundRobinAssigner",
    "StickyAssigner",
    # Core classes
    "Experiment",
    "ExperimentRunner",
    "ABTestingManager",
    "FeatureFlag",
    "ExperimentReporter",
    # Functions
    "create_experiment",
    "run_ab_test",
    "get_ab_testing_manager",
    "reset_ab_testing_manager",
]
