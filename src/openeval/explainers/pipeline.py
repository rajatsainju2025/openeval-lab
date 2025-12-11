"""Explainer pipeline for composable explanation workflows.

This module provides a pipeline abstraction for chaining multiple
explainers and transformations into composable workflows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from .types import CodeElement, ExplainLevel, ExplanationResult

T = TypeVar("T")
U = TypeVar("U")


class StageStatus(Enum):
    """Status of a pipeline stage execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult(Generic[T]):
    """Result from executing a pipeline stage."""

    stage_name: str
    status: StageStatus
    output: Optional[T] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if stage completed successfully."""
        return self.status == StageStatus.COMPLETED


@dataclass
class PipelineResult(Generic[T]):
    """Result from executing a complete pipeline."""

    pipeline_name: str
    stages: List[StageResult[Any]] = field(default_factory=list)
    final_output: Optional[T] = None
    total_duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if pipeline completed successfully."""
        return all(stage.success or stage.status == StageStatus.SKIPPED for stage in self.stages)

    @property
    def failed_stages(self) -> List[StageResult[Any]]:
        """Get list of failed stages."""
        return [s for s in self.stages if s.status == StageStatus.FAILED]


class PipelineStage(ABC, Generic[T, U]):
    """Abstract base class for pipeline stages."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize stage.

        Args:
            name: Name for this stage.
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def process(self, input_data: T) -> U:
        """Process input and produce output.

        Args:
            input_data: Input to process.

        Returns:
            Processed output.
        """
        pass

    def should_skip(self, input_data: T) -> bool:
        """Check if this stage should be skipped.

        Args:
            input_data: Input to check.

        Returns:
            True if stage should be skipped.
        """
        return False


class TransformStage(PipelineStage[T, U]):
    """Stage that transforms data using a function."""

    def __init__(
        self,
        transform_func: Callable[[T], U],
        name: Optional[str] = None,
        skip_condition: Optional[Callable[[T], bool]] = None,
    ) -> None:
        """Initialize transform stage.

        Args:
            transform_func: Function to transform input.
            name: Stage name.
            skip_condition: Optional condition to skip stage.
        """
        super().__init__(name or "transform")
        self._transform_func = transform_func
        self._skip_condition = skip_condition

    def process(self, input_data: T) -> U:
        """Apply transformation."""
        return self._transform_func(input_data)

    def should_skip(self, input_data: T) -> bool:
        """Check skip condition."""
        if self._skip_condition:
            return self._skip_condition(input_data)
        return False


class FilterStage(PipelineStage[T, T]):
    """Stage that filters data based on a predicate."""

    def __init__(
        self,
        predicate: Callable[[T], bool],
        name: Optional[str] = None,
    ) -> None:
        """Initialize filter stage.

        Args:
            predicate: Function returning True to keep data.
            name: Stage name.
        """
        super().__init__(name or "filter")
        self._predicate = predicate

    def process(self, input_data: T) -> T:
        """Pass through if predicate is True."""
        return input_data

    def should_skip(self, input_data: T) -> bool:
        """Skip if predicate is False."""
        return not self._predicate(input_data)


class ValidateStage(PipelineStage[T, T]):
    """Stage that validates data."""

    def __init__(
        self,
        validator: Callable[[T], bool],
        error_message: str = "Validation failed",
        name: Optional[str] = None,
    ) -> None:
        """Initialize validation stage.

        Args:
            validator: Function returning True if valid.
            error_message: Message to use if validation fails.
            name: Stage name.
        """
        super().__init__(name or "validate")
        self._validator = validator
        self._error_message = error_message

    def process(self, input_data: T) -> T:
        """Validate and pass through."""
        if not self._validator(input_data):
            raise ValueError(self._error_message)
        return input_data


class BranchStage(PipelineStage[T, T]):
    """Stage that branches based on a condition."""

    def __init__(
        self,
        condition: Callable[[T], bool],
        true_stage: PipelineStage[T, T],
        false_stage: Optional[PipelineStage[T, T]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize branch stage.

        Args:
            condition: Condition to check.
            true_stage: Stage to execute if condition is True.
            false_stage: Optional stage if condition is False.
            name: Stage name.
        """
        super().__init__(name or "branch")
        self._condition = condition
        self._true_stage = true_stage
        self._false_stage = false_stage

    def process(self, input_data: T) -> T:
        """Execute appropriate branch."""
        if self._condition(input_data):
            return self._true_stage.process(input_data)
        elif self._false_stage:
            return self._false_stage.process(input_data)
        return input_data


class ExplainerStage(PipelineStage[CodeElement, ExplanationResult]):
    """Pipeline stage that generates explanations."""

    def __init__(
        self,
        explainer_func: Callable[[CodeElement, ExplainLevel], ExplanationResult],
        level: ExplainLevel = ExplainLevel.SUMMARY,
        name: Optional[str] = None,
    ) -> None:
        """Initialize explainer stage.

        Args:
            explainer_func: Function to generate explanation.
            level: Explanation detail level.
            name: Stage name.
        """
        super().__init__(name or "explain")
        self._explainer_func = explainer_func
        self._level = level

    def process(self, input_data: CodeElement) -> ExplanationResult:
        """Generate explanation."""
        return self._explainer_func(input_data, self._level)


class EnrichmentStage(PipelineStage[ExplanationResult, ExplanationResult]):
    """Stage that enriches explanation results."""

    def __init__(
        self,
        enricher: Callable[[ExplanationResult], Dict[str, Any]],
        name: Optional[str] = None,
    ) -> None:
        """Initialize enrichment stage.

        Args:
            enricher: Function to generate additional metadata.
            name: Stage name.
        """
        super().__init__(name or "enrich")
        self._enricher = enricher

    def process(self, input_data: ExplanationResult) -> ExplanationResult:
        """Enrich the explanation result."""
        additional = self._enricher(input_data)
        # Create new result with enriched metadata
        metadata = {**input_data.analysis_metadata, **additional}
        return ExplanationResult(
            element=input_data.element,
            explanation=input_data.explanation,
            level=input_data.level,
            confidence=input_data.confidence,
            analysis_metadata=metadata,
            timestamp=input_data.timestamp,
            model_used=input_data.model_used,
        )


class Pipeline(Generic[T, U]):
    """Composable pipeline for processing data through stages."""

    def __init__(
        self,
        name: str = "pipeline",
        stages: Optional[List[PipelineStage[Any, Any]]] = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            name: Name for this pipeline.
            stages: Initial list of stages.
        """
        self.name = name
        self._stages: List[PipelineStage[Any, Any]] = stages or []

    def add_stage(self, stage: PipelineStage[Any, Any]) -> "Pipeline[T, U]":
        """Add a stage to the pipeline.

        Args:
            stage: Stage to add.

        Returns:
            Self for chaining.
        """
        self._stages.append(stage)
        return self

    def add_transform(
        self,
        transform_func: Callable[[Any], Any],
        name: Optional[str] = None,
    ) -> "Pipeline[T, U]":
        """Add a transform stage.

        Args:
            transform_func: Transformation function.
            name: Stage name.

        Returns:
            Self for chaining.
        """
        self._stages.append(TransformStage(transform_func, name))
        return self

    def add_filter(
        self,
        predicate: Callable[[Any], bool],
        name: Optional[str] = None,
    ) -> "Pipeline[T, U]":
        """Add a filter stage.

        Args:
            predicate: Filter predicate.
            name: Stage name.

        Returns:
            Self for chaining.
        """
        self._stages.append(FilterStage(predicate, name))
        return self

    def add_validation(
        self,
        validator: Callable[[Any], bool],
        error_message: str = "Validation failed",
        name: Optional[str] = None,
    ) -> "Pipeline[T, U]":
        """Add a validation stage.

        Args:
            validator: Validation function.
            error_message: Error message if validation fails.
            name: Stage name.

        Returns:
            Self for chaining.
        """
        self._stages.append(ValidateStage(validator, error_message, name))
        return self

    def execute(self, input_data: T) -> PipelineResult[U]:
        """Execute the pipeline.

        Args:
            input_data: Input to process.

        Returns:
            PipelineResult with all stage results.
        """
        import time

        start_time = time.perf_counter()
        stage_results: List[StageResult[Any]] = []
        current_data: Any = input_data

        for stage in self._stages:
            stage_start = time.perf_counter()

            try:
                if stage.should_skip(current_data):
                    stage_results.append(
                        StageResult(
                            stage_name=stage.name,
                            status=StageStatus.SKIPPED,
                            output=current_data,
                            duration_ms=0.0,
                        )
                    )
                    continue

                current_data = stage.process(current_data)
                duration_ms = (time.perf_counter() - stage_start) * 1000

                stage_results.append(
                    StageResult(
                        stage_name=stage.name,
                        status=StageStatus.COMPLETED,
                        output=current_data,
                        duration_ms=duration_ms,
                    )
                )

            except Exception as e:
                duration_ms = (time.perf_counter() - stage_start) * 1000
                stage_results.append(
                    StageResult(
                        stage_name=stage.name,
                        status=StageStatus.FAILED,
                        error=str(e),
                        duration_ms=duration_ms,
                    )
                )
                break

        total_duration = (time.perf_counter() - start_time) * 1000

        return PipelineResult(
            pipeline_name=self.name,
            stages=stage_results,
            final_output=current_data if stage_results and stage_results[-1].success else None,
            total_duration_ms=total_duration,
        )

    def __or__(self, other: "Pipeline[U, Any]") -> "Pipeline[T, Any]":
        """Compose pipelines using | operator.

        Args:
            other: Pipeline to chain after this one.

        Returns:
            New combined pipeline.
        """
        combined = Pipeline(f"{self.name}|{other.name}")
        combined._stages = self._stages + other._stages
        return combined


class ExplainerPipeline(Pipeline[CodeElement, ExplanationResult]):
    """Specialized pipeline for code explanation workflows."""

    def __init__(self, name: str = "explainer_pipeline") -> None:
        """Initialize explainer pipeline."""
        super().__init__(name)

    def add_explainer(
        self,
        explainer_func: Callable[[CodeElement, ExplainLevel], ExplanationResult],
        level: ExplainLevel = ExplainLevel.SUMMARY,
        name: Optional[str] = None,
    ) -> "ExplainerPipeline":
        """Add an explainer stage.

        Args:
            explainer_func: Function to generate explanation.
            level: Explanation detail level.
            name: Stage name.

        Returns:
            Self for chaining.
        """
        self._stages.append(ExplainerStage(explainer_func, level, name))
        return self

    def add_enrichment(
        self,
        enricher: Callable[[ExplanationResult], Dict[str, Any]],
        name: Optional[str] = None,
    ) -> "ExplainerPipeline":
        """Add an enrichment stage.

        Args:
            enricher: Function to generate additional metadata.
            name: Stage name.

        Returns:
            Self for chaining.
        """
        self._stages.append(EnrichmentStage(enricher, name))
        return self

    def add_post_processor(
        self,
        processor: Callable[[ExplanationResult], ExplanationResult],
        name: Optional[str] = None,
    ) -> "ExplainerPipeline":
        """Add a post-processing stage.

        Args:
            processor: Function to process explanation.
            name: Stage name.

        Returns:
            Self for chaining.
        """
        self._stages.append(TransformStage(processor, name or "post_process"))
        return self


class ParallelPipeline(Generic[T, U]):
    """Execute multiple pipelines in parallel and combine results."""

    def __init__(
        self,
        pipelines: List[Pipeline[T, U]],
        combiner: Optional[Callable[[List[U]], U]] = None,
        name: str = "parallel_pipeline",
    ) -> None:
        """Initialize parallel pipeline.

        Args:
            pipelines: Pipelines to run in parallel.
            combiner: Function to combine results.
            name: Pipeline name.
        """
        self.name = name
        self._pipelines = pipelines
        self._combiner = combiner

    def execute(self, input_data: T) -> List[PipelineResult[U]]:
        """Execute all pipelines.

        Args:
            input_data: Input to process.

        Returns:
            List of PipelineResults.
        """
        return [pipeline.execute(input_data) for pipeline in self._pipelines]

    def execute_and_combine(self, input_data: T) -> Optional[U]:
        """Execute all pipelines and combine results.

        Args:
            input_data: Input to process.

        Returns:
            Combined result or None.
        """
        results = self.execute(input_data)
        outputs = [r.final_output for r in results if r.success and r.final_output is not None]

        if not outputs:
            return None

        if self._combiner:
            return self._combiner(outputs)

        # Default: return first successful output
        return outputs[0]


class ConditionalPipeline(Generic[T, U]):
    """Execute different pipelines based on conditions."""

    def __init__(self, name: str = "conditional_pipeline") -> None:
        """Initialize conditional pipeline."""
        self.name = name
        self._conditions: List[tuple[Callable[[T], bool], Pipeline[T, U]]] = []
        self._default: Optional[Pipeline[T, U]] = None

    def when(
        self,
        condition: Callable[[T], bool],
        pipeline: Pipeline[T, U],
    ) -> "ConditionalPipeline[T, U]":
        """Add a conditional branch.

        Args:
            condition: Condition to check.
            pipeline: Pipeline to execute if condition is True.

        Returns:
            Self for chaining.
        """
        self._conditions.append((condition, pipeline))
        return self

    def otherwise(self, pipeline: Pipeline[T, U]) -> "ConditionalPipeline[T, U]":
        """Set default pipeline.

        Args:
            pipeline: Default pipeline to execute.

        Returns:
            Self for chaining.
        """
        self._default = pipeline
        return self

    def execute(self, input_data: T) -> Optional[PipelineResult[U]]:
        """Execute the appropriate pipeline.

        Args:
            input_data: Input to process.

        Returns:
            PipelineResult or None.
        """
        for condition, pipeline in self._conditions:
            if condition(input_data):
                return pipeline.execute(input_data)

        if self._default:
            return self._default.execute(input_data)

        return None


def create_pipeline(
    name: str = "pipeline",
    stages: Optional[List[Callable[[Any], Any]]] = None,
) -> Pipeline[Any, Any]:
    """Create a simple pipeline from functions.

    Args:
        name: Pipeline name.
        stages: List of transformation functions.

    Returns:
        Pipeline instance.
    """
    pipeline: Pipeline[Any, Any] = Pipeline(name)
    if stages:
        for i, func in enumerate(stages):
            pipeline.add_transform(func, f"stage_{i}")
    return pipeline


def compose(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose multiple functions into one.

    Args:
        *funcs: Functions to compose.

    Returns:
        Composed function.
    """
    return reduce(lambda f, g: lambda x: g(f(x)), funcs)
