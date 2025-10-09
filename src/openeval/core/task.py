"""Task abstractions for evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from .example import Example
from ..prompt import PromptTemplate

if TYPE_CHECKING:
    from .adapter import Adapter
    from .dataset import Dataset
    from .metric import Metric


class Task(ABC):
    """Abstract base class for evaluation tasks.

    A Task defines how to evaluate a model on a particular capability or behavior.
    It handles converting dataset examples into model-appropriate prompts and
    post-processing model outputs for evaluation.

    Tasks can use either a custom prompt-building implementation or a template-based
    approach. The template approach is recommended for simpler tasks as it provides
    better reproducibility and easier modification.

    Invariants:
        - Prompt building must be deterministic for given example and seed
        - Prompts must be valid for the target model/adapter
        - Post-processing must be consistent and preserve evaluation-critical information

    Attributes:
        name: A unique identifier for this task implementation.
        prompt_template: Optional template for generating prompts from examples.
    """

    name: str

    def __init__(self, prompt_template: Optional[Union[str, PromptTemplate]] = None):
        """Initialize task with optional prompt template.

        Args:
            prompt_template: Either a string template or PromptTemplate instance.
                If a string is provided, it will be converted to a PromptTemplate.
        """
        self._prompt_template_raw = prompt_template
        if isinstance(prompt_template, str):
            self.prompt_template = PromptTemplate(prompt_template)
        else:
            self.prompt_template = prompt_template

    @abstractmethod
    def build_prompt(self, ex: Example) -> str:
        """Convert an example into a model-ready prompt string.

        This is the core method that defines how examples are presented to the model.
        Must be implemented by concrete task classes unless using templates.

        Args:
            ex: The example to convert into a prompt.

        Returns:
            A string prompt ready to be sent to the model.
        """
        ...

    def build_prompt_with_template(self, ex: Example, **extra_vars: Any) -> str:
        """Build prompt using template if available, otherwise fallback to build_prompt.

        This method is used when a prompt template is provided. The template can
        access all example fields (input, reference, id) plus any metadata fields
        and extra variables provided.

        Args:
            ex: The example to convert into a prompt.
            **extra_vars: Additional variables to make available to the template.

        Returns:
            The rendered prompt string.
        """
        if self.prompt_template is not None:
            # Prepare template variables
            variables = {"input": ex.input, "reference": ex.reference, "id": ex.id, **extra_vars}
            # Add meta fields as top-level variables
            if ex.meta:
                variables.update(ex.meta)

            return self.prompt_template.render(**variables)
        else:
            return self.build_prompt(ex)

    def postprocess(self, raw_output: str) -> Any:
        """Post-process raw model output into evaluation-ready format.

        This method allows tasks to clean up or transform model outputs before
        evaluation. Common transformations include:
        - Extracting answers from verbose responses
        - Normalizing formatting (lowercase, remove punctuation)
        - Converting structured outputs (JSON, XML) to evaluation format

        Args:
            raw_output: The raw string output from the model.

        Returns:
            The processed output ready for evaluation. Type depends on the task.

        Note:
            Default implementation returns the raw output unchanged.
            Override for tasks that need output post-processing.
        """
        return raw_output

    def evaluate(
        self,
        adapter: "Adapter",
        dataset: "Dataset",
        metrics: List["Metric"],
        *,
        seed: Optional[int] = 0,
        collect_records: bool = False,
        concurrency: int = 1,
        max_retries: int = 0,
        request_timeout: Optional[float] = None,
        streaming_batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate the task using the provided adapter, dataset, and metrics.

        This is a convenience method that delegates to the evaluation engine.
        For more advanced usage, use the engine directly.

        Args:
            adapter: The model adapter to use for generation.
            dataset: The dataset to evaluate on.
            metrics: List of metrics to compute.
            seed: Random seed for reproducibility.
            collect_records: Whether to collect detailed records.
            concurrency: Number of concurrent requests.
            max_retries: Maximum number of retries per request.
            request_timeout: Timeout for individual requests.
            streaming_batch_size: Batch size for streaming evaluation.

        Returns:
            Dictionary containing evaluation results and metadata.
        """
        # Import here to avoid circular imports
        from ..engine import EvaluationEngine

        engine = EvaluationEngine()
        return engine.evaluate(
            self,
            adapter,
            dataset,
            metrics,
            seed=seed,
            collect_records=collect_records,
            concurrency=concurrency,
            max_retries=max_retries,
            request_timeout=request_timeout,
            streaming_batch_size=streaming_batch_size,
        )


__all__ = ["Task"]
