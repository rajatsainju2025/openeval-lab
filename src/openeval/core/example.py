"""Example data structure for evaluation instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Example:
    """A single evaluation example containing input, reference answer, and metadata.

    An Example represents one instance in an evaluation dataset. It contains the input
    that will be given to the model (after task-specific prompt construction), the
    reference answer(s) that will be used to evaluate the model's output, and optional
    metadata about the example.

    Attributes:
        id: A unique identifier for the example within its dataset.
        input: The raw input that will be processed by the task's prompt template.
            Can be a string for simple QA tasks or structured data for complex tasks.
        reference: The expected output or "ground truth" answer. Can be a string,
            list of strings for multiple references, or structured data.
        meta: Optional dictionary of metadata about this example (e.g., difficulty,
            source, tags). Accessible in prompt templates.

    Example:
        >>> example = Example(
        ...     id="qa-1",
        ...     input="What is the capital of France?",
        ...     reference="Paris",
        ...     meta={"difficulty": "easy", "category": "geography"}
        ... )
    """

    id: str
    input: Any
    reference: Any
    meta: Optional[Dict[str, Any]] = None


__all__ = ["Example"]
