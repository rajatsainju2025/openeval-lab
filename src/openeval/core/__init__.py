"""Core abstractions and protocols for the evaluation framework."""

from .adapter import Adapter
from .dataset import Dataset
from .errors import _categorize_error, _summarize_errors
from .example import Example
from .metric import Metric
from .task import Task

__all__ = [
    "Adapter",
    "Dataset",
    "Example",
    "Metric",
    "Task",
    "_categorize_error",
    "_summarize_errors",
]
