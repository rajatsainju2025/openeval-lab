"""Dataset abstractions for evaluation data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from .example import Example


class Dataset(ABC):
    """Abstract base class for evaluation datasets.

    A Dataset provides an iterable interface over evaluation Examples. It represents
    a collection of inputs and their corresponding reference outputs/answers that
    will be used to evaluate model performance.

    Implementations must provide an iterator over Examples. They should also provide
    a meaningful name that identifies the dataset. The length (number of examples)
    is computed automatically if not overridden.

    Invariants:
        - Iterator must be deterministic given a seed
        - Examples must have unique IDs within the dataset
        - Must support multiple iterations (reusable)

    Attributes:
        name: A unique identifier for this dataset implementation.

    Example:
        >>> class QADataset(Dataset):
        ...     name = "qa_dataset"
        ...     def __iter__(self):
        ...         yield Example(id="1", input="What is 2+2?", reference="4")
        ...         yield Example(id="2", input="What is pi?", reference="3.14159")
    """

    name: str

    @abstractmethod
    def __iter__(self) -> Iterator[Example]:
        """Iterate over examples in the dataset.

        Returns:
            An iterator over Example instances.

        Note:
            - Must be reusable (support multiple iterations)
            - Must be deterministic if seed is set
            - Should lazy load if possible for large datasets
        """
        ...

    def __len__(self) -> int:
        """Get the number of examples in the dataset.

        Returns:
            The total number of examples.

        Note:
            Default implementation consumes the iterator.
            Override for more efficient implementation.
        """
        return sum(1 for _ in iter(self))


__all__ = ["Dataset"]
