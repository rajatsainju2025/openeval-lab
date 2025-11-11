"""
Generator-based dataset streaming for memory-efficient processing.

Provides streaming interfaces that avoid loading entire datasets into memory,
enabling evaluation of arbitrarily large datasets with constant memory usage.
"""

from typing import Any, Generator, List, Optional


class StreamingDataset:
    """Wraps datasets to provide streaming interface."""

    def __init__(self, dataset: Any, batch_size: int = 1):
        """Initialize streaming dataset wrapper.

        Args:
            dataset: Dataset object supporting iteration
            batch_size: Batch size for processing (default: 1)
        """
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Generator[Any, None, None]:
        """Iterate over dataset items one at a time.

        Yields:
            Dataset items
        """
        for item in self.dataset:
            yield item

    def iter_batches(self, batch_size: Optional[int] = None) -> Generator[List[Any], None, None]:
        """Iterate over batches of items.

        Args:
            batch_size: Optional override for batch size

        Yields:
            Batches of items
        """
        batch_sz = batch_size or self.batch_size
        batch: List[Any] = []

        for item in self.dataset:
            batch.append(item)
            if len(batch) >= batch_sz:
                yield batch
                batch = []

        if batch:
            yield batch

    def iter_with_index(self) -> Generator[tuple[int, Any], None, None]:
        """Iterate with index.

        Yields:
            (index, item) tuples
        """
        for idx, item in enumerate(self.dataset):
            yield idx, item

    def iter_first_n(self, n: int) -> Generator[Any, None, None]:
        """Iterate over first n items.

        Args:
            n: Maximum number of items to yield

        Yields:
            Dataset items (up to n)
        """
        for idx, item in enumerate(self.dataset):
            if idx >= n:
                break
            yield item

    def iter_slice(self, start: int, end: int) -> Generator[Any, None, None]:
        """Iterate over slice of dataset.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Yields:
            Dataset items in slice
        """
        for idx, item in enumerate(self.dataset):
            if idx < start:
                continue
            if idx >= end:
                break
            yield item

    def count(self) -> int:
        """Count total items in dataset.

        Note: This consumes the dataset iterator.

        Returns:
            Total item count
        """
        count = 0
        for _ in self.dataset:
            count += 1
        return count

    def to_list(self) -> List[Any]:
        """Convert to list (loads entire dataset into memory).

        Note: Defeats the purpose of streaming. Use only for small datasets.

        Returns:
            List of all items
        """
        return list(self.dataset)


class FilteredStreamingDataset(StreamingDataset):
    """Streaming dataset with filtering."""

    def __init__(self, dataset: Any, filter_fn: Any, batch_size: int = 1):
        """Initialize filtered streaming dataset.

        Args:
            dataset: Base dataset
            filter_fn: Predicate function for filtering
            batch_size: Batch size for processing
        """
        super().__init__(dataset, batch_size)
        self.filter_fn = filter_fn

    def __iter__(self) -> Generator[Any, None, None]:
        """Iterate over filtered items.

        Yields:
            Items where filter_fn(item) is True
        """
        for item in self.dataset:
            if self.filter_fn(item):
                yield item


class MappedStreamingDataset(StreamingDataset):
    """Streaming dataset with transformation."""

    def __init__(self, dataset: Any, map_fn: Any, batch_size: int = 1):
        """Initialize mapped streaming dataset.

        Args:
            dataset: Base dataset
            map_fn: Transformation function
            batch_size: Batch size for processing
        """
        super().__init__(dataset, batch_size)
        self.map_fn = map_fn

    def __iter__(self) -> Generator[Any, None, None]:
        """Iterate over transformed items.

        Yields:
            Transformed items
        """
        for item in self.dataset:
            yield self.map_fn(item)


class ChainedStreamingDataset(StreamingDataset):
    """Chain multiple datasets into single stream."""

    def __init__(self, datasets: List[Any], batch_size: int = 1):
        """Initialize chained streaming dataset.

        Args:
            datasets: List of datasets to chain
            batch_size: Batch size for processing
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.dataset = None  # type: ignore

    def __iter__(self) -> Generator[Any, None, None]:
        """Iterate over all datasets in sequence.

        Yields:
            Items from each dataset in order
        """
        for dataset in self.datasets:
            for item in dataset:
                yield item


def stream_dataset(dataset: Any, batch_size: int = 1) -> StreamingDataset:
    """Wrap dataset with streaming interface.

    Args:
        dataset: Dataset to stream
        batch_size: Batch size for processing

    Returns:
        StreamingDataset wrapper
    """
    return StreamingDataset(dataset, batch_size)


def filter_streaming(dataset: Any, filter_fn: Any) -> FilteredStreamingDataset:
    """Create filtered streaming dataset.

    Args:
        dataset: Base dataset
        filter_fn: Predicate function

    Returns:
        FilteredStreamingDataset
    """
    return FilteredStreamingDataset(dataset, filter_fn)


def map_streaming(dataset: Any, map_fn: Any) -> MappedStreamingDataset:
    """Create mapped streaming dataset.

    Args:
        dataset: Base dataset
        map_fn: Transformation function

    Returns:
        MappedStreamingDataset
    """
    return MappedStreamingDataset(dataset, map_fn)


def chain_streaming(datasets: List[Any]) -> ChainedStreamingDataset:
    """Chain multiple datasets.

    Args:
        datasets: List of datasets to chain

    Returns:
        ChainedStreamingDataset
    """
    return ChainedStreamingDataset(datasets)
