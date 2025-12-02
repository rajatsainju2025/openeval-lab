"""
Memory-Optimized Dataset Processing

Provides streaming, lazy loading, and memory-efficient dataset operations
to handle large datasets without memory exhaustion.
"""

from __future__ import annotations

import json
import mmap
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union, Generator, Callable
from dataclasses import dataclass
import gc
import weakref

from .core import Dataset, Example
from .imports import HAS_PANDAS, pandas
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryConfig:
    """Memory optimization configuration."""

    chunk_size: int = 1000  # Process data in chunks
    max_cache_size: int = 10000  # Maximum items to cache
    enable_streaming: bool = True  # Use streaming for large files
    enable_mmap: bool = True  # Use memory mapping for files
    lazy_loading: bool = True  # Load data on demand
    gc_threshold: int = 500  # Trigger GC after processing N items


class StreamingDataset:
    """Memory-efficient streaming dataset that processes data lazily."""

    def __init__(
        self,
        path: Union[str, Path],
        config: Optional[MemoryConfig] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.path = Path(path)
        self.config = config or MemoryConfig()
        self.transform = transform
        self._file_size = self.path.stat().st_size
        self._cached_length: Optional[int] = None
        self._weak_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._processed_count = 0

    def __len__(self) -> int:
        """Get dataset length efficiently."""
        if self._cached_length is not None:
            return self._cached_length

        # Use fast line counting for JSONL files
        if self.path.suffix == ".jsonl":
            self._cached_length = self._count_lines_fast()
        else:
            self._cached_length = sum(1 for _ in self._stream_raw())

        return self._cached_length

    def _count_lines_fast(self) -> int:
        """Fast line counting using memory mapping."""
        if not self.config.enable_mmap:
            with open(self.path, "rb") as f:
                return sum(1 for _ in f)

        try:
            with open(self.path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return mm.read().count(b"\n")
        except (OSError, ValueError):
            # Fallback to regular counting
            with open(self.path, "rb") as f:
                return sum(1 for _ in f)

    def __iter__(self) -> Iterator[Example]:
        """Iterate over dataset with memory optimization."""
        try:
            if self.config.enable_streaming:
                yield from self._stream_optimized()
            else:
                yield from self._load_batched()
        finally:
            self._cleanup()

    def _stream_optimized(self) -> Generator[Example, None, None]:
        """Stream data with minimal memory footprint."""
        for chunk in self._stream_chunks():
            for item in chunk:
                if self.transform:
                    item = self.transform(item)

                example = self._to_example(item)
                yield example

                self._processed_count += 1
                if self._processed_count % self.config.gc_threshold == 0:
                    gc.collect()

    def _stream_chunks(self) -> Generator[List[Dict[str, Any]], None, None]:
        """Stream data in optimized chunks."""
        chunk = []

        for raw_item in self._stream_raw():
            chunk.append(raw_item)

            if len(chunk) >= self.config.chunk_size:
                yield chunk
                chunk = []

        # Yield final chunk
        if chunk:
            yield chunk

    def _stream_raw(self) -> Generator[Dict[str, Any], None, None]:
        """Stream raw data from file."""
        if self.path.suffix == ".jsonl":
            yield from self._stream_jsonl()
        elif self.path.suffix == ".json":
            yield from self._stream_json()
        elif self.path.suffix in (".csv", ".tsv"):
            yield from self._stream_csv()
        else:
            raise ValueError(f"Unsupported file format: {self.path.suffix}")

    def _stream_jsonl(self) -> Generator[Dict[str, Any], None, None]:
        """Stream JSONL file efficiently."""
        if self.config.enable_mmap and self._file_size < 100 * 1024 * 1024:  # 100MB threshold
            yield from self._stream_jsonl_mmap()
        else:
            yield from self._stream_jsonl_regular()

    def _stream_jsonl_mmap(self) -> Generator[Dict[str, Any], None, None]:
        """Stream JSONL using memory mapping."""
        try:
            with open(self.path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    for line in iter(mm.readline, b""):
                        line = line.decode("utf-8").strip()
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Skipping invalid JSON line: {e}")
        except (OSError, ValueError):
            # Fallback to regular streaming
            yield from self._stream_jsonl_regular()

    def _stream_jsonl_regular(self) -> Generator[Dict[str, Any], None, None]:
        """Stream JSONL using regular file I/O."""
        with open(self.path, "r", encoding="utf-8", buffering=8192) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

    def _stream_json(self) -> Generator[Dict[str, Any], None, None]:
        """Stream JSON file."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            yield from data
        else:
            yield data

    def _stream_csv(self) -> Generator[Dict[str, Any], None, None]:
        """Stream CSV file using pandas if available."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for CSV processing")

        # Read in chunks for memory efficiency
        chunk_size = min(self.config.chunk_size, 10000)

        try:
            for chunk in pandas.read_csv(
                self.path, chunksize=chunk_size, encoding="utf-8", low_memory=True
            ):
                # Convert to dict records
                for record in chunk.to_dict("records"):
                    yield record
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise

    def _load_batched(self) -> Generator[Example, None, None]:
        """Load data in batches for memory efficiency."""
        batch = []

        for raw_item in self._stream_raw():
            batch.append(raw_item)

            if len(batch) >= self.config.chunk_size:
                yield from self._process_batch(batch)
                batch = []
                gc.collect()

        # Process final batch
        if batch:
            yield from self._process_batch(batch)

    def _process_batch(self, batch: List[Dict[str, Any]]) -> Generator[Example, None, None]:
        """Process a batch of items."""
        for item in batch:
            if self.transform:
                item = self.transform(item)
            yield self._to_example(item)

    def _to_example(self, item: Dict[str, Any]) -> Example:
        """Convert dict to Example object."""
        # Flexible field mapping
        input_field = item.get("input") or item.get("question") or item.get("prompt", "")
        reference_field = item.get("reference") or item.get("answer") or item.get("target", "")

        return Example(
            id=item.get("id", f"example_{hash(str(item))}"),
            input=input_field,
            reference=reference_field,
            meta=item.get("metadata", {}),
        )

    def _cleanup(self):
        """Clean up resources."""
        self._weak_cache.clear()
        if self._processed_count > 0:
            gc.collect()
            self._processed_count = 0

    def slice(self, start: int, stop: Optional[int] = None) -> "StreamingDataset":
        """Create a sliced view of the dataset."""
        return SlicedStreamingDataset(self, start, stop)

    def batch(self, batch_size: int) -> Generator[List[Example], None, None]:
        """Yield batches of examples."""
        batch = []
        for example in self:
            batch.append(example)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


class SlicedStreamingDataset(StreamingDataset):
    """Memory-efficient sliced dataset view."""

    def __init__(self, parent: StreamingDataset, start: int, stop: Optional[int] = None):
        self.parent = parent
        self.start = start
        self.stop = stop
        self.config = parent.config

    def __len__(self) -> int:
        parent_len = len(self.parent)
        stop = self.stop if self.stop is not None else parent_len
        return max(0, min(stop, parent_len) - self.start)

    def __iter__(self) -> Iterator[Example]:
        """Iterate over slice efficiently."""
        for i, example in enumerate(self.parent):
            if i < self.start:
                continue
            if self.stop is not None and i >= self.stop:
                break
            yield example


class MemoryOptimizedDataset(Dataset):
    """Drop-in replacement for Dataset with memory optimizations."""

    def __init__(
        self,
        path: Union[str, Path],
        config: Optional[MemoryConfig] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.streaming_dataset = StreamingDataset(path, config, transform)
        self._examples: Optional[List[Example]] = None
        self.config = config or MemoryConfig()

    def __len__(self) -> int:
        return len(self.streaming_dataset)

    def __iter__(self) -> Iterator[Example]:
        if self.config.lazy_loading:
            yield from self.streaming_dataset
        else:
            if self._examples is None:
                self._examples = list(self.streaming_dataset)
            yield from self._examples

    def __getitem__(self, index: Union[int, slice]) -> Union[Example, List[Example]]:
        """Support indexing and slicing."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                # For stepped slices, we need to load everything
                if self._examples is None:
                    self._examples = list(self.streaming_dataset)
                return self._examples[index]
            else:
                # For simple slices, use streaming
                return list(self.streaming_dataset.slice(start, stop))
        else:
            # For single item access, use streaming with skip
            if index < 0:
                index += len(self)

            for i, example in enumerate(self.streaming_dataset):
                if i == index:
                    return example

            raise IndexError("Dataset index out of range")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get dataset metadata efficiently."""
        return {
            "size": len(self),
            "path": str(self.streaming_dataset.path),
            "file_size_mb": self.streaming_dataset._file_size / (1024 * 1024),
            "config": {
                "chunk_size": self.config.chunk_size,
                "streaming_enabled": self.config.enable_streaming,
                "lazy_loading": self.config.lazy_loading,
            },
        }

    def sample(self, n: int, random_state: Optional[int] = None) -> List[Example]:
        """Sample n examples efficiently."""
        if random_state is not None:
            import random

            random.seed(random_state)

        total_size = len(self)
        if n >= total_size:
            return list(self)

        # Use reservoir sampling for memory efficiency
        reservoir = []
        for i, example in enumerate(self):
            if len(reservoir) < n:
                reservoir.append(example)
            else:
                import random

                j = random.randint(0, i)
                if j < n:
                    reservoir[j] = example

        return reservoir


# Factory function for easy dataset creation
def create_memory_optimized_dataset(
    path: Union[str, Path],
    chunk_size: int = 1000,
    enable_streaming: bool = True,
    lazy_loading: bool = True,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> MemoryOptimizedDataset:
    """Create a memory-optimized dataset with specified configuration."""
    config = MemoryConfig(
        chunk_size=chunk_size,
        enable_streaming=enable_streaming,
        lazy_loading=lazy_loading,
    )
    return MemoryOptimizedDataset(path, config, transform)


__all__ = [
    "MemoryConfig",
    "StreamingDataset",
    "MemoryOptimizedDataset",
    "create_memory_optimized_dataset",
]
