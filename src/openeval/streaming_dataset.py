"""
Memory-Efficient Dataset Streaming for OpenEval Lab

This module provides memory-efficient dataset streaming capabilities that enable
processing of large datasets without loading everything into memory at once.
"""

from __future__ import annotations

import json
import gzip
import bz2
import lzma
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Union, Callable, TextIO
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import mmap
import io
import csv
from collections import deque

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from .enhanced_logging import get_logger
from .core import Example, Dataset

logger = get_logger(__name__)


class CompressionType:
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


@dataclass
class StreamingConfig:
    """Configuration for streaming datasets."""
    chunk_size: int = 1000  # Number of examples per chunk
    max_memory_mb: int = 100  # Maximum memory usage in MB
    compression: str = CompressionType.NONE
    buffer_size: int = 8192  # Buffer size for I/O operations
    prefetch_chunks: int = 2  # Number of chunks to prefetch
    cache_chunks: bool = True  # Whether to cache processed chunks


class StreamProcessor(ABC):
    """Abstract base class for stream processors."""

    @abstractmethod
    def process_line(self, line: str) -> Optional[Example]:
        """Process a single line into an Example."""
        pass

    @abstractmethod
    def validate_format(self, sample_lines: List[str]) -> bool:
        """Validate that the format is correct."""
        pass


class JSONLProcessor(StreamProcessor):
    """Processor for JSONL format."""

    def process_line(self, line: str) -> Optional[Example]:
        """Process a JSONL line."""
        line = line.strip()
        if not line:
            return None

        try:
            data = json.loads(line)
            return Example(
                id=str(data.get('id', '')),
                input=data.get('input', ''),
                reference=data.get('reference', ''),
                meta=data.get('meta', {})
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse JSONL line: {e}")
            return None

    def validate_format(self, sample_lines: List[str]) -> bool:
        """Validate JSONL format."""
        if not sample_lines:
            return False

        valid_count = 0
        for line in sample_lines[:5]:  # Check first 5 lines
            if self.process_line(line) is not None:
                valid_count += 1

        return valid_count >= len(sample_lines) * 0.8  # 80% valid


class CSVProcessor(StreamProcessor):
    """Processor for CSV format."""

    def __init__(self, delimiter: str = ',', has_header: bool = True):
        self.delimiter = delimiter
        self.has_header = has_header
        self.fieldnames = None

    def process_line(self, line: str) -> Optional[Example]:
        """Process a CSV line."""
        if not self.fieldnames:
            return None

        try:
            reader = csv.DictReader([line], fieldnames=self.fieldnames, delimiter=self.delimiter)
            row = next(reader)

            return Example(
                id=str(row.get('id', '')),
                input=row.get('input', ''),
                reference=row.get('reference', ''),
                meta={k: v for k, v in row.items() if k not in ['id', 'input', 'reference']}
            )
        except Exception as e:
            logger.warning(f"Failed to parse CSV line: {e}")
            return None

    def validate_format(self, sample_lines: List[str]) -> bool:
        """Validate CSV format."""
        if not sample_lines:
            return False

        try:
            # Try to detect header
            if self.has_header:
                self.fieldnames = sample_lines[0].strip().split(self.delimiter)
                data_lines = sample_lines[1:]
            else:
                # Assume standard fieldnames
                self.fieldnames = ['id', 'input', 'reference']
                data_lines = sample_lines

            valid_count = 0
            for line in data_lines[:3]:  # Check first 3 data lines
                if self.process_line(line) is not None:
                    valid_count += 1

            return valid_count >= len(data_lines) * 0.8

        except Exception:
            return False


class StreamingDataset(Dataset):
    """
    Memory-efficient streaming dataset that processes data on-demand.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        processor: StreamProcessor,
        config: Optional[StreamingConfig] = None,
        transform: Optional[Callable[[Example], Example]] = None
    ):
        self.file_path = Path(file_path)
        self.processor = processor
        self.config = config or StreamingConfig()
        self.transform = transform
        self._file_size = None
        self._estimated_count = None
        self._chunk_cache = {}

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Validate format with sample
        self._validate_dataset()

    def _validate_dataset(self) -> None:
        """Validate the dataset format."""
        sample_lines = self._read_sample_lines()
        if not self.processor.validate_format(sample_lines):
            raise ValueError(f"Invalid dataset format for file: {self.file_path}")

    def _read_sample_lines(self, num_lines: int = 10) -> List[str]:
        """Read sample lines for validation."""
        lines = []
        with self._open_file() as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                lines.append(line)
        return lines

    def _open_file(self) -> TextIO:
        """Open file with appropriate compression handling."""
        if self.config.compression == CompressionType.GZIP:
            return gzip.open(self.file_path, 'rt', encoding='utf-8')
        elif self.config.compression == CompressionType.BZIP2:
            return bz2.open(self.file_path, 'rt', encoding='utf-8')
        elif self.config.compression == CompressionType.LZMA:
            return lzma.open(self.file_path, 'rt', encoding='utf-8')
        else:
            return open(self.file_path, 'r', encoding='utf-8')

    def _get_file_size(self) -> int:
        """Get file size in bytes."""
        if self._file_size is None:
            self._file_size = self.file_path.stat().st_size
        return self._file_size

    def _estimate_count(self) -> int:
        """Estimate the number of examples in the dataset."""
        if self._estimated_count is not None:
            return self._estimated_count

        # Sample a portion of the file to estimate
        sample_size = min(100 * 1024, self._get_file_size() // 10)  # Sample 10% or 100KB
        if sample_size == 0:
            return 0

        sample_lines = []
        bytes_read = 0

        with self._open_file() as f:
            for line in f:
                sample_lines.append(line)
                bytes_read += len(line.encode('utf-8'))
                if bytes_read >= sample_size:
                    break

        if not sample_lines:
            return 0

        # Estimate total lines
        avg_line_size = bytes_read / len(sample_lines)
        total_estimated = int(self._get_file_size() / avg_line_size)

        # Validate by checking a few examples
        valid_examples = 0
        for line in sample_lines[:10]:
            if self.processor.process_line(line) is not None:
                valid_examples += 1

        if valid_examples < len(sample_lines) * 0.5:  # Less than 50% valid
            logger.warning(f"Low validity rate in sample: {valid_examples}/{len(sample_lines)}")

        self._estimated_count = total_estimated
        return total_estimated

    def __len__(self) -> int:
        """Get the estimated number of examples."""
        return self._estimate_count()

    def __iter__(self) -> Iterator[Example]:
        """Iterate through examples efficiently."""
        chunk_size = self.config.chunk_size
        chunk_cache = self._chunk_cache if self.config.cache_chunks else {}

        with self._open_file() as f:
            chunk = []
            chunk_idx = 0

            for line_num, line in enumerate(f):
                example = self.processor.process_line(line)
                if example is not None:
                    # Apply transformation if provided
                    if self.transform:
                        example = self.transform(example)

                    chunk.append(example)

                    # Yield chunk when it reaches the target size
                    if len(chunk) >= chunk_size:
                        if self.config.cache_chunks:
                            chunk_cache[chunk_idx] = chunk.copy()

                        yield from chunk
                        chunk = []
                        chunk_idx += 1

            # Yield remaining examples
            if chunk:
                if self.config.cache_chunks:
                    chunk_cache[chunk_idx] = chunk.copy()
                yield from chunk

    def get_chunk(self, chunk_idx: int) -> List[Example]:
        """Get a specific chunk of examples."""
        if self.config.cache_chunks and chunk_idx in self._chunk_cache:
            return self._chunk_cache[chunk_idx]

        chunk = []
        chunk_size = self.config.chunk_size
        start_idx = chunk_idx * chunk_size

        with self._open_file() as f:
            for line_num, line in enumerate(f):
                if line_num < start_idx:
                    continue

                example = self.processor.process_line(line)
                if example is not None:
                    if self.transform:
                        example = self.transform(example)
                    chunk.append(example)

                    if len(chunk) >= chunk_size:
                        break

        if self.config.cache_chunks:
            self._chunk_cache[chunk_idx] = chunk

        return chunk

    def iter_chunks(self) -> Iterator[List[Example]]:
        """Iterate through chunks of examples."""
        chunk_size = self.config.chunk_size

        with self._open_file() as f:
            chunk = []

            for line in f:
                example = self.processor.process_line(line)
                if example is not None:
                    if self.transform:
                        example = self.transform(example)
                    chunk.append(example)

                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []

            if chunk:
                yield chunk

    def sample(self, n: int, seed: Optional[int] = None) -> List[Example]:
        """Sample n examples from the dataset."""
        import random
        if seed is not None:
            random.seed(seed)

        examples = []
        total_count = len(self)

        if n >= total_count:
            # Return all examples
            examples = list(self)
        else:
            # Reservoir sampling for large datasets
            examples = []
            for i, example in enumerate(self):
                if len(examples) < n:
                    examples.append(example)
                else:
                    j = random.randint(0, i)
                    if j < n:
                        examples[j] = example

        return examples

    def filter(self, predicate: Callable[[Example], bool]) -> StreamingDataset:
        """Create a filtered version of the dataset."""
        def transform_filter(example: Example) -> Optional[Example]:
            if predicate(example):
                return self.transform(example) if self.transform else example
            return None

        # Create a new processor that applies the filter
        class FilteredProcessor(StreamProcessor):
            def __init__(self, base_processor: StreamProcessor, predicate: Callable[[Example], bool]):
                self.base_processor = base_processor
                self.predicate = predicate

            def process_line(self, line: str) -> Optional[Example]:
                example = self.base_processor.process_line(line)
                if example and self.predicate(example):
                    return example
                return None

            def validate_format(self, sample_lines: List[str]) -> bool:
                return self.base_processor.validate_format(sample_lines)

        return StreamingDataset(
            self.file_path,
            FilteredProcessor(self.processor, predicate),
            self.config
        )

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        cache_size = len(self._chunk_cache) if self.config.cache_chunks else 0
        cache_memory = sum(len(chunk) * 1000 for chunk in self._chunk_cache.values())  # Rough estimate

        return {
            "file_size_mb": self._get_file_size() / (1024 * 1024),
            "estimated_examples": self._estimate_count(),
            "cached_chunks": cache_size,
            "estimated_cache_memory_mb": cache_memory / (1024 * 1024),
            "chunk_size": self.config.chunk_size,
            "compression": self.config.compression
        }


class MemoryEfficientDatasetIterator:
    """
    Memory-efficient iterator that can handle very large datasets.
    """

    def __init__(self, dataset: StreamingDataset, buffer_size: int = 10000):
        self.dataset = dataset
        self.buffer_size = buffer_size
        self._buffer = deque(maxlen=buffer_size)
        self._buffer_start = 0
        self._exhausted = False

    def __iter__(self):
        return self

    def __next__(self) -> Example:
        if not self._buffer and not self._exhausted:
            self._fill_buffer()

        if not self._buffer:
            raise StopIteration

        return self._buffer.popleft()

    def _fill_buffer(self) -> None:
        """Fill the buffer with more examples."""
        try:
            chunk = self.dataset.get_chunk(self._buffer_start)
            if chunk:
                self._buffer.extend(chunk)
                self._buffer_start += 1
            else:
                self._exhausted = True
        except Exception as e:
            logger.error(f"Error filling buffer: {e}")
            self._exhausted = True

    def peek(self) -> Optional[Example]:
        """Peek at the next example without consuming it."""
        if not self._buffer and not self._exhausted:
            self._fill_buffer()

        return self._buffer[0] if self._buffer else None


# Utility functions for creating streaming datasets
def create_jsonl_streaming_dataset(
    file_path: Union[str, Path],
    config: Optional[StreamingConfig] = None
) -> StreamingDataset:
    """Create a streaming dataset from a JSONL file."""
    return StreamingDataset(file_path, JSONLProcessor(), config)


def create_csv_streaming_dataset(
    file_path: Union[str, Path],
    delimiter: str = ',',
    has_header: bool = True,
    config: Optional[StreamingConfig] = None
) -> StreamingDataset:
    """Create a streaming dataset from a CSV file."""
    return StreamingDataset(file_path, CSVProcessor(delimiter, has_header), config)


def create_compressed_streaming_dataset(
    file_path: Union[str, Path],
    compression: str,
    format_type: str = 'jsonl',
    config: Optional[StreamingConfig] = None
) -> StreamingDataset:
    """Create a streaming dataset from a compressed file."""
    config = config or StreamingConfig()
    config.compression = compression

    if format_type == 'jsonl':
        processor = JSONLProcessor()
    elif format_type == 'csv':
        processor = CSVProcessor()
    else:
        raise ValueError(f"Unsupported format: {format_type}")

    return StreamingDataset(file_path, processor, config)


# Memory monitoring utilities
def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def monitor_memory_usage(func: Callable) -> Callable:
    """Decorator to monitor memory usage of a function."""
    def wrapper(*args, **kwargs):
        start_memory = get_memory_usage()
        peak_memory = start_memory

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_memory = get_memory_usage()
            logger.info(
                f"Memory usage - Start: {start_memory:.1f}MB, "
                f"End: {end_memory:.1f}MB, "
                f"Peak: {peak_memory:.1f}MB, "
                f"Delta: {end_memory - start_memory:.1f}MB"
            )

    return wrapper