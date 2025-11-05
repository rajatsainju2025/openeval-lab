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
import asyncio
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Union, Callable, TextIO, AsyncIterator
from dataclasses import dataclass
from abc import ABC, abstractmethod
import csv
from collections import deque
from concurrent.futures import ThreadPoolExecutor

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

from .logging import get_logger
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
    adaptive_batching: bool = True  # Enable adaptive batch sizing
    parallel_processing: bool = True  # Enable parallel chunk processing
    max_workers: Optional[int] = None  # Max workers for parallel processing
    progress_tracking: bool = True  # Enable progress tracking
    memory_pressure_threshold: float = 0.8  # Memory pressure threshold (0-1)


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
                id=str(data.get("id", "")),
                input=data.get("input", ""),
                reference=data.get("reference", ""),
                meta=data.get("meta", {}),
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

    def __init__(self, delimiter: str = ",", has_header: bool = True):
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
                id=str(row.get("id", "")),
                input=row.get("input", ""),
                reference=row.get("reference", ""),
                meta={k: v for k, v in row.items() if k not in ["id", "input", "reference"]},
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
                self.fieldnames = ["id", "input", "reference"]
                data_lines = sample_lines

            valid_count = 0
            for line in data_lines[:3]:  # Check first 3 data lines
                if self.process_line(line) is not None:
                    valid_count += 1

            return valid_count >= len(data_lines) * 0.8

        except Exception:
            return False


class MemoryMonitor:
    """Monitor memory usage and pressure."""

    def __init__(self, pressure_threshold: float = 0.8):
        self.pressure_threshold = pressure_threshold
        self._last_memory_check = 0
        self._check_interval = 1.0  # Check every second

    def is_under_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        current_time = time.time()
        if current_time - self._last_memory_check < self._check_interval:
            return False

        self._last_memory_check = current_time
        memory_usage = get_memory_usage()

        # Get system memory info
        try:
            import psutil

            system_memory = psutil.virtual_memory()
            return (system_memory.percent / 100.0) > self.pressure_threshold
        except ImportError:
            # Fallback: check if usage exceeds threshold
            return memory_usage > (self.pressure_threshold * 1000)  # Rough estimate

    def get_optimal_chunk_size(self, base_size: int) -> int:
        """Get optimal chunk size based on memory pressure."""
        if self.is_under_pressure():
            return max(100, base_size // 2)  # Reduce chunk size
        return base_size


class IntelligentBatchSizer:
    """Intelligent batch sizer that adapts based on multiple factors."""

    def __init__(
        self,
        base_batch_size: int = 1000,
        min_batch_size: int = 10,
        max_batch_size: int = 10000,
        adaptation_interval: float = 30.0,  # Adapt every 30 seconds
        performance_window: int = 10,  # Keep last 10 measurements
    ):
        self.base_batch_size = base_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.adaptation_interval = adaptation_interval
        self.performance_window = performance_window

        self.current_batch_size = base_batch_size
        self.last_adaptation = time.time()
        self.performance_history: List[Dict[str, float]] = []
        self._lock = threading.Lock()

    def record_performance(
        self,
        batch_size: int,
        processing_time: float,
        throughput: float,
        memory_usage: float,
        cpu_usage: float,
    ) -> None:
        """Record performance metrics for a batch."""
        with self._lock:
            self.performance_history.append(
                {
                    "batch_size": batch_size,
                    "processing_time": processing_time,
                    "throughput": throughput,
                    "memory_usage": memory_usage,
                    "cpu_usage": cpu_usage,
                    "timestamp": time.time(),
                }
            )

            # Keep only recent history
            if len(self.performance_history) > self.performance_window:
                self.performance_history.pop(0)

    def get_optimal_batch_size(
        self,
        current_memory_pressure: float,
        current_cpu_usage: float,
        target_throughput: Optional[float] = None,
    ) -> int:
        """Calculate optimal batch size based on current conditions and history."""
        current_time = time.time()

        with self._lock:
            # Check if we should adapt
            if current_time - self.last_adaptation < self.adaptation_interval:
                return self.current_batch_size

            if not self.performance_history:
                return self.current_batch_size

            # Analyze performance history
            optimal_size = self._analyze_performance_history(
                current_memory_pressure, current_cpu_usage, target_throughput
            )

            # Apply bounds
            optimal_size = max(self.min_batch_size, min(self.max_batch_size, optimal_size))

            # Smooth transitions - don't change by more than 50% at once
            max_change = self.current_batch_size * 0.5
            if abs(optimal_size - self.current_batch_size) > max_change:
                if optimal_size > self.current_batch_size:
                    optimal_size = self.current_batch_size + max_change
                else:
                    optimal_size = self.current_batch_size - max_change

            self.current_batch_size = int(optimal_size)
            self.last_adaptation = current_time

            return self.current_batch_size

    def _analyze_performance_history(
        self,
        memory_pressure: float,
        cpu_usage: float,
        target_throughput: Optional[float],
    ) -> float:
        """Analyze performance history to find optimal batch size."""
        if not self.performance_history:
            return self.base_batch_size

        # Weight recent performance more heavily
        weights = [
            i / len(self.performance_history) for i in range(1, len(self.performance_history) + 1)
        ]

        # Calculate weighted averages
        total_weight = sum(weights)
        avg_throughput = (
            sum(p["throughput"] * w for p, w in zip(self.performance_history, weights))
            / total_weight
        )
        # avg_memory and avg_cpu calculated but not used directly in this version

        # Find best performing batch sizes
        best_sizes = sorted(
            self.performance_history,
            key=lambda x: x["throughput"]
            / (x["processing_time"] * (1 + x["memory_usage"] + x["cpu_usage"])),
            reverse=True,
        )[
            :3
        ]  # Top 3 performers

        # Calculate optimal size based on current conditions
        optimal_size = self.base_batch_size

        # Memory pressure adjustment
        if memory_pressure > 0.8:
            optimal_size *= 0.5  # Reduce batch size under high memory pressure
        elif memory_pressure < 0.3:
            optimal_size *= 1.5  # Increase batch size when memory is plentiful

        # CPU usage adjustment
        if cpu_usage > 0.9:
            optimal_size *= 0.7  # Reduce batch size under high CPU usage
        elif cpu_usage < 0.5:
            optimal_size *= 1.3  # Increase batch size when CPU is underutilized

        # Throughput target adjustment
        if target_throughput and avg_throughput > 0:
            throughput_ratio = target_throughput / avg_throughput
            optimal_size *= throughput_ratio**0.5  # Square root for smoother adjustment

        # Bias toward historically good sizes
        if best_sizes:
            historical_optimal = sum(s["batch_size"] for s in best_sizes) / len(best_sizes)
            optimal_size = (optimal_size + historical_optimal) / 2  # Average with historical

        return optimal_size


class ResourceAwareBatcher:
    """Resource-aware batch processor that optimizes for multiple constraints."""

    def __init__(
        self,
        batch_sizer: IntelligentBatchSizer,
        memory_monitor: MemoryMonitor,
        max_concurrent_batches: int = 4,
    ):
        self.batch_sizer = batch_sizer
        self.memory_monitor = memory_monitor
        self.max_concurrent_batches = max_concurrent_batches

        self._active_batches = 0
        self._batch_semaphore = threading.Semaphore(max_concurrent_batches)
        self._performance_stats: Dict[str, List[float]] = {
            "throughput": [],
            "latency": [],
            "memory_usage": [],
            "cpu_usage": [],
        }

    async def process_batches(
        self,
        items: List[Any],
        processor_func: Callable[[List[Any]], Any],
        target_throughput: Optional[float] = None,
    ) -> List[Any]:
        """Process items in optimally-sized batches."""
        results = []
        batch_start_times: Dict[int, float] = {}

        i = 0
        while i < len(items):
            # Get current resource usage
            memory_pressure = self._get_memory_pressure()
            cpu_usage = self._get_cpu_usage()

            # Calculate optimal batch size
            batch_size = self.batch_sizer.get_optimal_batch_size(
                memory_pressure, cpu_usage, target_throughput
            )

            # Adjust for concurrency limits
            effective_batch_size = min(batch_size, len(items) - i)

            # Wait for available batch slot
            await asyncio.get_event_loop().run_in_executor(None, self._batch_semaphore.acquire)

            batch = items[i : i + effective_batch_size]
            batch_start_times[len(results)] = time.time()

            # Process batch asynchronously
            task = asyncio.create_task(
                self._process_single_batch(batch, processor_func, len(results), batch_start_times)
            )
            results.append(task)

            i += effective_batch_size

        # Wait for all batches to complete
        completed_results = []
        for task in results:
            result = await task
            completed_results.append(result)

        return completed_results

    async def _process_single_batch(
        self,
        batch: List[Any],
        processor_func: Callable[[List[Any]], Any],
        batch_idx: int,
        start_times: Dict[int, float],
    ) -> Any:
        """Process a single batch and record performance."""
        try:
            start_time = start_times[batch_idx]

            # Process the batch
            result = processor_func(batch)

            batch_end = time.time()
            processing_time = batch_end - start_time

            # Record performance metrics
            throughput = len(batch) / processing_time if processing_time > 0 else 0
            memory_usage = self._get_memory_pressure()
            cpu_usage = self._get_cpu_usage()

            self.batch_sizer.record_performance(
                len(batch), processing_time, throughput, memory_usage, cpu_usage
            )

            # Update stats
            self._update_performance_stats(throughput, processing_time, memory_usage, cpu_usage)

            return result

        finally:
            self._batch_semaphore.release()

    def _get_memory_pressure(self) -> float:
        """Get current memory pressure (0-1)."""
        try:
            import psutil

            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            return self.memory_monitor.is_under_pressure() * 0.8  # Rough estimate

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage (0-1)."""
        try:
            import psutil

            return psutil.cpu_percent(interval=0.1) / 100.0
        except ImportError:
            return 0.5  # Default estimate

    def _update_performance_stats(
        self,
        throughput: float,
        latency: float,
        memory_usage: float,
        cpu_usage: float,
    ) -> None:
        """Update rolling performance statistics."""
        max_samples = 100

        for key, value in [
            ("throughput", throughput),
            ("latency", latency),
            ("memory_usage", memory_usage),
            ("cpu_usage", cpu_usage),
        ]:
            self._performance_stats[key].append(value)
            if len(self._performance_stats[key]) > max_samples:
                self._performance_stats[key].pop(0)

    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of recent performance metrics."""
        summary = {}
        for key, values in self._performance_stats.items():
            if values:
                summary[f"avg_{key}"] = sum(values) / len(values)
                summary[f"max_{key}"] = max(values)
                summary[f"min_{key}"] = min(values)
            else:
                summary[f"avg_{key}"] = 0.0
                summary[f"max_{key}"] = 0.0
                summary[f"min_{key}"] = 0.0

        return summary


class ProgressTracker:
    """Track progress of streaming operations."""

    def __init__(self):
        self.total_examples = 0
        self.processed_examples = 0
        self.start_time = time.time()
        self._last_report = 0
        self._report_interval = 5.0  # Report every 5 seconds

    def update(self, count: int = 1) -> None:
        """Update progress counter."""
        self.processed_examples += count
        self._maybe_report()

    def set_total(self, total: int) -> None:
        """Set total expected examples."""
        self.total_examples = total

    def _maybe_report(self) -> None:
        """Report progress if enough time has passed."""
        current_time = time.time()
        if current_time - self._last_report >= self._report_interval:
            self._report_progress()
            self._last_report = current_time

    def _report_progress(self) -> None:
        """Report current progress."""
        elapsed = time.time() - self.start_time
        if self.total_examples > 0:
            percentage = (self.processed_examples / self.total_examples) * 100
            rate = self.processed_examples / elapsed if elapsed > 0 else 0
            eta = (self.total_examples - self.processed_examples) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {self.processed_examples}/{self.total_examples} "
                f"({percentage:.1f}%) - {rate:.1f} examples/sec - ETA: {eta:.1f}s"
            )
        else:
            rate = self.processed_examples / elapsed if elapsed > 0 else 0
            logger.info(
                f"Progress: {self.processed_examples} examples processed "
                f"- {rate:.1f} examples/sec"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get progress statistics."""
        elapsed = time.time() - self.start_time
        return {
            "processed": self.processed_examples,
            "total": self.total_examples,
            "elapsed_seconds": elapsed,
            "rate_per_second": self.processed_examples / elapsed if elapsed > 0 else 0,
        }


class StreamingDataset(Dataset):
    """
    Memory-efficient streaming dataset that processes data on-demand.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        processor: StreamProcessor,
        config: Optional[StreamingConfig] = None,
        transform: Optional[Callable[[Example], Example]] = None,
    ):
        self.file_path = Path(file_path)
        self.processor = processor
        self.config = config or StreamingConfig()
        self.transform = transform
        self._file_size = None
        self._estimated_count = None
        self._chunk_cache = {}
        self._adaptive_chunk_size = self.config.chunk_size
        self._memory_monitor = MemoryMonitor(self.config.memory_pressure_threshold)
        self._progress_tracker = ProgressTracker() if self.config.progress_tracking else None
        self._thread_pool = (
            ThreadPoolExecutor(max_workers=self.config.max_workers)
            if self.config.parallel_processing
            else None
        )

        # Initialize intelligent batching components
        self._intelligent_batch_sizer = IntelligentBatchSizer(
            base_batch_size=self.config.chunk_size,
            min_batch_size=10,
            max_batch_size=self.config.chunk_size * 10,
        )
        self._resource_aware_batcher = ResourceAwareBatcher(
            self._intelligent_batch_sizer,
            self._memory_monitor,
            max_concurrent_batches=self.config.max_workers or 4,
        )

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
            return gzip.open(self.file_path, "rt", encoding="utf-8")
        elif self.config.compression == CompressionType.BZIP2:
            return bz2.open(self.file_path, "rt", encoding="utf-8")
        elif self.config.compression == CompressionType.LZMA:
            return lzma.open(self.file_path, "rt", encoding="utf-8")
        else:
            return open(self.file_path, "r", encoding="utf-8")

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
                bytes_read += len(line.encode("utf-8"))
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
        chunk_size = self._get_adaptive_chunk_size()
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

                    # Update progress
                    if self._progress_tracker:
                        self._progress_tracker.update()

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

    async def __aiter__(self) -> AsyncIterator[Example]:
        """Async iterate through examples."""
        chunk_size = self._get_adaptive_chunk_size()

        # Use thread pool for file I/O
        loop = asyncio.get_event_loop()

        with self._open_file() as f:
            chunk = []
            lines_iter = iter(f)

            while True:
                # Read lines asynchronously
                try:
                    line = await loop.run_in_executor(None, next, lines_iter)
                except StopIteration:
                    break

                example = self.processor.process_line(line)
                if example is not None:
                    if self.transform:
                        example = self.transform(example)

                    chunk.append(example)

                    if self._progress_tracker:
                        self._progress_tracker.update()

                    if len(chunk) >= chunk_size:
                        for ex in chunk:
                            yield ex
                        chunk = []

            # Yield remaining examples
            for ex in chunk:
                yield ex

    def _get_adaptive_chunk_size(self) -> int:
        """Get adaptive chunk size based on memory pressure."""
        if not self.config.adaptive_batching:
            return self.config.chunk_size

        return self._memory_monitor.get_optimal_chunk_size(self._adaptive_chunk_size)

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
        chunk_size = self._get_adaptive_chunk_size()

        with self._open_file() as f:
            chunk = []

            for line in f:
                example = self.processor.process_line(line)
                if example is not None:
                    if self.transform:
                        example = self.transform(example)
                    chunk.append(example)

                    if self._progress_tracker:
                        self._progress_tracker.update()

                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []

            if chunk:
                yield chunk

    def iter_chunks_parallel(self) -> Iterator[List[Example]]:
        """Iterate through chunks with parallel processing."""
        if not self.config.parallel_processing or self._thread_pool is None:
            yield from self.iter_chunks()
            return

        chunk_size = self._get_adaptive_chunk_size()
        futures = []

        with self._open_file() as f:
            chunk = []

            for line in f:
                example = self.processor.process_line(line)
                if example is not None:
                    if self.transform:
                        example = self.transform(example)
                    chunk.append(example)

                    if self._progress_tracker:
                        self._progress_tracker.update()

                    if len(chunk) >= chunk_size:
                        # Submit chunk for parallel processing
                        future = self._thread_pool.submit(
                            self._process_chunk_parallel, chunk.copy()
                        )
                        futures.append(future)
                        chunk = []

                        # Yield completed chunks
                        for completed_future in [f for f in futures if f.done()]:
                            yield completed_future.result()
                            futures.remove(completed_future)

            # Process remaining chunk
            if chunk:
                future = self._thread_pool.submit(self._process_chunk_parallel, chunk)
                futures.append(future)

            # Yield all remaining results
            for future in futures:
                yield future.result()

    def _process_chunk_parallel(self, chunk: List[Example]) -> List[Example]:
        """Process a chunk in parallel (placeholder for additional processing)."""
        # Update progress
        if self._progress_tracker:
            self._progress_tracker.update(len(chunk))

        return chunk

    async def evaluate_with_streaming_async(
        self,
        evaluator_func: Callable[[List[Example]], Any],
        batch_size: Optional[int] = None,
        parallel: bool = True,
        target_throughput: Optional[float] = None,
    ) -> AsyncIterator[Any]:
        """Evaluate dataset using streaming with intelligent batching."""
        # Collect all examples first (in a real implementation, this would be streaming)
        all_examples = []
        if parallel and self.config.parallel_processing and self._thread_pool:
            chunk_iter = self.iter_chunks_parallel()
        else:
            chunk_iter = self.iter_chunks()

        for chunk in chunk_iter:
            all_examples.extend(chunk)

        # Use intelligent batching for processing
        batch_results = await self._resource_aware_batcher.process_batches(
            all_examples, evaluator_func, target_throughput
        )

        for result in batch_results:
            yield result

    def evaluate_with_streaming(
        self,
        evaluator_func: Callable[[List[Example]], Any],
        batch_size: Optional[int] = None,
        parallel: bool = True,
    ) -> Iterator[Any]:
        """Evaluate dataset using streaming with automatic batching."""
        batch_size = batch_size or self._get_adaptive_chunk_size()

        if parallel and self.config.parallel_processing and self._thread_pool:
            chunk_iter = self.iter_chunks_parallel()
        else:
            chunk_iter = self.iter_chunks()

        for chunk in chunk_iter:
            if len(chunk) > 0:
                # Adaptive batching: split large chunks if needed
                for i in range(0, len(chunk), batch_size):
                    batch = chunk[i : i + batch_size]
                    yield evaluator_func(batch)

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
            def __init__(
                self, base_processor: StreamProcessor, predicate: Callable[[Example], bool]
            ):
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
            self.file_path, FilteredProcessor(self.processor, predicate), self.config
        )

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        cache_size = len(self._chunk_cache) if self.config.cache_chunks else 0
        cache_memory = sum(
            len(chunk) * 1000 for chunk in self._chunk_cache.values()
        )  # Rough estimate

        return {
            "file_size_mb": self._get_file_size() / (1024 * 1024),
            "estimated_examples": self._estimate_count(),
            "cached_chunks": cache_size,
            "estimated_cache_memory_mb": cache_memory / (1024 * 1024),
            "chunk_size": self.config.chunk_size,
            "compression": self.config.compression,
        }

    def get_batching_performance_stats(self) -> Dict[str, Any]:
        """Get intelligent batching performance statistics."""
        batcher_stats = self._resource_aware_batcher.get_performance_summary()
        sizer_stats = {
            "current_batch_size": self._intelligent_batch_sizer.current_batch_size,
            "performance_history_size": len(self._intelligent_batch_sizer.performance_history),
            "last_adaptation": self._intelligent_batch_sizer.last_adaptation,
        }

        return {
            "batcher_stats": batcher_stats,
            "sizer_stats": sizer_stats,
            "adaptive_batching_enabled": self.config.adaptive_batching,
            "intelligent_batching_enabled": True,
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
    file_path: Union[str, Path], config: Optional[StreamingConfig] = None
) -> StreamingDataset:
    """Create a streaming dataset from a JSONL file."""
    return StreamingDataset(file_path, JSONLProcessor(), config)


def create_csv_streaming_dataset(
    file_path: Union[str, Path],
    delimiter: str = ",",
    has_header: bool = True,
    config: Optional[StreamingConfig] = None,
) -> StreamingDataset:
    """Create a streaming dataset from a CSV file."""
    return StreamingDataset(file_path, CSVProcessor(delimiter, has_header), config)


def create_compressed_streaming_dataset(
    file_path: Union[str, Path],
    compression: str,
    format_type: str = "jsonl",
    config: Optional[StreamingConfig] = None,
) -> StreamingDataset:
    """Create a streaming dataset from a compressed file."""
    config = config or StreamingConfig()
    config.compression = compression

    if format_type == "jsonl":
        processor = JSONLProcessor()
    elif format_type == "csv":
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
