"""
Advanced Data Compression Module for OpenEval

This module provides advanced data compression techniques for efficient storage
and transmission of evaluation data, metrics, and results.
"""

import gzip
import bz2
import lzma
import zlib
import json
import pickle
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import defaultdict
import hashlib

try:
    import zstandard as zstd

    HAS_ZSTD = True
except ImportError:
    zstd = None
    HAS_ZSTD = False

try:
    import lz4.frame

    HAS_LZ4 = True
except ImportError:
    lz4 = None
    HAS_LZ4 = False

try:
    import snappy

    HAS_SNAPPY = True
except ImportError:
    snappy = None
    HAS_SNAPPY = False

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    original_size: int
    compressed_size: int
    compression_time: float
    decompression_time: float
    algorithm: str
    compression_ratio: float = field(init=False)
    compression_speed: float = field(init=False)  # MB/s
    decompression_speed: float = field(init=False)  # MB/s

    def __post_init__(self):
        self.compression_ratio = (
            self.original_size / self.compressed_size if self.compressed_size > 0 else 0
        )
        self.compression_speed = (
            (self.original_size / 1024 / 1024) / self.compression_time
            if self.compression_time > 0
            else 0
        )
        self.decompression_speed = (
            (self.original_size / 1024 / 1024) / self.decompression_time
            if self.decompression_time > 0
            else 0
        )


class CompressionAlgorithm(ABC):
    """Abstract base class for compression algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the compression algorithm."""
        pass

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        pass

    def benchmark(self, data: bytes, iterations: int = 5) -> CompressionResult:
        """Benchmark compression and decompression performance."""
        # Compress
        compressed = None
        compress_times = []
        for _ in range(iterations):
            start = time.time()
            compressed = self.compress(data)
            compress_times.append(time.time() - start)

        if compressed is None:
            raise RuntimeError("Compression failed")

        # Decompress
        decompress_times = []
        for _ in range(iterations):
            start = time.time()
            self.decompress(compressed)
            decompress_times.append(time.time() - start)

        return CompressionResult(
            original_size=len(data),
            compressed_size=len(compressed),
            compression_time=sum(compress_times) / len(compress_times),
            decompression_time=sum(decompress_times) / len(decompress_times),
            algorithm=self.name,
        )


class GzipCompression(CompressionAlgorithm):
    """Gzip compression algorithm."""

    def __init__(self, level: int = 6):
        self.level = level

    @property
    def name(self) -> str:
        return f"gzip-{self.level}"

    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=self.level)

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)


class Bzip2Compression(CompressionAlgorithm):
    """Bzip2 compression algorithm."""

    def __init__(self, level: int = 9):
        self.level = level

    @property
    def name(self) -> str:
        return f"bzip2-{self.level}"

    def compress(self, data: bytes) -> bytes:
        return bz2.compress(data, compresslevel=self.level)

    def decompress(self, data: bytes) -> bytes:
        return bz2.decompress(data)


class LzmaCompression(CompressionAlgorithm):
    """LZMA/XZ compression algorithm."""

    def __init__(self, preset: int = 6):
        self.preset = preset

    @property
    def name(self) -> str:
        return f"lzma-{self.preset}"

    def compress(self, data: bytes) -> bytes:
        return lzma.compress(data, preset=self.preset)

    def decompress(self, data: bytes) -> bytes:
        return lzma.decompress(data)


class ZlibCompression(CompressionAlgorithm):
    """Zlib compression algorithm."""

    def __init__(self, level: int = 6):
        self.level = level

    @property
    def name(self) -> str:
        return f"zlib-{self.level}"

    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data, level=self.level)

    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)


class ZstdCompression(CompressionAlgorithm):
    """Zstandard compression algorithm."""

    def __init__(self, level: int = 3):
        if not HAS_ZSTD:
            raise ImportError("zstandard not available")
        self.level = level

    @property
    def name(self) -> str:
        return f"zstd-{self.level}"

    def compress(self, data: bytes) -> bytes:
        if not HAS_ZSTD or zstd is None:
            raise ImportError("zstandard not available")
        ctx = zstd.ZstdCompressor(level=self.level)
        return ctx.compress(data)

    def decompress(self, data: bytes) -> bytes:
        if not HAS_ZSTD or zstd is None:
            raise ImportError("zstandard not available")
        ctx = zstd.ZstdDecompressor()
        return ctx.decompress(data)


class Lz4Compression(CompressionAlgorithm):
    """LZ4 compression algorithm."""

    @property
    def name(self) -> str:
        return "lz4"

    def compress(self, data: bytes) -> bytes:
        if not HAS_LZ4 or lz4 is None:
            raise ImportError("lz4 not available")
        return lz4.frame.compress(data)

    def decompress(self, data: bytes) -> bytes:
        if not HAS_LZ4 or lz4 is None:
            raise ImportError("lz4 not available")
        return lz4.frame.decompress(data)


class SnappyCompression(CompressionAlgorithm):
    """Snappy compression algorithm."""

    @property
    def name(self) -> str:
        return "snappy"

    def compress(self, data: bytes) -> bytes:
        if not HAS_SNAPPY or snappy is None:
            raise ImportError("snappy not available")
        return snappy.compress(data)

    def decompress(self, data: bytes) -> bytes:
        if not HAS_SNAPPY or snappy is None:
            raise ImportError("snappy not available")
        return snappy.decompress(data)


class AdaptiveCompressor:
    """Adaptive compressor that selects the best algorithm based on data characteristics."""

    def __init__(self):
        self.algorithms = self._initialize_algorithms()
        self.performance_cache: Dict[str, Dict[str, CompressionResult]] = defaultdict(dict)
        self.data_profiles: Dict[str, Dict[str, Any]] = {}

    def _initialize_algorithms(self) -> Dict[str, CompressionAlgorithm]:
        """Initialize available compression algorithms."""
        algorithms = {}

        # Always available algorithms
        algorithms["gzip-6"] = GzipCompression(6)
        algorithms["bzip2-9"] = Bzip2Compression(9)
        algorithms["lzma-6"] = LzmaCompression(6)
        algorithms["zlib-6"] = ZlibCompression(6)

        # Optional algorithms
        if HAS_ZSTD:
            algorithms["zstd-3"] = ZstdCompression(3)
        if HAS_LZ4:
            algorithms["lz4"] = Lz4Compression()
        if HAS_SNAPPY:
            algorithms["snappy"] = SnappyCompression()

        return algorithms

    def compress(
        self, data: Union[str, bytes, Dict, List], algorithm: Optional[str] = None
    ) -> Tuple[bytes, str, Dict[str, Any]]:
        """Compress data using the specified or best algorithm."""
        # Convert data to bytes
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
            data_type = "text"
        elif isinstance(data, (dict, list)):
            data_bytes = json.dumps(data, separators=(",", ":")).encode("utf-8")
            data_type = "json"
        elif isinstance(data, bytes):
            data_bytes = data
            data_type = "binary"
        else:
            data_bytes = pickle.dumps(data)
            data_type = "pickle"

        # Select algorithm
        if algorithm and algorithm in self.algorithms:
            selected_algorithm = self.algorithms[algorithm]
        else:
            selected_algorithm = self._select_best_algorithm(data_bytes, data_type)

        # Compress
        start_time = time.time()
        compressed = selected_algorithm.compress(data_bytes)
        compression_time = time.time() - start_time

        # Create metadata
        metadata = {
            "original_size": len(data_bytes),
            "compressed_size": len(compressed),
            "compression_ratio": len(data_bytes) / len(compressed) if len(compressed) > 0 else 0,
            "algorithm": selected_algorithm.name,
            "data_type": data_type,
            "compression_time": compression_time,
            "checksum": hashlib.md5(data_bytes).hexdigest(),
        }

        return compressed, selected_algorithm.name, metadata

    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> Any:
        """Decompress data using the specified algorithm."""
        algorithm_name = metadata.get("algorithm", "gzip-6")
        data_type = metadata.get("data_type", "binary")

        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown compression algorithm: {algorithm_name}")

        algorithm = self.algorithms[algorithm_name]

        # Decompress
        start_time = time.time()
        decompressed = algorithm.decompress(compressed_data)
        time.time() - start_time

        # Verify checksum if available
        if "checksum" in metadata:
            actual_checksum = hashlib.md5(decompressed).hexdigest()
            if actual_checksum != metadata["checksum"]:
                raise ValueError("Data integrity check failed")

        # Convert back to original type
        if data_type == "text":
            return decompressed.decode("utf-8")
        elif data_type == "json":
            return json.loads(decompressed.decode("utf-8"))
        elif data_type == "pickle":
            return pickle.loads(decompressed)
        else:
            return decompressed

    def _select_best_algorithm(self, data: bytes, data_type: str) -> CompressionAlgorithm:
        """Select the best compression algorithm based on data characteristics."""
        data_size = len(data)

        # For small data, use fast algorithms
        if data_size < 1024:  # < 1KB
            return self.algorithms.get("snappy", self.algorithms["gzip-6"])

        # For medium data, balance speed and compression
        elif data_size < 1024 * 1024:  # < 1MB
            if data_type == "text":
                return self.algorithms.get("zstd-3", self.algorithms["gzip-6"])
            else:
                return self.algorithms.get("lz4", self.algorithms["gzip-6"])

        # For large data, prioritize compression ratio
        else:
            if data_type == "text":
                return self.algorithms.get("bzip2-9", self.algorithms["gzip-6"])
            else:
                return self.algorithms.get("zstd-3", self.algorithms["lzma-6"])

    def benchmark_algorithms(
        self, test_data: bytes, algorithms: Optional[List[str]] = None
    ) -> Dict[str, CompressionResult]:
        """Benchmark compression algorithms on test data."""
        results = {}
        test_algorithms = algorithms or list(self.algorithms.keys())

        for alg_name in test_algorithms:
            if alg_name in self.algorithms:
                try:
                    result = self.algorithms[alg_name].benchmark(test_data)
                    results[alg_name] = result
                    logger.info(
                        f"Benchmarked {alg_name}: ratio={result.compression_ratio:.2f}, "
                        f"compress_speed={result.compression_speed:.1f} MB/s, "
                        f"decompress_speed={result.decompression_speed:.1f} MB/s"
                    )
                except Exception as e:
                    logger.warning(f"Failed to benchmark {alg_name}: {e}")

        return results

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            "available_algorithms": list(self.algorithms.keys()),
            "performance_cache_size": len(self.performance_cache),
            "data_profiles_count": len(self.data_profiles),
            "optional_algorithms": {"zstd": HAS_ZSTD, "lz4": HAS_LZ4, "snappy": HAS_SNAPPY},
        }


class CompressedStorage:
    """Storage system with automatic compression."""

    def __init__(self, base_path: str, compressor: Optional[AdaptiveCompressor] = None):
        self.base_path = base_path
        self.compressor = compressor or AdaptiveCompressor()
        self.index: Dict[str, Dict[str, Any]] = {}
        self._load_index()

    def store(self, key: str, data: Any, algorithm: Optional[str] = None) -> Dict[str, Any]:
        """Store data with compression."""
        compressed, alg_used, metadata = self.compressor.compress(data, algorithm)

        # Create file path
        file_path = f"{self.base_path}/{key}.compressed"

        # Store compressed data
        with open(file_path, "wb") as f:
            f.write(compressed)

        # Update index
        metadata.update({"file_path": file_path, "stored_at": time.time(), "key": key})
        self.index[key] = metadata
        self._save_index()

        logger.info(
            f"Stored {key}: {metadata['original_size']} -> {metadata['compressed_size']} bytes "
            f"(ratio: {metadata['compression_ratio']:.2f})"
        )
        return metadata

    def retrieve(self, key: str) -> Any:
        """Retrieve and decompress data."""
        if key not in self.index:
            raise KeyError(f"Key not found: {key}")

        metadata = self.index[key]
        file_path = metadata["file_path"]

        # Read compressed data
        with open(file_path, "rb") as f:
            compressed_data = f.read()

        # Decompress
        return self.compressor.decompress(compressed_data, metadata)

    def delete(self, key: str) -> None:
        """Delete stored data."""
        if key not in self.index:
            return

        metadata = self.index[key]
        file_path = metadata["file_path"]

        # Delete file
        try:
            import os

            os.remove(file_path)
        except FileNotFoundError:
            pass

        # Update index
        del self.index[key]
        self._save_index()

    def list_keys(self) -> List[str]:
        """List all stored keys."""
        return list(self.index.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_original = sum(meta["original_size"] for meta in self.index.values())
        total_compressed = sum(meta["compressed_size"] for meta in self.index.values())
        total_savings = total_original - total_compressed

        return {
            "total_keys": len(self.index),
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "total_space_savings": total_savings,
            "average_compression_ratio": (
                total_original / total_compressed if total_compressed > 0 else 0
            ),
            "algorithms_used": list(set(meta["algorithm"] for meta in self.index.values())),
        }

    def _load_index(self) -> None:
        """Load storage index from disk."""
        index_path = f"{self.base_path}/storage_index.json"
        try:
            with open(index_path, "r") as f:
                self.index = json.load(f)
        except FileNotFoundError:
            self.index = {}

    def _save_index(self) -> None:
        """Save storage index to disk."""
        import os

        os.makedirs(self.base_path, exist_ok=True)

        index_path = f"{self.base_path}/storage_index.json"
        with open(index_path, "w") as f:
            json.dump(self.index, f, indent=2)


class CompressionManager:
    """Main compression manager coordinating all compression operations."""

    def __init__(self):
        self.compressor = AdaptiveCompressor()
        self.storage = None
        self.compression_stats: Dict[str, Any] = defaultdict(int)

    def initialize_storage(self, base_path: str) -> None:
        """Initialize compressed storage."""
        self.storage = CompressedStorage(base_path, self.compressor)

    def compress_data(
        self, data: Any, algorithm: Optional[str] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress data with optional algorithm selection."""
        compressed, alg_used, metadata = self.compressor.compress(data, algorithm)

        # Update stats
        self.compression_stats["total_compressions"] += 1
        self.compression_stats[f"algorithm_{alg_used}"] += 1
        self.compression_stats["bytes_original"] += metadata["original_size"]
        self.compression_stats["bytes_compressed"] += metadata["compressed_size"]

        return compressed, metadata

    def decompress_data(self, compressed_data: bytes, metadata: Dict[str, Any]) -> Any:
        """Decompress data."""
        self.compression_stats["total_decompressions"] += 1
        return self.compressor.decompress(compressed_data, metadata)

    def benchmark_all_algorithms(self, test_data: bytes) -> Dict[str, CompressionResult]:
        """Benchmark all available compression algorithms."""
        return self.compressor.benchmark_algorithms(test_data)

    def get_compression_report(self) -> Dict[str, Any]:
        """Generate compression performance report."""
        stats = dict(self.compression_stats)

        if stats.get("bytes_original", 0) > 0:
            stats["overall_compression_ratio"] = stats["bytes_original"] / stats["bytes_compressed"]
            stats["total_space_savings"] = stats["bytes_original"] - stats["bytes_compressed"]

        if self.storage:
            stats["storage_stats"] = self.storage.get_stats()

        stats["available_algorithms"] = list(self.compressor.algorithms.keys())
        stats["compressor_stats"] = self.compressor.get_compression_stats()

        return stats
