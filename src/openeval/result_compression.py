"""Result compression utilities.

Compresses large JSON results with gzip before writing.
"""

import gzip
import json
from typing import Any, Dict
from pathlib import Path


def compress_results(data: Dict[str, Any], output_path: str, compression_level: int = 9) -> str:
    """Compress results to gzip file."""
    json_str = json.dumps(data, indent=2)

    with gzip.open(output_path, "wt", compresslevel=compression_level) as f:
        f.write(json_str)

    return output_path


def decompress_results(input_path: str) -> Dict[str, Any]:
    """Decompress results from gzip file."""
    with gzip.open(input_path, "rt") as f:
        return json.load(f)


def should_compress(data: Dict[str, Any], size_threshold_kb: int = 100) -> bool:
    """Check if data should be compressed based on size."""
    json_str = json.dumps(data)
    size_kb = len(json_str) / 1024
    return size_kb > size_threshold_kb


def get_compression_ratio(original_path: str, compressed_path: str) -> float:
    """Calculate compression ratio."""
    orig_size = Path(original_path).stat().st_size
    comp_size = Path(compressed_path).stat().st_size
    if orig_size == 0:
        return 0.0
    return (1 - comp_size / orig_size) * 100


__all__ = ["compress_results", "decompress_results", "should_compress", "get_compression_ratio"]
