"""Common constants used throughout OpenEval."""

from __future__ import annotations

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_SEED",
    "DEFAULT_TIMEOUT",
    "SUPPORTED_DATASET_FORMATS",
    "SUPPORTED_SPEC_FORMATS",
]

# Default directories and paths
DEFAULT_CACHE_DIR = ".cache"
DEFAULT_RUNS_DIR = "runs"
DEFAULT_ARTIFACTS_DIR = "artifacts"

# Default evaluation parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42
DEFAULT_TIMEOUT = 300  # 5 minutes

# Supported formats
SUPPORTED_DATASET_FORMATS = ["jsonl", "csv", "json", "parquet", "hf"]
SUPPORTED_SPEC_FORMATS = ["json", "yaml", "yml"]

# Metric computation
DEFAULT_PRECISION = 4  # Decimal places for metrics

# Caching
DEFAULT_CACHE_TTL = 86400  # 24 hours in seconds
DEFAULT_MAX_CACHE_SIZE_MB = 1000  # 1 GB

# Logging
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance
DEFAULT_MAX_WORKERS = 4
DEFAULT_CHUNK_SIZE = 100

# API and Web
DEFAULT_API_PORT = 8000
DEFAULT_API_HOST = "0.0.0.0"

# Version conventions
VERSION_PATTERN = r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+)?$"
