"""
Dataset Management and Validation System for OpenEval Lab

This module provides comprehensive dataset management, validation, preprocessing,
and quality assurance for evaluation datasets.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import random
import statistics
from collections import Counter, defaultdict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .enhanced_logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for a dataset."""
    name: str
    path: Path
    format: str  # json, jsonl, csv, tsv, etc.
    size: int  # Number of samples
    columns: List[str]
    data_types: Dict[str, str]
    checksum: str
    created_at: datetime
    last_modified: datetime
    statistics: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": str(self.path),
            "format": self.format,
            "size": self.size,
            "columns": self.columns,
            "data_types": self.data_types,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "statistics": self.statistics,
            "validation_results": self.validation_results
        }


@dataclass
class DatasetValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)

    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation."""
        self.recommendations.append(recommendation)

    def summary(self) -> str:
        """Get a summary of the validation result."""
        error_count = len(self.errors)
        warning_count = len(self.warnings)

        if self.is_valid and warning_count == 0:
            return "✅ Dataset validation passed with no issues."

        parts = []
        if error_count > 0:
            parts.append(f"❌ {error_count} error{'s' if error_count != 1 else ''}")
        if warning_count > 0:
            parts.append(f"⚠️ {warning_count} warning{'s' if warning_count != 1 else ''}")

        status = "❌ Invalid" if not self.is_valid else "⚠️ Valid with warnings"
        return f"{status} dataset ({', '.join(parts)})"


class DatasetValidator:
    """
    Comprehensive validator for evaluation datasets.
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    def validate_dataset(
        self,
        dataset_path: Union[str, Path],
        task_type: Optional[str] = None,
        expected_columns: Optional[List[str]] = None
    ) -> DatasetValidationResult:
        """
        Validate a dataset file.

        Args:
            dataset_path: Path to the dataset file
            task_type: Type of evaluation task (qa, code_eval, etc.)
            expected_columns: Expected column names

        Returns:
            DatasetValidationResult
        """
        result = DatasetValidationResult(is_valid=True)
        path = Path(dataset_path)

        if not path.exists():
            result.add_error(f"Dataset file does not exist: {path}")
            return result

        try:
            # Load dataset
            data = self._load_dataset(path)
            result.statistics["total_samples"] = len(data)

            if len(data) == 0:
                result.add_error("Dataset is empty")
                return result

            # Basic structure validation
            self._validate_basic_structure(data, result, expected_columns)

            # Task-specific validation
            if task_type:
                self._validate_task_specific(data, task_type, result)

            # Quality checks
            self._validate_data_quality(data, result)

            # Statistical analysis
            self._compute_statistics(data, result)

        except Exception as e:
            result.add_error(f"Failed to validate dataset: {e}")

        return result

    def _load_dataset(self, path: Path) -> List[Dict[str, Any]]:
        """Load dataset from file."""
        suffix = path.suffix.lower()

        if suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]

        elif suffix == '.jsonl':
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data

        elif suffix in ['.csv', '.tsv'] and HAS_PANDAS:
            sep = '\t' if suffix == '.tsv' else ','
            df = pd.read_csv(path, sep=sep)
            return [dict(row) for row in df.to_dict('records')]

        else:
            raise ValueError(f"Unsupported dataset format: {suffix}")

    def _validate_basic_structure(
        self,
        data: List[Dict[str, Any]],
        result: DatasetValidationResult,
        expected_columns: Optional[List[str]] = None
    ) -> None:
        """Validate basic dataset structure."""
        if not data:
            result.add_error("Dataset contains no samples")
            return

        # Check that all items are dictionaries
        non_dict_items = [i for i, item in enumerate(data) if not isinstance(item, dict)]
        if non_dict_items:
            result.add_error(f"Non-dictionary items found at indices: {non_dict_items[:10]}")

        # Get all unique keys
        all_keys = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())

        result.statistics["unique_columns"] = len(all_keys)
        result.statistics["all_columns"] = sorted(list(all_keys))

        # Check for expected columns
        if expected_columns:
            missing_columns = set(expected_columns) - all_keys
            if missing_columns:
                result.add_error(f"Missing expected columns: {missing_columns}")

        # Check for column consistency
        column_counts = Counter()
        for item in data:
            if isinstance(item, dict):
                column_counts[len(item)] += 1

        if len(column_counts) > 1:
            result.add_warning("Inconsistent number of columns across samples")
            result.statistics["column_distribution"] = dict(column_counts)

    def _validate_task_specific(
        self,
        data: List[Dict[str, Any]],
        task_type: str,
        result: DatasetValidationResult
    ) -> None:
        """Validate task-specific requirements."""
        task_validators = {
            "qa": self._validate_qa_dataset,
            "code_eval": self._validate_code_eval_dataset,
            "classification": self._validate_classification_dataset,
            "generation": self._validate_generation_dataset
        }

        validator = task_validators.get(task_type)
        if validator:
            validator(data, result)
        else:
            result.add_warning(f"Unknown task type: {task_type}")

    def _validate_qa_dataset(self, data: List[Dict[str, Any]], result: DatasetValidationResult) -> None:
        """Validate question-answering dataset."""
        required_fields = ["question", "answer"]
        suggested_fields = ["context", "explanation"]

        for i, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    result.add_error(f"Sample {i}: Missing required field '{field}'")
                elif not item[field] or str(item[field]).strip() == "":
                    result.add_error(f"Sample {i}: Empty '{field}' field")

            for field in suggested_fields:
                if field not in item:
                    result.add_warning(f"Sample {i}: Missing suggested field '{field}'")

    def _validate_code_eval_dataset(self, data: List[Dict[str, Any]], result: DatasetValidationResult) -> None:
        """Validate code evaluation dataset."""
        required_fields = ["code", "test_cases"]
        suggested_fields = ["language", "description", "expected_output"]

        for i, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    result.add_error(f"Sample {i}: Missing required field '{field}'")

            if "code" in item and not isinstance(item["code"], str):
                result.add_error(f"Sample {i}: 'code' field must be a string")

            if "test_cases" in item and not isinstance(item["test_cases"], list):
                result.add_error(f"Sample {i}: 'test_cases' field must be a list")

    def _validate_classification_dataset(self, data: List[Dict[str, Any]], result: DatasetValidationResult) -> None:
        """Validate classification dataset."""
        required_fields = ["text", "label"]

        for i, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    result.add_error(f"Sample {i}: Missing required field '{field}'")

    def _validate_generation_dataset(self, data: List[Dict[str, Any]], result: DatasetValidationResult) -> None:
        """Validate text generation dataset."""
        required_fields = ["input", "output"]

        for i, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    result.add_error(f"Sample {i}: Missing required field '{field}'")

    def _validate_data_quality(self, data: List[Dict[str, Any]], result: DatasetValidationResult) -> None:
        """Validate data quality aspects."""
        # Check for empty or null values
        empty_counts = defaultdict(int)
        total_samples = len(data)

        for item in data:
            for key, value in item.items():
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    empty_counts[key] += 1

        for field, count in empty_counts.items():
            percentage = (count / total_samples) * 100
            if percentage > 50:
                result.add_error(f"Field '{field}' has {percentage:.1f}% empty/null values")
            elif percentage > 10:
                result.add_warning(f"Field '{field}' has {percentage:.1f}% empty/null values")

        # Check for duplicate samples
        if len(data) > 1:
            # Simple duplicate detection based on string representation
            string_reps = [json.dumps(item, sort_keys=True) for item in data]
            duplicates = len(string_reps) - len(set(string_reps))

            if duplicates > 0:
                duplicate_percentage = (duplicates / len(data)) * 100
                if duplicate_percentage > 5:
                    result.add_error(f"{duplicate_percentage:.1f}% duplicate samples detected")
                else:
                    result.add_warning(f"{duplicate_percentage:.1f}% duplicate samples detected")

        # Text quality checks
        text_fields = ["question", "answer", "text", "input", "output", "context"]
        for item in data:
            for field in text_fields:
                if field in item and isinstance(item[field], str):
                    text = item[field]
                    # Check for very short texts
                    if len(text.strip()) < 5:
                        result.add_warning(f"Very short text in field '{field}': '{text}'")
                    # Check for excessive whitespace
                    if text != text.strip():
                        result.add_warning(f"Text in field '{field}' has leading/trailing whitespace")

    def _compute_statistics(self, data: List[Dict[str, Any]], result: DatasetValidationResult) -> None:
        """Compute dataset statistics."""
        stats = {}

        # Basic counts
        stats["total_samples"] = len(data)

        # Text length statistics
        text_lengths = defaultdict(list)
        for item in data:
            for key, value in item.items():
                if isinstance(value, str):
                    text_lengths[key].append(len(value))

        for field, lengths in text_lengths.items():
            if lengths:
                stats[f"{field}_length_stats"] = {
                    "min": min(lengths),
                    "max": max(lengths),
                    "mean": statistics.mean(lengths),
                    "median": statistics.median(lengths)
                }

        # Categorical field distributions
        categorical_fields = ["label", "language", "category"]
        for field in categorical_fields:
            values = [item.get(field) for item in data if field in item]
            if values:
                value_counts = Counter(values)
                stats[f"{field}_distribution"] = dict(value_counts.most_common(10))

        result.statistics.update(stats)


class DatasetManager:
    """
    Comprehensive dataset management system.
    """

    def __init__(self, datasets_dir: Optional[Path] = None):
        self.datasets_dir = datasets_dir or Path("datasets")
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.datasets_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.validator = DatasetValidator()

        # Cache for loaded datasets
        self._dataset_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._metadata_cache: Dict[str, DatasetMetadata] = {}

    def register_dataset(
        self,
        name: str,
        path: Union[str, Path],
        task_type: Optional[str] = None,
        validate: bool = True
    ) -> DatasetMetadata:
        """
        Register a dataset with the manager.

        Args:
            name: Dataset name
            path: Path to dataset file
            task_type: Evaluation task type
            validate: Whether to validate the dataset

        Returns:
            DatasetMetadata
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Validate if requested
        validation_result = None
        if validate:
            validation_result = self.validator.validate_dataset(path, task_type)
            if not validation_result.is_valid:
                logger.warning(f"Dataset validation failed: {validation_result.summary()}")

        # Compute checksum
        checksum = self._compute_checksum(path)

        # Create metadata
        metadata = DatasetMetadata(
            name=name,
            path=path,
            format=path.suffix[1:],  # Remove the dot
            size=self._get_dataset_size(path),
            columns=[],  # Will be filled during loading
            data_types={},
            checksum=checksum,
            created_at=datetime.now(),
            last_modified=datetime.fromtimestamp(path.stat().st_mtime)
        )

        # Load and analyze dataset for metadata
        try:
            data = self._load_dataset(path)
            metadata.size = len(data)
            metadata.columns = self._get_columns(data)
            metadata.data_types = self._infer_data_types(data)
            metadata.statistics = self._compute_basic_stats(data)

            if validation_result:
                metadata.validation_results = {
                    "is_valid": validation_result.is_valid,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                    "recommendations": validation_result.recommendations
                }

        except Exception as e:
            logger.error(f"Failed to analyze dataset {name}: {e}")

        # Save metadata
        self._save_metadata(metadata)

        # Cache metadata
        self._metadata_cache[name] = metadata

        logger.info(f"Registered dataset '{name}' with {metadata.size} samples")
        return metadata

    def load_dataset(
        self,
        name: str,
        cache: bool = True,
        validate: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Load a registered dataset.

        Args:
            name: Dataset name
            cache: Whether to use cached data
            validate: Whether to validate before loading

        Returns:
            Dataset samples
        """
        if cache and name in self._dataset_cache:
            return self._dataset_cache[name]

        metadata = self.get_metadata(name)
        if not metadata:
            raise ValueError(f"Dataset not found: {name}")

        # Validate if requested
        if validate:
            validation_result = self.validator.validate_dataset(metadata.path)
            if not validation_result.is_valid:
                raise ValueError(f"Dataset validation failed: {validation_result.summary()}")

        data = self._load_dataset(metadata.path)

        if cache:
            self._dataset_cache[name] = data

        return data

    def get_metadata(self, name: str) -> Optional[DatasetMetadata]:
        """Get metadata for a dataset."""
        if name in self._metadata_cache:
            return self._metadata_cache[name]

        metadata_file = self.metadata_dir / f"{name}.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                metadata = DatasetMetadata(**data)
                self._metadata_cache[name] = metadata
                return metadata
            except Exception as e:
                logger.error(f"Failed to load metadata for {name}: {e}")

        return None

    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        datasets = []
        for metadata_file in self.metadata_dir.glob("*.json"):
            datasets.append(metadata_file.stem)
        return sorted(datasets)

    def validate_dataset(
        self,
        name: str,
        task_type: Optional[str] = None
    ) -> DatasetValidationResult:
        """Validate a registered dataset."""
        metadata = self.get_metadata(name)
        if not metadata:
            result = DatasetValidationResult(is_valid=False)
            result.add_error(f"Dataset not found: {name}")
            return result

        return self.validator.validate_dataset(metadata.path, task_type)

    def split_dataset(
        self,
        name: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split a dataset into train/validation/test sets.

        Args:
            name: Dataset name
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        data = self.load_dataset(name)
        random.seed(seed)
        random.shuffle(data)

        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]

        logger.info(f"Split dataset '{name}': {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        return train_data, val_data, test_data

    def sample_dataset(
        self,
        name: str,
        n_samples: int,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Sample a subset of the dataset.

        Args:
            name: Dataset name
            n_samples: Number of samples to return
            seed: Random seed

        Returns:
            Sampled dataset
        """
        data = self.load_dataset(name)
        if n_samples >= len(data):
            return data

        random.seed(seed)
        return random.sample(data, n_samples)

    def _load_dataset(self, path: Path) -> List[Dict[str, Any]]:
        """Load dataset from file."""
        return self.validator._load_dataset(path)

    def _get_dataset_size(self, path: Path) -> int:
        """Get dataset size without loading all data."""
        suffix = path.suffix.lower()

        if suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return len(data) if isinstance(data, list) else 1

        elif suffix == '.jsonl':
            count = 0
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count

        elif suffix in ['.csv', '.tsv'] and HAS_PANDAS:
            sep = '\t' if suffix == '.tsv' else ','
            df = pd.read_csv(path, sep=sep)
            return len(df)

        return 0

    def _get_columns(self, data: List[Dict[str, Any]]) -> List[str]:
        """Get column names from dataset."""
        columns = set()
        for item in data:
            columns.update(item.keys())
        return sorted(list(columns))

    def _infer_data_types(self, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Infer data types for columns."""
        if not data:
            return {}

        data_types = {}
        sample_size = min(100, len(data))  # Sample first 100 items

        for col in self._get_columns(data):
            values = [item.get(col) for item in data[:sample_size] if col in item]
            if not values:
                data_types[col] = "unknown"
                continue

            # Infer type
            types = {type(v).__name__ for v in values if v is not None}
            if len(types) == 1:
                data_types[col] = list(types)[0]
            else:
                data_types[col] = "mixed"

        return data_types

    def _compute_basic_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute basic statistics."""
        stats: Dict[str, Any] = {"total_samples": len(data)}

        if not data:
            return stats

        # Text length stats
        text_cols = [col for col, dtype in self._infer_data_types(data).items() if dtype == "str"]
        for col in text_cols:
            lengths = [len(str(item.get(col, ""))) for item in data]
            if lengths:
                stats[f"{col}_avg_length"] = statistics.mean(lengths)

        return stats

    def _compute_checksum(self, path: Path) -> str:
        """Compute MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _save_metadata(self, metadata: DatasetMetadata) -> None:
        """Save metadata to file."""
        metadata_file = self.metadata_dir / f"{metadata.name}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)


def create_dataset_manager(datasets_dir: Optional[Path] = None) -> DatasetManager:
    """Create a dataset manager instance."""
    return DatasetManager(datasets_dir)


def validate_dataset_file(
    dataset_path: Union[str, Path],
    task_type: Optional[str] = None,
    expected_columns: Optional[List[str]] = None
) -> DatasetValidationResult:
    """
    Convenience function to validate a dataset file.

    Args:
        dataset_path: Path to dataset file
        task_type: Evaluation task type
        expected_columns: Expected column names

    Returns:
        DatasetValidationResult
    """
    validator = DatasetValidator()
    return validator.validate_dataset(dataset_path, task_type, expected_columns)