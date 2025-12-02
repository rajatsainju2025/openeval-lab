"""
Unified Validation System for OpenEval Lab

Consolidates all validation functionality into a single, efficient module.
Provides dataset validation, spec validation, schema validation with caching.
"""

from __future__ import annotations

import json
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Tuple

from .core import Dataset, Example
from .logging import get_logger

logger = get_logger(__name__)

# Try importing optional validation dependencies
try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

    # Create mock jsonschema for type checking
    class MockJsonSchema:
        class ValidationError(Exception):
            def __init__(self, message, absolute_path=None):
                self.message = message
                self.absolute_path = absolute_path or []

        class SchemaError(Exception):
            def __init__(self, message):
                self.message = message

        @staticmethod
        def validate(data, schema):
            pass

    jsonschema = MockJsonSchema()

try:
    from pydantic import BaseModel, ValidationError

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    ValidationError = Exception


class ValidationLevel(Enum):
    """Validation strictness levels."""

    STRICT = "strict"  # All validations, fail on warnings
    STANDARD = "standard"  # Standard validations, log warnings
    LENIENT = "lenient"  # Only critical validations


class ValidationSeverity(Enum):
    """Validation issue severity."""

    CRITICAL = "critical"  # Must fix
    ERROR = "error"  # Should fix
    WARNING = "warning"  # Nice to fix
    INFO = "info"  # Informational


@dataclass
class ValidationIssue:
    """Single validation issue."""

    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation operation."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def errors(self) -> List[str]:
        """Get error messages."""
        return [
            issue.message
            for issue in self.issues
            if issue.severity in (ValidationSeverity.CRITICAL, ValidationSeverity.ERROR)
        ]

    @property
    def warnings(self) -> List[str]:
        """Get warning messages."""
        return [
            issue.message for issue in self.issues if issue.severity == ValidationSeverity.WARNING
        ]

    def add_issue(self, severity: ValidationSeverity, message: str, **kwargs):
        """Add validation issue."""
        issue = ValidationIssue(severity=severity, message=message, **kwargs)
        self.issues.append(issue)
        if severity in (ValidationSeverity.CRITICAL, ValidationSeverity.ERROR):
            self.is_valid = False

    def __bool__(self) -> bool:
        return self.is_valid


class BaseValidator(ABC):
    """Base validator interface."""

    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        self.level = level

    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate data and return result."""
        pass

    def __call__(self, data: Any, **kwargs) -> ValidationResult:
        return self.validate(data, **kwargs)


@dataclass
class DatasetQualityReport:
    """Comprehensive dataset quality report."""

    total_examples: int
    valid_examples: int
    invalid_examples: int
    avg_input_length: float
    avg_reference_length: float
    unique_inputs: int
    unique_references: int
    duplicate_pairs: int
    empty_inputs: int
    empty_references: int
    encoding_issues: int
    format_issues: List[str]
    quality_score: float
    recommendations: List[str]
    validation_time_ms: float = 0.0


class DatasetValidator(BaseValidator):
    """Efficient dataset validator with quality assessment."""

    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD, strict: bool = False):
        super().__init__(level)
        self.strict = strict
        self._input_cache: Set[str] = set()
        self._reference_cache: Set[str] = set()

    def validate(self, data: Dataset, **kwargs) -> ValidationResult:
        """Validate dataset comprehensively."""
        start_time = time.perf_counter()
        result = ValidationResult(is_valid=True)

        try:
            # Validate dataset structure
            self._validate_dataset_structure(data, result)

            # Validate examples
            quality_report = self._validate_examples(data, result)
            result.metadata["quality_report"] = quality_report

            # Assess overall quality
            if quality_report.quality_score < 0.7:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Dataset quality score {quality_report.quality_score:.2f} is below recommended threshold of 0.7",
                )

        except Exception as e:
            result.add_issue(ValidationSeverity.CRITICAL, f"Dataset validation failed: {str(e)}")

        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _validate_dataset_structure(self, dataset: Dataset, result: ValidationResult):
        """Validate basic dataset structure."""
        try:
            # Check if dataset is iterable
            examples = list(dataset)
            if not examples:
                result.add_issue(ValidationSeverity.ERROR, "Dataset is empty")
                return

            result.metadata["example_count"] = len(examples)

        except Exception as e:
            result.add_issue(ValidationSeverity.CRITICAL, f"Cannot iterate dataset: {str(e)}")

    def _validate_examples(
        self, dataset: Dataset, result: ValidationResult
    ) -> DatasetQualityReport:
        """Validate individual examples and generate quality report."""
        total_examples = 0
        valid_examples = 0
        input_lengths = []
        reference_lengths = []
        unique_inputs = set()
        unique_references = set()
        duplicate_pairs = 0
        empty_inputs = 0
        empty_references = 0
        encoding_issues = 0
        format_issues = []

        for i, example in enumerate(dataset):
            total_examples += 1

            # Validate example structure
            is_valid = True

            if not hasattr(example, "input") or not hasattr(example, "reference"):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Example {i} missing required 'input' or 'reference' attributes",
                )
                is_valid = False
                continue

            input_str = str(example.input)
            reference_str = str(example.reference)

            # Check for empty inputs/references
            if not input_str.strip():
                empty_inputs += 1
                result.add_issue(ValidationSeverity.WARNING, f"Example {i} has empty input")
                is_valid = False

            if not reference_str.strip():
                empty_references += 1
                result.add_issue(ValidationSeverity.WARNING, f"Example {i} has empty reference")
                is_valid = False

            # Check encoding
            try:
                input_str.encode("utf-8")
                reference_str.encode("utf-8")
            except UnicodeEncodeError:
                encoding_issues += 1
                result.add_issue(ValidationSeverity.ERROR, f"Example {i} has encoding issues")
                is_valid = False

            # Track lengths
            input_lengths.append(len(input_str))
            reference_lengths.append(len(reference_str))

            # Track uniqueness
            input_hash = hashlib.md5(input_str.encode()).hexdigest()
            ref_hash = hashlib.md5(reference_str.encode()).hexdigest()
            pair_hash = f"{input_hash}:{ref_hash}"

            unique_inputs.add(input_hash)
            unique_references.add(ref_hash)

            # Check for duplicates
            if pair_hash in self._input_cache:
                duplicate_pairs += 1
                result.add_issue(ValidationSeverity.WARNING, f"Example {i} is duplicate")
            else:
                self._input_cache.add(pair_hash)

            if is_valid:
                valid_examples += 1

        # Calculate quality metrics
        avg_input_length = sum(input_lengths) / len(input_lengths) if input_lengths else 0
        avg_reference_length = (
            sum(reference_lengths) / len(reference_lengths) if reference_lengths else 0
        )

        # Calculate quality score (0-1 scale)
        quality_factors = [
            valid_examples / max(total_examples, 1),  # Valid ratio
            1.0 - (empty_inputs + empty_references) / max(total_examples * 2, 1),  # Non-empty ratio
            1.0 - encoding_issues / max(total_examples, 1),  # Encoding ratio
            min(len(unique_inputs) / max(total_examples, 1), 1.0),  # Input diversity
        ]
        quality_score = sum(quality_factors) / len(quality_factors)

        # Generate recommendations
        recommendations = []
        if empty_inputs > 0:
            recommendations.append(f"Remove {empty_inputs} examples with empty inputs")
        if empty_references > 0:
            recommendations.append(f"Remove {empty_references} examples with empty references")
        if encoding_issues > 0:
            recommendations.append(f"Fix {encoding_issues} examples with encoding issues")
        if duplicate_pairs > 0:
            recommendations.append(f"Remove {duplicate_pairs} duplicate examples")
        if len(unique_inputs) / max(total_examples, 1) < 0.8:
            recommendations.append("Increase input diversity to improve dataset quality")

        return DatasetQualityReport(
            total_examples=total_examples,
            valid_examples=valid_examples,
            invalid_examples=total_examples - valid_examples,
            avg_input_length=avg_input_length,
            avg_reference_length=avg_reference_length,
            unique_inputs=len(unique_inputs),
            unique_references=len(unique_references),
            duplicate_pairs=duplicate_pairs,
            empty_inputs=empty_inputs,
            empty_references=empty_references,
            encoding_issues=encoding_issues,
            format_issues=format_issues,
            quality_score=quality_score,
            recommendations=recommendations,
        )

    def validate_example(self, example: Example) -> Tuple[bool, List[str]]:
        """Validate single example - backward compatibility method."""
        issues = []

        if not hasattr(example, "input") or not hasattr(example, "reference"):
            issues.append("Example missing required 'input' or 'reference' attributes")
            return False, issues

        input_str = str(example.input).strip()
        reference_str = str(example.reference).strip()

        if not input_str:
            issues.append("Empty input field")
        if not reference_str:
            issues.append("Empty reference field")

        # Check encoding
        try:
            input_str.encode("utf-8")
            reference_str.encode("utf-8")
        except UnicodeEncodeError:
            issues.append("Encoding issues detected")

        is_valid = len(issues) == 0
        return is_valid, issues

    def assess_quality(self, dataset: Dataset) -> DatasetQualityReport:
        """Assess dataset quality and return report."""
        result = self.validate(dataset)
        return result.metadata.get(
            "quality_report",
            DatasetQualityReport(
                total_examples=0,
                valid_examples=0,
                invalid_examples=0,
                avg_input_length=0,
                avg_reference_length=0,
                unique_inputs=0,
                unique_references=0,
                duplicate_pairs=0,
                empty_inputs=0,
                empty_references=0,
                encoding_issues=0,
                format_issues=[],
                quality_score=0.0,
                recommendations=[],
            ),
        )


class SchemaValidator(BaseValidator):
    """JSON Schema validator."""

    def __init__(self, schema: Dict[str, Any], level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(level)
        self.schema = schema

    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate data against JSON schema."""
        start_time = time.perf_counter()
        result = ValidationResult(is_valid=True)

        if not HAS_JSONSCHEMA:
            result.add_issue(
                ValidationSeverity.WARNING, "jsonschema not available, skipping schema validation"
            )
            return result

        try:
            jsonschema.validate(data, self.schema)
        except jsonschema.ValidationError as e:
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Schema validation failed: {e.message}",
                field=".".join(str(p) for p in e.absolute_path),
            )
        except jsonschema.SchemaError as e:
            result.add_issue(ValidationSeverity.CRITICAL, f"Invalid schema: {e.message}")

        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result


class SpecValidator(BaseValidator):
    """Evaluation specification validator."""

    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate evaluation specification with caching."""
        start_time = time.perf_counter()
        result = ValidationResult(is_valid=True)

        try:
            if isinstance(data, str):
                try:
                    spec_dict = json.loads(data)
                except json.JSONDecodeError as e:
                    result.add_issue(ValidationSeverity.ERROR, f"Invalid JSON: {str(e)}")
                    return result
            else:
                spec_dict = data

            # Validate required fields
            required_fields = ["task", "dataset", "model"]
            for field in required_fields:
                if field not in spec_dict:
                    result.add_issue(
                        ValidationSeverity.ERROR, f"Missing required field: {field}", field=field
                    )

            # Validate task configuration
            if "task" in spec_dict:
                self._validate_task_config(spec_dict["task"], result)

            # Validate dataset configuration
            if "dataset" in spec_dict:
                self._validate_dataset_config(spec_dict["dataset"], result)

        except Exception as e:
            result.add_issue(
                ValidationSeverity.CRITICAL, f"Specification validation failed: {str(e)}"
            )

        result.duration_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _validate_task_config(self, task_config: Any, result: ValidationResult):
        """Validate task configuration."""
        if isinstance(task_config, dict):
            if "name" not in task_config:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "Task configuration missing 'name' field",
                    field="task.name",
                )
        elif not isinstance(task_config, str):
            result.add_issue(
                ValidationSeverity.ERROR, "Task must be string or object", field="task"
            )

    def _validate_dataset_config(self, dataset_config: Any, result: ValidationResult):
        """Validate dataset configuration."""
        if not isinstance(dataset_config, dict):
            result.add_issue(
                ValidationSeverity.ERROR, "Dataset configuration must be object", field="dataset"
            )
            return

        if "path" not in dataset_config:
            result.add_issue(
                ValidationSeverity.ERROR,
                "Dataset configuration missing 'path' field",
                field="dataset.path",
            )
        elif not isinstance(dataset_config["path"], str):
            result.add_issue(
                ValidationSeverity.ERROR, "Dataset path must be string", field="dataset.path"
            )


# Validation factory functions
def create_dataset_validator(
    strict: bool = False, level: ValidationLevel = ValidationLevel.STANDARD
) -> DatasetValidator:
    """Create dataset validator instance."""
    return DatasetValidator(level=level, strict=strict)


def create_schema_validator(
    schema: Dict[str, Any], level: ValidationLevel = ValidationLevel.STANDARD
) -> SchemaValidator:
    """Create schema validator instance."""
    return SchemaValidator(schema=schema, level=level)


def create_spec_validator(level: ValidationLevel = ValidationLevel.STANDARD) -> SpecValidator:
    """Create spec validator instance."""
    return SpecValidator(level=level)


# Convenience functions
def validate_dataset(dataset: Dataset, strict: bool = False) -> ValidationResult:
    """Validate dataset with default settings."""
    validator = create_dataset_validator(strict=strict)
    return validator.validate(dataset)


def validate_spec(spec_data: Union[str, Dict[str, Any]]) -> ValidationResult:
    """Validate spec with default settings."""
    validator = create_spec_validator()
    if isinstance(spec_data, dict):
        spec_data = json.dumps(spec_data)
    return validator.validate(spec_data)


__all__ = [
    # Enums
    "ValidationLevel",
    "ValidationSeverity",
    # Data structures
    "ValidationIssue",
    "ValidationResult",
    "DatasetQualityReport",
    # Validators
    "BaseValidator",
    "DatasetValidator",
    "SchemaValidator",
    "SpecValidator",
    # Factory functions
    "create_dataset_validator",
    "create_schema_validator",
    "create_spec_validator",
    # Convenience functions
    "validate_dataset",
    "validate_spec",
]
