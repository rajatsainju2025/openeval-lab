"""
Configuration Validation System for OpenEval Lab

This module provides comprehensive validation for evaluation configurations,
ensuring they are correct, complete, and compatible before execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

try:
    from pydantic import BaseModel, ValidationError, Field, validator

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

    # Fallback for when pydantic is not available
    class BaseModel:
        pass

    ValidationError = Exception

from .enhanced_logging import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationCategory(Enum):
    """Categories of validation issues."""

    STRUCTURE = "structure"
    TYPE = "type"
    VALUE = "value"
    COMPATIBILITY = "compatibility"
    DEPENDENCY = "dependency"
    SECURITY = "security"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    severity: ValidationSeverity
    category: ValidationCategory
    field_path: str
    message: str
    suggestion: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "field_path": self.field_path,
            "message": self.message,
            "suggestion": self.suggestion,
            "context": self.context,
        }


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.errors.append(issue)
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def summary(self) -> str:
        """Get a summary of the validation result."""
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        total_issues = len(self.issues)

        if self.is_valid and total_issues == 0:
            return "✅ Configuration is valid with no issues."

        parts = []
        if error_count > 0:
            parts.append(f"❌ {error_count} error{'s' if error_count != 1 else ''}")
        if warning_count > 0:
            parts.append(f"⚠️ {warning_count} warning{'s' if warning_count != 1 else ''}")

        status = "❌ Invalid" if not self.is_valid else "⚠️ Valid with warnings"
        return f"{status} configuration ({', '.join(parts)})"


class ConfigurationValidator:
    """
    Comprehensive validator for OpenEval Lab configurations.
    """

    def __init__(self, schema_dir: Optional[Path] = None):
        self.schema_dir = schema_dir or Path(__file__).parent / "schemas"
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        self.known_tasks: Set[str] = set()
        self.known_metrics: Set[str] = set()
        self.known_models: Set[str] = set()
        self.known_datasets: Set[str] = set()

        # Load known components
        self._load_known_components()

    def _load_known_components(self) -> None:
        """Load information about known tasks, metrics, models, and datasets."""
        try:
            # Load from project structure
            tasks_dir = Path(__file__).parent / "tasks"
            if tasks_dir.exists():
                self.known_tasks = {
                    f.stem for f in tasks_dir.glob("*.py") if not f.stem.startswith("_")
                }

            metrics_dir = Path(__file__).parent / "metrics"
            if metrics_dir.exists():
                self.known_metrics = {
                    f.stem for f in metrics_dir.glob("*.py") if not f.stem.startswith("_")
                }

            # Load from examples
            examples_dir = Path(__file__).parent.parent / "examples"
            if examples_dir.exists():
                for example_file in examples_dir.glob("*.json"):
                    try:
                        with open(example_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if "task" in data:
                                self.known_tasks.add(data["task"])
                            if "metrics" in data and isinstance(data["metrics"], list):
                                self.known_metrics.update(data["metrics"])
                    except Exception:
                        pass

        except Exception as e:
            logger.warning(f"Failed to load known components: {e}")

    def validate_configuration(self, config: Union[Dict[str, Any], str, Path]) -> ValidationResult:
        """
        Validate an evaluation configuration.

        Args:
            config: Configuration dictionary, file path, or JSON string

        Returns:
            ValidationResult with issues found
        """
        # Load configuration if needed
        if isinstance(config, (str, Path)):
            config = self._load_config_file(config)
        elif isinstance(config, str):
            try:
                config = json.loads(config)
            except json.JSONDecodeError as e:
                result = ValidationResult(is_valid=False)
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.STRUCTURE,
                        field_path="root",
                        message=f"Invalid JSON format: {e}",
                        suggestion="Ensure the configuration is valid JSON",
                    )
                )
                return result

        # At this point config should be a dict
        assert isinstance(config, dict), "Config should be a dictionary after loading"
        result = ValidationResult(is_valid=True)

        # Basic structure validation
        self._validate_basic_structure(config, result)

        # Task validation
        if "task" in config:
            self._validate_task_config(config["task"], result)

        # Model validation
        if "model" in config:
            self._validate_model_config(config["model"], result)

        # Dataset validation
        if "dataset" in config:
            self._validate_dataset_config(config["dataset"], result)

        # Metrics validation
        if "metrics" in config:
            self._validate_metrics_config(config["metrics"], result)

        # Evaluation parameters validation
        if "evaluation" in config:
            self._validate_evaluation_config(config["evaluation"], result)

        # Advanced validation
        self._validate_cross_references(config, result)

        return result

    def _load_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    def _validate_basic_structure(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate basic configuration structure."""
        required_fields = ["task"]
        recommended_fields = ["model", "dataset", "metrics"]

        # Check required fields
        for field in required_fields:
            if field not in config:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.STRUCTURE,
                        field_path=field,
                        message=f"Required field '{field}' is missing",
                        suggestion=f"Add the '{field}' field to your configuration",
                    )
                )

        # Check recommended fields
        for field in recommended_fields:
            if field not in config:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.STRUCTURE,
                        field_path=field,
                        message=f"Recommended field '{field}' is missing",
                        suggestion=f"Consider adding the '{field}' field for better evaluation control",
                    )
                )

        # Check for unknown top-level fields
        known_fields = {
            "task",
            "model",
            "dataset",
            "metrics",
            "evaluation",
            "output",
            "logging",
            "version",
            "metadata",
        }
        for field in config.keys():
            if field not in known_fields:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.STRUCTURE,
                        field_path=field,
                        message=f"Unknown configuration field '{field}'",
                        suggestion="Check the documentation for supported configuration fields",
                    )
                )

    def _validate_task_config(self, task_config: Any, result: ValidationResult) -> None:
        """Validate task configuration."""
        if not isinstance(task_config, str):
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.TYPE,
                    field_path="task",
                    message="Task must be a string",
                    suggestion="Provide the task name as a string (e.g., 'qa', 'code_eval')",
                )
            )
            return

        # Check if task is known
        if task_config not in self.known_tasks:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.VALUE,
                    field_path="task",
                    message=f"Unknown task '{task_config}'",
                    suggestion=f"Known tasks include: {', '.join(sorted(self.known_tasks))}",
                )
            )

    def _validate_model_config(self, model_config: Any, result: ValidationResult) -> None:
        """Validate model configuration."""
        if not isinstance(model_config, dict):
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.TYPE,
                    field_path="model",
                    message="Model configuration must be an object",
                    suggestion="Provide model configuration as an object with 'name' and other properties",
                )
            )
            return

        # Check required model fields
        if "name" not in model_config:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.STRUCTURE,
                    field_path="model.name",
                    message="Model name is required",
                    suggestion="Add 'name' field to model configuration",
                )
            )

        # Validate model parameters
        if "parameters" in model_config:
            params = model_config["parameters"]
            if not isinstance(params, dict):
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.TYPE,
                        field_path="model.parameters",
                        message="Model parameters must be an object",
                        suggestion="Provide model parameters as key-value pairs",
                    )
                )

    def _validate_dataset_config(self, dataset_config: Any, result: ValidationResult) -> None:
        """Validate dataset configuration."""
        if not isinstance(dataset_config, dict):
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.TYPE,
                    field_path="dataset",
                    message="Dataset configuration must be an object",
                    suggestion="Provide dataset configuration as an object",
                )
            )
            return

        # Check for path or name
        if "path" not in dataset_config and "name" not in dataset_config:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.STRUCTURE,
                    field_path="dataset",
                    message="Dataset should have 'path' or 'name' field",
                    suggestion="Add 'path' field pointing to dataset file or 'name' for built-in datasets",
                )
            )

        # Validate path if provided
        if "path" in dataset_config:
            path = dataset_config["path"]
            if not isinstance(path, str):
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.TYPE,
                        field_path="dataset.path",
                        message="Dataset path must be a string",
                        suggestion="Provide dataset path as a string",
                    )
                )
            elif not Path(path).exists():
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.VALUE,
                        field_path="dataset.path",
                        message=f"Dataset path does not exist: {path}",
                        suggestion="Ensure the dataset file exists at the specified path",
                    )
                )

    def _validate_metrics_config(self, metrics_config: Any, result: ValidationResult) -> None:
        """Validate metrics configuration."""
        if not isinstance(metrics_config, list):
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.TYPE,
                    field_path="metrics",
                    message="Metrics must be a list",
                    suggestion="Provide metrics as a list of metric names or configuration objects",
                )
            )
            return

        for i, metric in enumerate(metrics_config):
            field_path = f"metrics[{i}]"

            if isinstance(metric, str):
                # Simple metric name
                if metric not in self.known_metrics:
                    result.add_issue(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category=ValidationCategory.VALUE,
                            field_path=field_path,
                            message=f"Unknown metric '{metric}'",
                            suggestion=f"Known metrics include: {', '.join(sorted(self.known_metrics))}",
                        )
                    )
            elif isinstance(metric, dict):
                # Metric configuration object
                if "name" not in metric:
                    result.add_issue(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category=ValidationCategory.STRUCTURE,
                            field_path=f"{field_path}.name",
                            message="Metric configuration must have 'name' field",
                            suggestion="Add 'name' field to metric configuration",
                        )
                    )
                elif metric["name"] not in self.known_metrics:
                    result.add_issue(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category=ValidationCategory.VALUE,
                            field_path=f"{field_path}.name",
                            message=f"Unknown metric '{metric['name']}'",
                            suggestion=f"Known metrics include: {', '.join(sorted(self.known_metrics))}",
                        )
                    )
            else:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.TYPE,
                        field_path=field_path,
                        message="Metric must be a string or object",
                        suggestion="Provide metric as a string name or configuration object",
                    )
                )

    def _validate_evaluation_config(self, eval_config: Any, result: ValidationResult) -> None:
        """Validate evaluation configuration."""
        if not isinstance(eval_config, dict):
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.TYPE,
                    field_path="evaluation",
                    message="Evaluation configuration must be an object",
                    suggestion="Provide evaluation configuration as an object",
                )
            )
            return

        # Validate batch size
        if "batch_size" in eval_config:
            batch_size = eval_config["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.VALUE,
                        field_path="evaluation.batch_size",
                        message="Batch size must be a positive integer",
                        suggestion="Set batch_size to a positive integer (e.g., 8, 16, 32)",
                    )
                )

        # Validate max_samples
        if "max_samples" in eval_config:
            max_samples = eval_config["max_samples"]
            if not isinstance(max_samples, int) or max_samples <= 0:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.VALUE,
                        field_path="evaluation.max_samples",
                        message="Max samples must be a positive integer",
                        suggestion="Set max_samples to a positive integer or remove for unlimited samples",
                    )
                )

    def _validate_cross_references(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate cross-references between configuration sections."""
        task = config.get("task")
        metrics = config.get("metrics", [])

        # Check metric-task compatibility
        if task and metrics:
            incompatible_metrics = self._check_metric_task_compatibility(task, metrics)
            for metric in incompatible_metrics:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.COMPATIBILITY,
                        field_path="metrics",
                        message=f"Metric '{metric}' may not be compatible with task '{task}'",
                        suggestion="Review metric-task compatibility in the documentation",
                    )
                )

    def _check_metric_task_compatibility(self, task: str, metrics: List[Any]) -> List[str]:
        """Check compatibility between task and metrics."""
        # This is a simplified compatibility check
        # In a real implementation, this would use a compatibility matrix
        incompatible = []

        task_metric_map = {
            "qa": ["exact_match", "f1", "bleu", "rouge"],
            "code_eval": ["pass_rate", "syntax_accuracy"],
            "classification": ["accuracy", "precision", "recall", "f1"],
            "generation": ["bleu", "rouge", "perplexity"],
        }

        compatible_metrics = task_metric_map.get(task, [])
        if compatible_metrics:
            for metric in metrics:
                metric_name = metric if isinstance(metric, str) else metric.get("name", "")
                if metric_name and metric_name not in compatible_metrics:
                    incompatible.append(metric_name)

        return incompatible

    def validate_file(self, config_path: Union[str, Path]) -> ValidationResult:
        """Validate a configuration file."""
        try:
            config = self._load_config_file(config_path)
            return self.validate_configuration(config)
        except Exception as e:
            result = ValidationResult(is_valid=False)
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.STRUCTURE,
                    field_path="file",
                    message=f"Failed to load configuration file: {e}",
                    suggestion="Ensure the file exists and is valid JSON",
                )
            )
            return result

    def validate_and_fix(
        self, config: Union[Dict[str, Any], str, Path], auto_fix: bool = False
    ) -> Tuple[ValidationResult, Optional[Dict[str, Any]]]:
        """
        Validate configuration and optionally apply automatic fixes.

        Args:
            config: Configuration to validate
            auto_fix: Whether to apply automatic fixes

        Returns:
            Tuple of (validation result, fixed config or None)
        """
        if isinstance(config, (str, Path)):
            config = self._load_config_file(config)

        result = self.validate_configuration(config)
        fixed_config = None

        if auto_fix and result.has_errors:
            fixed_config = self._apply_auto_fixes(dict(config), result.errors)

        return result, fixed_config

    def _apply_auto_fixes(
        self, config: Dict[str, Any], errors: List[ValidationIssue]
    ) -> Dict[str, Any]:
        """Apply automatic fixes for certain types of errors."""
        fixed_config = dict(config)

        for error in errors:
            # Apply fixes for known error patterns
            if error.field_path == "task" and "must be a string" in error.message:
                # Try to extract task from other fields
                pass  # Simplified for this example

        return fixed_config

    def get_validation_schema(self, task: Optional[str] = None) -> Dict[str, Any]:
        """Get JSON schema for configuration validation."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["task"],
            "properties": {
                "task": {"type": "string", "description": "The evaluation task to perform"},
                "model": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "parameters": {"type": "object"}},
                    "required": ["name"],
                },
                "dataset": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "name": {"type": "string"}},
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "parameters": {"type": "object"},
                                },
                                "required": ["name"],
                            },
                        ]
                    },
                },
                "evaluation": {
                    "type": "object",
                    "properties": {
                        "batch_size": {"type": "integer", "minimum": 1},
                        "max_samples": {"type": "integer", "minimum": 1},
                    },
                },
            },
        }

        return schema


def validate_config(
    config: Union[Dict[str, Any], str, Path], schema_dir: Optional[Path] = None
) -> ValidationResult:
    """
    Convenience function to validate a configuration.

    Args:
        config: Configuration to validate
        schema_dir: Directory containing validation schemas

    Returns:
        ValidationResult
    """
    validator = ConfigurationValidator(schema_dir)
    return validator.validate_configuration(config)


def validate_config_file(
    config_path: Union[str, Path], schema_dir: Optional[Path] = None
) -> ValidationResult:
    """
    Convenience function to validate a configuration file.

    Args:
        config_path: Path to configuration file
        schema_dir: Directory containing validation schemas

    Returns:
        ValidationResult
    """
    validator = ConfigurationValidator(schema_dir)
    return validator.validate_file(config_path)
