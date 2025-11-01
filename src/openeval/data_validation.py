"""Data validation system with schema enforcement and type checking."""

import json
import hashlib
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

try:
    from pydantic import BaseModel, ValidationError, Field

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    ValidationError = Exception

    def Field(**kwargs):
        return None


try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    jsonschema = None

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationRule:
    """A validation rule with name, description, and validation function."""

    name: str
    description: str
    validator: Callable[[Any], bool]
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class DataValidator:
    """Comprehensive data validation system."""

    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._validation_cache: Dict[str, ValidationResult] = {}

    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.rules[rule.name] = rule

    def add_schema(self, name: str, schema: Dict[str, Any]):
        """Add a JSON schema for validation."""
        self.schemas[name] = schema

    def validate_data(self, data: Any, rules: Optional[List[str]] = None) -> ValidationResult:
        """Validate data against specified rules."""
        # Create cache key from data and rules
        data_hash = hashlib.md5(str(data).encode()).hexdigest()
        rules_key = tuple(sorted(rules)) if rules else None
        cache_key = f"{data_hash}:{rules_key}"

        # Check cache first
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]

        errors = []
        warnings = []
        metadata = {}

        # Apply specified rules or all rules
        rules_to_apply = rules or list(self.rules.keys())

        for rule_name in rules_to_apply:
            if rule_name in self.rules:
                rule = self.rules[rule_name]
                try:
                    result = rule.validator(data)
                    if not result:
                        if rule.severity == "error":
                            errors.append(f"Rule '{rule_name}': {rule.description}")
                        elif rule.severity == "warning":
                            warnings.append(f"Rule '{rule_name}': {rule.description}")
                except Exception as e:
                    errors.append(f"Rule '{rule_name}' failed: {str(e)}")

        result = ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings, metadata=metadata
        )

        # Cache the result
        self._validation_cache[cache_key] = result
        return result

    def clear_cache(self):
        """Clear the validation cache."""
        self._validation_cache.clear()

    def validate_json_schema(self, data: Any, schema_name: str) -> ValidationResult:
        """Validate data against a JSON schema."""
        if not HAS_JSONSCHEMA or jsonschema is None:
            return ValidationResult(
                is_valid=False,
                errors=["JSON schema validation not available (jsonschema not installed)"],
                warnings=[],
                metadata={},
            )

        if schema_name not in self.schemas:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema '{schema_name}' not found"],
                warnings=[],
                metadata={},
            )

        try:
            jsonschema.validate(data, self.schemas[schema_name])
            return ValidationResult(
                is_valid=True, errors=[], warnings=[], metadata={"schema": schema_name}
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema validation failed: {str(e)}"],
                warnings=[],
                metadata={"schema": schema_name},
            )


class PydanticDataValidator:
    """Pydantic-based data validation."""

    def __init__(self):
        if not HAS_PYDANTIC:
            raise ImportError("Pydantic is required for PydanticDataValidator")

    def validate_model(self, model_class: Any, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against a Pydantic model."""
        try:
            model = model_class(**data)
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                metadata={
                    "model": getattr(model_class, "__name__", str(model_class)),
                    "validated_data": getattr(
                        model, "model_dump", getattr(model, "dict", lambda: data)
                    )(),
                },
            )
        except Exception as e:
            errors = [str(e)]
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=[],
                metadata={"model": getattr(model_class, "__name__", str(model_class))},
            )


# Predefined validation rules
def create_standard_validator() -> DataValidator:
    """Create a validator with standard rules."""

    validator = DataValidator()

    # Rule: Check if data is not empty
    validator.add_rule(
        ValidationRule(
            name="not_empty",
            description="Data should not be empty",
            validator=lambda data: data is not None and data != "" and data != [],
        )
    )

    # Rule: Check if data is valid JSON
    validator.add_rule(
        ValidationRule(
            name="valid_json",
            description="Data should be valid JSON",
            validator=lambda data: isinstance(data, (dict, list))
            or (isinstance(data, str) and _is_valid_json(data)),
        )
    )

    # Rule: Check for required fields in dict
    validator.add_rule(
        ValidationRule(
            name="has_required_fields",
            description="Dictionary should have required fields",
            validator=lambda data: isinstance(data, dict) and len(data) > 0,
        )
    )

    return validator


def _is_valid_json(data: str) -> bool:
    """Check if string is valid JSON."""
    try:
        json.loads(data)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


# Example Pydantic models for common data types
if HAS_PYDANTIC and BaseModel is not object:

    class EvaluationSpec(BaseModel):
        """Pydantic model for evaluation specifications."""

        name: str
        task: str
        dataset: str
        metrics: List[str] = []
        model: Optional[str] = None

    class DatasetSpec(BaseModel):
        """Pydantic model for dataset specifications."""

        name: str
        path: Optional[str] = None
        format: str = "jsonl"
        split: str = "test"
        num_samples: Optional[int] = None
