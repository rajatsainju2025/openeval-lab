"""Validation caching system.

CONSOLIDATED: This module is now deprecated. All functionality has been
consolidated into validation_unified.py. Import from there instead.

For backward compatibility, this module re-exports key classes.
"""

# Re-export from unified module for backward compatibility
from .validation_unified import (
    SpecValidator,
    SchemaValidator,
    TypeValidator,
    DataQualityValidator,
    ValidationResult,
    spec_validator,
    schema_validator,
    type_validator,
    validate_spec,
    validate_schema,
    validate_type,
    validate_data_quality,
)

__all__ = [
    "SpecValidator",
    "SchemaValidator",
    "TypeValidator",
    "DataQualityValidator",
    "ValidationResult",
    "spec_validator",
    "schema_validator",
    "type_validator",
    "validate_spec",
    "validate_schema",
    "validate_type",
    "validate_data_quality",
]
