"""Model validation and testing framework for adapters.

CONSOLIDATED: This module is now deprecated. All functionality has been
consolidated into validation_unified.py. Import from there instead.

For backward compatibility, this module re-exports key classes.
"""

# Re-export from unified module for backward compatibility
from .validation_unified import (
    AdapterValidationResult as ValidationResult,
    AdapterValidator,
    BasicFunctionalityValidator,
    PerformanceValidator,
    SafetyValidator,
    AdapterTestSuite,
)

__all__ = [
    "ValidationResult",
    "AdapterValidator",
    "BasicFunctionalityValidator",
    "PerformanceValidator",
    "SafetyValidator",
    "AdapterTestSuite",
]
