"""Error Context Utilities (DEPRECATED).

CONSOLIDATED: This module is now deprecated. Functionality is integrated
into error_handling.py. Import from there instead.
"""

from .error_handling import (
    ErrorContext,
    with_error_context,
)

__all__ = ["ErrorContext", "with_error_context"]
