"""String Utilities (DEPRECATED).

CONSOLIDATED: This module is now deprecated. Functionality is integrated
into string_utils.py. Import from there instead.
"""

from .string_utils import (
    normalize_text,
    tokenize,
    strip_punctuation,
)

__all__ = ["normalize_text", "tokenize", "strip_punctuation"]
