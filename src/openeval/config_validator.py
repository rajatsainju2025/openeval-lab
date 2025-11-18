"""Configuration Validator (DEPRECATED).

CONSOLIDATED: This module is now deprecated. Functionality is integrated
into config.py. Import from there instead.
"""

from .config import (
    ConfigManager,
    OpenEvalConfig,
)

# Re-export with common validation-related names
__all__ = ["ConfigManager", "OpenEvalConfig"]
