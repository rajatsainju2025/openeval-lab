"""Structured Logging (DEPRECATED).

CONSOLIDATED: This module is now deprecated. Functionality is integrated
into logging.py. Import from there instead.
"""

from .logging import get_logger, set_log_level

__all__ = ["get_logger", "set_log_level"]
