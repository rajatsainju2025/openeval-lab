"""Configuration Consolidation Utilities (DEPRECATED).

CONSOLIDATED: This module is now deprecated. Functionality is integrated
into config.py. Import from there instead.
"""

from .config import (
    ConfigManager,
    OpenEvalConfig,
    load_config,
    save_config,
)

__all__ = ["ConfigManager", "OpenEvalConfig", "load_config", "save_config"]
