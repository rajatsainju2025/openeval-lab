"""CLI lazy import utilities.

Defers heavy module imports until needed to reduce startup time.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass

_lazy_modules = {}


def lazy_import(module_name: str) -> Any:
    """Import a module lazily on first access."""
    if module_name not in _lazy_modules:
        _lazy_modules[module_name] = __import__(module_name)
    return _lazy_modules[module_name]


def get_pandas():
    """Lazy import pandas."""
    return lazy_import("pandas")


def get_numpy():
    """Lazy import numpy."""
    return lazy_import("numpy")


def get_adapters():
    """Lazy import adapters."""
    return lazy_import("openeval.adapters")


__all__ = ["lazy_import", "get_pandas", "get_numpy", "get_adapters"]
