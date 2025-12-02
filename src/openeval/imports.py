"""
Unified Import Optimization System

Consolidates all lazy loading patterns into a single, efficient system.
Provides caching, error handling, and optional dependency management.
"""

from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache
from typing import Any, Optional, Dict, TypeVar, Union

T = TypeVar("T")

# Global registry for lazy imports
_IMPORT_CACHE: Dict[str, Any] = {}
_IMPORT_FLAGS: Dict[str, bool] = {}


class LazyModule:
    """Lazy module wrapper that imports on first attribute access."""

    def __init__(self, module_name: str, fallback: Optional[Any] = None):
        self._module_name = module_name
        self._module = None
        self._fallback = fallback
        self._import_failed = False

    def __getattr__(self, name: str) -> Any:
        if self._module is None and not self._import_failed:
            try:
                self._module = importlib.import_module(self._module_name)
                _IMPORT_CACHE[self._module_name] = self._module
                _IMPORT_FLAGS[self._module_name] = True
            except ImportError:
                self._import_failed = True
                _IMPORT_FLAGS[self._module_name] = False
                if self._fallback is not None:
                    self._module = self._fallback
                else:
                    raise ImportError(f"Failed to import {self._module_name}")

        if self._module is None:
            raise ImportError(f"Module {self._module_name} not available")

        return getattr(self._module, name)


@lru_cache(maxsize=128)
def lazy_import(
    module_name: str, fallback: Optional[Any] = None, cache: bool = True
) -> Union[Any, LazyModule]:
    """
    Optimized lazy import with caching and error handling.

    Args:
        module_name: Full module name to import
        fallback: Fallback value/object if import fails
        cache: Whether to cache the import result

    Returns:
        Imported module or LazyModule wrapper
    """
    if cache and module_name in _IMPORT_CACHE:
        return _IMPORT_CACHE[module_name]

    try:
        module = importlib.import_module(module_name)
        if cache:
            _IMPORT_CACHE[module_name] = module
            _IMPORT_FLAGS[module_name] = True
        return module
    except ImportError:
        _IMPORT_FLAGS[module_name] = False
        if fallback is not None:
            if cache:
                _IMPORT_CACHE[module_name] = fallback
            return fallback
        # Return lazy wrapper for delayed import attempts
        return LazyModule(module_name, fallback)


def is_available(module_name: str) -> bool:
    """Check if a module is available without importing it."""
    if module_name in _IMPORT_FLAGS:
        return _IMPORT_FLAGS[module_name]

    try:
        importlib.util.find_spec(module_name)
        _IMPORT_FLAGS[module_name] = True
        return True
    except (ImportError, ValueError, AttributeError):
        _IMPORT_FLAGS[module_name] = False
        return False


# Pre-cached common scientific computing modules
numpy = lazy_import("numpy", fallback=None)
pandas = lazy_import("pandas", fallback=None)
scipy = lazy_import("scipy", fallback=None)
sklearn = lazy_import("sklearn", fallback=None)

# Machine learning and AI modules
torch = lazy_import("torch", fallback=None)
transformers = lazy_import("transformers", fallback=None)

# Networking and async modules
httpx = lazy_import("httpx", fallback=None)
aiohttp = lazy_import("aiohttp", fallback=None)

# GPU and performance modules
cupy = lazy_import("cupy", fallback=None)
numba = lazy_import("numba", fallback=None)

# Availability flags for quick checks
HAS_NUMPY = is_available("numpy")
HAS_PANDAS = is_available("pandas")
HAS_SCIPY = is_available("scipy")
HAS_SKLEARN = is_available("sklearn")
HAS_TORCH = is_available("torch")
HAS_TRANSFORMERS = is_available("transformers")
HAS_HTTPX = is_available("httpx")
HAS_AIOHTTP = is_available("aiohttp")
HAS_CUPY = is_available("cupy")
HAS_NUMBA = is_available("numba")
HAS_GPU = HAS_CUPY or HAS_TORCH


# Convenience functions for common imports
def require_numpy():
    """Import numpy or raise informative error."""
    if not HAS_NUMPY:
        raise ImportError(
            "NumPy is required for this functionality. " "Install with: pip install numpy"
        )
    return numpy


def require_pandas():
    """Import pandas or raise informative error."""
    if not HAS_PANDAS:
        raise ImportError(
            "Pandas is required for this functionality. " "Install with: pip install pandas"
        )
    return pandas


def clear_cache():
    """Clear the import cache."""
    _IMPORT_CACHE.clear()
    _IMPORT_FLAGS.clear()
    lazy_import.cache_clear()


def get_import_stats() -> Dict[str, Any]:
    """Get statistics about imports."""
    return {
        "cached_modules": len(_IMPORT_CACHE),
        "availability_flags": len(_IMPORT_FLAGS),
        "available_modules": sum(_IMPORT_FLAGS.values()),
        "cache_hit_ratio": getattr(lazy_import, "cache_info", lambda: None)(),
    }


__all__ = [
    "lazy_import",
    "LazyModule",
    "is_available",
    "require_numpy",
    "require_pandas",
    "clear_cache",
    "get_import_stats",
    # Module instances
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "transformers",
    "httpx",
    "aiohttp",
    "cupy",
    "numba",
    # Flags
    "HAS_NUMPY",
    "HAS_PANDAS",
    "HAS_SCIPY",
    "HAS_SKLEARN",
    "HAS_TORCH",
    "HAS_TRANSFORMERS",
    "HAS_HTTPX",
    "HAS_AIOHTTP",
    "HAS_CUPY",
    "HAS_NUMBA",
    "HAS_GPU",
]
