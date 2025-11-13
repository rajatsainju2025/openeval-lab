"""
Lazy Loading and Import Optimization Utilities

Provides utilities for efficiently deferring expensive imports and
initializing heavy dependencies only when needed. This significantly
improves startup time for the OpenEval Lab package.

Patterns:
- Lazy module loading with caching
- Optional dependency handling
- Import-time error suppression
- Circular import prevention
"""

from __future__ import annotations

import importlib
from typing import Optional, Any, Callable, Dict, TypeVar
import functools
import warnings

T = TypeVar("T")

# Registry of lazy-loaded modules
_LAZY_MODULES: Dict[str, Any] = {}
_LAZY_CALLBACKS: Dict[str, Callable[[], Any]] = {}


def lazy_import(module_name: str, fallback: Optional[Any] = None) -> Any:
    """Import a module lazily on first access.

    Args:
        module_name: Full module name (e.g., 'numpy.random')
        fallback: Fallback value if import fails

    Returns:
        The imported module or fallback
    """
    if module_name in _LAZY_MODULES:
        return _LAZY_MODULES[module_name]

    try:
        module = importlib.import_module(module_name)
        _LAZY_MODULES[module_name] = module
        return module
    except ImportError:
        if fallback is not None:
            _LAZY_MODULES[module_name] = fallback
            return fallback
        raise


def optional_import(module_name: str, warn: bool = True) -> Optional[Any]:
    """Import a module, returning None if not available.

    Args:
        module_name: Full module name
        warn: Whether to warn if import fails

    Returns:
        Module or None
    """
    try:
        return lazy_import(module_name)
    except ImportError:
        if warn:
            warnings.warn(
                f"Optional dependency not available: {module_name}",
                ImportWarning,
                stacklevel=2,
            )
        return None


class LazyAttribute:
    """Descriptor for lazy-loaded attributes."""

    def __init__(self, loader: Callable[[], Any], doc: str = ""):
        """Initialize lazy attribute.

        Args:
            loader: Function to load the value
            doc: Documentation
        """
        self.loader = loader
        self.value: Any = None
        self.loaded = False
        self.__doc__ = doc

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Any:
        """Load value on first access."""
        if not self.loaded:
            self.value = self.loader()
            self.loaded = True
        return self.value


class LazyModule:
    """Lazy-loading module proxy."""

    def __init__(self, module_name: str):
        """Initialize lazy module.

        Args:
            module_name: Full module name
        """
        self._module_name = module_name
        self._module: Optional[Any] = None

    def __getattr__(self, name: str) -> Any:
        """Load module on attribute access."""
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return getattr(self._module, name)

    def __dir__(self) -> list:
        """Get module attributes."""
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return dir(self._module)


def lazy_function(loader: Callable[..., Any]) -> Callable:
    """Decorator to make a function lazy-loaded.

    The decorated function won't be called until first invocation.

    Args:
        loader: Function to call when lazy function is invoked

    Returns:
        Wrapped function
    """

    @functools.wraps(loader)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Import and call on first use
        return loader(*args, **kwargs)

    return wrapper


def require_dependency(dependency_name: str, import_name: Optional[str] = None):
    """Decorator to require a dependency for a function.

    Args:
        dependency_name: Name of the dependency
        import_name: Import name (if different from dependency_name)

    Returns:
        Decorator function
    """
    import_name = import_name or dependency_name

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                importlib.import_module(import_name)
            except ImportError:
                raise ImportError(
                    f"Function '{func.__name__}' requires '{dependency_name}' "
                    f"but it is not installed. "
                    f"Install with: pip install {dependency_name}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Commonly used lazy imports in OpenEval
def get_numpy():
    """Get numpy, loading lazily if needed."""
    return lazy_import("numpy", fallback=None)


def get_pandas():
    """Get pandas, loading lazily if needed."""
    return lazy_import("pandas", fallback=None)


def get_torch():
    """Get torch, loading lazily if needed."""
    return optional_import("torch", warn=False)


def get_transformers():
    """Get transformers, loading lazily if needed."""
    return optional_import("transformers", warn=False)


def get_sklearn():
    """Get sklearn, loading lazily if needed."""
    return optional_import("sklearn", warn=False)


def get_sqlalchemy():
    """Get sqlalchemy, loading lazily if needed."""
    return optional_import("sqlalchemy", warn=False)


def get_httpx():
    """Get httpx, loading lazily if needed."""
    return lazy_import("httpx", fallback=None)


def get_orjson():
    """Get orjson, loading lazily if needed."""
    return lazy_import("orjson", fallback=None)


def get_yaml():
    """Get yaml, loading lazily if needed."""
    return optional_import("yaml", warn=True)


def preload_common_dependencies() -> None:
    """Preload commonly-used dependencies for performance.

    Call this explicitly if you want dependencies loaded upfront.
    """
    get_httpx()
    get_orjson()


def get_import_stats() -> Dict[str, Any]:
    """Get statistics about lazy-loaded modules."""
    return {
        "loaded_modules": len(_LAZY_MODULES),
        "modules": list(_LAZY_MODULES.keys()),
    }


# Monkey-patch common modules with lazy loading
def install_lazy_loading() -> None:
    """Install lazy loading patches for common modules.

    This should be called early in the import chain.
    """
    # This is a placeholder - real implementation would hook into sys.meta_path
    pass
