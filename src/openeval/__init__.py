"""OpenEval Lab - LLM Evaluation Framework."""

# Lazy loading mapping for faster startup
_LAZY_IMPORTS = {
    # Core abstractions
    "Task": (".core", "Task"),
    "Dataset": (".core", "Dataset"),
    "Adapter": (".core", "Adapter"),
    "Metric": (".core", "Metric"),
    "Example": (".core", "Example"),
    # Spec loading
    "EvalSpec": (".spec", "EvalSpec"),
    # Profiling utilities
    "profile_time": (".profiling", "profile_time"),
    "profile_block": (".profiling", "profile_block"),
    "PerformanceTimer": (".profiling", "PerformanceTimer"),
    # Version utilities
    "bump_version": (".version_utils", "bump_version"),
    "generate_changelog": (".version_utils", "generate_changelog"),
}


def __getattr__(name: str):
    """Lazy import implementation for faster startup."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module

        module = import_module(module_name, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
