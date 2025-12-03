"""OpenEval Lab - LLM Evaluation Framework.

A comprehensive, extensible framework for evaluating LLMs, multimodal models,
and AI agents with enterprise-grade reproducibility and performance.

Quick Start:
    >>> from openeval import Task, Dataset, Adapter, Metric, EvalSpec
    >>> # Define your evaluation specification
    >>> spec = EvalSpec.from_file("examples/qa_spec.json")
    >>> # Run evaluation via CLI:
    >>> # $ openeval run spec examples/qa_spec.json --verbose

Core Abstractions:
    - Task: Defines how to evaluate model on specific capability
    - Dataset: Collection of examples for evaluation
    - Adapter: Interface to model API or local model
    - Metric: Computes evaluation scores
    - Example: A single evaluation instance

Utilities:
    - EvalSpec: Load and validate evaluation specifications
    - profile_time, profile_block: Performance profiling decorators
    - PerformanceTimer: Context manager for timing operations

For detailed documentation, see https://github.com/openeval-lab/openeval-lab
"""

__version__ = "0.1.0"

# Public API - explicit list of exported symbols
__all__ = [
    # Version
    "__version__",
    # Core abstractions
    "Task",
    "Dataset",
    "Adapter",
    "Metric",
    "Example",
    # Spec loading
    "EvalSpec",
    # Profiling utilities
    "profile_time",
    "profile_block",
    "PerformanceTimer",
    # Version utilities
    "bump_version",
    "generate_changelog",
    # Error handling
    "ErrorCategory",
    "ErrorContext",
    "categorize_error",
    # Memory management
    "MemorySnapshot",
    "memory_tracked_operation",
]

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
    # Error handling (new)
    "ErrorCategory": (".error_handling", "ErrorCategory"),
    "ErrorContext": (".error_handling", "ErrorContext"),
    "categorize_error": (".error_handling", "categorize_error"),
    # Memory management (new)
    "MemorySnapshot": (".memory_management", "MemorySnapshot"),
    "memory_tracked_operation": (".memory_management", "memory_tracked_operation"),
}


def __getattr__(name: str):
    """Lazy import implementation for faster startup."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module

        module = import_module(module_name, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
