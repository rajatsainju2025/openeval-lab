# Public vs Private API Convention

## Overview

OpenEval Lab follows Python conventions for distinguishing public and private APIs:

- **Public API**: Functions, classes, and modules meant for external use
- **Private API**: Internal implementation details that may change

## Convention

- **Public**: `my_function()`, `MyClass`, `module.public_api`
- **Private**: `_my_function()`, `_MyClass`, `_module` (single underscore prefix)
- **Internal**: `__my_function()` (double underscore - rarely used)

## Public API Modules

These are guaranteed to be stable and should not break in minor versions:

```
openeval/
  __init__.py          # Main entry point
  core.py              # Task, Dataset, Adapter, Metric
  cache.py             # Prediction caching
  streaming.py         # Dataset streaming
  profiling.py         # Performance profiling
  resource_monitor.py  # Resource monitoring
  validation_cache.py  # Validation result caching
  string_utils.py      # String building utilities
  batch_operations.py  # Batch cache operations
  error_context.py     # Error handling
  structured_logging.py # Performance logging

  cli/                 # Command-line interface
  adapters/            # Model adapters (OpenAI, HF, etc.)
  datasets/            # Dataset loaders
  metrics/             # Evaluation metrics
  tasks/               # Task implementations
```

## Private Implementation Modules

These are internal and should not be directly imported or relied upon:

```
openeval/
  _cache_internals.py  # Cache implementation details
  _async_utils.py      # Async infrastructure
  _validators.py       # Internal validation
  config/
    _config_parser.py   # Config parsing internals
```

## Breaking Changes Policy

Public APIs should maintain backward compatibility within a major version.

**Will be stable**: `openeval.run()`, `openeval.cache.PredictionCache`
**May change**: `openeval._validators.internal_check()`

## Migration Guide

If using private APIs, migrate to public equivalents:

```python
# WRONG - using private API
from openeval._cache_internals import _BloomFilter

# RIGHT - use public cache
from openeval.cache import PredictionCache
```

## Reporting Issues

Found private API in public documentation?
[Open an issue on GitHub](https://github.com/rajatsainju2025/openeval-lab/issues)
