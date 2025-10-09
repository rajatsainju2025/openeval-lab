"""
Architecture Refactoring Plan for OpenEval Lab

Current Issues:
1. Monolithic structure with 50+ modules in single package
2. Poor separation of concerns
3. Circular dependencies
4. Inconsistent naming and organization
5. Missing clear boundaries between components

New Architecture:
├── openeval/
│   ├── core/           # Core abstractions and interfaces
│   │   ├── __init__.py
│   │   ├── task.py
│   │   ├── dataset.py
│   │   ├── adapter.py
│   │   ├── metric.py
│   │   └── evaluation.py
│   ├── engine/         # Execution engine and orchestration
│   │   ├── __init__.py
│   │   ├── async_engine.py
│   │   ├── distributed_engine.py
│   │   ├── scheduler.py
│   │   └── worker.py
│   ├── storage/        # Data persistence and caching
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── results.py
│   │   ├── artifacts.py
│   │   └── compression.py
│   ├── config/         # Configuration management
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   ├── validation.py
│   │   └── templates.py
│   ├── monitoring/     # Observability and monitoring
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── alerts.py
│   │   ├── dashboard.py
│   │   └── telemetry.py
│   ├── security/       # Security and compliance
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── audit.py
│   │   ├── encryption.py
│   │   └── compliance.py
│   ├── cli/            # Command line interface
│   │   ├── __init__.py
│   │   ├── commands/
│   │   └── utils.py
│   └── plugins/        # Plugin system
│       ├── __init__.py
│       ├── marketplace.py
│       ├── loader.py
│       └── registry.py
"""

# Migration plan for existing modules
MIGRATION_MAP = {
    # Core components
    "core.py": "core/",
    "registry.py": "plugins/",
    "advanced_registry.py": "plugins/",
    # Engine components
    "async_evaluation_engine.py": "engine/",
    "distributed_engine.py": "engine/",
    "intelligent_scheduler.py": "engine/",
    "distributed_processor.py": "engine/",
    # Storage components
    "cache.py": "storage/",
    "optimized_cache.py": "storage/",
    "result_aggregation.py": "storage/",
    "results_analyzer.py": "storage/",
    "data_compression.py": "storage/",
    # Configuration
    "config.py": "config/",
    "unified_config.py": "config/",
    "config_manager.py": "config/",
    "config_validator.py": "config/",
    # Monitoring
    "monitoring_dashboard.py": "monitoring/",
    "observability.py": "monitoring/",
    "telemetry.py": "monitoring/",
    "performance.py": "monitoring/",
    "enhanced_logging.py": "monitoring/",
    # Security
    "security.py": "security/",
    "error_handling.py": "security/",
    # CLI
    "cli.py": "cli/",
    "commands/": "cli/commands/",
}
