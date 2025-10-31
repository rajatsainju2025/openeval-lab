"""
Commands module providing modular CLI command structure.

This module organizes CLI commands into logical groups:
- base: Core system commands (registry, docs, version, etc.)
- evaluation: Evaluation-specific commands (validate, compare, etc.)
- run: Evaluation execution commands

Each command group is implemented as a separate Typer app for better
organization and maintainability.
"""

from __future__ import annotations

import typer

# Import command implementations
from .base import (
    docs,
    doctor,
    registry_info,
    registry_list,
    version,
    tutorial,
)
from .evaluation import compare, validate_results, validate_spec
from .run import run

# Create command group apps
base_app = typer.Typer(no_args_is_help=True, help="Base system commands")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation management commands")
run_app = typer.Typer(no_args_is_help=True, help="Run evaluation commands")

__all__ = [
    "base_app",
    "eval_app",
    "run_app",
    "docs",
    "doctor",
    "registry_info",
    "registry_list",
    "version",
    "tutorial",
    "compare",
    "validate_results",
    "validate_spec",
    "run",
]
