"""
OpenEval Lab CLI Package

This package provides the command-line interface for OpenEval Lab.
"""


def __getattr__(name):
    if name == "app":
        from .cli import app

        return app
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
