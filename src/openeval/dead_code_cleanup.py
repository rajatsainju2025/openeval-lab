"""Dead code cleanup utilities.

Identifies and removes unused functions and imports.
"""


def cleanup_unused_imports(file_path: str) -> int:
    """Remove unused imports from a Python file."""
    try:
        import autoflake
    except ImportError:
        return 0

    with open(file_path, "r") as f:
        source = f.read()

    # Run autoflake to remove unused imports
    cleaned = autoflake.fix_code(source, remove_all_unused_imports=True)

    if cleaned != source:
        with open(file_path, "w") as f:
            f.write(cleaned)
        return 1
    return 0


def find_dead_functions(file_path: str) -> list:
    """Find potentially dead functions in a file."""
    try:
        import vulture
    except ImportError:
        return []

    v = vulture.Vulture()
    v.scavenge([file_path])

    return [str(unused) for unused in v.get_unused_code()]


__all__ = ["cleanup_unused_imports", "find_dead_functions"]
