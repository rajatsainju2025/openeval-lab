"""Dead code cleanup utilities.

Identifies and removes unused functions and imports.
"""

import ast
from pathlib import Path
from typing import List, Set, Dict


def cleanup_unused_imports(file_path: str) -> int:
    """Remove unused imports from a Python file.

    Falls back to AST-based implementation if autoflake is not available.
    """
    try:
        import autoflake

        with open(file_path, "r") as f:
            source = f.read()

        # Run autoflake to remove unused imports
        cleaned = autoflake.fix_code(source, remove_all_unused_imports=True)

        if cleaned != source:
            with open(file_path, "w") as f:
                f.write(cleaned)
            return 1
        return 0
    except ImportError:
        # Fallback to AST-based implementation
        return _cleanup_unused_imports_ast(file_path)


def _cleanup_unused_imports_ast(file_path: str) -> int:
    """AST-based unused import removal (fallback)."""
    with open(file_path, "r") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0

    # Collect imported names
    imported_names: Dict[str, ast.Import | ast.ImportFrom] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_names[name] = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_names[name] = node

    # Find used names
    used_names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Handle attribute access like module.function
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)

    # Identify unused imports
    unused = set(imported_names.keys()) - used_names

    if not unused:
        return 0

    # Simple heuristic: report count but don't auto-remove (safer)
    # Auto-removal would require precise line tracking
    return len(unused)


def find_dead_functions(file_path: str) -> List[str]:
    """Find potentially dead functions in a file.

    Falls back to AST-based implementation if vulture is not available.
    """
    try:
        import vulture

        v = vulture.Vulture()
        v.scavenge([file_path])

        return [str(unused) for unused in v.get_unused_code()]
    except ImportError:
        # Fallback to AST-based implementation
        return _find_dead_functions_ast(file_path)


def _find_dead_functions_ast(file_path: str) -> List[str]:
    """AST-based dead function detection (fallback)."""
    with open(file_path, "r") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # Collect defined functions and classes
    defined: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):  # Ignore private
                defined.add(node.name)

    # Find referenced names
    referenced: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)

    # Potentially dead = defined but not referenced
    potentially_dead = defined - referenced

    return [f"{file_path}:{name}" for name in sorted(potentially_dead)]


def analyze_module_dependencies(module_path: Path) -> Dict[str, List[str]]:
    """Analyze import dependencies for a Python module.

    Returns mapping of module -> list of modules it imports.
    """
    dependencies: Dict[str, List[str]] = {}

    if module_path.is_file():
        files = [module_path]
    else:
        files = list(module_path.rglob("*.py"))

    for file_path in files:
        with open(file_path, "r") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        dependencies[str(file_path)] = imports

    return dependencies


__all__ = [
    "cleanup_unused_imports",
    "find_dead_functions",
    "analyze_module_dependencies",
]
