"""Semantic analysis for code understanding.

Performs variable tracking, scope analysis, and dependency analysis.
"""

import ast
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import CodeAnalyzer
from .types import AnalysisResult, CodeElement


class VariableScope:
    """Represents a scope for variable tracking."""

    def __init__(self, name: str, parent: Optional["VariableScope"] = None) -> None:
        """Initialize a scope.

        Args:
            name: Scope name (e.g., function name).
            parent: Parent scope.
        """
        self.name = name
        self.parent = parent
        self.variables: Dict[str, Set[str]] = {}  # var_name -> {types}
        self.children: List["VariableScope"] = []
        if parent:
            parent.children.append(self)

    def declare_variable(self, name: str, var_type: str) -> None:
        """Declare a variable in this scope.

        Args:
            name: Variable name.
            var_type: Inferred type.
        """
        if name not in self.variables:
            self.variables[name] = set()
        self.variables[name].add(var_type)

    def get_variable_type(self, name: str) -> Optional[str]:
        """Get variable type, checking parent scopes.

        Args:
            name: Variable name.

        Returns:
            Type string, or None if not found.
        """
        if name in self.variables and self.variables[name]:
            return list(self.variables[name])[0]

        if self.parent:
            return self.parent.get_variable_type(name)

        return None

    def get_all_variables(self) -> Dict[str, Set[str]]:
        """Get all variables in this scope including parents."""
        result = {}
        if self.parent:
            result.update(self.parent.get_all_variables())
        result.update(self.variables)
        return result


class PythonSemanticAnalyzer(CodeAnalyzer):
    """Analyze semantic structure of Python code."""

    def analyze(self, code: str) -> AnalysisResult:
        """Analyze code for semantic information.

        Args:
            code: Python source code.

        Returns:
            AnalysisResult with semantic analysis metadata.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError(f"Failed to parse code: {e}") from e

        elements = self.extract_elements(code)
        scopes = self._build_scope_tree(tree)
        dependencies = self.get_dependencies(code)
        variable_analysis = self._analyze_variables(tree)

        return AnalysisResult(
            code=code,
            elements=elements,
            dependencies=dependencies,
            metadata={
                "language": "python",
                "scopes": len(scopes),
                "variable_analysis": variable_analysis,
                "semantic_analysis": True,
            },
        )

    def extract_elements(self, code: str) -> List[CodeElement]:
        """Extract code elements (delegated to AST analyzer)."""
        # Basic extraction without full AST analysis
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        elements = []
        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                element = self._extract_function_semantic(node, lines)
                if element:
                    elements.append(element)
            elif isinstance(node, ast.ClassDef):
                element = self._extract_class_semantic(node, lines)
                if element:
                    elements.append(element)

        return elements

    def get_dependencies(self, code: str) -> List[str]:
        """Extract dependencies from code."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        dependencies = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if module:
                        dependencies.add(module)

        return list(dependencies)

    def _extract_function_semantic(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, lines: List[str]
    ):
        """Extract function with semantic info."""
        try:
            from .types import CodeElement, CodeElementType

            start_line = node.lineno - 1
            end_line = node.end_lineno or node.lineno
            source_lines = lines[start_line:end_line]
            source_code = "\n".join(source_lines)

            # Analyze parameters
            param_types = self._infer_parameter_types(node)

            metadata = {
                "parameters": {arg.arg: param_types.get(arg.arg, "Any") for arg in node.args.args},
                "has_return": any(isinstance(n, ast.Return) for n in ast.walk(node)),
            }

            return CodeElement(
                type=CodeElementType.FUNCTION,
                name=node.name,
                source_code=source_code,
                line_start=node.lineno,
                line_end=end_line,
                docstring=ast.get_docstring(node),
                metadata=metadata,
            )
        except Exception:
            return None

    def _extract_class_semantic(self, node: ast.ClassDef, lines: List[str]):
        """Extract class with semantic info."""
        try:
            from .types import CodeElement, CodeElementType

            start_line = node.lineno - 1
            end_line = node.end_lineno or node.lineno
            source_lines = lines[start_line:end_line]
            source_code = "\n".join(source_lines)

            # Analyze methods and attributes
            methods = [
                n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            attributes = self._extract_class_attributes(node)

            metadata = {
                "methods": methods,
                "attributes": attributes,
                "has_init": "__init__" in methods,
            }

            return CodeElement(
                type=CodeElementType.CLASS,
                name=node.name,
                source_code=source_code,
                line_start=node.lineno,
                line_end=end_line,
                docstring=ast.get_docstring(node),
                metadata=metadata,
            )
        except Exception:
            return None

    def _build_scope_tree(self, tree: ast.AST) -> Dict[str, VariableScope]:
        """Build scope hierarchy for the code.

        Args:
            tree: AST tree.

        Returns:
            Dictionary mapping scope names to VariableScope objects.
        """
        scopes: Dict[str, VariableScope] = {"global": VariableScope("global")}

        class ScopeBuilder(ast.NodeVisitor):
            def __init__(self):
                self.current = scopes["global"]
                self.scopes = scopes

            def visit_FunctionDef(self, node):
                new_scope = VariableScope(node.name, self.current)
                self.scopes[node.name] = new_scope
                old_scope = self.current
                self.current = new_scope

                # Add parameters to scope
                for arg in node.args.args:
                    self.current.declare_variable(arg.arg, "Any")

                self.generic_visit(node)
                self.current = old_scope

            def visit_ClassDef(self, node):
                new_scope = VariableScope(node.name, self.current)
                self.scopes[node.name] = new_scope
                old_scope = self.current
                self.current = new_scope
                self.generic_visit(node)
                self.current = old_scope

            def visit_Assign(self, node):
                for target in ast.walk(node.value):
                    if isinstance(target, ast.Name):
                        self.current.declare_variable(target.id, "Any")
                self.generic_visit(node)

        builder = ScopeBuilder()
        builder.visit(tree)
        return scopes

    def _analyze_variables(self, tree: ast.AST) -> Dict[str, Any]:
        """Perform variable analysis on code.

        Args:
            tree: AST tree.

        Returns:
            Dictionary with variable analysis results.
        """
        var_uses: Dict[str, int] = {}
        var_defs: Dict[str, int] = {}
        reassignments: Dict[str, int] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    var_defs[node.id] = var_defs.get(node.id, 0) + 1
                elif isinstance(node.ctx, ast.Load):
                    var_uses[node.id] = var_uses.get(node.id, 0) + 1

        # Find reassignments
        for var, def_count in var_defs.items():
            if def_count > 1:
                reassignments[var] = def_count

        return {
            "total_variables": len(set(var_defs.keys()) | set(var_uses.keys())),
            "definitions": len(var_defs),
            "uses": len(var_uses),
            "reassignments": len(reassignments),
            "reassignment_vars": reassignments,
            "unused_variables": [v for v in var_defs if v not in var_uses],
        }

    def _infer_parameter_types(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Dict[str, str]:
        """Infer parameter types from annotations.

        Args:
            node: FunctionDef node.

        Returns:
            Dictionary mapping parameter names to type strings.
        """
        types = {}

        for arg in node.args.args:
            if arg.annotation:
                types[arg.arg] = ast.unparse(arg.annotation)
            else:
                types[arg.arg] = "Any"

        return types

    def _extract_class_attributes(self, node: ast.ClassDef) -> List[str]:
        """Extract class attributes.

        Args:
            node: ClassDef node.

        Returns:
            List of attribute names.
        """
        attributes = []

        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)

        return attributes

    def get_variable_lifetime(self, code: str) -> Dict[str, Tuple[int, int]]:
        """Get variable definition and last use line numbers.

        Args:
            code: Python source code.

        Returns:
            Dictionary mapping variable names to (def_line, last_use_line) tuples.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}

        lifetimes: Dict[str, Tuple[int, int]] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    if node.id not in lifetimes:
                        lifetimes[node.id] = (node.lineno, node.lineno)
                    else:
                        start, _ = lifetimes[node.id]
                        lifetimes[node.id] = (start, node.lineno)
                elif isinstance(node.ctx, ast.Load):
                    if node.id not in lifetimes:
                        lifetimes[node.id] = (node.lineno, node.lineno)
                    else:
                        start, _ = lifetimes[node.id]
                        lifetimes[node.id] = (start, node.lineno)

        return lifetimes

    def find_data_flow_dependencies(self, code: str) -> Dict[str, List[str]]:
        """Identify data dependencies between variables.

        Args:
            code: Python source code.

        Returns:
            Dictionary mapping variables to their dependencies.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}

        dependencies: Dict[str, Set[str]] = {}

        class DataFlowAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.current_assign_target: Optional[str] = None

            def visit_Assign(self, node):
                # Get target variable
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.current_assign_target = target.id
                        if target.id not in dependencies:
                            dependencies[target.id] = set()

                        # Find all names in the value
                        for n in ast.walk(node.value):
                            if isinstance(n, ast.Name) and n.id != target.id:
                                dependencies[target.id].add(n.id)

                self.generic_visit(node)

        analyzer = DataFlowAnalyzer()
        analyzer.visit(tree)

        return {k: list(v) for k, v in dependencies.items()}
