"""AST-based code analyzer for Python.

Extracts structural information using abstract syntax trees.
"""

import ast
from typing import Dict, List, Optional, Set

from .base import CodeAnalyzer
from .types import AnalysisResult, CodeElement, CodeElementType


class PythonASTAnalyzer(CodeAnalyzer):
    """Analyze Python code structure using Abstract Syntax Trees."""

    def __init__(self) -> None:
        """Initialize the AST analyzer."""
        self.current_scope: List[str] = []
        self.imports: List[str] = []
        self.dependencies: Set[str] = set()

    def analyze(self, code: str) -> AnalysisResult:
        """Analyze Python code and extract structural information.

        Args:
            code: Python source code to analyze.

        Returns:
            AnalysisResult with extracted elements and metadata.

        Raises:
            SyntaxError: If code has syntax errors.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError(f"Failed to parse code: {e}") from e

        # Reset state
        self.current_scope = []
        self.imports = []
        self.dependencies = set()

        # Extract elements
        elements = self.extract_elements(code)
        imports = self._extract_imports(tree)
        dependencies = self._extract_dependencies(tree)

        return AnalysisResult(
            code=code,
            elements=elements,
            dependencies=list(dependencies),
            imports=imports,
            metadata={
                "language": "python",
                "total_elements": len(elements),
                "imports_count": len(imports),
            },
        )

    def extract_elements(self, code: str) -> List[CodeElement]:
        """Extract functions, classes, and other code elements.

        Args:
            code: Python source code.

        Returns:
            List of CodeElement objects.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        elements = []
        lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                element = self._extract_function(node, code, lines)
                if element:
                    elements.append(element)

            elif isinstance(node, ast.ClassDef):
                element = self._extract_class(node, code, lines)
                if element:
                    elements.append(element)

        return elements

    def get_dependencies(self, code: str) -> List[str]:
        """Extract code dependencies and imports.

        Args:
            code: Python source code.

        Returns:
            List of dependency module names.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        return self._extract_imports(tree) + list(self._extract_dependencies(tree))

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        code: str,
        lines: List[str],
    ) -> Optional[CodeElement]:
        """Extract function element information.

        Args:
            node: AST FunctionDef node.
            code: Full source code.
            lines: Code split by lines.

        Returns:
            CodeElement for the function, or None if extraction fails.
        """
        try:
            # Get function source
            start_line = node.lineno - 1
            end_line = node.end_lineno or node.lineno
            source_lines = lines[start_line:end_line]
            source_code = "\n".join(source_lines)

            # Get docstring
            docstring = ast.get_docstring(node)

            # Build metadata
            metadata = {
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "args": len(node.args.args),
                "has_return": any(isinstance(n, ast.Return) for n in ast.walk(node)),
                "decorators": [
                    d.id if isinstance(d, ast.Name) else ast.unparse(d) for d in node.decorator_list
                ],
            }

            return CodeElement(
                type=CodeElementType.FUNCTION,
                name=node.name,
                source_code=source_code,
                line_start=node.lineno,
                line_end=end_line,
                docstring=docstring,
                metadata=metadata,
            )
        except Exception:
            return None

    def _extract_class(
        self,
        node: ast.ClassDef,
        code: str,
        lines: List[str],
    ) -> Optional[CodeElement]:
        """Extract class element information.

        Args:
            node: AST ClassDef node.
            code: Full source code.
            lines: Code split by lines.

        Returns:
            CodeElement for the class, or None if extraction fails.
        """
        try:
            start_line = node.lineno - 1
            end_line = node.end_lineno or node.lineno
            source_lines = lines[start_line:end_line]
            source_code = "\n".join(source_lines)

            docstring = ast.get_docstring(node)

            # Count methods and attributes
            methods = [
                n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            attributes = []
            for n in node.body:
                if isinstance(n, ast.Assign):
                    for target in n.targets:
                        if isinstance(target, ast.Name):
                            attributes.append(target.id)

            metadata = {
                "methods": methods,
                "method_count": len(methods),
                "attribute_count": len(attributes),
                "bases": [b.id if isinstance(b, ast.Name) else ast.unparse(b) for b in node.bases],
            }

            return CodeElement(
                type=CodeElementType.CLASS,
                name=node.name,
                source_code=source_code,
                line_start=node.lineno,
                line_end=end_line,
                docstring=docstring,
                metadata=metadata,
            )
        except Exception:
            return None

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from AST.

        Args:
            tree: AST tree.

        Returns:
            List of imported module names.
        """
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        imports.append(f"{module}.*")
                    else:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)

        return list(set(imports))  # Remove duplicates

    def _extract_dependencies(self, tree: ast.AST) -> Set[str]:
        """Extract non-local references and dependencies.

        Args:
            tree: AST tree.

        Returns:
            Set of dependency names.
        """
        dependencies = set()

        for node in ast.walk(tree):
            # Function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    dependencies.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        dependencies.add(node.func.value.id)

            # Attribute access
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    dependencies.add(node.value.id)

        return dependencies

    def get_call_graph(self, code: str) -> Dict[str, List[str]]:
        """Extract function call relationships.

        Args:
            code: Python source code.

        Returns:
            Dictionary mapping function names to their callees.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}

        class CallGraphExtractor(ast.NodeVisitor):
            def __init__(self):
                self.graph: Dict[str, List[str]] = {}
                self.current_fn: Optional[str] = None

            def visit_FunctionDef(self, node):
                old_fn = self.current_fn
                self.current_fn = node.name
                self.graph[node.name] = []
                self.generic_visit(node)
                self.current_fn = old_fn

            def visit_Call(self, node):
                if self.current_fn and isinstance(node.func, ast.Name):
                    self.graph[self.current_fn].append(node.func.id)
                self.generic_visit(node)

        extractor = CallGraphExtractor()
        extractor.visit(tree)
        return extractor.graph

    def get_control_flow_complexity(self, code: str) -> Dict[str, int]:
        """Calculate control flow complexity metrics per function.

        Args:
            code: Python source code.

        Returns:
            Dictionary mapping function names to complexity counts.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}

        class ComplexityCounter(ast.NodeVisitor):
            def __init__(self):
                self.current_fn: Optional[str] = None
                self.complexity: Dict[str, int] = {}

            def visit_FunctionDef(self, node):
                old_fn = self.current_fn
                self.current_fn = node.name
                self.complexity[node.name] = 1
                self.generic_visit(node)
                self.current_fn = old_fn

            def visit_If(self, node):
                if self.current_fn:
                    self.complexity[self.current_fn] = self.complexity.get(self.current_fn, 0) + 1
                self.generic_visit(node)

            def visit_For(self, node):
                if self.current_fn:
                    self.complexity[self.current_fn] = self.complexity.get(self.current_fn, 0) + 1
                self.generic_visit(node)

            def visit_While(self, node):
                if self.current_fn:
                    self.complexity[self.current_fn] = self.complexity.get(self.current_fn, 0) + 1
                self.generic_visit(node)

            def visit_ExceptHandler(self, node):
                if self.current_fn:
                    self.complexity[self.current_fn] = self.complexity.get(self.current_fn, 0) + 1
                self.generic_visit(node)

        counter = ComplexityCounter()
        counter.visit(tree)
        return counter.complexity
