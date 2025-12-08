"""Code complexity metrics calculation.

Computes cyclomatic complexity, lines of code, nesting depth, and other metrics.
"""

import ast
from typing import Dict, Optional

from .base import ComplexityAnalyzer
from .types import ComplexityMetrics


class PythonComplexityAnalyzer(ComplexityAnalyzer):
    """Analyze Python code complexity."""

    def calculate(self, code: str) -> ComplexityMetrics:
        """Calculate comprehensive complexity metrics.

        Args:
            code: Python source code.

        Returns:
            ComplexityMetrics with all computed values.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Cannot analyze non-parseable code: {e}") from e

        cc = self.calculate_cyclomatic_complexity(code)
        loc = self._calculate_lines_of_code(code)
        comment_ratio = self._calculate_comment_ratio(code)
        max_nesting = self._calculate_max_nesting_depth(tree)
        fn_count = self._count_functions(tree)
        class_count = self._count_classes(tree)
        avg_fn_length = self._calculate_average_function_length(code, tree)

        return ComplexityMetrics(
            cyclomatic_complexity=cc,
            lines_of_code=loc,
            comment_ratio=comment_ratio,
            nesting_depth=max_nesting,
            function_count=fn_count,
            class_count=class_count,
            average_function_length=avg_fn_length,
        )

    def calculate_cyclomatic_complexity(self, code: str) -> float:
        """Calculate cyclomatic complexity (McCabe complexity).

        Higher values indicate more complex code.
        Rules: Base = 1, +1 for each decision point (if, for, while, except, etc)

        Args:
            code: Python source code.

        Returns:
            Cyclomatic complexity score.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0

        complexity = 1  # Base complexity

        class ComplexityCounter(ast.NodeVisitor):
            def __init__(self):
                self.count = 0

            def visit_If(self, node):
                self.count += 1
                self.generic_visit(node)

            def visit_For(self, node):
                self.count += 1
                self.generic_visit(node)

            def visit_While(self, node):
                self.count += 1
                self.generic_visit(node)

            def visit_ExceptHandler(self, node):
                self.count += 1
                self.generic_visit(node)

            def visit_BoolOp(self, node):
                # Add 1 for each 'and'/'or' operator
                self.count += len(node.values) - 1
                self.generic_visit(node)

        counter = ComplexityCounter()
        counter.visit(tree)
        complexity += counter.count

        return float(complexity)

    def _calculate_lines_of_code(self, code: str) -> int:
        """Count non-empty, non-comment lines.

        Args:
            code: Python source code.

        Returns:
            Number of executable lines.
        """
        loc = 0
        in_multiline_comment = False

        for line in code.split("\n"):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Track multiline strings/comments
            if '"""' in stripped or "'''" in stripped:
                in_multiline_comment = not in_multiline_comment
                continue

            if in_multiline_comment:
                continue

            # Skip comment lines
            if stripped.startswith("#"):
                continue

            loc += 1

        return loc

    def _calculate_comment_ratio(self, code: str) -> float:
        """Calculate ratio of comment lines to total lines.

        Args:
            code: Python source code.

        Returns:
            Ratio between 0.0 and 1.0.
        """
        total_lines = len([line for line in code.split("\n") if line.strip()])
        if total_lines == 0:
            return 0.0

        comment_lines = len([line for line in code.split("\n") if line.strip().startswith("#")])

        return min(comment_lines / total_lines, 1.0)

    def _calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth.

        Args:
            tree: AST tree.

        Returns:
            Maximum nesting depth.
        """

        class DepthCalculator(ast.NodeVisitor):
            def __init__(self):
                self.depth = 0
                self.max_depth = 0

            def generic_visit(self, node):
                # Track nesting for control structures
                if isinstance(
                    node,
                    (
                        ast.If,
                        ast.For,
                        ast.While,
                        ast.With,
                        ast.Try,
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                    ),
                ):
                    self.depth += 1
                    self.max_depth = max(self.max_depth, self.depth)
                    super().generic_visit(node)
                    self.depth -= 1
                else:
                    super().generic_visit(node)

        calculator = DepthCalculator()
        calculator.visit(tree)
        return calculator.max_depth

    def _count_functions(self, tree: ast.AST) -> int:
        """Count function definitions.

        Args:
            tree: AST tree.

        Returns:
            Number of functions.
        """
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                count += 1
        return count

    def _count_classes(self, tree: ast.AST) -> int:
        """Count class definitions.

        Args:
            tree: AST tree.

        Returns:
            Number of classes.
        """
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                count += 1
        return count

    def _calculate_average_function_length(self, code: str, tree: ast.AST) -> float:
        """Calculate average function length in lines.

        Args:
            code: Python source code.
            tree: AST tree.

        Returns:
            Average function length.
        """
        lengths = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    hasattr(node, "lineno")
                    and hasattr(node, "end_lineno")
                    and node.end_lineno is not None
                ):
                    length = node.end_lineno - node.lineno + 1
                    lengths.append(length)

        return sum(lengths) / len(lengths) if lengths else 0.0

    def get_complexity_by_function(self, code: str) -> Dict[str, float]:
        """Calculate complexity for each function.

        Args:
            code: Python source code.

        Returns:
            Dictionary mapping function names to complexity scores.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}

        complexities: Dict[str, float] = {}

        class FunctionComplexityCalculator(ast.NodeVisitor):
            def __init__(self):
                self.current_fn: Optional[str] = None
                self.current_complexity = 1

            def visit_FunctionDef(self, node):
                old_fn = self.current_fn
                old_cc = self.current_complexity
                self.current_fn = node.name
                self.current_complexity = 1
                self.generic_visit(node)
                complexities[node.name] = float(self.current_complexity)
                self.current_fn = old_fn
                self.current_complexity = old_cc

            def visit_If(self, node):
                if self.current_fn:
                    self.current_complexity += 1
                self.generic_visit(node)

            def visit_For(self, node):
                if self.current_fn:
                    self.current_complexity += 1
                self.generic_visit(node)

            def visit_While(self, node):
                if self.current_fn:
                    self.current_complexity += 1
                self.generic_visit(node)

            def visit_ExceptHandler(self, node):
                if self.current_fn:
                    self.current_complexity += 1
                self.generic_visit(node)

        calculator = FunctionComplexityCalculator()
        calculator.visit(tree)
        return complexities

    def get_maintainability_index(self, code: str) -> float:
        """Calculate Maintainability Index (0-100).

        Higher is better. Formula: MI = 171 - 5.2*ln(avgHalstead) - 0.23*CC + 50*sqrt(2.46*LOCM)
        Simplified version for quick assessment.

        Args:
            code: Python source code.

        Returns:
            Maintainability index 0-100.
        """
        metrics = self.calculate(code)

        # Simplified MI calculation
        mi = 100  # Start at max

        # Penalty for cyclomatic complexity
        if metrics.cyclomatic_complexity > 10:
            mi -= (metrics.cyclomatic_complexity - 10) * 2

        # Penalty for lines of code
        if metrics.lines_of_code > 200:
            mi -= (metrics.lines_of_code - 200) / 10

        # Penalty for nesting depth
        if metrics.nesting_depth > 5:
            mi -= (metrics.nesting_depth - 5) * 3

        # Bonus for comments
        if metrics.comment_ratio > 0.1:
            mi += metrics.comment_ratio * 5

        return max(0.0, min(100.0, mi))

    def rate_complexity(self, code: str) -> str:
        """Rate code complexity as string.

        Args:
            code: Python source code.

        Returns:
            Rating: "Simple", "Moderate", "Complex", or "Very Complex".
        """
        cc = self.calculate_cyclomatic_complexity(code)

        if cc <= 5:
            return "Simple"
        elif cc <= 10:
            return "Moderate"
        elif cc <= 20:
            return "Complex"
        else:
            return "Very Complex"
