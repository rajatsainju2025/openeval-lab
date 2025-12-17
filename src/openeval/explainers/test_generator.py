"""Test case generator for generating tests from code analysis.

This module provides functionality to automatically generate test cases
based on code analysis, helping developers create comprehensive test suites.

Example:
    >>> from openeval.explainers import TestGenerator, generate_tests
    >>> code = '''
    ... def add(a: int, b: int) -> int:
    ...     return a + b
    ... '''
    >>> generator = get_test_generator()
    >>> tests = generator.generate(code)
    >>> print(tests.test_code)
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TestType(Enum):
    """Types of test cases."""

    UNIT = "unit"
    INTEGRATION = "integration"
    EDGE_CASE = "edge_case"
    BOUNDARY = "boundary"
    ERROR = "error"
    PERFORMANCE = "performance"
    PROPERTY = "property"


class TestFramework(Enum):
    """Supported test frameworks."""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    DOCTEST = "doctest"
    HYPOTHESIS = "hypothesis"


class CoverageType(Enum):
    """Types of test coverage."""

    LINE = "line"
    BRANCH = "branch"
    PATH = "path"
    MUTATION = "mutation"


@dataclass
class TestCase:
    """A single test case."""

    name: str
    test_type: TestType
    description: str
    test_code: str
    inputs: dict[str, Any]
    expected_output: Any
    assertions: list[str]
    setup_code: str = ""
    teardown_code: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_pytest(self) -> str:
        """Convert to pytest format."""
        lines = []
        if self.setup_code:
            lines.append(self.setup_code)
        lines.append(f"def test_{self.name}():")
        lines.append(f'    """Test: {self.description}"""')
        lines.append(self.test_code)
        for assertion in self.assertions:
            lines.append(f"    {assertion}")
        if self.teardown_code:
            lines.append(self.teardown_code)
        return "\n".join(lines)

    def to_unittest(self) -> str:
        """Convert to unittest format."""
        lines = []
        lines.append(f"def test_{self.name}(self):")
        lines.append(f'    """Test: {self.description}"""')
        if self.setup_code:
            lines.append(f"    {self.setup_code}")
        lines.append(f"    {self.test_code}")
        for assertion in self.assertions:
            lines.append(f"    {assertion}")
        return "\n".join(lines)


@dataclass
class TestSuite:
    """A collection of test cases."""

    name: str
    description: str
    test_cases: list[TestCase]
    framework: TestFramework
    imports: list[str] = field(default_factory=list)
    fixtures: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def test_code(self) -> str:
        """Generate complete test code."""
        lines = []

        # Imports
        if self.framework == TestFramework.PYTEST:
            lines.append("import pytest")
        elif self.framework == TestFramework.UNITTEST:
            lines.append("import unittest")

        for imp in self.imports:
            lines.append(imp)

        lines.append("")

        # Fixtures
        for fixture in self.fixtures:
            lines.append(fixture)
            lines.append("")

        # Test cases
        if self.framework == TestFramework.UNITTEST:
            lines.append(f"class Test{self.name}(unittest.TestCase):")
            lines.append(f'    """Test suite: {self.description}"""')
            lines.append("")
            for tc in self.test_cases:
                lines.append("    " + tc.to_unittest().replace("\n", "\n    "))
                lines.append("")
        else:
            for tc in self.test_cases:
                lines.append(tc.to_pytest())
                lines.append("")

        return "\n".join(lines)

    @property
    def total_tests(self) -> int:
        """Total number of test cases."""
        return len(self.test_cases)


@dataclass
class GenerationResult:
    """Result of test generation."""

    source_code: str
    test_suite: TestSuite
    coverage_estimate: float
    suggestions: list[str]
    statistics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


@dataclass
class FunctionSignature:
    """Extracted function signature."""

    name: str
    parameters: list[dict[str, Any]]
    return_type: str | None
    docstring: str | None
    is_async: bool
    decorators: list[str]
    line_number: int


class TestStrategy(ABC):
    """Abstract base class for test generation strategies."""

    @abstractmethod
    def generate_tests(
        self,
        signature: FunctionSignature,
        source_code: str,
    ) -> list[TestCase]:
        """Generate test cases for a function."""
        pass


class BasicTestStrategy(TestStrategy):
    """Basic test generation strategy."""

    def generate_tests(
        self,
        signature: FunctionSignature,
        source_code: str,
    ) -> list[TestCase]:
        """Generate basic tests."""
        tests = []
        test_id = 0

        # Generate a basic test for normal execution
        test_id += 1
        inputs = self._generate_sample_inputs(signature)
        tests.append(
            TestCase(
                name=f"{signature.name}_basic_{test_id}",
                test_type=TestType.UNIT,
                description=f"Basic test for {signature.name}",
                test_code=self._generate_call_code(signature, inputs),
                inputs=inputs,
                expected_output=None,
                assertions=["assert result is not None"],
            )
        )

        return tests

    def _generate_sample_inputs(self, signature: FunctionSignature) -> dict[str, Any]:
        """Generate sample inputs for a function."""
        inputs = {}
        type_samples = {
            "int": 1,
            "float": 1.0,
            "str": '"test"',
            "bool": True,
            "list": [],
            "dict": {},
            "None": None,
        }

        for param in signature.parameters:
            param_name = param["name"]
            param_type = param.get("type", "Any")
            if param_type in type_samples:
                inputs[param_name] = type_samples[param_type]
            else:
                inputs[param_name] = None

        return inputs

    def _generate_call_code(
        self,
        signature: FunctionSignature,
        inputs: dict[str, Any],
    ) -> str:
        """Generate code to call the function."""
        args = ", ".join(
            f"{k}={v}" if not isinstance(v, str) else f"{k}={v}" for k, v in inputs.items()
        )
        if signature.is_async:
            return f"    result = await {signature.name}({args})"
        return f"    result = {signature.name}({args})"


class EdgeCaseTestStrategy(TestStrategy):
    """Edge case test generation strategy."""

    def generate_tests(
        self,
        signature: FunctionSignature,
        source_code: str,
    ) -> list[TestCase]:
        """Generate edge case tests."""
        tests = []
        test_id = 0

        for param in signature.parameters:
            param_type = param.get("type", "Any")

            # None test
            if "None" not in param_type and "Optional" not in param_type:
                test_id += 1
                tests.append(
                    TestCase(
                        name=f"{signature.name}_none_{param['name']}_{test_id}",
                        test_type=TestType.EDGE_CASE,
                        description=f"Test {signature.name} with None for {param['name']}",
                        test_code=f"    with pytest.raises((TypeError, ValueError)):\n        {signature.name}({param['name']}=None)",
                        inputs={param["name"]: None},
                        expected_output="Exception",
                        assertions=[],
                        tags=["edge_case", "null_input"],
                    )
                )

            # Empty string test for str params
            if param_type == "str":
                test_id += 1
                tests.append(
                    TestCase(
                        name=f"{signature.name}_empty_string_{test_id}",
                        test_type=TestType.EDGE_CASE,
                        description=f"Test {signature.name} with empty string",
                        test_code=f'    result = {signature.name}({param["name"]}="")',
                        inputs={param["name"]: ""},
                        expected_output=None,
                        assertions=["# Verify handling of empty string"],
                        tags=["edge_case", "empty_input"],
                    )
                )

            # Empty list test for list params
            if "list" in param_type.lower():
                test_id += 1
                tests.append(
                    TestCase(
                        name=f"{signature.name}_empty_list_{test_id}",
                        test_type=TestType.EDGE_CASE,
                        description=f"Test {signature.name} with empty list",
                        test_code=f'    result = {signature.name}({param["name"]}=[])',
                        inputs={param["name"]: []},
                        expected_output=None,
                        assertions=["# Verify handling of empty list"],
                        tags=["edge_case", "empty_input"],
                    )
                )

        return tests


class BoundaryTestStrategy(TestStrategy):
    """Boundary value test generation strategy."""

    def generate_tests(
        self,
        signature: FunctionSignature,
        source_code: str,
    ) -> list[TestCase]:
        """Generate boundary value tests."""
        tests = []
        test_id = 0

        for param in signature.parameters:
            param_type = param.get("type", "Any")

            if param_type == "int":
                # Test with 0
                test_id += 1
                tests.append(
                    TestCase(
                        name=f"{signature.name}_zero_{param['name']}_{test_id}",
                        test_type=TestType.BOUNDARY,
                        description=f"Test {signature.name} with zero",
                        test_code=f'    result = {signature.name}({param["name"]}=0)',
                        inputs={param["name"]: 0},
                        expected_output=None,
                        assertions=["# Verify zero handling"],
                        tags=["boundary", "zero"],
                    )
                )

                # Test with negative
                test_id += 1
                tests.append(
                    TestCase(
                        name=f"{signature.name}_negative_{param['name']}_{test_id}",
                        test_type=TestType.BOUNDARY,
                        description=f"Test {signature.name} with negative",
                        test_code=f'    result = {signature.name}({param["name"]}=-1)',
                        inputs={param["name"]: -1},
                        expected_output=None,
                        assertions=["# Verify negative handling"],
                        tags=["boundary", "negative"],
                    )
                )

                # Test with large number
                test_id += 1
                tests.append(
                    TestCase(
                        name=f"{signature.name}_large_{param['name']}_{test_id}",
                        test_type=TestType.BOUNDARY,
                        description=f"Test {signature.name} with large value",
                        test_code=f'    result = {signature.name}({param["name"]}=10**9)',
                        inputs={param["name"]: 10**9},
                        expected_output=None,
                        assertions=["# Verify large value handling"],
                        tags=["boundary", "large_value"],
                    )
                )

        return tests


class ErrorTestStrategy(TestStrategy):
    """Error handling test generation strategy."""

    def generate_tests(
        self,
        signature: FunctionSignature,
        source_code: str,
    ) -> list[TestCase]:
        """Generate error handling tests."""
        tests = []
        test_id = 0

        # Check for raise statements in source
        if "raise" in source_code:
            # Extract exception types
            exceptions = re.findall(r"raise\s+(\w+)", source_code)
            for exc in set(exceptions):
                test_id += 1
                tests.append(
                    TestCase(
                        name=f"{signature.name}_raises_{exc.lower()}_{test_id}",
                        test_type=TestType.ERROR,
                        description=f"Test that {signature.name} raises {exc}",
                        test_code=f"    with pytest.raises({exc}):\n        {signature.name}()",
                        inputs={},
                        expected_output=f"{exc} exception",
                        assertions=[],
                        tags=["error_handling", exc.lower()],
                    )
                )

        # Generate type error test
        if signature.parameters:
            test_id += 1
            tests.append(
                TestCase(
                    name=f"{signature.name}_type_error_{test_id}",
                    test_type=TestType.ERROR,
                    description=f"Test {signature.name} with wrong types",
                    test_code=f"    with pytest.raises(TypeError):\n        {signature.name}('invalid', 'types')",
                    inputs={"invalid": "types"},
                    expected_output="TypeError",
                    assertions=[],
                    tags=["error_handling", "type_error"],
                )
            )

        return tests


class CodeAnalyzer:
    """Analyzes code to extract testable elements."""

    def extract_functions(self, code: str) -> list[FunctionSignature]:
        """Extract function signatures from code."""
        functions = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = self._extract_signature(node)
                    functions.append(sig)

        except SyntaxError:
            # Fallback to regex extraction
            functions = self._regex_extract_functions(code)

        return functions

    def _extract_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionSignature:
        """Extract signature from AST node."""
        parameters = []

        for arg in node.args.args:
            param = {"name": arg.arg, "type": "Any"}
            if arg.annotation:
                param["type"] = ast.unparse(arg.annotation)
            parameters.append(param)

        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        docstring = ast.get_docstring(node)

        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)

        return FunctionSignature(
            name=node.name,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=decorators,
            line_number=node.lineno,
        )

    def _regex_extract_functions(self, code: str) -> list[FunctionSignature]:
        """Fallback regex extraction."""
        functions = []
        pattern = r"(async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:"

        for match in re.finditer(pattern, code):
            is_async = match.group(1) is not None
            name = match.group(2)
            params_str = match.group(3)
            return_type = match.group(4)

            parameters = []
            if params_str:
                for param in params_str.split(","):
                    param = param.strip()
                    if param and param != "self":
                        if ":" in param:
                            p_name, p_type = param.split(":", 1)
                            parameters.append({"name": p_name.strip(), "type": p_type.strip()})
                        else:
                            parameters.append({"name": param, "type": "Any"})

            functions.append(
                FunctionSignature(
                    name=name,
                    parameters=parameters,
                    return_type=return_type.strip() if return_type else None,
                    docstring=None,
                    is_async=is_async,
                    decorators=[],
                    line_number=0,
                )
            )

        return functions


class TestGenerator:
    """Main class for generating test cases."""

    def __init__(self, framework: TestFramework = TestFramework.PYTEST) -> None:
        """Initialize the test generator.

        Args:
            framework: Target test framework.
        """
        self.framework = framework
        self.analyzer = CodeAnalyzer()
        self._strategies: list[TestStrategy] = [
            BasicTestStrategy(),
            EdgeCaseTestStrategy(),
            BoundaryTestStrategy(),
            ErrorTestStrategy(),
        ]

    def register_strategy(self, strategy: TestStrategy) -> None:
        """Register a custom test generation strategy.

        Args:
            strategy: The strategy to register.
        """
        self._strategies.append(strategy)

    def generate(
        self,
        code: str,
        suite_name: str = "GeneratedTests",
        include_types: list[TestType] | None = None,
    ) -> GenerationResult:
        """Generate tests for the given code.

        Args:
            code: Source code to generate tests for.
            suite_name: Name for the test suite.
            include_types: Types of tests to include (all if None).

        Returns:
            GenerationResult containing generated tests.
        """
        # Extract functions
        functions = self.analyzer.extract_functions(code)

        all_test_cases = []
        suggestions = []
        warnings = []

        for func in functions:
            # Skip private functions
            if func.name.startswith("_") and not func.name.startswith("__"):
                suggestions.append(f"Consider adding tests for private function '{func.name}'")
                continue

            # Generate tests using all strategies
            for strategy in self._strategies:
                try:
                    tests = strategy.generate_tests(func, code)

                    # Filter by type if specified
                    if include_types:
                        tests = [t for t in tests if t.test_type in include_types]

                    all_test_cases.extend(tests)
                except Exception as e:
                    warnings.append(
                        f"Strategy {strategy.__class__.__name__} failed " f"for {func.name}: {e}"
                    )

        # Add suggestions based on analysis
        if not functions:
            suggestions.append("No functions found to test")

        for func in functions:
            if not func.docstring:
                suggestions.append(f"Add docstring to '{func.name}' for better tests")
            if not func.return_type:
                suggestions.append(f"Add return type annotation to '{func.name}'")

        # Create test suite
        test_suite = TestSuite(
            name=suite_name,
            description="Generated tests for code module",
            test_cases=all_test_cases,
            framework=self.framework,
            imports=self._generate_imports(code),
        )

        # Estimate coverage
        coverage = self._estimate_coverage(functions, all_test_cases)

        return GenerationResult(
            source_code=code,
            test_suite=test_suite,
            coverage_estimate=coverage,
            suggestions=suggestions,
            statistics={
                "functions_analyzed": len(functions),
                "tests_generated": len(all_test_cases),
                "by_type": {
                    tt.value: sum(1 for t in all_test_cases if t.test_type == tt) for tt in TestType
                },
            },
            warnings=warnings,
        )

    def _generate_imports(self, code: str) -> list[str]:
        """Generate import statements for tests."""
        imports = []

        # Try to detect module name
        module_match = re.search(r'"""([^"]+)"""', code)
        if module_match:
            imports.append(f"# Source: {module_match.group(1)[:50]}")

        return imports

    def _estimate_coverage(
        self,
        functions: list[FunctionSignature],
        tests: list[TestCase],
    ) -> float:
        """Estimate test coverage."""
        if not functions:
            return 0.0

        # Simple heuristic: tests per function
        tested_functions = set()
        for test in tests:
            # Extract function name from test name
            for func in functions:
                if func.name in test.name:
                    tested_functions.add(func.name)

        return len(tested_functions) / len(functions) if functions else 0.0

    def generate_for_function(
        self,
        code: str,
        function_name: str,
    ) -> list[TestCase]:
        """Generate tests for a specific function.

        Args:
            code: Source code.
            function_name: Name of function to test.

        Returns:
            List of test cases.
        """
        functions = self.analyzer.extract_functions(code)
        target = next((f for f in functions if f.name == function_name), None)

        if not target:
            return []

        tests = []
        for strategy in self._strategies:
            tests.extend(strategy.generate_tests(target, code))

        return tests


# Global instance
_test_generator: TestGenerator | None = None


def get_test_generator() -> TestGenerator:
    """Get the global test generator instance.

    Returns:
        The global TestGenerator instance.
    """
    global _test_generator
    if _test_generator is None:
        _test_generator = TestGenerator()
    return _test_generator


def reset_test_generator() -> None:
    """Reset the global test generator instance."""
    global _test_generator
    _test_generator = None


def generate_tests(
    code: str,
    framework: TestFramework = TestFramework.PYTEST,
    **kwargs: Any,
) -> GenerationResult:
    """Generate tests for code.

    Args:
        code: Source code to generate tests for.
        framework: Target test framework.
        **kwargs: Additional options.

    Returns:
        GenerationResult with tests.
    """
    generator = TestGenerator(framework)
    return generator.generate(code, **kwargs)


def create_test_generator(
    framework: TestFramework = TestFramework.PYTEST,
) -> TestGenerator:
    """Create a new test generator.

    Args:
        framework: Target test framework.

    Returns:
        New TestGenerator instance.
    """
    return TestGenerator(framework)


def generate_test_suite(
    code: str,
    suite_name: str = "Tests",
    framework: TestFramework = TestFramework.PYTEST,
) -> TestSuite:
    """Generate a complete test suite.

    Args:
        code: Source code.
        suite_name: Name for the suite.
        framework: Target framework.

    Returns:
        TestSuite with generated tests.
    """
    result = generate_tests(code, framework=framework, suite_name=suite_name)
    return result.test_suite
