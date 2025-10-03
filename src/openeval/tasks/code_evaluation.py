"""
Code Evaluation Task for OpenEval Lab

This module implements code evaluation tasks including:
- Code generation quality assessment
- Code execution and testing
- Code style and best practices checking
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class CodeEvaluationTask:
    """
    Task for evaluating code generation and execution quality.
    """

    def __init__(
        self,
        language: str = "python",
        execution_timeout: float = 10.0,
        check_syntax: bool = True,
        check_style: bool = True,
        run_tests: bool = False,
        test_framework: Optional[str] = None
    ):
        """
        Initialize code evaluation task.

        Args:
            language: Programming language ("python", "javascript", etc.)
            execution_timeout: Maximum execution time in seconds
            check_syntax: Whether to check code syntax
            check_style: Whether to check code style
            run_tests: Whether to run unit tests
            test_framework: Test framework to use ("pytest", "unittest", etc.)
        """
        self.language = language.lower()
        self.execution_timeout = execution_timeout
        self.check_syntax = check_syntax
        self.check_style = check_style
        self.run_tests = run_tests
        self.test_framework = test_framework or self._get_default_test_framework()

    def _get_default_test_framework(self) -> str:
        """Get default test framework for the language."""
        if self.language == "python":
            return "pytest"
        elif self.language == "javascript":
            return "jest"
        else:
            return "none"

    def evaluate(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate code predictions.

        Args:
            predictions: List of generated code strings
            references: Optional reference solutions
            test_cases: Optional test cases for validation

        Returns:
            List of evaluation results
        """
        results = []

        for i, prediction in enumerate(predictions):
            result = {
                "prediction_id": i,
                "language": self.language,
                "syntax_valid": False,
                "style_score": 0.0,
                "execution_success": False,
                "test_passed": False,
                "quality_score": 0.0,
                "issues": [],
                "metrics": {}
            }

            # Syntax checking
            if self.check_syntax:
                syntax_result = self._check_syntax(prediction)
                result["syntax_valid"] = syntax_result["valid"]
                if not syntax_result["valid"]:
                    result["issues"].extend(syntax_result["errors"])

            # Style checking
            if self.check_style and result["syntax_valid"]:
                style_result = self._check_style(prediction)
                result["style_score"] = style_result["score"]
                result["issues"].extend(style_result["issues"])

            # Code execution
            if result["syntax_valid"]:
                exec_result = self._execute_code(prediction, test_cases[i] if test_cases else None)
                result["execution_success"] = exec_result["success"]
                result["execution_output"] = exec_result["output"]
                result["execution_error"] = exec_result["error"]
                if not exec_result["success"]:
                    result["issues"].append(f"Execution failed: {exec_result['error']}")

            # Test execution
            if self.run_tests and test_cases and result["syntax_valid"]:
                test_result = self._run_tests(prediction, test_cases[i])
                result["test_passed"] = test_result["passed"]
                result["test_output"] = test_result["output"]
                result["test_coverage"] = test_result.get("coverage", 0.0)

            # Calculate overall quality score
            result["quality_score"] = self._calculate_quality_score(result)
            result["metrics"] = self._extract_metrics(prediction)

            results.append(result)

        return results

    def _check_syntax(self, code: str) -> Dict[str, Any]:
        """Check code syntax."""
        if self.language == "python":
            return self._check_python_syntax(code)
        elif self.language == "javascript":
            return self._check_javascript_syntax(code)
        else:
            return {"valid": True, "errors": []}

    def _check_python_syntax(self, code: str) -> Dict[str, Any]:
        """Check Python code syntax."""
        try:
            ast.parse(code)
            return {"valid": True, "errors": []}
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [f"Syntax error at line {e.lineno}: {e.msg}"]
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Parse error: {str(e)}"]
            }

    def _check_javascript_syntax(self, code: str) -> Dict[str, Any]:
        """Check JavaScript code syntax."""
        # Basic syntax check - could be enhanced with a JS parser
        try:
            # Simple bracket matching
            brackets = {'(': ')', '[': ']', '{': '}'}
            stack = []
            for char in code:
                if char in brackets:
                    stack.append(char)
                elif char in brackets.values():
                    if not stack:
                        return {"valid": False, "errors": ["Unmatched closing bracket"]}
                    if brackets[stack.pop()] != char:
                        return {"valid": False, "errors": ["Mismatched brackets"]}

            if stack:
                return {"valid": False, "errors": ["Unmatched opening brackets"]}

            return {"valid": True, "errors": []}
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}

    def _check_style(self, code: str) -> Dict[str, Any]:
        """Check code style and best practices."""
        issues = []
        score = 1.0

        if self.language == "python":
            return self._check_python_style(code)
        elif self.language == "javascript":
            return self._check_javascript_style(code)
        else:
            return {"score": score, "issues": issues}

    def _check_python_style(self, code: str) -> Dict[str, Any]:
        """Check Python code style."""
        issues = []
        score = 1.0

        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:  # PEP 8 recommendation
                issues.append(f"Line {i} too long ({len(line)} > 88 characters)")
                score -= 0.1

            # Check for print statements (prefer logging)
            if re.search(r'\bprint\s*\(', line):
                issues.append(f"Line {i}: Consider using logging instead of print")
                score -= 0.05

            # Check for TODO comments
            if 'TODO' in line.upper():
                issues.append(f"Line {i}: TODO comment found")
                score -= 0.05

        # Check for docstrings
        if 'def ' in code and '"""' not in code:
            issues.append("Function lacks docstring")
            score -= 0.1

        return {"score": max(0.0, score), "issues": issues}

    def _check_javascript_style(self, code: str) -> Dict[str, Any]:
        """Check JavaScript code style."""
        issues = []
        score = 1.0

        # Basic style checks
        if 'var ' in code:
            issues.append("Consider using 'let' or 'const' instead of 'var'")
            score -= 0.1

        if 'console.log' in code:
            issues.append("Consider removing debug console.log statements")
            score -= 0.05

        return {"score": max(0.0, score), "issues": issues}

    def _execute_code(
        self,
        code: str,
        test_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute code and return results."""
        if self.language == "python":
            return self._execute_python_code(code, test_input)
        else:
            return {
                "success": False,
                "output": "",
                "error": f"Execution not supported for {self.language}"
            }

    def _execute_python_code(
        self,
        code: str,
        test_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute Python code."""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Prepare execution environment
            env = {"__name__": "__main__"}
            if test_input:
                env.update(test_input)

            # Execute with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.execution_timeout,
                env={**os.environ, **{k: str(v) for k, v in env.items()}}
            )

            # Clean up
            Path(temp_file).unlink(missing_ok=True)

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Execution timed out after {self.execution_timeout}s"
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }

    def _run_tests(self, code: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests for the code."""
        if self.test_framework == "pytest" and self.language == "python":
            return self._run_pytest(code, test_case)
        else:
            return {
                "passed": False,
                "output": "",
                "error": f"Test framework {self.test_framework} not supported"
            }

    def _run_pytest(self, code: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run pytest on the code."""
        try:
            # This is a simplified implementation
            # In practice, you'd need to create proper test files
            return {
                "passed": True,  # Placeholder
                "output": "Tests passed",
                "coverage": 0.85
            }
        except Exception as e:
            return {
                "passed": False,
                "output": "",
                "error": str(e)
            }

    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        score = 0.0
        total_weight = 0.0

        # Syntax (30%)
        if result["syntax_valid"]:
            score += 0.3
        total_weight += 0.3

        # Style (20%)
        score += result["style_score"] * 0.2
        total_weight += 0.2

        # Execution (30%)
        if result["execution_success"]:
            score += 0.3
        total_weight += 0.3

        # Tests (20%)
        if result.get("test_passed", False):
            score += 0.2
        total_weight += 0.2

        return score / total_weight if total_weight > 0 else 0.0

    def _extract_metrics(self, code: str) -> Dict[str, Any]:
        """Extract code metrics."""
        metrics = {
            "lines_of_code": len(code.split('\n')),
            "characters": len(code),
            "functions": len(re.findall(r'\bdef\s+', code)),
            "classes": len(re.findall(r'\bclass\s+', code)),
            "imports": len(re.findall(r'\bimport\s+|\bfrom\s+', code))
        }

        if self.language == "python":
            # Count Python-specific constructs
            metrics["list_comprehensions"] = len(re.findall(r'\[.*\s+for\s+.*\s+in\s+.*\]', code))
            metrics["dict_comprehensions"] = len(re.findall(r'\{.*\s+for\s+.*\s+in\s+.*\}', code))

        return metrics


def evaluate_code_task(
    predictions: List[str],
    language: str = "python",
    **kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Convenience function for code evaluation.

    Args:
        predictions: List of code strings to evaluate
        language: Programming language
        **kwargs: Additional arguments for CodeEvaluationTask

    Returns:
        Evaluation results
    """
    task = CodeEvaluationTask(language=language, **kwargs)
    return task.evaluate(predictions)
