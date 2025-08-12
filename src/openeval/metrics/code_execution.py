"""
Code execution metric with pass@k evaluation.

Evaluates generated code by running it and measuring pass@k scores,
commonly used for code generation benchmarks like HumanEval.
"""

from typing import Any, Dict, Iterable, List, Optional, Union
import subprocess
import tempfile
import os
import sys
import contextlib
import io
import traceback
import multiprocessing
import time
from pathlib import Path

from ..core import Metric


class CodeExecutionError(Exception):
    """Exception raised when code execution fails."""

    pass


class TimeoutError(Exception):
    """Exception raised when code execution times out."""

    pass


def execute_code_safely(
    code: str, test_cases: List[str], timeout: float = 5.0, language: str = "python"
) -> Dict[str, Any]:
    """
    Execute code safely with test cases.

    Args:
        code: The code to execute
        test_cases: List of test cases to run
        timeout: Execution timeout in seconds
        language: Programming language (currently only Python)

    Returns:
        Dictionary with execution results
    """
    if language.lower() != "python":
        raise NotImplementedError(f"Language {language} not supported yet")

    results = {
        "passed": 0,
        "total": len(test_cases),
        "errors": [],
        "outputs": [],
        "execution_time": 0.0,
    }

    start_time = time.time()

    for i, test_case in enumerate(test_cases):
        try:
            # Create complete code with test case
            full_code = code + "\n\n" + test_case

            # Execute in isolated environment
            result = _execute_python_code(full_code, timeout)

            if result["success"]:
                results["passed"] += 1
                results["outputs"].append(result["output"])
            else:
                results["errors"].append(f"Test {i+1}: {result['error']}")
                results["outputs"].append(result["output"])

        except Exception as e:
            results["errors"].append(f"Test {i+1}: {str(e)}")
            results["outputs"].append("")

    results["execution_time"] = time.time() - start_time
    return results


def _execute_python_code(code: str, timeout: float) -> Dict[str, Any]:
    """Execute Python code in a subprocess with timeout."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Run code in subprocess with timeout
            process = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir(),
            )

            return {
                "success": process.returncode == 0,
                "output": process.stdout,
                "error": process.stderr if process.returncode != 0 else None,
            }

        finally:
            # Clean up temporary file
            os.unlink(temp_file)

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Code execution timed out after {timeout} seconds",
        }
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


def calculate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    Calculate pass@k metric.

    Args:
        num_samples: Total number of samples generated
        num_correct: Number of samples that passed all tests
        k: Number of samples to consider for pass@k

    Returns:
        pass@k score between 0 and 1
    """
    if num_samples < k:
        return 0.0

    if num_correct == 0:
        return 0.0

    # For pass@1, it's simply the accuracy
    if k == 1:
        return num_correct / num_samples

    # For k > 1, use binomial formula
    # pass@k = 1 - C(n-c, k) / C(n, k)
    # where C(n, k) is binomial coefficient, n=num_samples, c=num_correct

    from math import comb

    # If we have enough correct samples to guarantee success, return 1.0
    if num_correct >= k:
        return 1.0

    # Calculate probability that all k samples fail
    try:
        prob_all_fail = comb(num_samples - num_correct, k) / comb(num_samples, k)
        return 1.0 - prob_all_fail
    except (ValueError, ZeroDivisionError):
        return 0.0


class CodeExecutionMetric(Metric):
    """Metric for evaluating code by execution with test cases."""

    name = "code_execution"

    def __init__(
        self,
        timeout: float = 5.0,
        language: str = "python",
        k_values: Optional[List[int]] = None,
        safe_execution: bool = True,
    ):
        """
        Initialize code execution metric.

        Args:
            timeout: Execution timeout in seconds
            language: Programming language
            k_values: List of k values for pass@k (default: [1, 5, 10])
            safe_execution: Whether to use safe subprocess execution
        """
        super().__init__()
        self.timeout = timeout
        self.language = language
        self.k_values = k_values or [1, 5, 10]
        self.safe_execution = safe_execution

    def compute(
        self, predictions: Iterable[Any], references: Iterable[Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Compute code execution metrics.

        Args:
            predictions: Generated code samples
            references: Test cases (or lists of test cases)

        Returns:
            Dictionary with execution results and pass@k scores
        """
        pred_list = list(predictions)
        ref_list = list(references)
        if len(pred_list) != len(ref_list):
            raise ValueError("Predictions and references must have same length")

        total_correct = 0
        total_tests = 0
        execution_results = []
        errors = []

        for i, (pred, ref) in enumerate(zip(pred_list, ref_list)):
            try:
                # Handle single test case or multiple test cases
                if isinstance(ref, str):
                    test_cases = [ref]
                elif isinstance(ref, list):
                    test_cases = ref
                else:
                    test_cases = [str(ref)]

                # Execute code with test cases
                result = execute_code_safely(
                    str(pred), test_cases, timeout=self.timeout, language=self.language
                )

                execution_results.append(result)

                # Check if all tests passed
                if result["passed"] == result["total"]:
                    total_correct += 1

                total_tests += 1

                # Collect errors for debugging
                if result["errors"]:
                    errors.extend([f"Sample {i+1}: {err}" for err in result["errors"]])

            except Exception as e:
                errors.append(f"Sample {i+1}: Execution error: {str(e)}")
                execution_results.append(
                    {
                        "passed": 0,
                        "total": 1,
                        "errors": [str(e)],
                        "outputs": [""],
                        "execution_time": 0.0,
                    }
                )
                total_tests += 1

        # Calculate pass@k for each k value
        pass_at_k = {}
        for k in self.k_values:
            if k <= total_tests:
                pass_at_k[f"pass@{k}"] = calculate_pass_at_k(total_tests, total_correct, k)

        # Calculate additional metrics
        accuracy = total_correct / total_tests if total_tests > 0 else 0.0
        total_execution_time = sum(res["execution_time"] for res in execution_results)
        avg_execution_time = (
            total_execution_time / len(execution_results) if execution_results else 0.0
        )

        return {
            "accuracy": accuracy,
            "total_correct": total_correct,
            "total_tests": total_tests,
            "pass_at_k": pass_at_k,
            "execution_results": execution_results,
            "total_execution_time": total_execution_time,
            "avg_execution_time": avg_execution_time,
            "error_count": len(errors),
            "errors": errors[:10] if errors else [],  # Limit errors for output size
        }


class HumanEvalMetric(CodeExecutionMetric):
    """Specialized metric for HumanEval-style code evaluation."""

    name = "humaneval"

    def __init__(self, **kwargs):
        """Initialize HumanEval metric with standard settings."""
        super().__init__(
            timeout=3.0,  # HumanEval uses shorter timeout
            language="python",
            k_values=[1, 10, 100],  # Standard HumanEval k values
            **kwargs,
        )

    def compute(
        self, predictions: Iterable[Any], references: Iterable[Any], **kwargs
    ) -> Dict[str, Any]:
        """Compute HumanEval metrics with additional formatting."""
        result = super().compute(predictions, references, **kwargs)

        # Add HumanEval-specific formatting
        result["humaneval_score"] = result["pass_at_k"].get("pass@1", 0.0)
        result["humaneval_pass_at_10"] = result["pass_at_k"].get("pass@10", 0.0)

        return result
