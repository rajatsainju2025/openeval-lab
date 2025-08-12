"""Tests for code execution metric."""

import pytest
import tempfile
import os

from openeval.metrics.code_execution import (
    CodeExecutionMetric,
    HumanEvalMetric,
    execute_code_safely,
    calculate_pass_at_k,
)
from openeval.datasets.code import CodeDataset, HumanEvalDataset, CodeExample, HumanEvalExample


class TestCodeExecutionUtils:
    """Test utility functions for code execution."""

    def test_execute_code_safely_success(self):
        """Test successful code execution."""
        code = "def add(a, b):\n    return a + b"
        test_cases = ["assert add(2, 3) == 5", "assert add(0, 0) == 0"]

        result = execute_code_safely(code, test_cases, timeout=5.0)

        assert result["passed"] == 2
        assert result["total"] == 2
        assert len(result["errors"]) == 0
        assert result["execution_time"] > 0

    def test_execute_code_safely_failure(self):
        """Test code execution with failures."""
        code = "def add(a, b):\n    return a - b"  # Wrong implementation
        test_cases = ["assert add(2, 3) == 5"]

        result = execute_code_safely(code, test_cases, timeout=5.0)

        assert result["passed"] == 0
        assert result["total"] == 1
        assert len(result["errors"]) > 0

    def test_execute_code_safely_syntax_error(self):
        """Test code execution with syntax errors."""
        code = "def add(a, b\n    return a + b"  # Missing closing parenthesis
        test_cases = ["assert add(2, 3) == 5"]

        result = execute_code_safely(code, test_cases, timeout=5.0)

        assert result["passed"] == 0
        assert result["total"] == 1
        assert len(result["errors"]) > 0

    def test_execute_code_safely_timeout(self):
        """Test code execution timeout."""
        code = "import time\ndef slow():\n    time.sleep(10)\n    return 1"
        test_cases = ["assert slow() == 1"]

        result = execute_code_safely(code, test_cases, timeout=1.0)

        assert result["passed"] == 0
        assert result["total"] == 1
        assert "timed out" in result["errors"][0].lower()

    def test_calculate_pass_at_k(self):
        """Test pass@k calculation."""
        # All correct
        assert calculate_pass_at_k(10, 10, 1) == 1.0
        assert calculate_pass_at_k(10, 10, 5) == 1.0

        # None correct
        assert calculate_pass_at_k(10, 0, 1) == 0.0
        assert calculate_pass_at_k(10, 0, 5) == 0.0

        # Some correct - pass@1 should be accuracy
        score_1 = calculate_pass_at_k(10, 5, 1)
        assert score_1 == 0.5  # 5/10 = 0.5

        # pass@k for k>1 should be higher than pass@1
        score_2 = calculate_pass_at_k(10, 5, 2)
        assert score_2 > score_1

        # Edge case: k > n
        assert calculate_pass_at_k(5, 3, 10) == 0.0


class TestCodeExecutionMetric:
    """Test the code execution metric."""

    def test_basic_metric(self):
        """Test basic metric functionality."""
        metric = CodeExecutionMetric(timeout=5.0)

        predictions = [
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return a - b",  # Wrong
        ]
        references = ["assert add(2, 3) == 5", "assert add(2, 3) == 5"]

        result = metric.compute(predictions, references)

        assert "accuracy" in result
        assert "total_correct" in result
        assert "total_tests" in result
        assert "pass_at_k" in result
        assert "execution_results" in result

        assert result["accuracy"] == 0.5  # 1 out of 2 correct
        assert result["total_correct"] == 1
        assert result["total_tests"] == 2

    def test_multiple_test_cases(self):
        """Test with multiple test cases per sample."""
        metric = CodeExecutionMetric()

        predictions = ["def add(a, b):\n    return a + b"]
        references = [["assert add(2, 3) == 5", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]]

        result = metric.compute(predictions, references)

        assert result["total_correct"] == 1  # All tests passed
        assert result["execution_results"][0]["passed"] == 3
        assert result["execution_results"][0]["total"] == 3

    def test_pass_at_k_calculation(self):
        """Test pass@k calculation in metric."""
        metric = CodeExecutionMetric(k_values=[1, 2, 3])

        # 2 out of 4 correct
        predictions = [
            "def add(a, b):\n    return a + b",  # Correct
            "def add(a, b):\n    return a - b",  # Wrong
            "def add(a, b):\n    return a + b",  # Correct
            "def add(a, b):\n    return a * b",  # Wrong
        ]
        references = ["assert add(2, 3) == 5"] * 4

        result = metric.compute(predictions, references)

        assert "pass@1" in result["pass_at_k"]
        assert "pass@2" in result["pass_at_k"]
        assert "pass@3" in result["pass_at_k"]

        # pass@1 should be 0.5 (2 out of 4)
        assert result["pass_at_k"]["pass@1"] == 0.5


class TestHumanEvalMetric:
    """Test HumanEval-specific metric."""

    def test_humaneval_metric(self):
        """Test HumanEval metric with standard settings."""
        metric = HumanEvalMetric()

        predictions = ["def solution():\n    return 42"]
        references = ["assert solution() == 42"]

        result = metric.compute(predictions, references)

        assert "humaneval_score" in result
        assert "humaneval_pass_at_10" in result
        assert result["humaneval_score"] == result["pass_at_k"].get("pass@1", 0.0)


class TestCodeDataset:
    """Test code dataset functionality."""

    def test_code_dataset_loading(self):
        """Test loading code dataset."""
        dataset = CodeDataset(path="examples/code_toy.jsonl")
        examples = list(dataset)

        assert len(examples) > 0

        # Check first example
        ex = examples[0]
        assert hasattr(ex, "prompt")
        assert hasattr(ex, "solution")
        assert hasattr(ex, "test_cases")
        assert ex.id == "test/0"

    def test_humaneval_dataset_loading(self):
        """Test loading HumanEval dataset."""
        dataset = HumanEvalDataset(path="examples/code_toy.jsonl")
        examples = list(dataset)

        assert len(examples) > 0

        # Check first example
        ex = examples[0]
        assert isinstance(ex, HumanEvalExample)
        assert hasattr(ex, "entry_point")
        assert ex.entry_point == "add_two_numbers"
