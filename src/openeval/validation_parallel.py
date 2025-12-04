"""Parallel validation utilities for faster adapter testing.

Extends validation.py with parallel test execution capabilities
to significantly reduce validation time for multiple adapters.

Usage:
    from openeval.validation_parallel import ParallelAdapterValidator

    validator = ParallelAdapterValidator(max_workers=4)
    results = validator.validate_multiple(adapters)
"""

from __future__ import annotations

import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .core import Adapter
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParallelValidationResult:
    """Result of parallel adapter validation."""

    adapter_name: str
    passed: bool
    test_results: Dict[str, bool]
    response_time: float
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class ParallelAdapterValidator:
    """Validates adapters in parallel for faster testing."""

    def __init__(self, max_workers: int = 4):
        """Initialize validator.

        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers

    def validate_multiple(self, adapters: List[Adapter]) -> Dict[str, ParallelValidationResult]:
        """Validate multiple adapters in parallel.

        Args:
            adapters: List of adapters to validate

        Returns:
            Dictionary mapping adapter names to validation results
        """
        results: Dict[str, ParallelValidationResult] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_adapter = {
                executor.submit(self._validate_single, adapter): adapter for adapter in adapters
            }

            for future in as_completed(future_to_adapter):
                adapter = future_to_adapter[future]
                try:
                    result = future.result()
                    results[adapter.name] = result
                except Exception as e:
                    logger.error(f"Failed to validate {adapter.name}: {e}")
                    results[adapter.name] = ParallelValidationResult(
                        adapter_name=adapter.name,
                        passed=False,
                        test_results={},
                        response_time=0.0,
                        error_message=str(e),
                    )

        return results

    def _validate_single(self, adapter: Adapter) -> ParallelValidationResult:
        """Validate a single adapter with parallel test execution."""
        start_time = time.time()

        # Run individual tests in parallel
        test_results, warnings = self._run_tests_parallel(adapter)

        response_time = time.time() - start_time
        passed = all(test_results.values())

        return ParallelValidationResult(
            adapter_name=adapter.name,
            passed=passed,
            test_results=test_results,
            response_time=response_time,
            warnings=warnings,
        )

    def _run_tests_parallel(self, adapter: Adapter) -> tuple[Dict[str, bool], List[str]]:
        """Run validation tests in parallel."""
        warnings: List[str] = []

        # Define test functions
        def test_simple():
            try:
                response = adapter.generate("Hello, world!")
                if response and len(response.strip()) > 0:
                    return ("simple_generation", True, None)
                return ("simple_generation", False, "Empty response")
            except Exception as e:
                return ("simple_generation", False, f"Error: {str(e)}")

        def test_empty():
            try:
                response = adapter.generate("")
                if response is not None:
                    return ("empty_prompt", True, None)
                return ("empty_prompt", False, "Returned None")
            except Exception as e:
                return ("empty_prompt", False, f"Error: {str(e)}")

        def test_long():
            try:
                long_prompt = "This is a very long prompt. " * 100
                response = adapter.generate(long_prompt)
                if response:
                    return ("long_prompt", True, None)
                return ("long_prompt", False, "Empty response")
            except Exception as e:
                return ("long_prompt", False, f"Error: {str(e)}")

        def test_special():
            try:
                special_prompt = "Test with émojis 🚀 and symbols: @#$%^&*()"
                response = adapter.generate(special_prompt)
                if response:
                    return ("special_characters", True, None)
                return ("special_characters", False, "Empty response")
            except Exception as e:
                return ("special_characters", False, f"Error: {str(e)}")

        # Execute tests in parallel
        tests = [test_simple, test_empty, test_long, test_special]
        test_results: Dict[str, bool] = {}

        with ThreadPoolExecutor(max_workers=min(len(tests), self.max_workers)) as executor:
            futures = {executor.submit(test): test for test in tests}

            for future in as_completed(futures):
                test_name, passed, warning = future.result()
                test_results[test_name] = passed
                if warning:
                    warnings.append(f"{test_name}: {warning}")

        return test_results, warnings

    def validate_with_profiling(
        self, adapters: List[Adapter]
    ) -> Dict[str, ParallelValidationResult]:
        """Validate adapters with detailed timing information.

        Args:
            adapters: List of adapters to validate

        Returns:
            Dictionary with validation results and timing breakdown
        """
        logger.info(f"Starting parallel validation of {len(adapters)} adapters")
        start = time.time()

        results = self.validate_multiple(adapters)

        total_time = time.time() - start
        logger.info(f"Validation completed in {total_time:.2f}s")

        # Calculate statistics
        passed_count = sum(1 for r in results.values() if r.passed)
        failed_count = len(results) - passed_count
        avg_time = sum(r.response_time for r in results.values()) / len(results)

        logger.info(
            f"Results: {passed_count} passed, {failed_count} failed, " f"avg time {avg_time:.2f}s"
        )

        return results


__all__ = ["ParallelAdapterValidator", "ParallelValidationResult"]
