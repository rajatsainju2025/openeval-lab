# Ensure the src/ directory is on sys.path when running tests from repo root
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# =============================================================================
# Common Test Fixtures
# =============================================================================


@pytest.fixture
def sample_qa_data() -> List[Dict[str, Any]]:
    """Provide sample Q&A data for testing."""
    return [
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe.",
            "answer": "Paris",
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "context": "Romeo and Juliet is a famous tragedy.",
            "answer": "William Shakespeare",
        },
        {
            "question": "What is 2 + 2?",
            "context": "Basic arithmetic.",
            "answer": "4",
        },
    ]


@pytest.fixture
def sample_spec() -> Dict[str, Any]:
    """Provide a valid sample evaluation specification."""
    return {
        "task": "qa",
        "dataset": "openeval.datasets.jsonl.JSONLinesDataset",
        "adapter": "openeval.adapters.echo.EchoAdapter",
        "metrics": ["accuracy", "f1"],
        "dataset_kwargs": {"path": "examples/qa_toy.jsonl"},
        "output": "results.json",
    }


@pytest.fixture
def invalid_spec() -> Dict[str, Any]:
    """Provide an invalid specification for error testing."""
    return {
        "task": "unknown_task_type",
        # Missing required fields: dataset, adapter, metrics
    }


@pytest.fixture
def temp_jsonl_file(tmp_path: Path, sample_qa_data: List[Dict[str, Any]]) -> Path:
    """Create a temporary JSONL file with sample data."""
    import json

    filepath = tmp_path / "test_data.jsonl"
    with open(filepath, "w") as f:
        for item in sample_qa_data:
            f.write(json.dumps(item) + "\n")
    return filepath


@pytest.fixture
def temp_results_file(tmp_path: Path) -> Path:
    """Create a temporary results file."""
    import json

    results = {
        "task": "qa",
        "adapter": "echo",
        "metrics": {"accuracy": 0.85, "f1": 0.82},
        "size": 100,
        "predictions": [{"input": "test", "output": "test", "reference": "test"}],
    }
    filepath = tmp_path / "results.json"
    with open(filepath, "w") as f:
        json.dump(results, f)
    return filepath


# =============================================================================
# Mock Fixtures
# =============================================================================


class MockAdapter:
    """Mock adapter for testing without API calls."""

    name = "mock"

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        return self.responses.get(prompt, f"Mock response for: {prompt[:50]}")


@pytest.fixture
def mock_adapter() -> MockAdapter:
    """Provide a mock adapter for testing."""
    return MockAdapter(
        responses={
            "What is 2 + 2?": "4",
            "Capital of France?": "Paris",
        }
    )


# =============================================================================
# Performance Testing Utilities
# =============================================================================


@pytest.fixture
def timer() -> Generator[dict, None, None]:
    """Context manager fixture for timing test sections.

    Usage:
        def test_performance(timer):
            # test code here
            assert timer['elapsed'] < 1.0
    """
    import time

    data = {"start": time.perf_counter(), "elapsed": 0.0}
    yield data
    data["elapsed"] = time.perf_counter() - data["start"]


# =============================================================================
# Test Markers
# =============================================================================

# Register custom markers for test categorization


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "api: marks tests that require external API access")
