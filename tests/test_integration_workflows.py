"""Integration tests for key OpenEval workflows."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from openeval.cli import app
from openeval.core import Example, Dataset, Task, Adapter, Metric
from typer.testing import CliRunner

runner = CliRunner()


class MockDataset(Dataset):
    """Mock dataset for testing."""

    name = "mock_dataset"

    def __init__(self, examples=None):
        self.examples = examples or [
            Example(id="1", input="Hello", reference="Hi"),
            Example(id="2", input="Goodbye", reference="Bye"),
            Example(id="3", input="How are you?", reference="Good"),
        ]

    def __iter__(self):
        return iter(self.examples)


class MockAdapter(Adapter):
    """Mock adapter for testing."""

    name = "mock_adapter"

    def generate(self, prompt: str, **kwargs) -> str:
        # Simple echo with transformation
        if "Hello" in prompt:
            return "Hi"
        elif "Goodbye" in prompt:
            return "Bye"
        elif "How are you" in prompt:
            return "Good"
        else:
            return "OK"


class MockTask(Task):
    """Mock task for testing."""

    name = "mock_task"

    def build_prompt(self, ex: Example) -> str:
        return f"Input: {ex.input}\nOutput:"


class MockMetric(Metric):
    """Mock metric for testing."""

    name = "mock_accuracy"

    def compute(self, predictions, references):
        pred_list = list(predictions)
        ref_list = list(references)

        if not pred_list or not ref_list:
            return {"accuracy": 0.0}

        correct = sum(1 for p, r in zip(pred_list, ref_list) if str(p).strip() == str(r).strip())
        return {"accuracy": correct / len(pred_list)}


def test_full_evaluation_workflow():
    """Test complete evaluation from spec to results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create spec file
        spec_data = {
            "task": "mock_task",
            "dataset": "mock_dataset",
            "adapter": "mock_adapter",
            "metrics": [{"name": "mock_accuracy"}],
            "output": "results.json",
        }

        spec_file = tmpdir_path / "test_spec.json"
        with open(spec_file, "w") as f:
            json.dump(spec_data, f)

        # Mock the components
        with patch("openeval.spec.import_class") as mock_import:

            def mock_import_side_effect(path):
                if path == "mock_task":
                    return MockTask
                elif path == "mock_dataset":
                    return MockDataset
                elif path == "mock_adapter":
                    return MockAdapter
                elif path == "mock_accuracy":
                    return MockMetric
                else:
                    raise ImportError(f"Unknown class: {path}")

            mock_import.side_effect = mock_import_side_effect

            # Run evaluation
            result = runner.invoke(
                app, ["run", str(spec_file), "--records", "--artifacts", str(tmpdir_path / "runs")]
            )

            assert result.exit_code == 0

            # Check that results were written
            runs_dir = tmpdir_path / "runs"
            assert runs_dir.exists()

            result_files = list(runs_dir.glob("*.json"))
            assert len(result_files) > 0

            # Check result content
            with open(result_files[0]) as f:
                results = json.load(f)

            assert "metrics" in results
            assert "mock_accuracy" in results["metrics"]
            assert results["metrics"]["mock_accuracy"]["accuracy"] == 1.0  # Perfect match

            assert "records" in results
            assert len(results["records"]) == 3


def test_spec_validation():
    """Test spec validation workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Valid spec
        valid_spec = {
            "task": "mock_task",
            "dataset": "mock_dataset",
            "adapter": "mock_adapter",
            "metrics": [{"name": "mock_accuracy"}],
            "output": "results.json",
        }

        valid_spec_file = tmpdir_path / "valid_spec.json"
        with open(valid_spec_file, "w") as f:
            json.dump(valid_spec, f)

        # Invalid spec (missing required field)
        invalid_spec = {
            "task": "mock_task",
            # Missing dataset
            "adapter": "mock_adapter",
            "output": "results.json",
        }

        invalid_spec_file = tmpdir_path / "invalid_spec.json"
        with open(invalid_spec_file, "w") as f:
            json.dump(invalid_spec, f)

        with patch("openeval.spec.import_class") as mock_import:

            def mock_import_side_effect(path):
                if path.startswith("mock_"):
                    return MagicMock
                raise ImportError(f"Unknown class: {path}")

            mock_import.side_effect = mock_import_side_effect

            # Test valid spec
            result = runner.invoke(app, ["validate", str(valid_spec_file)])
            assert result.exit_code == 0

            # Test invalid spec
            result = runner.invoke(app, ["validate", str(invalid_spec_file)])
            assert result.exit_code != 0


def test_dataset_validation_workflow():
    """Test dataset validation workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create valid JSONL dataset
        valid_data = [
            {"id": "1", "input": "Hello", "reference": "Hi"},
            {"id": "2", "input": "Goodbye", "reference": "Bye"},
            {"id": "3", "input": "How are you?", "reference": "Good"},
        ]

        valid_jsonl = tmpdir_path / "valid_data.jsonl"
        with open(valid_jsonl, "w") as f:
            for item in valid_data:
                f.write(json.dumps(item) + "\n")

        # Create invalid JSONL (with issues)
        invalid_data = [
            {"id": "1", "input": "", "reference": "Hi"},  # Empty input
            {"id": "2", "input": "Hello", "reference": ""},  # Empty reference
            {"malformed json": True},  # This line will be invalid JSON
        ]

        invalid_jsonl = tmpdir_path / "invalid_data.jsonl"
        with open(invalid_jsonl, "w") as f:
            for item in invalid_data[:-1]:
                f.write(json.dumps(item) + "\n")
            f.write("invalid json line\n")

        # Test valid dataset
        result = runner.invoke(
            app,
            [
                "validate-dataset",
                str(valid_jsonl),
                "--output",
                str(tmpdir_path / "valid_report.json"),
            ],
        )
        assert result.exit_code == 0

        # Check report was created
        report_file = tmpdir_path / "valid_report.json"
        assert report_file.exists()

        with open(report_file) as f:
            report = json.load(f)

        assert report["total_examples"] == 3
        assert report["valid_examples"] == 3
        assert report["quality_score"] > 0.8

        # Test invalid dataset
        result = runner.invoke(app, ["validate-dataset", str(invalid_jsonl), "--strict"])
        assert result.exit_code != 0


def test_comparison_workflow():
    """Test model comparison workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create two mock result files
        results1 = {
            "metrics": {"accuracy": {"accuracy": 0.8}},
            "records": [
                {"id": "1", "prediction": "Hi", "reference": "Hi"},
                {"id": "2", "prediction": "Bye", "reference": "Bye"},
                {"id": "3", "prediction": "OK", "reference": "Good"},
            ],
        }

        results2 = {
            "metrics": {"accuracy": {"accuracy": 0.9}},
            "records": [
                {"id": "1", "prediction": "Hi", "reference": "Hi"},
                {"id": "2", "prediction": "Bye", "reference": "Bye"},
                {"id": "3", "prediction": "Good", "reference": "Good"},
            ],
        }

        results1_file = tmpdir_path / "results1.json"
        results2_file = tmpdir_path / "results2.json"

        with open(results1_file, "w") as f:
            json.dump(results1, f)

        with open(results2_file, "w") as f:
            json.dump(results2, f)

        # Test comparison
        result = runner.invoke(
            app, ["compare", str(results1_file), str(results2_file), "--bootstrap", "100"]
        )

        assert result.exit_code == 0
        # Should output comparison statistics


def test_caching_workflow():
    """Test caching functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        cache_dir = tmpdir_path / "cache"

        spec_data = {
            "task": "mock_task",
            "dataset": "mock_dataset",
            "adapter": "mock_adapter",
            "metrics": [{"name": "mock_accuracy"}],
            "output": "results.json",
        }

        spec_file = tmpdir_path / "test_spec.json"
        with open(spec_file, "w") as f:
            json.dump(spec_data, f)

        with patch("openeval.spec.import_class") as mock_import:

            def mock_import_side_effect(path):
                if path.startswith("mock_"):
                    if path == "mock_task":
                        return MockTask
                    elif path == "mock_dataset":
                        return MockDataset
                    elif path == "mock_adapter":
                        return MockAdapter
                    elif path == "mock_accuracy":
                        return MockMetric
                raise ImportError(f"Unknown class: {path}")

            mock_import.side_effect = mock_import_side_effect

            # First run with cache write
            result1 = runner.invoke(
                app, ["run", str(spec_file), "--cache-dir", str(cache_dir), "--cache", "write"]
            )

            assert result1.exit_code == 0
            assert cache_dir.exists()

            # Second run with cache read
            result2 = runner.invoke(
                app, ["run", str(spec_file), "--cache-dir", str(cache_dir), "--cache", "read"]
            )

            assert result2.exit_code == 0


def test_interactive_mode():
    """Test interactive mode (mocked)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        spec_data = {
            "task": "mock_task",
            "dataset": "mock_dataset",
            "adapter": "mock_adapter",
            "metrics": [{"name": "mock_accuracy"}],
            "output": "results.json",
        }

        spec_file = tmpdir_path / "test_spec.json"
        with open(spec_file, "w") as f:
            json.dump(spec_data, f)

        with patch("openeval.spec.import_class") as mock_import:

            def mock_import_side_effect(path):
                if path.startswith("mock_"):
                    if path == "mock_task":
                        return MockTask
                    elif path == "mock_dataset":
                        return MockDataset
                    elif path == "mock_adapter":
                        return MockAdapter
                    elif path == "mock_accuracy":
                        return MockMetric
                raise ImportError(f"Unknown class: {path}")

            mock_import.side_effect = mock_import_side_effect

            # Mock user input to skip examples
            with patch("builtins.input", side_effect=["s", "s", "s", "q"]):
                result = runner.invoke(app, ["run", str(spec_file), "--interactive"])

                # Should exit gracefully even in interactive mode
                assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__])
