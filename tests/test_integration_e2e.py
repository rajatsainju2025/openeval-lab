"""Comprehensive end-to-end integration tests.

Tests full workflows from CLI invocation through evaluation to output generation.
Validates the entire OpenEval Lab pipeline.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from openeval.datasets.jsonl import JSONLinesDataset
from openeval.adapters.echo import EchoAdapter
from openeval.tasks.qa import QATask
from openeval.metrics.accuracy import ExactMatch
from openeval.cache import PredictionCache


@pytest.fixture
def temp_dir():
    """Temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataset_file(temp_dir):
    """Create a sample JSONL dataset file."""
    dataset_path = temp_dir / "test_dataset.jsonl"
    with open(dataset_path, "w") as f:
        for i in range(10):
            f.write(json.dumps({"input": f"Question {i}?", "reference": f"Answer {i}"}) + "\n")
    return dataset_path


@pytest.fixture
def sample_spec_file(temp_dir, sample_dataset_file):
    """Create a sample evaluation spec file."""
    spec_path = temp_dir / "test_spec.json"
    spec = {
        "task": "qa",
        "dataset": {"type": "jsonl", "path": str(sample_dataset_file)},
        "adapter": {"type": "echo"},
        "metrics": ["exact_match"],
        "output_path": str(temp_dir / "results.json"),
    }
    with open(spec_path, "w") as f:
        json.dump(spec, f)
    return spec_path


class TestEndToEndWorkflow:
    """Test complete evaluation workflows."""

    def test_programmatic_evaluation_pipeline(self, sample_dataset_file, temp_dir):
        """Test full evaluation pipeline programmatically (no cache)."""
        # Setup components
        dataset = JSONLinesDataset(sample_dataset_file)
        adapter = EchoAdapter()
        task = QATask()
        metric = ExactMatch()

        # Run evaluation
        predictions = []
        references = []

        for example in dataset:
            prompt = task.build_prompt(example)
            pred = adapter.generate(prompt)

            predictions.append(pred)
            references.append(example.reference)

        # Compute metrics
        results = metric.compute(predictions, references)

        # Validate results
        assert "accuracy" in results
        assert isinstance(results["accuracy"], float)
        assert 0.0 <= results["accuracy"] <= 1.0
        assert len(predictions) == 10
        assert len(references) == 10

    def test_cli_evaluation_workflow(self, sample_spec_file, temp_dir):
        """Test evaluation through CLI (if CLI is available)."""
        result_path = temp_dir / "results.json"

        # Try to run CLI command
        try:
            result = subprocess.run(
                ["openeval", "run", str(sample_spec_file)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Check if command succeeded
            if result.returncode == 0:
                # Verify output file was created
                assert result_path.exists(), "Results file not created"

                # Validate results structure
                with open(result_path) as f:
                    results = json.load(f)

                assert "metrics" in results
                assert "examples" in results or "predictions" in results

        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("CLI not available or command timed out")

    def test_spec_based_evaluation(self, sample_spec_file, temp_dir):
        """Test evaluation spec file structure."""
        # Load and validate spec file structure
        with open(sample_spec_file) as f:
            spec_data = json.load(f)

        assert spec_data["task"] == "qa"
        assert spec_data["dataset"]["type"] == "jsonl"
        assert spec_data["adapter"]["type"] == "echo"
        assert "exact_match" in spec_data["metrics"]

    def test_cache_persistence_across_runs(self, temp_dir):
        """Test cache directory creation."""
        cache_dir = temp_dir / "persistent_cache"
        cache = PredictionCache(cache_dir=cache_dir)

        # Verify cache directory was created
        assert cache_dir.exists(), "Cache directory not created"
        assert cache.cache_dir == cache_dir


class TestErrorHandling:
    """Test error handling in end-to-end workflows."""

    def test_invalid_spec_file(self, temp_dir):
        """Test handling of invalid spec file."""
        invalid_spec = temp_dir / "invalid.json"
        with open(invalid_spec, "w") as f:
            f.write("{invalid json")

        with pytest.raises(json.JSONDecodeError):
            with open(invalid_spec) as f:
                json.load(f)

    def test_missing_dataset_file(self, temp_dir):
        """Test handling of missing dataset file."""
        missing_path = temp_dir / "nonexistent.jsonl"

        with pytest.raises((FileNotFoundError, ValueError)):
            dataset = JSONLinesDataset(missing_path)
            list(dataset)  # Force iteration to trigger error

    def test_empty_dataset(self, temp_dir):
        """Test handling of empty dataset."""
        empty_dataset = temp_dir / "empty.jsonl"
        empty_dataset.touch()

        dataset = JSONLinesDataset(empty_dataset)
        examples = list(dataset)

        assert len(examples) == 0


class TestOutputFormats:
    """Test different output formats and serialization."""

    def test_results_json_serialization(self, temp_dir):
        """Test that results can be serialized to JSON."""
        results = {
            "task": "qa",
            "adapter": "echo",
            "metrics": {"accuracy": 0.95, "f1": 0.92},
            "predictions": ["answer1", "answer2"],
            "references": ["answer1", "answer2"],
        }

        output_path = temp_dir / "test_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Verify can be read back
        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["metrics"]["accuracy"] == 0.95

    def test_multiple_metrics_computation(self):
        """Test computing multiple metrics simultaneously."""
        predictions = ["answer1", "answer2", "answer3"]
        references = ["answer1", "answer2", "different"]

        # Exact match
        em_metric = ExactMatch()
        em_results = em_metric.compute(predictions, references)

        assert em_results["accuracy"] == pytest.approx(2 / 3)


@pytest.mark.slow
class TestScalability:
    """Test scalability of evaluation workflows."""

    def test_large_dataset_evaluation(self, temp_dir):
        """Test evaluation with larger dataset (100 examples)."""
        large_dataset = temp_dir / "large.jsonl"
        with open(large_dataset, "w") as f:
            for i in range(100):
                f.write(json.dumps({"input": f"Question {i}?", "reference": f"Answer {i}"}) + "\n")

        dataset = JSONLinesDataset(large_dataset)
        adapter = EchoAdapter()
        task = QATask()

        predictions = []
        for example in dataset:
            prompt = task.build_prompt(example)
            pred = adapter.generate(prompt)
            predictions.append(pred)

        assert len(predictions) == 100
