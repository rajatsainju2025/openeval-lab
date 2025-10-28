"""End-to-end integration tests for OpenEval Lab workflows."""

import json

import pytest

from openeval.adapters.echo import EchoAdapter
from openeval.core.example import Example
from openeval.datasets.jsonl import JSONLinesDataset
from openeval.metrics.accuracy import ExactMatch
from openeval.tasks.qa import QATask


def test_full_evaluation_pipeline(tmp_path):
    """Test complete evaluation pipeline from spec to results."""
    # Create dataset
    dataset_path = tmp_path / "qa_dataset.jsonl"
    examples = [
        {"id": "q1", "input": "What is 2+2?", "reference": "4"},
        {"id": "q2", "input": "What is the capital of France?", "reference": "Paris"},
        {"id": "q3", "input": "What color is the sky?", "reference": "blue"},
    ]
    dataset_path.write_text("\n".join(json.dumps(ex) for ex in examples))

    # Initialize components
    task = QATask()
    adapter = EchoAdapter()
    dataset = JSONLinesDataset(path=dataset_path)
    metric = ExactMatch()

    # Run evaluation
    result = task.evaluate(
        adapter,  # type: ignore
        dataset,
        [metric],
        concurrency=2,
        max_retries=1,
        request_timeout=10.0,
    )

    # Verify results
    assert result["size"] == 3
    assert metric.name in result["metrics"]
    assert "accuracy" in result["metrics"][metric.name]


def test_evaluation_with_output_file(tmp_path):
    """Test evaluation saves results to file."""
    # Create minimal dataset
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text('{"id": "1", "input": "test", "reference": "test"}\n')

    output_path = tmp_path / "results.json"

    # Create and run evaluation
    task = QATask()
    adapter = EchoAdapter()
    dataset = JSONLinesDataset(path=dataset_path)
    metric = ExactMatch()

    result = task.evaluate(
        adapter,  # type: ignore
        dataset,
        [metric],
        concurrency=1,
        max_retries=0,
        request_timeout=None,
    )

    # Save results manually (simulating what CLI does)
    output_path.write_text(json.dumps(result, indent=2))

    # Verify output file
    assert output_path.exists()
    loaded_result = json.loads(output_path.read_text())
    assert loaded_result["size"] == 1
    assert metric.name in loaded_result["metrics"]


def test_evaluation_with_custom_instruction(tmp_path):
    """Test QA task with custom instruction."""
    dataset_path = tmp_path / "custom.jsonl"
    dataset_path.write_text('{"id": "1", "input": "Hi", "reference": "Hello"}\n')

    # Custom instruction
    custom_instruction = "Translate the following to a greeting:"
    task = QATask(instruction=custom_instruction)

    # Build prompt to verify instruction
    example = Example(id="test", input="Hi", reference="Hello")
    prompt = task.build_prompt(example)

    assert custom_instruction in prompt
    assert "Hi" in prompt


def test_evaluation_with_few_shot_examples(tmp_path):
    """Test QA task with few-shot learning."""
    dataset_path = tmp_path / "fewshot.jsonl"
    dataset_path.write_text('{"id": "1", "input": "What is 3+3?", "reference": "6"}\n')

    # Few-shot examples
    few_shot = [
        {"input": "What is 1+1?", "reference": "2"},
        {"input": "What is 2+2?", "reference": "4"},
    ]

    task = QATask(few_shot_examples=few_shot)

    # Build prompt to verify few-shot
    example = Example(id="test", input="What is 3+3?", reference="6")
    prompt = task.build_prompt(example)

    assert "What is 1+1?" in prompt
    assert "2" in prompt
    assert "What is 2+2?" in prompt
    assert "4" in prompt
    assert "What is 3+3?" in prompt


def test_multiple_metrics(tmp_path):
    """Test evaluation with multiple metrics."""
    dataset_path = tmp_path / "multi_metric.jsonl"
    examples = [
        {"id": "1", "input": "Q1", "reference": "A1"},
        {"id": "2", "input": "Q2", "reference": "A2"},
    ]
    dataset_path.write_text("\n".join(json.dumps(ex) for ex in examples))

    task = QATask()
    adapter = EchoAdapter()
    dataset = JSONLinesDataset(path=dataset_path)

    # Multiple metrics (if available)
    metrics = [ExactMatch()]

    result = task.evaluate(
        adapter,  # type: ignore
        dataset,
        metrics,
        concurrency=1,
        max_retries=0,
        request_timeout=None,
    )

    # Verify all metrics computed
    for metric in metrics:
        assert metric.name in result["metrics"]


def test_dataset_validation(tmp_path):
    """Test dataset loading and validation."""
    # Test valid dataset
    valid_path = tmp_path / "valid.jsonl"
    valid_path.write_text('{"id": "1", "input": "test", "reference": "answer"}\n')

    ds = JSONLinesDataset(path=valid_path)
    assert len(ds) == 1
    examples = list(ds)
    assert len(examples) == 1
    assert examples[0].input == "test"
    assert examples[0].reference == "answer"


def test_evaluation_error_handling(tmp_path):
    """Test error handling during evaluation."""
    # Create dataset with valid data
    dataset_path = tmp_path / "error_test.jsonl"
    dataset_path.write_text('{"id": "1", "input": "test", "reference": "ref"}\n')

    task = QATask()
    adapter = EchoAdapter()
    dataset = JSONLinesDataset(path=dataset_path)
    metric = ExactMatch()

    # This should succeed with echo adapter
    result = task.evaluate(
        adapter,  # type: ignore
        dataset,
        [metric],
        concurrency=1,
        max_retries=0,
        request_timeout=None,
    )

    assert result["size"] == 1


def test_concurrent_requests(tmp_path):
    """Test concurrent request handling."""
    # Create larger dataset
    dataset_path = tmp_path / "concurrent.jsonl"
    examples = [
        {"id": str(i), "input": f"Question {i}", "reference": f"Answer {i}"} for i in range(10)
    ]
    dataset_path.write_text("\n".join(json.dumps(ex) for ex in examples))

    task = QATask()
    adapter = EchoAdapter()
    dataset = JSONLinesDataset(path=dataset_path)
    metric = ExactMatch()

    # Test with different concurrency levels
    for concurrency in [1, 2, 5]:
        result = task.evaluate(
            adapter,  # type: ignore
            dataset,
            [metric],
            concurrency=concurrency,
            max_retries=0,
            request_timeout=None,
        )
        assert result["size"] == 10
        assert metric.name in result["metrics"]


def test_prompt_template_rendering(tmp_path):
    """Test prompt template rendering with variables."""
    task = QATask(instruction="Answer this question carefully.")

    example = Example(id="test-1", input="What is AI?", reference="Artificial Intelligence")

    prompt = task.build_prompt(example)

    # Verify instruction and input are in prompt
    assert "Answer this question carefully" in prompt
    assert "What is AI?" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
