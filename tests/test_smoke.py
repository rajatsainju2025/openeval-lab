from openeval.tasks.qa import QATask
from openeval.adapters.echo import EchoAdapter
from openeval.datasets.jsonl import JSONLinesDataset
from openeval.metrics.accuracy import ExactMatch
from openeval.core.example import Example


def test_smoke(tmp_path):
    # Prepare a tiny dataset
    p = tmp_path / "toy.jsonl"
    p.write_text('{"id": 1, "input": "What is 2+2?", "reference": "4"}\n')

    task = QATask()
    adapter = EchoAdapter()
    ds = JSONLinesDataset(path=p)
    metric = ExactMatch()

    result = task.evaluate(
        adapter, ds, [metric], concurrency=1, max_retries=0, request_timeout=None  # type: ignore
    )
    assert result["size"] == 1
    assert metric.name in result["metrics"]
    assert "accuracy" in result["metrics"][metric.name]


def test_empty_dataset(tmp_path):
    """Test handling of empty datasets"""
    p = tmp_path / "empty.jsonl"
    p.write_text("")  # Empty file

    ds = JSONLinesDataset(path=p)
    assert len(ds) == 0

    task = QATask()
    adapter = EchoAdapter()
    metric = ExactMatch()

    result = task.evaluate(
        adapter, ds, [metric], concurrency=1, max_retries=0, request_timeout=None  # type: ignore
    )
    assert result["size"] == 0
    assert metric.name in result["metrics"]


def test_metric_calculation():
    """Test metric computation directly"""
    metric = ExactMatch()
    predictions = ["4", "five", "6"]
    references = ["4", "5", "6"]

    scores = metric.compute(predictions, references)
    assert scores["accuracy"] == 2 / 3  # 2 out of 3 correct

    # Test perfect accuracy
    perfect_scores = metric.compute(["a", "b"], ["a", "b"])
    assert perfect_scores["accuracy"] == 1.0

    # Test zero accuracy
    zero_scores = metric.compute(["x", "y"], ["a", "b"])
    assert zero_scores["accuracy"] == 0.0


def test_dataset_iteration(tmp_path):
    """Test dataset iteration and data access"""
    p = tmp_path / "test.jsonl"
    p.write_text(
        '{"id": 1, "input": "Q1", "reference": "A1"}\n'
        '{"id": 2, "input": "Q2", "reference": "A2"}\n'
    )

    ds = JSONLinesDataset(path=p)
    assert len(ds) == 2

    examples = list(ds)
    assert len(examples) == 2
    assert examples[0].input == "Q1"
    assert examples[0].reference == "A1"
    assert examples[1].input == "Q2"
    assert examples[1].reference == "A2"


def test_adapter_functionality():
    """Test adapter basic functionality"""
    adapter = EchoAdapter()

    # Test single prediction
    result = adapter.generate("test input")
    assert result == "test input"  # EchoAdapter returns input as-is

    # Test with logprobs
    logprob_result = adapter.generate_with_logprobs("test input")
    assert logprob_result["text"] == "test input"
    assert "tokens" in logprob_result
    assert "logprobs" in logprob_result
    assert "usage" in logprob_result


def test_task_prompt_building():
    """Test task prompt building"""
    task = QATask()

    # Create a proper Example instance
    example = Example(id="test-1", input="What is 2+2?", reference="4")
    prompt = task.build_prompt(example)
    assert "What is 2+2?" in prompt
    assert "Answer the question concisely" in prompt


def test_cli(tmp_path):
    from openeval.cli import app
    from typer.testing import CliRunner

    spec = tmp_path / "spec.json"
    data = {
        "task": "openeval.tasks.qa.QATask",
        "dataset": "openeval.datasets.jsonl.JSONLinesDataset",
        "adapter": "openeval.adapters.echo.EchoAdapter",
        "dataset_kwargs": {"path": str(tmp_path / "toy.jsonl")},
        "metrics": [{"name": "openeval.metrics.accuracy.ExactMatch"}],
        "output": str(tmp_path / "out.json"),
    }
    (tmp_path / "toy.jsonl").write_text('{"id":1,"input":"x","reference":"x"}\n')
    spec.write_text(__import__("json").dumps(data))

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "run",
            "spec",
            str(spec),
            "--max-concurrent",
            "2",
            "--request-timeout",
            "1",
        ],
    )
    assert res.exit_code == 0
    assert (tmp_path / "out.json").exists()


def test_cli_validation_error(tmp_path):
    """Test CLI validation with invalid spec"""
    from openeval.cli import app
    from typer.testing import CliRunner

    spec = tmp_path / "invalid_spec.json"
    # Invalid spec - missing required fields
    data = {"task": "invalid.task"}
    spec.write_text(__import__("json").dumps(data))

    runner = CliRunner()
    res = runner.invoke(app, ["run", "spec", str(spec)])

    # Should fail with validation error
    assert res.exit_code != 0


def test_concurrent_evaluation(tmp_path):
    """Test evaluation with concurrency"""
    p = tmp_path / "concurrent.jsonl"
    # Create multiple examples to test concurrency
    data = "\n".join(
        [f'{{"id": {i}, "input": "Question {i}", "reference": "Answer {i}"}}' for i in range(1, 6)]
    )
    p.write_text(data)

    task = QATask()
    adapter = EchoAdapter()
    ds = JSONLinesDataset(path=p)
    metric = ExactMatch()

    # Test with concurrency > 1
    result = task.evaluate(
        adapter, ds, [metric], concurrency=2, max_retries=0, request_timeout=None  # type: ignore
    )
    assert result["size"] == 5
    assert metric.name in result["metrics"]
    assert "accuracy" in result["metrics"][metric.name]
