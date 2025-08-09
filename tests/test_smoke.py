from openeval.core import Example
from openeval.tasks.qa import QATask
from openeval.adapters.echo import EchoAdapter
from openeval.datasets.jsonl import JSONLinesDataset
from openeval.metrics.accuracy import ExactMatch


def test_smoke(tmp_path):
    # Prepare a tiny dataset
    p = tmp_path / "toy.jsonl"
    p.write_text('{"id": 1, "input": "What is 2+2?", "reference": "4"}\n')

    task = QATask()
    adapter = EchoAdapter()
    ds = JSONLinesDataset(path=p)
    metric = ExactMatch()

    result = task.evaluate(adapter, ds, [metric], concurrency=1, max_retries=0, request_timeout=None)
    assert result["size"] == 1
    assert metric.name in result["metrics"]
    assert "accuracy" in result["metrics"][metric.name]


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
    res = runner.invoke(app, [
        "run",
        str(spec),
        "--concurrency",
        "2",
        "--max-retries",
        "1",
        "--request-timeout",
        "1",
    ])
    assert res.exit_code == 0
    assert (tmp_path / "out.json").exists()
