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

    result = task.evaluate(adapter, ds, [metric])
    assert result["size"] == 1
    assert metric.name in result["metrics"]
    assert "accuracy" in result["metrics"][metric.name]
