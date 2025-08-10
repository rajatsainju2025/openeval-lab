from openeval.tasks.summarization import SummarizationTask
from openeval.adapters.echo import EchoAdapter
from openeval.datasets.jsonl import JSONLinesDataset


def test_summarization_smoke(tmp_path):
    p = tmp_path / "sum.jsonl"
    p.write_text('{"id":1, "input":"A. B.", "reference":"A B"}\n')
    task = SummarizationTask(max_words=10)
    adapter = EchoAdapter()
    ds = JSONLinesDataset(path=p)
    result = task.evaluate(adapter, ds, metrics=[], concurrency=1)
    assert result["size"] == 1
