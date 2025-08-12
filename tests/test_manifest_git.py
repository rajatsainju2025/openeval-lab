from openeval.tasks.qa import QATask
from openeval.adapters.echo import EchoAdapter
from openeval.datasets.jsonl import JSONLinesDataset
from openeval.metrics.accuracy import ExactMatch


def test_manifest_includes_git_optional(tmp_path):
    p = tmp_path / "toy.jsonl"
    p.write_text('{"id": 1, "input": "Q?", "reference": "A"}\n')

    task = QATask()
    adapter = EchoAdapter()
    ds = JSONLinesDataset(path=p)
    metric = ExactMatch()

    result = task.evaluate(adapter, ds, [metric])
    # git info may or may not be present depending on git availability
    manifest = result.get("manifest", {})
    if "git" in manifest:
        assert isinstance(manifest["git"], dict)
        assert "commit" in manifest["git"]
