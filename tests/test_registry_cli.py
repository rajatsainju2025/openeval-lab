from typer.testing import CliRunner

from openeval.cli import app


runner = CliRunner()


def test_registry_list_tasks():
    res = runner.invoke(app, ["registry-list", "task"])
    assert res.exit_code == 0
    assert "Registry: task" in res.stdout
    assert "qa" in res.stdout


def test_registry_info_metric():
    res = runner.invoke(app, ["registry-info", "metric", "rouge_l"])
    assert res.exit_code == 0
    assert "rouge_l" in res.stdout
    assert "path" in res.stdout
