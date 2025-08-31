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


def test_doctor_runs():
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 0
    # Should print sections
    assert "Environment Checks" in res.stdout
    assert "API Keys" in res.stdout
    assert "Filesystem" in res.stdout
