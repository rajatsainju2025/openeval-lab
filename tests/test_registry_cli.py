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


def test_doctor_json_output():
    res = runner.invoke(app, ["doctor", "--json"])
    assert res.exit_code == 0
    # Output should be JSON with required keys
    import json as _json
    payload = _json.loads(res.stdout)
    assert "python" in payload
    assert "packages" in payload
    assert "api_keys" in payload
    assert "filesystem" in payload
    assert "registry" in payload


def test_runs_collect_writes_index(tmp_path):
    # Use a temporary runs dir with no JSONs
    out = tmp_path / "index.json"
    res = runner.invoke(app, ["runs", "collect", "--dir", str(tmp_path), "--out", str(out)])
    assert res.exit_code == 0
    assert out.exists()
    data = out.read_text()
    assert "runs" in data
