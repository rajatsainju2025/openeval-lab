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


def test_registry_new_metrics():
    """Test that newly added metrics are available in registry."""
    # Test f1_score metric
    res = runner.invoke(app, ["registry-info", "metric", "f1_score"])
    assert res.exit_code == 0
    assert "f1_score" in res.stdout
    assert "precision and recall" in res.stdout

    # Test calibration_error metric
    res = runner.invoke(app, ["registry-info", "metric", "calibration_error"])
    assert res.exit_code == 0
    assert "calibration_error" in res.stdout
    assert "Expected Calibration Error" in res.stdout


def test_registry_new_adapters():
    """Test that newly added adapters are available in registry."""
    # Test anthropic adapter
    res = runner.invoke(app, ["registry-info", "adapter", "anthropic"])
    assert res.exit_code == 0
    assert "anthropic" in res.stdout
    assert "Anthropic Claude" in res.stdout

    # Test vllm adapter
    res = runner.invoke(app, ["registry-info", "adapter", "vllm"])
    assert res.exit_code == 0
    assert "vllm" in res.stdout
    assert "high-throughput" in res.stdout


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


def test_registry_invalid_kind():
    """Test error handling for invalid registry kind."""
    res = runner.invoke(app, ["registry-list", "invalid_kind"])
    assert res.exit_code == 1
    assert "Unknown registry kind" in res.stdout
    assert "Available kinds:" in res.stdout
    assert "task" in res.stdout
    assert "dataset" in res.stdout
    assert "adapter" in res.stdout
    assert "metric" in res.stdout


def test_runs_collect_writes_index(tmp_path):
    # Use a temporary runs dir with no JSONs
    out = tmp_path / "index.json"
    res = runner.invoke(app, ["runs", "collect", "--dir", str(tmp_path), "--out", str(out)])
    assert res.exit_code == 0
    assert out.exists()
    data = out.read_text()
    assert "runs" in data
