import json
from typer.testing import CliRunner

from openeval.cli import app


runner = CliRunner()


def test_results_schema_prints_json():
    res = runner.invoke(app, ["results-schema"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload.get("title") == "OpenEval Results"
    assert payload.get("type") == "object"


def test_validate_results_happy_path(tmp_path):
    # Minimal valid payload
    payload = {
        "task": "qa",
        "dataset": "jsonl",
        "adapter": "echo",
        "size": 1,
        "metrics": {"exact_match": {"value": 1.0}}
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(payload))

    res = runner.invoke(app, ["validate-results", str(p)])
    assert res.exit_code == 0
    out = json.loads(res.stdout)
    assert out["valid"] is True


def test_validate_results_strict_failure(tmp_path):
    # Missing required key 'size'
    payload = {
        "task": "qa",
        "dataset": "jsonl",
        "adapter": "echo",
        "metrics": {}
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(payload))

    res = runner.invoke(app, ["validate-results", str(p), "--strict"])
    assert res.exit_code != 0
    out = json.loads(res.stdout)
    assert out["valid"] is False
    assert any("missing required key" in e for e in out["errors"])
