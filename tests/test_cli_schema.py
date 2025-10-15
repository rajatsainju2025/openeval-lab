from typer.testing import CliRunner
import json

from openeval.cli import app


runner = CliRunner()


def test_schema_prints_json():
    res = runner.invoke(app, ["schema"])
    assert res.exit_code == 0
    # Output should be JSON; parse the first/last braces
    out = res.stdout.strip()
    assert out.startswith("{") and out.endswith("}")


def test_schema_writes_file(tmp_path):
    out = tmp_path / "schema.json"
    res = runner.invoke(app, ["schema", "--out", str(out)])
    assert res.exit_code == 0
    data = json.loads(out.read_text())
    assert data.get("title")


def test_results_schema_prints_json():
    res = runner.invoke(app, ["results-schema"])
    assert res.exit_code == 0
    out = res.stdout.strip()
    assert out.startswith("{") and out.endswith("}")


def test_results_schema_writes_file(tmp_path):
    out = tmp_path / "results_schema.json"
    res = runner.invoke(app, ["results-schema", "--out", str(out)])
    assert res.exit_code == 0
    data = json.loads(out.read_text())
    assert data.get("title") == "OpenEval Results"
