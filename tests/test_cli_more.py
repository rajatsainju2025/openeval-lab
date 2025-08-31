from pathlib import Path
import json
from typer.testing import CliRunner

from openeval.cli import app


runner = CliRunner()


def test_schema_prints_json():
    res = runner.invoke(app, ["schema"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert isinstance(payload, dict)
    assert "properties" in payload or "$defs" in payload


def test_init_writes_json_and_yaml(tmp_path):
    out_json = tmp_path / "spec.json"
    res = runner.invoke(app, ["init", str(out_json)])
    assert res.exit_code == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert "task" in data and "dataset" in data and "adapter" in data

    out_yaml = tmp_path / "spec.yaml"
    res2 = runner.invoke(app, ["init", str(out_yaml), "--fmt", "yaml"])
    # If yaml extras aren't installed, exit code may be 2
    assert res2.exit_code in (0, 2)
    if out_yaml.exists():
        content = out_yaml.read_text()
        assert "task:" in content


def test_validate_example_spec():
    spec = Path(__file__).resolve().parents[1] / "examples" / "qa_spec.json"
    assert spec.exists()
    res = runner.invoke(app, ["validate", str(spec)])
    assert res.exit_code == 0


def test_docs_lists_files():
    res = runner.invoke(app, ["docs"])
    assert res.exit_code == 0
    assert "Documentation" in res.stdout
    assert "Tutorial" in res.stdout


def test_version_outputs():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "Python version" in res.stdout


def test_write_out_preview():
    spec = Path(__file__).resolve().parents[1] / "examples" / "qa_spec.json"
    res = runner.invoke(app, ["write_out", str(spec), "--limit", "2", "--preview", "1"])
    assert res.exit_code == 0
    assert "preview_count" in res.stdout
