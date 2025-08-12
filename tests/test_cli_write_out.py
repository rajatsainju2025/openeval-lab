from pathlib import Path
from typer.testing import CliRunner

from openeval.cli import app


def test_write_out_preview(tmp_path: Path):
    # create tiny spec and dataset
    ds = tmp_path / "toy.jsonl"
    ds.write_text('{"id": 1, "input": "Q?", "reference": "A"}\n')
    spec = tmp_path / "spec.json"
    spec.write_text(
        __import__("json").dumps(
            {
                "task": "qa",
                "dataset": "jsonl",
                "adapter": "echo",
                "dataset_kwargs": {"path": str(ds)},
                "metrics": [{"name": "exact_match"}],
                "output": str(tmp_path / "out.json"),
            }
        )
    )
    res = CliRunner().invoke(app, ["write_out", str(spec), "--limit", "1"])
    assert res.exit_code == 0
    assert "preview_count" in res.stdout


def test_write_out_file(tmp_path: Path):
    ds = tmp_path / "toy.jsonl"
    ds.write_text('{"id": 1, "input": "Q?", "reference": "A"}\n')
    out = tmp_path / "prompts.jsonl"
    spec = tmp_path / "spec.json"
    spec.write_text(
        __import__("json").dumps(
            {
                "task": "qa",
                "dataset": "jsonl",
                "adapter": "echo",
                "dataset_kwargs": {"path": str(ds)},
                "metrics": [{"name": "exact_match"}],
                "output": str(tmp_path / "out.json"),
            }
        )
    )
    res = CliRunner().invoke(app, ["write_out", str(spec), "--out", str(out)])
    assert res.exit_code == 0
    assert out.exists()
    # one JSONL line
    assert len(out.read_text().strip().splitlines()) == 1
