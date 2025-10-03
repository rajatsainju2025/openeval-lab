"""
Test dashboard export functionality.
"""

import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from openeval.web.app import app


def test_export_json():
    """Test JSON export endpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake run file
        runs_dir = Path(tmpdir) / "runs"
        runs_dir.mkdir()
        run_file = runs_dir / "test_run.json"

        test_data = {
            "task": "qa",
            "dataset": "toy",
            "adapter": "echo",
            "metrics": {"exact_match": {"accuracy": 0.8}},
            "records": [
                {"id": "1", "input": "Q1", "reference": "A1", "prediction": "A1"},
                {"id": "2", "input": "Q2", "reference": "A2", "prediction": "A2"},
            ],
        }
        run_file.write_text(json.dumps(test_data))

        # Change to tmpdir to make runs/ discoverable
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            with TestClient(app) as client:
                response = client.get("/export/test_run.json?format=json")
                assert response.status_code == 200
                assert response.headers["content-type"] == "application/json"
                assert "attachment" in response.headers["content-disposition"]

                exported_data = json.loads(response.content)
                assert exported_data["task"] == "qa"
                assert len(exported_data["records"]) == 2
        finally:
            os.chdir(original_cwd)


def test_export_csv():
    """Test CSV export endpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake run file with records
        runs_dir = Path(tmpdir) / "runs"
        runs_dir.mkdir()
        run_file = runs_dir / "test_run.json"

        test_data = {
            "task": "qa",
            "records": [
                {
                    "id": "1",
                    "input": "Q1",
                    "reference": "A1",
                    "prediction": "A1",
                    "latency_ms": 100.0,
                },
                {
                    "id": "2",
                    "input": "Q2",
                    "reference": "A2",
                    "prediction": "A2",
                    "latency_ms": 150.0,
                },
            ],
        }
        run_file.write_text(json.dumps(test_data))

        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            with TestClient(app) as client:
                response = client.get("/export/test_run.json?format=csv")
                assert response.status_code == 200
                assert "text/csv" in response.headers["content-type"]
                assert "test_run.csv" in response.headers["content-disposition"]

                csv_content = response.content.decode()
                assert "id,input,reference,prediction,latency_ms" in csv_content
                assert "1,Q1,A1,A1,100.0" in csv_content
        finally:
            os.chdir(original_cwd)


def test_export_nonexistent_file():
    """Test export endpoint with nonexistent file."""
    with TestClient(app) as client:
        response = client.get("/export/nonexistent.json")
        assert response.status_code == 404
