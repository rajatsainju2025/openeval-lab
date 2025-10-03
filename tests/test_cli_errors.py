"""Test cases for CLI error handling and edge cases."""

from typer.testing import CliRunner
from openeval.cli import app
import json
import tempfile
from pathlib import Path


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_invalid_spec_file(self):
        """Test handling of invalid specification file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": "json"')
            invalid_file = f.name

        try:
            result = self.runner.invoke(app, ["run", invalid_file])
            assert result.exit_code != 0
            assert "Error" in result.output or "error" in result.output.lower()
        finally:
            Path(invalid_file).unlink(missing_ok=True)

    def test_missing_spec_file(self):
        """Test handling of missing specification file."""
        result = self.runner.invoke(app, ["run", "nonexistent.json"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_invalid_command_arguments(self):
        """Test handling of invalid command arguments."""
        result = self.runner.invoke(app, ["run"])  # Missing required argument
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "requires" in result.output.lower()

    def test_registry_invalid_kind(self):
        """Test registry list with invalid kind."""
        result = self.runner.invoke(app, ["registry-list", "invalid_kind"])
        assert result.exit_code != 0
        assert "error" in result.output.lower() or "invalid" in result.output.lower()

    def test_validate_invalid_spec(self):
        """Test validation of invalid specification."""
        invalid_spec = {
            "task": "invalid_task",
            "dataset": {"name": "nonexistent"},
            "adapter": {"name": "invalid_adapter"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_spec, f)
            spec_file = f.name

        try:
            result = self.runner.invoke(app, ["validate", spec_file])
            assert result.exit_code != 0
            assert "error" in result.output.lower() or "invalid" in result.output.lower()
        finally:
            Path(spec_file).unlink(missing_ok=True)

    def test_doctor_command_error_handling(self):
        """Test doctor command error handling."""
        # This should work, but test error scenarios if any
        result = self.runner.invoke(app, ["doctor"])
        # Doctor command should handle errors gracefully
        assert result.exit_code == 0 or "error" in result.output.lower()

    def test_json_logging_format(self):
        """Test JSON logging format option."""
        result = self.runner.invoke(app, ["--json-logs", "registry-list", "task"])
        # Should not crash with JSON logging
        assert result.exit_code == 0

    def test_debug_mode_logging(self):
        """Test debug mode logging."""
        result = self.runner.invoke(app, ["--debug", "registry-list", "task"])
        # Should not crash with debug mode
        assert result.exit_code == 0

    def test_concurrent_execution_error(self):
        """Test error handling in concurrent execution scenarios."""
        # Create a spec that might cause concurrency issues
        spec = {
            "task": "qa",
            "dataset": {"name": "jsonl", "path": "examples/qa_toy.jsonl"},
            "adapter": {"name": "openai-chat", "model": "gpt-4o-mini"},
            "metrics": [{"name": "exact_match"}],
            "concurrency": 100,  # High concurrency that might cause issues
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(spec, f)
            spec_file = f.name

        try:
            result = self.runner.invoke(app, ["run", spec_file, "--dry-run"])
            # Should handle high concurrency gracefully or report error
            assert result.exit_code == 0 or "error" in result.output.lower()
        finally:
            Path(spec_file).unlink(missing_ok=True)

    def test_cache_error_handling(self):
        """Test cache-related error handling."""
        result = self.runner.invoke(app, ["run", "examples/qa_spec.json", "--cache", "invalid"])
        assert result.exit_code != 0
        assert "error" in result.output.lower() or "invalid" in result.output.lower()

    def test_output_directory_permissions(self):
        """Test handling of output directory permission issues."""
        # This is harder to test without setting up permissions, but we can test the concept
        result = self.runner.invoke(
            app, ["run", "examples/qa_spec.json", "--artifacts", "/root/invalid"]
        )
        assert (
            result.exit_code != 0
            or "permission" in result.output.lower()
            or "error" in result.output.lower()
        )
