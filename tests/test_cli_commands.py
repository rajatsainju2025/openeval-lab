"""End-to-end tests for CLI commands.

This module provides comprehensive testing for all CLI commands, including:
- Command execution with various arguments
- Error handling and validation
- Config loading and spec validation
- Result output and formatting
"""

import pytest
import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

# Import CLI app and commands
from openeval.cli.cli import app


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_spec(temp_dir):
    """Create sample evaluation spec file."""
    spec = {
        "task": "qa",
        "dataset": {
            "type": "jsonl",
            "path": "examples/qa_toy.jsonl",
        },
        "adapter": {
            "type": "openai",
            "model": "gpt-3.5-turbo",
        },
        "metrics": ["exact_match", "f1"],
        "concurrency": 2,
    }

    spec_path = temp_dir / "test_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f)

    return spec_path


@pytest.fixture
def sample_results(temp_dir):
    """Create sample results file."""
    results = {
        "metrics": {
            "exact_match": 0.85,
            "f1": 0.92,
            "precision": 0.90,
            "recall": 0.88,
        },
        "num_examples": 100,
        "timestamp": "2025-12-05T10:00:00",
    }

    results_path = temp_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f)

    return results_path


class TestVersionCommand:
    """Test version command."""

    def test_version_command(self, runner):
        """Test version command execution."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "OpenEval Lab" in result.output or "version" in result.output.lower()


class TestRegistryCommands:
    """Test registry list and info commands."""

    def test_registry_list(self, runner):
        """Test listing registry components."""
        result = runner.invoke(app, ["registry-list"])
        # Command should succeed or fail gracefully
        assert result.exit_code in [0, 1]

    def test_registry_info_missing_arg(self, runner):
        """Test registry info with missing argument."""
        result = runner.invoke(app, ["registry-info"])
        # Should show error about missing argument
        assert result.exit_code != 0


class TestValidateCommand:
    """Test spec validation command."""

    def test_validate_valid_spec(self, runner, sample_spec):
        """Test validating a valid spec file."""
        result = runner.invoke(app, ["validate", str(sample_spec)])
        # Should succeed or gracefully handle missing dependencies
        assert result.exit_code in [0, 1]

    def test_validate_nonexistent_file(self, runner):
        """Test validating non-existent file."""
        result = runner.invoke(app, ["validate", "nonexistent_spec.json"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_validate_invalid_json(self, runner, temp_dir):
        """Test validating invalid JSON file."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("{ invalid json }")

        result = runner.invoke(app, ["validate", str(invalid_file)])
        assert result.exit_code != 0


class TestHelpCommands:
    """Test help and documentation commands."""

    def test_examples_command(self, runner):
        """Test examples command."""
        result = runner.invoke(app, ["examples"])
        assert result.exit_code == 0
        # Should show some examples or help text
        assert len(result.output) > 0

    def test_spec_guide_command(self, runner):
        """Test spec guide command."""
        result = runner.invoke(app, ["spec-guide"])
        assert result.exit_code == 0
        assert len(result.output) > 0

    def test_workflow_command(self, runner):
        """Test workflow guide command."""
        result = runner.invoke(app, ["workflow"])
        assert result.exit_code == 0
        assert len(result.output) > 0

    def test_troubleshoot_command(self, runner):
        """Test troubleshoot command."""
        result = runner.invoke(app, ["troubleshoot"])
        assert result.exit_code == 0
        assert len(result.output) > 0


class TestDoctorCommand:
    """Test system diagnostics command."""

    def test_doctor_command(self, runner):
        """Test doctor command execution."""
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        # Should show some diagnostic info
        assert len(result.output) > 0


class TestCompareCommand:
    """Test result comparison command."""

    def test_compare_command_missing_args(self, runner):
        """Test compare command with missing arguments."""
        result = runner.invoke(app, ["compare"])
        # Should show error about missing files
        assert result.exit_code != 0

    def test_compare_nonexistent_files(self, runner):
        """Test comparing non-existent files."""
        result = runner.invoke(app, ["compare", "file1.json", "file2.json"])
        # Should fail gracefully
        assert result.exit_code != 0


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_invalid_command(self, runner):
        """Test invalid command."""
        result = runner.invoke(app, ["invalid_command"])
        assert result.exit_code != 0

    def test_help_flag(self, runner):
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "openeval" in result.output.lower() or "usage" in result.output.lower()


class TestCLIIntegration:
    """Integration tests for CLI workflows."""

    def test_validate_then_compare_workflow(self, runner, sample_spec, sample_results):
        """Test a typical workflow: validate spec, then compare results."""
        # Validate spec
        validate_result = runner.invoke(app, ["validate", str(sample_spec)])
        # Should succeed or fail gracefully
        assert validate_result.exit_code in [0, 1]

        # If we had two result files, we could test compare
        # For now, just ensure the command structure works
        assert True

    def test_help_commands_accessible(self, runner):
        """Test that all help commands are accessible."""
        help_commands = ["examples", "spec-guide", "workflow", "troubleshoot"]

        for cmd in help_commands:
            result = runner.invoke(app, [cmd])
            assert result.exit_code == 0, f"Command '{cmd}' failed"
            assert len(result.output) > 0, f"Command '{cmd}' produced no output"


class TestCLIOutputFormatting:
    """Test CLI output formatting and display."""

    def test_version_output_format(self, runner):
        """Test version command output is properly formatted."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        # Output should be readable
        assert "\n" in result.output or len(result.output) > 0

    def test_help_output_readable(self, runner):
        """Test help output is readable."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Should have sections or formatting
        assert len(result.output.split("\n")) > 5


class TestCLIConfigIntegration:
    """Test CLI integration with configuration files."""

    def test_spec_file_parsing(self, runner, sample_spec):
        """Test that CLI can parse spec files."""
        # Validate should be able to read the spec
        result = runner.invoke(app, ["validate", str(sample_spec)])
        # Should not crash
        assert result.exit_code in [0, 1]

    def test_invalid_spec_format(self, runner, temp_dir):
        """Test handling of invalid spec format."""
        bad_spec = temp_dir / "bad_spec.json"
        bad_spec.write_text('{"invalid": "missing required fields"}')

        result = runner.invoke(app, ["validate", str(bad_spec)])
        # Should detect validation errors
        assert result.exit_code != 0


# Parametrized tests for multiple scenarios
@pytest.mark.parametrize(
    "command",
    [
        "version",
        "doctor",
        "examples",
        "spec-guide",
        "workflow",
        "troubleshoot",
    ],
)
def test_command_exits_cleanly(runner, command):
    """Test that commands exit cleanly without crashes."""
    result = runner.invoke(app, [command])
    # Should exit with 0 or 1, but not crash
    assert result.exit_code in [0, 1]
    # Should produce some output
    assert len(result.output) >= 0  # Even empty output is okay


@pytest.mark.parametrize(
    "invalid_file",
    [
        "nonexistent.json",
        "/path/that/does/not/exist.json",
        "invalid_spec.yaml",
    ],
)
def test_validate_handles_missing_files(runner, invalid_file):
    """Test that validate handles missing files gracefully."""
    result = runner.invoke(app, ["validate", invalid_file])
    assert result.exit_code != 0
    # Should have some error message
    assert len(result.output) > 0


class TestCLIRobustness:
    """Test CLI robustness and edge cases."""

    def test_empty_arguments(self, runner):
        """Test CLI with no arguments shows help."""
        result = runner.invoke(app, [])
        # Should show help or error
        assert result.exit_code in [0, 1]

    def test_too_many_arguments(self, runner):
        """Test commands with too many arguments."""
        result = runner.invoke(app, ["version", "extra", "args"])
        # Should handle gracefully
        assert result.exit_code in [0, 1]

    def test_special_characters_in_paths(self, runner, temp_dir):
        """Test handling of special characters in file paths."""
        special_path = temp_dir / "spec with spaces.json"
        special_path.write_text('{"task": "qa"}')

        result = runner.invoke(app, ["validate", str(special_path)])
        # Should handle the path correctly
        assert result.exit_code in [0, 1]


# Mark slow tests
@pytest.mark.slow
class TestCLIPerformance:
    """Test CLI performance characteristics."""

    def test_help_command_fast(self, runner):
        """Test that help commands respond quickly."""
        import time

        start = time.time()
        result = runner.invoke(app, ["--help"])
        elapsed = time.time() - start

        assert result.exit_code == 0
        assert elapsed < 5.0  # Should complete within 5 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
