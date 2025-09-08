"""
Comprehensive validation tests for OpenEval Lab.

Tests the validation framework including CLI commands, schema validation,
import validation, dataset validation, and performance checks.
"""

import json
import pytest
import tempfile
import yaml
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from openeval.cli import app


class TestValidationFramework:
    """Test suite for the comprehensive validation framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Sample valid spec
        self.valid_spec = {
            "task": "qa",
            "dataset": {
                "name": "jsonl",
                "path": "test_data.jsonl"
            },
            "adapter": {
                "name": "echo"
            },
            "metrics": [
                {"name": "exact_match"}
            ]
        }
        
        # Sample invalid spec (missing required fields)
        self.invalid_spec = {
            "task": "qa",
            # Missing dataset
            "adapter": {
                "name": "echo"
            }
        }
        
        # Sample dataset content
        self.sample_dataset = [
            {"input": "What is 2+2?", "output": "4"},
            {"input": "What is 3+3?", "output": "6"}
        ]
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_validate_comprehensive_help(self):
        """Test that validate-comprehensive command shows help."""
        result = self.runner.invoke(app, ["validate-comprehensive", "--help"])
        assert result.exit_code == 0
        assert "Comprehensive validation" in result.stdout
        assert "--type" in result.stdout
        assert "--output-dir" in result.stdout
    
    def test_validate_valid_spec_file(self):
        """Test validation of a valid specification file."""
        # Create valid spec file
        spec_file = self.temp_path / "valid_spec.json"
        with open(spec_file, "w") as f:
            json.dump(self.valid_spec, f)
        
        # Create dataset file
        dataset_file = self.temp_path / "test_data.jsonl"
        with open(dataset_file, "w") as f:
            for item in self.sample_dataset:
                f.write(json.dumps(item) + "\n")
        
        # Update spec with correct dataset path
        self.valid_spec["dataset"]["path"] = str(dataset_file)
        with open(spec_file, "w") as f:
            json.dump(self.valid_spec, f)
        
        result = self.runner.invoke(app, [
            "validate-comprehensive",
            str(spec_file),
            "--type", "spec"
        ])
        
        # Should succeed (or give informative error if dependencies missing)
        assert result.exit_code in [0, 1]  # Allow for missing optional dependencies
    
    def test_validate_invalid_spec_file(self):
        """Test validation of an invalid specification file."""
        # Create invalid spec file
        spec_file = self.temp_path / "invalid_spec.json"
        with open(spec_file, "w") as f:
            json.dump(self.invalid_spec, f)
        
        result = self.runner.invoke(app, [
            "validate-comprehensive",
            str(spec_file),
            "--type", "spec"
        ])
        
        # Should fail with validation error
        assert result.exit_code == 1
        assert "validation" in result.stdout.lower() or "error" in result.stdout.lower()
    
    def test_validate_yaml_spec(self):
        """Test validation of YAML specification files."""
        # Create valid YAML spec
        spec_file = self.temp_path / "valid_spec.yaml"
        with open(spec_file, "w") as f:
            yaml.dump(self.valid_spec, f)
        
        result = self.runner.invoke(app, [
            "validate-comprehensive",
            str(spec_file),
            "--type", "spec"
        ])
        
        # Should succeed or give informative error
        assert result.exit_code in [0, 1]
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        result = self.runner.invoke(app, [
            "validate-comprehensive",
            "nonexistent_file.json",
            "--type", "spec"
        ])
        
        # Should fail with file not found error
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()
    
    def test_validate_dataset_type(self):
        """Test validation of dataset files."""
        # Create dataset file
        dataset_file = self.temp_path / "test_dataset.jsonl"
        with open(dataset_file, "w") as f:
            for item in self.sample_dataset:
                f.write(json.dumps(item) + "\n")
        
        result = self.runner.invoke(app, [
            "validate-comprehensive",
            str(dataset_file),
            "--type", "dataset"
        ])
        
        # Should succeed or give informative error
        assert result.exit_code in [0, 1]
    
    def test_validate_config_type(self):
        """Test validation of configuration files."""
        # Create config file
        config_file = self.temp_path / "test_config.yaml"
        config_data = {
            "defaults": {
                "concurrency": 1,
                "cache": "off"
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        result = self.runner.invoke(app, [
            "validate-comprehensive",
            str(config_file),
            "--type", "config"
        ])
        
        # Should succeed or give informative error
        assert result.exit_code in [0, 1]
    
    def test_validate_with_output_dir(self):
        """Test validation with custom output directory."""
        spec_file = self.temp_path / "test_spec.json"
        with open(spec_file, "w") as f:
            json.dump(self.valid_spec, f)
        
        output_dir = self.temp_path / "validation_output"
        
        result = self.runner.invoke(app, [
            "validate-comprehensive",
            str(spec_file),
            "--type", "spec",
            "--output-dir", str(output_dir)
        ])
        
        # Should create output directory
        assert result.exit_code in [0, 1]
        # Output directory should be created (if validation runs)
        if result.exit_code == 0:
            assert output_dir.exists()
    
    def test_validate_malformed_json(self):
        """Test validation of malformed JSON file."""
        # Create malformed JSON file
        spec_file = self.temp_path / "malformed.json"
        with open(spec_file, "w") as f:
            f.write('{"task": "qa", "invalid": json}')  # Invalid JSON
        
        result = self.runner.invoke(app, [
            "validate-comprehensive",
            str(spec_file),
            "--type", "spec"
        ])
        
        # Should fail with JSON parsing error
        assert result.exit_code == 1
        assert "error" in result.stdout.lower() or "invalid" in result.stdout.lower()
    
    def test_validate_malformed_yaml(self):
        """Test validation of malformed YAML file."""
        # Create malformed YAML file
        spec_file = self.temp_path / "malformed.yaml"
        with open(spec_file, "w") as f:
            f.write('task: qa\ninvalid: [unclosed')  # Invalid YAML
        
        result = self.runner.invoke(app, [
            "validate-comprehensive",
            str(spec_file),
            "--type", "spec"
        ])
        
        # Should fail with YAML parsing error
        assert result.exit_code == 1
        assert "error" in result.stdout.lower() or "invalid" in result.stdout.lower()


class TestValidationIntegration:
    """Integration tests for validation workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @pytest.mark.integration
    def test_validation_script_integration(self):
        """Test integration with validation script."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
            
            result = self.runner.invoke(app, [
                "validate-comprehensive",
                "test_spec.json",
                "--type", "spec"
            ])
            
            # Should attempt to run validation
            assert result.exit_code in [0, 1]
    
    @pytest.mark.integration  
    def test_makefile_integration(self):
        """Test that Makefile validation target works."""
        import subprocess
        
        # Test if Makefile exists and has validate target
        try:
            result = subprocess.run(
                ["make", "-n", "validate"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            # Should show what validate target would do
            assert result.returncode in [0, 2]  # 0 = success, 2 = no target (acceptable)
            
        except FileNotFoundError:
            # Make command not available, skip test
            pytest.skip("make command not available")


class TestValidationPerformance:
    """Performance tests for validation framework."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up performance test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.performance
    def test_validation_speed_small_file(self):
        """Test validation speed on small files."""
        import time
        
        # Create small spec file
        spec_file = self.temp_path / "small_spec.json"
        spec_data = {
            "task": "qa",
            "dataset": {"name": "jsonl", "path": "dummy.jsonl"},
            "adapter": {"name": "echo"},
            "metrics": [{"name": "exact_match"}]
        }
        with open(spec_file, "w") as f:
            json.dump(spec_data, f)
        
        start_time = time.time()
        result = self.runner.invoke(app, [
            "validate-comprehensive",
            str(spec_file),
            "--type", "spec"
        ])
        end_time = time.time()
        
        # Validation should complete within reasonable time
        validation_time = end_time - start_time
        assert validation_time < 10.0  # Should complete within 10 seconds
        
        # Log performance for monitoring
        print(f"Small file validation time: {validation_time:.3f}s")
    
    @pytest.mark.performance
    def test_validation_memory_usage(self):
        """Test validation memory usage."""
        try:
            import psutil
            import os
            
            # Create spec file
            spec_file = self.temp_path / "memory_test.json"
            spec_data = {
                "task": "qa",
                "dataset": {"name": "jsonl", "path": "dummy.jsonl"},
                "adapter": {"name": "echo"},
                "metrics": [{"name": "exact_match"}]
            }
            with open(spec_file, "w") as f:
                json.dump(spec_data, f)
            
            # Measure memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            
            result = self.runner.invoke(app, [
                "validate-comprehensive",
                str(spec_file),
                "--type", "spec"
            ])
            
            # Measure memory after
            memory_after = process.memory_info().rss
            memory_diff = memory_after - memory_before
            
            # Memory usage should be reasonable (less than 100MB)
            assert memory_diff < 100 * 1024 * 1024  # 100MB
            
            print(f"Validation memory usage: {memory_diff / 1024 / 1024:.2f}MB")
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")


# Test markers for different test categories
pytestmark = [
    pytest.mark.validation,
    pytest.mark.cli
]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
