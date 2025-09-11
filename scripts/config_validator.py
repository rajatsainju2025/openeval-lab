#!/usr/bin/env python3
"""
Configuration Validator for OpenEval Lab

This script validates evaluation configuration files against schemas,
checks for common configuration errors, and provides recommendations.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
import re

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class ConfigValidator:
    """Comprehensive configuration validator."""

    def __init__(self):
        self.schemas = self._load_schemas()

    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load validation schemas."""
        schemas = {}

        # Basic evaluation spec schema
        schemas['evaluation_spec'] = {
            "type": "object",
            "required": ["task", "dataset", "model"],
            "properties": {
                "task": {"type": "string", "enum": ["qa", "summarization", "generation", "classification", "translation"]},
                "dataset": {"type": "object", "required": ["path"]},
                "model": {"type": "object", "required": ["name"]},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "adapter": {"type": "object"},
                "evaluation": {"type": "object"},
                "output": {"type": "object"}
            }
        }

        # Dataset schema
        schemas['dataset'] = {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string"},
                "format": {"type": "string", "enum": ["json", "jsonl", "csv", "parquet"]},
                "split": {"type": "string"},
                "subset": {"type": "string"},
                "validation": {"type": "object"}
            }
        }

        # Model schema
        schemas['model'] = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": ["api", "local", "huggingface", "openai", "anthropic"]},
                "parameters": {"type": "object"},
                "api_key": {"type": "string"},
                "endpoint": {"type": "string"}
            }
        }

        return schemas

    def validate_file(self, config_path: Path) -> ValidationResult:
        """Validate a configuration file."""
        errors = []
        warnings = []
        suggestions = []

        try:
            # Load configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                    try:
                        import yaml
                        config = yaml.safe_load(f)
                    except ImportError:
                        errors.append("PyYAML required for YAML validation")
                        return ValidationResult(False, errors, warnings, suggestions)
                else:
                    config = json.load(f)

        except Exception as e:
            errors.append(f"Failed to load configuration: {e}")
            return ValidationResult(False, errors, warnings, suggestions)

        # Validate against schema
        if HAS_JSONSCHEMA:
            schema_errors = self._validate_schema(config)
            errors.extend(schema_errors)

        # Check for common issues
        self._check_common_issues(config, errors, warnings, suggestions)

        # Check file references
        self._check_file_references(config, config_path.parent, errors, warnings)

        # Check API keys and sensitive data
        self._check_sensitive_data(config, warnings, suggestions)

        # Generate suggestions
        self._generate_suggestions(config, suggestions)

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, suggestions)

    def _validate_schema(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against JSON schema."""
        errors = []

        try:
            # Validate top-level structure
            if 'task' in config and HAS_JSONSCHEMA:
                jsonschema.validate(config, self.schemas['evaluation_spec'])

            # Validate nested objects
            if 'dataset' in config and HAS_JSONSCHEMA:
                jsonschema.validate(config['dataset'], self.schemas['dataset'])

            if 'model' in config and HAS_JSONSCHEMA:
                jsonschema.validate(config['model'], self.schemas['model'])

        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")

        return errors

    def _check_common_issues(
        self,
        config: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
        suggestions: List[str]
    ) -> None:
        """Check for common configuration issues."""

        # Check required fields
        required_fields = ['task', 'dataset', 'model']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Check dataset configuration
        if 'dataset' in config:
            dataset = config['dataset']
            if 'path' not in dataset:
                errors.append("Dataset missing 'path' field")
            elif not isinstance(dataset['path'], str):
                errors.append("Dataset 'path' must be a string")

        # Check model configuration
        if 'model' in config:
            model = config['model']
            if 'name' not in model:
                errors.append("Model missing 'name' field")

            # Check for API key requirements
            model_name = model.get('name', '').lower()
            if any(api_provider in model_name for api_provider in ['openai', 'anthropic', 'claude', 'gpt']):
                if 'api_key' not in model and 'api_key' not in os.environ:
                    warnings.append("API key not found in config or environment variables")

        # Check metrics configuration
        if 'metrics' in config:
            metrics = config['metrics']
            if not isinstance(metrics, list):
                errors.append("'metrics' must be a list")
            else:
                valid_metrics = [
                    'accuracy', 'f1', 'precision', 'recall', 'bleu', 'rouge',
                    'meteor', 'bert_score', 'semantic_similarity'
                ]
                for metric in metrics:
                    if metric not in valid_metrics:
                        warnings.append(f"Unknown metric: {metric}")

        # Check evaluation parameters
        if 'evaluation' in config:
            eval_config = config['evaluation']
            if 'concurrency' in eval_config:
                concurrency = eval_config['concurrency']
                if not isinstance(concurrency, int) or concurrency < 1:
                    errors.append("'concurrency' must be a positive integer")
                elif concurrency > 100:
                    warnings.append("High concurrency may cause rate limiting")

    def _check_file_references(
        self,
        config: Dict[str, Any],
        base_path: Path,
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Check file references in configuration."""
        if 'dataset' in config and 'path' in config['dataset']:
            dataset_path = config['dataset']['path']
            if not os.path.isabs(dataset_path):
                full_path = base_path / dataset_path
            else:
                full_path = Path(dataset_path)

            if not full_path.exists():
                warnings.append(f"Dataset file not found: {full_path}")

    def _check_sensitive_data(
        self,
        config: Dict[str, Any],
        warnings: List[str],
        suggestions: List[str]
    ) -> None:
        """Check for sensitive data in configuration."""
        sensitive_patterns = [
            r'api_key', r'password', r'secret', r'token', r'key'
        ]

        def check_dict(d: Dict[str, Any], path: str = "") -> None:
            for key, value in d.items():
                current_path = f"{path}.{key}" if path else key
                if any(re.search(pattern, key, re.IGNORECASE) for pattern in sensitive_patterns):
                    if isinstance(value, str) and len(value) > 10:
                        warnings.append(f"Potential sensitive data in {current_path}")
                        suggestions.append(f"Consider moving {current_path} to environment variables")
                elif isinstance(value, dict):
                    check_dict(value, current_path)

        check_dict(config)

    def _generate_suggestions(
        self,
        config: Dict[str, Any],
        suggestions: List[str]
    ) -> None:
        """Generate configuration improvement suggestions."""
        # Suggest adding missing optional fields
        if 'evaluation' not in config:
            suggestions.append("Consider adding 'evaluation' section for performance tuning")

        if 'output' not in config:
            suggestions.append("Consider adding 'output' section for result formatting")

        if 'metrics' in config and len(config['metrics']) == 0:
            suggestions.append("Add evaluation metrics to assess model performance")

        # Suggest optimization based on task type
        task = config.get('task', '')
        if task == 'qa':
            if 'metrics' not in config or 'f1' not in config['metrics']:
                suggestions.append("For QA tasks, consider adding F1 score metric")
        elif task == 'summarization':
            if 'metrics' not in config or 'rouge' not in config['metrics']:
                suggestions.append("For summarization, consider adding ROUGE metrics")
        elif task == 'generation':
            if 'metrics' not in config or 'bleu' not in config['metrics']:
                suggestions.append("For generation, consider adding BLEU score")

    def validate_directory(self, config_dir: Path) -> Dict[str, ValidationResult]:
        """Validate all configuration files in a directory."""
        results = {}

        config_files = list(config_dir.glob("*.json")) + list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

        for config_file in config_files:
            print(f"🔍 Validating {config_file.name}...")
            result = self.validate_file(config_file)
            results[config_file.name] = result

            if result.errors:
                print(f"❌ {len(result.errors)} errors found")
            elif result.warnings:
                print(f"⚠️  {len(result.warnings)} warnings found")
            else:
                print("✅ Configuration is valid")

        return results


def main():
    """Main entry point for configuration validation."""
    if len(sys.argv) < 2:
        print("Usage: python config_validator.py <config_file_or_directory>")
        print("Supported formats: .json, .yaml, .yml")
        sys.exit(1)

    target_path = Path(sys.argv[1])
    validator = ConfigValidator()

    if target_path.is_file():
        print(f"🔍 Validating configuration file: {target_path}")
        result = validator.validate_file(target_path)

        if result.errors:
            print("\n❌ Validation Errors:")
            for error in result.errors:
                print(f"  • {error}")

        if result.warnings:
            print("\n⚠️  Validation Warnings:")
            for warning in result.warnings:
                print(f"  • {warning}")

        if result.suggestions:
            print("\n💡 Suggestions:")
            for suggestion in result.suggestions:
                print(f"  • {suggestion}")

        if result.is_valid:
            print("\n✅ Configuration is valid!")
        else:
            print("\n❌ Configuration has errors!")
            sys.exit(1)

    elif target_path.is_dir():
        print(f"🔍 Validating configuration directory: {target_path}")
        results = validator.validate_directory(target_path)

        total_files = len(results)
        valid_files = sum(1 for r in results.values() if r.is_valid)
        files_with_warnings = sum(1 for r in results.values() if r.warnings)

        print("\n📊 Summary:")
        print(f"  Total files: {total_files}")
        print(f"  Valid files: {valid_files}")
        print(f"  Files with warnings: {files_with_warnings}")

        if valid_files < total_files:
            sys.exit(1)

    else:
        print(f"Error: Path not found: {target_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
