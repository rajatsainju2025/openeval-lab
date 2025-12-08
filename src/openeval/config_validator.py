"""
Configuration File Validator

Simple, user-friendly validation for OpenEval configuration files (JSON/YAML).
Provides clear error messages with actionable suggestions.

Example:
    >>> from openeval.config_validator import validate_config_file
    >>>
    >>> # Validate a spec file
    >>> result = validate_config_file("my_spec.json")
    >>> if result["valid"]:
    ...     print("✅ Configuration is valid!")
    >>> else:
    ...     for error in result["errors"]:
    ...         print(error)
"""

from pathlib import Path
from typing import Dict, List, Any, Set
import json


def validate_config_file(file_path: str) -> Dict[str, Any]:
    """Validate a configuration file with helpful error messages.

    Args:
        file_path: Path to JSON or YAML config file

    Returns:
        Dictionary with validation results:
        - valid: bool - whether the config is valid
        - errors: list of error messages
        - warnings: list of warning messages
        - suggestions: list of helpful suggestions

    Example:
        >>> result = validate_config_file("examples/qa_spec.json")
        >>> if not result["valid"]:
        ...     print("\\n".join(result["errors"]))
    """
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

    # Check file exists
    path = Path(file_path)
    if not path.exists():
        return {
            "valid": False,
            "errors": [
                f"❌ File not found: {file_path}",
                "   💡 Check the file path is correct",
            ],
            "warnings": [],
            "suggestions": ["Verify the file exists and the path is correct"],
        }

    # Try to load file
    try:
        with open(path, "r") as f:
            if path.suffix in [".json"]:
                config = json.load(f)
            elif path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml

                    config = yaml.safe_load(f)
                except ImportError:
                    return {
                        "valid": False,
                        "errors": [
                            "❌ YAML file detected but PyYAML not installed",
                            "   💡 Install with: pip install pyyaml",
                        ],
                        "warnings": [],
                        "suggestions": ["Install PyYAML to parse YAML files"],
                    }
            else:
                return {
                    "valid": False,
                    "errors": [
                        f"❌ Unsupported file format: {path.suffix}",
                        "   💡 Use .json or .yaml/.yml files",
                    ],
                    "warnings": [],
                    "suggestions": ["Convert to JSON or YAML format"],
                }
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "errors": [
                f"❌ Invalid JSON syntax at line {e.lineno}, column {e.colno}",
                f"   {e.msg}",
                "   💡 Check for missing commas, quotes, or brackets",
            ],
            "warnings": [],
            "suggestions": [
                "Use a JSON validator or linter",
                "Common issues: trailing commas, unquoted keys, missing closing brackets",
            ],
        }
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"❌ Failed to parse file: {str(e)}"],
            "warnings": [],
            "suggestions": ["Check file syntax and formatting"],
        }

    # Validate structure
    if not isinstance(config, dict):
        errors.append(
            "❌ Configuration must be a JSON object (dictionary), not " + type(config).__name__
        )
        suggestions.append("Wrap your configuration in curly braces { }")

    # Required fields
    required_fields: Set[str] = {"task", "dataset", "adapter", "metrics"}
    missing = required_fields - set(config.keys())
    if missing:
        missing_list = ", ".join(f"'{f}'" for f in sorted(missing))
        errors.append(f"❌ Missing required fields: {missing_list}")
        errors.append("   💡 Add these fields to your configuration:")
        for field in sorted(missing):
            if field == "task":
                errors.append('   - task: "qa"  # or "code", "summarization", etc.')
            elif field == "dataset":
                errors.append('   - dataset: "path/to/data.jsonl"')
            elif field == "adapter":
                errors.append('   - adapter: {"type": "openai", "model": "gpt-4"}')
            elif field == "metrics":
                errors.append('   - metrics: ["exact_match", "f1"]')

    # Validate task
    valid_tasks = {"qa", "summarization", "code", "agent", "multimodal", "loglikelihood"}
    task = config.get("task")
    if task and task not in valid_tasks:
        valid_list = ", ".join(f"'{t}'" for t in sorted(valid_tasks))
        errors.append(f"❌ Unknown task type: '{task}'")
        errors.append(f"   💡 Valid task types: {valid_list}")

    # Validate metrics
    metrics = config.get("metrics", [])
    if not isinstance(metrics, list):
        errors.append(f"❌ Metrics must be a list, not {type(metrics).__name__}")
        errors.append('   💡 Example: "metrics": ["exact_match", "f1"]')

    # Validate dataset
    dataset = config.get("dataset")
    if dataset and isinstance(dataset, str):
        dataset_path = Path(dataset)
        if not dataset_path.exists() and not dataset.startswith(("http://", "https://")):
            warnings.append(f"⚠️  Dataset file not found: '{dataset}'")
            suggestions.append("Ensure the dataset file exists or provide a full path/URL")

    # Validate adapter
    adapter = config.get("adapter")
    if adapter:
        if isinstance(adapter, dict):
            if "type" not in adapter and "name" not in adapter:
                errors.append("❌ Adapter missing 'type' or 'name' field")
                errors.append('   💡 Example: {"type": "openai", "model": "gpt-4"}')
        elif not isinstance(adapter, str):
            errors.append(f"❌ Adapter must be string or dict, not {type(adapter).__name__}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
    }


def validate_and_print(file_path: str) -> bool:
    """Validate a config file and print results.

    Args:
        file_path: Path to config file

    Returns:
        True if valid, False otherwise

    Example:
        >>> if validate_and_print("my_spec.json"):
        ...     print("Ready to run!")
    """
    result = validate_config_file(file_path)

    if result["valid"]:
        print(f"✅ Configuration is valid: {file_path}")
        if result["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in result["warnings"]:
                print(f"  {warning}")
        return True
    else:
        print(f"❌ Configuration validation failed: {file_path}\n")
        for error in result["errors"]:
            print(error)

        if result["suggestions"]:
            print("\n💡 Suggestions:")
            for suggestion in result["suggestions"]:
                print(f"  • {suggestion}")

        return False
