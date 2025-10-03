#!/usr/bin/env python3
"""
Data Validation Tool for OpenEval Lab

This script provides comprehensive data validation for evaluation datasets,
including format checking, quality assessment, and statistical analysis.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter, defaultdict
import statistics

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class DataValidator:
    """Comprehensive data validator for evaluation datasets."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.issues: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {}
        self.data: List[Dict[str, Any]] = []

    def validate(self) -> bool:
        """Run complete validation suite."""
        print(f"🔍 Validating dataset: {self.dataset_path}")

        # Load data
        if not self._load_data():
            return False

        # Run validations
        self._validate_format()
        self._validate_required_fields()
        self._validate_data_quality()
        self._validate_duplicates()
        self._compute_statistics()

        # Generate report
        self._generate_report()

        return len(self.issues) == 0

    def _load_data(self) -> bool:
        """Load dataset from file."""
        try:
            if self.dataset_path.suffix == ".jsonl":
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    self.data = [json.loads(line.strip()) for line in f if line.strip()]
            elif self.dataset_path.suffix == ".json":
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        self.data = loaded
                    elif isinstance(loaded, dict) and "data" in loaded:
                        self.data = loaded["data"]
                    else:
                        self.data = [loaded]
            elif self.dataset_path.suffix == ".csv":
                if not HAS_PANDAS:
                    self.issues.append(
                        {
                            "severity": "ERROR",
                            "category": "DEPENDENCY",
                            "description": "pandas required for CSV validation",
                            "recommendation": "Install pandas: pip install pandas",
                        }
                    )
                    return False
                df = pd.read_csv(self.dataset_path)
                self.data = df.to_dict("records")
            else:
                self.issues.append(
                    {
                        "severity": "ERROR",
                        "category": "FORMAT",
                        "description": f"Unsupported file format: {self.dataset_path.suffix}",
                        "recommendation": "Use .json, .jsonl, or .csv files",
                    }
                )
                return False

            print(f"✅ Loaded {len(self.data)} examples")
            return True

        except Exception as e:
            self.issues.append(
                {
                    "severity": "ERROR",
                    "category": "LOADING",
                    "description": f"Failed to load dataset: {e}",
                    "recommendation": "Check file format and encoding",
                }
            )
            return False

    def _validate_format(self) -> None:
        """Validate data format consistency."""
        if not self.data:
            return

        first_example = self.data[0]
        expected_keys = set(first_example.keys())

        for i, example in enumerate(self.data[1:], 1):
            current_keys = set(example.keys())
            if current_keys != expected_keys:
                missing = expected_keys - current_keys
                extra = current_keys - expected_keys

                if missing:
                    self.issues.append(
                        {
                            "severity": "WARNING",
                            "category": "FORMAT",
                            "description": f"Example {i} missing fields: {missing}",
                            "recommendation": "Ensure all examples have consistent fields",
                        }
                    )

                if extra:
                    self.issues.append(
                        {
                            "severity": "INFO",
                            "category": "FORMAT",
                            "description": f"Example {i} has extra fields: {extra}",
                            "recommendation": "Review extra fields for consistency",
                        }
                    )

    def _validate_required_fields(self) -> None:
        """Validate presence of required fields."""
        if not self.data:
            return

        # Common required fields for different task types
        task_indicators = {
            "question": ["question", "answer"],
            "summarization": ["text", "summary"],
            "translation": ["source", "target"],
            "generation": ["prompt", "completion"],
            "classification": ["text", "label"],
        }

        # Try to infer task type from first example
        first_keys = set(self.data[0].keys())
        inferred_task = None

        for task, required in task_indicators.items():
            if any(req in first_keys for req in required):
                inferred_task = task
                break

        if inferred_task:
            required_fields = task_indicators[inferred_task]
            for i, example in enumerate(self.data):
                missing = [field for field in required_fields if field not in example]
                if missing:
                    self.issues.append(
                        {
                            "severity": "ERROR",
                            "category": "REQUIRED_FIELDS",
                            "description": f"Example {i} missing required fields for {inferred_task}: {missing}",
                            "recommendation": f"Add {missing} fields to example {i}",
                        }
                    )

    def _validate_data_quality(self) -> None:
        """Validate data quality metrics."""
        if not self.data:
            return

        text_fields = [
            "text",
            "question",
            "answer",
            "summary",
            "prompt",
            "completion",
            "source",
            "target",
        ]

        for i, example in enumerate(self.data):
            for field in text_fields:
                if field in example:
                    value = example[field]
                    if not isinstance(value, str):
                        self.issues.append(
                            {
                                "severity": "WARNING",
                                "category": "QUALITY",
                                "description": f"Example {i} field '{field}' is not a string: {type(value)}",
                                "recommendation": "Ensure text fields contain strings",
                            }
                        )
                        continue

                    # Check for empty strings
                    if not value.strip():
                        self.issues.append(
                            {
                                "severity": "WARNING",
                                "category": "QUALITY",
                                "description": f"Example {i} field '{field}' is empty",
                                "recommendation": "Provide meaningful content for text fields",
                            }
                        )

                    # Check for very short content
                    if len(value.strip()) < 5:
                        self.issues.append(
                            {
                                "severity": "INFO",
                                "category": "QUALITY",
                                "description": f"Example {i} field '{field}' is very short: {len(value)} chars",
                                "recommendation": "Review short content for completeness",
                            }
                        )

                    # Check for placeholder text
                    placeholders = ["lorem ipsum", "placeholder", "todo", "tbd", "n/a"]
                    if any(placeholder in value.lower() for placeholder in placeholders):
                        self.issues.append(
                            {
                                "severity": "WARNING",
                                "category": "QUALITY",
                                "description": f"Example {i} field '{field}' contains placeholder text",
                                "recommendation": "Replace placeholder content with real data",
                            }
                        )

    def _validate_duplicates(self) -> None:
        """Check for duplicate examples."""
        if not self.data:
            return

        # Convert examples to hashable tuples for duplicate detection
        seen = set()
        duplicates = []

        for i, example in enumerate(self.data):
            # Create a hashable representation
            try:
                example_tuple = tuple(sorted(example.items()))
                if example_tuple in seen:
                    duplicates.append(i)
                else:
                    seen.add(example_tuple)
            except TypeError:
                # Skip if example contains unhashable types
                continue

        if duplicates:
            self.issues.append(
                {
                    "severity": "WARNING",
                    "category": "DUPLICATES",
                    "description": f"Found {len(duplicates)} duplicate examples at indices: {duplicates[:10]}{'...' if len(duplicates) > 10 else ''}",
                    "recommendation": "Remove duplicate examples to ensure dataset quality",
                }
            )

    def _compute_statistics(self) -> None:
        """Compute dataset statistics."""
        if not self.data:
            return

        self.stats = {
            "total_examples": len(self.data),
            "field_counts": defaultdict(int),
            "text_length_stats": {},
            "label_distribution": {},
        }

        # Field presence
        for example in self.data:
            for field in example.keys():
                self.stats["field_counts"][field] += 1

        # Text length statistics
        text_fields = ["text", "question", "answer", "summary", "prompt", "completion"]
        for field in text_fields:
            lengths = []
            for example in self.data:
                if field in example and isinstance(example[field], str):
                    lengths.append(len(example[field]))

            if lengths:
                self.stats["text_length_stats"][field] = {
                    "count": len(lengths),
                    "min": min(lengths),
                    "max": max(lengths),
                    "mean": statistics.mean(lengths),
                    "median": statistics.median(lengths),
                }

        # Label distribution (if applicable)
        if "label" in self.data[0]:
            labels = [ex.get("label") for ex in self.data if "label" in ex]
            if labels:
                self.stats["label_distribution"] = dict(Counter(labels))

    def _generate_report(self) -> None:
        """Generate validation report."""
        report_path = self.dataset_path.with_suffix(".validation_report.json")

        report = {
            "dataset": str(self.dataset_path),
            "validation_timestamp": None,  # Will be set when saving
            "is_valid": len(self.issues) == 0,
            "issue_count": len(self.issues),
            "issues": self.issues,
            "statistics": self.stats,
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📄 Validation report saved to {report_path}")

        # Print summary
        if self.issues:
            print(f"\n🚨 Found {len(self.issues)} issues:")
            severity_counts = defaultdict(int)
            for issue in self.issues:
                severity_counts[issue["severity"]] += 1

            for severity, count in severity_counts.items():
                print(f"  {severity}: {count}")
        else:
            print("\n✅ Dataset validation passed!")

        # Print key statistics
        if self.stats:
            print("\n📊 Dataset Statistics:")
            print(f"  Total examples: {self.stats['total_examples']}")
            if self.stats["text_length_stats"]:
                print(f"  Text fields: {list(self.stats['text_length_stats'].keys())}")


def main():
    """Main entry point for data validation."""
    if len(sys.argv) != 2:
        print("Usage: python data_validator.py <dataset_file>")
        print("Supported formats: .json, .jsonl, .csv")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    validator = DataValidator(dataset_path)
    is_valid = validator.validate()

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
