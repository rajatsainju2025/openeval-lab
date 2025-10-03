"""
Validate example spec files by attempting to load them via the library's spec loader.
Exits with non-zero code if any spec fails to load.
"""

from pathlib import Path
import sys
import json
from typing import Dict, Any

from openeval.spec import load_spec


def validate_with_comprehensive_check(spec_path: Path) -> Dict[str, Any]:
    """Run comprehensive validation on a spec file."""
    try:
        # Basic load test first
        task, dataset, adapter, metrics, out = load_spec(spec_path)

        # Extended validation checks
        issues = []
        metadata = {
            "task": str(type(task).__name__),
            "adapter": str(type(adapter).__name__),
            "dataset": str(type(dataset).__name__),
            "metrics": [str(type(m).__name__) for m in metrics],
        }

        # Check for basic requirements
        if not hasattr(adapter, "generate"):
            issues.append(
                {
                    "severity": "error",
                    "message": "Adapter missing generate method",
                    "category": "interface",
                }
            )

        if not hasattr(task, "evaluate"):
            issues.append(
                {
                    "severity": "warning",
                    "message": "Task missing evaluate method",
                    "category": "interface",
                }
            )

        # Try to get a small sample from dataset
        try:
            examples = list(dataset)[:3]
            if not examples:
                issues.append(
                    {"severity": "error", "message": "Dataset is empty", "category": "data"}
                )
            else:
                # Check example structure
                for i, ex in enumerate(examples):
                    if not hasattr(ex, "input"):
                        issues.append(
                            {
                                "severity": "error",
                                "message": f"Example {i} missing input field",
                                "category": "data",
                            }
                        )
                    if not hasattr(ex, "reference"):
                        issues.append(
                            {
                                "severity": "warning",
                                "message": f"Example {i} missing reference field",
                                "category": "data",
                            }
                        )
        except Exception as e:
            issues.append(
                {"severity": "error", "message": f"Cannot iterate dataset: {e}", "category": "data"}
            )

        error_count = len([i for i in issues if i["severity"] == "error"])
        warning_count = len([i for i in issues if i["severity"] == "warning"])

        return {
            "valid": error_count == 0,
            "path": str(spec_path),
            "issues": issues,
            "metadata": metadata,
            "error_count": error_count,
            "warning_count": warning_count,
        }

    except Exception as e:
        return {
            "valid": False,
            "path": str(spec_path),
            "error": str(e),
            "error_count": 1,
            "warning_count": 0,
        }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    examples = (root / "examples").glob("*.*")
    targets = [p for p in examples if p.suffix in {".json", ".yaml", ".yml"} and "spec" in p.name]

    failed = []
    validation_results = []
    total_errors = 0
    total_warnings = 0

    print("🔍 Running comprehensive validation on example specs...")
    print(f"Found {len(targets)} spec files to validate\n")

    for p in sorted(targets):
        print(f"Validating {p.name}...", end=" ")

        result = validate_with_comprehensive_check(p)
        validation_results.append(result)

        if result["valid"]:
            if result["warning_count"] > 0:
                print(f"✅ (with {result['warning_count']} warnings)")
            else:
                print("✅")
        else:
            print("❌")
            failed.append(p)

        total_errors += result["error_count"]
        total_warnings += result["warning_count"]

        # Show issues if any
        if "issues" in result:
            for issue in result["issues"]:
                severity_icon = "❌" if issue["severity"] == "error" else "⚠️"
                print(f"  {severity_icon} {issue['message']}")
        elif "error" in result:
            print(f"  ❌ {result['error']}")

        print()

    # Summary
    print("=" * 60)
    print("📊 Validation Summary:")
    print(f"  Total specs: {len(targets)}")
    print(f"  Passed: {len(targets) - len(failed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total errors: {total_errors}")
    print(f"  Total warnings: {total_warnings}")

    if failed:
        print("\n❌ Failed specs:")
        for p in failed:
            print(f"  - {p.name}")
        print("\n🔧 Please fix the errors above before proceeding.")
        return 1
    else:
        if total_warnings > 0:
            print(f"\n⚠️  All specs passed but there are {total_warnings} warnings to review.")
        else:
            print("\n🎉 All specs validated successfully!")

        # Save validation results
        results_file = root / "validation_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "timestamp": "2025-09-03T00:00:00Z",
                    "summary": {
                        "total_specs": len(targets),
                        "passed": len(targets) - len(failed),
                        "failed": len(failed),
                        "total_errors": total_errors,
                        "total_warnings": total_warnings,
                    },
                    "results": validation_results,
                },
                f,
                indent=2,
            )

        print(f"📄 Detailed results saved to: {results_file}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
