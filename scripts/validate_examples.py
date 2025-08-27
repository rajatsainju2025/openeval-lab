"""
Validate example spec files by attempting to load them via the library's spec loader.
Exits with non-zero code if any spec fails to load.
"""
from pathlib import Path
import sys

from openeval.spec import load_spec


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    examples = (root / "examples").glob("*.*")
    targets = [p for p in examples if p.suffix in {".json", ".yaml", ".yml"} and "spec" in p.name]
    failed = []
    for p in sorted(targets):
        try:
            # load_spec should raise SystemExit on validation errors; catch broadly
            load_spec(p)
            print({"validated": str(p)})
        except SystemExit as e:
            print({"error": str(p), "code": getattr(e, "code", 1)})
            failed.append(p)
        except Exception as e:
            print({"error": str(p), "exception": str(e)})
            failed.append(p)
    if failed:
        print({"failed": [str(p) for p in failed]})
        return 1
    print({"validated_count": len(targets)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
